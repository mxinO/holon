"""L0 for kernels.ggml_cuda (M0.1 design §4.6): llama.cpp's quantized kernels against gguf-py and the q8_1 emulator.

- `qlinear`, every supported type, `in` ∈ {768, 2304, 4096, 12288} on random-byte blocks with realistic fp16 scales
  (negative, zero, subnormal, up to 1) and real tensors from the cached Qwen3-8B files; `out` = 300 (not a multiple
  of the mmq tile); the bytes right after the weights are poisoned (NaN scales). Rows 1..9 (every mmvq column
  count, and mmq above the type's threshold on this device), 64, 128, 256, 512, 513, 1025, max_step_tokens;
  activations hold an all-zero row and a row with one large outlier. Per element `l0.check(bf16, in)` against the
  emulated reference of the path taken (judge `q8_1`, MMA kernels: cc >= 7.5), NMSE <= 5e-4 against the
  unquantized oracle, run-to-run identical.
- mmq's schedules at 512 rows, asserted from llama.cpp's own choice (tile I x J, nsm): stream-k with the fixup
  pass (out = 300), one block per tile (out = nsm * I), and the model's out 4096 and 12288. Under each, its bf16
  output equals its f32 output cast (docs/m07-perf.md D4: the fixup's partials stay f32 until one rounding), and
  NVFP4's `scale2` multiplies the f32 sum before that rounding (`torch.equal`, the fixup forced; native W4A4 too on
  sm_120).
- `convert` and `get_rows` (`qembedding`) against `gguf.quants.dequantize` (Q2_0, which gguf-py 0.19 lacks: against
  `dequantize`'s reading of its block): bit-identical for every type. FMA
  decision, measured on sm_86: with llama.cpp's flags (`-use_fast_math`, FMA contraction on) every value matches
  gguf-py bit for bit, affine types (Q4_1, Q5_1, Q2_K, Q4_K, Q5_K, IQ1_S) included, so nothing is built with
  `-fmad=false`; the one exception is the sign of zero in Q4_0's convert ((8 - 8) * d is +0 for negative d).
- CUDA graphs: captured calls replay bitwise like eager ones (PDL launches on sm_120a), also on new inputs; the
  fused gate/up and a prepared input read by two consumers too.
- Workspace: every call takes at most what `workspace_bytes` sizes for its rows (the pool's high-water mark,
  `Sized`), so an undersized formula fails at any row count.
- FP4 (docs/m06-lowp.md D8, §4.8; MXFP4 and NVFP4 are in the type sweeps above as ggml blocks): the HF layouts'
  `prepare` is lossless — gguf-py on its blocks × scale2 equals `reference.py` by value, the blocks are llama.cpp's
  converter's byte for byte, the source parts untouched — and the repack plants (N2, N3, M1) break it; `scale2` on
  every path and output dtype against the q8_1 emulator in f64 on the reference weight, also in CUDA graphs; in
  mmvq's epilogue at rows 2–8 by value
equal to the f32 multiply it replaced (docs/m07-perf.md D11). The
  precision check: mmq's precision is the spec's, and the default one (the N6 plant) is an error on sm_120 only;
  `quant.act = a4` selects native W4A4 on sm_120 and q8_1 elsewhere. Native FP4 activations: llama.cpp's quantizer
  against the emulator (`holon_judge.w4a4`: MXFP4 runs on any class, NVFP4 on sm_120), codes differing at a rate
  ≤ 1e-3 and only in fast-math's tie classes. sm_120: native W4A4 mmq against its own decoded activation codes (bf16
  and f32 inputs), the zeroed tail of `prepare` at `in` 768 (without it, plant M4: NaN output).

Skips without a GPU; builds the extension (prebuild) into the user cache if it is missing.
"""
import dataclasses
import os
import pathlib
import subprocess

import gguf
import numpy as np
import pytest
import torch
from holon.api import Layout, QTensor, Scheme
from holon.core.build import BuildContext
from holon.core.plugins import PluginSet, discover
from holon_judge import l0, q8_1, w4a4
from holon_judge.hf_quant import converter, dequantize_blocks, ulps
from holon_judge.planted import PLANTS
from holon_plugins.kernels.ggml_cuda import extension
from holon_plugins.kernels.ggml_cuda.plugin import (
    CHUNK,
    CROSSOVER,
    FP4,
    GLU_TYPES,
    NATIVE_FP4,
    PREC,
    Q8_1,
    TYPES,
    mmvq_rows,
)
from holon_std.ops import reference
from holon_std.weights.hf import reference as hf
from holon_testing.qwen3_5_files import LLAMA
from huggingface_hub.constants import HF_HUB_CACHE

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

INS = (768, 2304, 4096, 12288)
OUT = 300
GLU_OUT = 296                   # a merged gate/up of OUT's first rows: halves of a multiple of 4 rows (qlinear_glu)
MAX_STEP_TOKENS = 4096          # the engine's default step budget
NMSE = 5e-4                     # llama.cpp test-backend-ops MUL_MAT threshold
OUTLIER = 1000.0
GGUF_DIR = pathlib.Path(HF_HUB_CACHE) / "models--unsloth--Qwen3-8B-GGUF" / "snapshots"
FILES = sorted(GGUF_DIR.glob("*/*.gguf"))
# Byte offsets of each block's fp16 scales (d, and dmin / m where the type has one); d at 0 otherwise. IQ1_M's d is
# the top nibbles of the four uint16 scales at 48.
SCALES = {"q4_1": (0, 2), "q5_1": (0, 2), "q2_k": (80, 82), "q3_k": (108,), "q4_k": (0, 2), "q5_k": (0, 2),
          "q6_k": (208,), "iq1_m": (48,)}
# The (values, bytes) per block of the types gguf-py 0.19 lacks: Q2_0 (ggml-common.h's block_q2_0).
BLOCK_SIZES = {"q2_0": (64, 18)}


@pytest.fixture(scope="module")
def sm() -> int:
    major, minor = torch.cuda.get_device_capability()
    sm = 10 * major + minor
    assert sm >= 75, "q8_1.MMQ emulates llama.cpp's MMA kernels (turing_mma_available)"
    try:
        extension.load(sm)
    except extension.NotBuilt:
        extension.build(sm, os.cpu_count())
    return sm


@pytest.fixture(scope="module")
def ctx(sm) -> BuildContext:
    return BuildContext(PluginSet(discover()), torch.device("cuda"))


@pytest.fixture(scope="module")
def real() -> dict[tuple[str, int], np.ndarray]:
    """The first OUT rows of the first tensor of each (type, in) in the cached Qwen3-8B files."""
    found = {}
    for path in FILES:
        for t in gguf.GGUFReader(path).tensors:
            key = (t.tensor_type.name.lower(), int(t.shape[0]))
            if key not in found and "token_embd" not in t.name and len(t.shape) == 2 and t.shape[1] >= OUT:
                found[key] = np.array(t.data[:OUT])
    return found


def qlinear(ctx, name: str, k: int, out_dtype: str = "bfloat16", out: int = OUT, act_dtype: str = "bfloat16",
            act_quant: str = "a8", op: str = "qlinear@2"):
    """A `qlinear` of ggml type `name` (or of an HF layout, `"nvfp4/hf"`), held to its workspace formula; with `op`
    "qlinear_glu@1", its fused SwiGLU on a merged gate/up of `out` rows."""
    facts = {"layout": name if "/" in name else f"ggml.{name}/gguf.native", "in": k, "out": out,
             "act_dtype": act_dtype, "out_dtype": out_dtype, "act_quant": act_quant}
    facts |= {"glu": "swiglu"} if op == "qlinear_glu@1" else {}
    spec = ctx.select_kernel(op, site="l0", **facts)
    assert spec.provider == "kernels.ggml_cuda" and {"graph_safe", "deterministic"} <= spec.props
    kernel, device = spec.impl(ctx, **facts), torch.cuda.current_device()
    return Sized(kernel, lambda x, parts: kernel.lib.workspace_bytes(device, k, min(len(_rows(x)), kernel.call_rows)))


def _rows(x) -> torch.Tensor:
    """A call's activations: `x`, or a `prepare_input` handle's."""
    return x if isinstance(x, torch.Tensor) else x.x


class Sized:
    """A kernel instance whose every call is held to its workspace formula: the most bytes the call took from its
    workspace (the pool's high-water mark) are at most what `sized(*args)` gives for its rows, so an undersized
    formula fails at any row count and pin, not only when a whole chunk overflows its buffer."""

    def __init__(self, kernel, sized) -> None:
        self.kernel, self.sized = kernel, sized

    def __getattr__(self, name: str):
        return getattr(self.kernel, name)

    def __call__(self, *args, **kw):
        self.kernel.lib.workspace_high_water()                  # reset
        y = self.kernel(*args, **kw)
        used, sized = self.kernel.lib.workspace_high_water(), self.sized(*args)
        assert used <= sized, f"a call took {used} workspace bytes, sized for {sized}"
        return y


def random_blocks(name: str, rows: int, k: int, seed: int) -> np.ndarray:
    """Random bytes, with every fp16 scale realistic: |d| log-uniform in [1e-6, 1] (fp16 subnormals below 6e-5),
    random sign, 3 % exact zeros; FP4 scales as `fp4_scales`."""
    rng = np.random.default_rng(seed)
    block, size = BLOCK_SIZES.get(name) or gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType[name.upper()]]
    b = rng.integers(0, 256, size=(rows, k // block, size), dtype=np.uint8)
    if name in FP4:
        n = 4 if name == "nvfp4" else 1
        b[..., :n] = fp4_scales(name, (rows, k // block, n), rng)
        return b.reshape(rows, -1)
    for off in SCALES.get(name, (0,)):
        d = rng.choice([-1.0, 1.0], size=b.shape[:2]) * 10 ** rng.uniform(-6, 0, size=b.shape[:2])
        d[rng.random(d.shape) < 0.03] = 0
        h = d.astype(np.float16).view(np.uint16)
        if name == "iq1_m":
            sc = b[..., off:off + 8].copy().view(np.uint16)
            sc[:] = sc & 0x0FFF | (h[..., None] >> 4 * np.arange(4) & 0xF) << 12
            b[..., off:off + 8] = sc.view(np.uint8)
        else:
            b[..., off:off + 2] = h.view(np.uint8).reshape(*b.shape[:2], 2)
    return b.reshape(rows, -1)


def fp4_scales(name: str, shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    """Realistic FP4 block scales: MXFP4 e8m0 2^-10..2^0; NVFP4 UE4M3 up to 1.0 (0x38), subnormals (0x01..0x07)
    included, 3 % zeros (never a sign bit or NaN: the HF sources refuse them)."""
    if name == "mxfp4":
        return rng.integers(117, 128, size=shape, dtype=np.uint8)
    return np.where(rng.random(shape) < 0.03, 0, rng.integers(1, 0x39, size=shape)).astype(np.uint8)


def hf_fp4(elem: str, out: int, k: int, seed: int, experts: int = 0) -> dict[str, torch.Tensor]:
    """An HF FP4 weight's normalized parts (docs/m06-lowp.md D2), random codes and realistic scales; NVFP4's scale2
    near a real checkpoint's (amax / (6 · 448)), one per expert when `experts`."""
    rng = np.random.default_rng(seed)
    lead = (experts,) if experts else ()
    group = 16 if elem == "nvfp4" else 32
    parts = {"weight": torch.from_numpy(rng.integers(0, 256, size=(*lead, out, k // 2), dtype=np.uint8)),
             "scale": torch.from_numpy(fp4_scales(elem, (*lead, out, k // group), rng))}
    if elem == "nvfp4":
        parts["scale2"] = torch.from_numpy(np.asarray(rng.uniform(1, 2, size=lead) / 2688)).float()
    return parts


def hf_weight(elem: str, parts: dict[str, torch.Tensor]) -> torch.Tensor:
    """The reference's fp32 weight of `hf_fp4` parts ([E, out, in] for a stack: per expert)."""
    if parts["weight"].dim() == 2:
        return hf.dequantize(elem, parts)
    return torch.stack([hf.dequantize(elem, {r: p[e] for r, p in parts.items()}) for e in range(len(parts["weight"]))])


def prepared(kernel, elem: str, parts: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """What the kernel's `prepare` makes of HF parts (on the GPU), checked not to modify them."""
    cuda = {r: p.cuda() for r, p in parts.items()}
    before = {r: p.clone() for r, p in cuda.items()}
    q = kernel.prepare(QTensor(Layout(Scheme(elem), "hf"), (*parts["weight"].shape[:-1], 2 * parts["weight"].shape[-1]),
                               cuda))
    assert all(torch.equal(cuda[r], before[r]) for r in cuda), "prepare modified its input"
    assert q.layout == Layout(Scheme(f"ggml.{elem}"), "gguf.native") and next(iter(q.parts)) == "blocks"
    return dict(q.parts)


def same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    ints = {torch.float32: torch.int32, torch.bfloat16: torch.int16}[a.dtype]
    return a.shape == b.shape and torch.equal(a.view(ints), b.to(a.dtype).view(ints))


def device_blocks(blocks: np.ndarray) -> torch.Tensor:
    """The blocks on the GPU, followed by poison: bytes that decode to NaN scales if a kernel reads past them."""
    buf = torch.full((blocks.size + (1 << 16),), 0xFF, dtype=torch.uint8, device="cuda")
    buf[: blocks.size] = torch.from_numpy(blocks.reshape(-1)).cuda()
    return buf[: blocks.size].view(blocks.shape)


def dequantize(name: str, blocks: np.ndarray) -> torch.Tensor:
    """gguf-py's dequantization; Q2_0's, which gguf-py 0.19 lacks, as ggml-cuda's `dequantize_q2_0` reads the block:
    fp16 d, then 2-bit codes c (4 per byte, from the low bits), value (c − 1)·d."""
    if name == "q2_0":
        b = blocks.reshape(len(blocks), -1, 18)
        d = b[..., :2].copy().view(np.float16).astype(np.float32)
        c = b[..., 2:, None] >> np.arange(0, 8, 2, dtype=np.uint8) & 3
        return torch.from_numpy(((c.reshape(*b.shape[:2], 64) - 1.0).astype(np.float32) * d).reshape(len(b), -1)).cuda()
    return torch.from_numpy(gguf.quants.dequantize(blocks, gguf.GGMLQuantizationType[name.upper()])).cuda()


def activations(rows: int, k: int, seed: int) -> torch.Tensor:
    """Random rows; row 1 all zero and row 2 with one large outlier when there are that many."""
    x = torch.randn(rows, k, generator=torch.Generator().manual_seed(seed)).to(torch.bfloat16).cuda()
    x[1:2] = 0
    x[2:3, k // 3] = OUTLIER
    return x


def check_qlinear(kernel, name: str, blocks: np.ndarray | None, rows_list, parts=None, weight=None,
                  act_dtype: torch.dtype = torch.bfloat16, glu=None, other=None) -> None:
    """The kernel against the q8_1 emulator of each row count's path (the weight: gguf-py's of `blocks`, or the
    given one with its `parts`), its zero and outlier rows on their own, NMSE against the exact product. D6: `glu`,
    the fused SwiGLU on the same weight read as a merged gate/up, at one row against silu(gate) · up of the emulated
    sums; `other`, a (kernel, parts) pair of another type and the same `in`: on the mmvq path one `prepare_input`
    serves both, bitwise as their own calls, and a handle a later `prepare_input` overwrote is not read."""
    k = kernel.k
    weight = dequantize(name, blocks) if weight is None else weight.cuda()
    w = {"blocks": device_blocks(blocks)} if parts is None else parts
    for rows in rows_list:
        path = q8_1.MMVQ if rows <= kernel.mmvq_rows else q8_1.MMQ
        x = activations(rows, k, seed=rows).to(act_dtype)
        y = kernel(x, w)
        assert torch.equal(y, kernel(x, w)), f"{name} rows={rows}: not run-to-run identical"
        ref = q8_1.matmul(x, weight, blocks, name.split("/")[0], path)
        if glu is not None:                                   # the first glu.out rows read as a merged gate/up
            fused = glu(x, {r: p[:glu.out] if r == "blocks" else p for r, p in w.items()})
            assert (fused is None) == (rows != 1), f"{name} rows={rows}: glu fuses one row only"
            if fused is not None:
                gate, up = ref[:, :glu.out].chunk(2, dim=-1)
                r = l0.check(fused, torch.nn.functional.silu(gate) * up, torch.bfloat16, k)
                assert r.passed, f"{name} in={k} glu: {r}"
        if other is not None and path == q8_1.MMVQ:
            (second, second_parts), h = other, kernel.prepare_input(x)
            alone = second(x, second_parts)
            assert torch.equal(kernel(h, w), y) and torch.equal(second(h, second_parts), alone), f"{name} rows={rows}"
            second.prepare_input(activations(rows, k, seed=rows + 100).to(act_dtype))     # overwrites h's buffer
            assert torch.equal(kernel(h, w), y), f"{name} rows={rows}: a stale prepared input was read"
        normal = [i for i in range(rows) if i not in (1, 2)]
        for sel in (normal, [i for i in (1, 2) if i < rows]):
            if sel:
                r = l0.check(y[sel], ref[sel], torch.bfloat16, k)
                assert r.passed, f"{name} in={k} rows={rows} {path}: {r}"
        if rows > 1:
            assert not y[1].any(), f"{name} rows={rows}: zero row"
        if parts is not None and "scale2" in parts and 1 < rows <= kernel.mmvq_rows:   # D11, by value: as the f32
            f32 = torch.empty(rows, kernel.out, device="cuda")                       # multiply it replaces
            kernel.lib.mmvq(kernel.type, parts["blocks"], x, f32, None, kernel.workspace)
            assert torch.equal(y, (f32 * parts["scale2"]).to(y.dtype)), f"{name} rows={rows}: scale2 in mmvq"
        exact = reference.qlinear(x[normal], weight).double()
        nmse = ((y[normal].double() - exact) ** 2).sum() / (exact**2).sum()
        assert nmse <= NMSE, f"{name} in={k} rows={rows} {path}: NMSE {nmse:.2e}"


def test_link(sm):
    """The module resolves every ggml symbol in the vendored code itself (the shim), and its build applied the
    patch (the static library is built from the committed, patched sources)."""
    undefined = subprocess.run(["nm", "-D", "--undefined-only", str(extension.module_path(sm))],
                               capture_output=True, text=True, check=True).stdout
    assert not [s for s in undefined.split() if "ggml" in s or "gguf" in s]
    assert "Holon:" in (extension.CUDA_DIR / "quantize.cu").read_text()


@pytest.mark.parametrize("k", INS)
@pytest.mark.parametrize("name", sorted(TYPES))
def test_qlinear_random_blocks(ctx, name, k):
    kernel = qlinear(ctx, name, k)
    t = kernel.mmvq_rows
    assert 1 <= t <= 8
    rows = sorted({*range(1, 10), 64, 128, 256, CHUNK, CHUNK + 1, 2 * CHUNK + 1, MAX_STEP_TOKENS})
    second = "q6_k" if name == "q4_k" else "q4_k"                           # a consumer of another type (D6)
    other = (qlinear(ctx, second, k, out=64), {"blocks": device_blocks(random_blocks(second, 64, k, seed=1))})
    glu, blocks = qlinear(ctx, name, k, out=GLU_OUT, op="qlinear_glu@1"), random_blocks(name, OUT, k, seed=k)
    check_qlinear(kernel, name, blocks, rows, glu=glu if name in GLU_TYPES else None, other=other)
    x, w = activations(1, k, seed=1), {"blocks": device_blocks(blocks)}
    halves_of_150 = qlinear(ctx, name, k, op="qlinear_glu@1")          # not 4-row blocks: served, and composed
    assert halves_of_150(x, w) is None and (name in GLU_TYPES or glu(x, w) is None)       # the others compose


def test_qlinear_real_tensors(ctx, real):
    tested = {(name, k) for name, k in real if name in TYPES}
    if not tested:
        pytest.skip("no cached Qwen3-8B GGUF file")
    for name, k in sorted(tested):
        kernel = qlinear(ctx, name, k)
        t = kernel.mmvq_rows
        check_qlinear(kernel, name, real[name, k], (1, t, t + 1, 64, CHUNK + 1))


def test_qlinear_f32_out_and_empty(ctx):
    kernel = qlinear(ctx, "q4_k", 4096, out_dtype="float32")
    blocks = random_blocks("q4_k", OUT, 4096, seed=1)
    check_qlinear(kernel, "q4_k", blocks, (1, kernel.mmvq_rows + 1, CHUNK + 1))
    x = torch.empty(0, 4096, dtype=torch.bfloat16, device="cuda")
    assert kernel(x, {"blocks": device_blocks(blocks)}).shape == (0, OUT)


# The classes locks are written for (plans/<model id>/<class>/<toolchain>/N.lock) and those with D2 entries.
PLANS = pathlib.Path(__file__).resolve().parents[2] / "plans"
LOCK_CLASSES = sorted({lock.parent.parent.name for lock in PLANS.rglob("*.lock")} | {c[0] for c in CROSSOVER})


def test_qlinear_dispatch_is_mmvq_rows(sm):
    """D2 (docs/m07-perf.md): the path `QLinear` actually takes per row count (its counted call) is `mmvq_rows`'s, for
    every lock class (the build's fingerprint names the class; llama.cpp's rule is asked at this device's capability)
    on every shape the class has an entry for and a k-quant it has none for; and llama.cpp's rule answers for another
    class's compute capability (Q4_K: mmvq up to 8 rows on Ada, 5 on Blackwell)."""
    lib, here = extension.load(sm), BuildContext(PluginSet(discover()), torch.device("cuda")).fingerprint
    assert [mmvq_rows(dataclasses.replace(here, sm=s), "q4_k", "q8_1", 4096, OUT, lib) for s in (89, 120)] == [8, 5]
    assert LOCK_CLASSES and all(1 <= rows <= 8 for rows in CROSSOVER.values())
    for cls in LOCK_CLASSES:
        ctx = BuildContext(PluginSet(discover()), torch.device("cuda"))
        ctx.fingerprint = dataclasses.replace(here, gpu=cls)
        shapes = {(t, k, out) for c, t, k, out, mode in CROSSOVER if (c, mode) == (cls, "q8_1")} | {("q4_k", 4096, OUT)}
        for name, k, out in sorted(shapes):
            kernel = qlinear(ctx, name, k, out=out).kernel
            want = mmvq_rows(ctx.fingerprint, name, "q8_1", k, out, lib)
            blocks = {"blocks": torch.zeros(out, lib.row_bytes(TYPES[name], k), dtype=torch.uint8, device="cuda")}
            for rows in range(1, 10):
                kernel.calls.clear()
                kernel(activations(rows, k, seed=rows), blocks)
                assert kernel.calls == {f"{'mmvq' if rows <= want else 'mmq'}/q8_1": 1}, (cls, name, k, out, rows)


@pytest.mark.parametrize("name", sorted(TYPES))
def test_mmq_schedules(ctx, sm, name):
    """mmq at CHUNK rows under each of llama.cpp's schedules, asserted from its own choice (`mmq_schedule`:
    0 tiling, 1 stream-k with one block per tile, 2 stream-k with the fixup pass) and from the kernels that launch
    (so a refresh that moves llama.cpp's choice away from our mirror fails here); its bf16 output is its f32 output
    cast (D4), and with NVFP4's `scale2` (an HF layout) the f32 output times scale2, cast."""
    lib, k = extension.load(sm), 4096
    nsm = torch.cuda.get_device_properties().multi_processor_count
    kind, tile_i, _ = lib.mmq_schedule(0, TYPES[name], PREC["q8_1"], OUT, CHUNK)
    assert kind == 2, f"{name}: out={OUT} should take the stream-k fixup"
    x = activations(CHUNK, k, seed=0)
    for out in (OUT, nsm * tile_i, 4096, 12288):
        kind = lib.mmq_schedule(0, TYPES[name], PREC["q8_1"], out, CHUNK)[0]
        assert out != nsm * tile_i or kind in (0, 1), f"{name}: out={out} (tiles a multiple of nsm) takes the fixup"
        kernel, blocks = qlinear(ctx, name, k, out=out), random_blocks(name, out, k, seed=out)
        w = {"blocks": device_blocks(blocks)}
        launched = _launches(kernel, x, w)
        fixup = any("stream_k_fixup" in n for n in launched)
        assert fixup == (kind == 2), f"{name}: out={out} schedule {kind}, fixup launched: {fixup}"
        check_qlinear(kernel, name, blocks, (CHUNK,))
        assert torch.equal(kernel(x, w), qlinear(ctx, name, k, "float32", out=out)(x, w).to(torch.bfloat16)), out
    if name == "nvfp4":                                  # scale2, the fixup forced (out = OUT); native W4A4 on sm_120
        for act_quant in ("a8", "a4") if sm == 120 else ("a8",):
            kernel, f32 = (qlinear(ctx, "nvfp4/hf", k, dtype, act_quant=act_quant) for dtype in ("bfloat16", "float32"))
            parts = prepared(kernel, name, hf_fp4(name, OUT, k, seed=1))
            plain = f32(x, {"blocks": parts["blocks"]}) * parts["scale2"]          # today's: f32, then the multiply
            assert torch.equal(f32(x, parts), plain) and torch.equal(kernel(x, parts), plain.to(torch.bfloat16))
            assert kernel.calls == {f"mmq/{kernel.act['mmq']}": 1}


def test_the_large_call_plants_fire_on_the_fixup(ctx, sm):
    """docs/v1-moe-prefill.md D5's plants (judge `qlinear-partials-bf16`, `qlinear-large-no-fixup`): on a dense call
    of two chunks that takes the stream-k fixup, the counter sees the fixup, and each plant changes the output (the
    partials rounded to bf16; the fixup skipped) and is counted; a one-chunk call is never planted."""
    lib, k, rows = extension.load(sm), 4096, 2 * CHUNK
    assert lib.mmq_schedule(0, TYPES["q4_k"], PREC["q8_1"], OUT, rows)[0] == 2
    kernel = qlinear(ctx, "q4_k", k, "float32")
    w, x = {"blocks": device_blocks(random_blocks("q4_k", OUT, k, seed=3))}, activations(rows, k, seed=3)
    lib.mmq_fixups()
    clean, one = kernel(x, w), kernel(x[:CHUNK], w)
    assert lib.mmq_fixups() == (1, 0)
    for bits in (1, 2):
        lib.mmq_plant(bits)
        try:
            planted, chunk = kernel(x, w), kernel(x[:CHUNK], w)
        finally:
            lib.mmq_plant(0)
        assert lib.mmq_fixups() == (1, 1) and not torch.equal(planted, clean) and torch.equal(chunk, one), bits


def _launches(kernel, x: torch.Tensor, w: dict[str, torch.Tensor]) -> set[str]:
    """The names of the CUDA kernels one call launches."""
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        kernel(x, w)
        torch.cuda.synchronize()
    return {e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA}


@pytest.mark.parametrize("name", sorted(TYPES))
def test_convert_and_get_rows_match_gguf(ctx, sm, name):
    lib = extension.load(sm)
    for k in (768, 4096):
        blocks = random_blocks(name, OUT, k, seed=7)
        ref, w = dequantize(name, blocks), device_blocks(blocks)
        y = torch.empty(OUT, k, device="cuda")
        lib.dequantize(TYPES[name], w, y)
        zero = 0.0 if name == "q4_0" else -0.0   # x + 0.0 maps -0 to +0: Q4_0's one exception (docstring)
        assert same_bits(y + zero, ref + zero), f"{name}: convert differs from gguf-py"
        if name == "nvfp4":                       # get_rows has no NVFP4
            continue
        facts = {"layout": f"ggml.{name}/gguf.native", "vocab": OUT, "hidden": k}
        ids = torch.tensor([0, OUT - 1, 5, 5, 123], device="cuda")
        for dtype in ("float32", "bfloat16"):
            spec = ctx.select_kernel("qembedding@1", site="l0", dtype=dtype, **facts)
            got = spec.impl(ctx, dtype=dtype, **facts)(ids, w)
            assert same_bits(got, ref[ids]), f"{name}: get_rows ({dtype}) differs from gguf-py"


@pytest.mark.parametrize("name", ["iq4_nl", "q4_0", "q5_0", "q8_0"])
def test_get_rows_of_rows_narrower_than_256(ctx, name):
    """Holon: `qembedding` at a row of whole blocks narrower than QK_K (160: PLE's IQ4_NL rows, FN) is gguf-py's."""
    blocks = random_blocks(name, OUT, 160, seed=9)
    ref, w = dequantize(name, blocks), device_blocks(blocks)
    facts = {"layout": f"ggml.{name}/gguf.native", "vocab": OUT, "hidden": 160}
    ids = torch.tensor([0, OUT - 1, 5, 5, 123], device="cuda")
    for dtype in ("float32", "bfloat16"):
        got = ctx.select_kernel("qembedding@1", site="l0", dtype=dtype, **facts).impl(ctx, dtype=dtype, **facts)(ids, w)
        assert same_bits(got, ref[ids]), f"{name}: get_rows ({dtype}) differs from gguf-py"


def test_qembedding_real_table(ctx):
    path = next((p for p in FILES if p.name.endswith("Q4_K_M.gguf")), None)
    if path is None:
        pytest.skip("Qwen3-8B-Q4_K_M.gguf not cached")
    t = next(t for t in gguf.GGUFReader(path).tensors if t.name == "token_embd.weight")
    vocab, hidden = int(t.shape[1]), int(t.shape[0])
    facts = {"layout": f"ggml.{t.tensor_type.name.lower()}/gguf.native", "vocab": vocab, "hidden": hidden,
             "dtype": "bfloat16"}
    kernel = ctx.select_kernel("qembedding@1", site="l0", **facts).impl(ctx, **facts)
    ids = torch.tensor([0, 1, 151643, 151645, vocab - 1, 4096])
    rows = np.array(t.data[ids.numpy()])
    got = kernel(ids.cuda(), device_blocks(np.array(t.data)))
    assert same_bits(got, dequantize(t.tensor_type.name.lower(), rows))


@pytest.mark.parametrize(("name", "use"), [*((n, "qlinear") for n in ("q4_k", "q6_k", "q2_k", "iq3_xxs", "iq2_s",
                                                                       "q8_0", "nvfp4/hf")),
                                           ("q4_k", "glu"), ("nvfp4/hf", "glu"), ("q4_k", "prepared")])
def test_cuda_graph_replays_bitwise(ctx, name, use):
    """A captured call replays bitwise like the eager one, also on new inputs: `qlinear`; the fused gate/up at one row
    (D6, `glu`); one `prepare_input` read by two consumers of the same `in` (D6, `prepared`: a Q4_K and a Q6_K
    linear), whose host-side decision to read the prepared buffer is captured — against their eager calls, each
    quantizing for itself."""
    k = 4096
    out = GLU_OUT if use == "glu" else OUT
    kernel = qlinear(ctx, name, k, out=out, op="qlinear_glu@1" if use == "glu" else "qlinear@2")
    w = prepared(kernel, "nvfp4", hf_fp4("nvfp4", out, k, seed=3)) if name == "nvfp4/hf" else \
        {"blocks": device_blocks(random_blocks(name, out, k, seed=3))}
    second = qlinear(ctx, "q6_k", k, out=64), {"blocks": device_blocks(random_blocks("q6_k", 64, k, seed=4))}

    def call(x, eager=False):
        if use != "prepared":
            return (kernel(x, w),)
        h = x if eager else kernel.prepare_input(x)
        return kernel(h, w), second[0](h, second[1])

    for rows in (1,) if use == "glu" else (1, 2, kernel.mmvq_rows) if use == "prepared" else \
            (1, 2, kernel.mmvq_rows + 1, CHUNK + 1):
        x = activations(rows, k, seed=11)
        eager = call(x, eager=True)
        assert all(e is not None for e in eager), f"{name} {use} rows={rows}: not fused"
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            call(x)                                        # warm-up on the capture stream
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            y = call(x)
        graph.replay()
        assert all(map(torch.equal, y, eager)), f"{name} {use} rows={rows}: replay differs from eager"
        x.copy_(activations(rows, k, seed=12))
        graph.replay()
        assert all(map(torch.equal, y, call(x, eager=True))), f"{name} {use} rows={rows}: replay on new input differs"


def test_pdl_survives_graph_capture(ctx, sm):
    """On sm_90+ llama.cpp launches mmvq's kernels with PDL (`GGML_CUDA_PDL` replaced by "on"); captured by torch,
    they are joined by programmatic edges from the launch-completion port."""
    if sm < 90:
        pytest.skip("PDL needs sm_90+")
    from cuda.bindings import runtime as rt

    kernel = qlinear(ctx, "q4_k", 4096)
    w, x = {"blocks": device_blocks(random_blocks("q4_k", OUT, 4096, seed=3))}, activations(1, 4096, seed=1)
    kernel(x, w)
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        kernel(x, w)
    raw = rt.cudaGraph_t(init_value=graph.raw_cuda_graph())
    n = rt.cudaGraphGetEdges(raw)[-1]
    data = rt.cudaGraphGetEdges(raw, n)[3][:n]
    assert data and all(d.type == rt.cudaGraphDependencyType.cudaGraphDependencyTypeProgrammatic for d in data)


SLACK = 512 * 4 + 3 * 256     # the formulas' spare: the 512-int load-chunk allowance, then three 256-byte pool pads


def reserve(rows: int) -> int:
    """The workspace's native NVFP4 per-row scale (256-byte padded), which q8_1 calls leave unused."""
    return -(-rows * 4 // 256) * 256


def test_the_workspace_formula_is_tight(ctx, sm):
    """Negative control on a J = 128 shape with the stream-k fixup (out 300 at 512 rows): the formula exceeds what
    the call takes by at most its spare, so a too-generous formula (one that would hide an allocation) fails too. The
    test entry reads and resets the high-water mark, at least the q8_1 activations (144 bytes per 128 values)."""
    kernel, k = qlinear(ctx, "q4_k", 4096), 4096
    assert kernel.lib.mmq_schedule(torch.cuda.current_device(), TYPES["q4_k"], PREC["q8_1"], OUT, CHUNK) == \
        (2, 128, 128)
    kernel.kernel(activations(CHUNK, k, seed=0), {"blocks": device_blocks(random_blocks("q4_k", OUT, k, seed=0))})
    used = kernel.lib.workspace_high_water()
    assert used >= CHUNK * k // 128 * 144 and kernel.lib.workspace_high_water() == 0
    assert 0 <= kernel.lib.workspace_bytes(torch.cuda.current_device(), k, CHUNK) - used <= SLACK + reserve(CHUNK)


@pytest.mark.parametrize("elem", FP4)
def test_hf_fp4_prepare_is_lossless_and_scale2_holds_on_every_path(ctx, elem):
    """Acceptance 4 on random weights: `prepare`'s ggml blocks, dequantized by gguf-py × scale2, equal the reference
    by value, and are llama.cpp's converter's byte for byte (with the pin's checkout, git-ignored under spikes/); the
    repack plants change them. Then `scale2` (one value
    per tensor: mmvq's epilogue at every row count (D11), mmq's store (D4))
    against the q8_1 emulator on the reference weight, both output dtypes and both input dtypes."""
    plants = {"nvfp4": ("nvfp4-repack-nibbles-swapped", "nvfp4-repack-scales-permuted"),
              "mxfp4": ("mxfp4-repack-scale-off-by-one",)}[elem]
    theirs = converter(LLAMA) if (LLAMA / "conversion" / "base.py").is_file() else None
    for k, out_dtype, act_dtype in ((768, "bfloat16", torch.bfloat16), (4096, "float32", torch.float32)):
        kernel, parts = qlinear(ctx, f"{elem}/hf", k, out_dtype, act_dtype=str(act_dtype)[6:]), hf_fp4(elem, OUT, k, k)
        weight, got = hf_weight(elem, parts), prepared(kernel, elem, parts)
        assert set(got) == ({"blocks", "scale2"} if elem == "nvfp4" else {"blocks"})
        assert torch.equal(dequantize_blocks(elem, got["blocks"], got.get("scale2")), weight)     # by value
        if theirs is not None:
            assert torch.equal(got["blocks"].cpu(), theirs(elem, parts["weight"], parts["scale"]))
        for plant in plants:
            with PLANTS[plant]():
                bad = prepared(kernel, elem, parts)["blocks"]
            assert not torch.equal(dequantize_blocks(elem, bad, got.get("scale2")), weight), plant
        t = kernel.mmvq_rows
        glu = qlinear(ctx, f"{elem}/hf", k, out_dtype, GLU_OUT, act_dtype=str(act_dtype)[6:], op="qlinear_glu@1")
        check_qlinear(kernel, f"{elem}/hf", None, (*range(1, t + 2), 64, CHUNK + 1), got, weight, act_dtype,
                      glu if elem in GLU_TYPES else None)
        assert set(kernel.calls) == {"mmvq/q8_1", "mmq/q8_1"}


def test_the_mmq_precision_is_the_spec_s(ctx, sm):
    """llama.cpp's precision for each type: q8_1 when asked for it; at the default (dst's op_params[3] zero, plant
    N6) native W4A4 for the FP4 types on sm_120, which the pre-launch check refuses for the q8_1 spec, and q8_1 below
    it, which passes. `quant.act = a4` selects the native spec on sm_120 only (§4.7); `a8` never does."""
    lib, device = extension.load(sm), torch.cuda.current_device()
    for name, t in TYPES.items():
        assert lib.mmq_prec(device, t, PREC["q8_1"]) == PREC["q8_1"]
        assert lib.mmq_prec(device, t, 0) == PREC["fp4" if sm == 120 and name in FP4 else "q8_1"], name
    with PLANTS["ggml-prec-default"]():
        if sm == 120:
            with pytest.raises(RuntimeError, match="spec's mode 'q8_1'"):
                qlinear(ctx, "nvfp4/hf", 4096)
        else:
            qlinear(ctx, "nvfp4/hf", 4096)
    for name, act_quant in (("nvfp4/hf", "a4"), ("mxfp4", "a4"), ("nvfp4/hf", "a8"), ("q4_k", "a4")):
        kernel = qlinear(ctx, name, 4096, act_quant=act_quant)
        native = sm == 120 and act_quant == "a4" and name.split("/")[0] in FP4
        assert kernel.act == (NATIVE_FP4 if native else Q8_1), (name, act_quant)


def fp4_activations(rows: int, k: int, seed: int) -> torch.Tensor:
    """f32 rows with row scales from 1e-3 to 1e3, row 1 all zero and row 2 with one large outlier."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, k, generator=gen) * 10 ** torch.empty(rows, 1).uniform_(-3, 3, generator=gen)
    x[1:2] = 0
    x[2:3, k // 3] = OUTLIER * 100
    return x.cuda()


def native_codes(lib, elem: str, x: torch.Tensor):
    """llama.cpp's native FP4 activations of f32 `x` (the test entry): (codes, scales, values f64, row scales)."""
    blocks = torch.empty(-(-x.shape[1] // 512) * 2, len(x), 144, dtype=torch.uint8, device=x.device)
    row_scale = torch.zeros(len(x), device=x.device)
    lib.quantize_fp4(TYPES[elem], x, blocks, row_scale)
    return (*w4a4.unpack(elem, blocks, row_scale, x.shape[1]), row_scale)


@pytest.mark.parametrize("elem", FP4)
def test_native_fp4_activations_are_the_emulator_s(sm, elem):
    """The codes llama.cpp's FP4 quantizer gives against `holon_judge.w4a4`: at most 1e-3 differ, every differing
    scale group inside a fast-math tie class (MXFP4's quantizer runs on every class; NVFP4's on sm_120 only), and
    NVFP4's row scale is amax / (6 · 448) within the tie class's 4 ulp."""
    if elem == "nvfp4" and sm != 120:
        pytest.skip("llama.cpp's NVFP4 activation quantizer is Blackwell's")
    lib = extension.load(sm)
    for rows, k in ((64, 768), (300, 4096), (7, 12288)):
        x = fp4_activations(rows, k, seed=rows)
        codes, scales, values, row_scale = native_codes(lib, elem, x)
        differ, unexplained = w4a4.check_codes(elem, x, codes, scales, row_scale)
        assert unexplained == 0 and differ <= 1e-3, (rows, k, differ, unexplained)
        assert not values[1].any() and values.isfinite().all()
        if elem == "nvfp4":
            want = (x.abs().amax(1).double() / (6 * 448)).float()
            assert (ulps(row_scale, want) <= w4a4.NUDGE).all(), (row_scale, want)


@pytest.mark.parametrize("elem", FP4)
@pytest.mark.parametrize("act_dtype", [torch.bfloat16, torch.float32])
def test_native_w4a4_against_its_own_codes(ctx, sm, elem, act_dtype):
    """sm_120, `quant.act = a4`: mmq at native W4A4 against the f64 product of its own decoded activation codes
    (exact operands: the budget is the reduction's), mmvq at q8_1 against the emulator; `prepare` appends the zeroed
    tail at `in` 768, without which (plant M4) MXFP4's NaN scales reach the output."""
    if sm != 120:
        pytest.skip("native W4A4 is Blackwell's")
    lib = extension.load(sm)
    for k in (768, 4096):
        kernel = qlinear(ctx, f"{elem}/hf", k, act_dtype=str(act_dtype)[6:], act_quant="a4")
        assert kernel.act == NATIVE_FP4
        parts = hf_fp4(elem, OUT, k, seed=k)
        weight, got = hf_weight(elem, parts).cuda().double(), prepared(kernel, elem, parts)
        blocks, tail = got["blocks"], kernel.tail
        assert (tail > 0) == (k % 512 != 0)
        storage = torch.empty(0, dtype=torch.uint8, device="cuda").set_(blocks.untyped_storage())
        assert len(storage) == blocks.numel() + tail and not storage[blocks.numel():].any()
        t = kernel.mmvq_rows
        for rows in (1, t, t + 1, 64, CHUNK + 1, MAX_STEP_TOKENS):
            x = activations(rows, k, seed=rows).to(act_dtype)
            y = kernel(x, got)
            if rows <= t:
                ref = q8_1.matmul(x, weight, None, elem, q8_1.MMVQ)
            else:
                ref = torch.cat([native_codes(lib, elem, x[a:a + CHUNK].float())[2].cuda() @ weight.T
                                 for a in range(0, rows, CHUNK)])
            r = l0.check(y, ref, torch.bfloat16, k)
            assert r.passed, f"{elem} in={k} rows={rows}: {r}"
        assert set(kernel.calls) == {"mmvq/q8_1", "mmq/fp4"}
        if elem == "mxfp4" and tail:
            with PLANTS["fp4-tail-not-zeroed"]():
                planted = prepared(kernel, elem, parts)
            assert kernel(activations(CHUNK, k, seed=1).to(act_dtype), planted).isnan().any()
