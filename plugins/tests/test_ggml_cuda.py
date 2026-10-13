"""L0 for kernels.ggml_cuda (M0.1 design §4.6): llama.cpp's quantized kernels against gguf-py and the q8_1 emulator.

- `qlinear`, every supported type, `in` ∈ {768, 2304, 4096, 12288} on random-byte blocks with realistic fp16 scales
  (negative, zero, subnormal, up to 1) and real tensors from the cached Qwen3-8B files; `out` = 300 (not a multiple
  of the mmq tile); the bytes right after the weights are poisoned (NaN scales). Rows 1..9 (every mmvq column
  count, and mmq above the type's threshold on this device), 64, 128, 256, 512, 513, 1025, max_step_tokens;
  activations hold an all-zero row and a row with one large outlier. Per element `l0.check(bf16, in)` against the
  emulated reference of the path taken (judge `q8_1`, MMA kernels: cc >= 7.5), NMSE <= 5e-4 against the
  unquantized oracle, run-to-run identical.
- mmq's schedules at 512 rows, asserted from llama.cpp's own choice (tile I x J, nsm): stream-k with the fixup
  pass (out = 300), one block per tile (out = nsm * I), and the model's out 4096 and 12288.
- `convert` and `get_rows` (`qembedding`) against `gguf.quants.dequantize`: bit-identical for every type. FMA
  decision, measured on sm_86: with llama.cpp's flags (`-use_fast_math`, FMA contraction on) every value matches
  gguf-py bit for bit, affine types (Q4_1, Q5_1, Q2_K, Q4_K, Q5_K, IQ1_S) included, so nothing is built with
  `-fmad=false`; the one exception is the sign of zero in Q4_0's convert ((8 - 8) * d is +0 for negative d).
- CUDA graphs: captured calls replay bitwise like eager ones (PDL launches on sm_120a), also on new inputs.
- Workspace: every call takes at most what `workspace_bytes` sizes for its rows (the pool's high-water mark,
  `Sized`), so an undersized formula fails at any row count; `mmq_schedule` refuses FP4 types.

Skips without a GPU; builds the extension (prebuild) into the user cache if it is missing.
"""
import os
import pathlib
import subprocess
import sys

import gguf
import numpy as np
import pytest
import torch
from holon.core.build import BuildContext
from holon.core.plugins import PluginSet, discover
from holon_judge import l0, q8_1
from holon_plugins.kernels.ggml_cuda import extension
from holon_plugins.kernels.ggml_cuda.plugin import CHUNK, TYPES
from holon_std.ops import reference
from huggingface_hub.constants import HF_HUB_CACHE

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

INS = (768, 2304, 4096, 12288)
OUT = 300
MAX_STEP_TOKENS = 4096          # the engine's default step budget
NMSE = 5e-4                     # llama.cpp test-backend-ops MUL_MAT threshold
OUTLIER = 1000.0
GGUF_DIR = pathlib.Path(HF_HUB_CACHE) / "models--unsloth--Qwen3-8B-GGUF" / "snapshots"
FILES = sorted(GGUF_DIR.glob("*/*.gguf"))
# Byte offsets of each block's fp16 scales (d, and dmin / m where the type has one); d at 0 otherwise.
SCALES = {"q4_1": (0, 2), "q5_1": (0, 2), "q2_k": (80, 82), "q3_k": (108,), "q4_k": (0, 2), "q5_k": (0, 2),
          "q6_k": (208,)}


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


def qlinear(ctx, name: str, k: int, out_dtype: str = "bfloat16", out: int = OUT):
    facts = {"layout": f"ggml.{name}/gguf.native", "in": k, "out": out, "act_dtype": "bfloat16",
             "out_dtype": out_dtype, "act_quant": "a8"}
    spec = ctx.select_kernel("qlinear@2", site="l0", **facts)
    assert spec.provider == "kernels.ggml_cuda" and {"graph_safe", "deterministic"} <= spec.props
    kernel, device = spec.impl(ctx, **facts), torch.cuda.current_device()
    return Sized(kernel, lambda x, parts: kernel.lib.workspace_bytes(device, k, min(len(x), CHUNK)))


class Sized:
    """A kernel instance whose every call is held to its workspace formula: the most bytes the call took from its
    workspace (the pool's high-water mark) are at most what `sized(*args)` gives for its rows, so an undersized
    formula fails at any row count and pin, not only when a whole chunk overflows its buffer."""

    def __init__(self, kernel, sized) -> None:
        self.kernel, self.sized = kernel, sized

    def __getattr__(self, name: str):
        return getattr(self.kernel, name)

    def __call__(self, *args):
        self.kernel.lib.workspace_high_water()                  # reset
        y = self.kernel(*args)
        used, sized = self.kernel.lib.workspace_high_water(), self.sized(*args)
        assert used <= sized, f"a call took {used} workspace bytes, sized for {sized}"
        return y


def random_blocks(name: str, rows: int, k: int, seed: int) -> np.ndarray:
    """Random bytes, with every fp16 scale realistic: |d| log-uniform in [1e-6, 1] (fp16 subnormals below 6e-5),
    random sign, 3 % exact zeros."""
    rng = np.random.default_rng(seed)
    qtype = gguf.GGMLQuantizationType[name.upper()]
    block, size = gguf.GGML_QUANT_SIZES[qtype]
    b = rng.integers(0, 256, size=(rows, k // block, size), dtype=np.uint8)
    for off in SCALES.get(name, (0,)):
        d = rng.choice([-1.0, 1.0], size=b.shape[:2]) * 10 ** rng.uniform(-6, 0, size=b.shape[:2])
        d[rng.random(d.shape) < 0.03] = 0
        b[..., off:off + 2] = d.astype(np.float16).view(np.uint8).reshape(*b.shape[:2], 2)
    return b.reshape(rows, -1)


def same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    ints = {torch.float32: torch.int32, torch.bfloat16: torch.int16}[a.dtype]
    return a.shape == b.shape and torch.equal(a.view(ints), b.to(a.dtype).view(ints))


def device_blocks(blocks: np.ndarray) -> torch.Tensor:
    """The blocks on the GPU, followed by poison: bytes that decode to NaN scales if a kernel reads past them."""
    buf = torch.full((blocks.size + (1 << 16),), 0xFF, dtype=torch.uint8, device="cuda")
    buf[: blocks.size] = torch.from_numpy(blocks.reshape(-1)).cuda()
    return buf[: blocks.size].view(blocks.shape)


def dequantize(name: str, blocks: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(gguf.quants.dequantize(blocks, gguf.GGMLQuantizationType[name.upper()])).cuda()


def activations(rows: int, k: int, seed: int) -> torch.Tensor:
    """Random rows; row 1 all zero and row 2 with one large outlier when there are that many."""
    x = torch.randn(rows, k, generator=torch.Generator().manual_seed(seed)).to(torch.bfloat16).cuda()
    x[1:2] = 0
    x[2:3, k // 3] = OUTLIER
    return x


def check_qlinear(kernel, name: str, blocks: np.ndarray, rows_list) -> None:
    k, weight, w = kernel.k, dequantize(name, blocks), {"blocks": device_blocks(blocks)}
    for rows in rows_list:
        path = q8_1.MMVQ if rows <= kernel.mmvq_rows else q8_1.MMQ
        x = activations(rows, k, seed=rows)
        y = kernel(x, w)
        assert torch.equal(y, kernel(x, w)), f"{name} rows={rows}: not run-to-run identical"
        ref = q8_1.matmul(x, weight, blocks, name, path)
        normal = [i for i in range(rows) if i not in (1, 2)]
        for sel in (normal, [i for i in (1, 2) if i < rows]):
            if sel:
                r = l0.check(y[sel], ref[sel], torch.bfloat16, k)
                assert r.passed, f"{name} in={k} rows={rows} {path}: {r}"
        if rows > 1:
            assert not y[1].any(), f"{name} rows={rows}: zero row"
        exact = reference.qlinear(x[normal], weight).double()
        nmse = ((y[normal].double() - exact) ** 2).sum() / (exact**2).sum()
        assert nmse <= NMSE, f"{name} in={k} rows={rows} {path}: NMSE {nmse:.2e}"


def test_link(sm):
    """The module resolves every ggml symbol in the vendored code itself (the shim), and its build applied the
    patch (the static library is built from the committed, patched sources)."""
    import subprocess

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
    check_qlinear(kernel, name, random_blocks(name, OUT, k, seed=k), rows)


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


@pytest.mark.parametrize("name", sorted(TYPES))
def test_mmq_schedules(ctx, sm, name):
    """mmq at CHUNK rows under each of llama.cpp's schedules, asserted from its own choice (`mmq_schedule`:
    0 tiling, 1 stream-k with one block per tile, 2 stream-k with the fixup pass) and from the kernels that launch
    (so a refresh that moves llama.cpp's choice away from our mirror fails here)."""
    lib, k = extension.load(sm), 4096
    nsm = torch.cuda.get_device_properties().multi_processor_count
    kind, tile_i, _ = lib.mmq_schedule(0, TYPES[name], OUT, CHUNK)
    assert kind == 2, f"{name}: out={OUT} should take the stream-k fixup"
    for out in (OUT, nsm * tile_i, 4096, 12288):
        kind = lib.mmq_schedule(0, TYPES[name], out, CHUNK)[0]
        assert out != nsm * tile_i or kind in (0, 1), f"{name}: out={out} (tiles a multiple of nsm) takes the fixup"
        kernel, blocks = qlinear(ctx, name, k, out=out), random_blocks(name, out, k, seed=out)
        launched = _launches(kernel, activations(CHUNK, k, seed=0), {"blocks": device_blocks(blocks)})
        fixup = any("stream_k_fixup" in n for n in launched)
        assert fixup == (kind == 2), f"{name}: out={out} schedule {kind}, fixup launched: {fixup}"
        check_qlinear(kernel, name, blocks, (CHUNK,))


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
        facts = {"layout": f"ggml.{name}/gguf.native", "vocab": OUT, "hidden": k}
        ids = torch.tensor([0, OUT - 1, 5, 5, 123], device="cuda")
        for dtype in ("float32", "bfloat16"):
            spec = ctx.select_kernel("qembedding@1", site="l0", dtype=dtype, **facts)
            got = spec.impl(ctx, dtype=dtype, **facts)(ids, w)
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


@pytest.mark.parametrize("name", ["q4_k", "q6_k", "q2_k", "iq3_xxs", "iq2_s", "q8_0"])
def test_cuda_graph_replays_bitwise(ctx, name):
    k = 4096
    kernel = qlinear(ctx, name, k)
    w = {"blocks": device_blocks(random_blocks(name, OUT, k, seed=3))}
    for rows in (1, kernel.mmvq_rows + 1, CHUNK + 1):
        x = activations(rows, k, seed=11)
        eager = kernel(x, w)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            kernel(x, w)                                   # warm-up on the capture stream
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            y = kernel(x, w)
        graph.replay()
        assert torch.equal(y, eager), f"{name} rows={rows}: replay differs from eager"
        x.copy_(activations(rows, k, seed=12))
        graph.replay()
        assert torch.equal(y, kernel(x, w)), f"{name} rows={rows}: replay on new input differs"


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


def test_the_workspace_formula_is_tight(ctx, sm):
    """Negative control on a J = 128 shape with the stream-k fixup (out 300 at 512 rows): the formula exceeds what
    the call takes by at most its spare, so a too-generous formula (one that would hide an allocation) fails too. The
    test entry reads and resets the high-water mark, at least the q8_1 activations (144 bytes per 128 values)."""
    kernel, k = qlinear(ctx, "q4_k", 4096), 4096
    assert kernel.lib.mmq_schedule(torch.cuda.current_device(), TYPES["q4_k"], OUT, CHUNK) == (2, 128, 128)
    kernel.kernel(activations(CHUNK, k, seed=0), {"blocks": device_blocks(random_blocks("q4_k", OUT, k, seed=0))})
    used = kernel.lib.workspace_high_water()
    assert used >= CHUNK * k // 128 * 144 and kernel.lib.workspace_high_water() == 0
    assert 0 <= kernel.lib.workspace_bytes(torch.cuda.current_device(), k, CHUNK) - used <= SLACK


def test_mmq_schedule_refuses_fp4_types(sm):
    """Their activation precision is chosen per call (M0.6 A3 passes it): the schedule aborts rather than guess
    (shown on NVFP4, GGML_TYPE 40; MXFP4 takes the same check, and A3 replaces both)."""
    code = ("from holon_plugins.kernels.ggml_cuda import extension; "
            f"extension.load({sm}).mmq_schedule(0, 40, 300, 512)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    assert r.returncode != 0 and "an FP4 type needs its precision" in r.stderr, r.stderr[-2000:]
