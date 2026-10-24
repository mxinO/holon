"""L0 for kernels.ggml_cuda's MoE kernels (M0.2 design §4.3, §4.6): `linear_experts` (llama.cpp's MUL_MAT_ID, mmvq
and mmq with ids) against the q8_1 emulator, `moe_topk` and `moe_combine` against the std references.

- Accuracy: every supported type on both paths (1, t, t + 1 and 64 tokens; t the type's mmvq limit), the merged
  gate/up stack (one input row per token, bf16) and down (one input per slot, f32, distinct per slot): per element
  `l0.check(bf16, in)` against `q8_1.linear_experts` on the path the plugin takes for each 512-token chunk;
  random-byte blocks with realistic scales (M0.1's), a zero and an outlier token. Up to 2049 tokens on Q4_K and
  IQ3_XXS (below): over millions of outputs IQ1_S's mmq residual tail (its weights are `d·(g + 1)` plus a cancelling
  offset term, which magnifies the quantizer's fast-math rounding the emulator leaves to the budget) reaches the
  budget, in dense `qlinear` as well. `l0.check` models a reduction's error; SwiGLU multiplies two of them, so the
  `glu="swiglu"` output is checked against our SwiGLU of the checked `glu="none"` output instead: bitwise where mmq
  ran, within the fp32 budget for mmvq's fused gate; our SwiGLU kernel itself is checked against f64.
- Merge: the merged stack against separate gate and up stacks, bitwise on mmvq and on mmq without the stream-k fixup.
- Workspace: every call within `workspace_bytes_id` for its tokens (`test_ggml_cuda.Sized`, the high-water mark).
- Routings, at E = 128, k = 8: uniform; every token on the same 8 experts at 512 and 513 tokens (several column
  tiles inside one expert); half the experts empty; ids 0 and E−1 in every row with poisoned bytes before and after
  the stack. Token counts 0, 1, t, t + 1, 9, 64, 512, 513, 1025, 2049 (t: the type's mmvq limit); the two paths
  agree across the limit. Mixed types in one std.moe layer at 7 tokens (sm_86: IQ3_S gate/up on mmq, IQ4_XS down on
  mmvq); ids straight from `moe_topk`; CUDA graphs replayed with routing unlike the capture's at every decode
  bucket.
- Placement (docs/v1-offload.md D4): mmq with ids always tiles, over a compact list of the active slots (or every
  slot, where that is at most 64 more), so an expert's rows are bitwise the same in arenas of 12, 13, 16, 74 and 128
  slots under any placement, at every tile width, and in compacted launches padded with null-slot rows to the type's
  mmvq limit + 1, on both id helpers (per slot, compact; the SM's `PASS_PROVEN` types; gate/up and down).
- Violated preconditions set their bit in the kernel's `errors` word and leave only their own rows uncomputed: an id
  out of range, more slots than the binding's bound, a slot repeated inside rows beyond mmq's column tiles (within
  them it is computed; llama.cpp's helper instead left stale entries that wrote over a real row, which M0.2's padding
  plant relied on; retired, docs/m02-moe.md §4.6).
- `moe_topk`: the reference's picks up to near-ties (fast-math `__expf` moves probabilities by a few ulps), its
  weights; ±inf and all-NaN rows give unique in-range ids; token counts off the kernel's 8-row blocks.
- `moe_combine`: within 1 bf16 ulp of the reference, plus the f32 accumulation's rounding (the kernel's FMA) for
  sums that cancel; its `add` bitwise `residual + out` (cancelling and subnormal residuals too).
- Precision (f64 references written out here): our f32 SwiGLU within the fp32 budget, which a bf16 SwiGLU misses,
  with per-expert gate and up scales too; a std.moe layer binds `linear_experts` with act_dtype bf16 and out_dtype
  f32, and its expert outputs are f32.
- NVFP4 `scale2` per expert (docs/m06-lowp.md §4.5): HF stacks through `prepare`, the merged gate/up stack's [2, E]
  rows (distinct per expert and half) and down's [E], on both paths: down against the emulator in f64 on the
  reference weight, and bitwise the unscaled stack's output times each slot's expert scale; gate/up's SwiGLU against
  ours on the unscaled stack's output with the scales (bitwise on mmq); plants N4 (scale by slot) and N5 (scale after
  SwiGLU) fail them. sm_120, `quant.act = a4`: native W4A4 mmq with ids against the f64 product of its own decoded
  activation codes, gate/up (bf16, one row per token) and down (f32, one per slot).

Skips without a GPU; conftest builds the extension (prebuild) into the user cache if it is missing.
"""
import dataclasses
import math
import re
import types
from unittest import mock

import numpy as np
import pytest
import torch
from holon.api import Layout, QTensor, Ref, Scheme
from holon.core.build import BuildContext
from holon.core.executor import BUCKETS
from holon.core.plugins import PluginSet, discover
from holon_judge import l0, q8_1
from holon_judge.planted import PLANTS
from holon_plugins.kernels.ggml_cuda import extension
from holon_plugins.kernels.ggml_cuda.plugin import (
    CHUNK,
    FP4,
    ID_ERRORS,
    IDS_J_TOKENS,
    IDS_J_TOKENS_DEFAULT,
    NATIVE_FP4,
    PASS_PROVEN,
    PASS_ROWS,
    PREC,
    TYPES,
    Prepared,
    q8_1_bytes,
)
from holon_std.moe import reference
from holon_testing.pass_rows import check_pass_rows

from .test_ggml_cuda import (
    SLACK,
    Sized,
    activations,
    dequantize,
    hf_fp4,
    hf_weight,
    native_codes,
    prepared,
    random_blocks,
    reserve,
)

pytestmark = [pytest.mark.l0, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")]
PREBUILT = ("ggml_cuda",)  # CUDA extensions built if missing (conftest)

E, K = 32, 8                    # the type sweep's stacks (E = 128, Qwen3-30B-A3B's, for routings and graphs)
EQ = 128
H, INTER = 2048, 768            # hidden (gate/up in, down out) and expert width: Qwen3-30B-A3B's
POISON = 1 << 16                # bytes of 0xFF (NaN fp16 scales) on each side of a stack
SIZES = (0, 1, 9, 64, CHUNK, CHUNK + 1, 2 * CHUNK + 1, 4 * CHUNK + 1)
TIE = 1e-5                      # relative: two probabilities this close may swap under `__expf`


@pytest.fixture(scope="module")
def sm() -> int:
    major, minor = torch.cuda.get_device_capability()
    sm = 10 * major + minor
    assert sm >= 75, "q8_1.MMQ emulates llama.cpp's MMA kernels (turing_mma_available)"
    return sm


@pytest.fixture(scope="module")
def ctx(sm) -> BuildContext:
    return BuildContext(PluginSet(discover()), torch.device("cuda"))


@pytest.fixture(scope="module")
def lib(sm):
    return extension.load(sm)


@dataclasses.dataclass
class Stack:
    """An expert stack [E·out, row_bytes] of one ggml type: its blocks, on the GPU between poisoned bytes, and
    dequantized by gguf-py [E, out, in]."""

    name: str
    blocks: np.ndarray
    w: torch.Tensor
    weight: torch.Tensor
    _offs: dict[int, torch.Tensor] = dataclasses.field(default_factory=dict)

    def offs(self, path: str) -> torch.Tensor | None:
        """The emulator's offsets for `path`, computed once."""
        g = q8_1.blocking(self.name, path).sum_group
        if g and g not in self._offs:
            self._offs[g] = q8_1.offsets(self.blocks, self.name, g).view(*self.weight.shape[:2], -1).cuda()
        return self._offs.get(g)

    @property
    def parts(self) -> dict[str, torch.Tensor]:
        return {"blocks": self.w.view(len(self.weight), -1, self.w.shape[-1])}       # per expert (the contract)


def stack(name: str, experts: int, out: int, k: int, seed: int, blocks: np.ndarray | None = None) -> Stack:
    blocks = random_blocks(name, experts * out, k, seed) if blocks is None else blocks
    buf = torch.full((blocks.size + 2 * POISON,), 0xFF, dtype=torch.uint8, device="cuda")
    buf[POISON:POISON + blocks.size] = torch.from_numpy(blocks.reshape(-1)).cuda()
    w = buf[POISON:POISON + blocks.size].view(blocks.shape)
    return Stack(name, blocks, w, dequantize(name, blocks).view(experts, out, k))


def experts(ctx, name: str, k: int, out: int, rows_per_token: int, glu: str = "none", n: int = E,
            act_dtype: str = "bfloat16", act_quant: str = "a8"):
    """A `linear_experts` of ggml type `name` (or of an HF layout, `"nvfp4/hf"`), held to its workspace formula."""
    facts = {"layout": name if "/" in name else f"ggml.{name}/gguf.native", "experts": n, "in": k, "out": out,
             "top_k": K, "rows_per_token": rows_per_token, "glu": glu, "act_dtype": act_dtype, "out_dtype": "float32",
             "act_quant": act_quant}
    spec = ctx.select_kernel("linear_experts@2", site="l0", **facts)
    assert spec.provider == "kernels.ggml_cuda" and {"graph_safe", "deterministic"} <= spec.props
    kernel, device = spec.impl(ctx, **facts), torch.cuda.current_device()
    return Sized(kernel, lambda x, parts, ids: kernel.lib.workspace_bytes_id(device, k, min(len(ids), kernel.pass_rows),
                                                                            K, kernel.slots))


def uniform(tokens: int, n: int, seed: int) -> torch.Tensor:
    """k distinct random experts per token."""
    return torch.rand(tokens, n, generator=torch.Generator().manual_seed(seed)).argsort(-1)[:, :K].int().cuda()


def gate_up_input(tokens: int, seed: int) -> torch.Tensor:
    """[T, 1, H] bf16: token 1 all zero, token 2 with one large outlier (when there are that many)."""
    return activations(tokens, H, seed)[:, None]


def layer_input(tokens: int, seed: int) -> torch.Tensor:
    """[T, H] bf16 hidden states for gate/up followed by down: random-byte stacks have weights up to ~1e3, so unit
    inputs would give down inputs past what q8_1's fp16 scale holds (amax/127 ≤ 65504)."""
    return (1e-3 * torch.randn(tokens, H, generator=torch.Generator().manual_seed(seed))).bfloat16().cuda()


def down_input(tokens: int, seed: int) -> torch.Tensor:
    """[T, k, INTER] f32, a distinct row per slot (as down reads the SwiGLU output of its own slot)."""
    return torch.randn(tokens, K, INTER, generator=torch.Generator().manual_seed(seed)).cuda()


def emulated(kernel, s: Stack, x: torch.Tensor, ids: torch.Tensor, glu: str = "none") -> torch.Tensor:
    """The emulator on the path the plugin takes for each 512-token chunk."""
    chunks = []
    for a in range(0, len(ids), CHUNK):
        path = q8_1.MMVQ if len(ids[a:a + CHUNK]) <= kernel.mmvq_tokens else q8_1.MMQ
        chunks.append(q8_1.linear_experts(x[a:a + CHUNK], s.weight, s.blocks, ids[a:a + CHUNK], s.name, path, glu,
                                          s.offs(path)))
    return torch.cat(chunks) if chunks else torch.zeros(0, K, s.weight.shape[1], dtype=torch.float64)


def launched(fn) -> set[str]:
    """The names of the CUDA kernels `fn()` launches."""
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return {e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA}


def check(y: torch.Tensor, ref: torch.Tensor, k: int, what: str, special: bool = False) -> None:
    """Per element under `l0.check(bf16, k)`; with `special` (a `gate_up_input`) its zero and outlier tokens on
    their own."""
    assert y.dtype == torch.float32 and y.shape == ref.shape, what
    special = [t for t in (1, 2) if t < len(y)] if special else []
    normal = [t for t in range(len(y)) if t not in special]
    for sel in (normal, special):
        if sel:
            r = l0.check(y[sel], ref[sel], torch.bfloat16, k)
            assert r.passed, f"{what}: {r}"
    if special:
        assert not y[1].any(), f"{what}: zero token"


def check_swiglu(lib, fused, none, s: Stack, x: torch.Tensor, ids: torch.Tensor, what: str,
                 fused_parts: dict | None = None) -> None:
    """`glu="swiglu"` against our SwiGLU of the `glu="none"` output: bitwise where mmq ran, within the fp32 budget
    where mmvq's fused gate did. `fused_parts`: the fused kernel's (the stack's with its [2, E] scale2), the scales
    then applied by ours."""
    fused_parts = s.parts if fused_parts is None else fused_parts
    y, gate_up = fused(x, fused_parts, ids), none(x, s.parts, ids)
    ours = torch.empty_like(y)
    scale2 = fused_parts.get("scale2")
    if len(y):
        lib.swiglu(gate_up, ours, *((None,) * 3 if scale2 is None else (scale2[0], scale2[1], ids.view(-1))))
    for a in range(0, len(ids), CHUNK):
        c = slice(a, a + CHUNK)
        if len(ids[c]) > fused.mmvq_tokens:
            assert torch.equal(y[c], ours[c]), f"{what} tokens {a}+: mmq's SwiGLU differs from ours"
        else:
            r = l0.check(y[c], ours[c], torch.float32, 1)
            assert r.passed, f"{what} tokens {a}+: the fused gate vs ours: {r}"


def test_link(sm):
    """The new vendored files are compiled in and patched: every ggml symbol resolves in the shim (nm), and the
    patch lines are there."""
    import subprocess

    undefined = subprocess.run(["nm", "-D", "--undefined-only", str(extension.module_path(sm))],
                               capture_output=True, text=True, check=True).stdout
    assert not [s for s in undefined.split() if "ggml" in s or "gguf" in s]
    for f in ("topk-moe.cu", "moe-weighted-reduction.cu", "mmq.cu", "mmq.cuh", "mmq-load-tiles.cuh",
              "mmq-config-ampere.cuh", "template-instances/mmq-instance-iq1_m.cu"):
        assert "Holon:" in (extension.CUDA_DIR / f).read_text()


@pytest.mark.parametrize("name", sorted(TYPES))
def test_linear_experts_every_type(ctx, lib, name):
    gate_up, fused = experts(ctx, name, H, 2 * INTER, 1), experts(ctx, name, H, 2 * INTER, 1, "swiglu")
    down = experts(ctx, name, INTER, H, K)
    t = gate_up.mmvq_tokens
    assert 1 <= t <= 8 and fused.mmvq_tokens == down.mmvq_tokens == t
    assert fused.glu_store, f"{name}: D4's store not taken at out {2 * INTER}"       # every type, on every MMA SM
    sg, sd = stack(name, E, 2 * INTER, H, seed=1), stack(name, E, H, INTER, seed=2)
    for tokens in (1, t, t + 1, 64):
        ids, x, h = uniform(tokens, E, seed=tokens), gate_up_input(tokens, seed=tokens), down_input(tokens, tokens)
        check(gate_up(x, sg.parts, ids), emulated(gate_up, sg, x, ids), H, f"gate_up {name} T={tokens}", True)
        check_swiglu(lib, fused, gate_up, sg, x, ids, f"swiglu {name} T={tokens}")
        y = down(h, sd.parts, ids)
        assert torch.equal(y, down(h, sd.parts, ids)), f"down {name} T={tokens}: not run-to-run identical"
        check(y, emulated(down, sd, h, ids), INTER, f"down {name} T={tokens}")


@pytest.mark.parametrize("name", sorted(TYPES))
def test_linear_experts_every_type_at_prefill_chunks(ctx, sm, name):
    """mmq's prefill kernels on every type: the tile width J follows the chunk's tokens (128 at a 512-token chunk,
    64 at the sweep's 64), and each J is its own kernel. A small stack (8 experts, 256 rows) keeps it to ~1M
    outputs; 513 tokens adds the one-token mmvq tail chunk. D0's passes (docs/v1-moe-prefill.md): `pass_rows` is
    PASS_ROWS on a pair of `PASS_PROVEN` (this SM's), else CHUNK, and where it is larger a call of up to 4,608 rows
    gives each row its 512-row chunk's bits (holon_testing's check; Qwen3-30B-A3B's 128 experts: the NaN test)."""
    n, out = 8, 256
    kernel, s = experts(ctx, name, H, out, 1, n=n), stack(name, n, out, H, seed=5)
    assert kernel.pass_rows == (PASS_ROWS if (sm, name, "q8_1") in PASS_PROVEN else CHUNK)
    for tokens in (CHUNK, CHUNK + 1):
        ids, x = uniform(tokens, n, seed=tokens), gate_up_input(tokens, seed=tokens)
        check(kernel(x, s.parts, ids), emulated(kernel, s, x, ids), H, f"{name} T={tokens}", True)
    if kernel.pass_rows > CHUNK:
        ids, x = uniform(9 * CHUNK, n, seed=9), gate_up_input(9 * CHUNK, seed=9)
        check_pass_rows(kernel, s.parts, x, ids, name)


@pytest.mark.parametrize("name", ["q4_k", "iq3_xxs", "q6_k", "q4_0", "q2_k"])
def test_merged_stack_equals_separate_stacks(ctx, lib, name):
    """Gate and up merged per expert give each stack's own output bitwise on mmvq and on mmq without the fixup (with
    it, the stream-k split moves with the number of row tiles). E = the SM count: every tile count is a multiple of
    it, so llama.cpp tiles one block per tile and never runs the fixup (`mmq.cuh`, ≥ 90% efficiency), on any GPU."""
    n = torch.cuda.get_device_properties(0).multi_processor_count
    sep, merged = experts(ctx, name, H, INTER, 1, n=n), experts(ctx, name, H, 2 * INTER, 1, n=n)
    fused = experts(ctx, name, H, 2 * INTER, 1, "swiglu", n=n)
    gate, up = random_blocks(name, n * INTER, H, seed=3), random_blocks(name, n * INTER, H, seed=4)
    both = np.concatenate([gate.reshape(n, INTER, -1), up.reshape(n, INTER, -1)], axis=1).reshape(n * 2 * INTER, -1)
    sg, su, sm_ = ({"blocks": torch.from_numpy(b).cuda().view(n, -1, b.shape[-1])}       # compared bitwise only
                   for b in (gate, up, both))
    t = merged.mmvq_tokens
    for tokens in (1, t, t + 1, 64, CHUNK):
        ids, x = uniform(tokens, n, seed=tokens), gate_up_input(tokens, seed=tokens)
        calls = launched(lambda x=x, ids=ids: [sep(x, sg, ids), sep(x, su, ids), merged(x, sm_, ids)])
        assert not any("stream_k_fixup" in c for c in calls), f"{name} T={tokens}: mmq took the fixup at E = {n}"
        y = merged(x, sm_, ids)
        assert torch.equal(y[..., :INTER], sep(x, sg, ids)), f"{name} T={tokens}: gate rows differ"
        assert torch.equal(y[..., INTER:], sep(x, su, ids)), f"{name} T={tokens}: up rows differ"
        if tokens <= t:
            ours = torch.empty(tokens, K, INTER, device="cuda")
            lib.swiglu(y, ours)
            r = l0.check(fused(x, sm_, ids), ours, torch.float32, 1)
            assert r.passed, f"{name} T={tokens}: the fused gate vs ours: {r}"


@pytest.fixture(scope="module")
def qwen_stacks():
    """Q4_K and IQ3_XXS stacks at Qwen3-30B-A3B's E = 128 (gate/up merged, down)."""
    return {name: (stack(name, EQ, 2 * INTER, H, seed=5), stack(name, EQ, H, INTER, seed=6))
            for name in ("q4_k", "iq3_xxs")}


@pytest.mark.parametrize("name", ["q2_0", "iq4_nl"])
def test_linear_experts_at_in_640(ctx, lib, name):
    """T6 (docs/v1-moe-prefill.md §10): FN's routed down projection (`in` 640, a whole number of the type's blocks
    but not of 256) on mmvq and mmq with ids, at a chunk and at a pass of two; mmq's last K step stays inside each
    row (the bytes after the stack are poison). Both types are `PASS_PROVEN`: a pass gives each row its chunk's bits
    at 640 too (a small stack, as in the every-type test). mmq takes its `tail` instances there only: at 768 (whole
    K steps) upstream's."""
    down = experts(ctx, name, 640, H, K)
    sd, t = stack(name, E, H, 640, seed=13), down.mmvq_tokens
    for tokens in (1, t, t + 1, 64, 2 * CHUNK + 1):
        ids = uniform(tokens, E, seed=tokens)
        h = torch.randn(tokens, K, 640, generator=torch.Generator().manual_seed(tokens)).cuda()
        y = down(h, sd.parts, ids)
        check(y, emulated(down, sd, h, ids), 640, f"down {name} at 640, T={tokens}")
    ids = uniform(64, E, seed=64)
    for k, tail in ((640, "true"), (INTER, "false")):
        kern, s = (down, sd) if k == 640 else (experts(ctx, name, k, H, K), stack(name, E, H, k, seed=15))
        h = torch.randn(64, K, k, generator=torch.Generator().manual_seed(k)).cuda()
        names = [n for n in launched(lambda kern=kern, h=h, s=s: kern(h, s.parts, ids))
                 if n.startswith("void mul_mat_q<")]
        assert names and {re.match(r"void mul_mat_q<([^>]*)>", n)[1].split(", ")[-1] for n in names} == {tail}, names
    small = experts(ctx, name, 640, 256, 1, n=8)
    if small.pass_rows > CHUNK:
        h = torch.randn(9 * CHUNK, 1, 640, generator=torch.Generator().manual_seed(9)).cuda()
        check_pass_rows(small, stack(name, 8, 256, 640, seed=14).parts, h, uniform(9 * CHUNK, 8, seed=9), name)


@pytest.mark.slow
@pytest.mark.parametrize("name", ["q4_k", "iq3_xxs"])
def test_routings_and_sizes(ctx, lib, qwen_stacks, name):
    sg, sd = qwen_stacks[name]
    gate_up, fused = experts(ctx, name, H, 2 * INTER, 1, n=EQ), experts(ctx, name, H, 2 * INTER, 1, "swiglu", n=EQ)
    down = experts(ctx, name, INTER, H, K, n=EQ)
    assert fused.glu_store, f"{name}: D4's store not taken at out {2 * INTER}"
    t = gate_up.mmvq_tokens
    rng = torch.Generator().manual_seed(7)
    same = torch.randperm(EQ, generator=rng)[:K]
    routings = {"uniform": lambda n: uniform(n, EQ, seed=n),
                "same 8 experts": lambda n: same[torch.rand(n, K, generator=rng).argsort(-1)].int().cuda(),
                "half empty": lambda n: (2 * uniform(n, EQ // 2, seed=n)).int(),
                "ids 0 and E-1": lambda n: torch.cat([torch.tensor([[0, EQ - 1]] * n, dtype=torch.int32).cuda(),
                                                      1 + uniform(n, EQ - 2, seed=n)[:, :K - 2]], 1)}
    cases = [("uniform", n) for n in sorted({*SIZES, t, t + 1})] + \
            [("same 8 experts", n) for n in (CHUNK, CHUNK + 1)] + [(r, 64) for r in ("half empty", "ids 0 and E-1")]
    for routing, tokens in cases:
        what = f"{name} {routing} T={tokens}"
        ids, x, h = routings[routing](tokens), gate_up_input(tokens, seed=tokens), down_input(tokens, seed=tokens)
        assert all(len(set(row)) == K for row in ids.tolist())
        y = gate_up(x, sg.parts, ids)
        assert y.shape == (tokens, K, 2 * INTER)
        check(y, emulated(gate_up, sg, x, ids), H, f"gate_up {what}", True)
        check_swiglu(lib, fused, gate_up, sg, x, ids, f"swiglu {what}")
        check(down(h, sd.parts, ids), emulated(down, sd, h, ids), INTER, f"down {what}")
    # across the mmvq limit the two paths agree on the tokens they share
    ids, x, h = uniform(t + 1, EQ, seed=8), gate_up_input(t + 1, seed=8), down_input(t + 1, seed=8)
    for kern, s, inp, k in ((gate_up, sg, x, H), (down, sd, h, INTER)):
        r = l0.check(kern(inp[:t], s.parts, ids[:t]), kern(inp, s.parts, ids)[:t], torch.bfloat16, k)
        assert r.passed, f"{name} in={k}: mmvq at {t} tokens vs mmq at {t + 1}: {r}"


@pytest.mark.parametrize("name", ["q4_k", "q6_k"])
def test_every_tile_of_every_entry_is_written(ctx, lib, name):
    """D2 (docs/v1-moe-prefill.md): mmq with ids launches only column tiles that hold rows, each block finding its
    entry by a binary search over the entries' bounds, at a tile width chosen against the mean rows per entry. Into an
    output pre-filled with NaN every row of every entry is written, through each id helper (calls of 9 to 4,096 rows
    at E = 128), with routings that give one entry many tiles (every token on 8 experts, a skewed popularity) or many
    entries one partial tile (uniform); each row's bits are its row's in the 4,096-row call (the tile width does not
    move them), that call's rows are its 512-row chunks' calls' (D0's `pass_rows` at E = 128), calls of 512, 2,048
    and 4,096 rows take the per-slot, compact and device-wide helpers, and no stream-k fixup is launched."""
    device, errors, slots = torch.cuda.current_device(), torch.zeros(1, dtype=torch.int32, device="cuda"), EQ + K
    rng = torch.Generator().manual_seed(17)
    hot = 1 / torch.arange(1, EQ + 1, dtype=torch.float64)[torch.randperm(EQ, generator=rng)]
    routings = {"uniform": uniform(PASS_ROWS, EQ, seed=17),
                "same 8 experts": torch.randperm(EQ, generator=rng)[:K][torch.rand(PASS_ROWS, K, generator=rng)
                                                                         .argsort(-1)].int().cuda(),
                "skewed": torch.multinomial(hot.expand(PASS_ROWS, EQ), K, generator=rng).int().cuda()}
    for stage, (k, out, _) in STAGES.items():
        s = stack(name, EQ, out, k, seed=18)
        w = s.w.view(-1, s.w.shape[-1])
        x = (gate_up_input if stage == "gate_up" else down_input)(PASS_ROWS, seed=19)

        def call(n, ids, a=0, x=x, w=w, k=k, out=out):
            y = torch.full((n, K, out), float("nan"), device="cuda")
            ws = torch.empty(lib.workspace_bytes_id(device, k, n, K, slots), dtype=torch.uint8, device="cuda")
            lib.mmq_id(TYPES[name], PREC["q8_1"], w, x[a:a + n], ids[a:a + n], y, ws, slots, errors, J_AGAINST_T)
            return y
        for routing, ids in routings.items():
            what = f"{name} {stage} {routing}"
            assert all(len(set(row)) == K for row in ids[:64].tolist())
            ref = call(PASS_ROWS, ids)
            assert not ref.isnan().any(), what
            for n in (9, 64, CHUNK, 4 * CHUNK):
                y = call(n, ids)
                assert not y.isnan().any() and torch.equal(y, ref[:n]), f"{what} T={n}"
            chunks = torch.cat([call(CHUNK, ids, a) for a in range(0, PASS_ROWS, CHUNK)])
            assert torch.equal(ref, chunks), f"{what}: the 4,096-row call against its chunks' calls"
            names = set().union(*(launched(lambda n=n, ids=ids: call(n, ids)) for n in (CHUNK, 4 * CHUNK, PASS_ROWS)))
            assert not any("stream_k_fixup" in c for c in names), what
            helpers = ("mm_ids_helper_per_slot", "mm_ids_helper_compact", "mm_ids_device")
            assert all(any(h in c for c in names) for h in helpers), f"{what}: {sorted(names)}"
    assert int(errors) == 0


@pytest.mark.slow
def test_mixed_types_in_one_layer(ctx, sm):
    """A std.moe layer with IQ3_S gate/up (mmq above 6 tokens on sm_86) and IQ4_XS down (mmvq up to 8) at 7 tokens:
    each stack takes its own path, as the per-instance counters show, and matches the emulator."""
    if sm != 86:
        pytest.skip("the IQ3_S / IQ4_XS limits that split 7 tokens across the paths are sm_86's")
    unit, (sg, su, sd) = moe_layer(ctx, "iq3_s", "iq4_xs")
    x = layer_input(7, seed=9)
    weights, ids, _ = unit._route(x)
    h = unit._gate_up(x[:, None], ids)
    y = unit.down(h, ids)
    assert unit._gate_up._kernel.calls == {"mmq/q8_1": 1} and unit.down._kernel.calls == {"mmvq/q8_1": 1}
    merged = stack("iq3_s", EQ, 2 * INTER, H, 0, unit._gate_up.weight.flatten(0, 1).cpu().numpy())
    none = experts(ctx, "iq3_s", H, 2 * INTER, 1, n=EQ)
    check(none(x[:, None], merged.parts, ids), emulated(none, merged, x[:, None], ids), H, "gate_up iq3_s T=7")
    check_swiglu(extension.load(sm), unit._gate_up._kernel, none, merged, x[:, None], ids, "swiglu iq3_s T=7")
    check(y, emulated(unit.down._kernel, sd, h, ids), INTER, "down iq4_xs T=7")
    out = torch.empty_like(x)
    unit._combine(y, weights, out)
    assert torch.equal(unit(x), out)


def moe_layer(ctx, gate_up: str, down: str):
    """A std.moe unit (E = 128, k = 8) with random gate/up stacks of `gate_up`'s type and a down stack of `down`'s,
    bound like a loaded model's; also returns the stacks."""
    with torch.device("meta"):
        unit = ctx.unit(Ref("std.moe@1", experts=EQ, top_k=K, inter=INTER, renormalize=True), hidden=H, layer=0)
    router = torch.randn(EQ, H, generator=torch.Generator().manual_seed(10)).cuda() / H**0.5
    torch.utils.swap_tensors(unit.router, torch.nn.Parameter(router, requires_grad=False))
    stacks = []
    for part, name, out, k, seed in (("gate", gate_up, INTER, H, 11), ("up", gate_up, INTER, H, 12),
                                     ("down", down, H, INTER, 13)):
        s = stack(name, EQ, out, k, seed)
        q = QTensor(Layout(Scheme(f"ggml.{name}"), "gguf.native"), (EQ, out, k), {"blocks": torch.from_numpy(s.blocks)})
        assert getattr(unit, part).adopt_weights(ctx, "weight", q)
        stacks.append(s)
    unit.finalize_weights(ctx)
    for module in unit.modules():
        if hasattr(module, "bind_kernels"):
            module.bind_kernels(ctx)
    return unit, stacks


def test_ids_from_moe_topk(ctx, qwen_stacks):
    sg, sd = qwen_stacks["q4_k"]
    route, down = topk(ctx, EQ), experts(ctx, "q4_k", INTER, H, K, n=EQ)
    for tokens, dtype in ((1, "bfloat16"), (5, "float32"), (77, "float32")):   # act_dtype f32 too
        gate_up = experts(ctx, "q4_k", H, 2 * INTER, 1, n=EQ, act_dtype=dtype)
        _, ids = route(torch.randn(tokens, EQ, generator=torch.Generator().manual_seed(tokens)).cuda())
        x, h = gate_up_input(tokens, seed=tokens).to(getattr(torch, dtype)), down_input(tokens, seed=tokens)
        check(gate_up(x, sg.parts, ids), emulated(gate_up, sg, x, ids), H, f"gate_up topk T={tokens}", True)
        check(down(h, sd.parts, ids), emulated(down, sd, h, ids), INTER, f"down topk T={tokens}")


@pytest.mark.parametrize("name", ["q4_k", "iq3_xxs", "nvfp4/hf"])
def test_graph_replay_follows_the_routing(ctx, qwen_stacks, name):
    """Captured at every decode bucket (mmvq up to the limit, mmq above), replayed with other ids and inputs: the
    result is the eager one on the new routing, bitwise (NVFP4: its per-expert scale2 gathered on the device) — and
    so with passed buffers (docs/v1-moe-prefill.md D10), static, NaN-poisoned: gate/up's and down's `out`, and one
    `scratch` for both calls."""
    if name == "nvfp4/hf":
        gate_up, down, pg, pd, _ = nvfp4_stacks(ctx, EQ)
    else:
        sg, sd = qwen_stacks[name]
        pg, pd = sg.parts, sd.parts
        gate_up, down = experts(ctx, name, H, 2 * INTER, 1, "swiglu", n=EQ), experts(ctx, name, INTER, H, K, n=EQ)
    for tokens in BUCKETS:
        x, ids = layer_input(tokens, seed=tokens)[:, None], uniform(tokens, EQ, seed=tokens)

        def layer(x=x, ids=ids):
            return down(gate_up(x, pg, ids), pd, ids)

        eager = layer()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            layer()                                         # warm-up on the capture stream
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            y = layer()
        graph.replay()
        assert torch.equal(y, eager), f"{name} T={tokens}: replay differs from eager"
        x.copy_(layer_input(tokens, seed=tokens + 1)[:, None])
        ids.copy_(uniform(tokens, EQ, seed=tokens + 1))
        graph.replay()
        assert torch.equal(y, layer()), f"{name} T={tokens}: replay on new routing differs from eager"
        h, out = torch.full((tokens, K, INTER), torch.nan, device="cuda"), torch.full_like(y, torch.nan)
        scratch = torch.full((max(gate_up.scratch_bytes(tokens), down.scratch_bytes(tokens)),), 0xFF,
                             dtype=torch.uint8, device="cuda")
        assert (gate_up.scratch_bytes(tokens) > 0) == (tokens > gate_up.mmvq_tokens)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            down(gate_up(x, pg, ids, out=h, scratch=scratch), pd, ids, out=out, scratch=scratch)
        x.copy_(layer_input(tokens, seed=tokens + 2)[:, None])
        ids.copy_(uniform(tokens, EQ, seed=tokens + 2))
        graph.replay()
        assert torch.equal(out, layer()), f"{name} T={tokens}: replay with passed buffers differs from eager"


# per SM, the types D0 proves there (`PASS_PROVEN`: the lock classes' expert types and FN's MoE quants)
SM = 10 * torch.cuda.get_device_capability()[0] + torch.cuda.get_device_capability()[1] if torch.cuda.is_available() \
    else 0
PLACED = sorted({t for s, t, _ in PASS_PROVEN if s == SM})
EP = 12                                                     # experts placed in an arena, k = 8 of them per token
SLOTS = (EP, EP + 1, 16, 74)                                # docs/v1-offload.md D4's slot counts
STAGES = {"gate_up": (H, 2 * INTER, 1), "down": (INTER, H, K)}          # in, out, input rows per token


@dataclasses.dataclass
class Arena:
    """The EP experts of a stack placed among the slots of a class arena (D5), every other slot poisoned (0xFF bytes:
    NaN scales) but the top K when there is room for them: the per-rank null slots, zero weights. Called with ids
    < EP (an expert) or EP + j (rank j's null slot). The kernel is bound to the layer's EP experts, so mmq's compact
    launch covers min(slots, T·k, EP + K) entries."""

    kernel: Sized
    parts: dict[str, torch.Tensor]
    table: torch.Tensor                                     # id -> slot

    def __call__(self, x: torch.Tensor, ids: torch.Tensor) -> tuple[torch.Tensor, str]:
        """The output and the path the kernel took."""
        calls = dict(self.kernel.calls)
        y = self.kernel(x, self.parts, self.table[ids.long()])
        return y, next(p for p, n in self.kernel.calls.items() if n != calls.get(p, 0))


def arena(ctx, s: Stack, stage: str, slots: int, placement: str) -> Arena:
    k, out, rows = STAGES[stage]
    null = slots >= EP + K
    free = slots - K if null else slots
    place = {"in order": torch.arange(EP), "reversed": torch.arange(EP).flip(0),
             "random": torch.randperm(free, generator=torch.Generator().manual_seed(slots))[:EP]}[placement]
    blocks = torch.full((slots, out, s.w.shape[-1]), 0xFF, dtype=torch.uint8, device="cuda")
    blocks[place.cuda()] = s.parts["blocks"]
    if null:
        blocks[free:] = 0
    table = torch.cat([place, torch.arange(free, slots)] if null else [place]).int().cuda()
    return Arena(experts(ctx, s.name, k, out, rows, n=EP), {"blocks": blocks}, table)


def placed_inputs(stage: str, tokens: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    return (gate_up_input if stage == "gate_up" else down_input)(tokens, seed), uniform(tokens, EP, seed)


# mmq's two id helpers (mmid.cuh), each with an arena size that takes it for a binding of EP experts (top_k K: at most
# EP + K slots per call): per slot where the launch lists every slot (at most 64 more than EP + K, SLOTS included),
# compact where the arena has more
HELPERS = {"per_slot": EP + K, "compact": 128}
# mmq with ids picks J against a call's tokens up to this many, above against its entries' mean (the plugin's rule)
J_AGAINST_T = IDS_J_TOKENS.get(SM, IDS_J_TOKENS_DEFAULT)


def helper(fn) -> str:
    """The id helper `fn()` launches."""
    return next(h for h in HELPERS for name in launched(fn) if f"mm_ids_helper_{h}" in name)


@pytest.mark.parametrize("name", PLACED)
@pytest.mark.parametrize("stage", sorted(STAGES))
def test_expert_rows_do_not_depend_on_the_slots_or_the_tile_width(ctx, lib, name, stage):
    """D4's placement invariance on mmq with ids: an expert's rows are bitwise the same wherever an arena places it
    (each of SLOTS with a random placement, per-slot id helper, and 128 slots, compact helper; at EP also reversed:
    permutations) and at every mmq tile width (token counts from the mmvq limit + 1 to 320 on the 128-slot arena,
    whose launch lists at most EP + K entries: J from 16 up to 128, chosen against the tokens up to J_AGAINST_T and
    above against the entries' mean rows, docs/v1-moe-prefill.md D2), all against one launch over EP slots in order at
    512 tokens. llama.cpp switched mmq with ids to stream-k below 90 % tile efficiency, splitting K at points that
    move with the slot count: on every GPU at least one of SLOTS did (the dense schedule over the same grid says so),
    and its rows differed."""
    k, out, _ = STAGES[stage]
    s = stack(name, EP, out, k, seed=21)
    x, ids = placed_inputs(stage, CHUNK, seed=22)
    ref, path = arena(ctx, s, stage, EP, "in order")(x, ids)
    assert path == "mmq/q8_1"
    device, typ = torch.cuda.current_device(), TYPES[name]
    assert any(lib.mmq_schedule(device, typ, PREC["q8_1"], out * n, 64)[0] == 2 for n in SLOTS)
    launches = [(n, "random", t) for n in (*SLOTS, HELPERS["compact"]) for t in (64, CHUNK)] + [(EP, "reversed", 64)]
    a = arena(ctx, s, stage, HELPERS["compact"], "random")
    tiles = sorted({a.kernel.mmvq_tokens + 1, 16, 24, 32, 48, 64, 96, 112, 128, 160, 180, 200, 256, 320})
    rows = [t if t <= J_AGAINST_T else -(-t * K // a.kernel.slots) for t in tiles]     # what J is chosen against
    widths = {lib.mmq_schedule(device, typ, PREC["q8_1"], out, r)[2] for r in rows}
    assert min(widths) <= 16 and max(widths) == 128 and len(widths) >= 9
    for n, placement, tokens in launches + [(HELPERS["compact"], "random", t) for t in tiles]:
        same = (n, placement) == (HELPERS["compact"], "random")
        y, path = (a if same else arena(ctx, s, stage, n, placement))(x[:tokens], ids[:tokens])
        assert path == "mmq/q8_1" and torch.equal(y, ref[:tokens]), f"{name} {stage}: {n} slots {placement}, T={tokens}"


@pytest.mark.parametrize("name", ["q4_k", "q6_k"])
@pytest.mark.parametrize("stage", sorted(STAGES))
def test_an_arena_call_of_many_chunks_gives_the_chunks_rows(ctx, name, stage):
    """D8's expert passes: the arena binding (`linear_experts_arena@1`) takes 4,096 rows per mmq call at top-8 (32,768
    pairs: the id helper's device-wide path, past one block's 16,384), in whole 512-row chunks, and each row is bitwise
    what a 512-row call — the reference chunk's — gives it, with every expert of the layer in its arena; with passed
    buffers too (`passed_buffers`). On that path the helper flags as the others do: ids out of range, more slots than
    the bound, a slot past mmq's column tiles."""
    k, out, rows = STAGES[stage]
    s = stack(name, EP, out, k, seed=31)
    facts = {"layout": f"ggml.{name}/gguf.native", "experts": EP, "in": k, "out": out, "top_k": K,
             "rows_per_token": rows, "glu": "none", "act_dtype": "bfloat16", "out_dtype": "float32", "act_quant": "a8",
             "slots": EP + K}
    kernel = ctx.select_kernel("linear_experts_arena@1", site="l0", **facts).impl(ctx, **facts)
    assert kernel.pass_rows == 4096
    parts = {"blocks": torch.cat([s.parts["blocks"], torch.zeros_like(s.parts["blocks"][:K])])}
    x, ids = placed_inputs(stage, 4608, seed=32)
    y = kernel(x, parts, ids)
    assert torch.equal(y, torch.cat([kernel(x[a:a + CHUNK], parts, ids[a:a + CHUNK]) for a in range(0, 4608, CHUNK)]))
    assert any("mm_ids_device" in n for n in launched(lambda: kernel(x[:4096], parts, ids[:4096])))
    passed_buffers(ctx, facts, parts, x, ids)
    if name != "q4_k" or stage != "gate_up":
        return
    x, ids, errors = x[:4096], ids[:4096], kernel.errors

    def run(i):
        errors.zero_()
        return kernel(x, parts, i), int(errors)
    bad, keep = ids.clone(), torch.ones_like(ids, dtype=torch.bool)
    bad[3, 2], bad[4000, 7] = EP + K, -1
    keep[3, 2] = keep[4000, 7] = False
    got, err = run(bad)
    assert err == ID_ERRORS["range"] and torch.equal(got[keep], y[:4096][keep]), "ids out of range"
    over = ids.clone()
    over[:, :] = ids[0, 0]                                          # 32,768 pairs on one slot: past its column tiles
    assert run(over)[1] == ID_ERRORS["columns"]
    two = facts | {"experts": 2}                                    # bound to 2 + K = 10 of the 12 slots named
    narrow = ctx.select_kernel("linear_experts_arena@1", site="l0", **two).impl(ctx, **two)
    errors.zero_()
    narrow(x, parts, ids)
    assert int(errors) == ID_ERRORS["slots"], "more slots than the bound"



@pytest.mark.parametrize(("stage", "glu"), [("gate_up", "none"), ("gate_up", "swiglu"), ("down", "none")])
def test_mmvq_skips_an_arenas_null_slots(ctx, stage, glu):
    """An arena binding's mmvq calls (one token, and the multi-token kernel at the type's limit) store 0 for an entry
    on a null slot (below top_k, std.moe.kernels) without reading the slot: NaN-poisoned null slots give the bits of
    zeroed ones, and the other entries' rows are unchanged."""
    k, out, rows = STAGES[stage]
    s = stack("q4_k", EP, out, k, seed=41)
    facts = {"layout": "ggml.q4_k/gguf.native", "experts": EP, "in": k, "out": out, "top_k": K,
             "rows_per_token": rows, "glu": glu, "act_dtype": "bfloat16", "out_dtype": "float32", "act_quant": "a8",
             "slots": EP + K}
    kernel = ctx.select_kernel("linear_experts_arena@1", site="l0", **facts).impl(ctx, **facts)
    zero = torch.cat([torch.zeros_like(s.parts["blocks"][:K]), s.parts["blocks"]])
    poisoned = zero.clone()
    poisoned[:K] = 0xFF
    x, ids = placed_inputs(stage, kernel.mmvq_tokens, seed=42)
    slots = ids.int() + K
    slots[:, ::2] = torch.arange(0, K, 2, dtype=torch.int32, device="cuda")      # even ranks on their null slots
    for tokens in sorted({1, kernel.mmvq_tokens}):
        want = kernel(x[:tokens], {"blocks": zero}, slots[:tokens])
        got = kernel(x[:tokens], {"blocks": poisoned}, slots[:tokens])
        assert torch.equal(got, want) and not got[:, ::2].any() and got[:, 1::2].any(), f"T={tokens}"
    assert {p for p, n in kernel.calls.items() if n} == {"mmvq/q8_1"}, kernel.calls


@pytest.mark.parametrize("glu", ["none", "swiglu"])
@pytest.mark.parametrize("stage", sorted(STAGES))
def test_entries_split_across_calls_compose_the_one_call(ctx, stage, glu):
    """A pipelined pass (FN step 7): an entry's id −1 is skipped on purpose — mmq's id helper lists it last without a
    range error, mmvq leaves its rows, our SwiGLU after an unfused mmq leaves them too —, so calls over a partition of a
    pass's entries, each with the others' ids −1, write into one `out` exactly the one call's rows (mmq at 300 tokens,
    a 1-token and an mmvq-limit tail)."""
    k, out, rows = STAGES[stage]
    s = stack("q4_k", EP, out, k, seed=51)
    facts = {"layout": "ggml.q4_k/gguf.native", "experts": EP, "in": k, "out": out, "top_k": K,
             "rows_per_token": rows, "glu": glu, "act_dtype": "bfloat16", "out_dtype": "float32", "act_quant": "a8",
             "slots": EP + K}
    kernel = ctx.select_kernel("linear_experts_arena@1", site="l0", **facts).impl(ctx, **facts)
    kernel.glu_store = False                                     # (swiglu: mmq's gate/up, then our SwiGLU)
    parts = {"blocks": torch.cat([s.parts["blocks"], torch.zeros_like(s.parts["blocks"][:K])])}
    for tokens in (300, 1, kernel.mmvq_tokens):
        x, ids = placed_inputs(stage, tokens, seed=52)
        want = kernel(x, parts, ids)
        part = torch.randint(-1, 3, ids.shape, generator=torch.Generator().manual_seed(tokens)).cuda()
        got = torch.full_like(want, torch.nan)
        kernel.errors.zero_()
        for i in range(-1, 3):
            kernel(x, parts, torch.where(part == i, ids, -1), out=got)
        assert torch.equal(got, want) and not int(kernel.errors), f"T={tokens}"


def test_scaled_slots_leave_minus_one_entries_untouched(ctx):
    """An expert scale (NVFP4's scale2) applied after an arena call's mmq multiplies only its own entries' rows: a −1
    entry's row, which another call of the pass wrote, is untouched (it had taken the last expert's scale)."""
    from holon_plugins.kernels.ggml_cuda.plugin import LinearExperts
    y = torch.randn(9, K, 64, device="cuda")
    ids = torch.randint(0, EP, (9, K), device="cuda", dtype=torch.int32)
    ids[::2, 0] = -1
    scale = torch.rand(EP, device="cuda") + 0.5
    got = y.clone()
    LinearExperts._scale_slots(got, scale, ids)
    want = torch.where((ids >= 0)[..., None], y * scale[ids.clamp_min(0)][..., None], y)
    assert torch.equal(got, want) and torch.equal(got[::2, 0], y[::2, 0])


def passed_buffers(ctx, facts: dict, parts: dict, x: torch.Tensor, ids: torch.Tensor) -> None:
    """D10 (docs/v1-moe-prefill.md) on an arena binding, glu none and swiglu (D4's store, and unfused): NaN-poisoned
    `out` and `scratch` of `scratch_bytes` (one for every launch of a call: 4,096 rows, then a chunk, or then an mmvq
    tail) give the allocating call's bits; `scratch` is gate/up's f32 output (none for D4's store), then the mmq
    launches' id maps and q8_1 rows (the persistent workspace holds the mmvq calls'); mmvq calls do none of D10's
    work; a short or misaligned buffer, or one
    overlapping another or `x`, is refused before anything is launched."""
    for glu, fused in (("none", True), ("swiglu", True), ("swiglu", False)):
        f = facts | {"glu": glu}
        kernel = ctx.select_kernel("linear_experts_arena@1", site="l0", **f).impl(ctx, **f)
        kernel.glu_store &= fused
        assert kernel.scratch_bytes(kernel.mmvq_tokens) == 0
        small, empty = kernel.mmvq_tokens, torch.empty(0, dtype=torch.uint8, device="cuda")
        want = kernel(x[:small], parts, ids[:small])
        with (mock.patch.object(kernel, "_gate_up_out", side_effect=AssertionError),        # mmvq calls do none of
              mock.patch.object(kernel, "scratch_bytes", side_effect=AssertionError)):     # D10's work (host-bound)
            assert torch.equal(kernel(x[:small], parts, ids[:small], scratch=empty), want)
        assert kernel.scratch_bytes(64) == (0 if glu == "none" or kernel.glu_store else 64 * K * kernel.out * 4) + \
            kernel._id_bytes(64)
        for tokens in (len(ids), len(ids) - CHUNK + 1):
            want = kernel(x[:tokens], parts, ids[:tokens])
            out, scratch = torch.full_like(want, torch.nan), torch.full((kernel.scratch_bytes(tokens),), 0xFF,
                                                                        dtype=torch.uint8, device="cuda")
            got = kernel(x[:tokens], parts, ids[:tokens], out=out, scratch=scratch)
            assert got is out and torch.equal(out, want), f"{glu}, T={tokens}: passed buffers"
    shape, nbytes = (len(ids), K, kernel.out_y), kernel.scratch_bytes(len(ids))
    room = torch.empty(nbytes + 512, dtype=torch.uint8, device="cuda")
    at = -room.data_ptr() % 256
    scratch, size = room[at:at + nbytes], math.prod(shape) * 4
    flat = torch.empty(x.nbytes + size, dtype=torch.uint8, device="cuda")          # out's first bytes are x's last
    for kw, why in (({"scratch": scratch[:-1]}, "scratch"), ({"scratch": room[at + 1:at + 1 + nbytes]}, "scratch"),
                    ({"out": scratch[:size].view(torch.float32).view(shape), "scratch": scratch}, "overlaps"),
                    ({"out": torch.empty(*shape[:2], shape[2] + 1, device="cuda")}, "contiguous f32"),
                    ({"out": flat[x.nbytes - 256:][:size].view(torch.float32).view(shape),
                      "x": flat[:x.nbytes].view(x.dtype).view(x.shape)}, "overlaps"),
                    ({"scratch": scratch, "x": scratch[256:256 + x.nbytes].view(x.dtype).view(x.shape)}, "overlaps")):
        xs = kw.pop("x", x)

        def call(kw=kw, xs=xs, why=why):
            with pytest.raises(ValueError, match=why):
                kernel(xs, parts, ids, **kw)
        assert not launched(call), why


@pytest.mark.parametrize("name", PLACED)
@pytest.mark.parametrize("stage", sorted(STAGES))
@pytest.mark.parametrize("path", sorted(HELPERS))
def test_compacted_launches_equal_the_chunk_s(ctx, name, stage, path):
    """D4's compaction: a launch over a subset of a chunk's rows, in any order, padded with null-slot rows (zero
    activations, rank j's ids on its own null slot, repeated across the padding rows) up to the type's mmvq limit + 1
    so the kernel takes mmq, gives those rows bitwise as the chunk's launch does, and zero padding rows; a subset
    above the limit needs no padding. On both id helpers. Chunks of 64 rows, and of 8, between the limits on Turing
    and Ampere (Q4_K, Q6_K, IQ2_XXS 8: the chunk takes mmvq and is never compacted; IQ2_S, IQ3_XXS 7: mmq,
    compacted); on Ada and Blackwell every type's expert mmvq limit is 8, so the 8-row chunk takes mmvq throughout."""
    k, out, _ = STAGES[stage]
    a = arena(ctx, stack(name, EP, out, k, seed=21), stage, HELPERS[path], "random")
    t, gen = a.kernel.mmvq_tokens, torch.Generator().manual_seed(24)
    for chunk in (8, 64):
        x, ids = placed_inputs(stage, chunk, seed=chunk)
        y, path_ = a(x, ids)
        if chunk == 64:
            assert helper(lambda x=x, ids=ids: a(x, ids)) == path
        if chunk <= t:
            assert path_ == "mmvq/q8_1"
            continue
        for r in sorted({1, t, min(2 * t, chunk - 1)}):
            rows, pad = torch.randperm(chunk, generator=gen)[:r].cuda(), max(0, t + 1 - r)
            xg = torch.cat([x[rows], x.new_zeros(pad, *x.shape[1:])])
            ids_g = torch.cat([ids[rows], (EP + torch.arange(K, dtype=torch.int32, device="cuda")).expand(pad, K)])
            yg, path_ = a(xg, ids_g)
            what = f"{name} {stage}: {r} of {chunk} rows + {pad} padding"
            assert path_ == "mmq/q8_1" and torch.equal(yg[:r], y[rows]) and not yg[r:].any(), what


@pytest.mark.parametrize("path", sorted(HELPERS))
def test_violated_preconditions_are_flagged(ctx, path):
    """mmq's id helpers report a violated precondition in the kernel's `errors` word and leave the rows it names
    uncomputed, every other row as without it: an id out of range (past the stack, or negative); more distinct slots
    than the binding's bound (compact helper only: the per-slot one lists every slot); a slot with more pairs than
    mmq's column tiles (three rows with all k ids on one expert at 16 tokens, one tile of 16). An id repeated inside a
    row within that bound is computed: both slots hold the expert's product, unflagged."""
    a = arena(ctx, stack("q4_k", EP, 2 * INTER, H, seed=21), "gate_up", HELPERS[path], "random")
    tokens, errors = 16, a.kernel.errors
    x, ids = placed_inputs("gate_up", tokens, seed=25)
    slots = a.table[ids.long()]

    def run(kernel, s):
        errors.zero_()
        return kernel(x, a.parts, s), int(errors)

    ref, err = run(a.kernel, slots)
    assert err == 0 and helper(lambda: run(a.kernel, slots)) == path
    uses = {s: int((slots == s).sum()) for s in slots.unique().tolist()}
    j0 = min(range(K), key=lambda j: uses[int(slots[0, j])])           # token 0's least used slot, repeated
    assert uses[int(slots[0, j0])] < tokens
    dup, j1 = slots.clone(), (j0 + 1) % K
    dup[0, j1] = slots[0, j0]
    y, err = run(a.kernel, dup)
    keep = torch.ones(tokens, K, dtype=torch.bool, device="cuda")
    keep[0, j1] = False
    assert err == 0 and torch.equal(y[0, j1], y[0, j0]) and torch.equal(y[keep], ref[keep]), "repeat within the bound"
    over = slots.clone()
    over[-3:] = slots[0, 0]
    y, err = run(a.kernel, over)
    assert err == ID_ERRORS["columns"] and torch.equal(y[:-3], ref[:-3]), "repeat beyond the column tiles"
    bad, keep = slots.clone(), torch.ones(tokens, K, dtype=torch.bool, device="cuda")
    bad[3, 2], bad[5, 7] = HELPERS[path], -1
    keep[3, 2] = keep[5, 7] = False
    y, err = run(a.kernel, bad)
    assert err == ID_ERRORS["range"] and torch.equal(y[keep], ref[keep]), "ids out of range"
    if path == "compact":
        narrow = experts(ctx, "q4_k", H, 2 * INTER, 1, n=2)               # bound to 2 + K = 10 of the 12 slots
        y, err = run(narrow, slots)
        listed = torch.isin(slots, slots.unique()[:narrow.slots])
        assert err == ID_ERRORS["slots"] and torch.equal(y[listed], ref[listed]), "more slots than the bound"


def topk(ctx, n: int, k: int = K, renormalize: bool = True):
    facts = {"experts": n, "top_k": k, "renormalize": renormalize}
    spec = ctx.select_kernel("moe_topk@1", site="l0", **facts)
    assert spec.provider == "kernels.ggml_cuda" and {"graph_safe", "deterministic"} <= spec.props
    return spec.impl(ctx, **facts)


def check_topk(logits: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor, k: int, renormalize: bool) -> None:
    """The reference's picks up to near-ties, its weights within 1e-5, ids unique and in range."""
    want_w, want_ids = reference.moe_topk(logits, k, renormalize)
    assert ids.dtype == torch.int32 and ids.is_contiguous() and weights.is_contiguous()
    assert all(len(set(row)) == k and 0 <= min(row) and max(row) < logits.shape[1] for row in ids.tolist())
    x = logits.double()
    p = torch.where(x.isnan(), torch.finfo(torch.float32).min, x).softmax(-1)
    got, want = p.gather(1, ids.long()), p.gather(1, want_ids.long())
    tied = (got - want).abs() <= TIE * want
    assert tied.all(), f"picks differ beyond near-ties at {(~tied).nonzero()[:4].tolist()}"
    torch.testing.assert_close(weights, want_w, rtol=TIE, atol=1e-7)


@pytest.mark.parametrize("n", [16, 64, 128, 512])
def test_moe_topk(ctx, n):
    for k, renormalize in ((K, True), (K, False), (2, True)):
        route = topk(ctx, n, k, renormalize)
        for tokens in (1, 13, 256):
            gen = torch.Generator().manual_seed(tokens)
            logits = (2 * torch.randn(tokens, n, generator=gen)).cuda()
            check_topk(logits, *route(logits), k, renormalize)
            near = torch.randn(tokens, n, generator=gen).round(decimals=1)
            near += 1e-6 * torch.randn(tokens, n, generator=gen)
            check_topk(near.cuda(), *route(near.cuda()), k, renormalize)
        w, ids = route(torch.empty(0, n, device="cuda"))
        assert w.shape == ids.shape == (0, k)


@pytest.mark.parametrize("n", [16, 128, 512])
def test_moe_topk_non_finite_rows(ctx, n):
    """Rows with NaN or ±inf logits still give k unique ids in range, as the contract requires; their weights are
    unspecified. llama.cpp replaces NaN by −FLT_MAX after the softmax, so a NaN or +inf logit makes every probability
    NaN: the row routes to experts 0..k−1, its renormalised weights −inf (−FLT_MAX over the clamped sum). Rows of
    finite values and −inf match the reference."""
    inf, nan = float("inf"), float("nan")
    rows = torch.randn(7, n)
    rows[0] = nan
    rows[1] = -inf
    rows[2] = inf
    rows[3, 5] = inf
    rows[4, ::3] = nan
    rows[5, ::2] = -inf
    rows[6, 1], rows[6, 2], rows[6, 3] = inf, -inf, nan
    weights, ids = topk(ctx, n)(rows.cuda())
    assert all(len(set(row)) == K and 0 <= min(row) and max(row) < n for row in ids.tolist())
    assert ids[4].tolist() == list(range(K)) and (weights[4] == -float("inf")).all()
    torch.testing.assert_close(ids[5], reference.moe_topk(rows[5:6], K, True)[1][0].cuda(), rtol=0, atol=0)


@pytest.mark.parametrize("n,hidden", [(16, 4096), (128, H), (512, 1536)])
def test_moe_router(ctx, lib, n, hidden):
    """`moe_router@1` (docs/v1-offload.md §14, step 3c), its fused path (up to `rows` tokens): its logits within the
    f32 budget of the f64 product from x's exact values, bf16 and f32 x; its weights and ids bitwise `moe_topk` on
    those logits, near-ties and NaN rows included (k unique ids in range); a row's logits the same at any token count
    and tile (row-invariant: 1- to 16-row calls against each other), exact ties included (two equal router rows; the
    q8_1 loop runs several rounds at 16 experts of 4,096); the tile counters left zeroed, so a second call
    (a graph replay) gives the same bits; x's q8_1 activations, written in the same launch, bitwise quantize_q8_1's.
    Through the plugin: the fused path's bits up to `rows` tokens, above them the composition's (the f32 linear, then
    `moe_topk`); for prompt chunks (CHUNK rows and more, experts a multiple of 64: docs/v1-moe-prefill.md D3) the
    router GEMM's logits, within the f32 budget and row-invariant across calls of 512 rows and more, then `moe_topk`'s
    selection on them, bitwise."""
    gen = torch.Generator().manual_seed(n)
    router = (torch.randn(n, hidden, generator=gen) / hidden ** 0.5).cuda()
    router[1] = router[0]                                               # tied logits: the lower id first
    x32 = torch.randn(700, hidden, generator=gen).cuda()
    x32[7, 5] = float("nan")
    rows = lib.moe_router_rows()
    done = torch.zeros(lib.moe_router_tiles(rows), dtype=torch.int32, device="cuda")
    for k, renormalize in ((min(K, n), True), (2, False)):
        route, choose = (select(ctx, "moe_router@1", hidden=hidden, experts=n, top_k=k, renormalize=renormalize,
                                dtype="bfloat16"), topk(ctx, n, k, renormalize))
        for x in (x32.bfloat16(), x32):
            def call(rows_: torch.Tensor, k=k, renormalize=renormalize):
                logits = torch.full((len(rows_), n), float("nan"), device="cuda")
                w, ids = torch.empty(len(rows_), k, device="cuda"), torch.empty(len(rows_), k, dtype=torch.int32,
                                                                                    device="cuda")
                xq, want = (torch.full((q8_1_bytes(len(rows_), hidden),), 7, dtype=torch.uint8, device="cuda")
                            for _ in range(2))
                lib.moe_router(rows_, router, logits, done, w, ids, renormalize, xq=xq)
                lib.quantize_q8_1(TYPES["q4_k"], rows_, want)
                assert torch.equal(xq, want), "x's q8_1 is not quantize_q8_1's"
                return logits, w, ids
            whole = [call(x[a:a + rows]) for a in range(0, len(x), rows)]
            logits, w, ids = (torch.cat(t) for t in zip(*whole, strict=True))
            assert not done.any(), "tile counters left set"
            finite = torch.ones(len(x), dtype=torch.bool)
            finite[7] = False
            r = l0.check(logits[finite], x[finite].double() @ router.double().T, torch.float32, hidden)
            assert r.passed, r
            assert logits[7].isnan().all() and all(len(set(row)) == k for row in ids.tolist())
            assert all(map(bitwise, choose(logits), (w, ids)))
            for a, b in ((0, 1), (5, 8), (13, 16), (690, 700)):           # row-invariant, and repeatable
                assert all(map(bitwise, call(x[a:b]), (logits[a:b], w[a:b], ids[a:b]))), f"rows {a}:{b}"
            if x.dtype == torch.bfloat16:
                assert all(map(bitwise, route(x[:rows], router), (w[:rows], ids[:rows])))
                if n % 64:
                    assert all(map(bitwise, route(x, router), choose(torch.nn.functional.linear(x.float(), router))))
                    continue
                def logits_of(rows_: torch.Tensor) -> torch.Tensor:
                    out = torch.full((len(rows_), n), float("nan"), device="cuda")
                    lib.moe_router_logits(rows_, router, torch.empty(lib.moe_router_partials(n), *out.shape,
                                                                     device="cuda"), out)
                    return out
                gemm = logits_of(x)
                r = l0.check(gemm[finite], x[finite].double() @ router.double().T, torch.float32, hidden)
                assert r.passed, f"router GEMM: {r}"
                for a, b in ((0, CHUNK), (len(x) - CHUNK, len(x)), (3, 70)):
                    assert bitwise(logits_of(x[a:b]), gemm[a:b]), f"router GEMM rows {a}:{b}"
                assert all(map(bitwise, route(x, router), choose(gemm)))
        assert route(x32[:0].bfloat16(), router)[1].shape == (0, k)
    rb = router.bfloat16()                 # a bf16 router (moe.router_dtype): its values as f32 or held bf16, same bits
    for x in (x32.bfloat16(), x32):
        held = []
        for r in (rb.float(), rb):
            logits = torch.empty(rows, n, device="cuda")
            w, ids = torch.empty(rows, K, device="cuda"), torch.empty(rows, K, dtype=torch.int32, device="cuda")
            lib.moe_router(x[:rows], r, logits, done, w[:, :min(K, n)], ids[:, :min(K, n)], True)
            gemm = None
            if n % 64 == 0:
                gemm = torch.empty(len(x), n, device="cuda")
                lib.moe_router_logits(x, r, torch.empty(lib.moe_router_partials(n), *gemm.shape, device="cuda"), gemm)
            held.append((logits, w[:, :min(K, n)], ids[:, :min(K, n)], gemm))
        assert all(bitwise(a, b) for a, b in zip(*held, strict=True) if a is not None), "a bf16 router's bits"


def test_the_router_prepares_the_gate_up_input(ctx, lib):
    """`route_prepare` (step 3c): at every token count on the gate/up's mmvq path the handle it returns gives the
    gate/up output bitwise that of the layer input itself (its q8_1 written by the router's launch, read in place of
    a quantize: other activations written into the handle's buffer are what it reads); a handle whose buffer a later
    write took falls back to its x; another provider's handle is read as its `.x`; above the path it returns the
    tensor."""
    unit, _ = moe_layer(ctx, "q4_k", "q4_k")
    gate_up = unit._gate_up.kernel
    assert unit._prepare is not None and 0 < gate_up.mmvq_tokens <= 8
    for t in range(1, gate_up.mmvq_tokens + 2):
        x, other = layer_input(t, seed=t), layer_input(t, seed=100 + t)
        weights, ids, xin = unit._prepare(x, unit.router, gate_up)
        assert all(map(bitwise, (weights, ids), unit._router(x, unit.router)))
        want = unit._gate_up(x[:, None], ids)
        if t > gate_up.mmvq_tokens:
            assert isinstance(xin, torch.Tensor) and torch.equal(xin, x[:, None])
            continue
        assert isinstance(xin, Prepared) and torch.equal(unit._gate_up(xin, ids), want)
        assert torch.equal(unit._gate_up(types.SimpleNamespace(x=x[:, None]), ids), want), "a foreign handle's .x"
        lib.quantize_q8_1(TYPES["q4_k"], other, xin.buffer)             # the buffer, generation unchanged: read
        assert torch.equal(unit._gate_up(xin, ids), unit._gate_up(other[:, None], ids)), "the handle's q8_1 unread"
        unit._prepare(other, unit.router, gate_up)                     # the buffer's next write
        assert torch.equal(unit._gate_up(xin, ids), want), "a stale handle read the buffer"


def bitwise(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Equal bits, NaN in the same places."""
    return a.dtype == b.dtype and torch.equal(a.isnan(), b.isnan()) and torch.equal(a.nan_to_num(), b.nan_to_num())


def select(ctx, op: str, **facts):
    spec = ctx.select_kernel(op, site="l0", **facts)
    assert spec.provider == "kernels.ggml_cuda" and {"graph_safe", "deterministic"} <= spec.props
    return spec.impl(ctx, **facts)


def test_moe_combine_within_one_ulp(ctx):
    facts = {"hidden": 2048, "top_k": K, "dtype": "bfloat16", "y_dtype": "float32"}
    spec = ctx.select_kernel("moe_combine@1", site="l0", **facts)
    assert spec.provider == "kernels.ggml_cuda" and {"graph_safe", "deterministic"} <= spec.props
    combine = spec.impl(ctx, **facts)
    for tokens in (0, 1, 13, CHUNK):
        gen = torch.Generator().manual_seed(tokens)
        y = (torch.randn(tokens, K, 2048, generator=gen) * 10 ** torch.empty(tokens, K, 1).uniform_(-3, 3,
                                                                                                    generator=gen))
        y[..., 128:192] = 0                                                  # sums of 0 (subnormal residuals below)
        weights = torch.rand(tokens, K, generator=gen).softmax(-1)
        out = torch.full((tokens, 2048), float("nan"), dtype=torch.bfloat16, device="cuda")
        combine(y.cuda(), weights.cuda(), out)
        want = reference.moe_combine(y.cuda(), weights.cuda(), torch.bfloat16)
        ulp = torch.maximum(torch.nextafter(want, torch.full_like(want, float("inf"))) - want,
                            want - torch.nextafter(want, torch.full_like(want, -float("inf")))).float()
        rounding = 2 * K * l0.U32 * (weights[..., None] * y).abs().sum(1).cuda()   # k f32 roundings each side
        assert ((out.float() - want.float()).abs() <= ulp + rounding).all(), f"T={tokens}: more than 1 ulp off"
        # `add`: bitwise the unfused residual add after the combine, also where they cancel and on subnormals
        residual = (torch.randn(tokens, 2048, generator=gen) * 10 ** torch.empty(tokens, 1).uniform_(-3, 3,
                                                                                                generator=gen))
        residual = residual.to(torch.bfloat16).cuda()
        residual[:, :64] = -out[:, :64]
        residual[:, 64:128] = -out[:, 64:128] * 1.0078125
        residual[:, 128:192] = 3e-39 * torch.randn(tokens, 64, generator=gen).cuda()
        fused = torch.full_like(out, float("nan"))
        combine.add(y.cuda(), weights.cuda(), residual, fused)
        assert torch.equal(fused, residual + out), f"T={tokens}: the fused add is not residual + the combine"


def test_swiglu_kernel_within_the_fp32_budget(lib):
    """Our f32 SwiGLU against f64, written out here (not `reference.py`): within the fp32 budget, also with each
    row's expert scales on gate and up (NVFP4's scale2, before the product); a bf16 SwiGLU cast to f32 misses it."""
    gen = torch.Generator().manual_seed(15)
    scale = 10 ** torch.empty(4099, 1).uniform_(-2, 1.5, generator=gen)          # gates from ±0.01 to ±100
    x = (torch.randn(4099, 2 * INTER, generator=gen) * scale).cuda()
    y = torch.empty(4099, INTER, device="cuda")
    g, u = x[:, :INTER].double(), x[:, INTER:].double()
    ids = torch.randint(0, E, (4099,), generator=gen, dtype=torch.int32).cuda()
    s = (2.0 ** torch.empty(2, E).uniform_(-4, 4, generator=gen)).cuda()        # [2, E]: gate's row, up's
    for scales in (None, (s[0], s[1], ids)):
        lib.swiglu(x, y, *(scales or (None,) * 3))
        sg, su = (s[0][ids.long()].double()[:, None], s[1][ids.long()].double()[:, None]) if scales else (1, 1)
        want = g * sg / (1 + torch.exp(-g * sg)) * u * su
        r = l0.check(y, want, torch.float32, 1)
        assert r.passed, r
    gb, ub = x[:, :INTER].bfloat16(), x[:, INTER:].bfloat16()
    want = g / (1 + torch.exp(-g)) * u
    assert not l0.check((torch.nn.functional.silu(gb) * ub).float(), want, torch.float32, 1).passed


def test_router_logits_within_the_fp32_budget(ctx, monkeypatch):
    # D3: the router is an f32 product of the bf16 hidden states (an f64 reference at the fp32 budget); a bf16
    # router or TF32 (`allow_tf32`, e.g. via `set_float32_matmul_precision("high")`) must miss it. The layer routes
    # through `moe_router@1`: its fused path's logits are test_moe_router's; its prompt chunks (here 300 rows) compose
    # the f32 linear and `moe_topk`, held here.
    unit, _ = moe_layer(ctx, "q4_k", "q4_k")
    assert unit._router is not None and unit._topk is None
    x = torch.randn(300, H, generator=torch.Generator().manual_seed(17)).bfloat16().cuda()
    want = x.double() @ unit.router.double().T
    seen = []
    route = unit._router.topk
    monkeypatch.setattr(unit._router, "topk", lambda logits: seen.append(logits.clone()) or route(logits))

    def logits() -> torch.Tensor:
        seen.clear()
        unit(x)
        return seen[0]

    assert torch.backends.cuda.matmul.allow_tf32 is False
    r = l0.check(logits(), want, torch.float32, H)
    assert r.passed, r
    bf16 = torch.nn.functional.linear(x, unit.router.bfloat16()).float()
    assert not l0.check(bf16, want, torch.float32, H).passed
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    assert not l0.check(logits(), want, torch.float32, H).passed


def test_moe_layer_binds_its_activation_dtypes_and_f32_outputs(ctx):
    unit, _ = moe_layer(ctx, "q4_k", "q4_k")
    sites = {v["facts"]["glu"]: v for k, v in ctx.kernels.items()
             if k.endswith("/linear_experts@2") and "experts=128" in k}
    assert set(sites) == {"swiglu", "none"} and all(v["provider"] == "kernels.ggml_cuda"
                                                    and v["facts"]["out_dtype"] == "float32" for v in sites.values())
    assert sites["swiglu"]["facts"]["act_dtype"] == "bfloat16"         # gate/up: the layer input
    assert sites["none"]["facts"]["act_dtype"] == "float32"            # down: the f32 SwiGLU output
    x = layer_input(3, seed=16)
    _, ids, _ = unit._route(x)
    h = unit._gate_up(x[:, None], ids)
    assert h.dtype == unit.down(h, ids).dtype == torch.float32 and unit(x).dtype == torch.bfloat16


def test_the_workspace_formula_with_ids_is_tight(ctx, lib):
    """Negative control with ids on a J = 128 shape where llama.cpp took the stream-k fixup: 8 experts of 384 rows at
    512 tokens are 3 · 8 · 4 = 96 tiles (dense mmq_schedule over out · E rows mirrors that grid when out is a multiple
    of I). With ids mmq now always tiles, and the formula, which has no fixup term, exceeds what the call takes by at
    most its spare."""
    n, out, device = 8, 384, torch.cuda.current_device()
    if lib.mmq_schedule(device, TYPES["q4_k"], PREC["q8_1"], out * n, CHUNK) != (2, 128, 128):
        pytest.skip("96 tiles need no stream-k fixup on this GPU's SM count")
    kernel, s = experts(ctx, "q4_k", H, out, 1, n=n), stack("q4_k", n, out, H, seed=5)
    kernel.kernel(gate_up_input(CHUNK, seed=1), s.parts, uniform(CHUNK, n, seed=2))
    used = lib.workspace_high_water()
    assert used > 0 and 0 <= lib.workspace_bytes_id(device, H, CHUNK, K, n) - used <= SLACK + reserve(CHUNK * K)


def nvfp4_stacks(ctx, n: int = E):
    """NVFP4 HF stacks through their kernels' `prepare`: the merged gate/up (fused SwiGLU) with scale2 [2, E] and
    down with [E] (distinct per expert and half); also down's HF parts."""
    fused = experts(ctx, "nvfp4/hf", H, 2 * INTER, 1, "swiglu", n)
    down = experts(ctx, "nvfp4/hf", INTER, H, K, n=n, act_dtype="float32")
    gate, up, dn = (hf_fp4("nvfp4", out, k, seed, experts=n) for out, k, seed in ((INTER, H, 1), (INTER, H, 2),
                                                                                   (H, INTER, 3)))
    merged = {role: torch.cat([gate[role], up[role]], 1) for role in ("weight", "scale")}
    gu = prepared(fused, "nvfp4", merged | {"scale2": torch.stack([gate["scale2"], up["scale2"]])})
    dw = prepared(down, "nvfp4", dn)
    assert gu["scale2"].shape == (2, n) and dw["scale2"].shape == (n,)
    return fused, down, gu, dw, dn


def test_nvfp4_scale2_per_expert_on_both_paths(ctx, lib):
    fused, down, gu, dw, dn = nvfp4_stacks(ctx)
    sg, sd = (stack("nvfp4", E, out, k, 0, p["blocks"].flatten(0, 1).cpu().numpy())        # unscaled, same blocks
              for p, out, k in ((gu, 2 * INTER, H), (dw, H, INTER)))
    none, down_none = experts(ctx, "nvfp4", H, 2 * INTER, 1), experts(ctx, "nvfp4", INTER, H, K, act_dtype="float32")
    t = fused.mmvq_tokens
    weighted = Stack("nvfp4", sd.blocks, sd.w, hf_weight("nvfp4", dn).cuda())   # the reference weight, scale2 in it
    for tokens in (1, t, t + 1, 64, CHUNK + 1):
        ids, x, h = uniform(tokens, E, seed=tokens), gate_up_input(tokens, seed=tokens), down_input(tokens, tokens)
        check_swiglu(lib, fused, none, sg, x, ids, f"swiglu T={tokens}", gu)
        y = down(h, dw, ids)
        check(y, emulated(down, weighted, h, ids), INTER, f"down T={tokens}")
        assert torch.equal(y, down_none(h, sd.parts, ids) * dw["scale2"][ids.long()][..., None]), f"down T={tokens}"
    ids, x, h = uniform(64, E, seed=5), gate_up_input(64, seed=5), down_input(64, 5)
    # D4: the fused store (docs/v1-moe-prefill.md) against ours with halves scaled to f32's smallest normals, their
    # products below them (flushed alike)
    assert fused.glu_store and check_swiglu(lib, fused, none, sg, x, ids, "subnormal halves",
                                            gu | {"scale2": gu["scale2"] * 1e-36}) is None
    with pytest.raises(ValueError, match=r"scale2 \(32,\) for 32 experts, glu swiglu"):    # one row for two halves
        fused(x, gu | {"scale2": dw["scale2"]}, ids)
    with PLANTS["nvfp4-scale2-by-slot"]():                                                                  # N4
        assert not torch.equal(down(h, dw, ids), down_none(h, sd.parts, ids) * dw["scale2"][ids.long()][..., None])
    with PLANTS["nvfp4-scale2-after-swiglu"](), pytest.raises(AssertionError, match="mmq's SwiGLU differs"):  # N5
        check_swiglu(lib, experts(ctx, "nvfp4/hf", H, 2 * INTER, 1, "swiglu"), none, sg, x, ids, "N5", gu)


@pytest.mark.parametrize("name", FP4)
def test_native_w4a4_experts_against_their_own_codes(ctx, lib, sm, name):
    """sm_120, `quant.act = a4`: mmq with ids at native W4A4 (gate/up: bf16 copied to f32, one row per token,
    scattered to its slots; down: f32, one row per slot) against the f64 product of its own decoded activations;
    each 512-token chunk on its own path (513 tokens: a one-token mmvq chunk, q8_1, against the emulator)."""
    if sm != 120:
        pytest.skip("native W4A4 is Blackwell's")
    gate_up = experts(ctx, name, H, 2 * INTER, 1, act_quant="a4")
    down = experts(ctx, name, INTER, H, K, act_dtype="float32", act_quant="a4")
    assert gate_up.act == down.act == NATIVE_FP4
    sg, sd = stack(name, E, 2 * INTER, H, seed=1), stack(name, E, H, INTER, seed=2)
    pg, pdn = (k.prepare(QTensor(Layout(Scheme(f"ggml.{name}"), "gguf.native"), tuple(s.weight.shape), s.parts)).parts
               for k, s in ((gate_up, sg), (down, sd)))
    for tokens in (gate_up.mmvq_tokens + 1, 64, CHUNK + 1):
        ids, x, h = uniform(tokens, E, seed=tokens), gate_up_input(tokens, seed=tokens), down_input(tokens, tokens)
        for kern, s, parts, inp, k in ((gate_up, sg, pg, x, H), (down, sd, pdn, h, INTER)):
            want = []
            for a in range(0, tokens, CHUNK):
                rows, ids_c = inp[a:a + CHUNK], ids[a:a + CHUNK]
                if len(ids_c) <= kern.mmvq_tokens:
                    want.append(q8_1.linear_experts(rows, s.weight, s.blocks, ids_c, name, q8_1.MMVQ))
                    continue
                xhat = native_codes(lib, name, rows.float().flatten(0, 1))[2].cuda().view(*rows.shape)
                want.append(decoded_experts(xhat, s.weight, ids_c))
            check(kern(inp, parts, ids), torch.cat(want), k, f"{name} in={k} T={tokens}")
    assert set(gate_up.calls) == set(down.calls) == {"mmq/fp4", "mmvq/q8_1"}


def decoded_experts(xhat: torch.Tensor, weight: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Slot j of token t: expert ids[t, j] of `weight` [E, out, in] on xhat[t, 0] or xhat[t, j], in f64."""
    t, k = ids.shape
    src = xhat.expand(t, k, -1) if xhat.shape[1] == 1 else xhat
    y = torch.zeros(t, k, weight.shape[1], dtype=torch.float64, device=xhat.device)
    for e in ids.unique().tolist():
        rows = ids == e
        y[rows] = src[rows].double() @ weight[e].double().T
    return y
