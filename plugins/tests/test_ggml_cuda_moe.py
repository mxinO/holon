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
- Routings, at E = 128, k = 8: uniform; every token on the same 8 experts at 512 and 513 tokens (several column
  tiles inside one expert); half the experts empty; ids 0 and E−1 in every row with poisoned bytes before and after
  the stack. Token counts 0, 1, t, t + 1, 9, 64, 512, 513, 1025, 2049 (t: the type's mmvq limit); the two paths
  agree across the limit. Mixed types in one std.moe layer at 7 tokens (sm_86: IQ3_S gate/up on mmq, IQ4_XS down on
  mmvq); ids straight from `moe_topk`; CUDA graphs replayed with routing unlike the capture's at every decode
  bucket.
- The padding plant (§4.6): padding rows with all k slots on one expert and the workspace pre-filled with a real
  row's index change real rows on mmq and leave them alone on mmvq.
- `moe_topk`: the reference's picks up to near-ties (fast-math `__expf` moves probabilities by a few ulps), its
  weights; ±inf and all-NaN rows give unique in-range ids; token counts off the kernel's 8-row blocks.
- `moe_combine`: within 1 bf16 ulp of the reference, plus the f32 accumulation's rounding (the kernel's FMA) for
  sums that cancel.
- Precision (f64 references written out here): our f32 SwiGLU within the fp32 budget, which a bf16 SwiGLU misses;
  a std.moe layer binds `linear_experts` with act_dtype bf16 and out_dtype f32, and its expert outputs are f32.

Skips without a GPU; builds the extension (prebuild) into the user cache if it is missing.
"""
import dataclasses
import os

import numpy as np
import pytest
import torch
from holon.api import Layout, QTensor, Ref, Scheme
from holon.core.build import BuildContext
from holon.core.executor import BUCKETS
from holon.core.plugins import PluginSet, discover
from holon_judge import l0, q8_1
from holon_plugins.kernels.ggml_cuda import extension
from holon_plugins.kernels.ggml_cuda.plugin import CHUNK, TYPES
from holon_std.moe import reference

from .test_ggml_cuda import activations, dequantize, random_blocks

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

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
    try:
        extension.load(sm)
    except extension.NotBuilt:
        extension.build(sm, os.cpu_count())
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
        return {"blocks": self.w}


def stack(name: str, experts: int, out: int, k: int, seed: int, blocks: np.ndarray | None = None) -> Stack:
    blocks = random_blocks(name, experts * out, k, seed) if blocks is None else blocks
    buf = torch.full((blocks.size + 2 * POISON,), 0xFF, dtype=torch.uint8, device="cuda")
    buf[POISON:POISON + blocks.size] = torch.from_numpy(blocks.reshape(-1)).cuda()
    w = buf[POISON:POISON + blocks.size].view(blocks.shape)
    return Stack(name, blocks, w, dequantize(name, blocks).view(experts, out, k))


def experts(ctx, name: str, k: int, out: int, rows_per_token: int, glu: str = "none", n: int = E,
            act_dtype: str = "bfloat16"):
    facts = {"layout": f"ggml.{name}/gguf.native", "experts": n, "in": k, "out": out, "top_k": K,
             "rows_per_token": rows_per_token, "glu": glu, "act_dtype": act_dtype, "out_dtype": "float32"}
    spec = ctx.select_kernel("linear_experts@1", site="l0", **facts)
    assert spec.provider == "kernels.ggml_cuda" and {"graph_safe", "deterministic"} <= spec.props
    return spec.impl(ctx, **facts)


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


def check_swiglu(lib, fused, none, s: Stack, x: torch.Tensor, ids: torch.Tensor, what: str) -> None:
    """`glu="swiglu"` against our SwiGLU of the `glu="none"` output: bitwise where mmq ran, within the fp32 budget
    where mmvq's fused gate did."""
    y, gate_up = fused(x, s.parts, ids), none(x, s.parts, ids)
    ours = torch.empty_like(y)
    if len(y):
        lib.swiglu(gate_up, ours)
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
    for f in ("topk-moe.cu", "moe-weighted-reduction.cu"):
        assert "Holon:" in (extension.CUDA_DIR / f).read_text()


@pytest.mark.parametrize("name", sorted(TYPES))
def test_linear_experts_every_type(ctx, lib, name):
    gate_up, fused = experts(ctx, name, H, 2 * INTER, 1), experts(ctx, name, H, 2 * INTER, 1, "swiglu")
    down = experts(ctx, name, INTER, H, K)
    t = gate_up.mmvq_tokens
    assert 1 <= t <= 8 and fused.mmvq_tokens == down.mmvq_tokens == t
    sg, sd = stack(name, E, 2 * INTER, H, seed=1), stack(name, E, H, INTER, seed=2)
    for tokens in (1, t, t + 1, 64):
        ids, x, h = uniform(tokens, E, seed=tokens), gate_up_input(tokens, seed=tokens), down_input(tokens, tokens)
        check(gate_up(x, sg.parts, ids), emulated(gate_up, sg, x, ids), H, f"gate_up {name} T={tokens}", True)
        check_swiglu(lib, fused, gate_up, sg, x, ids, f"swiglu {name} T={tokens}")
        y = down(h, sd.parts, ids)
        assert torch.equal(y, down(h, sd.parts, ids)), f"down {name} T={tokens}: not run-to-run identical"
        check(y, emulated(down, sd, h, ids), INTER, f"down {name} T={tokens}")


@pytest.mark.parametrize("name", sorted(TYPES))
def test_linear_experts_every_type_at_prefill_chunks(ctx, name):
    """mmq's prefill kernels on every type: the tile width J follows the chunk's tokens (128 at a 512-token chunk,
    64 at the sweep's 64), and each J is its own kernel. A small stack (8 experts, 256 rows) keeps it to ~1M
    outputs; 513 tokens adds the one-token mmvq tail chunk."""
    n, out = 8, 256
    kernel, s = experts(ctx, name, H, out, 1, n=n), stack(name, n, out, H, seed=5)
    for tokens in (CHUNK, CHUNK + 1):
        ids, x = uniform(tokens, n, seed=tokens), gate_up_input(tokens, seed=tokens)
        check(kernel(x, s.parts, ids), emulated(kernel, s, x, ids), H, f"{name} T={tokens}", True)


@pytest.mark.parametrize("name", ["q4_k", "iq3_xxs", "q6_k", "q4_0", "q2_k"])
def test_merged_stack_equals_separate_stacks(ctx, lib, name):
    """Gate and up merged per expert give each stack's own output bitwise on mmvq and on mmq without the fixup (with
    it, the stream-k split moves with the number of row tiles)."""
    sep, merged = experts(ctx, name, H, INTER, 1), experts(ctx, name, H, 2 * INTER, 1)
    fused = experts(ctx, name, H, 2 * INTER, 1, "swiglu")
    gate, up = random_blocks(name, E * INTER, H, seed=3), random_blocks(name, E * INTER, H, seed=4)
    both = np.concatenate([gate.reshape(E, INTER, -1), up.reshape(E, INTER, -1)], axis=1).reshape(E * 2 * INTER, -1)
    sg, su, sm_ = (stack(name, E, n, H, 0, b) for n, b in ((INTER, gate), (INTER, up), (2 * INTER, both)))
    t, exact_mmq = merged.mmvq_tokens, 0
    for tokens in (1, t, t + 1, 64, CHUNK):
        ids, x = uniform(tokens, E, seed=tokens), gate_up_input(tokens, seed=tokens)
        calls = launched(lambda x=x, ids=ids: [sep(x, sg.parts, ids), sep(x, su.parts, ids), merged(x, sm_.parts, ids)])
        if any("stream_k_fixup" in n for n in calls):
            continue
        exact_mmq += tokens > t
        y = merged(x, sm_.parts, ids)
        assert torch.equal(y[..., :INTER], sep(x, sg.parts, ids)), f"{name} T={tokens}: gate rows differ"
        assert torch.equal(y[..., INTER:], sep(x, su.parts, ids)), f"{name} T={tokens}: up rows differ"
        if tokens <= t:
            ours = torch.empty(tokens, K, INTER, device="cuda")
            lib.swiglu(y, ours)
            r = l0.check(fused(x, sm_.parts, ids), ours, torch.float32, 1)
            assert r.passed, f"{name} T={tokens}: the fused gate vs ours: {r}"
    if not exact_mmq:       # the fixup runs unless the tiles divide evenly over the SMs: never here at these sizes
        sms = torch.cuda.get_device_properties(0).multi_processor_count
        pytest.skip(f"{name}: every mmq size took the stream-k fixup on this GPU ({sms} SMs): mmvq compared, mmq not")


@pytest.fixture(scope="module")
def qwen_stacks():
    """Q4_K and IQ3_XXS stacks at Qwen3-30B-A3B's E = 128 (gate/up merged, down)."""
    return {name: (stack(name, EQ, 2 * INTER, H, seed=5), stack(name, EQ, H, INTER, seed=6))
            for name in ("q4_k", "iq3_xxs")}


@pytest.mark.parametrize("name", ["q4_k", "iq3_xxs"])
def test_routings_and_sizes(ctx, lib, qwen_stacks, name):
    sg, sd = qwen_stacks[name]
    gate_up, fused = experts(ctx, name, H, 2 * INTER, 1, n=EQ), experts(ctx, name, H, 2 * INTER, 1, "swiglu", n=EQ)
    down = experts(ctx, name, INTER, H, K, n=EQ)
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


def test_mixed_types_in_one_layer(ctx, sm):
    """A std.moe layer with IQ3_S gate/up (mmq above 6 tokens on sm_86) and IQ4_XS down (mmvq up to 8) at 7 tokens:
    each stack takes its own path, as the per-instance counters show, and matches the emulator."""
    if sm != 86:
        pytest.skip("the IQ3_S / IQ4_XS limits that split 7 tokens across the paths are sm_86's")
    unit, (sg, su, sd) = moe_layer(ctx, "iq3_s", "iq4_xs")
    x = layer_input(7, seed=9)
    weights, ids = unit._topk(torch.nn.functional.linear(x.float(), unit.router))
    h = unit._gate_up(x[:, None], ids)
    y = unit.down(h, ids)
    assert unit._gate_up._kernel.calls == {"mmvq": 0, "mmq": 1} and unit.down._kernel.calls == {"mmvq": 1, "mmq": 0}
    merged = stack("iq3_s", EQ, 2 * INTER, H, 0, unit._gate_up.weight.cpu().numpy())
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
        assert getattr(unit, part).adopt(ctx, "weight", q)
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


@pytest.mark.parametrize("name", ["q4_k", "iq3_xxs"])
def test_graph_replay_follows_the_routing(ctx, qwen_stacks, name):
    """Captured at every decode bucket (mmvq up to the limit, mmq above), replayed with other ids and inputs: the
    result is the eager one on the new routing, bitwise."""
    sg, sd = qwen_stacks[name]
    gate_up, down = experts(ctx, name, H, 2 * INTER, 1, "swiglu", n=EQ), experts(ctx, name, INTER, H, K, n=EQ)
    for tokens in BUCKETS:
        x, ids = layer_input(tokens, seed=tokens)[:, None], uniform(tokens, EQ, seed=tokens)

        def layer(x=x, ids=ids):
            return down(gate_up(x, sg.parts, ids), sd.parts, ids)

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


@pytest.mark.parametrize("path", ["mmvq", "mmq"])
@pytest.mark.parametrize("stage", ["gate_up", "down"])
def test_padding_plant_changes_real_rows_on_mmq_only(ctx, qwen_stacks, path, stage):
    """The plant of design §4.6: padding rows with all k slots on one expert (a duplicate inside a row), the
    `ggml_cuda.id` workspace pre-filled with a real row's index. mmq's compaction counts the duplicates in
    `expert_bounds` but compacts each token once per expert, so the expert's range ends in k − 1 stale entries per
    padding row: they write over the real row (and, on gate/up, scatter the padding input over its compact row).
    mmvq applies each slot on its own: real rows are unchanged, bitwise. The stale entries are asserted in the
    workspace (`ids_src1`, `ids_dst`, `expert_bounds`, each padded to 256 B, as mmq allocates them); which writer of a
    real row lands last is a race, so the changed output is the secondary check (it held in 100/100 routings)."""
    sg, sd = qwen_stacks["q4_k"]
    s, kern = (sg, experts(ctx, "q4_k", H, 2 * INTER, 1, n=EQ)) if stage == "gate_up" else \
        (sd, experts(ctx, "q4_k", INTER, H, K, n=EQ))
    tokens = kern.mmvq_tokens if path == "mmvq" else 16
    real, k_in = tokens - 2, s.weight.shape[2]
    ids = uniform(tokens, EQ, seed=14)
    x = gate_up_input(tokens, seed=14) if stage == "gate_up" else down_input(tokens, seed=14)
    clean = kern(x, s.parts, ids)
    victim = K + int(ids[1].argmin())          # token 1's slot on its lowest expert: computed before the plant's
    planted = ids.clone()
    planted[real:] = EQ - 2                    # after the victim's expert, and not the last (whose tail is unused)
    kern.workspace.view(torch.int32).fill_(victim)
    y = kern(x, s.parts, planted)
    if path == "mmvq":
        assert torch.equal(y[:real], clean[:real])
    else:
        ws, n = kern.workspace.view(torch.int32), -(-tokens * K * 4 // 256) * 64
        ids_dst, bounds = ws[n:2 * n], ws[2 * n:2 * n + EQ + 1]
        lo, hi = int(bounds[EQ - 2]), int(bounds[EQ - 1])
        assert hi - lo == int((planted == EQ - 2).sum()), "expert_bounds counts every slot of the expert"
        stale = (K - 1) * (tokens - real)      # one compacted entry per token, k per padding row in the bounds
        assert (ids_dst[hi - stale:hi] == victim).all() and (ids_dst[lo:hi - stale] != victim).all()
        ref = emulated(kern, s, x, ids)[:real]
        assert l0.check(clean[:real], ref, torch.bfloat16, k_in).passed
        assert not l0.check(y[:real], ref, torch.bfloat16, k_in).passed, "the plant left the real rows alone"


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


def test_moe_combine_within_one_ulp(ctx):
    facts = {"hidden": 2048, "top_k": K, "dtype": "bfloat16", "y_dtype": "float32"}
    spec = ctx.select_kernel("moe_combine@1", site="l0", **facts)
    assert spec.provider == "kernels.ggml_cuda" and {"graph_safe", "deterministic"} <= spec.props
    combine = spec.impl(ctx, **facts)
    for tokens in (0, 1, 13, CHUNK):
        gen = torch.Generator().manual_seed(tokens)
        y = (torch.randn(tokens, K, 2048, generator=gen) * 10 ** torch.empty(tokens, K, 1).uniform_(-3, 3,
                                                                                                    generator=gen))
        weights = torch.rand(tokens, K, generator=gen).softmax(-1)
        out = torch.full((tokens, 2048), float("nan"), dtype=torch.bfloat16, device="cuda")
        combine(y.cuda(), weights.cuda(), out)
        want = reference.moe_combine(y.cuda(), weights.cuda(), torch.bfloat16)
        ulp = torch.maximum(torch.nextafter(want, torch.full_like(want, float("inf"))) - want,
                            want - torch.nextafter(want, torch.full_like(want, -float("inf")))).float()
        rounding = 2 * K * l0.U32 * (weights[..., None] * y).abs().sum(1).cuda()   # k f32 roundings each side
        assert ((out.float() - want.float()).abs() <= ulp + rounding).all(), f"T={tokens}: more than 1 ulp off"


def test_swiglu_kernel_within_the_fp32_budget(lib):
    """Our f32 SwiGLU against f64, written out here (not `reference.py`): within the fp32 budget; a bf16 SwiGLU cast
    to f32 misses it."""
    gen = torch.Generator().manual_seed(15)
    scale = 10 ** torch.empty(4099, 1).uniform_(-2, 1.5, generator=gen)          # gates from ±0.01 to ±100
    x = (torch.randn(4099, 2 * INTER, generator=gen) * scale).cuda()
    y = torch.empty(4099, INTER, device="cuda")
    lib.swiglu(x, y)
    g, u = x[:, :INTER].double(), x[:, INTER:].double()
    want = g / (1 + torch.exp(-g)) * u
    r = l0.check(y, want, torch.float32, 1)
    assert r.passed, r
    gb, ub = x[:, :INTER].bfloat16(), x[:, INTER:].bfloat16()
    assert not l0.check((torch.nn.functional.silu(gb) * ub).float(), want, torch.float32, 1).passed


def test_router_logits_within_the_fp32_budget(ctx, monkeypatch):
    # D3: the router is an f32 product of the bf16 hidden states (an f64 reference at the fp32 budget); a bf16
    # router or TF32 (`allow_tf32`, e.g. via `set_float32_matmul_precision("high")`) must miss it.
    unit, _ = moe_layer(ctx, "q4_k", "q4_k")
    x = torch.randn(300, H, generator=torch.Generator().manual_seed(17)).bfloat16().cuda()
    want = x.double() @ unit.router.double().T
    seen = []
    route = unit._topk
    monkeypatch.setattr(unit, "_topk", lambda logits: seen.append(logits.clone()) or route(logits))

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
             if k.endswith("/linear_experts@1") and "experts=128" in k}
    assert set(sites) == {"swiglu", "none"} and all(v["provider"] == "kernels.ggml_cuda"
                                                    and v["facts"]["out_dtype"] == "float32" for v in sites.values())
    assert sites["swiglu"]["facts"]["act_dtype"] == "bfloat16"         # gate/up: the layer input
    assert sites["none"]["facts"]["act_dtype"] == "float32"            # down: the f32 SwiGLU output
    x = layer_input(3, seed=16)
    _, ids = unit._topk(torch.nn.functional.linear(x.float(), unit.router))
    h = unit._gate_up(x[:, None], ids)
    assert h.dtype == unit.down(h, ids).dtype == torch.float32 and unit(x).dtype == torch.bfloat16
