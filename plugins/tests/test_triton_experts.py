"""L0 for `kernels.triton_experts` (docs/m02-moe.md §4.6): the dense grouped GEMM against the `linear_experts`
reference at the fp32 budget, routing edge cases, masks, poisoned memory past the stack, the merged stack, run-to-run
equality, graph replay with routing unlike the capture's, and the operand precision against an f64 reference written
out here (not from reference.py)."""
import pytest
import torch
from holon_judge import l0
from holon_plugins.kernels.triton_experts.kernel import BM, grouped
from holon_std.moe import reference as moe

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
E, K = 8, 2


@pytest.fixture(autouse=True)
def _seeded():
    torch.manual_seed(0)                                  # every test sees the same inputs on every run


def stack(experts: int, out: int, inp: int, seed: int) -> torch.Tensor:
    """bf16 weights followed by a NaN-poisoned expert: a read past the stack makes NaNs."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    buf = torch.randn(experts + 1, out, inp, device="cuda", generator=gen).bfloat16() * 0.1
    buf[experts] = float("nan")
    return buf[:experts]


def routing(t: int, kind: str, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    if t == 0:
        return torch.empty(0, K, dtype=torch.int32, device="cuda")
    if kind == "uniform":
        return torch.stack([torch.randperm(E, device="cuda", generator=gen)[:K] for _ in range(t)]).int()
    if kind == "one-expert":                                # every token's first slot on expert 3: many row blocks
        rest = torch.stack([torch.randperm(E - 1, device="cuda", generator=gen)[:K - 1] for _ in range(t)])
        return torch.cat([torch.full((t, 1), 3, device="cuda"), rest + (rest >= 3)], 1).int()
    if kind == "edges":                                    # only experts 0 and E-1; the rest are empty
        return torch.tensor([[0, E - 1]] * t, device="cuda").int()
    raise ValueError(kind)


def exact(x, w, ids, glu):
    """The contract in f64: the activation rounded to the weight dtype, exact products and sums, SwiGLU in f64."""
    xs = x.expand(ids.shape[0], K, x.shape[-1]) if x.shape[1] == 1 else x
    y = torch.einsum("tki,tkoi->tko", xs.to(w.dtype).double(), w.double()[ids.long()])
    if glu == "swiglu":
        gate, up = y.chunk(2, -1)
        y = torch.nn.functional.silu(gate) * up
    return y


def check(x, w, ids, glu):
    got = grouped(x, w, ids, glu)
    terms = w.shape[2] * (2 if glu == "swiglu" else 1)     # SwiGLU multiplies two reductions: twice the rounding
    r = l0.check(got, exact(x, w, ids, glu), torch.float32, terms)
    assert r.passed, r
    if ids.numel():                                       # and the oracle agrees with the contract
        assert l0.check(moe.linear_experts(x, w, ids, glu, operand_dtype=w.dtype), exact(x, w, ids, glu),
                        torch.float32, terms).passed


@pytest.mark.parametrize("kind", ["uniform", "one-expert", "edges"])
@pytest.mark.parametrize("t", [0, 1, BM // K, BM // K + 1, 64, 513])
@pytest.mark.parametrize("glu", ["none", "swiglu"])
def test_matches_the_reference(kind, t, glu):
    w = stack(E, 2 * 48 if glu == "swiglu" else 70, 100, seed=1)          # off the tile multiples: masks
    x = torch.randn(t, 1, 100, device="cuda", generator=torch.Generator(device="cuda").manual_seed(t)).bfloat16()
    check(x, w, routing(t, kind), glu)


def test_down_reads_each_slots_own_row():
    w = stack(E, 64, 96, seed=2)
    x = torch.randn(40, K, 96, device="cuda")                              # f32: rounded to bf16 by the kernel
    check(x, w, routing(40, "uniform"), "none")


def test_the_merged_stack_equals_two_separate_ones():
    gate, up = stack(E, 48, 128, seed=3), stack(E, 48, 128, seed=4)
    x, ids = torch.randn(77, 1, 128, device="cuda").bfloat16(), routing(77, "uniform")
    merged = grouped(x, torch.cat([gate, up], 1), ids, "none")
    assert torch.equal(merged, torch.cat([grouped(x, gate, ids, "none"), grouped(x, up, ids, "none")], -1))


def test_run_to_run_and_graph_replay_with_new_routing():
    w = stack(E, 2 * 64, 128, seed=5)
    x = torch.randn(24, 1, 128, device="cuda").bfloat16()
    ids = routing(24, "uniform", seed=1)
    first = grouped(x, w, ids, "swiglu")
    assert torch.equal(first, grouped(x, w, ids, "swiglu"))
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):                           # torch's default error mode: a host sync would fail
        out = grouped(x, w, ids, "swiglu")
    ids.copy_(routing(24, "one-expert", seed=2))
    graph.replay()
    assert torch.equal(out, grouped(x, w, ids, "swiglu"))


def test_operand_precision_against_f64():
    """bf16(h) · W at the fp32 budget: a TF32 product (10 mantissa bits, not bf16's 7) or an f32 operand misses it."""
    w = stack(E, 64, 768, seed=6)
    h = torch.randn(32, K, 768, device="cuda")                             # the f32 SwiGLU output, as down sees it
    ids = routing(32, "uniform", seed=3)
    want = exact(h, w, ids, "none")
    r = l0.check(grouped(h, w, ids, "none"), want, torch.float32, 768)
    assert r.passed, r
    tf32 = (h.view(torch.int32) + 0x1000 & ~0x1FFF).view(torch.float32)   # 10 mantissa bits (round half up)
    for operand in (h, tf32):                                              # the plants, products exact in f64
        plant = torch.einsum("tki,tkoi->tko", operand.double(), w.double()[ids.long()])
        assert not l0.check(plant, want, torch.float32, 768).passed
