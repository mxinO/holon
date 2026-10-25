"""L0 for GLM-5.3-Flash's mHC kernels (`kernels.hc_triton`: `hc_mhc_mix@1`, `hc_mhc_writeback@1`; docs/v1-glm.md D3,
acceptance 3) against their oracles, through the build context, at GLM's n = 4 streams on H = 4,096 and the tiny GPU
fixture's 512.

- `hc_mhc_mix@1` within `l0.check` of the oracle (x at its dtype; post and comb in float32, over n·H terms; R over
  the write-back's n) for 0, 1, 7, 77 and 300 tokens, with and without a pending write-back, on **peaked** combs (base
  and scale spread so the Sinkhorn has work left at its last iterations): one iteration short (the plant
  `mhc_sinkhorn_short`, 19) is 156–539× its comb budget, where the kernel's comb is ≤ 0.31× (RTX A5000).
- The pending write-back leaves R bitwise what `hc_mhc_writeback@1` leaves; a block as the unit runs it (mix, the
  write-back riding in the next mix, its last write-back) leaves R within L0 of the oracle's block (≤ 0.75×; the
  oracle's own GPU run against its CPU run is at 0.35–0.51× on these shapes, ours 0.28–0.42×).
- Row invariance (the contract): a row's x, post and comb are bitwise the same alone, among 300 rows and among 2,000,
  with and without a pending write-back — one code path for every T.
- The in-place write-back under stress: many programs, repeated under side-stream load of varying size (so programs
  run in other orders): R bitwise the same every repeat and as the standalone write-back, within L0 of the
  out-of-place oracle; a write-back split over streams (each program writing one stream in place, reading all four)
  fails that check.
- Graph-safe: a captured mix and write-back replay to the eager bits.

Skips without a GPU.
"""
import pytest
import torch
import triton
import triton.language as tl
from holon.core.build import BuildContext
from holon.core.plugins import PluginSet, discover
from holon_judge import l0
from holon_std.hyper import kernels as hc_kernels
from holon_std.hyper import reference as hc

pytestmark = [pytest.mark.l0, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")]

N, EPS, HC_EPS, ITERS = 4, 1e-5, 1e-6, 20


def _kernels(hidden: int, iters: int = ITERS):
    pins = {"hc_mhc_mix": "kernels.hc_triton", "hc_mhc_writeback": "kernels.hc_triton"}
    ctx = BuildContext(PluginSet(discover()), torch.device("cuda"), options={"kernels": pins})
    facts = {"streams": N, "hidden": hidden, "eps": EPS, "hc_eps": HC_EPS, "iters": iters, "dtype": "bfloat16"}
    mix = ctx.select_kernel(hc_kernels.MHC_MIX, site="test", **facts).impl(ctx, **facts)
    facts = {"streams": N, "hidden": hidden, "dtype": "bfloat16"}
    return mix, ctx.select_kernel(hc_kernels.MHC_WRITEBACK, site="test", **facts).impl(ctx, **facts)


def _weights(hidden: int, g: torch.Generator, spread: float = 4.0):
    """fn, base, scale: comb logits spread by `spread` (peaked combs, the Sinkhorn far from converged early)."""
    fn = torch.randn(24, N * hidden, generator=g) / (N * hidden) ** 0.5
    return fn, torch.randn(24, generator=g) * spread, torch.tensor([1.0, 1.0, spread])


def _inputs(t: int, hidden: int, g: torch.Generator):
    r = torch.randn(t, N * hidden, generator=g) * torch.linspace(0.5, 4.0, N).repeat_interleave(hidden)
    y = torch.randn(t, hidden, generator=g).bfloat16()
    post = 2 * torch.rand(t, N, generator=g)
    comb = torch.softmax(torch.randn(t, N, N, generator=g) * 3, -1)
    return r, (y, post, comb)


def _cuda(*xs):
    return [x.cuda() if x is not None else None for x in xs]


def _usage(ours, ref, dtype, n, what):
    report = l0.check(ours, ref, dtype, n)
    print(f"\n{what}: usage {report.usage:.3g}")
    return report


@pytest.mark.parametrize("hidden", [512, 4096])
def test_mhc(hidden):
    g = torch.Generator().manual_seed(0)
    weights = _weights(hidden, g)
    mix, writeback = _kernels(hidden)
    short, _ = _kernels(hidden, ITERS - 1)
    for t in (0, 1, 7, 77, 300):
        r, pending = _inputs(t, hidden, g)
        for extra in ((None, None, None), pending):
            what = f"H {hidden}, {t} tokens{', pending' if extra[0] is not None else ''}"
            ours = r.cuda()
            x, post, comb = mix(ours, *_cuda(*weights), *_cuda(*extra))
            r_ref = r.clone()
            x_ref, post_ref, comb_ref = hc.hc_mhc_mix(r_ref, *weights, *extra, N, EPS, HC_EPS, ITERS, torch.bfloat16)
            assert x.dtype == torch.bfloat16 and x.shape == (t, hidden) and comb.shape == (t, N, N)
            for o, ref, dt, n, name in ((x, x_ref, torch.bfloat16, N * hidden, "x"),
                                        (post, post_ref, torch.float32, N * hidden, "post"),
                                        (comb, comb_ref, torch.float32, N * hidden, "comb"),
                                        (ours, r_ref, torch.float32, N, "r")):
                assert _usage(o, ref, dt, n, f"{name}, {what}").passed, (name, what)
            if extra[0] is not None:                         # R as hc_mhc_writeback leaves it, bit for bit
                alone = r.cuda()
                writeback(alone, *_cuda(*extra))
                assert torch.equal(alone, ours)
            if t >= 77 and extra[0] is None:                 # the plant: one Sinkhorn iteration short fails L0
                _, _, comb19 = short(r.cuda(), *_cuda(*weights))
                assert _usage(comb19, comb_ref, torch.float32, N * hidden, f"comb at 19, {what}").usage > 100
        if t:                                                # a block as the unit runs it: R complete at its end
            y_attn, y_ffn = (torch.randn(t, hidden, generator=g).bfloat16() for _ in range(2))
            ours = r.cuda()
            _, p_a, c_a = mix(ours, *_cuda(*weights))
            _, p_f, c_f = mix(ours, *_cuda(*weights), y_attn.cuda(), p_a, c_a)
            writeback(ours, y_ffn.cuda(), p_f, c_f)
            r_ref = r.clone()
            _, pr_a, cr_a = hc.hc_mhc_mix(r_ref, *weights, None, None, None, N, EPS, HC_EPS, ITERS, torch.bfloat16)
            _, pr_f, cr_f = hc.hc_mhc_mix(r_ref, *weights, y_attn, pr_a, cr_a, N, EPS, HC_EPS, ITERS, torch.bfloat16)
            hc.hc_mhc_writeback(r_ref, y_ffn, pr_f, cr_f)
            assert _usage(ours, r_ref, torch.float32, N * hidden, f"block, H {hidden}, {t} tokens").passed


def test_mhc_rows_are_invariant():
    hidden, g = 4096, torch.Generator().manual_seed(1)
    weights = _cuda(*_weights(hidden, g))
    mix, _ = _kernels(hidden)
    r, pending = _inputs(2000, hidden, g)
    r, pending = r.cuda(), _cuda(*pending)
    for extra in ((None, None, None), pending):
        outs = []
        for t in (2000, 300, 1):                             # row 0 among 2,000, among 300, alone
            ours = r[:t].clone()
            got = mix(ours, *weights, *[e[:t] if e is not None else None for e in extra])
            outs.append((*(o[:1] for o in got), ours[:1]))
        for other in outs[1:]:
            assert all(torch.equal(a, b) for a, b in zip(outs[0], other, strict=True))


@triton.jit
def _split_writeback(r_ptr, y_ptr, post_ptr, comb_ptr, H: tl.constexpr, BLOCK: tl.constexpr):
    """A write-back split over streams (the bug D3 forbids): program (row, d, block) writes stream d in place, reading
    all four — a later stream's program can read a stream already written."""
    t, d = tl.program_id(0).to(tl.int64), tl.program_id(1)
    h = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    at = r_ptr + t * 4 * H + h
    acc = tl.load(comb_ptr + t * 16 + d) * tl.load(at)
    for s in tl.static_range(1, 4):
        acc = tl.fma(tl.load(comb_ptr + t * 16 + s * 4 + d), tl.load(at + s * H), acc)
    acc = tl.fma(tl.load(post_ptr + t * 4 + d), tl.load(y_ptr + t * H + h).to(tl.float32), acc)
    tl.store(at + d * H, acc)


def _stress(writeback, r, extra, want_ref, side, load) -> bool:
    """Repeats of `writeback` under side-stream load: every repeat bitwise the first, the first within L0 of the
    out-of-place oracle."""
    first = None
    for rep in range(8):
        with torch.cuda.stream(side):                        # other work on the card: programs scheduled differently
            for _ in range(rep % 4):
                load @ load
        ours = r.clone()
        writeback(ours, *extra)
        torch.cuda.synchronize()
        first = ours if first is None else first
        if not torch.equal(ours, first):
            return False
    return l0.check(first, want_ref, torch.float32, N).passed


def test_mhc_writeback_in_place_stress():
    hidden, g = 4096, torch.Generator().manual_seed(2)
    weights = _cuda(*_weights(hidden, g))
    mix, writeback = _kernels(hidden)
    side, load = torch.cuda.Stream(), torch.randn(2048, 2048, device="cuda")

    def split(r, y, post, comb):
        _split_writeback[(len(r), N, hidden // 512)](r, y, post, comb, hidden, 512)

    for t in (1, 16, 77, 300):
        r, extra = _inputs(t, hidden, g)
        want = r.clone()
        hc.hc_mhc_writeback(want, *extra)                    # out of place: torch's matmul reads the old R
        r, extra = r.cuda(), _cuda(*extra)
        assert _stress(writeback, r, extra, want, side, load), t
        assert _stress(lambda r, *e: mix(r, *weights, *e), r, extra, want, side, load), t   # the mix's pending one
        if t >= 77:                                          # (fewer rows: every program may run in one wave)
            assert not _stress(split, r, extra, want, side, load), t                        # the check catches it


def test_mhc_graph():
    hidden, g = 4096, torch.Generator().manual_seed(3)
    weights = _cuda(*_weights(hidden, g))
    mix, writeback = _kernels(hidden)
    r, extra = _inputs(1, hidden, g)
    r, extra = r.cuda(), _cuda(*extra)
    eager = r.clone()
    want = mix(eager, *weights, *extra)
    writeback(eager, extra[0], want[1], want[2])
    static = r.clone()
    mix(static.clone(), *weights, *extra)                   # warm-up (compile) outside the capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got = mix(static, *weights, *extra)
        writeback(static, extra[0], got[1], got[2])
    static.copy_(r)
    graph.replay()
    torch.cuda.synchronize()
    assert all(torch.equal(a, b) for a, b in zip(got, want, strict=True)) and torch.equal(static, eager)
