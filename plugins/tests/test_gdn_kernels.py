"""L0 for the Gated DeltaNet kernels (docs/m03-qwen38.md D3, §4.7): `kernels.fla`'s `gated_delta_rule` and
`kernels.gdn_triton`'s `causal_conv1d` and `gated_rmsnorm` (with FN's sigmoid `out_gate` too), selected through the
build context, on Qwen3.8-27B's shapes (Hk 16, Hv 48: head ratio 3, conv channels 10240) and Qwen3.5's (Hv 32: ratio 2,
8192 channels), Dk = Dv = 128, K = 4, in both v-head orders, plus a small shape at head dim 64.

- Batches mix lengths around the 64-token chunk (1, 2, 63, 64, 65, 200, 1000), rows with and without has_init, slot 0
  and the last slot, and a zero-token row. Every unit starts poisoned with NaN except those of has_init rows and of
  the zero-token row: units of rows without has_init must not be read, and units no row writes (the unused ones and
  the zero-token row's) must stay bitwise untouched.
- `gated_delta_rule` against an f64 recurrence written out here: per row, the relative Frobenius error of the output
  and of the final state within 2× that of HF's bf16 torch path on the same inputs (transformers' own functions:
  l2norm and β in bf16, f32 core, bf16 output; its chunked path for prefill rows, its recurrent one for decode),
  for one step and for decode steps continuing a prefill.
- `causal_conv1d` within `l0.check(bf16, K)` of the oracle (in f32, unrounded) and its state bitwise; `gated_rmsnorm`
  within `l0.check(dtype, dim)` of the oracle.
- Exactness: a 64-aligned split (128 then 72 tokens) bitwise equal to one shot (both ops); a decode step (conv, rule,
  norm) replayed from a CUDA graph with new inputs bitwise equal to eager; runs repeat bitwise; plan and call
  never sync with the host.
- `gdn_rewind` (docs/m07-perf.md D5) bitwise equal to its oracle on three layers at once, every row accepting a
  different number of drafts, at the widths of MTP and DFlash2.
- Plants that must fail: conv1d without its state carry; has_init ignored (states read when False); a row on slot 0
  skipped (fla's original `<= 0` sentinel, rebuilt from the vendored source).

Run with `-s` to see the worst error ratios. Skips without a GPU.
"""
import importlib.util
import inspect

import pytest
import torch
import torch.nn.functional as F
from holon.core.build import BuildContext
from holon.core.plugins import PluginSet, discover
from holon_judge import l0
from holon_plugins.kernels.fla import plugin as fla_plugin
from holon_plugins.kernels.fla.vendor import fused_sigmoid_gating
from holon_std.mixers.gdn import reference as ref

pytestmark = [pytest.mark.l0, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")]
hf = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")

SHAPES = {"qwen3.8-27b": (16, 48, 128), "qwen3.5": (16, 32, 128), "small": (2, 6, 64)}   # Hk, Hv, D
KERNEL, UNITS, EPS = 4, 16, 1e-6
LENS = [1, 2, 63, 64, 0, 65, 200, 1000]
INIT = [True, False, True, False, True, True, False, True]
SLOTS = [0, UNITS - 1, 3, 7, 9, 1, 12, 5]                      # row 4 has no tokens: unit 9 stays as it is
DECODE_INIT = [True, False, True, True, False, True]
DECODE_SLOTS = [0, UNITS - 1, 4, 2, 11, 6]
RATIO = 2.0                                                    # × HF's bf16 error
FLOOR = 2e-2                                                   # HF's bf16 error itself stays below this
# The yardstick's transformers, the release in uv.lock: the bound was set at 5.6.0.dev0, whose torch path 5.6.2 keeps
# (l2norm in bf16; the chunked path differs only by a mask the decay mask already applies). Later versions
# l2-normalise q/k in f32, a smaller HF error against which the same kernels measure ≈ 2.3× on mixed batches, so the
# ratio holds only for the torch path it was set on.
HF_VERSION = "5.6.2"


@pytest.fixture(autouse=True)
def _seeded():
    import transformers

    if transformers.__version__ != HF_VERSION:
        pytest.fail(f"transformers {transformers.__version__}: the 2x bound is set against HF's torch path at "
                    f"{HF_VERSION} (l2norm in bf16)")
    torch.manual_seed(0)


def channels(hk: int, hv: int, d: int) -> int:
    return 2 * hk * d + hv * d                                 # the conv runs over q, k and v


def select(op: str, **facts):
    ctx = BuildContext(PluginSet(discover()), torch.device("cuda"))
    spec = ctx.select_kernel(op, site="test", **facts)
    assert spec.provider in ("kernels.fla", "kernels.gdn_triton")
    return spec.impl(ctx, **facts)


def gdn(shape: str, order: str):
    hk, hv, d = SHAPES[shape]
    return select("gated_delta_rule@1", heads_k=hk, heads_v=hv, head_k_dim=d, head_v_dim=d, v_head_order=order,
                  dtype="bfloat16")


def conv(c: int, dtype=torch.bfloat16):
    return select("causal_conv1d@1", channels=c, kernel_size=KERNEL, dtype=str(dtype).removeprefix("torch."))


def query_start(lens) -> torch.Tensor:
    return torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32, device="cuda")


def dev(values, dtype=torch.int32) -> torch.Tensor:
    return torch.tensor(values, dtype=dtype, device="cuda")


# --- gated_delta_rule: inputs, the f64 recurrence and HF's bf16 path ---------------------------------------------

def tokens(t: int, hk: int, hv: int, d: int) -> dict:
    """bf16 q/k/v/a/b and f32 gate parameters, decays spread from ~0.7 to ~0.999 per token across heads."""
    return dict(q=torch.randn(t, hk, d, device="cuda").bfloat16(), k=torch.randn(t, hk, d, device="cuda").bfloat16(),
                v=torch.randn(t, hv, d, device="cuda").bfloat16(), a=torch.randn(t, hv, device="cuda").bfloat16(),
                b=torch.randn(t, hv, device="cuda").bfloat16())


def gates(hv: int) -> dict:
    return dict(a_log=torch.linspace(-4.0, 0.5, hv, device="cuda"), dt_bias=torch.full((hv,), -2.0, device="cuda"))


def rows(x: dict, lo: int, hi: int) -> dict:
    return {n: x[n][lo:hi] for n in ("q", "k", "v", "a", "b")}


def exact(x: dict, g: dict, h0: torch.Tensor | None, order: str) -> tuple[torch.Tensor, torch.Tensor]:
    """The contract in f64, token by token: (o [T, Hv, D], final state [Hv, Dv, Dk])."""
    hk, hv, d = x["q"].shape[1], x["v"].shape[1], x["v"].shape[2]
    idx = ref.head_map(hk, hv, order).cuda()

    def l2(y):
        return y * torch.rsqrt((y * y).sum(-1, keepdim=True) + ref.L2_EPS)

    q, k = l2(x["q"].double())[:, idx] * d ** -0.5, l2(x["k"].double())[:, idx]
    decay = (-g["a_log"].double().exp() * F.softplus(x["a"].double() + g["dt_bias"].double())).exp()
    beta, v = torch.sigmoid(x["b"].double()), x["v"].double()
    s = torch.zeros(hv, d, d, dtype=torch.float64, device="cuda") if h0 is None else h0.double().clone()
    out = []
    for t in range(len(v)):
        s = s * decay[t][:, None, None]
        s = s + (beta[t][:, None] * (v[t] - (s @ k[t][:, :, None])[..., 0]))[:, :, None] * k[t][:, None, :]
        out.append((s @ q[t][:, :, None])[..., 0])
    return torch.stack(out) if out else v.new_zeros(0, hv, d), s


def hf_bf16(x: dict, g: dict, h0: torch.Tensor | None, order: str) -> tuple[torch.Tensor, torch.Tensor]:
    """What HF's bf16 model computes on these inputs (its torch path), in the kernels' state layout."""
    idx = ref.head_map(x["q"].shape[1], x["v"].shape[1], order).cuda()
    gate = -g["a_log"].exp() * F.softplus(x["a"].float() + g["dt_bias"])
    fn = hf.torch_recurrent_gated_delta_rule if len(x["v"]) == 1 and h0 is not None else hf.torch_chunk_gated_delta_rule
    o, s = fn(x["q"][:, idx][None], x["k"][:, idx][None], x["v"][None], gate[None], x["b"].sigmoid()[None],
              initial_state=None if h0 is None else h0.transpose(-1, -2)[None], output_final_state=True,
              use_qk_l2norm_in_kernel=True)
    return o[0], s[0].transpose(-1, -2)


def fro(got: torch.Tensor, want: torch.Tensor) -> float:
    return ((got.double() - want).norm() / want.norm()).item()


def ratio(got: torch.Tensor, want: torch.Tensor, floor: torch.Tensor) -> float:
    """The kernel's error over HF's bf16 error, both vs f64; HF's must itself be small (a broken floor fails)."""
    assert fro(floor, want) < FLOOR, fro(floor, want)
    return fro(got, want) / fro(floor, want)


def warm_states(x: dict, g: dict, order: str, units: int) -> torch.Tensor:
    """A pool of NaN units; realistic states (64 tokens of the f64 recurrence) are written where a test wants them."""
    hv, d = x["v"].shape[1], x["v"].shape[2]
    pool = torch.full((units, hv, d, d), float("nan"), device="cuda")
    warm = tokens(64, x["q"].shape[1], hv, d)
    return pool, exact(warm, g, None, order)[1].float()


def judge_rows(o, state, x, g, h0s, lens, slots, order, what: str) -> float:
    """Per row with tokens: output and final-state errors vs f64, each within RATIO × HF's bf16; the worst ratio."""
    worst, qs = 0.0, query_start(lens).tolist()
    for r, (lo, hi) in enumerate(zip(qs[:-1], qs[1:], strict=True)):
        if hi == lo:
            continue
        want_o, want_s = exact(rows(x, lo, hi), g, h0s[r], order)
        hf_o, hf_s = hf_bf16(rows(x, lo, hi), g, h0s[r], order)
        for name, got, want, floor in (("o", o[lo:hi], want_o, hf_o), ("state", state[slots[r]], want_s, hf_s)):
            assert torch.isfinite(got).all(), (what, r, name)
            x_ratio = ratio(got, want, floor)
            assert x_ratio <= RATIO, (what, r, hi - lo, name, fro(got, want), fro(floor, want))
            worst = max(worst, x_ratio)
    return worst


def run_batch(kernel, x, g, pool, lens, init, slots):
    return kernel(kernel.plan(lens), x["q"], x["k"], x["v"], x["a"], x["b"], g["a_log"], g["dt_bias"], pool,
                  dev(slots), dev(init, torch.bool))


# --- gated_delta_rule ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("order", ["grouped", "tiled"])
def test_the_f64_recurrence_is_the_oracle(order):
    hk, hv, d = SHAPES["small"]
    x, g = tokens(70, hk, hv, d), gates(hv)
    h0 = warm_states(x, g, order, 1)[1]
    state = h0[None].clone()
    got = ref.gated_delta_rule(x["q"], x["k"], x["v"], x["a"], x["b"], g["a_log"], g["dt_bias"], state, dev([0]),
                               query_start([70]), dev([True], torch.bool), order)
    want_o, want_s = exact(x, g, h0, order)
    assert l0.check(got, want_o, torch.bfloat16, d).passed
    torch.testing.assert_close(state[0].double(), want_s, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("order", ["grouped", "tiled"])
@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("batch", ["mixed", "decode"])
def test_delta_rule_within_twice_hf_bf16(shape, order, batch):
    hk, hv, d = SHAPES[shape]
    lens, init, slots = (LENS, INIT, SLOTS) if batch == "mixed" else ([1] * len(DECODE_SLOTS), DECODE_INIT,
                                                                       DECODE_SLOTS)
    x, g = tokens(sum(lens), hk, hv, d), gates(hv)
    pool, warm = warm_states(x, g, order, UNITS)
    h0s = []
    for r, (unit, has) in enumerate(zip(slots, init, strict=True)):
        if has or lens[r] == 0:                                 # zero-token rows hold a state they must keep
            pool[unit] = warm * (1 + r / 8)
        h0s.append(pool[unit].clone() if has else None)
    before = pool.clone()
    o = run_batch(gdn(shape, order), x, g, pool, lens, init, slots)
    assert o.shape == x["v"].shape and o.dtype == torch.bfloat16
    written = {u for u, n in zip(slots, lens, strict=True) if n}
    untouched = [u for u in range(UNITS) if u not in written]
    assert torch.equal(pool[untouched].view(torch.int32), before[untouched].view(torch.int32))
    worst = judge_rows(o, pool, x, g, h0s, lens, slots, order, f"{shape}/{order}/{batch}")
    print(f"gated_delta_rule {shape} {order} {batch}: worst error / HF bf16 error = {worst:.3f}")


@pytest.mark.parametrize("order", ["grouped", "tiled"])
def test_decode_continues_prefill_step_by_step(order):
    """Prefill rows (chunked path), then 8 decode steps (recurrent path), each row's outputs over all steps and its
    final state within RATIO × HF's (chunked prefill, then its recurrence token by token)."""
    hk, hv, d = SHAPES["qwen3.8-27b"]
    prefill, steps, slots = [63, 200, 1, 64], 8, [0, UNITS - 1, 6, 2]
    g, kernel = gates(hv), gdn("qwen3.8-27b", order)
    streams = [tokens(n + steps, hk, hv, d) for n in prefill]
    pool = torch.full((UNITS, hv, d, d), float("nan"), device="cuda")
    batch = {n: torch.cat([s[n][:p] for s, p in zip(streams, prefill, strict=True)]) for n in streams[0]}
    outs = [[o] for o in run_batch(kernel, batch, g, pool, prefill, [False] * 4, slots).split(prefill)]
    for j in range(steps):
        step = {n: torch.cat([s[n][p + j:p + j + 1] for s, p in zip(streams, prefill, strict=True)]) for n in batch}
        for out, o in zip(outs, run_batch(kernel, step, g, pool, [1] * 4, [True] * 4, slots).split(1), strict=True):
            out.append(o)
    worst = 0.0
    for r, (s, p) in enumerate(zip(streams, prefill, strict=True)):
        want_o, want_s = exact(s, g, None, order)
        hf_o, hf_s = hf_bf16(rows(s, 0, p), g, None, order)
        hf_outs = [hf_o]
        for j in range(steps):
            o, hf_s = hf_bf16(rows(s, p + j, p + j + 1), g, hf_s, order)
            hf_outs.append(o)
        for got, want, floor in ((torch.cat(outs[r]), want_o, torch.cat(hf_outs)), (pool[slots[r]], want_s, hf_s)):
            x_ratio = ratio(got, want, floor)
            assert x_ratio <= RATIO, (r, fro(got, want), fro(floor, want))
            worst = max(worst, x_ratio)
    print(f"gated_delta_rule prefill + decode {order}: worst error / HF bf16 error = {worst:.3f}")


@pytest.mark.parametrize("order", ["grouped", "tiled"])
def test_delta_rule_aligned_split_and_repeat_are_bitwise(order):
    hk, hv, d = SHAPES["qwen3.8-27b"]
    x, g, kernel = tokens(200, hk, hv, d), gates(hv), gdn("qwen3.8-27b", order)
    one = torch.full((2, hv, d, d), float("nan"), device="cuda")
    split = one.clone()
    o_one = run_batch(kernel, x, g, one, [200], [False], [1])
    o_a = run_batch(kernel, rows(x, 0, 128), g, split, [128], [False], [1])
    o_b = run_batch(kernel, rows(x, 128, 200), g, split, [72], [True], [1])
    assert torch.equal(torch.cat([o_a, o_b]), o_one) and torch.equal(split[1], one[1])
    for lens, init in (([200], [False]), ([1], [True])):          # chunked and recurrent paths, twice each
        x1 = rows(x, 0, lens[0])
        pools = [torch.zeros(2, hv, d, d, device="cuda") + 0.01 for _ in range(2)]
        outs = [run_batch(kernel, x1, g, p, lens, init, [0]) for p in pools]
        assert torch.equal(outs[0], outs[1]) and torch.equal(pools[0], pools[1])
    # the recurrent path reads column views of wider rows (the projections' and conv's) in place: the same bits
    x4 = rows(x, 0, 4)
    wide = {n: torch.cat([v.reshape(4, -1), torch.full_like(v.reshape(4, -1), float("nan"))], 1) for n, v in x4.items()}
    views = {n: wide[n][:, :x4[n][0].numel()].view(x4[n].shape) for n in x4}
    pools = [torch.zeros(4, hv, d, d, device="cuda") + 0.01 for _ in range(2)]
    outs = [run_batch(kernel, y, g, p, [1] * 4, [True] * 4, [0, 1, 2, 3])
            for y, p in ((x4, pools[0]), (views, pools[1]))]
    assert not views["q"].is_contiguous() and torch.equal(outs[0], outs[1]) and torch.equal(pools[0], pools[1])
    # … and per-token rows of several tokens (a verify pass's, its rewind's replays) step their views token by token
    for lens in ([4], [2, 3], [1, 4, 2]):
        t, xs = sum(lens), rows(x, 0, sum(lens))
        wide = {n: torch.cat([v.reshape(t, -1), torch.full_like(v.reshape(t, -1), float("nan"))], 1)
                for n, v in xs.items()}
        views = {n: wide[n][:, :xs[n][0].numel()].view(xs[n].shape) for n in xs}
        pools = [torch.zeros(len(lens), hv, d, d, device="cuda") + 0.01 for _ in range(2)]
        outs = [kernel(kernel.plan(lens, per_token=True), y["q"], y["k"], y["v"], y["a"], y["b"], g["a_log"],
                       g["dt_bias"], p, dev(list(range(len(lens)))), dev([True] * len(lens), torch.bool))
                for y, p in ((xs, pools[0]), (views, pools[1]))]
        assert torch.equal(outs[0], outs[1]) and torch.equal(pools[0], pools[1]), lens


def test_delta_rule_never_syncs():
    hk, hv, d = SHAPES["small"]
    kernel, g = gdn("small", "tiled"), gates(hv)
    pool = torch.zeros(UNITS, hv, d, d, device="cuda")
    for lens, init, slots in ((LENS, INIT, SLOTS), ([1] * 6, DECODE_INIT, DECODE_SLOTS)):
        x, s, h = tokens(sum(lens), hk, hv, d), dev(slots), dev(init, torch.bool)
        kernel(kernel.plan(lens), x["q"], x["k"], x["v"], x["a"], x["b"], g["a_log"], g["dt_bias"], pool, s, h)
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            kernel(kernel.plan(lens), x["q"], x["k"], x["v"], x["a"], x["b"], g["a_log"], g["dt_bias"], pool, s, h)
        finally:
            torch.cuda.set_sync_debug_mode("default")


def test_plant_has_init_ignored_reads_stale_state():
    hk, hv, d = SHAPES["qwen3.5"]
    g = gates(hv)
    for lens, init, slots in ((LENS, INIT, SLOTS), ([1] * 6, DECODE_INIT, DECODE_SLOTS)):
        x = tokens(sum(lens), hk, hv, d)
        pool = torch.full((UNITS, hv, d, d), float("nan"), device="cuda")      # stale units of rows without init
        pool[[u for u, has in zip(slots, init, strict=True) if has]] = 0.0
        o = run_batch(gdn("qwen3.5", "grouped"), x, g, pool, lens, [True] * len(lens), slots)
        assert not torch.isfinite(o).all()


def test_plant_slot_zero_skipped_by_the_old_sentinel(tmp_path, monkeypatch):
    """fla's own sentinel (`<= 0`) rebuilt from the vendored source: the row on slot 0 is skipped and L0 fails."""
    src = inspect.getsource(fused_sigmoid_gating)
    old = src.replace("if state_idx < 0:", "if state_idx <= 0:").replace("if final_state_idx >= 0:",
                                                                         "if final_state_idx > 0:")
    assert old.count("<= 0") == src.count("<= 0") + 1 and old != src
    path = tmp_path / "fused_sigmoid_gating_old_sentinel.py"
    path.write_text(old)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    hk, hv, d = SHAPES["qwen3.5"]
    x, g = tokens(len(DECODE_SLOTS), hk, hv, d), gates(hv)
    pool, warm = warm_states(x, g, "grouped", UNITS)
    pool[DECODE_SLOTS] = warm
    h0 = pool[0].clone()
    kernel = gdn("qwen3.5", "grouped")
    good = run_batch(kernel, x, g, pool.clone(), [1] * 6, [True] * 6, DECODE_SLOTS)
    monkeypatch.setattr(fla_plugin, "fused_sigmoid_gating_delta_rule_update",
                        module.fused_sigmoid_gating_delta_rule_update)
    planted = pool.clone()
    bad = run_batch(kernel, x, g, planted, [1] * 6, [True] * 6, DECODE_SLOTS)
    want_o, want_s = exact(rows(x, 0, 1), g, h0, "grouped")
    hf_o, hf_s = hf_bf16(rows(x, 0, 1), g, h0, "grouped")
    assert fro(good[:1], want_o) <= RATIO * fro(hf_o, want_o)
    assert torch.equal(planted[0], h0)                          # slot 0's state was never written
    assert not fro(planted[0], want_s) <= RATIO * fro(hf_s, want_s)
    assert torch.equal(bad[1:], good[1:])                       # the other rows are unaffected


# --- causal_conv1d -------------------------------------------------------------------------------------------------

def conv_case(c: int, lens, init, slots, dtype=torch.bfloat16):
    x = torch.randn(sum(lens), c, device="cuda").to(dtype)
    w = (torch.randn(c, KERNEL, device="cuda") * 0.5).to(dtype)
    state = torch.full((UNITS, c, KERNEL - 1), float("nan"), device="cuda").to(dtype)
    for r, (unit, has) in enumerate(zip(slots, init, strict=True)):
        if has or lens[r] == 0:
            state[unit] = torch.randn(c, KERNEL - 1, device="cuda").to(dtype)
    return x, w, state


def conv_oracle(x, w, state, slots, lens, init):
    """The oracle in f32 (unrounded output) and the state it leaves, rounded back to the state dtype."""
    s = state.float()
    y = ref.causal_conv1d(x.float(), w, s, dev(slots), query_start(lens), dev(init, torch.bool))
    want = state.clone()
    used = [u for u, n in zip(slots, lens, strict=True) if n]
    want[used] = s[used].to(state.dtype)
    return y, want


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("batch", ["mixed", "decode"])
def test_conv1d_matches_the_oracle(shape, batch, dtype):
    c = channels(*SHAPES[shape])
    lens, init, slots = (LENS, INIT, SLOTS) if batch == "mixed" else ([1] * 6, DECODE_INIT, DECODE_SLOTS)
    x, w, state = conv_case(c, lens, init, slots, dtype)
    y_want, s_want = conv_oracle(x, w, state, slots, lens, init)
    y = conv(c, dtype)(x, w, state, dev(slots), query_start(lens), dev(init, torch.bool))
    r = l0.check(y, y_want, dtype, KERNEL)
    assert r.passed, r
    assert torch.equal(state.view(torch.int16), s_want.view(torch.int16))      # NaN units bitwise untouched too
    print(f"causal_conv1d {shape} {batch} {dtype}: usage {r.usage:.3f}")


def test_conv1d_aligned_split_is_bitwise_and_the_carry_plant_fails():
    c = channels(*SHAPES["qwen3.8-27b"])
    x, w, _ = conv_case(c, [200], [False], [1])
    kernel = conv(c)
    one = torch.full((2, c, KERNEL - 1), float("nan"), device="cuda").bfloat16()
    split, dropped = one.clone(), one.clone()
    y_one = kernel(x, w, one, dev([1]), query_start([200]), dev([False], torch.bool))
    y_a = kernel(x[:128], w, split, dev([1]), query_start([128]), dev([False], torch.bool))
    y_b = kernel(x[128:], w, split, dev([1]), query_start([72]), dev([True], torch.bool))
    assert torch.equal(torch.cat([y_a, y_b]), y_one) and torch.equal(split[1], one[1])
    y_want, _ = conv_oracle(x, w, torch.zeros_like(one), [1], [200], [False])
    kernel(x[:128], w, dropped, dev([1]), query_start([128]), dev([False], torch.bool))
    dropped[1] = 0                                              # the plant: the carry dropped between chunks
    y_c = kernel(x[128:], w, dropped, dev([1]), query_start([72]), dev([True], torch.bool))
    assert l0.check(y_b, y_want[128:], torch.bfloat16, KERNEL).passed
    assert not l0.check(y_c, y_want[128:], torch.bfloat16, KERNEL).passed


def test_conv1d_plant_has_init_ignored():
    c = channels(*SHAPES["qwen3.5"])
    x, w, state = conv_case(c, LENS, INIT, SLOTS)
    y_want, _ = conv_oracle(x, w, state, SLOTS, LENS, INIT)
    y = conv(c)(x, w, state, dev(SLOTS), query_start(LENS), dev([True] * len(LENS), torch.bool))
    assert not l0.check(y, y_want, torch.bfloat16, KERNEL).passed


@pytest.mark.parametrize("kernel", [2, 4, 8])
def test_conv1d_many_rows_and_kernel_sizes(kernel):
    """150 rows (several passes of the row search), zero-token rows spread through, every supported K's extremes."""
    c, n = 640, 150
    lens = [(0, 1, 1, 3, 9)[r % 5] for r in range(n)]
    init, slots = [r % 3 != 0 for r in range(n)], list(range(n))
    x = torch.randn(sum(lens), c, device="cuda").bfloat16()
    w = (torch.randn(c, kernel, device="cuda") * 0.5).bfloat16()
    state = torch.full((n + 1, c, kernel - 1), float("nan"), device="cuda").bfloat16()
    for r in range(n):
        if init[r] or lens[r] == 0:
            state[r] = torch.randn(c, kernel - 1, device="cuda").bfloat16()
    y_want, s_want = conv_oracle(x, w, state, slots, lens, init)
    y = select("causal_conv1d@1", channels=c, kernel_size=kernel, dtype="bfloat16")(
        x, w, state, dev(slots), query_start(lens), dev(init, torch.bool))
    assert l0.check(y, y_want, torch.bfloat16, kernel).passed
    assert torch.equal(state.view(torch.int16), s_want.view(torch.int16))


# --- gated_rmsnorm -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(("t", "h", "d", "gate"), [(0, 48, 128, "silu"), (37, 48, 128, "silu"), (300, 32, 128, "silu"),
                                                  (5, 6, 64, "silu"), (37, 48, 128, "sigmoid")])  # FN's GDN (D8)
def test_gated_rmsnorm_matches_the_oracle(t, h, d, gate, dtype):
    facts = {"dim": d, "eps": EPS, "dtype": str(dtype).removeprefix("torch.")} | (
        {"out_gate": gate} if gate != "silu" else {})
    norm = select("gated_rmsnorm@1", **facts)
    o = (torch.randn(t, h, d, device="cuda") * 3).to(dtype)
    zw = torch.randn(t, h + 1, d, device="cuda").to(dtype)                    # z as a strided view
    z, w = zw[:, 1:], (1 + 0.2 * torch.randn(d, device="cuda")).to(dtype)
    for weight in (w, w.float()):                               # an f32 weight: the product stays in f32
        got, want = norm(o, z, weight), ref.gated_rmsnorm(o, z, weight, EPS, gate)
        r = l0.check(got, want, dtype, d)
        assert r.passed, r
        print(f"gated_rmsnorm {t}x{h}x{d} {dtype} weight {weight.dtype}: usage {r.usage:.3f}")


# --- a decode step in a CUDA graph -----------------------------------------------------------------------------------

@pytest.mark.parametrize("order", ["grouped", "tiled"])
def test_decode_step_graph_replay_equals_eager(order):
    hk, hv, d = SHAPES["qwen3.8-27b"]
    c, n = channels(hk, hv, d), len(DECODE_SLOTS)
    rule, conv_k = gdn("qwen3.8-27b", order), conv(c)
    norm = select("gated_rmsnorm@1", dim=d, eps=EPS, dtype="bfloat16")
    g, w_conv = gates(hv), (torch.randn(c, KERNEL, device="cuda") * 0.5).bfloat16()
    w_norm = (1 + 0.2 * torch.randn(d, device="cuda")).bfloat16()
    slots, init, qs = dev(DECODE_SLOTS), dev(DECODE_INIT, torch.bool), query_start([1] * n)
    handle = rule.plan([1] * n)
    inputs = dict(x=torch.randn(n, c, device="cuda").bfloat16(), z=torch.randn(n, hv, d, device="cuda").bfloat16(),
                  **{k: v for k, v in tokens(n, hk, hv, d).items() if k in ("a", "b")})
    pool = torch.randn(UNITS, hv, d, d, device="cuda") * 0.05
    conv_state = torch.randn(UNITS, c, KERNEL - 1, device="cuda").bfloat16()

    def step(p, cs):
        y = conv_k(inputs["x"], w_conv, cs, slots, qs, init)
        q, k, v = y.split([hk * d, hk * d, hv * d], 1)
        o = rule(handle, q.reshape(n, hk, d), k.reshape(n, hk, d), v.reshape(n, hv, d), inputs["a"], inputs["b"],
                 g["a_log"], g["dt_bias"], p, slots, init)
        return norm(o, inputs["z"], w_norm)

    p0, c0 = pool.clone(), conv_state.clone()
    step(pool.clone(), conv_state.clone())                      # compile outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = step(pool, conv_state)
    for name, value in tokens(n, hk, hv, d).items():            # new inputs, the captured states restored
        if name in inputs:
            inputs[name].copy_(value)
    inputs["x"].copy_(torch.randn(n, c, device="cuda").bfloat16())
    inputs["z"].copy_(torch.randn(n, hv, d, device="cuda").bfloat16())
    slots.copy_(slots.flip(0))                                  # rows on other units, flags flipped
    init.copy_(~init)
    pool.copy_(p0)
    conv_state.copy_(c0)
    graph.replay()
    p_eager, c_eager = p0.clone(), c0.clone()
    eager = step(p_eager, c_eager)
    assert torch.equal(out, eager) and torch.equal(pool, p_eager) and torch.equal(conv_state, c_eager)


# --- in-step snapshots (docs/m03-in-step-snapshot.md §3.1) ---------------------------------------------------------

def _call(kernel, x, g, state, lens, init, slots, snapshots=(), view=None):
    h = kernel.plan(lens, snapshots)
    return kernel(h, x["q"], x["k"], x["v"], x["a"], x["b"], g["a_log"], g["dt_bias"], state, dev(slots),
                  dev(init, torch.bool), snapshot=view)


@pytest.mark.parametrize("order", ["grouped", "tiled"])
def test_a_snapshot_is_the_final_state_of_a_split_there(order):
    """Rows of 200 (snapshot after 64 = chunk 1), 0, 150 (after 128: the last full chunk, a partial tail) and 90 (none,
    with an initial state): each snapshot is bitwise the final state of a call over the row's first tokens; outputs
    and final states are bitwise those of the call without snapshots; untouched units stay NaN."""
    hk, hv, d = SHAPES["qwen3.8-27b"]
    lens, init, slots = [200, 0, 150, 90], [False, False, True, True], [0, 5, 2, 7]
    kernel, x, g = gdn("qwen3.8-27b", order), tokens(sum(lens), hk, hv, d), gates(hv)
    pool, warm = warm_states(x, g, order, UNITS)
    for unit in (2, 7):
        pool[unit] = warm
    payload = torch.full((4, hv, d, d), float("nan"), device="cuda")
    snapped, plain = pool.clone(), pool.clone()
    o1 = _call(kernel, x, g, snapped, lens, init, slots, [(0, 64, 1), (2, 128, 3)], payload)
    o0 = _call(kernel, x, g, plain, lens, init, slots)
    assert torch.equal(o0, o1) and torch.equal(snapped.nan_to_num(), plain.nan_to_num())
    assert payload[0].isnan().all() and payload[2].isnan().all()
    qs = query_start(lens).tolist()
    for row, offset, unit in [(0, 64, 1), (2, 128, 3)]:
        split = pool.clone()
        lo = qs[row]
        _call(kernel, rows(x, lo, lo + offset), g, split, [offset], [init[row]], [slots[row]])
        assert torch.equal(payload[unit], split[slots[row]]), (row, offset)
        late = pool.clone()                                    # plant: one chunk late is another state
        _call(kernel, rows(x, lo, lo + offset + 64), g, late, [offset + 64], [init[row]], [slots[row]])
        assert not torch.equal(payload[unit], late[slots[row]])
        assert not torch.equal(payload[unit], split[slots[row]].bfloat16().float())   # plant: fla's bf16 `h`


@pytest.mark.parametrize("lens, snaps, subs", [
    ([1100], [], 5), ([1100], [(0, 384, 1)], 5), ([1100], [(0, 512, 1)], 5), ([1100], [(0, 640, 1)], 5),
    ([700, 0, 450, 90], [(0, 512, 0), (2, 192, 1)], 3 + 2 + 1),       # two prompts' chunks and a short row in one step
])
def test_a_step_in_sub_chunks_is_one_call(lens, snaps, subs, monkeypatch):
    """A step past SUB_TOKENS runs row by row, each row in sub-chunks from the state the one before left (FN §23.3):
    its outputs, final states and snapshots — inside a sub-chunk (384, 640, 192) or at one's start (512) — are
    bitwise one call's, for one row or several (SUB_TOKENS 256 here)."""
    from holon_plugins.kernels.fla import plugin as fla
    hk, hv, d = SHAPES["qwen3.8-27b"]
    n = len(lens)
    init, slots = [r % 2 == 0 for r in range(n)], [2 + r for r in range(n)]
    kernel, x, g = gdn("qwen3.8-27b", "grouped"), tokens(sum(lens), hk, hv, d), gates(hv)
    pool, warm = warm_states(x, g, "grouped", UNITS)
    for s in slots:
        pool[s] = warm
    runs = []
    for sub in (1 << 20, 256):
        monkeypatch.setattr(fla, "SUB_TOKENS", sub)
        state, payload = pool.clone(), torch.full((2, hv, d, d), float("nan"), device="cuda")
        runs.append((_call(kernel, x, g, state, lens, init, slots, snaps, payload), state, payload))
    assert len(kernel.plan(lens, snaps).subs) == subs
    (o0, s0, p0), (o1, s1, p1) = runs
    assert torch.equal(o0, o1) and torch.equal(s0.nan_to_num(), s1.nan_to_num())       # (untouched units: NaN)
    assert torch.equal(p0.nan_to_num(), p1.nan_to_num())
    assert all(not p1[unit].isnan().any() for _, _, unit in snaps)


def test_a_step_of_two_rows_peaks_no_higher_than_one_row():
    """The startup profile measures a one-row prompt step (its rows of at most max_model_len): a step of the same
    tokens in two rows must not need more of the chunked rule's temporaries, or a deployment serving two prompts at
    once runs past its reserve (the RTX 5090's exact.py run did, at 64 sequences)."""
    hk, hv, d = SHAPES["qwen3.8-27b"]
    kernel, x, g = gdn("qwen3.8-27b", "grouped"), tokens(4096, hk, hv, d), gates(hv)
    pool, _ = warm_states(x, g, "grouped", UNITS)
    peaks = []
    for lens in ([4096], [2048, 2048], [3000, 1096]):
        state = pool.clone()
        h = kernel.plan(lens)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        kernel(h, x["q"], x["k"], x["v"], x["a"], x["b"], g["a_log"], g["dt_bias"], state,
               dev(list(range(len(lens)))), dev([False] * len(lens), torch.bool))
        torch.cuda.synchronize()
        peaks.append(torch.cuda.max_memory_allocated() - base)
    assert max(peaks[1:]) <= peaks[0], peaks


def test_snapshots_must_fall_inside_their_row_on_a_chunk():
    kernel = gdn("small", "grouped")
    for bad in [(0, 0, 0), (0, 32, 0), (0, 192, 0), (1, 64, 0)]:   # chunk 0, off a chunk, past the end, a 0-token row
        with pytest.raises(ValueError, match="snapshot"):
            kernel.plan([150, 0], [bad])


# --- speculative verify rows (docs/m04-spec.md D6): per-token rows, out_slots, masked replay ---------------------
# Bitwise: the masked replay equals one call over the accepted tokens alone (the same kernel specialization). Against
# one-token decode calls (a specialization storing inside its loop) the states agree within STATE_ULPS·eps(f32) of
# their scale — the design's fallback (§1.3) — and a long run of verify/replay steps stays there without drifting.
STATE_ULPS = 8


def _verify_replay(rule, x, g, s0, slots, accepted, q, sink=None):
    """A verify step over rows of q tokens (written to `sink`, else to the rows' units) and its masked replay."""
    n = len(slots)
    init = dev([True] * n, torch.bool)
    out = None
    if sink is not None:
        out = rule(rule.plan([q] * n, per_token=True), *rows(x, 0, n * q).values(), g["a_log"], g["dt_bias"], s0,
                   dev(slots), init, out_slots=dev([sink] * n))
    keep = torch.arange(q, device="cuda")[None, :] < 1 + torch.tensor(accepted, device="cuda")[:, None]
    masked = dict(rows(x, 0, n * q))
    for name in ("a", "b"):
        masked[name] = masked[name].masked_fill(~keep.reshape(-1, 1), -1e4)
    rule(rule.plan([q] * n, per_token=True), *masked.values(), g["a_log"], g["dt_bias"], s0, dev(slots), init,
         out_slots=dev(slots))
    return out


def _decode_steps(rule, x, g, state, slots, accepted, q) -> dict:
    """1 + n one-token calls per row; their outputs by (row, token)."""
    outs = {}
    for j in range(1 + max(accepted)):
        live = [r for r, n in enumerate(accepted) if j <= n]
        at = torch.tensor([r * q + j for r in live], device="cuda")
        o = rule(rule.plan([1] * len(live)), *{n: t[at] for n, t in rows(x, 0, len(slots) * q).items()}.values(),
                 g["a_log"], g["dt_bias"], state, dev([slots[r] for r in live]), dev([True] * len(live), torch.bool))
        outs |= {(r, j): o[i] for i, r in enumerate(live)}
    return outs


def _close(a: torch.Tensor, b: torch.Tensor, ulps: int = STATE_ULPS) -> bool:
    return bool((a - b).abs().max() <= ulps * torch.finfo(torch.float32).eps * b.abs().max())


@pytest.mark.parametrize("order", ["grouped", "tiled"])
@pytest.mark.parametrize("shape", ["qwen3.8-27b", "qwen3.5"])
def test_verify_rows_replay_bitwise_to_the_accepted_tokens(shape, order):
    hk, hv, d = SHAPES[shape]
    rule = gdn(shape, order)
    q, slots, accepted, sink = 4, [3, 0, UNITS - 1], [0, 2, 3], 9
    x, g = tokens(len(slots) * q, hk, hv, d), gates(hv)
    s0 = torch.randn(UNITS, hv, d, d, device="cuda") * 0.1

    state = s0.clone()
    keep_s0 = state.clone()
    out = rule(rule.plan([q] * len(slots), per_token=True), *rows(x, 0, len(slots) * q).values(), g["a_log"],
               g["dt_bias"], state, dev(slots), dev([True] * len(slots), torch.bool),
               out_slots=dev([sink] * len(slots)))
    others = [u for u in range(UNITS) if u != sink]
    assert torch.equal(state[others], keep_s0[others])               # the verify step moved nothing but the sink

    replay = s0.clone()
    _verify_replay(rule, x, g, replay, slots, accepted, q)
    alone = s0.clone()
    take = torch.cat([torch.arange(r * q, r * q + 1 + n, device="cuda") for r, n in enumerate(accepted)])
    rule(rule.plan([1 + n for n in accepted], per_token=True),
         *{n: t[take] for n, t in rows(x, 0, len(slots) * q).items()}.values(), g["a_log"], g["dt_bias"], alone,
         dev(slots), dev([True] * len(slots), torch.bool), out_slots=dev(slots))
    assert torch.equal(alone, replay)                                # bitwise: every unit

    decode = s0.clone()
    outs = _decode_steps(rule, x, g, decode, slots, accepted, q)
    untouched = [u for u in range(UNITS) if u not in slots]
    assert torch.equal(replay[untouched], s0[untouched]) and torch.equal(decode[untouched], s0[untouched])
    assert _close(replay[slots], decode[slots])
    for (r, j), o in outs.items():                                   # verify outputs: decode's within a bf16 step
        assert torch.allclose(out[r * q + j].float(), o.float(), rtol=2 ** -7, atol=1e-3)


@pytest.mark.parametrize("shape", ["qwen3.8-27b"])
def test_verify_replay_does_not_drift_from_decoding(shape):
    # 60 verify/replay rounds (up to 240 tokens per row, random acceptance) against the same tokens decoded one at a
    # time: the states stay within STATE_ULPS of each other — the specializations' rounding does not accumulate.
    hk, hv, d = SHAPES[shape]
    rule = gdn(shape, "tiled")
    q, slots = 4, [2, 5]
    g = gates(hv)
    spec = torch.randn(UNITS, hv, d, d, device="cuda") * 0.1
    plain = spec.clone()
    gen = torch.Generator().manual_seed(7)
    for _ in range(60):
        accepted = torch.randint(0, q, (len(slots),), generator=gen).tolist()
        x = tokens(len(slots) * q, hk, hv, d)
        _verify_replay(rule, x, g, spec, slots, accepted, q)
        _decode_steps(rule, x, g, plain, slots, accepted, q)
    assert _close(spec[slots], plain[slots])


@pytest.mark.parametrize("width", [4, 8])
@pytest.mark.parametrize("shape", ["qwen3.8-27b", "small"])
def test_rewind_glue_is_its_oracle(shape, width):
    # Rows on slot 0, the last unit and a middle one (a fourth row buffer unused), accepting 0, width − 1 and some
    # drafts; the oracle leaves every other unit, row and gate untouched, so whole-tensor equality covers them.
    hk, hv, d = SHAPES[shape]
    c, layers, slots = channels(hk, hv, d), 3, [0, UNITS - 1, 6]
    glue = select("gdn_rewind@1", channels=c, kernel_size=KERNEL, heads_v=hv, dtype="bfloat16")
    accepted = torch.zeros(UNITS, dtype=torch.long, device="cuda")
    accepted[slots] = dev([0, width - 1, width // 2], torch.long)
    bufs = [torch.randn(layers, UNITS, c, KERNEL - 1, device="cuda").bfloat16(),
            torch.randn(layers, 4, c, KERNEL - 1, device="cuda").bfloat16(),
            torch.randn(layers, 4 * width, c, device="cuda").bfloat16(),
            torch.randn(layers, 4 * width, hv, device="cuda").bfloat16(),
            torch.randn(layers, 4 * width, hv, device="cuda").bfloat16()]
    got, want = [t.clone() for t in bufs], [t.clone() for t in bufs]
    glue(*got, dev(slots, torch.long), accepted, width)
    ref.gdn_rewind(*want, dev(slots, torch.long), accepted, width)
    assert all(torch.equal(g, w) for g, w in zip(got, want, strict=True))
    assert not torch.equal(got[0], bufs[0]) and not torch.equal(got[3], bufs[3])
