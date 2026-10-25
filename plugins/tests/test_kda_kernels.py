"""L0 for `kernels.fla`'s `kda@1` (GLM-5.3-Flash's Kimi delta attention, docs/v1-glm.md D4), selected through the
build context, at GLM's shape (64 heads × 128) and the tiny GPU fixture's (2 × 128) and a small one (2 × 64).

- The yardstick (`test_gdn_kernels.py`'s form): per row, the relative Frobenius error of the output and of the final
  state against an f64 recurrence written out here, within RATIOS × that of a bf16 torch path on the same inputs —
  HF's KDA recurrence (transformers 5.19.0 `recurrent_kimi_delta_attention`, f32 throughout) with q and k
  l2-normalised and β taken in bf16, as the GDN yardstick's HF path does; the kernel's chunks round their operands
  (k, β, the WY factors, h) to bf16 alike. Mixed batches (lengths around the 64-token chunk, rows with and without
  has_init, slot 0 and the last slot, a zero-token row), decode batches, decode continuing a prefill, and the gate at
  its bound (−5 on every channel of every token: the in-chunk decay exponents' worst case, C-m4).
- Exactness: a split at 64-token boundaries (KDAPlanner's sub-batches) bitwise one call, with the carried f32 state;
  a step in SUB_TOKENS sub-calls bitwise one call; an in-step snapshot bitwise the final state of a split there; runs
  repeat bitwise; per-token rows give their one-token steps' outputs bitwise; a decode step replayed from a CUDA
  graph bitwise eager; plan and call never sync.
- The gate at its bound: the in-chunk decay exponents stay ≤ 0 where they are used (FLA's sub-chunk form masks the
  rest), so it is finite and within budget; a 64-row sub-chunk probe (C-m4) measured the same.
- Units of rows without has_init are never read (NaN-poisoned), units no row writes stay bitwise untouched.
- The slot-0 case (D4): one sequence decoding at slot 0 against the oracle; it fails with fla's original `<= 0`
  sentinel, rebuilt from the vendored source.
- `gdn_rewind@1` at KDA's gate width (H·Dk) is bitwise its oracle; tokens it masks (f = b = −inf) leave a row's state
  bitwise unchanged on the recurrent and the chunked paths.

Run with `-s` to see the worst error ratios. Skips without a GPU.
"""
import importlib.util
import inspect
import math

import pytest
import torch
from holon.core.build import BuildContext
from holon.core.plugins import PluginSet, discover
from holon_plugins.kernels.fla import plugin as fla_plugin
from holon_plugins.kernels.fla.vendor import fused_sigmoid_gating
from holon_std.mixers.gdn import reference as ref
from holon_std.mixers.gdn.kernels import MASKED

pytestmark = [pytest.mark.l0, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")]

SHAPES = {"glm-5.3-flash": (64, 128), "tiny-gpu": (2, 128), "small": (2, 64)}         # H, D
LB, UNITS, KERNEL = -5.0, 16, 4
LENS = [1, 2, 63, 64, 0, 65, 200, 1000]
INIT = [True, False, True, False, True, True, False, True]
SLOTS = [0, UNITS - 1, 3, 7, 9, 1, 12, 5]                      # row 4 has no tokens: unit 9 stays as it is
DECODE_INIT = [True, False, True, True, False, True]
DECODE_SLOTS = [0, UNITS - 1, 4, 2, 11, 6]
# × the bf16 yardstick's error, by head count (docs/v1-glm.md D4's table, 16 seeds of the mixed batch): GLM's 64 heads
# hold GDN's 2.0 (worst 1.33); the 2-head shapes need 4.0 (worst 2.86, a one-token row's output on the chunked path,
# e.g. a prompt step's 1-token tail). The cause: FLA's two per-head bf16 casts (`kda.py:928` q·scale, `:961` Aqk),
# coherent over a head's outputs, which two heads cannot average, plus small-sample tails; a wrong gate (the plants
# below) or state is ≥ 10×.
RATIOS = {64: 2.0, 2: 4.0}


def budget(x: dict) -> float:
    return RATIOS[x["q"].shape[1]]
FLOOR = 2e-2                                                   # the yardstick's error itself stays below this
STATE_ULPS = 8


@pytest.fixture(autouse=True)
def _seeded():
    torch.manual_seed(0)


def kda(shape: str):
    h, d = SHAPES[shape]
    facts = dict(heads=h, head_k_dim=d, head_v_dim=d, lower_bound=LB, dtype="bfloat16")
    ctx = BuildContext(PluginSet(discover()), torch.device("cuda"))
    spec = ctx.select_kernel("kda@1", site="test", **facts)
    assert spec.provider == "kernels.fla"
    return spec.impl(ctx, **facts)


def dev(values, dtype=torch.int32) -> torch.Tensor:
    return torch.tensor(values, dtype=dtype, device="cuda")


def query_start(lens) -> torch.Tensor:
    return dev([0, *torch.tensor(lens).cumsum(0).tolist()])


def tokens(t: int, h: int, d: int, bound: bool = False) -> dict:
    """bf16 q/k/v/f/b; f spread so the gate covers (−5, 0) — or at the bound: σ(·) = 1, g = −5 everywhere."""
    f = torch.full((t, h * d), 1e4, device="cuda") if bound else torch.randn(t, h * d, device="cuda") * 2
    return dict(q=torch.randn(t, h, d, device="cuda").bfloat16(), k=torch.randn(t, h, d, device="cuda").bfloat16(),
                v=torch.randn(t, h, d, device="cuda").bfloat16(), f=f.bfloat16(),
                b=torch.randn(t, h, device="cuda").bfloat16())


def gates(h: int, d: int) -> dict:
    return dict(a_log=torch.linspace(-3.0, 1.0, h, device="cuda"), dt_bias=torch.randn(h * d, device="cuda") * 0.5)


def rows(x: dict, lo: int, hi: int) -> dict:
    return {n: x[n][lo:hi] for n in ("q", "k", "v", "f", "b")}


def _recurrence(x: dict, g: dict, h0, dtype, l2_dtype):
    h, d = x["q"].shape[1], x["q"].shape[2]

    def l2(y):
        y = y.to(l2_dtype)
        return y / torch.sqrt((y * y).sum(-1, keepdim=True) + ref.L2_EPS)

    q, k = l2(x["q"]).to(dtype) * d ** -0.5, l2(x["k"]).to(dtype)
    gate = LB * torch.sigmoid(g["a_log"].to(dtype).exp()[:, None] * (x["f"].to(dtype).view(-1, h, d)
                                                                     + g["dt_bias"].to(dtype).view(h, d)))
    beta, v = torch.sigmoid(x["b"].to(l2_dtype)).to(dtype), x["v"].to(dtype)
    s = torch.zeros(h, d, d, dtype=dtype, device="cuda") if h0 is None else h0.to(dtype).clone()
    out = []
    for t in range(len(v)):
        s = s * gate[t].exp()[:, None, :]
        s = s + (beta[t][:, None] * (v[t] - (s * k[t][:, None, :]).sum(-1)))[:, :, None] * k[t][:, None, :]
        out.append((s * q[t][:, None, :]).sum(-1))
    return (torch.stack(out) if out else v.new_zeros(0, h, d)), s


def exact(x: dict, g: dict, h0) -> tuple[torch.Tensor, torch.Tensor]:
    """The contract in f64, token by token: (o [T, H, D], final state [H, Dv, Dk])."""
    return _recurrence(x, g, h0, torch.float64, torch.float64)


def bf16_path(x: dict, g: dict, h0) -> tuple[torch.Tensor, torch.Tensor]:
    """The yardstick: HF's f32 KDA recurrence with q, k l2-normalised and β in bf16, the output rounded to bf16."""
    o, s = _recurrence(x, g, h0, torch.float32, torch.bfloat16)
    return o.bfloat16(), s


def fro(got: torch.Tensor, want: torch.Tensor) -> float:
    return ((got.double() - want).norm() / want.norm()).item()


def ratio(got: torch.Tensor, want: torch.Tensor, floor: torch.Tensor) -> float:
    assert fro(floor, want) < FLOOR, fro(floor, want)
    return fro(got, want) / fro(floor, want)


def warm(h: int, d: int, g: dict) -> torch.Tensor:
    """A realistic state: 64 tokens of the f64 recurrence."""
    return exact(tokens(64, h, d), g, None)[1].float()


def run(kernel, x, g, pool, lens, init, slots, snapshots=(), view=None, per_token=False):
    return kernel(kernel.plan(lens, snapshots, per_token=per_token), x["q"], x["k"], x["v"], x["f"], x["b"],
                  g["a_log"], g["dt_bias"], pool, dev(slots), dev(init, torch.bool), snapshot=view)


def judge_rows(o, state, x, g, h0s, lens, slots, what: str, limit: float | None = None) -> float:
    """Per row: output and state errors vs f64 over the bf16 yardstick's, each within `limit` (default the shape's
    budget); the worst ratio."""
    limit, worst, qs = budget(x) if limit is None else limit, 0.0, query_start(lens).tolist()
    for r, (lo, hi) in enumerate(zip(qs[:-1], qs[1:], strict=True)):
        if hi == lo:
            continue
        want_o, want_s = exact(rows(x, lo, hi), g, h0s[r])
        bf_o, bf_s = bf16_path(rows(x, lo, hi), g, h0s[r])
        for name, got, want, floor in (("o", o[lo:hi], want_o, bf_o), ("state", state[slots[r]], want_s, bf_s)):
            assert torch.isfinite(got).all(), (what, r, name)
            x_ratio = ratio(got, want, floor)
            assert x_ratio <= limit, (what, r, hi - lo, name, fro(got, want), fro(floor, want))
            worst = max(worst, x_ratio)
    return worst


# --- the rule against f64 ------------------------------------------------------------------------------------------

def test_the_f64_recurrence_is_the_oracle():
    h, d = SHAPES["small"]
    x, g = tokens(70, h, d), gates(h, d)
    h0 = warm(h, d, g)
    state = h0[None].clone()
    got = ref.kda(x["q"], x["k"], x["v"], x["f"], x["b"], g["a_log"], g["dt_bias"], state, dev([0]),
                  query_start([70]), dev([True], torch.bool), LB)
    want_o, want_s = exact(x, g, h0)
    assert fro(got.float(), want_o) < 4e-3                     # the oracle's own bf16 output rounding
    torch.testing.assert_close(state[0].double(), want_s, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("batch", ["mixed", "decode", "bound"])
def test_kda_within_twice_the_bf16_path(shape, batch):
    h, d = SHAPES[shape]
    lens, init, slots = (([1] * len(DECODE_SLOTS), DECODE_INIT, DECODE_SLOTS) if batch == "decode" else
                         (LENS, INIT, SLOTS))
    x, g = tokens(sum(lens), h, d, bound=batch == "bound"), gates(h, d)
    pool, w = torch.full((UNITS, h, d, d), float("nan"), device="cuda"), warm(h, d, g)
    h0s = []
    for r, (unit, has) in enumerate(zip(slots, init, strict=True)):
        if has or lens[r] == 0:                                 # zero-token rows hold a state they must keep
            pool[unit] = w * (1 + r / 8)
        h0s.append(pool[unit].clone() if has else None)
    before = pool.clone()
    o = run(kda(shape), x, g, pool, lens, init, slots)
    assert o.shape == x["v"].shape and o.dtype == torch.bfloat16
    written = {u for u, n in zip(slots, lens, strict=True) if n}
    untouched = [u for u in range(UNITS) if u not in written]
    assert torch.equal(pool[untouched].view(torch.int32), before[untouched].view(torch.int32))
    worst = judge_rows(o, pool, x, g, h0s, lens, slots, f"{shape}/{batch}")
    print(f"kda {shape} {batch}: worst error / bf16 path error = {worst:.3f}")


def test_decode_continues_prefill_step_by_step():
    h, d = SHAPES["glm-5.3-flash"]
    prefill, steps, slots = [63, 200, 1, 64], 8, [0, UNITS - 1, 6, 2]
    g, kernel = gates(h, d), kda("glm-5.3-flash")
    streams = [tokens(n + steps, h, d) for n in prefill]
    pool = torch.full((UNITS, h, d, d), float("nan"), device="cuda")
    batch = {n: torch.cat([s[n][:p] for s, p in zip(streams, prefill, strict=True)]) for n in streams[0]}
    outs = [[o] for o in run(kernel, batch, g, pool, prefill, [False] * 4, slots).split(prefill)]
    for j in range(steps):
        step = {n: torch.cat([s[n][p + j:p + j + 1] for s, p in zip(streams, prefill, strict=True)]) for n in batch}
        for out, o in zip(outs, run(kernel, step, g, pool, [1] * 4, [True] * 4, slots).split(1), strict=True):
            out.append(o)
    worst = 0.0
    for r, s in enumerate(streams):
        want_o, want_s = exact(s, g, None)
        bf_o, bf_s = bf16_path(s, g, None)
        for got, want, floor in ((torch.cat(outs[r]), want_o, bf_o), (pool[slots[r]], want_s, bf_s)):
            x_ratio = ratio(got, want, floor)
            assert x_ratio <= RATIOS[h], (r, fro(got, want), fro(floor, want))
            worst = max(worst, x_ratio)
    print(f"kda prefill + decode: worst error / bf16 path error = {worst:.3f}")


# --- exactness ------------------------------------------------------------------------------------------------------

def test_aligned_splits_repeats_and_per_token_rows_are_bitwise():
    """D4 / A-m3: a row run as pieces split at multiples of 64 (KDAPlanner's sub-batches: 128, then 64, then 8),
    each continuing the carried f32 state, is bitwise one call; runs repeat bitwise on both paths; per-token rows
    (verify rows) give their one-token steps' outputs bitwise, states within STATE_ULPS (`test_gdn_kernels.py`'s
    bound: the kernel stores once after the last token instead of in its loop)."""
    h, d = SHAPES["glm-5.3-flash"]
    x, g, kernel = tokens(200, h, d), gates(h, d), kda("glm-5.3-flash")
    one = torch.full((2, h, d, d), float("nan"), device="cuda")
    split = one.clone()
    o_one = run(kernel, x, g, one, [200], [False], [1])
    parts = [run(kernel, rows(x, lo, hi), g, split, [hi - lo], [lo > 0], [1])
             for lo, hi in ((0, 128), (128, 192), (192, 200))]
    assert torch.equal(torch.cat(parts), o_one) and torch.equal(split[1], one[1])
    for lens, init in (([200], [False]), ([1], [True])):
        x1 = rows(x, 0, lens[0])
        pools = [torch.zeros(2, h, d, d, device="cuda") + 0.01 for _ in range(2)]
        outs = [run(kernel, x1, g, p, lens, init, [0]) for p in pools]
        assert torch.equal(outs[0], outs[1]) and torch.equal(pools[0], pools[1])
    x3, w = rows(x, 0, 3), warm(h, d, g)
    multi, single = w[None].clone(), w[None].clone()
    o_multi = run(kernel, x3, g, multi, [3], [True], [0], per_token=True)
    o_single = torch.cat([run(kernel, rows(x3, t, t + 1), g, single, [1], [True], [0]) for t in range(3)])
    assert torch.equal(o_multi, o_single)                       # the state: another store, within a few ulps
    assert (multi - single).abs().max() <= STATE_ULPS * torch.finfo(torch.float32).eps * single.abs().max()


@pytest.mark.parametrize("lens, snaps", [([1100], [(0, 384, 1)]), ([700, 0, 450, 90], [(0, 512, 0), (2, 192, 1)])])
def test_sub_calls_and_snapshots_are_bitwise_one_call(lens, snaps, monkeypatch):
    """A step past SUB_TOKENS (256 here) runs in calls each from the state the one before left: outputs, final
    states and snapshots bitwise one call's; each snapshot is bitwise the final state of a call over the row's first
    tokens (one chunk later is another state)."""
    h, d = SHAPES["tiny-gpu"]
    n = len(lens)
    init, slots = [r % 2 == 0 for r in range(n)], [2 + r for r in range(n)]
    kernel, x, g = kda("tiny-gpu"), tokens(sum(lens), h, d), gates(h, d)
    pool, w = torch.full((UNITS, h, d, d), float("nan"), device="cuda"), warm(h, d, g)
    for s in slots:
        pool[s] = w
    runs = []
    for sub in (1 << 20, 256):
        monkeypatch.setattr(fla_plugin, "SUB_TOKENS", sub)
        state, payload = pool.clone(), torch.full((2, h, d, d), float("nan"), device="cuda")
        runs.append((run(kernel, x, g, state, lens, init, slots, snaps, payload), state, payload))
    (o0, s0, p0), (o1, s1, p1) = runs
    assert torch.equal(o0, o1) and torch.equal(s0.nan_to_num(), s1.nan_to_num())
    assert torch.equal(p0.nan_to_num(), p1.nan_to_num())
    qs = query_start(lens).tolist()
    for row, offset, unit in snaps:
        lo, split, late = qs[row], pool.clone(), pool.clone()
        run(kernel, rows(x, lo, lo + offset), g, split, [offset], [init[row]], [slots[row]])
        assert torch.equal(p0[unit], split[slots[row]]), (row, offset)
        run(kernel, rows(x, lo, lo + offset + 64), g, late, [offset + 64], [init[row]], [slots[row]])
        assert not torch.equal(p0[unit], late[slots[row]])


def test_decode_step_graph_replay_equals_eager():
    h, d = SHAPES["glm-5.3-flash"]
    n, kernel, g = len(DECODE_SLOTS), kda("glm-5.3-flash"), gates(h, d)
    slots, init, handle = dev(DECODE_SLOTS), dev(DECODE_INIT, torch.bool), kernel.plan([1] * n)
    inputs = tokens(n, h, d)
    pool = torch.randn(UNITS, h, d, d, device="cuda") * 0.05

    def step(p):
        return kernel(handle, inputs["q"], inputs["k"], inputs["v"], inputs["f"], inputs["b"], g["a_log"],
                      g["dt_bias"], p, slots, init)

    p0 = pool.clone()
    step(pool.clone())                                          # compile outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = step(pool)
    for name, value in tokens(n, h, d).items():                # new inputs, the captured state restored
        inputs[name].copy_(value)
    slots.copy_(slots.flip(0))
    init.copy_(~init)
    pool.copy_(p0)
    graph.replay()
    p_eager = p0.clone()
    eager = step(p_eager)
    assert torch.equal(out, eager) and torch.equal(pool, p_eager)


def test_kda_never_syncs():
    h, d = SHAPES["small"]
    kernel, g = kda("small"), gates(h, d)
    pool = torch.zeros(UNITS, h, d, d, device="cuda")
    for lens, init, slots in ((LENS, INIT, SLOTS), ([1] * 6, DECODE_INIT, DECODE_SLOTS)):
        x, s, i = tokens(sum(lens), h, d), dev(slots), dev(init, torch.bool)
        args = (x["q"], x["k"], x["v"], x["f"], x["b"], g["a_log"], g["dt_bias"], pool, s, i)
        kernel(kernel.plan(lens), *args)
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            kernel(kernel.plan(lens), *args)
        finally:
            torch.cuda.set_sync_debug_mode("default")


# --- the slot-0 case (D4) -------------------------------------------------------------------------------------------

def test_one_sequence_decodes_at_slot_zero_and_the_old_sentinel_fails(tmp_path, monkeypatch):
    """At `max_seqs` 1 the only sequence is slot 0: its decode steps against the oracle. fla's original sentinel
    (`<= 0`: slot 0 is padding), rebuilt from the vendored source, returns it unwritten."""
    h, d = SHAPES["glm-5.3-flash"]
    kernel, g, x = kda("glm-5.3-flash"), gates(h, d), tokens(4, h, d)
    h0 = warm(h, d, g)
    pool = torch.full((UNITS, h, d, d), float("nan"), device="cuda")
    pool[0] = h0
    outs = [run(kernel, rows(x, t, t + 1), g, pool, [1], [True], [0]) for t in range(4)]
    want_o, want_s = exact(x, g, h0)
    bf_o, bf_s = bf16_path(x, g, h0)
    assert ratio(torch.cat(outs), want_o, bf_o) <= RATIOS[h] and ratio(pool[0], want_s, bf_s) <= RATIOS[h]
    src = inspect.getsource(fused_sigmoid_gating)
    old = src.replace("if state_idx < 0:", "if state_idx <= 0:").replace("if final_state_idx >= 0:",
                                                                         "if final_state_idx > 0:")
    assert old != src
    path = tmp_path / "fused_sigmoid_gating_old_sentinel.py"
    path.write_text(old)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(fla_plugin, "fused_sigmoid_gating_delta_rule_update",
                        module.fused_sigmoid_gating_delta_rule_update)
    planted = torch.full((UNITS, h, d, d), float("nan"), device="cuda")
    planted[0] = h0
    run(kernel, rows(x, 0, 1), g, planted, [1], [True], [0])
    assert torch.equal(planted[0], h0)                          # never written: the sequence was skipped


# --- the rewind at KDA's gate width (D4) ---------------------------------------------------------------------------

@pytest.mark.parametrize("width", [4, 8])
def test_rewind_at_kdas_gate_width_is_its_oracle(width):
    h, d = SHAPES["glm-5.3-flash"]
    layers, rows_n, c = 3, 3, 3 * h * d
    facts = dict(channels=c, kernel_size=KERNEL, heads_v=h, dtype="bfloat16", gate_width=h * d)
    ctx = BuildContext(PluginSet(discover()), torch.device("cuda"))
    spec = ctx.select_kernel("gdn_rewind@1", site="test", **facts)
    assert spec.provider == "kernels.fla"
    rewind = spec.impl(ctx, **facts)
    t = rows_n * width
    state = torch.randn(layers, UNITS, c, KERNEL - 1, device="cuda").bfloat16()
    window = torch.randn(layers, rows_n, c, KERNEL - 1, device="cuda").bfloat16()
    xs = torch.randn(layers, t, c, device="cuda").bfloat16()
    a = torch.randn(layers, t, h * d, device="cuda").bfloat16()
    b = torch.randn(layers, t, h, device="cuda").bfloat16()
    slots = dev([5, 0, 9])
    accepted = torch.zeros(UNITS, dtype=torch.int64, device="cuda")
    accepted[[5, 0, 9]] = torch.tensor([0, width - 1, width // 2], device="cuda")
    want = [y.clone() for y in (state, a, b)]
    ref.gdn_rewind(*want[:1], window, xs, *want[1:], slots, accepted, width)
    rewind(state, window, xs, a, b, slots, accepted, width)
    for got, w in zip((state, a, b), want, strict=True):
        assert torch.equal(got, w)
    assert torch.isneginf(a[0, 1:width].float()).all() and torch.isneginf(b[0, 1:width].float()).all()


@pytest.mark.parametrize("path", ["recurrent", "chunked"])
def test_masked_tokens_leave_the_state_bitwise_unchanged(path):
    """What the rewind leaves past a row's accepted tokens (f = b = −inf: g = −5 · σ(−inf) = −0, β = 0) is the rule's
    identity step: the row's state after its tokens and the masked ones is bitwise its state after its tokens alone,
    token by token (per-token rows, the recurrent kernel) and in one chunked call."""
    h, d = SHAPES["glm-5.3-flash"]
    kernel, g = kda("glm-5.3-flash"), gates(h, d)
    n, extra = (3, 2) if path == "recurrent" else (100, 20)
    x = tokens(n + extra, h, d)
    for name in ("f", "b"):
        x[name][n:] = MASKED
    w = warm(h, d, g)
    kept, masked = w[None].clone(), w[None].clone()
    per_token = path == "recurrent"
    run(kernel, rows(x, 0, n), g, kept, [n], [True], [0], per_token=per_token)
    run(kernel, x, g, masked, [n + extra], [True], [0], per_token=per_token)
    assert torch.equal(kept, masked)


@pytest.mark.parametrize("batch", ["mixed", "decode"])
def test_plant_a_looser_gate_bound_fails(batch, monkeypatch):
    """The gate's bound at −4 instead of the fact's −5, on both paths, at GLM's shape: ≥ 5× its budget."""
    h, d = SHAPES["glm-5.3-flash"]
    lens, init, slots = (LENS, INIT, SLOTS) if batch == "mixed" else ([1] * 6, DECODE_INIT, DECODE_SLOTS)
    x, g, kernel = tokens(sum(lens), h, d), gates(h, d), kda("glm-5.3-flash")
    kernel.lower_bound, kernel._recurrent = -4.0, {"is_kda": True, "lower_bound": -4.0}
    pool, w = torch.zeros(UNITS, h, d, d, device="cuda"), warm(h, d, g)
    pool[slots] = w
    h0s = [w if has else None for has in init]
    worst = judge_rows(run(kernel, x, g, pool, lens, init, slots), pool, x, g, h0s, lens, slots, "planted", math.inf)
    print(f"plant gate bound -4 ({batch}): worst ratio {worst:.1f} (budget {budget(x)})")
    assert worst >= 5 * budget(x)                   # (measured 20-21: a bound one unit off is the subtlest gate plant)


def test_plant_kda_gate_softplus_fails():
    """§6.3's `kda_gate_softplus`: the recurrent (decode) kernel's gate unbounded (GDN's softplus form per channel)
    fails the decode L0 by ≥ 10× its budget."""
    from holon_judge.planted import PLANTS

    h, d = SHAPES["tiny-gpu"]
    lens, init, slots = [1] * len(DECODE_SLOTS), DECODE_INIT, DECODE_SLOTS
    x, g = tokens(sum(lens), h, d), gates(h, d)
    with PLANTS["kda_gate_softplus"]():
        kernel = kda("tiny-gpu")
    pool, w = torch.zeros(UNITS, h, d, d, device="cuda"), warm(h, d, g)
    pool[slots] = w
    h0s = [w if has else None for has in init]
    worst = judge_rows(run(kernel, x, g, pool, lens, init, slots), pool, x, g, h0s, lens, slots, "planted", math.inf)
    print(f"plant kda_gate_softplus: worst ratio {worst:.1f} (budget {budget(x)})")
    assert worst >= 10 * budget(x)
