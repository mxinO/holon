"""L0 for Qwen3.8-Flash-Next's hyper-connection kernels (`kernels.hc_triton`: `hc_gr_mix@1`, `hc_gr_combine@1`) and
PLE kernels (`kernels.ple_triton`: `qwen4_exp.ple_rows@1`, `qwen4_exp.ple_block@1`) against their oracles
(docs/dev/v1-flash-next.md acceptance 2, 3), on FN's shapes (4 streams of 2,560, rank 320; 16 hash heads, trigram, EOS
248044; conv 4 taps at dilation 3), through the build context.

- `hc_gr_mix@1` within `l0.check` of the oracle (x at its dtype, w in float32, over n·H terms), with and without
  `inject` (f32 beside bf16 `down`), with and without a pending combine (y_prev, w_prev: the same block's previous
  half), for 0, 1, 7 and 77 tokens; the residual a pending combine leaves bitwise what `hc_gr_combine@1` leaves, and
  that within float32's elementwise budget. A block as the unit runs it (mix, combine riding in the next mix, its
  last combine applied) leaves R as the oracle's block does.
- The in-place update of R under stress (the GEMV mix's race, FN step 5, §17.1): with a pending combine, at 1 to 300
  tokens (the GEMV and tiled paths, few to many programs), 4 and 8 warps, repeated under side-stream load of varying
  size (so programs run in other orders): R bitwise what `hc_gr_combine@1` leaves, x bitwise the same every repeat.
  The racy kernel of 30c7fd7 fails it on sm_86.
- `qwen4_exp.ple_rows@1` bitwise (rows and history), with wrapping int64 products, EOS at a row's start, inside it and
  in the history, decode rows, a zero-token row and rows without has_init.
- `qwen4_exp.ple_block@1` within `l0.check(f32, H)` of the oracle (r, state and the returned conv inputs C); state
  units of rows without has_init are NaN and must not be read, and units no row writes stay untouched. Acceptance 2's
  GPU unit test, chunked half: every row split into chunks, down to 1-token decode steps, bitwise one shot (r, state
  and C); its replay-rewind half runs the PLE unit (`test_qwen4_exp_units.py`).

Skips without a GPU.
"""
from unittest import mock

import pytest
import torch
from holon.core.build import BuildContext
from holon.core.plugins import PluginSet, discover
from holon_judge import l0
from holon_plugins.models.qwen4_exp import kernels as ple_kernels
from holon_plugins.models.qwen4_exp import reference as ple
from holon_std.hyper import kernels as hc_kernels
from holon_std.hyper import reference as hc

pytestmark = [pytest.mark.l0, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")]

STREAMS, HIDDEN, RANK, EPS = 4, 2560, 320, 1e-6
NGRAM, HEADS_PER, EOS, VOCAB = 3, 8, 248044, 248320
TAPS, DILATION, UNITS = 4, 3, 8
WINDOW = (TAPS - 1) * DILATION


def _kernel(op: str, provider: str, **facts):
    ctx = BuildContext(PluginSet(discover()), torch.device("cuda"), options={"kernels": {op.split("@")[0]: provider}})
    return ctx.select_kernel(op, site="test", **facts).impl(ctx, **facts)


def _check(ours: torch.Tensor, ref: torch.Tensor, dtype: torch.dtype, n: int, what: str) -> None:
    report = l0.check(ours, ref, dtype, n)
    print(f"\n{what}: usage {report.usage:.3f}")
    assert report.passed, (what, report)


@pytest.mark.parametrize("inject", [True, False])
def test_hc(inject):
    g = torch.Generator().manual_seed(0)
    width = STREAMS * HIDDEN
    norm = (torch.randn(width, generator=g) * 0.1).bfloat16()
    down = (torch.randn(RANK, width, generator=g) / width ** 0.5).bfloat16()
    up = (torch.randn(width, RANK, generator=g) / RANK ** 0.5).bfloat16()
    gate = torch.randn(STREAMS, width, generator=g) / width ** 0.5 if inject else None   # f32 beside bf16 down
    weights = [x.cuda() if x is not None else None for x in (norm, down, up, gate)]
    mix = _kernel(hc_kernels.MIX, "kernels.hc_triton", streams=STREAMS, hidden=HIDDEN, rank=RANK, eps=EPS,
                  dtype="bfloat16", inject=inject)
    combine = _kernel(hc_kernels.COMBINE, "kernels.hc_triton", streams=STREAMS, hidden=HIDDEN, dtype="bfloat16")
    for t in (0, 1, 7, 77):
        r = torch.randn(t, width, generator=g) * torch.linspace(0.5, 4.0, STREAMS).repeat_interleave(HIDDEN)
        y = torch.randn(t, HIDDEN, generator=g).bfloat16()
        w_prev = 2 * torch.rand(t, STREAMS, generator=g)
        for pending in (False, True):                    # a pending combine (the block's previous half) rides along
            extra = (y, w_prev) if pending else (None, None)
            ours = r.cuda()
            x, w = mix(ours, *weights, *[e.cuda() if e is not None else None for e in extra])
            r_ref = r.clone()
            x_ref, w_ref = hc.hc_gr_mix(r_ref, norm, down, up, gate, *extra, STREAMS, EPS, torch.bfloat16)
            what = f"{t} tokens{', pending' if pending else ''}"
            assert x.dtype == torch.bfloat16 and x.shape == (t, HIDDEN) and (w is None) == (not inject)
            _check(x, x_ref, torch.bfloat16, width, f"mix x, {what}")
            _check(ours, r_ref, torch.float32, 1, f"mix r, {what}")
            if inject:
                _check(w, w_ref, torch.float32, width, f"mix w, {what}")
            if pending:                                  # R as hc_gr_combine leaves it, bit for bit
                alone = r.cuda()
                combine(alone, y.cuda(), w_prev.cuda())
                assert torch.equal(alone, ours)
        if inject and t:                                 # a block as the unit runs it: R complete when it returns
            y_attn, y_ffn = (torch.randn(t, HIDDEN, generator=g).bfloat16() for _ in range(2))
            ours = r.cuda()
            _, w_attn = mix(ours, *weights)
            _, w_ffn = mix(ours, *weights, y_attn.cuda(), w_attn)          # the attention half's combine rides here
            combine(ours, y_ffn.cuda(), w_ffn)                              # the block's last combine, applied
            r_ref = r.clone()
            _, w_a = hc.hc_gr_mix(r_ref, norm, down, up, gate, None, None, STREAMS, EPS, torch.bfloat16)
            _, w_f = hc.hc_gr_mix(r_ref, norm, down, up, gate, y_attn, w_a, STREAMS, EPS, torch.bfloat16)
            hc.hc_gr_combine(r_ref, y_ffn, w_f)
            _check(ours, r_ref, torch.float32, width, f"block, {t} tokens")


def test_hc_prefill_bf16x3():
    """`hc.prefill=bf16x3` (FN step 7 item 7): prefill's projections as three bf16 tensor-core products of the
    activation's exact split, float32 accumulation — x, w and R within `l0.check` of the oracle as the SGEMMs are, at
    FN's bf16 weights (inject too), with and without a pending combine."""
    g = torch.Generator().manual_seed(1)
    width = STREAMS * HIDDEN
    norm = (torch.randn(width, generator=g) * 0.1).bfloat16()
    down = (torch.randn(RANK, width, generator=g) / width ** 0.5).bfloat16()
    up = (torch.randn(width, RANK, generator=g) / RANK ** 0.5).bfloat16()
    gate = (torch.randn(STREAMS, width, generator=g) / width ** 0.5).bfloat16()
    weights = [x.cuda() for x in (norm, down, up, gate)]
    ctx = BuildContext(PluginSet(discover()), torch.device("cuda"),
                       options={"kernels": {"hc_gr_mix": "kernels.hc_triton"}, "plugins": {"hc.prefill": "bf16x3"}})
    facts = {"streams": STREAMS, "hidden": HIDDEN, "rank": RANK, "eps": EPS, "dtype": "bfloat16", "inject": True}
    mix = ctx.select_kernel(hc_kernels.MIX, site="test", **facts).impl(ctx, **facts)
    assert mix.prefill == "bf16x3"
    for t in (77, 300):
        r = torch.randn(t, width, generator=g) * torch.linspace(0.5, 4.0, STREAMS).repeat_interleave(HIDDEN)
        y, w_prev = torch.randn(t, HIDDEN, generator=g).bfloat16(), 2 * torch.rand(t, STREAMS, generator=g)
        for extra in ((None, None), (y, w_prev)):
            ours = r.cuda()
            x, w = mix(ours, *weights, *[e.cuda() if e is not None else None for e in extra])
            r_ref = r.clone()
            x_ref, w_ref = hc.hc_gr_mix(r_ref, norm, down, up, gate, *extra, STREAMS, EPS, torch.bfloat16)
            what = f"bf16x3, {t} tokens{', pending' if extra[0] is not None else ''}"
            _check(x, x_ref, torch.bfloat16, width, f"mix x, {what}")
            _check(w, w_ref, torch.float32, width, f"mix w, {what}")
            _check(ours, r_ref, torch.float32, 1, f"mix r, {what}")
    # its passes of PREFILL_ROWS_X3 rows (whole `_mm3` row tiles): the same bits as one pass
    from holon_plugins.kernels.hc_triton import kernel as hc_kernel
    r = (torch.randn(300, width, generator=g) * 2).cuda()
    outs = []
    for rows in (128, 4096):
        with mock.patch.object(hc_kernel, "PREFILL_ROWS_X3", rows):
            ours = r.clone()
            outs.append((*mix(ours, *weights, None, None), ours))
    assert all(torch.equal(a, b) for a, b in zip(*outs, strict=True))


@pytest.mark.parametrize("warps", [4, 8])
def test_hc_in_place_stress(warps, monkeypatch):
    from holon_plugins.kernels.hc_triton import kernel
    monkeypatch.setattr(kernel, "WARPS", warps)
    g = torch.Generator().manual_seed(1)
    width = STREAMS * HIDDEN
    norm = (torch.randn(width, generator=g) * 0.1).bfloat16().cuda()
    down = (torch.randn(RANK, width, generator=g) / width ** 0.5).bfloat16().cuda()
    up = (torch.randn(width, RANK, generator=g) / RANK ** 0.5).bfloat16().cuda()
    gate = (torch.randn(STREAMS, width, generator=g) / width ** 0.5).cuda()
    mix = _kernel(hc_kernels.MIX, "kernels.hc_triton", streams=STREAMS, hidden=HIDDEN, rank=RANK, eps=EPS,
                  dtype="bfloat16", inject=True)
    combine = _kernel(hc_kernels.COMBINE, "kernels.hc_triton", streams=STREAMS, hidden=HIDDEN, dtype="bfloat16")
    side, load = torch.cuda.Stream(), torch.randn(2048, 2048, device="cuda")
    for t in (1, 4, 16, 17, 77, 300):
        r = torch.randn(t, width, generator=g).cuda()
        y = torch.randn(t, HIDDEN, generator=g).bfloat16().cuda()
        w_prev = 2 * torch.rand(t, STREAMS, generator=g).cuda()
        want = r.clone()
        combine(want, y, w_prev)
        first = None
        for rep in range(8):
            with torch.cuda.stream(side):                # other work on the card: programs scheduled differently
                for _ in range(rep % 4):
                    load @ load
            ours = r.clone()
            x, _ = mix(ours, norm, down, up, gate, y, w_prev)
            torch.cuda.synchronize()
            assert torch.equal(ours, want), (t, warps, rep)
            first = x if first is None else first
            assert torch.equal(x, first), (t, warps, rep)


# (slot, has_init, tokens): a prefill, decode rows, a zero-token row, rows with and without has_init
ROWS = [(2, False, 37), (5, True, 1), (0, True, 0), (7, True, 5), (3, False, 2), (1, True, 130), (6, True, 1)]


def _rows(rows=ROWS):
    slots = torch.tensor([s for s, _, _ in rows], dtype=torch.int32)
    has_init = torch.tensor([i for _, i, _ in rows], dtype=torch.bool)
    query_start = torch.tensor([0] + [n for _, _, n in rows], dtype=torch.int32).cumsum(0, dtype=torch.int32)
    return slots, query_start, has_init


def test_ple_rows():
    g = torch.Generator().manual_seed(1)
    slots, query_start, has_init = _rows()
    ids = torch.randint(VOCAB, (int(query_start[-1]),), generator=g)
    starts = query_start[:-1].long()
    ids[starts[0]] = ids[starts[0] + 10] = ids[starts[0] + 11] = EOS              # at the start, two in a row
    ids[starts[3] + 1] = ids[starts[5] + 2] = EOS
    history = torch.randint(VOCAB, (UNITS, 2), generator=g)
    history[5, 1] = history[7, 0] = EOS                                            # EOS just before / two before
    heads = (NGRAM - 1) * HEADS_PER
    multipliers = torch.randint(1 << 62, (NGRAM,), generator=g) * 2 + 1            # products wrap int64
    sizes = torch.randint(10 ** 7, 3 * 10 ** 7, (heads,), generator=g)
    offsets = torch.cat((torch.zeros(1, dtype=torch.long), sizes.cumsum(0)[:-1]))
    kernel = _kernel(ple_kernels.ROWS, "kernels.ple_triton", ngram=NGRAM, heads_per_ngram=HEADS_PER, eos=EOS)
    ours_history = history.cuda()
    ours = kernel(ids.cuda(), ours_history, slots.cuda(), query_start.cuda(), has_init.cuda(), multipliers.cuda(),
                  sizes.cuda(), offsets.cuda())
    oracle = ple.ple_rows(ids, history, slots, query_start, has_init, multipliers, sizes, offsets, NGRAM, HEADS_PER,
                          EOS)
    assert torch.equal(ours.cpu(), oracle) and torch.equal(ours_history.cpu(), history)


def _block_inputs(t: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    width = STREAMS * HIDDEN
    key = torch.randn(t, width, generator=g).bfloat16()
    value = torch.randn(t, HIDDEN, generator=g).bfloat16()
    r = torch.randn(t, width, generator=g) * 3
    weights = [(torch.randn(width, generator=g) * 0.1).bfloat16() for _ in range(3)]
    conv = (torch.randn(width, TAPS, generator=g) * 0.5).bfloat16()
    state = torch.randn(UNITS, WINDOW, width, generator=g)
    return key, value, r, weights, conv, state


def _block_kernel():
    return _kernel(ple_kernels.BLOCK, "kernels.ple_triton", streams=STREAMS, hidden=HIDDEN, kernel=TAPS,
                   dilation=DILATION, eps=EPS, dtype="bfloat16")


def test_ple_block():
    slots, query_start, has_init = _rows()
    key, value, r, weights, conv, state = _block_inputs(int(query_start[-1]), 2)
    for s, init, _ in ROWS:
        if not init:
            state[s] = float("nan")                                             # must not be read
    ours_r, ours_state = r.cuda(), state.cuda()
    ours_c = _block_kernel()(key.cuda(), value.cuda(), ours_r, *[w.cuda() for w in weights], conv.cuda(), ours_state,
                             slots.cuda(), query_start.cuda(), has_init.cuda())
    c = ple.ple_block(key, value, r, *weights, conv, state, slots, query_start, has_init, STREAMS, DILATION, EPS)
    _check(ours_r, r, torch.float32, HIDDEN, "ple_block r")
    _check(ours_c, c, torch.float32, HIDDEN, "ple_block C")
    _check(ours_state, state, torch.float32, HIDDEN, "ple_block state")   # untouched units: equal NaN or values


def test_ple_block_chunks():
    """Every row split into chunks (straddling the 9-row window, then 1-token steps), bitwise one shot."""
    cuts = {2: [4, 33], 7: [1, 1, 3], 3: [1, 1], 1: [61, 1, 1, 67]}
    rows = [row for row in ROWS if row[2]]
    slots, query_start, has_init = _rows(rows)
    key, value, r, weights, conv, state = [x.cuda() if isinstance(x, torch.Tensor) else [w.cuda() for w in x]
                                           for x in _block_inputs(int(query_start[-1]), 3)]
    kernel = _block_kernel()
    one_r, one_state = r.clone(), state.clone()
    one_c = kernel(key, value, one_r, *weights, conv, one_state, slots.cuda(), query_start.cuda(), has_init.cuda())
    done, c = [0] * len(rows), torch.empty_like(one_c)
    for step in range(max(len(c) for c in cuts.values())):
        part = [(i, cuts.get(s, [n])[step]) for i, (s, _, n) in enumerate(rows) if step < len(cuts.get(s, [n]))]
        tokens = torch.cat([torch.arange(n) + int(query_start[i]) + done[i] for i, n in part]).cuda()
        p_slots, p_start, _ = _rows([(rows[i][0], False, n) for i, n in part])
        p_init = torch.tensor([rows[i][1] or done[i] > 0 for i, _ in part])
        chunk = r[tokens]
        c[tokens] = kernel(key[tokens], value[tokens], chunk, *weights, conv, state, p_slots.cuda(), p_start.cuda(),
                           p_init.cuda())
        r[tokens] = chunk
        for i, n in part:
            done[i] += n
    assert torch.equal(r, one_r) and torch.equal(state, one_state) and torch.equal(c, one_c)
