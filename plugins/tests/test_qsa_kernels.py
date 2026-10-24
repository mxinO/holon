"""L0 for `kernels.qsa_triton` (docs/v1-flash-next.md acceptance 3) against std.mixers.qsa's oracles, on
Qwen3.8-Flash-Next's shapes (indexer 4 × 128, rotary 64 at θ 1e7; attention 24 q / 2 kv heads of 256; ratio 4,
budget 2,048), through the build context.

- `qsa_index@1` (one launch): prefill in two chunks per sequence, the second starting inside a block (its pooled keys
  read raw keys of the first chunk and of its own), and padding tokens; k in bf16 and in f32 (pooled from k as raw
  stores it). `raw` bitwise; `pooled` within
  `l0.check(f32, Dk)`; slots no token writes stay untouched. The kernels read `raw` and `pooled` as the unit lays them
  out: views of one segment per unit (raw keys, then pooled; units strided), as `std.mixers.qsa@1`'s `_idx_layer`.
- `qsa_select@1`, exactly equal to the oracle where its float32 scores have no near-ties: pooled keys drawn from six
  vectors, one of them zero, so every boundary is a large tie that D2's lower-block rule decides, a tie at ReLU's 0.0
  among them (asserted). Rows: 32K-token queries (8,192 blocks, and 8,195: past the kernel's in-register row), queries
  just above, at and below the budget, open blocks, padding. With continuous keys, a valid top-k of the oracle's
  scores within float32 rounding. Shape invariance (B-1): a query's scores and selection are the same bits alone and
  in a 128-row call; sub-batching changes no bit.
- `qsa_attend@1` for prefill, decode and verify rows (sparse past 2,051 tokens, dense below, padding) in bf16 and fp16,
  and from the q8_0 and fp8 caches (fp8 from sm_89), within the attention budget of test_attention_kernels against the
  oracle reading the same stored cache, with and without the output gate (a strided view, as gqa's gated q layout
  gives it; NaN in padding rows' gates still gives 0); batches whose shapes split the selected tokens 32, 8, 4 ways
  and not at all. A plant must fail: the open block reading one token more.

Skips without a GPU.
"""
import numpy as np
import pytest
import torch
from holon.core.build import BuildContext
from holon.core.plugins import PluginSet, discover
from holon_judge import l0
from holon_plugins.kernels.qsa_triton import kernel as qsa_kernel
from holon_std.mixers.gqa import reference as gqa
from holon_std.mixers.qsa import kernels as qsa_kernels
from holon_std.mixers.qsa import reference as ref

from .test_attention_kv_quant import empty_cache, fp8_device, usage

pytestmark = [pytest.mark.l0, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")]

PAGE, SLOTS, RATIO, BUDGET = 16, 8, 4, 2048          # Qwen3.8-Flash-Next's indexer: 4-token blocks, 2,048 tokens
HI, DK, ROT, THETA = 4, 128, 64, 1e7
HQ, HKV, D = 24, 2, 256
WIDTH = qsa_kernels.width(BUDGET, RATIO)


def _kernel(op: str, **facts):
    ctx = BuildContext(PluginSet(discover()), torch.device("cuda"),
                       options={"kernels": {op.split("@")[0]: "kernels.qsa_triton"}})
    return ctx.select_kernel(op, site="test", **facts).impl(ctx, **facts)


def _table(max_len: int, seed: int) -> torch.Tensor:
    """[SLOTS, pages] distinct unit ids, shuffled, with a few units no slot uses."""
    pages = -(-max_len // PAGE)
    units = np.random.default_rng(seed).permutation(SLOTS * pages + 5)[: SLOTS * pages]
    return torch.from_numpy(units.astype(np.int32)).view(SLOTS, pages)


def _segments(units: int) -> tuple[torch.Tensor, torch.Tensor]:
    """`raw` [units, PAGE, DK] bf16 and `pooled` [units, PAGE / RATIO, DK] f32 on the GPU, as the unit views its pool:
    one byte segment per unit, raw keys then pooled keys, so units are strided in both (`QSA._idx_layer`)."""
    raw_bytes = PAGE * DK * 2
    seg = torch.empty(units, raw_bytes + PAGE // RATIO * DK * 4, dtype=torch.uint8, device="cuda")
    return (seg[:, :raw_bytes].view(torch.bfloat16).view(-1, PAGE, DK),
            seg[:, raw_bytes:].view(torch.float32).view(-1, PAGE // RATIO, DK))


def _addresses(rows: list[tuple[int, int, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """rows: (slot, first position, tokens); slot −1 gives that many padding tokens."""
    seq = [s for s, _, n in rows for _ in range(n)]
    pos = [p + i if s >= 0 else 0 for s, p, n in rows for i in range(n)]
    return torch.tensor(seq, dtype=torch.int32), torch.tensor(pos, dtype=torch.int64)


@pytest.mark.parametrize("k_dtype", [torch.bfloat16, torch.float32])         # f32 k: pooled as stored (bf16)
def test_index(k_dtype):
    lengths, cut = {3: 37, 0: 130, 6: 5}, {3: 13, 0: 66, 6: 2}        # second chunks start inside a block
    table = _table(max(lengths.values()), 0)
    units = int(table.max()) + 1
    g = torch.Generator().manual_seed(0)
    keys = {s: torch.randn(n, DK, generator=g).to(k_dtype) for s, n in lengths.items()}
    weight = torch.randn(DK, generator=g) * 0.1
    cos_sin = ref.cos_sin(max(lengths.values()), ROT, THETA)
    raw = torch.full((units, PAGE, DK), 7.0, dtype=torch.bfloat16)
    pooled = torch.full((units, PAGE // RATIO, DK), float("nan"))
    ours = _segments(units)
    for view, init in zip(ours, (raw, pooled), strict=True):
        view.copy_(init)
    kernel = _kernel(qsa_kernels.INDEX, head_dim=DK, ratio=RATIO, page=PAGE, rotary_dim=ROT,
                     max_positions=len(cos_sin), eps=1e-6, dtype="bfloat16")
    for chunk in (lambda s: (0, cut[s]), lambda s: (cut[s], lengths[s])):
        rows = [(-1, 0, 2)] + [(s, *(lambda a, b: (a, b - a))(*chunk(s))) for s in lengths] + [(-1, 0, 1)]
        seq, pos = _addresses(rows)
        k = torch.cat([torch.zeros(n, DK, dtype=k_dtype) if s < 0 else keys[s][p:p + n] for s, p, n in rows])
        ref.qsa_index(k, raw, pooled, table, seq, pos, weight, cos_sin, RATIO, 1e-6)
        kernel(k.cuda(), *ours, table.cuda(), seq.cuda(), pos.cuda(), weight.cuda(), cos_sin.cuda())
    assert torch.equal(ours[0].cpu(), raw)
    assert (~pooled.isnan()).sum() == sum(n // RATIO for n in lengths.values()) * DK
    report = l0.check(ours[1], pooled, torch.float32, DK)                 # NaN (never written) must match in place
    assert report.passed, report


SELECT_ROWS = [(2, 32 * 1024 - 1, 3), (3, 32 * 1024 + 12, 1), (5, 2048, 1), (1, 2047, 2), (4, 2043, 1), (7, 100, 2),
               (-1, 0, 2), (6, 0, 1), (0, 2052, 4), (1, 2400, 1)]
# 8,192 blocks from registers, 8,195 re-read per round; at 600 blocks the k-th score is ReLU's 0.0, tied many times


def _pooled(table: torch.Tensor, palette: int | None, seed: int) -> torch.Tensor:
    """Pooled keys for every unit; with a palette, from that many distinct vectors, the first a zero vector (score
    0.0) drawn for 85% of the blocks."""
    g = torch.Generator().manual_seed(seed)
    shape = (int(table.max()) + 1, PAGE // RATIO, DK)
    if palette is None:
        return torch.randn(shape, generator=g)
    vectors = torch.cat((torch.zeros(1, DK), torch.randn(palette - 1, DK, generator=g)))
    weights = torch.tensor([0.85] + [0.15 / (palette - 1)] * (palette - 1))
    return vectors[torch.multinomial(weights, shape[0] * shape[1], replacement=True, generator=g).view(shape[:2])]


def _oracle_scores(q, pooled, table, seq: int, n: int) -> torch.Tensor:
    """The oracle's scores of a query's n blocks, in float64."""
    first = torch.arange(n) * RATIO
    keys = pooled[table[seq, first // PAGE].long(), (first % PAGE) // RATIO].double()
    return (keys @ q.double().T).relu().sum(-1) / DK ** 0.5


def _select(palette: int | None, seed: int = 0):
    table = _table(32 * 1024 + 8, seed)
    pooled = _pooled(table, palette, seed)
    seq, pos = _addresses(SELECT_ROWS)
    q = torch.randn(len(seq), HI, DK, generator=torch.Generator().manual_seed(seed + 1)).bfloat16()
    kernel = _kernel(qsa_kernels.SELECT, heads=HI, head_dim=DK, ratio=RATIO, budget=BUDGET, page=PAGE)
    args = [x.cuda() for x in (q, pooled, table, seq, pos)]
    args[1] = _segments(len(pooled))[1]
    args[1].copy_(pooled)
    return kernel(*args).cpu(), ref.qsa_select(q, pooled, table, seq, pos, RATIO, BUDGET), (q, pooled, table), args


def test_select_ties():
    ours, oracle, (q, pooled, table), args = _select(palette=6)
    assert torch.equal(ours, oracle)
    small = qsa_kernel.qsa_select(*args, RATIO, BUDGET, sub_batch=3)
    assert torch.equal(small.cpu(), ours)
    seq, pos = _addresses(SELECT_ROWS)                   # a boundary tie at 0.0 occurs, many keys deep
    k, t = BUDGET // RATIO, len(seq) - 1
    scores = _oracle_scores(q[t], pooled, table, int(seq[t]), (int(pos[t]) + 1) // RATIO).sort(descending=True).values
    assert scores[k - 1] == 0.0 == scores[k] and (scores[:k] > 0).sum() < k - 10


def test_select_shape_invariant():
    """A query's scores (and selection) are the same bits alone and in a 128-row call of mixed sequences."""
    g = torch.Generator().manual_seed(7)
    table = _table(32 * 1024 + 8, 7).cuda()
    pooled = _pooled(table, None, 7).cuda()
    seq = torch.randint(SLOTS, (128,), generator=g).int().cuda()
    pos = torch.randint(100, 32 * 1024, (128,), generator=g).cuda()
    q = torch.randn(128, HI, DK, generator=g).bfloat16().cuda()
    width = table.shape[1] * PAGE // RATIO
    together = torch.full((128, width), float("nan"), device="cuda")
    qsa_kernel.scores(q, pooled, table, seq, pos, RATIO, together)
    blocks = qsa_kernel.qsa_select(q, pooled, table, seq, pos, RATIO, BUDGET, 512)
    for t in range(0, 128, 9):
        alone = torch.full((1, width), float("nan"), device="cuda")
        qsa_kernel.scores(q[t:t + 1], pooled, table, seq[t:t + 1], pos[t:t + 1], RATIO, alone)
        n = (int(pos[t]) + 1) // RATIO
        assert torch.equal(alone[0, :n], together[t, :n])
        assert torch.equal(qsa_kernel.qsa_select(q[t:t + 1], pooled, table, seq[t:t + 1], pos[t:t + 1], RATIO,
                                                 BUDGET, 512)[0], blocks[t])


def test_select_continuous():
    ours, oracle, (q, pooled, table), _ = _select(palette=None, seed=1)
    seq, pos = _addresses(SELECT_ROWS)
    top = BUDGET // RATIO
    for t in range(len(seq)):
        n = (int(pos[t]) + 1) // RATIO if seq[t] >= 0 else 0
        k = min(n, top)
        chosen = ours[t, :k].long()
        assert torch.equal(ours[t, k:k + 1], oracle[t, k:k + 1]) and (ours[t, k + 1:] == -1).all()
        assert (chosen.diff() > 0).all() and (chosen < n).all()
        if n > top:                                      # the oracle's scores of every block, in float64
            scores = _oracle_scores(q[t], pooled, table, int(seq[t]), n)
            mask = torch.zeros(n, dtype=torch.bool)
            mask[chosen] = True
            assert scores[mask].min() >= scores[~mask].max() - 1e-5 * scores.abs().max()
            assert len(set(chosen.tolist()) ^ set(oracle[t, :k].tolist())) <= 4    # near-ties only


# batches by the splits of the selected tokens their shape gives (32 for decode, ... 1 for prefill)
ATTEND_BATCHES = {32: [(1, 5000, 1), (-1, 0, 1), (6, 9, 1)],
                  8: [(5, 4093, 3), (0, 2051, 1), (2, 2047, 6), (-1, 0, 1)],
                  4: [(3, 0, 37), (1, 5000, 1), (5, 4093, 3), (-1, 0, 1), (0, 2051, 1), (2, 2047, 6), (6, 9, 1)],
                  1: [(3, 0, 130), (1, 4800, 3)]}


@pytest.mark.parametrize("layout,dtype", [("kv.hnd", torch.bfloat16), ("kv.hnd", torch.float16),
                                          ("kv.hnd.q8_0", torch.bfloat16), ("kv.hnd.fp8", torch.bfloat16)])
def test_attend(layout, dtype):
    if layout == "kv.hnd.fp8" and not fp8_device():
        pytest.skip("fp8 KV kernels are selected from sm_89 on")
    table = _table(5008, 2)
    units = int(table.max()) + 1
    g = torch.Generator().manual_seed(2)
    values = torch.randn(units, 2, HKV, PAGE, D, generator=g) * torch.tensor([0.5, 2.0])[None, None, :, None, None]
    kv = empty_cache(units, HKV, D, layout, dtype)
    if layout == "kv.hnd.q8_0":
        q8 = gqa.quantize_q8_0(values)
        kv[0].copy_(q8[0])
        kv[1].copy_(q8[1])
    else:
        kv.copy_(gqa.to_fp8(values) if layout == "kv.hnd.fp8" else values.to(dtype))
    stored = tuple(x.cpu() for x in kv) if isinstance(kv, tuple) else kv.cpu()
    kernel = _kernel(qsa_kernels.ATTEND, heads=HQ, kv_heads=HKV, head_dim=D, ratio=RATIO, width=WIDTH, page=PAGE,
                     layout=layout, q_dtype=str(dtype).split(".")[1],
                     kv_dtype={"kv.hnd.q8_0": "q8_0", "kv.hnd.fp8": "float8_e4m3fn"}.get(layout, str(dtype)[6:]))
    for splits, rows in ATTEND_BATCHES.items():
        seq, pos = _addresses(rows)
        blocks = torch.full((len(seq), WIDTH), -1, dtype=torch.int32)
        for t in range(len(seq)):                        # dense below the budget, a random subset above it
            n = (int(pos[t]) + 1) // RATIO
            if seq[t] < 0:
                continue
            chosen = torch.arange(n) if n <= WIDTH - 1 else torch.randperm(n, generator=g)[: WIDTH - 1].sort().values
            blocks[t, :len(chosen)] = chosen.int()
            if (int(pos[t]) + 1) % RATIO:
                blocks[t, len(chosen)] = n
        qg = torch.randn(len(seq), HQ, 2 * D, generator=g).to(dtype)        # gqa's gated layout: q, then gate
        qg[seq < 0, :, D:] = float("nan")                # stale padding: its output must still be 0
        q, gate = qg[..., :D].contiguous(), qg[..., D:]
        for gated in (False, True):
            ours = kernel(q.cuda(), kv, blocks.cuda(), table.cuda(), seq.cuda(), pos.cuda(),
                          qg.cuda()[..., D:] if gated else None).cpu()
            oracle = ref.qsa_attend(q, stored, blocks, table, seq, pos, gate if gated else None, RATIO)
            assert (ours[seq < 0] == 0).all()
            worst = usage(ours, oracle)
            print(f"\n{layout} {dtype}, {splits} splits{', gated' if gated else ''}: attention budget {worst:.3f}")
            assert worst <= 1.0
        if splits == 32:                                 # a plant: the open block reading one token more (p + 1)
            mutated = ref.qsa_attend(q, stored, blocks, table, seq, pos + 1, gate, RATIO)
            assert usage(ours, mutated) > 1.0
