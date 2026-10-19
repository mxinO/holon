"""L0 for `kernels.qsa_triton` (docs/v1-flash-next.md acceptance 3) against std.mixers.qsa's oracles, on
Qwen3.8-Flash-Next's shapes (indexer 4 × 128, rotary 64 at θ 1e7; attention 24 q / 2 kv heads of 256; ratio 4,
budget 2,048), through the build context.

- `qsa_index@1`: prefill in two chunks per sequence, the second starting inside a block (its pooled keys read raw keys
  of the first chunk), and padding tokens. `raw` bitwise; `pooled` within `l0.check(f32, Dk)`; slots no token writes
  stay untouched.
- `qsa_select@1`, exactly equal to the oracle where its float32 scores have no near-ties: pooled keys drawn from six
  vectors, so every boundary is a large tie that D2's lower-block rule decides. Rows: 32K-token queries (8,192 blocks,
  and 8,195: past the kernel's in-register row), queries just above, at and below the budget, open blocks, padding.
  With continuous keys, a valid top-k of the oracle's scores within float32 rounding. Sub-batching changes no bit.
- `qsa_attend@1` for prefill, decode and verify rows (sparse past 2,051 tokens, dense below, padding) in bf16 and fp16,
  and from the q8_0 and fp8 caches (fp8 from sm_89), within the attention budget of test_attention_kernels against the
  oracle reading the same stored cache; batches whose shapes split the selected tokens 32, 8, 4 ways and not at all.

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

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

PAGE, SLOTS, RATIO, BUDGET = 16, 8, qsa_kernels.RATIO, qsa_kernels.BUDGET
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


def _addresses(rows: list[tuple[int, int, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """rows: (slot, first position, tokens); slot −1 gives that many padding tokens."""
    seq = [s for s, _, n in rows for _ in range(n)]
    pos = [p + i if s >= 0 else 0 for s, p, n in rows for i in range(n)]
    return torch.tensor(seq, dtype=torch.int32), torch.tensor(pos, dtype=torch.int64)


def test_index():
    lengths, cut = {3: 37, 0: 130, 6: 5}, {3: 13, 0: 66, 6: 2}        # second chunks start inside a block
    table = _table(max(lengths.values()), 0)
    units = int(table.max()) + 1
    g = torch.Generator().manual_seed(0)
    keys = {s: torch.randn(n, DK, generator=g).bfloat16() for s, n in lengths.items()}
    weight = torch.randn(DK, generator=g) * 0.1
    cos_sin = ref.cos_sin(max(lengths.values()), ROT, THETA)
    raw = torch.full((units, PAGE, DK), 7.0, dtype=torch.bfloat16)
    pooled = torch.full((units, PAGE // RATIO, DK), float("nan"))
    ours = [x.cuda() for x in (raw, pooled)]
    kernel = _kernel(qsa_kernels.INDEX, head_dim=DK, ratio=RATIO, page=PAGE, rotary_dim=ROT,
                     max_positions=len(cos_sin), eps=1e-6, dtype="bfloat16")
    for chunk in (lambda s: (0, cut[s]), lambda s: (cut[s], lengths[s])):
        rows = [(-1, 0, 2)] + [(s, *(lambda a, b: (a, b - a))(*chunk(s))) for s in lengths] + [(-1, 0, 1)]
        seq, pos = _addresses(rows)
        k = torch.cat([torch.zeros(n, DK, dtype=torch.bfloat16) if s < 0 else keys[s][p:p + n] for s, p, n in rows])
        ref.qsa_index(k, raw, pooled, table, seq, pos, weight, cos_sin, RATIO, 1e-6)
        kernel(k.cuda(), *ours, table.cuda(), seq.cuda(), pos.cuda(), weight.cuda(), cos_sin.cuda())
    assert torch.equal(ours[0].cpu(), raw)
    assert (~pooled.isnan()).sum() == sum(n // RATIO for n in lengths.values()) * DK
    report = l0.check(ours[1], pooled, torch.float32, DK)                 # NaN (never written) must match in place
    assert report.passed, report


SELECT_ROWS = [(2, 32 * 1024 - 1, 3), (3, 32 * 1024 + 12, 1), (5, 2048, 1), (1, 2047, 2), (4, 2043, 1), (7, 100, 2),
               (-1, 0, 2), (6, 0, 1), (0, 2052, 4)]      # 8,192 blocks from registers, 8,195 re-read per round


def _pooled(table: torch.Tensor, palette: int | None, seed: int) -> torch.Tensor:
    """Pooled keys for every unit; from `palette` distinct vectors when given."""
    g = torch.Generator().manual_seed(seed)
    shape = (int(table.max()) + 1, PAGE // RATIO, DK)
    if palette is None:
        return torch.randn(shape, generator=g)
    vectors = torch.randn(palette, DK, generator=g)
    return vectors[torch.randint(palette, shape[:2], generator=g)]


def _select(palette: int | None, seed: int = 0):
    table = _table(32 * 1024 + 8, seed)
    pooled = _pooled(table, palette, seed)
    seq, pos = _addresses(SELECT_ROWS)
    q = torch.randn(len(seq), HI, DK, generator=torch.Generator().manual_seed(seed + 1)).bfloat16()
    kernel = _kernel(qsa_kernels.SELECT, heads=HI, head_dim=DK, ratio=RATIO, budget=BUDGET, page=PAGE)
    args = [x.cuda() for x in (q, pooled, table, seq, pos)]
    return kernel(*args).cpu(), ref.qsa_select(q, pooled, table, seq, pos, RATIO, BUDGET), (q, pooled, table), args


def test_select_ties():
    ours, oracle, _, args = _select(palette=6)
    assert torch.equal(ours, oracle)
    small = qsa_kernel.qsa_select(*args, RATIO, BUDGET, sub_batch=3)
    assert torch.equal(small.cpu(), ours)


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
            idx = torch.arange(n) * RATIO
            keys = pooled[table[int(seq[t]), idx // PAGE].long(), (idx % PAGE) // RATIO].double()
            scores = (keys @ q[t].double().T).relu().sum(-1) / DK ** 0.5
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
        q = torch.randn(len(seq), HQ, D, generator=g).to(dtype)
        ours = kernel(q.cuda(), kv, blocks.cuda(), table.cuda(), seq.cuda(), pos.cuda()).cpu()
        oracle = ref.qsa_attend(q, stored, blocks, table, seq, pos, RATIO)
        assert (ours[seq < 0] == 0).all()
        worst = usage(ours, oracle)
        print(f"\n{layout} {dtype}, {splits} splits: worst use of the attention budget {worst:.3f}")
        assert worst <= 1.0
