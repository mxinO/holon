"""L0 for GLM-5.3-Flash's DSA kernels in `kernels.qsa_triton` (docs/v1-glm.md D5, D12, acceptance 3) against
std.mixers.qsa's oracles, on GLM's shapes (indexer 32 heads × 128 with signed head weights; MLA 64 heads over the
512-wide latent, scale 1/16; ratio 4, budget 512) and the tiny GPU fixture's (indexer 4 × 64; MLA 2 heads over 128),
through the build context.

- `qsa_select@1` with head weights: exactly the oracle where its float32 scores have no near-ties (pooled keys from a
  few vectors, one of them zero: large ties the lower-block rule decides, at 0.0 and among negative scores); with
  continuous keys a valid top-k of the oracle's float64 scores within a recorded ε (near-tie swaps at the top-k edge
  only); a query's scores the same bits alone and in a 128-row call.
- `dsa_index@1`: prefill in two chunks per sequence (the second starting inside a block), padding tokens; raw keys
  and gates bitwise, pooled keys within `l0.check(f32, ratio)`; units laid out as the unit views its pool (raw keys,
  gates, pooled keys per unit).
- `mla_attend@1` for prefill, decode and verify rows (sparse past 515 tokens, every block below), padding, q in f32
  and bf16: relative L2 per row ≤ 1.5e-3 (q̃ as one bfloat16 part: ≈ 9e-4 measured; step 7 judges it on the real
  model, docs/v1-glm.md §13) against the oracle on the same stored latent; a plant (the open block reading one token
  more) fails it. A row's output is the same bits alone (a program a chunk, then the merge), beside a long row (one
  program over its chunks) and split across calls (the dense path is the sparse kernel, sub-batching changes no bit).
- Decode (index, select, attend) captured in a CUDA graph replays to the eager bits.

Skips without a GPU.
"""
import numpy as np
import pytest
import torch
from holon.core.build import BuildContext
from holon.core.plugins import PluginSet, discover
from holon_judge import l0
from holon_plugins.kernels.qsa_triton import kernel as qsa_kernel
from holon_std.mixers.qsa import kernels as qsa_kernels
from holon_std.mixers.qsa import reference as ref

pytestmark = [pytest.mark.l0, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")]

SLOTS, RATIO, BUDGET = 8, 4, 512
WIDTH = qsa_kernels.width(BUDGET, RATIO)
GLM, TINY = (32, 128, 64, 512), (4, 64, 2, 128)          # (indexer heads, Dk, MLA heads, latent)
EPS_SELECT = 2e-6       # continuous keys: a chosen block's oracle score ≥ an unchosen one's − ε · max |score|


def _kernel(op: str, **facts):
    ctx = BuildContext(PluginSet(discover()), torch.device("cuda"),
                       options={"kernels": {op.split("@")[0]: "kernels.qsa_triton"}})
    return ctx.select_kernel(op, site="test", **facts).impl(ctx, **facts)


def _table(max_len: int, page: int, seed: int) -> torch.Tensor:
    """[SLOTS, pages] distinct unit ids, shuffled, with a few units no slot uses."""
    pages = -(-max_len // page)
    units = np.random.default_rng(seed).permutation(SLOTS * pages + 5)[: SLOTS * pages]
    return torch.from_numpy(units.astype(np.int32)).view(SLOTS, pages)


def _addresses(rows: list[tuple[int, int, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """rows: (slot, first position, tokens); slot −1 gives that many padding tokens."""
    seq = [s for s, _, n in rows for _ in range(n)]
    pos = [p + i if s >= 0 else 0 for s, p, n in rows for i in range(n)]
    return torch.tensor(seq, dtype=torch.int32), torch.tensor(pos, dtype=torch.int64)


# ---------------------------------------------------------------------------------------------------------- selection

SELECT_ROWS = [(2, 32 * 1024 - 1, 3), (3, 32 * 1024 + 12, 1), (5, 512, 1), (1, 511, 2), (4, 515, 1), (7, 100, 2),
               (-1, 0, 2), (6, 0, 1), (0, 600, 4), (1, 2400, 1)]
PAGE_SELECT = 16


def _pooled(table: torch.Tensor, dk: int, palette: int | None, seed: int, zero: float = 0.85) -> torch.Tensor:
    """Pooled keys for every unit; with a palette, from that many distinct vectors, the first a zero vector (score
    ±0.0) drawn for a `zero` share of the blocks."""
    g = torch.Generator().manual_seed(seed)
    shape = (int(table.max()) + 1, PAGE_SELECT // RATIO, dk)
    if palette is None:
        return torch.randn(shape, generator=g)
    vectors = torch.cat((torch.zeros(1, dk), torch.randn(palette - 1, dk, generator=g)))
    weights = torch.tensor([zero] + [(1 - zero) / (palette - 1)] * (palette - 1))
    return vectors[torch.multinomial(weights, shape[0] * shape[1], replacement=True, generator=g).view(shape[:2])]


def _weights(t: int, hi: int, seed: int) -> torch.Tensor:
    """Signed head weights [t, hi] (÷ √hi, as the unit scales them): every third row all negative, so its k-th score
    is negative."""
    w = torch.randn(t, hi, generator=torch.Generator().manual_seed(seed)) * hi ** -0.5
    w[::3] = -w[::3].abs()
    return w


def _oracle_scores(q, w, pooled, table, seq: int, n: int) -> torch.Tensor:
    first = torch.arange(n) * RATIO
    keys = pooled[table[seq, first // PAGE_SELECT].long(), (first % PAGE_SELECT) // RATIO].double()
    return ((keys @ q.double().T).relu() * w.double()).sum(-1) / q.shape[-1] ** 0.5


def _select(shape, palette: int | None, seed: int, zero: float = 0.85):
    hi, dk = shape[:2]
    table = _table(32 * 1024 + 16, PAGE_SELECT, seed)
    pooled = _pooled(table, dk, palette, seed, zero)
    seq, pos = _addresses(SELECT_ROWS)
    q = torch.randn(len(seq), hi, dk, generator=torch.Generator().manual_seed(seed + 1)).bfloat16()
    w = _weights(len(seq), hi, seed + 2)
    kernel = _kernel(qsa_kernels.SELECT, heads=hi, head_dim=dk, ratio=RATIO, budget=BUDGET, page=PAGE_SELECT,
                     head_weights=True)
    args = [x.cuda() for x in (q, pooled, table, seq, pos, w)]
    oracle = ref.qsa_select(q, pooled, table, seq, pos, RATIO, BUDGET, w)
    return kernel(*args).cpu(), oracle, (q, w, pooled, table), args


@pytest.mark.parametrize("shape", [GLM, TINY], ids=["glm", "tiny"])
@pytest.mark.parametrize("zero", [0.85, 0.0])
def test_select_weights_ties(shape, zero):
    """With the zero vector on most blocks the all-negative row's boundary is a tie at 0.0 (−0.0 and +0.0 among it);
    without it, among negative scores."""
    ours, oracle, (q, w, pooled, table), args = _select(shape, palette=6, seed=0, zero=zero)
    assert torch.equal(ours, oracle)
    small = qsa_kernel.qsa_select(*args[:5], RATIO, BUDGET, 3, args[5])          # sub-batches of 3 rows
    assert torch.equal(small.cpu(), ours)
    seq, pos = _addresses(SELECT_ROWS)
    k = BUDGET // RATIO
    s = _oracle_scores(q[0], w[0], pooled, table, int(seq[0]), (int(pos[0]) + 1) // RATIO).sort(descending=True)
    assert s.values[k - 1] == s.values[k]                # 8,192 blocks, all-negative weights: a tie at the boundary
    assert s.values[k - 1] == 0.0 if zero else s.values[k - 1] < 0


@pytest.mark.parametrize("shape", [GLM, TINY], ids=["glm", "tiny"])
def test_select_weights_continuous(shape):
    ours, oracle, (q, w, pooled, table), _ = _select(shape, palette=None, seed=1)
    seq, pos = _addresses(SELECT_ROWS)
    top, worst, swaps = BUDGET // RATIO, 0.0, 0
    for t in range(len(seq)):
        n = (int(pos[t]) + 1) // RATIO if seq[t] >= 0 else 0
        k = min(n, top)
        chosen = ours[t, :k].long()
        assert torch.equal(ours[t, k:k + 1], oracle[t, k:k + 1]) and (ours[t, k + 1:] == -1).all()
        assert (chosen.diff() > 0).all() and (chosen < n).all()
        if n > top:
            scores = _oracle_scores(q[t], w[t], pooled, table, int(seq[t]), n)
            mask = torch.zeros(n, dtype=torch.bool)
            mask[chosen] = True
            gap = float((scores[~mask].max() - scores[mask].min()) / scores.abs().max())
            worst, swaps = max(worst, gap), swaps + len(set(chosen.tolist()) ^ set(oracle[t, :k].tolist())) // 2
    print(f"\nselect {shape[:2]}: worst near-tie gap {worst:.2e} of max |score| (ε {EPS_SELECT:.0e}), {swaps} swaps")
    assert worst <= EPS_SELECT and swaps <= 4


def test_select_weights_shape_invariant():
    """A query's weighted scores (and selection) are the same bits alone and in a 128-row call."""
    hi, dk = GLM[:2]
    g = torch.Generator().manual_seed(7)
    table = _table(32 * 1024 + 16, PAGE_SELECT, 7).cuda()
    pooled = _pooled(table.cpu(), dk, None, 7).cuda()
    seq = torch.randint(SLOTS, (128,), generator=g).int().cuda()
    pos = torch.randint(100, 32 * 1024, (128,), generator=g).cuda()
    q = torch.randn(128, hi, dk, generator=g).bfloat16().cuda()
    w = _weights(128, hi, 8).cuda()
    width = table.shape[1] * PAGE_SELECT // RATIO
    together = torch.full((128, width), float("nan"), device="cuda")
    qsa_kernel.scores(q, pooled, table, seq, pos, RATIO, together, w)
    blocks = qsa_kernel.qsa_select(q, pooled, table, seq, pos, RATIO, BUDGET, 512, w)
    for t in range(0, 128, 9):
        alone = torch.full((1, width), float("nan"), device="cuda")
        qsa_kernel.scores(q[t:t + 1], pooled, table, seq[t:t + 1], pos[t:t + 1], RATIO, alone, w[t:t + 1])
        n = (int(pos[t]) + 1) // RATIO
        assert torch.equal(alone[0, :n], together[t, :n])
        assert torch.equal(qsa_kernel.qsa_select(q[t:t + 1], pooled, table, seq[t:t + 1], pos[t:t + 1], RATIO,
                                                 BUDGET, 512, w[t:t + 1])[0], blocks[t])


# -------------------------------------------------------------------------------------------------------------- index

def _segments(units: int, page: int, dk: int) -> tuple[torch.Tensor, ...]:
    """raw, gates [units, page, Dk] bf16 and pooled [units, page / ratio, Dk] f32 on the GPU, as `DSAMLA._idx_layer`
    views its pool: one byte segment per unit, so units are strided."""
    half = page * dk * 2
    seg = torch.empty(units, 2 * half + page // RATIO * dk * 4, dtype=torch.uint8, device="cuda")
    view = lambda a, b: seg[:, a:b].view(torch.bfloat16).view(-1, page, dk)      # noqa: E731
    return view(0, half), view(half, 2 * half), seg[:, 2 * half:].view(torch.float32).view(-1, page // RATIO, dk)


@pytest.mark.parametrize("shape", [GLM, TINY], ids=["glm", "tiny"])
@pytest.mark.parametrize("page", [16, 64])
def test_index(shape, page):
    dk = shape[1]
    lengths, cut = {3: 37, 0: 130, 6: 5}, {3: 13, 0: 66, 6: 2}        # second chunks start inside a block
    table = _table(max(lengths.values()), page, 0)
    units = int(table.max()) + 1
    g = torch.Generator().manual_seed(0)
    keys = {s: torch.randn(n, dk, generator=g).bfloat16() for s, n in lengths.items()}
    gate = {s: (torch.randn(n, dk, generator=g) * 3).bfloat16() for s, n in lengths.items()}
    ape = torch.randn(RATIO, dk, generator=g)
    raw = torch.full((units, page, dk), 7.0, dtype=torch.bfloat16)
    gates = torch.full((units, page, dk), -7.0, dtype=torch.bfloat16)
    pooled = torch.full((units, page // RATIO, dk), float("nan"))
    ours = _segments(units, page, dk)
    for view, init in zip(ours, (raw, gates, pooled), strict=True):
        view.copy_(init)
    kernel = _kernel(qsa_kernels.DSA_INDEX, head_dim=dk, ratio=RATIO, page=page, dtype="bfloat16")
    for chunk in (lambda s: (0, cut[s]), lambda s: (cut[s], lengths[s])):
        rows = [(-1, 0, 2)] + [(s, *(lambda a, b: (a, b - a))(*chunk(s))) for s in lengths] + [(-1, 0, 1)]
        seq, pos = _addresses(rows)
        k, gt = (torch.cat([torch.zeros(n, dk, dtype=torch.bfloat16) if s < 0 else src[s][p:p + n]
                            for s, p, n in rows]) for src in (keys, gate))
        ref.dsa_index(k, gt, raw, gates, pooled, ape, table, seq, pos, RATIO)
        kernel(k.cuda(), gt.cuda(), *ours, ape.cuda(), table.cuda(), seq.cuda(), pos.cuda())
    assert torch.equal(ours[0].cpu(), raw) and torch.equal(ours[1].cpu(), gates)
    assert (~pooled.isnan()).sum() == sum(n // RATIO for n in lengths.values()) * dk
    report = l0.check(ours[2], pooled, torch.float32, RATIO)             # NaN (never written) must match in place
    print(f"\ndsa_index {dk}, page {page}: {report}")
    assert report.passed, report


# ------------------------------------------------------------------------------------------------------------ attend

PAGE_ATTEND = 16
# decode at 32K (sparse), verify rows, a prefill chunk below the dense limit and one crossing it, padding
ATTEND_ROWS = [(1, 32 * 1024 - 3, 1), (-1, 0, 1), (6, 9, 1), (5, 4093, 3), (0, 515, 1), (2, 511, 6), (3, 0, 37),
               (4, 480, 64)]


def _blocks(seq: torch.Tensor, pos: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """Every block below the budget (`all_blocks`), a random ascending subset above it, then the open block."""
    blocks = torch.full((len(seq), WIDTH), -1, dtype=torch.int32)
    for t in range(len(seq)):
        n = (int(pos[t]) + 1) // RATIO
        if seq[t] < 0:
            continue
        chosen = torch.arange(n) if n <= WIDTH - 1 else torch.randperm(n, generator=g)[: WIDTH - 1].sort().values
        blocks[t, :len(chosen)] = chosen.int()
        if (int(pos[t]) + 1) % RATIO:
            blocks[t, len(chosen)] = n
    return blocks


def _attend_case(shape, q_dtype, rows=ATTEND_ROWS, seed=2):
    heads, dc = shape[2:]
    table = _table(32 * 1024, PAGE_ATTEND, seed)
    units = int(table.max()) + 1
    g = torch.Generator().manual_seed(seed)
    kv = (torch.randn(units, PAGE_ATTEND, dc, generator=g) * 0.7).bfloat16()
    seq, pos = _addresses(rows)
    blocks = _blocks(seq, pos, g)
    q = (torch.randn(len(seq), heads, dc, generator=g) * 0.4).to(q_dtype)          # q̃: scores of a few units
    kernel = _kernel(qsa_kernels.MLA_ATTEND, heads=heads, latent=dc, ratio=RATIO, width=WIDTH, page=PAGE_ATTEND,
                     scale=1 / 16, q_dtype=str(q_dtype)[6:], kv_dtype="bfloat16")
    args = [x.cuda() for x in (q, kv, blocks, table, seq, pos)]
    return kernel, args, (q, kv, blocks, table, seq, pos)


def _rel_l2(ours: torch.Tensor, oracle: torch.Tensor) -> torch.Tensor:
    """Per row, ‖ours − oracle‖ / ‖oracle‖ over its heads and latent."""
    d = (ours.double() - oracle.double()).flatten(1).norm(dim=1)
    return d / oracle.double().flatten(1).norm(dim=1).clamp_min(1e-30)


@pytest.mark.parametrize("shape", [GLM, TINY], ids=["glm", "tiny"])
@pytest.mark.parametrize("q_dtype", [torch.float32, torch.bfloat16])
def test_attend(shape, q_dtype):
    kernel, args, (q, kv, blocks, table, seq, pos) = _attend_case(shape, q_dtype)
    ours = kernel(*args).cpu()
    assert ours.dtype == q_dtype and (ours[seq < 0] == 0).all()
    oracle = ref.mla_attend(q.float(), kv, blocks, table, seq, pos, RATIO, 1 / 16)
    rel = _rel_l2(ours, oracle)[seq >= 0]
    floor = _rel_l2(oracle.to(q_dtype), oracle)[seq >= 0].max()      # the output's own rounding
    print(f"\nmla_attend {shape[2:]} q {q_dtype}: rel L2 max {rel.max():.2e} (its rounding {floor:.2e})")
    assert rel.max() <= 1.5e-3 + floor
    planted = ref.mla_attend(q.float(), kv, blocks, table, seq, pos + 1, RATIO, 1 / 16)   # the open block, one more
    assert _rel_l2(ours, planted)[seq >= 0].max() > 1e-2


def test_attend_row_bits_alone_and_split():
    """A row's output: the same bits alone (a call of few rows: a program a chunk, then `_merge`), beside a long
    prefill chunk (one program walking the row's chunks), and with the call split in two (the dense path is the sparse
    kernel over every block; sub-batching changes no bit)."""
    from holon_plugins.kernels.qsa_triton import mla

    kernel, args, (_, _, _, _, seq, _) = _attend_case(GLM, torch.float32)
    groups = GLM[2] // mla.HEADS
    assert groups < mla.SPLIT_BELOW <= len(seq) * groups and WIDTH * RATIO > 2 * mla.CHUNK    # both paths, 3 chunks
    together = kernel(*args)
    q, kv, blocks, table, seq_, pos = args
    for t in (0, 2, 5, len(seq) - 1):                    # decode, a short prefill row, verify, a long chunk's last
        alone = kernel(q[t:t + 1], kv, blocks[t:t + 1], table, seq_[t:t + 1], pos[t:t + 1])
        assert torch.equal(alone[0], together[t])
    cut = len(seq) // 3
    split = torch.cat([kernel(q[a:b], kv, blocks[a:b], table, seq_[a:b], pos[a:b])
                       for a, b in ((0, cut), (cut, len(seq)))])
    assert torch.equal(split, together)


def test_decode_graph():
    """Decode's three DSA kernels captured in a CUDA graph replay to the eager bits."""
    hi, dk, heads, dc = GLM
    page, slots = 16, 2
    table = _table(4096, page, 3).cuda()
    units = int(table.max()) + 1
    g = torch.Generator().manual_seed(3)
    raw, gates, pooled = _segments(units, page, dk)
    for v in (raw, gates, pooled):
        v.copy_(torch.randn(v.shape, generator=g))
    latent = (torch.randn(units, page, dc, generator=g) * 0.7).bfloat16().cuda()
    seq = torch.arange(slots, dtype=torch.int32).cuda()
    pos = torch.tensor([4091, 2050]).cuda()
    k, gate = (torch.randn(slots, dk, generator=g).bfloat16().cuda() for _ in range(2))
    q_i = torch.randn(slots, hi, dk, generator=g).bfloat16().cuda()
    w = _weights(slots, hi, 4).cuda()
    q = torch.randn(slots, heads, dc, generator=g).cuda() * 0.4
    ape = torch.randn(RATIO, dk, generator=g).cuda()
    index = _kernel(qsa_kernels.DSA_INDEX, head_dim=dk, ratio=RATIO, page=page, dtype="bfloat16")
    select = _kernel(qsa_kernels.SELECT, heads=hi, head_dim=dk, ratio=RATIO, budget=BUDGET, page=page,
                     head_weights=True)
    attend = _kernel(qsa_kernels.MLA_ATTEND, heads=heads, latent=dc, ratio=RATIO, width=WIDTH, page=page,
                     scale=1 / 16, q_dtype="float32", kv_dtype="bfloat16")

    def step():
        index(k, gate, raw, gates, pooled, ape, table, seq, pos)
        return attend(q, latent, select(q_i, pooled, table, seq, pos, w), table, seq, pos)

    eager = step().clone()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        step()                                           # warm (compile) outside the capture
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = step()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)
