"""The op contracts of QSA, Qwen3.8-Flash-Next's indexed sparse attention (docs/dev/v1-flash-next.md D6). Their oracles
are in `reference.py`; kernels come from kernel plugins. Kernel instances are stateless across steps and safe to call
inside CUDA graphs. (The unit, `std.mixers.qsa@1`, lands with FN's core seams; these contracts stand alone.)

**Blocks.** A sequence's tokens form blocks of `ratio` (4) consecutive positions; block b holds positions
ratio·b … ratio·b + ratio − 1. A query at position p sees n = (p + 1) // ratio complete blocks and, when
(p + 1) % ratio ≠ 0, the open block n (its positions up to p).

**State** (two paged specs of the mixer, in separate page groups, `page % ratio == 0`): the gqa KV cache (`"kv"`, any
of `std.mixers.gqa`'s layouts) and the index keys (`"idx"`): per unit, `raw` [units, page, Dk] in the model dtype —
every token's index key as projected (unnormed, unrotated) — and `pooled` [units, page / ratio, Dk] float32, one key per
complete block. Block b of a sequence lives in its page (ratio·b) // page, at (ratio·b % page) / ratio. Within a unit
both are dense rows; units may be strided (a layer's segment of a unit holds its raw keys, then its pooled keys):
kernels address units by `raw.stride(0)` and `pooled.stride(0)`.

**Addressing** (every op): `seq` [T] int32, each token's row of `table` [slots, max_pages] (int32 unit ids; −1: a
padding token, which reads nothing and writes nothing), and `positions` [T] int64, its position in that sequence.
**Precondition:** each sequence's tokens are consecutive rows of the call, in position order (the batch layout).

- `qsa_index@1`: instance `(k [T, Dk] (its rows may be strided), raw, pooled, table, seq, positions, weight [Dk],
  cos_sin) -> None`: stores each
  token's k into `raw` at its position, then, for every token that completes a block (p % ratio = ratio − 1), pools that
  block **from the raw keys as stored after the call** (a position the call stores reads the value it stores; never
  values kept from an earlier step, so a block completed by rejected drafts is rewritten when its positions are
  appended again): `pooled[b] = rope(norm(bf16(mean_f32(raw[ratio·b … ratio·b + ratio − 1]))))`:
  the mean in float32 rounded once to the model dtype, then a zero-centred RMSNorm (x · rsqrt(mean(x²) + eps) ·
  (1 + weight)) and a neox rotation of the first `rotary_dim` dims at position ratio·b (`cos_sin`: `rope@1`'s table),
  both in float32, stored float32. Facts: head_dim, ratio, page, rotary_dim, max_positions, eps, dtype.
- `qsa_select@1`: instance `(q [T, Hi, Dk], pooled, table, seq, positions) -> blocks [T, budget / ratio + 1] int32`:
  per query (q normed and rotated by the unit, any float dtype), the scores of its n complete blocks
  `s_b = Σ_h relu(⟨q_h, pooled_b⟩) / √Dk` in float32 with IEEE products, and its top min(budget / ratio, n) blocks, ties
  to the lower block (D2's rule); written ascending, then the open block n when there is one, then −1. A query with
  n ≤ budget / ratio thus selects every block (dense). Rows of padding tokens are all −1. Score workspaces are
  sub-batched (at most 512 queries per pass). **Shape-invariant:** a query's scores, and so its selection, are the same
  bits alone and inside any call, whatever its other rows and their number. Facts: heads, head_dim, ratio, budget,
  page.
- `qsa_attend@1` (prefill, decode and verify rows alike): instance `(q [T, Hq, D], kv, blocks, table, seq, positions,
  gate [T, Hq, D] | None) -> [T, Hq, D]` of q's dtype: per query, softmax(q k^T / √D) v over exactly the tokens of its
  `blocks` (all `ratio` of a complete block; the open block's up to p), kv heads shared by Hq / Hkv query heads,
  float32 softmax and sums; with `gate` (the attention's output gate, any row and head strides), times σ(gate) in
  float32 before the one rounding, so the gate costs no launch of its own.
  **Every `kv` layout of `std.mixers.gqa` is read in the kernel**, `kv.hnd.q8_0` (values int8 and fp16 scales per 32)
  included, so a q8_0 deployment never dequantizes the cache first (panel C, step 0). A padding token's output is 0.
  Facts: heads, kv_heads, head_dim, ratio, width (budget / ratio + 1), page, layout, q_dtype, kv_dtype.

A step whose rows all have host-known lengths ≤ (budget / ratio) · ratio + ratio − 1 (2,051) runs the dense gqa ops
instead (D6).
"""
from __future__ import annotations

INDEX, SELECT, ATTEND = "qsa_index@1", "qsa_select@1", "qsa_attend@1"


def width(budget: int, ratio: int) -> int:
    """`qsa_select@1`'s output width: the selected complete blocks and the open one."""
    return budget // ratio + 1


def dense_limit(budget: int, ratio: int) -> int:
    """The longest sequence whose every query selects all its blocks (2,051 for Flash-Next): D6's dense rule."""
    return budget + ratio - 1
