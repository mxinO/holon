"""Paged attention for decode (and, from M0.4, verify) rows, reading the block table and lengths on the device.

q:      [B * NQ, Hq, D]                  NQ query rows per sequence (1 = decode), uniform per launch
kv:     [units, 2, Hkv, page, D]         one layer of HND paged KV; this step's tokens are already written
table:  [slots, max_pages] int32         persistent block table, one row per request slot
rows:   [B] int32                        the slot (table row) of each batch row
lens:   [B] int32                        KV length per row including this step's tokens

Row i of sequence b sits at position lens[b] - NQ + i and attends causally. One program handles all NQ * group
query rows of one (sequence, kv head, split), so K/V are read once per kv head. The split count is a launch
constant chosen on the host from an upper bound on the lengths, never from the lengths themselves; split
boundaries are computed on the device and partials are merged in a fixed order, so the output is run-to-run
deterministic.
"""
from __future__ import annotations

import dataclasses
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_attn(Q, K, V, TABLE, ROWS, LENS, OUT, WS_ML, WS_ACC,
                stride_qt, stride_qh, stride_ku, stride_kh, stride_kt, stride_table,
                qk_scale,
                NQ: tl.constexpr, G: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, SPLITS: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)
    hq = tl.num_programs(1) * G

    n = tl.load(LENS + b)
    slot = tl.load(ROWS + b)
    m = tl.arange(0, BLOCK_M)
    qi = m // G
    head = h * G + m % G
    row_ok = m < NQ * G
    d = tl.arange(0, D)
    q = tl.load(Q + (b * NQ + qi)[:, None] * stride_qt + head[:, None] * stride_qh + d[None, :],
                mask=row_ok[:, None], other=0.0)
    last_visible = n - NQ + qi                                    # causal bound per row

    per_split = tl.cdiv(tl.cdiv(n, SPLITS), BLOCK_N) * BLOCK_N
    lo = s * per_split
    hi = tl.minimum(lo + per_split, n)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    k_head = K + h * stride_kh
    v_head = V + h * stride_kh
    for start in range(lo, hi, BLOCK_N):
        tok = start + tl.arange(0, BLOCK_N)
        tok_ok = tok < hi
        unit = tl.load(TABLE + slot * stride_table + tok // PAGE, mask=tok_ok, other=0)
        off = unit[:, None].to(tl.int64) * stride_ku + (tok % PAGE)[:, None] * stride_kt + d[None, :]
        k = tl.load(k_head + off, mask=tok_ok[:, None], other=0.0)
        qk = tl.dot(q, tl.trans(k)) * qk_scale
        visible = (tok[None, :] <= last_visible[:, None]) & tok_ok[None, :]
        qk = tl.where(visible, qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp2(qk - m_safe[:, None])
        alpha = tl.exp2(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(v_head + off, mask=tok_ok[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    row = (b * NQ + qi) * hq + head                               # flat (token, head) index
    if SPLITS == 1:
        out = acc / tl.where(l_i == 0.0, 1.0, l_i)[:, None]
        tl.store(OUT + row[:, None] * D + d[None, :], out.to(OUT.dtype.element_ty), mask=row_ok[:, None])
    else:
        tl.store(WS_ML + (row * SPLITS + s) * 2, m_i, mask=row_ok)
        tl.store(WS_ML + (row * SPLITS + s) * 2 + 1, l_i, mask=row_ok)
        tl.store(WS_ACC + (row * SPLITS + s)[:, None] * D + d[None, :], acc, mask=row_ok[:, None])


@triton.jit
def _merge_splits(WS_ML, WS_ACC, OUT, D: tl.constexpr, SPLITS: tl.constexpr, SPLITS_P2: tl.constexpr):
    row = tl.program_id(0)
    s = tl.arange(0, SPLITS_P2)
    s_ok = s < SPLITS
    m = tl.load(WS_ML + (row * SPLITS + s) * 2, mask=s_ok, other=float("-inf"))
    l = tl.load(WS_ML + (row * SPLITS + s) * 2 + 1, mask=s_ok, other=0.0)  # noqa: E741
    m_max = tl.max(m, 0)
    w = tl.where(m == float("-inf"), 0.0, tl.exp2(m - m_max))
    d = tl.arange(0, D)
    acc = tl.load(WS_ACC + (row * SPLITS + s)[:, None] * D + d[None, :], mask=s_ok[:, None], other=0.0)
    den = tl.sum(l * w, 0)
    out = tl.sum(acc * w[:, None], 0) / tl.where(den == 0.0, 1.0, den)   # length-0 (padding) rows give zeros
    tl.store(OUT + row * D + d, out.to(OUT.dtype.element_ty))


# Pinned launch configs (spike S1 sweeps): the most robust config per (sm major, head_dim), overridden per
# fingerprint class where the consumer-card sweeps found a better one; anything unswept gets DEFAULT, which is
# correct everywhere and merely untuned. No autotuning at run time.
DEFAULT = (64, 4, 1)
CONFIGS = {
    (8, 128): (64, 4, 1), (8, 256): (32, 4, 3),
    (12, 128): (64, 4, 1), (12, 256): (64, 4, 2),
}
CLASS_CONFIGS = {
    ("rtx-4090", 128): (64, 8, 1), ("rtx-4090", 256): (32, 4, 3),
    ("rtx-5080", 128): (32, 4, 3), ("rtx-5080", 256): (64, 8, 3),
    ("rtx-5090", 128): (64, 4, 3), ("rtx-5090", 256): (64, 4, 3),
}


@dataclasses.dataclass(frozen=True)
class Config:
    block_n: int
    num_warps: int
    num_stages: int

    @classmethod
    def pinned(cls, gpu_class: str | None, sm: int, head_dim: int) -> Config:
        return cls(*CLASS_CONFIGS.get((gpu_class, head_dim)) or CONFIGS.get((sm // 10, head_dim), DEFAULT))


def splits_for(batch: int, kv_heads: int, len_bound: int, num_sms: int, min_tokens: int = 64) -> int:
    """KV splits from the batch size and an upper bound on lengths only: about two programs per SM."""
    want = -(-2 * num_sms // (batch * kv_heads))
    return max(1, min(want, -(-len_bound // min_tokens)))


def paged_attention(q: torch.Tensor, kv: torch.Tensor, table: torch.Tensor, rows: torch.Tensor,
                    lens: torch.Tensor, *, nq: int, splits: int, cfg: Config) -> torch.Tensor:
    """See the module docstring; returns [B * nq, Hq, D]."""
    t, hq, d = q.shape
    hkv, page = kv.shape[2], kv.shape[3]
    b, g = rows.numel(), hq // hkv
    k, v = kv[:, 0], kv[:, 1]
    assert q.stride(-1) == 1 and k.stride(-1) == 1
    out = torch.empty(q.shape, dtype=q.dtype, device=q.device)     # the kernel writes it densely
    ws_ml = torch.empty(t * hq * splits * 2 if splits > 1 else 1, dtype=torch.float32, device=q.device)
    ws_acc = torch.empty(t * hq * splits * d if splits > 1 else 1, dtype=torch.float32, device=q.device)
    _paged_attn[(b, hkv, splits)](
        q, k, v, table, rows, lens, out, ws_ml, ws_acc,
        q.stride(0), q.stride(1), k.stride(0), k.stride(1), k.stride(2), table.stride(0),
        1.0 / math.sqrt(d) * math.log2(math.e),
        NQ=nq, G=g, D=d, PAGE=page, BLOCK_M=max(16, triton.next_power_of_2(nq * g)), BLOCK_N=cfg.block_n,
        SPLITS=splits, num_warps=cfg.num_warps, num_stages=cfg.num_stages)
    if splits > 1:
        _merge_splits[(t * hq,)](ws_ml, ws_acc, out, D=d, SPLITS=splits,
                                 SPLITS_P2=triton.next_power_of_2(splits), num_warps=4)
    return out
