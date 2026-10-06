"""Prototype Triton paged attention for decode and verify rows, reading page table and lengths on the device.

q:     [B*NQ, Hq, D]           NQ query rows per sequence (1 = decode, k+1 = verify), uniform per launch
k, v:  [pages, Hkv, P, D]      HND paged KV; the NQ new tokens are already written
pt:    [B, max_pages] int32    page table (device; the SeqTable view)
lens:  [B] int32               total KV length incl. the NQ new tokens (device; may be unknown to the host)
q8_0 KV (optional): k, v int8 plus scales ks, vs [pages, Hkv, P, D/32] fp16 (one scale per 32 elements, as in
GGUF q8_0); dequantized to bf16 in registers right after the load.

Row i of sequence b sits at position lens[b] - NQ + i and attends causally. One program handles all
NQ * group query rows of one (sequence, kv head, split), so K/V are read once per kv head. The number of
KV splits is a launch constant chosen on the host from the graph bucket (an upper bound on lengths), never
from the actual lengths; the split boundaries are computed on the device. Partials are merged by a second
kernel in a fixed order, so outputs are run-to-run deterministic.
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _dequant_q8_0(x, scale, BLOCK_N: tl.constexpr, D: tl.constexpr, OUT: tl.constexpr):
    x = tl.reshape(x.to(tl.float32), [BLOCK_N, D // 32, 32]) * scale.to(tl.float32)[:, :, None]
    return tl.reshape(x, [BLOCK_N, D]).to(OUT)


@triton.jit
def _paged_attn(Q, K, V, KS, VS, PT, LENS, O, WS_ML, WS_ACC,
                stride_qt, stride_qh, stride_kp, stride_kh, stride_ks, stride_sp, stride_sh, stride_ss, stride_pt,
                qk_scale,
                NQ: tl.constexpr, G: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, SPLITS: tl.constexpr, Q8: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)
    Hq = tl.num_programs(1) * G

    n = tl.load(LENS + b)
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
    kv_head = K + h * stride_kh
    v_head = V + h * stride_kh
    for start in range(lo, hi, BLOCK_N):
        tok = start + tl.arange(0, BLOCK_N)
        tok_ok = tok < hi
        page = tl.load(PT + b * stride_pt + tok // PAGE, mask=tok_ok, other=0)
        off = page[:, None].to(tl.int64) * stride_kp + (tok % PAGE)[:, None] * stride_ks + d[None, :]
        k = tl.load(kv_head + off, mask=tok_ok[:, None], other=0.0)
        if Q8:
            s_off = (page[:, None].to(tl.int64) * stride_sp + h * stride_sh + (tok % PAGE)[:, None] * stride_ss
                     + tl.arange(0, D // 32)[None, :])
            k = _dequant_q8_0(k, tl.load(KS + s_off, mask=tok_ok[:, None], other=0.0), BLOCK_N, D, Q.dtype.element_ty)
        qk = tl.dot(q, tl.trans(k)) * qk_scale
        visible = (tok[None, :] <= last_visible[:, None]) & tok_ok[None, :]
        qk = tl.where(visible, qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp2(qk - m_safe[:, None])
        alpha = tl.exp2(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(v_head + off, mask=tok_ok[:, None], other=0.0)
        if Q8:
            v = _dequant_q8_0(v, tl.load(VS + s_off, mask=tok_ok[:, None], other=0.0), BLOCK_N, D, Q.dtype.element_ty)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    row = (b * NQ + qi) * Hq + head                               # flat (token, head) index
    if SPLITS == 1:
        out = acc / tl.where(l_i == 0.0, 1.0, l_i)[:, None]
        tl.store(O + row[:, None] * D + d[None, :], out.to(O.dtype.element_ty), mask=row_ok[:, None])
    else:
        tl.store(WS_ML + (row * SPLITS + s) * 2, m_i, mask=row_ok)
        tl.store(WS_ML + (row * SPLITS + s) * 2 + 1, l_i, mask=row_ok)
        tl.store(WS_ACC + (row * SPLITS + s)[:, None] * D + d[None, :], acc, mask=row_ok[:, None])


@triton.jit
def _merge_splits(WS_ML, WS_ACC, O, D: tl.constexpr, SPLITS: tl.constexpr, SPLITS_P2: tl.constexpr):
    row = tl.program_id(0)
    s = tl.arange(0, SPLITS_P2)
    s_ok = s < SPLITS
    m = tl.load(WS_ML + (row * SPLITS + s) * 2, mask=s_ok, other=float("-inf"))
    l = tl.load(WS_ML + (row * SPLITS + s) * 2 + 1, mask=s_ok, other=0.0)
    m_max = tl.max(m, 0)
    w = tl.where(m == float("-inf"), 0.0, tl.exp2(m - m_max))
    d = tl.arange(0, D)
    acc = tl.load(WS_ACC + (row * SPLITS + s)[:, None] * D + d[None, :], mask=s_ok[:, None], other=0.0)
    out = tl.sum(acc * w[:, None], 0) / tl.sum(l * w, 0)
    tl.store(O + row * D + d, out.to(O.dtype.element_ty))


def pick_splits(batch: int, hkv: int, max_len_bound: int, num_sms: int, min_tokens: int = 64) -> int:
    """Host-side, from the bucket (batch, upper bound on length) only: aim for ~2 programs per SM."""
    want = -(-2 * num_sms // (batch * hkv))
    return max(1, min(want, -(-max_len_bound // min_tokens)))


# Pinned per (arch, head_dim): the most robust config of sweep_triton.py (worst case within a few % of the
# per-shape best over decode/verify shapes). No autotune at run time.
PINNED = {
    (8, 128): {"BLOCK_N": 64, "num_warps": 4, "num_stages": 1},
    (8, 256): {"BLOCK_N": 32, "num_warps": 4, "num_stages": 3},
    (12, 128): {"BLOCK_N": 64, "num_warps": 4, "num_stages": 1},
    (12, 256): {"BLOCK_N": 64, "num_warps": 4, "num_stages": 2},
}


def default_config(d: int) -> dict:
    return PINNED[(torch.cuda.get_device_capability()[0], d)]


class TritonPagedAttention:
    """Static buffers sized for one graph bucket (batch, nq, splits)."""

    def __init__(self, hq: int, hkv: int, d: int, page_size: int, batch: int, nq: int, splits: int,
                 config: dict | None = None):
        self.hq, self.hkv, self.d, self.page, self.nq, self.splits = hq, hkv, d, page_size, nq, splits
        self.g = hq // hkv
        self.block_m = max(16, triton.next_power_of_2(nq * self.g))
        self.cfg = config or default_config(d)
        rows = batch * nq * hq
        self.out = torch.empty(batch * nq, hq, d, dtype=torch.bfloat16, device="cuda")
        self.ws_ml = torch.empty(rows * splits * 2 if splits > 1 else 1, dtype=torch.float32, device="cuda")
        self.ws_acc = torch.empty(rows * splits * d if splits > 1 else 1, dtype=torch.float32, device="cuda")
        self.scale = 1.0 / math.sqrt(d) * 1.4426950408889634

    def run(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, pt: torch.Tensor,
            lens: torch.Tensor, ks: torch.Tensor | None = None, vs: torch.Tensor | None = None) -> torch.Tensor:
        B = lens.numel()
        assert k.stride() == v.stride() and k.stride(-1) == 1 and q.stride(-1) == 1
        q8 = ks is not None
        sst = ks.stride()[:3] if q8 else (0, 0, 0)
        _paged_attn[(B, self.hkv, self.splits)](
            q, k, v, ks if q8 else k, vs if q8 else v, pt, lens, self.out, self.ws_ml, self.ws_acc,
            q.stride(0), q.stride(1), k.stride(0), k.stride(1), k.stride(2), *sst, pt.stride(0),
            self.scale, NQ=self.nq, G=self.g, D=self.d, PAGE=self.page, BLOCK_M=self.block_m,
            BLOCK_N=self.cfg["BLOCK_N"], SPLITS=self.splits, Q8=q8,
            num_warps=self.cfg["num_warps"], num_stages=self.cfg["num_stages"])
        if self.splits > 1:
            _merge_splits[(B * self.nq * self.hq,)](self.ws_ml, self.ws_acc, self.out, D=self.d,
                                                     SPLITS=self.splits,
                                                     SPLITS_P2=triton.next_power_of_2(self.splits),
                                                     num_warps=4)
        return self.out


class TritonOnPagedKV(TritonPagedAttention):
    """Spike-harness adapter: run(q, kv) with kv carrying k, v, page_table, lens."""

    def run(self, q, kv):
        return super().run(q, kv.k, kv.v, kv.page_table, kv.lens)
