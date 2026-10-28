"""FlashInfer call paths under test. Each builder plans on the host and returns (run(layer) -> out, info).

All paged paths use HND layout ([pages, hkv, page_size, d]) and the CUDA-graph wrappers, i.e. the
configuration a FULL decode/verify graph would use.
"""

import math

import torch
import flashinfer
import triton
import triton.language as tl

from common import Geom, PagedKV, csr_from_table

WORKSPACE_BYTES = 256 << 20
_workspace = {}


def workspace(tag: str) -> torch.Tensor:
    # One float workspace per wrapper instance (FlashInfer requires zero-init on first use).
    if tag not in _workspace:
        _workspace[tag] = torch.zeros(WORKSPACE_BYTES, dtype=torch.uint8, device="cuda")
    return _workspace[tag]


def scales(kv: PagedKV) -> dict:
    return {"k_scale": 1.0, "v_scale": 1.0} if kv.k.dtype != torch.bfloat16 else {}


class _FIPaged:
    """Common setup for the paged wrappers.

    meta="host":   CUDA-graph wrapper; plan() runs on the host every step and copies CSR metadata into its
                   static buffers (FlashInfer chooses split-KV).
    meta="device": non-graph wrapper planned once with disable_split_kv=True, so the plan does not depend on
                   KV lengths; it aliases our device CSR buffers, which prepare() rewrites inside the graph
                   from the device page table and lengths. (A CUDA-graph wrapper cannot be used here: with
                   disable_split_kv it launches padded CTAs that read unwritten request indices; see graph_test.)
    meta="graph_nosplit": the CUDA-graph wrapper with disable_split_kv=True (kept to demonstrate that hazard).
    """

    def __init__(self, g: Geom, kv: PagedKV, meta: str):
        B, maxp = kv.page_table.shape
        self.g, self.kv, self.meta = g, kv, meta
        self.indptr = torch.zeros(B + 1, dtype=torch.int32, device="cuda")
        self.indices = torch.zeros(B * maxp, dtype=torch.int32, device="cuda")
        self.last = torch.zeros(B, dtype=torch.int32, device="cuda")
        self.graph_wrapper = meta != "device"
        self.extra = {"disable_split_kv": True} if meta != "host" else {}

    def csr(self, kv: PagedKV | None):
        if self.meta == "device":
            self.prepare(kv)
            return self.indptr, self.indices, self.last
        return csr_from_table(kv or self.kv)

    def prepare(self, kv: PagedKV | None = None):
        write_csr_on_device(kv or self.kv, self.indptr, self.indices, self.last)

    def run(self, q: torch.Tensor, kv: PagedKV) -> torch.Tensor:
        return self.w.run(q, (kv.k, kv.v), out=self.out, **scales(kv))


class FIDecode(_FIPaged):
    """BatchDecodeWithPagedKVCacheWrapper (use_tensor_cores selects the FA2 prefill kernel)."""

    def __init__(self, g: Geom, kv: PagedKV, use_tc: bool, meta: str = "host", tag: str = "decode"):
        super().__init__(g, kv, meta)
        bufs = dict(paged_kv_indptr_buffer=self.indptr, paged_kv_indices_buffer=self.indices,
                    paged_kv_last_page_len_buffer=self.last) if self.graph_wrapper else {}
        self.w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace(tag), "HND", use_cuda_graph=self.graph_wrapper, use_tensor_cores=use_tc, **bufs)
        self.out = torch.empty(kv.lens.numel(), g.hq, g.d, dtype=torch.bfloat16, device="cuda")

    def plan(self, kv: PagedKV | None = None, csr=None):
        indptr, indices, last = csr or self.csr(kv)
        self.w.plan(indptr, indices, last, self.g.hq, self.g.hkv, self.g.d, self.kv.page_size,
                    q_data_type=torch.bfloat16, kv_data_type=self.kv.k.dtype, **self.extra)


class FIPagedPrefill(_FIPaged):
    """BatchPrefillWithPagedKVCacheWrapper with `nq` causal query rows per sequence (verify rows or prefill)."""

    def __init__(self, g: Geom, kv: PagedKV, nq: int, meta: str = "host", graph: bool = True,
                 tag: str = "prefill"):
        super().__init__(g, kv, meta)
        B = kv.lens.numel()
        self.nq = nq
        self.graph_wrapper = self.graph_wrapper and graph
        self.qo_indptr = torch.arange(B + 1, dtype=torch.int32, device="cuda") * nq
        self.qo_indptr_host = self.qo_indptr.cpu()
        bufs = dict(qo_indptr_buf=self.qo_indptr, paged_kv_indptr_buf=self.indptr,
                    paged_kv_indices_buf=self.indices,
                    paged_kv_last_page_len_buf=self.last) if self.graph_wrapper else {}
        self.w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace(tag), "HND",
                                                                use_cuda_graph=self.graph_wrapper, **bufs)
        self.out = torch.empty(B * nq, g.hq, g.d, dtype=torch.bfloat16, device="cuda")

    def plan(self, kv: PagedKV | None = None, csr=None):
        indptr, indices, last = csr or self.csr(kv)
        self.w.plan(self.qo_indptr_host, indptr, indices, last, self.g.hq, self.g.hkv, self.g.d, self.kv.page_size,
                    causal=True, q_data_type=torch.bfloat16, kv_data_type=self.kv.k.dtype, **self.extra)


class FIRaggedPrefill:
    """BatchPrefillWithRaggedKVCacheWrapper: first chunk, K/V given as contiguous [tokens, hkv, d]."""

    def __init__(self, g: Geom, lens: list[int], tag: str = "ragged"):
        self.g = g
        indptr = torch.zeros(len(lens) + 1, dtype=torch.int32)
        indptr[1:] = torch.cumsum(torch.tensor(lens), 0)
        self.w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace(tag), "NHD")
        self.w.plan(indptr, indptr, g.hq, g.hkv, g.d, causal=True, q_data_type=torch.bfloat16)
        self.out = torch.empty(int(indptr[-1]), g.hq, g.d, dtype=torch.bfloat16, device="cuda")

    def run(self, q, k, v):
        return self.w.run(q, k, v, out=self.out)


@triton.jit
def _csr_kernel(LENS, PT, INDPTR, INDICES, LAST, B, stride_pt, PAGE: tl.constexpr, B_P2: tl.constexpr,
                BLOCK: tl.constexpr):
    b = tl.program_id(0)
    i = tl.arange(0, B_P2)
    npages_all = (tl.load(LENS + i, mask=i < B, other=0) + PAGE - 1) // PAGE
    start = tl.sum(tl.where(i < b, npages_all, 0), 0)          # exclusive prefix sum, B is small
    n = tl.load(LENS + b)
    npages = (n + PAGE - 1) // PAGE
    tl.store(INDPTR + b + 1, start + npages)
    if b == 0:
        tl.store(INDPTR, 0)
    tl.store(LAST + b, n - (npages - 1) * PAGE)
    for j in range(0, npages, BLOCK):
        cols = j + tl.arange(0, BLOCK)
        tl.store(INDICES + start + cols, tl.load(PT + b * stride_pt + cols, mask=cols < npages), mask=cols < npages)


def write_csr_on_device(kv: PagedKV, indptr: torch.Tensor, indices: torch.Tensor, last: torch.Tensor):
    """Graph-capturable: FlashInfer CSR metadata from the device page table and device lengths (one launch)."""
    B = kv.lens.numel()
    _csr_kernel[(B,)](kv.lens, kv.page_table, indptr, indices, last, B, kv.page_table.stride(0),
                      PAGE=kv.page_size, B_P2=triton.next_power_of_2(B), BLOCK=256)


class XQA:
    """flashinfer.xqa: device page table + device seq_lens, no plan (SM90/100/120 only)."""

    def __init__(self, g: Geom, kv: PagedKV, nq: int, tag: str = "xqa"):
        B = kv.lens.numel()
        self.g, self.nq = g, nq
        ws = workspace(tag)
        self.sem = ws[: 8 << 20].view(torch.uint32)
        self.scratch = ws[8 << 20:]
        self.sem.zero_()
        self.seq_lens = torch.empty(B, 1, dtype=torch.uint32, device="cuda")
        self.out = torch.empty(B * nq, g.hq, g.d, dtype=torch.bfloat16, device="cuda")
        self.mask = None
        if nq > 1:  # causal among the nq new tokens, bit-packed per row
            words = (nq + 31) // 32
            bits = (torch.arange(32 * words)[None, :] <= torch.arange(nq)[:, None]).to(torch.int64)
            packed = (bits.view(nq, words, 32) << torch.arange(32)).sum(-1).to(torch.uint32)
            self.mask = packed.view(torch.uint16).reshape(1, nq, 2 * words).expand(B, -1, -1).contiguous().cuda()

    def run(self, q: torch.Tensor, kv: PagedKV) -> torch.Tensor:
        # uint32 lengths: copy inside the graph from the int32 SeqTable view.
        self.seq_lens.copy_(kv.lens.view(-1, 1).view(torch.uint32))
        B = kv.lens.numel()
        qv = q.view(B, 1, *((self.nq,) if self.nq > 1 else ()), self.g.hq, self.g.d)
        kv_scale = 1.0
        flashinfer.xqa(qv, kv.k, kv.v, kv.page_table, self.seq_lens, self.out.view(qv.shape), self.scratch, self.sem,
                       self.g.hkv, kv.page_size, q_scale=1.0, kv_scale=kv_scale, kv_layout="HND",
                       q_seq_len=self.nq, mask=self.mask, enable_pdl=False)
        return self.out


def sm_scale(g: Geom) -> float:
    return 1.0 / math.sqrt(g.d)
