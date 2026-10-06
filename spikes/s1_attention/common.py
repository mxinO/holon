"""Shared helpers for spike S1: geometries, paged KV construction, torch reference, graph timing."""

import json
import os
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = HERE / "out"


@dataclass(frozen=True)
class Geom:
    name: str
    hq: int
    hkv: int
    d: int

    @property
    def group(self) -> int:
        return self.hq // self.hkv


# Full-attention geometries (HF configs in ~/.cache/huggingface/hub).
GEOMS = {
    "qwen3_8b": Geom("qwen3_8b", 32, 8, 128),          # group 4
    "qwen3_5_9b": Geom("qwen3_5_9b", 16, 4, 256),      # group 4
    "qwen3_8_27b": Geom("qwen3_8_27b", 24, 4, 256),    # group 6
}


def fingerprint() -> dict:
    p = torch.cuda.get_device_properties(0)
    import flashinfer
    import triton
    return {
        "gpu": p.name, "sm": f"{p.major}{p.minor}", "num_sms": p.multi_processor_count,
        "host": platform.node(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "triton": triton.__version__, "flashinfer": flashinfer.__version__,
        "driver": subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader", "-i",
                                  os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]],
                                 capture_output=True, text=True).stdout.strip(),
    }


def class_name() -> str:
    """Fingerprint class of GPU 0, by product name (5080/5090/PRO 6000 all report sm_120)."""
    name = torch.cuda.get_device_properties(0).name
    for key, cls in (("A5000", "a5000"), ("RTX PRO 6000", "rtx-pro-6000"), ("RTX 4090", "rtx-4090"),
                     ("RTX 5080", "rtx-5080"), ("RTX 5090", "rtx-5090")):
        if key in name:
            return cls
    major, minor = torch.cuda.get_device_capability(0)
    return f"sm{major}{minor}"


def save(name: str, payload: dict) -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}_{class_name()}.json"
    path.write_text(json.dumps({"fingerprint": fingerprint(), **payload}, indent=1))
    return path


@dataclass
class PagedKV:
    """One layer of HND paged KV ([pages, hkv, page_size, d]) plus a per-seq page table and lengths."""
    k: torch.Tensor
    v: torch.Tensor
    page_table: torch.Tensor   # [B, max_pages] int32 (device, the SeqTable view)
    lens: torch.Tensor         # [B] int32 (device), total KV length incl. the new query tokens
    page_size: int


def make_paged_kv(g: Geom, lens: list[int], page_size: int, max_pages: int | None = None,
                  dtype=torch.bfloat16, seed: int = 0, layers: int = 1) -> list[PagedKV]:
    """`layers` independent KV caches sharing one page table (like a model's layers sharing one plan).

    Pages are assigned in a random permutation so page tables are non-contiguous.
    """
    gen = torch.Generator(device="cuda").manual_seed(seed)
    need = [(n + page_size - 1) // page_size for n in lens]
    max_pages = max_pages or max(need)
    total = sum(need)
    perm = torch.randperm(total, device="cuda", generator=gen).to(torch.int32)
    pt = torch.zeros(len(lens), max_pages, dtype=torch.int32, device="cuda")
    off = 0
    for b, n in enumerate(need):
        pt[b, :n] = perm[off:off + n]
        off += n
    lens_t = torch.tensor(lens, dtype=torch.int32, device="cuda")
    out = []
    for _ in range(layers):
        shape = (total, g.hkv, page_size, g.d)
        k = torch.randn(shape, device="cuda", generator=gen, dtype=torch.bfloat16)
        v = torch.randn(shape, device="cuda", generator=gen, dtype=torch.bfloat16)
        if dtype != torch.bfloat16:
            k, v = k.to(dtype), v.to(dtype)
        out.append(PagedKV(k, v, pt, lens_t, page_size))
    return out


def csr_from_table(kv: PagedKV):
    """FlashInfer CSR metadata (indptr, indices, last_page_len) from page table + lengths (host-side form)."""
    lens = kv.lens.long()
    npages = (lens + kv.page_size - 1) // kv.page_size
    indptr = torch.zeros(len(lens) + 1, dtype=torch.int32, device="cuda")
    indptr[1:] = torch.cumsum(npages, 0)
    cols = torch.arange(kv.page_table.shape[1], device="cuda")
    indices = kv.page_table[cols[None, :] < npages[:, None]].contiguous()
    last = (lens - (npages - 1) * kv.page_size).to(torch.int32)
    return indptr, indices, last


def reference(q: torch.Tensor, kv: PagedKV, nq: int, scale: float, k_scale: float = 1.0,
              v_scale: float = 1.0) -> torch.Tensor:
    """fp32 causal attention. q: [B*nq, hq, d]; query row i of seq b sits at position lens[b]-nq+i."""
    B = kv.lens.numel()
    hq, d = q.shape[1], q.shape[2]
    hkv = kv.k.shape[1]
    out = torch.empty(B * nq, hq, d, dtype=torch.float32, device="cuda")
    for b in range(B):
        n = int(kv.lens[b])
        pages = kv.page_table[b, :(n + kv.page_size - 1) // kv.page_size].long()
        k = kv.k[pages].float().permute(1, 0, 2, 3).reshape(hkv, -1, d)[:, :n] * k_scale
        v = kv.v[pages].float().permute(1, 0, 2, 3).reshape(hkv, -1, d)[:, :n] * v_scale
        k = k.repeat_interleave(hq // hkv, 0)
        v = v.repeat_interleave(hq // hkv, 0)
        qb = q[b * nq:(b + 1) * nq].float().transpose(0, 1)               # [hq, nq, d]
        s = qb @ k.transpose(1, 2) * scale                                  # [hq, nq, n]
        pos = n - nq + torch.arange(nq, device="cuda")
        s.masked_fill_(torch.arange(n, device="cuda")[None, None, :] > pos[None, :, None], float("-inf"))
        out[b * nq:(b + 1) * nq] = (s.softmax(-1) @ v).transpose(0, 1)
    return out


def err(out: torch.Tensor, ref: torch.Tensor) -> dict:
    diff = (out.float() - ref).abs()
    return {"max_abs": diff.max().item(), "mean_abs": diff.mean().item(), "nan": bool(out.isnan().any())}


def l2_bytes() -> int:
    return torch.cuda.get_device_properties(0).L2_cache_size


def graph_time_us(fns, warmup: int = 3, reps: int = 10) -> float:
    """Median time of one call, capturing all `fns` (e.g. one per layer replica) in a single CUDA graph.

    Using several KV replicas whose total size exceeds L2 keeps small problems from running out of cache,
    as they would not in a real model with dozens of layers.
    """
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in fns:
            f()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for f in fns:
            f()
    for _ in range(warmup):
        graph.replay()
    times = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        graph.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1e3 / len(fns))
    del graph
    times.sort()
    return times[len(times) // 2]


def replicas_for(bytes_per_layer: int, budget: int = 2 << 30, cap: int = 64) -> int:
    """Enough KV replicas to cover 3x L2, within a memory budget."""
    want = -(-3 * l2_bytes() // max(bytes_per_layer, 1))
    return max(1, min(want, cap, budget // max(bytes_per_layer, 1)))
