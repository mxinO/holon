"""FN step 7 item 2 (`k_fetch` to link rate): GB/s of a decode layer's miss fetch from a mapped host pool into VRAM
slots — the store's `expert_fetch@1` (8 CTAs, zero-copy), a Triton zero-copy gather at more CTAs, and copy-engine
DMA — over host pages of 4 KiB and transparent huge pages (`madvise(MADV_HUGEPAGE)`), for 1–16 units of a decode
layer's sizes, and the cost of page-locking each. Usage: fetch_bw.py [GiB of host pool]"""
from __future__ import annotations

import ctypes
import functools
import json
import mmap
import sys
import time

import torch
import triton
import triton.language as tl
from holon.core import platform
from holon_plugins.kernels.expert_io import extension

UNIT = 1_048_576 + 524_288                   # gate/up's and down's bytes of an IQ2_XS expert, as one unit
SM = torch.cuda.get_device_capability()[0] * 10 + torch.cuda.get_device_capability()[1]


@triton.jit
def _gather(src, dst, fetch, n, words, BLOCK: tl.constexpr):
    """Program p copies words [p·BLOCK, …) of every fetched unit (int4 words as 4 int32 lanes)."""
    pid, progs = tl.program_id(0), tl.num_programs(0)
    count = tl.load(n)
    for b in range(count):
        s, d = tl.load(fetch + b).to(tl.int64), tl.load(fetch + 64 + b).to(tl.int64)
        for start in range(pid * BLOCK, words, progs * BLOCK):
            off = start + tl.arange(0, BLOCK)
            m = off < words
            tl.store(dst + d * words + off, tl.load(src + s * words + off, mask=m), mask=m)


def pool(nbytes: int, huge: bool) -> tuple[torch.Tensor, torch.Tensor, float]:
    region = mmap.mmap(-1, nbytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    if huge:
        region.madvise(mmap.MADV_HUGEPAGE)
    host = torch.frombuffer(region, dtype=torch.uint8)
    host.fill_(1)                                         # faulted in (huge pages where the kernel gives them)
    t0 = time.perf_counter()
    err = torch.cuda.cudart().cudaHostRegister(host.data_ptr(), nbytes, 1 | 2)
    assert int(err) == 0, err
    took = time.perf_counter() - t0
    host._region = region
    return host, platform.device_alias(host, torch.device("cuda")), took


def _dma(dst: torch.Tensor, host: torch.Tensor, ids: list[int]) -> None:
    for i, u in enumerate(ids):
        dst[i].copy_(host[u], non_blocking=True)


def timed(fn, reps: int = 50) -> float:
    fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / reps / 1e3


def main() -> None:
    gib = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
    units = int(gib * 2**30) // UNIT
    lib = extension.load(SM)
    dst = torch.empty(64, UNIT, dtype=torch.uint8, device="cuda")
    out = {"gpu": torch.cuda.get_device_name(), "unit_bytes": UNIT, "pool_gib": gib, "rows": []}
    for huge in (False, True):
        host, alias, took = pool(units * UNIT, huge)
        smaps = open("/proc/self/smaps_rollup").read()
        anon_huge = [ln for ln in smaps.splitlines() if ln.startswith("AnonHugePages")]
        out[f"register_s_per_gib_{'thp' if huge else '4k'}"] = round(took / gib, 3)
        out[f"anon_huge_{'thp' if huge else '4k'}"] = anon_huge
        src = alias.view(units, UNIT)
        g = torch.Generator().manual_seed(0)
        for n in (1, 2, 4, 8, 16):
            ids = torch.randperm(units, generator=g)[:n].int()
            fetch = torch.zeros(2, 64, dtype=torch.int32)
            fetch[0, :n], fetch[1, :n] = ids, torch.arange(n, dtype=torch.int32)
            fetch, cnt = fetch.cuda(), torch.tensor([n], dtype=torch.int32, device="cuda")
            row = {"pages": "thp" if huge else "4k", "units": n}
            sec = timed(functools.partial(lib.fetch, fetch, cnt, src, dst, False))
            row["k_fetch_gbs"] = round(n * UNIT / sec / 1e9, 1)
            s32, d32 = src.view(torch.int32), dst.view(torch.int32)
            for ctas in (8, 32, 128):
                sec = timed(functools.partial(_gather[(ctas,)], s32, d32, fetch, cnt, UNIT // 4, BLOCK=4096))
                row[f"triton_{ctas}_gbs"] = round(n * UNIT / sec / 1e9, 1)
            hv = host.view(units, UNIT)
            sec = timed(functools.partial(_dma, dst, hv, ids.tolist()))
            row["dma_gbs"] = round(n * UNIT / sec / 1e9, 1)
            out["rows"].append(row)
            print(json.dumps(row), flush=True)
        ctypes.c_int(int(torch.cuda.cudart().cudaHostUnregister(host.data_ptr())))
        del host, alias, src
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
