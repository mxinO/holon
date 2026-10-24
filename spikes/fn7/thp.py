"""FN's speed round D (the PCIe rate): a host pool of the deployment's size over 4 KiB pages and over transparent huge
pages (`madvise(MADV_HUGEPAGE)` before page-locking), each card's cost to allocate and page-lock it (the faults,
the kernel's compaction and `cudaHostRegister`), the device memory its mapping takes (the GPU's page tables), and
its rates over it: the deployed `gather_roles` (two roles, the 5090's IQ3_XXS gate/up and IQ4_NL down units,
`expert_io.fetch_ctas` per role) at a decode layer's, a verify pass's and a prompt pass's units per role, and
copy-engine DMA (`copy_batch`, the ring's) of a prompt pass's units.
Usage: thp.py <GiB of host pool> [fetch_ctas per role]"""
from __future__ import annotations

import functools
import gc
import json
import mmap
import sys
import time

import torch
from holon.core import platform
from holon_plugins.kernels.expert_io import extension

ROLES = (1_254_400, 921_600)                 # bytes of a gate/up unit and a down unit
SM = torch.cuda.get_device_capability()[0] * 10 + torch.cuda.get_device_capability()[1]


def pool(nbytes: int, huge: bool) -> tuple[torch.Tensor, float]:
    t0 = time.perf_counter()
    region = mmap.mmap(-1, nbytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    if huge:
        region.madvise(mmap.MADV_HUGEPAGE)
    host = torch.frombuffer(region, dtype=torch.uint8)
    assert int(torch.cuda.cudart().cudaHostRegister(host.data_ptr(), nbytes, 1 | 2)) == 0
    took = time.perf_counter() - t0
    host._region = region
    return host, took


def dma(lib, fetch, n, hosts, dsts, stream) -> None:
    for f, h, d in zip(fetch, hosts, dsts, strict=True):
        lib.copy_batch(f, n, h, d, stream)


def timed(fn, reps: int = 30) -> float:
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
    gib = float(sys.argv[1])
    ctas = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    lib = extension.load(SM)
    half = int(gib * 2**30) // 2
    out = {"gpu": torch.cuda.get_device_name(), "pool_gib": gib, "ctas_per_role": ctas, "rows": []}
    for huge in (False, True):
        torch.cuda.synchronize()
        free0 = torch.cuda.mem_get_info()[0]
        host, took = pool(2 * half, huge)
        mode = "thp" if huge else "4k"
        alias = platform.device_alias(host, torch.device("cuda"))
        torch.cuda.synchronize()
        out[f"device_mib_{mode}"] = (free0 - torch.cuda.mem_get_info()[0]) >> 20   # its mapping's page tables
        out[f"lock_s_per_gib_{mode}"] = round(took / gib, 3)
        out[f"anon_huge_gib_{mode}"] = round(int(next(ln for ln in open("/proc/self/smaps_rollup")
                                                      if ln.startswith("AnonHugePages")).split()[1]) / 2**20, 1)
        srcs = [alias[r * half:r * half + (half // u) * u].view(-1, u) for r, u in enumerate(ROLES)]
        g = torch.Generator().manual_seed(0)
        for per_role in (3, 10, 64):
            dsts = [torch.empty(per_role, u, dtype=torch.uint8, device="cuda") for u in ROLES]
            fetch = torch.zeros(len(ROLES), 2, per_role, dtype=torch.int32)
            for r, s in enumerate(srcs):
                fetch[r, 0] = torch.randperm(s.shape[0], generator=g)[:per_role].int()
                fetch[r, 1] = torch.arange(per_role, dtype=torch.int32)
            fetch_d, n = fetch.cuda(), torch.full((len(ROLES),), per_role, dtype=torch.int32, device="cuda")
            nbytes = per_role * sum(ROLES)
            sec = timed(functools.partial(lib.fetch_roles, fetch_d, n, srcs, dsts, ctas * len(ROLES), 0, 1))
            row = {"pages": mode, "units_per_role": per_role, "gather_gbs": round(nbytes / sec / 1e9, 1)}
            if per_role == 64:
                hosts = [host[r * half:r * half + (half // u) * u].view(-1, u) for r, u in enumerate(ROLES)]
                fl = [fetch[r].contiguous() for r in range(len(ROLES))]
                side = torch.cuda.Stream()                   # (batched copies refuse the legacy stream)
                torch.cuda.set_stream(side)
                sec = timed(functools.partial(dma, lib, fl, per_role, hosts, dsts, side.cuda_stream))
                torch.cuda.set_stream(torch.cuda.default_stream())
                row["dma_gbs"] = round(nbytes / sec / 1e9, 1)
            out["rows"].append(row)
            print(json.dumps(row), flush=True)
        torch.cuda.synchronize()
        assert int(torch.cuda.cudart().cudaHostUnregister(host.data_ptr())) == 0
        del alias, srcs, host, hosts, dsts, fl                   # the map goes with its last view
        gc.collect()
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
