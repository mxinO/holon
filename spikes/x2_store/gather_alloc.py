"""The in-graph gather (`expert_fetch@1`, kernels.expert_io: 8 CTAs x 256 threads, 16-byte loads, unroll 8) from host
memory allocated four ways (docs/v1-offload.md §13, panel C-r4.5-5): `cudaHostAlloc` (torch's pinned allocator; mapped
under unified addressing), and an anonymous map registered with `cudaHostRegister(portable | mapped)` — shared or
private, with and without MADV_HUGEPAGE. Expert-sized blobs (2.65 and 3.06 MB: Qwen3-30B-A3B Q4_K_M experts) are
gathered 64 at a time from a 2 GiB arena, in a CUDA graph; GB/s per allocation, and its ratio to `cudaHostAlloc`.

    CUDA_VISIBLE_DEVICES=0 python spikes/x2_store/gather_alloc.py --out evidence/x2_gather_alloc_<class>_<date>.json
"""
from __future__ import annotations

import argparse
import json
import mmap
import statistics
import subprocess

import torch
from holon.core import platform
from holon.core.build import BuildContext
from holon.core.plugins import PluginSet, discover

ARENA = 2 << 30
BLOBS = {"2.65MB": 2_654_208, "3.06MB": 3_059_712}           # Q4_K gate/up + Q4_K down; + Q6_K down
GATHER, REPS, ROUNDS = 64, 20, 5


def anon_huge_kb() -> int:
    """This process's anonymous memory in transparent huge pages (did MADV_HUGEPAGE take?)."""
    with open("/proc/self/smaps_rollup") as f:
        return next(int(line.split()[1]) for line in f if line.startswith("AnonHugePages:"))


def registered(nbytes: int, private: bool, huge: bool) -> tuple[torch.Tensor, object]:
    flags = mmap.MAP_ANONYMOUS | (mmap.MAP_PRIVATE if private else mmap.MAP_SHARED)
    region = mmap.mmap(-1, nbytes, flags=flags)
    if huge:
        region.madvise(mmap.MADV_HUGEPAGE)
    host = torch.frombuffer(region, dtype=torch.uint8)
    host.fill_(1)                                                 # touched before it is registered
    err = torch.cuda.cudart().cudaHostRegister(host.data_ptr(), nbytes, 1 | 2)
    assert int(err) == 0, err
    return host, region


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    args = p.parse_args()
    device = torch.device("cuda")
    ctx = BuildContext(PluginSet(discover()), device)
    results = {}
    for blob, unit in BLOBS.items():
        units = ARENA // unit
        fetch = ctx.select_kernel("expert_fetch@1", site="bench", unit_bytes=unit, staging=GATHER).impl(
            ctx, unit_bytes=unit, staging=GATHER)
        dst = torch.empty(GATHER + 8, unit, dtype=torch.uint8, device=device)
        n = torch.tensor([GATHER], dtype=torch.int32, device=device)
        g = torch.Generator().manual_seed(0)
        lists = [torch.randperm(units, generator=g)[:GATHER].int().to(device) for _ in range(REPS)]
        size = units * unit
        allocs = {"cudaHostAlloc": None, "mmap_shared": (False, False), "mmap_private": (True, False),
                  "mmap_private_hugepage": (True, True)}
        for name, how in allocs.items():
            huge = anon_huge_kb()
            host, region = (torch.empty(size, dtype=torch.uint8, pin_memory=True), None) if how is None else \
                registered(size, *how)
            huge = anon_huge_kb() - huge
            src = platform.device_alias(host, device).view(units, unit)
            graph = torch.cuda.CUDAGraph()
            for fl in lists[:1]:
                fetch(fl, n, src, dst, 8)
            torch.cuda.synchronize()
            with torch.cuda.graph(graph):
                for fl in lists:
                    fetch(fl, n, src, dst, 8)
            rates = []
            for _ in range(ROUNDS):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                rates.append(REPS * GATHER * unit / (start.elapsed_time(end) * 1e-3) / 1e9)
            results.setdefault(blob, {})[name] = {"gb_s": round(statistics.median(rates), 2),
                                                  "rounds": [round(r, 2) for r in rates], "anon_huge_kb": huge}
            del graph, src
            if region is not None:
                torch.cuda.cudart().cudaHostUnregister(host.data_ptr())
            del host, region
        base = results[blob]["cudaHostAlloc"]["gb_s"]
        for r in results[blob].values():
            r["vs_cudaHostAlloc"] = round(r["gb_s"] / base, 3)
    record = {"command": "python spikes/x2_store/gather_alloc.py", "gpu": platform.fingerprint(device).cls,
              "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
              "torch": torch.__version__, "gather": {"ctas": 8, "threads": 256, "unroll": 8, "blobs_per_call": GATHER,
                                                     "calls_per_graph": REPS, "arena_bytes": ARENA},
              "results": results}
    print(json.dumps(results, indent=1))
    with open(args.out, "w") as f:
        json.dump(record, f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()
