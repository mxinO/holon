"""Where a deployment's VRAM goes (FN step 7, lever 1: the cache fit): the resident parameters and buffers per
component, the ledger's device pools, the worst-case step's activation peak and the graphs' pool the state budget
reserves, the allocator's and the driver's totals. Usage: vram.py <model> '<engine flags>' [max_model_len]"""
from __future__ import annotations

import collections
import json
import sys

import torch
from _engine import engine
from holon.core import platform

COMPONENTS = ("mtp", "hc", "gdn", "linear_attn", "self_attn", "qsa", "shared", "experts", "router", "gate", "ple",
              "embed", "lm_head", "norm")


def component(name: str) -> str:
    parts = name.split(".")
    head = "mtp." if "mtp" in parts[:2] or name.startswith("aux") else ""
    for c in COMPONENTS[1:]:
        if any(c in p for p in parts):
            return head + c
    return head + (parts[-2] if len(parts) > 1 else parts[0])


def main() -> None:
    model, flags = sys.argv[1:3]
    max_len = int(sys.argv[3]) if len(sys.argv) > 3 else 33792
    seen = {}
    peak, growth = platform.peak_memory, platform.reserved_growth

    def peak_(device, fn):
        seen["worst_step_peak"] = peak(device, fn)
        return seen["worst_step_peak"]

    def growth_(device, fn):
        seen["graphs_pool"] = growth(device, fn)
        return seen["graphs_pool"]

    platform.peak_memory, platform.reserved_growth = peak_, growth_
    torch.cuda.memory._record_memory_history(max_entries=500_000, stacks="python")
    e = engine(model, flags, max_seqs=1, max_model_len=max_len)
    snap = torch.cuda.memory._snapshot()
    torch.cuda.memory._record_memory_history(enabled=None)
    per, seen_ptr = collections.Counter(), set()
    aux = getattr(e.ctx, "aux", None)
    modules = [("", e.executor.model)] + [(f"mtp.{ns}.", m) for ns, m in (aux.modules if aux is not None else [])]
    for prefix, mod in modules:
        for name, t in list(mod.named_parameters()) + list(mod.named_buffers()):
            if t.device.type != "cuda" or t.data_ptr() in seen_ptr:
                continue
            seen_ptr.add(t.data_ptr())
            per[component(prefix + name)] += t.numel() * t.element_size()
    pools = {p.name: {"unit_bytes": p.unit_bytes, "capacity": p.capacity, "bytes": p.unit_bytes * p.capacity,
                      "fill": p.fill, "paged": p.tokens_per_unit is not None}
             for p in e.ledger.pools.values() if not p.host and p.capacity}
    # every live block not a parameter, buffer or pool, by its allocating frame (the innermost in this repository)
    owned = set(seen_ptr) | {p.buffer.data_ptr() for p in e.ledger.pools.values()
                             if not p.host and p.capacity and p.buffer is not None}
    other = collections.Counter()
    for seg in snap["segments"]:
        addr = seg["address"]
        for b in seg["blocks"]:
            at, addr = addr, addr + b["size"]
            if b["state"] != "active_allocated" or at in owned:
                continue
            frames = b.get("frames", [])
            where = next((f"{f['filename'].split('fn-core/')[-1]}:{f['line']}" for f in frames
                          if any(f"/{r}/" in f["filename"] for r in ("holon", "std", "plugins", "core"))), "other")
            other[where] += b["size"]
    statics = collections.Counter()           # the build's named statics and workspaces (`ctx.static`)
    for name, buf in getattr(e.ctx, "_buffers", {}).items():
        if buf.is_cuda:
            statics[name.split("[")[0]] += buf.numel() * buf.element_size()
    free, total = torch.cuda.mem_get_info()
    out = {"flags": flags, "max_model_len": max_len, "params_and_buffers": dict(per.most_common()),
           "params_total": sum(per.values()),
           "other_live": {k: round(v / 2**20, 1) for k, v in other.most_common(25)},
           "other_live_total": sum(other.values()),
           "statics": {k: round(v / 2**20, 1) for k, v in statics.most_common(25)},
           "pools": pools, "pools_total": sum(p["bytes"] for p in pools.values()),
           **seen, "allocated": torch.cuda.memory_allocated(), "reserved": torch.cuda.memory_reserved(),
           "driver_used": total - free, "total": total}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
