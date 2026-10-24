"""FN's speed round A–B: a deployment's VRAM ledger at a given `max_seqs` — parameters and buffers per component, the
ledger's device pools (the experts' arena, the state pools), the startup profile's reserve (the worst step's live peak
as measured, the allocator's reserved growth over that same step: its segments, and the graphs' pool with the device
memory their execs keep), the step-peak loan, the cached share of the target's and the head's expert bytes, what is
left free after startup, and the CUDA context; the arena's composition (per class its null, staging and cache
slots). With `--peak`, the worst step's live tensors at its peak, by allocating frame and its caller (MiB, blocks,
the largest block's MiB; the allocator's trace replayed).
Usage: ledger.py <model> '<engine flags>' [--seqs 1 4 8 16] [--max-len 33792] [--peak]"""
from __future__ import annotations

import argparse
import collections
import gc
import json
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("flags")
    ap.add_argument("--seqs", type=int, nargs="+", default=[1])
    ap.add_argument("--max-len", type=int, default=33792)
    ap.add_argument("--peak", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT / "spikes/fn6"))
    from _engine import STORE, engine
    from holon.core import executor, platform
    from vram import component
    peak, growth = platform.peak_memory, platform.reserved_growth
    state_memory, capture = executor.Executor.state_memory, executor.Executor.capture_graphs

    def outside() -> float:                        # device bytes outside torch's allocator (GiB)
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        return round((total - free - torch.cuda.memory_reserved()) / 2**30, 3)
    for seqs in a.seqs:
        seen = {"peaks": [], "graphs": None}

        def peak_(device, fn, seen=seen):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            reserved = torch.cuda.memory_reserved(device)
            tracing = a.peak and not seen["peaks"]               # the worst step, the first one profiled
            if tracing:
                torch.cuda.memory._record_memory_history(max_entries=2_000_000, stacks="python")
            out = peak(device, fn)
            if tracing:
                seen["top_at_peak"] = top_at_peak(torch.cuda.memory._snapshot())
                torch.cuda.memory._record_memory_history(enabled=None)
            seen["peaks"].append({"live": out, "reserved_growth": torch.cuda.memory_reserved(device) - reserved})
            return out

        def growth_(device, fn, copies=1, seen=seen):
            more = [copies] if copies != 1 else []               # (a tree before the execs' reserve: two args)
            seen["graphs"] = {"reserve": growth(device, fn, *more), "buckets": copies}
            return seen["graphs"]["reserve"]
        platform.peak_memory, platform.reserved_growth = peak_, growth_
        stages = {"start": outside()}

        def state_memory_(self, ledger, fraction, stages=stages):
            stages["model_built"] = outside()
            out = state_memory(self, ledger, fraction)
            stages["profiled"] = outside()
            return out

        def capture_(self, stages=stages):
            stages["before_capture"] = outside()
            capture(self)
            stages["captured"] = outside()
        executor.Executor.state_memory, executor.Executor.capture_graphs = state_memory_, capture_
        free0, total = torch.cuda.mem_get_info()
        e = engine(a.model, a.flags, max_seqs=seqs, max_model_len=a.max_len)
        platform.peak_memory, platform.reserved_growth = peak, growth
        executor.Executor.state_memory, executor.Executor.capture_graphs = state_memory, capture
        per, ptrs = collections.Counter(), set()
        aux = getattr(e.ctx, "aux", None)
        modules = [("", e.executor.model)] + [(f"mtp.{ns}.", m) for ns, m in (aux.modules if aux is not None else [])]
        for prefix, mod in modules:
            for name, t in list(mod.named_parameters()) + list(mod.named_buffers()):
                if t.device.type == "cuda" and t.data_ptr() not in ptrs:
                    ptrs.add(t.data_ptr())
                    per[component(prefix + name)] += t.numel() * t.element_size()
        pools = {p.name: p.unit_bytes * p.capacity for p in e.ledger.pools.values() if not p.host and p.capacity}
        s = e.ctx.shared_objects.get(STORE)
        share, arena = {}, {}
        if s is not None:
            arena = {c.name: {"null": c.null, "staging": c.staging, "cache": c.size - c.null - c.staging,
                              "unit_bytes": c.unit_bytes} for c in s.classes.values()}
            for aux_ in (False, True):
                cls = [c for c in s.classes.values() if c.name.startswith("aux.") == aux_]
                share["head" if aux_ else "target"] = round(
                    sum((c.size - c.null - c.staging) * c.unit_bytes for c in cls) /
                    max(sum(len(c.pieces) * c.unit_bytes for c in cls), 1), 4)
        statics = sum(b.numel() * b.element_size() for b in getattr(e.ctx, "_buffers", {}).values() if b.is_cuda)
        torch.cuda.synchronize()
        free, _ = torch.cuda.mem_get_info()
        out = {"max_seqs": seqs, "total": total, "free_before": free0, "free_after_startup": free,
               "params_and_buffers": dict(per.most_common()), "params_total": sum(per.values()),
               "statics_total": statics, "pools": pools, "pools_total": sum(pools.values()),
               "profile": seen, "loan_bytes": getattr(s, "_loan_bytes", 0) if s else 0,
               "lending": getattr(s, "lending", None), "own_tail": getattr(s, "_own_tail", None),
               "cached_share": share, "arena": arena, "allocated": torch.cuda.memory_allocated(),
               "reserved": torch.cuda.memory_reserved(),
               "outside_torch": total - free - torch.cuda.memory_reserved(), "outside_stages_gib": stages}
        print(json.dumps(out), flush=True)
        e.close()
        del e, s
        gc.collect()
        torch.cuda.empty_cache()


def top_at_peak(snap: dict, n: int = 30) -> dict:
    """The trace's live blocks at its most-allocated moment, by the innermost frame in this repository (MiB)."""
    events = snap["device_traces"][0] if snap.get("device_traces") else []
    live, now, best, at_best = {}, 0, 0, {}
    for ev in events:
        if ev["action"] == "alloc":
            live[ev["addr"]] = ev
            now += ev["size"]
            if now > best:
                best, at_best = now, dict(live)
        elif ev["action"] == "free_completed" and ev["addr"] in live:
            now -= live.pop(ev["addr"])["size"]
    by, count, biggest = collections.Counter(), collections.Counter(), collections.Counter()
    for ev in at_best.values():
        frames = [f"{'/'.join(f['filename'].rsplit('/', 2)[-2:])}:{f['line']}" for f in ev.get("frames", [])
                  if any(f"/{r}/" in f["filename"] for r in ("holon", "holon_std", "holon_plugins"))]
        where = " < ".join(frames[:2]) or "other"
        by[where] += ev["size"]
        count[where] += 1
        biggest[where] = max(biggest[where], ev["size"])
    return {"peak_mib": round(best / 2**20, 1),
            "by_frame": {k: [round(v / 2**20, 1), count[k], round(biggest[k] / 2**20, 1)]
                         for k, v in by.most_common(n)}}


if __name__ == "__main__":
    main()
