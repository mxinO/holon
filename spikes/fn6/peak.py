"""What the worst-case step's activation peak — the bytes the state budget reserves for activations — is made of:
the startup profile's step replayed under the allocator's history, the blocks alive at its peak grouped by their
allocating frame (the innermost in this repository). Usage: peak.py <model> '<engine flags>' [max_model_len]"""
from __future__ import annotations

import collections
import json
import sys

import torch
from _engine import engine
from holon.core import platform

ROOTS = ("holon", "plugins", "std", "core")


def site(frames: list[dict]) -> str:
    for f in frames:
        name = f.get("filename", "")
        if any(f"/{r}/" in name for r in ROOTS) and "torch" not in name.split("/")[-1]:
            return f"{name.split('fn-core/')[-1]}:{f.get('line')} {f.get('name')}"
    return "other"


def main() -> None:
    model, flags = sys.argv[1:3]
    max_len = int(sys.argv[3]) if len(sys.argv) > 3 else 33792
    out = {}
    peak = platform.peak_memory

    def traced(device, fn):
        torch.cuda.memory._record_memory_history(max_entries=2_000_000, stacks="python")
        result = peak(device, fn)
        snap = torch.cuda.memory._snapshot()
        torch.cuda.memory._record_memory_history(enabled=None)
        alive, now, best, at_best = {}, 0, 0, {}
        for ev in snap["device_traces"][torch.cuda.current_device()]:
            if ev["action"] == "alloc":
                alive[ev["addr"]] = (ev["size"], site(ev.get("frames", [])))
                now += ev["size"]
            elif ev["action"] in ("free_completed",) and ev["addr"] in alive:
                now -= alive.pop(ev["addr"])[0]
            if now > best:
                best, at_best = now, dict(alive)
        groups = collections.Counter()
        for size, where in at_best.values():
            groups[where] += size
        out.update(peak=result, traced_peak=best,
                   sites={k: round(v / 2**20, 1) for k, v in groups.most_common(40)})
        return result

    platform.peak_memory = traced
    engine(model, flags, max_seqs=1, max_model_len=max_len)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
