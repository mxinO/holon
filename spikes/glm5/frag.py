"""GLM step 5 (the lead's call on `--memory-fraction` 0.99): the caching allocator's fragmentation against the slack
the reserve does not count. The deployment's engine at a given fraction serves (1) the snapshot trace
(`holon_judge.snapshot`: a long prompt, its prefix-hit extension, 62 short requests through every graph bucket) and
(2) the bench plan's shapes (natural-text prompts of 1K, 4K and 32K tokens, 256 decode tokens each, 3 of each), one
request at a time for (2). Per step it records the step's peak allocated and peak reserved bytes and the device's free
memory after it; the report gives, per phase, the worst bloat (peak reserved − peak allocated) and the least free
memory, against the slack (the device's memory past the fraction).

    python spikes/glm5/frag.py <maya-s -00001 shard> '<engine flags>' --prompts <glm v1-long set> --out <json>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("flags")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--long-len", type=int, default=24000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT / "spikes/fn7/tasks"))
    import run as tasks
    from holon.api import SamplingParams
    from holon.core.engine import Engine, EngineConfig
    from holon_judge import snapshot
    from holon_judge.recording import PromptSet

    cfg = tasks.flags(a.flags)
    e = Engine(EngineConfig(a.model, **cfg))
    total = torch.cuda.get_device_properties(0).total_memory
    slack = total - int(cfg.get("memory_fraction", 0.9) * total)
    rows: dict[str, list] = {"trace": [], "bench": []}
    step = e.step

    def recorded(phase: str):
        def run():
            torch.cuda.reset_peak_memory_stats()
            out = step()
            torch.cuda.synchronize()
            alloc, res = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
            rows[phase].append((alloc, res, torch.cuda.mem_get_info()[0]))
            return out
        return run
    prompts = PromptSet.load(a.prompts)
    e.step = recorded("trace")
    snapshot.record(e, snapshot.trace(prompts, long_len=a.long_len))
    text = [t for p in prompts.prompts for t in p.ids]
    e.step = recorded("bench")
    params = SamplingParams(temperature=0.0, max_tokens=256, ignore_eos=True)
    for n in (1024, 4096, 32768):
        for k in range(3):
            at = (k * 7919 + n) % max(1, len(text) - n)
            e.add(text[at:at + n], params)
            while e.has_work():
                e.step()
    e.close()
    out = {"fraction": cfg.get("memory_fraction"), "slack_gib": round(slack / 2**30, 3), "total_gib": total / 2**30}
    for phase, r in rows.items():
        out[phase] = {"steps": len(r), "worst_bloat_gib": round(max(res - al for al, res, _ in r) / 2**30, 3),
                      "peak_reserved_gib": round(max(res for _, res, _ in r) / 2**30, 3),
                      "peak_allocated_gib": round(max(al for al, _, _ in r) / 2**30, 3),
                      "least_free_gib": round(min(f for *_, f in r) / 2**30, 3)}
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
