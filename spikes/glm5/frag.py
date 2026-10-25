"""GLM step 5 (the lead's call on `--memory-fraction` 0.99): the caching allocator's fragmentation against the slack
the reserve does not count. The deployment's engine at a given fraction serves (1) the snapshot trace
(`holon_judge.snapshot`: a long prompt, its prefix-hit extension, 62 short requests through every graph bucket) and
(2) the bench plan's shapes (natural-text prompts of 1K, 4K and 32K tokens, 256 decode tokens each, 3 of each) and
mid-size prompts with prompt logprobs (2K-12K, the gates' kind), one request at a time for (2). Per step it records
the step's peak allocated and peak reserved bytes, the device's free memory after it, its prompt tokens and its
activation loan (the store's `_step_loan`) and the loan's overflow (bytes its segments took outside the range: the
device's own memory); the report gives, per phase, the worst bloat (peak reserved − peak allocated), the least free
memory against the slack (the device's memory past the fraction), and the steps whose loan overflowed. An
out-of-memory error ends the run and is recorded with its step.

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

    cfg = {"max_model_len": 33792} | tasks.flags(a.flags)
    e = Engine(EngineConfig(a.model, **cfg))
    store = e.ctx.shared_objects.get("std.moe.offload")
    loan = getattr(store, "_loan", None)
    total = torch.cuda.get_device_properties(0).total_memory
    slack = total - int(cfg.get("memory_fraction", 0.9) * total)
    rows: dict[str, list] = {"trace": [], "bench": [], "prompt_logprobs": []}
    step = e.step

    def recorded(phase: str):
        def run():
            torch.cuda.reset_peak_memory_stats()
            out = step()
            torch.cuda.synchronize()
            alloc, res = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
            over = loan.lib.loan_overflow() if loan is not None else 0
            rows[phase].append((alloc, res, torch.cuda.mem_get_info()[0], store.step_tokens if store else 0,
                                getattr(store, "_step_loan", 0), over))
            return out
        return run
    prompts = PromptSet.load(a.prompts)
    text = [t for p in prompts.prompts for t in p.ids]
    oom = None
    try:
        e.step = recorded("trace")
        snapshot.record(e, snapshot.trace(prompts, long_len=a.long_len))
        e.step = recorded("bench")
        params = SamplingParams(temperature=0.0, max_tokens=256, ignore_eos=True)
        for n in (1024, 4096, 32768):
            for k in range(3):
                at = (k * 7919 + n) % max(1, len(text) - n)
                e.add(text[at:at + n], params)
                while e.has_work():
                    e.step()
        e.step = recorded("prompt_logprobs")
        for n in (2048, 3000, 5451, 8000, 12000):
            e.add(text[n:2 * n], SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1))
            while e.has_work():
                e.step()
    except torch.OutOfMemoryError as err:
        oom = {"phase": [k for k, v in rows.items() if v][-1], "error": str(err)[:300]}
    e.close()
    out = {"fraction": cfg.get("memory_fraction"), "slack_gib": round(slack / 2**30, 3), "total_gib": total / 2**30,
           "oom": oom}
    gib = lambda b: round(b / 2**30, 3)                                     # noqa: E731
    for phase, r in rows.items():
        if not r:
            continue
        out[phase] = {"steps": len(r), "worst_bloat_gib": gib(max(res - al for al, res, *_ in r)),
                      "peak_reserved_gib": gib(max(res for _, res, *_ in r)),
                      "peak_allocated_gib": gib(max(al for al, *_ in r)),
                      "least_free_gib": gib(min(f for _, _, f, *_ in r)),
                      "loan_overflow_gib": gib(max(o for *_, o in r)),
                      "lent_steps (tokens, loan GiB)": sorted({(t, gib(ln)) for _, _, _, t, ln, _ in r if ln})}
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
