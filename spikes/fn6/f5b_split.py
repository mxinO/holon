"""F5b on an offload deployment (docs/dev/v1-offload.md D4): tune's split measured at batch 1 with MTP (the RTX 5080 MTP
rows' question, owner 2026-10-21) — F5a, then the CPU's share at each thread count, the best share. Writes the record
and the F5b options at the best share and threads, whatever the gain. Usage: f5b_split.py <model> '<engine flags>'
<prompts.jsonl> <out.json>"""
from __future__ import annotations

import dataclasses
import json
import sys

sys.path.insert(0, "spikes/fn7/tasks")

import run  # noqa: E402
from holon.core.cli import layered  # noqa: E402
from holon.core.engine import Engine, EngineConfig  # noqa: E402
from holon_judge.recording import PromptSet  # noqa: E402
from holon_tools import tune  # noqa: E402


def main() -> None:
    model, flags, prompts_path, out = sys.argv[1:5]
    base = EngineConfig(model, **layered({"max_seqs": 1, "max_model_len": 8192}, run.flags(flags)))
    e = Engine(base)
    store = e.ctx.shared_objects.get("std.moe.offload")
    misses, isa = store.top_k * base.max_seqs * (1 + base.max_draft_tokens), e.ctx.fingerprint.cpu_isa
    e.close()
    del e, store
    prompts = [p.ids for p in PromptSet.load(prompts_path).prompts]

    def rate(extra: dict) -> float:
        cfg = dataclasses.replace(base, plugin_options=base.plugin_options | extra)
        return tune.decode_rate(cfg, 1, tokens=128, prompts=prompts)

    taken, record = tune.split(rate, misses, isa)
    best = record["best"]
    record["options"] = {"moe.cpu_plan": tune.plan(best["share"], misses), "moe.cpu_isa": isa,
                         "cpu_task.threads": str(best["threads"])}
    record["taken_by_tune"] = taken
    json.dump(record, open(out, "w"), indent=1)
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
