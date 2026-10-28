"""FN acceptance 4, offload exactness: the snapshot trace (holon_judge.snapshot: a long prompt, its prefix-hit
extension, 62 short requests through every graph bucket, logprobs and seeded sampling) recorded under named
configurations of one deployment, each on a fresh engine with the same state pools, and diffed against the first
run of the first: run-identity (`--runs`), cross-placement identity (other `moe.*` placements), offload against
all-resident (a configuration without the offload store), and plants (a configuration under `holon_judge.planted`'s
plant, which must differ).
Usage: exact.py <model> '<engine flags>' --config NAME='k=v ...' [--config ...] [--plant NAME=PLANT:'k=v ...']
       [--runs 3] [--long-len 8192] [--prompts judge/prompts/v1-long.jsonl] [--out <json>]"""
from __future__ import annotations

import argparse
import contextlib
import gc
import json
import pathlib
import shlex

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]


def engine(model: str, flags: str, **base):
    """An engine from `holon serve`-style flags over `base` (the flags win; `plugin_options` given in `base` merge on
    top of theirs)."""
    from holon.core.cli import engine_flags, engine_settings, layered
    from holon.core.engine import Engine, EngineConfig
    parser = argparse.ArgumentParser()
    engine_flags(parser, given_only=True)
    extra = base.pop("plugin_options", None)
    cfg = layered(base, engine_settings(parser.parse_args(shlex.split(flags))))
    return Engine(EngineConfig(model, **(layered(cfg, {"plugin_options": extra}) if extra else cfg)))


def options(text: str) -> dict:
    return dict(item.split("=", 1) for item in text.split())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("flags")
    ap.add_argument("--config", action="append", default=[], help="NAME='k=v ...' (plugin options over the flags')")
    ap.add_argument("--plant", action="append", default=[], help="NAME=PLANT:'k=v ...' (must differ)")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--long-len", type=int, default=8192)
    ap.add_argument("--prompts", default=str(ROOT / "judge/prompts/v1-long.jsonl"))
    ap.add_argument("--out")
    a = ap.parse_args()
    from holon_judge import planted, snapshot
    from holon_judge.recording import PromptSet

    trace = snapshot.trace(PromptSet.load(a.prompts), long_len=a.long_len)
    runs = [(name, extra, None, i) for name, _, extra in (c.partition("=") for c in a.config)
            for i in range(a.runs if not runs_of_first_only(a, name) else 1)]
    runs += [(name, rest.partition(":")[2], rest.partition(":")[0], 0) for name, _, rest in
             (p.partition("=") for p in a.plant)]
    budget, ref, out = None, None, {"steps": None, "runs": []}
    for name, extra, plant, i in runs:
        with planted.PLANTS[plant]() if plant else contextlib.nullcontext():
            e = engine(a.model, a.flags, plugin_options=options(extra),
                       **({} if budget is None else {"state_memory": budget}))
            pools = {k: p.capacity for k, p in e.ledger.pools.items() if k.startswith("state.")}   # (the schedule's)
            budget = budget or e.ledger.device_bytes()
            snap = snapshot.record(e, trace)
            e.close()
        del e
        gc.collect()
        torch.cuda.empty_cache()
        if ref is None:
            ref, out["steps"] = snap, len(snap.steps)
        where = snapshot.diff(ref, snap)
        out["runs"].append({"config": name, "run": i, "plant": plant, "state_pools": pools,
                            "identical": where is None, "first_difference": where})
        print(json.dumps(out["runs"][-1]), flush=True)
    # (a placement that frees VRAM may grow the state pools; the snapshot's per-step records pin the schedule anyway)
    out["pass"] = all(r["identical"] != bool(r["plant"]) for r in out["runs"])
    out["state_pools_equal"] = all(r["state_pools"] == out["runs"][0]["state_pools"] for r in out["runs"])
    print(json.dumps(out))
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(out, indent=1) + "\n")


def runs_of_first_only(a, name: str) -> bool:
    """Run-identity is checked on the first configuration; the others run once."""
    return name != a.config[0].partition("=")[0]


if __name__ == "__main__":
    main()
