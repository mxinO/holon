"""GLM acceptance 2, exactness (docs/v1-glm.md §6.2): the snapshot trace (holon_judge.snapshot: a long prompt, its
prefix-hit extension, 62 short requests through every graph bucket, logprobs and seeded sampling) recorded under
named configurations of one model at one `max_step_tokens`, each on a fresh engine, and diffed against a reference:
the first run of the first configuration, or `--ref` (a snapshot another process saved). Run-identity (`--runs` of
the first configuration), identity across fits, placement policies, lending, the ring and the RAM tier (other
configurations), and plants (a configuration under `holon_judge.planted`'s plant, which must differ, or raise from
the store's own checks: recorded as such). FN's `spikes/fn7/exact.py` with a configuration's own engine flags (not
only plugin options: the all-resident reference and the store's deployments take different budgets) and every
snapshot saved (`--save`).

A configuration is `NAME='<holon serve flags>'`, layered over the base flags (plugin options merged per key, `k=`
deleting one). Each engine measures its own pools unless the base flags give `--state-memory` (not FN's first-engine
budget: an explicit budget skips the startup profile, so an offload deployment's step-peak loan (`moe.lend_peak`)
never binds and a large step runs out of memory); the state pools are recorded per run.

Usage: exact.py <model> '<base flags>' --prompts <set.jsonl> --config NAME='<flags>' [--config ...]
       [--plant NAME=PLANT:'<flags>'] [--runs 3] [--long-len 24000] [--ref <snapshot.json>] [--save <dir>]
       [--out <json>]"""
from __future__ import annotations

import argparse
import contextlib
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
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--config", action="append", default=[], help="NAME='<flags>' (over the base flags)")
    ap.add_argument("--plant", action="append", default=[], help="NAME=PLANT:'<flags>' (must differ)")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--long-len", type=int, default=24000)
    ap.add_argument("--ref", help="a saved snapshot to diff against (else the first run)")
    ap.add_argument("--save", help="a directory for every run's snapshot (<config>.<run>.json)")
    ap.add_argument("--out")
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT / "spikes/fn7/tasks"))
    import run as tasks
    from holon.core.cli import layered
    from holon.core.engine import Engine, EngineConfig
    from holon_judge import planted, snapshot
    from holon_judge.recording import PromptSet

    base = tasks.flags(a.flags)
    trace = snapshot.trace(PromptSet.load(a.prompts), long_len=a.long_len)
    runs = [(name, extra, None, i) for k, (name, _, extra) in enumerate(c.partition("=") for c in a.config)
            for i in range(a.runs if k == 0 else 1)]
    runs += [(name, rest.partition(":")[2], rest.partition(":")[0], 0) for name, _, rest in
             (p.partition("=") for p in a.plant)]
    ref = snapshot.Snapshot.load(a.ref) if a.ref else None
    out = {"ref": "--ref" if a.ref else "the first run", "steps": None if ref is None else len(ref.steps), "runs": []}
    for name, extra, plant, i in runs:
        cfg = layered(base, tasks.flags(extra))
        e = None
        try:
            with planted.PLANTS[plant]() if plant else contextlib.nullcontext():
                e = Engine(EngineConfig(a.model, **cfg))
                pools = {k: p.capacity for k, p in e.ledger.pools.items() if k.startswith("state.")}  # (the schedule's)
                snap = snapshot.record(e, trace)
                stats = e.stats()
        except RuntimeError as err:          # a plant the store's own checks catch (its error words): it failed
            # — but only the store's own refusals count: an out-of-memory error or a CUDA error (a dead context) is
            # not a catch, it fails the run and stops it (step 5's review A m1)
            if not plant or isinstance(err, (torch.OutOfMemoryError, torch.AcceleratorError)) or \
                    "CUDA error" in str(err) or not str(err).startswith("moe"):
                raise
            out["runs"].append({"config": name, "run": i, "plant": plant, "identical": False,
                                "first_difference": f"the store refused: {str(err)[:300]}"})
            print(json.dumps(out["runs"][-1]), flush=True)
            continue
        finally:
            if e is not None:
                e.close()
            del e
            gc.collect()
            torch.cuda.empty_cache()
        if a.save:
            pathlib.Path(a.save).mkdir(parents=True, exist_ok=True)
            snap.save(pathlib.Path(a.save) / f"{name}.{i}.json")
        if ref is None:
            ref, out["steps"] = snap, len(snap.steps)
        where = snapshot.diff(ref, snap)
        out["runs"].append({"config": name, "run": i, "plant": plant, "state_pools": pools,
                            "identical": where is None, "first_difference": where,
                            "stats": stats})
        print(json.dumps(out["runs"][-1]), flush=True)
    out["pass"] = all(r["identical"] != bool(r["plant"]) for r in out["runs"])
    out["state_pools_equal"] = all(r.get("state_pools", out["runs"][0]["state_pools"]) == out["runs"][0]["state_pools"]
                                   for r in out["runs"])
    print(json.dumps({k: v for k, v in out.items() if k != "runs"}))
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
