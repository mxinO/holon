"""Offload exactness (FN acceptance 4, GLM acceptance 2): the snapshot trace (holon_judge.snapshot: a long prompt, its
prefix-hit extension, 62 short requests through every graph bucket, logprobs and seeded sampling) recorded under
named configurations of one deployment, each on a fresh engine, and diffed against a reference: the first run of the
first configuration, or `--ref` (a snapshot another process saved). Run-identity (`--runs` of the first
configuration), cross-placement identity (other `moe.*` placements, fits, lending, the ring and the RAM tier),
offload against all-resident (a configuration without the offload store), and plants (a configuration under
`holon_judge.planted`'s plant, which must differ, or raise from the store's own checks: recorded as such).

A configuration is `NAME='<words>'`, layered over the base flags: `holon serve` flags, led by bare `k=v` plugin options
if any (`--config ram='moe.ram=4096'` is `--config ram='--plugin-options moe.ram=4096'`); plugin options merge per
key, `k=` deleting one. `--pools first` (the default) gives every engine the first one's state budget, so the state
pools are the same; `--pools own` lets each measure its own (an explicit budget skips the startup profile, so an
offload deployment's step-peak loan, `moe.lend_peak`, never binds and its large steps run out of memory: the
deployments that lend need it). The state pools are recorded per run; `--save` keeps every run's snapshot.

Usage: exact.py <model> '<base flags>' --config NAME='<words>' [--config ...] [--plant NAME=PLANT:'<words>']
       [--runs 3] [--long-len 8192] [--prompts judge/prompts/v1-long.jsonl] [--pools first|own]
       [--ref <snapshot.json>] [--save <dir>] [--out <json>]"""
from __future__ import annotations

import argparse
import contextlib
import gc
import itertools
import json
import pathlib
import shlex

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]


def config(words: str) -> dict:
    """A configuration's settings: its leading bare `k=v` words are plugin options, the rest `holon serve` flags."""
    from holon.core.cli import layered
    from holon_judge.engine import flag_settings
    split = shlex.split(words)
    bare = list(itertools.takewhile(lambda w: not w.startswith("-"), split))
    options = dict(w.split("=", 1) for w in bare)
    return layered(flag_settings(shlex.join(split[len(bare):])), {"plugin_options": options})


def refused(err: RuntimeError) -> bool:
    """A plant the store's own checks catch (its error words, `moe…`): a failure. An out-of-memory error or a CUDA
    error (a dead context) is not a catch: it fails the run and stops it."""
    return not isinstance(err, (torch.OutOfMemoryError, torch.AcceleratorError)) and "CUDA error" not in str(err) \
        and str(err).startswith("moe")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("flags")
    ap.add_argument("--config", action="append", default=[], help="NAME='<words>' (over the base flags)")
    ap.add_argument("--plant", action="append", default=[], help="NAME=PLANT:'<words>' (must differ)")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--long-len", type=int, default=8192)
    ap.add_argument("--prompts", default=str(ROOT / "judge/prompts/v1-long.jsonl"))
    ap.add_argument("--pools", choices=("first", "own"), default="first", help="the state budget: the first engine's"
                    " for all, or each engine's own")
    ap.add_argument("--ref", help="a saved snapshot to diff against (else the first run)")
    ap.add_argument("--save", help="a directory for every run's snapshot (<config>.<run>.json)")
    ap.add_argument("--out")
    a = ap.parse_args()
    from holon.core.cli import layered
    from holon.core.engine import Engine, EngineConfig
    from holon_judge import planted, snapshot
    from holon_judge.engine import flag_settings
    from holon_judge.recording import PromptSet

    base = flag_settings(a.flags)
    trace = snapshot.trace(PromptSet.load(a.prompts), long_len=a.long_len)
    runs = [(name, words, None, i) for k, (name, _, words) in enumerate(c.partition("=") for c in a.config)
            for i in range(a.runs if k == 0 else 1)]
    runs += [(name, rest.partition(":")[2], rest.partition(":")[0], 0) for name, _, rest in
             (p.partition("=") for p in a.plant)]
    ref = snapshot.Snapshot.load(a.ref) if a.ref else None
    budget = None
    out = {"ref": "--ref" if a.ref else "the first run", "pools": a.pools,
           "steps": None if ref is None else len(ref.steps), "runs": []}
    for name, words, plant, i in runs:
        cfg = layered({} if budget is None else {"state_memory": budget}, layered(base, config(words)))
        e = None
        try:
            with planted.PLANTS[plant]() if plant else contextlib.nullcontext():
                e = Engine(EngineConfig(a.model, **cfg))
                pools = {k: p.capacity for k, p in e.ledger.pools.items() if k.startswith("state.")}  # (the schedule's)
                if a.pools == "first":
                    budget = budget or e.ledger.device_bytes()
                snap = snapshot.record(e, trace)
                stats = e.stats()
        except RuntimeError as err:
            if not plant or not refused(err):
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
                            "identical": where is None, "first_difference": where, "stats": stats})
        print(json.dumps(out["runs"][-1]), flush=True)
    # (a placement that frees VRAM may grow the state pools; the snapshot's per-step records pin the schedule anyway)
    out["pass"] = all(r["identical"] != bool(r["plant"]) for r in out["runs"])
    pools = [r["state_pools"] for r in out["runs"] if "state_pools" in r]
    out["state_pools_equal"] = all(p == pools[0] for p in pools)
    print(json.dumps({k: v for k, v in out.items() if k != "runs"}))
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
