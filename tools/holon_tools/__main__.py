"""Command line: the `holon impact | propagate | tune` commands of design §4.4 and §4.6.

    python -m holon_tools impact <unit | path | A..B>... [--plans plans]
    python -m holon_tools propagate <model> [--class C] [--new-model] [--approve] [--plans plans]
                                    --golden <fp32 golden> --floor <floor> [--prompts judge/prompts/v1.jsonl]
                                    [engine flags: --device --dtype --max-seqs ...]
    python -m holon_tools tune <model> [--ops op ...] [--batch 8] [--tokens 64] [the flags of propagate]

`propagate` and `tune` exit 0 when nothing needs doing or the lock was written, 2 when the new lock needs an
approval (re-run with --approve to write it), 1 on failure.

The lock's kernel pins carry over (they isolate a model from kernels installed since); engine settings come from the
flags, so changing one is a visible plan change that goes through the checks.
"""
from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys

import torch
from holon.api import Lock
from holon.core import plans, platform
from holon.core.cli import engine_config, engine_flags
from holon.core.engine import Engine, EngineConfig, resolve_model
from holon.core.plugins import PluginSet, discover
from holon_judge.golden import resolve_model as checkpoint_dir
from holon_judge.recording import PromptSet
from holon_judge.snapshot import Snapshot, trace

from . import impact, locks, propagate, tune

DEFAULT_PROMPTS = pathlib.Path(__file__).resolve().parents[2] / "judge" / "prompts" / "v1.jsonl"


def _impact(a: argparse.Namespace) -> int:
    try:
        units = impact.changed_units(a.changes, plans.unit_dirs(discover()), impact.repo_root())
    except ValueError as e:
        raise SystemExit(str(e)) from None
    print(f"changed units: {', '.join(sorted(units)) or '-'}")
    for lock, hit in impact.impact(locks.heads(a.plans), units):
        print(f"{lock.plan.model_id}  {lock.plan.fingerprint_class}: {', '.join(hit)}")
    return 0


def _start(a: argparse.Namespace, config: EngineConfig) -> tuple[Lock | None, dict[str, str]]:
    """The lock to start from — the newest of `--class`, else of this machine's class, else of the nearest class
    (`--new-model` asserts there is none) — and the kernel pins it implies here."""
    plugins = PluginSet(discover())
    model_id = resolve_model(plugins, config)[0].identity()
    fp = platform.fingerprint(torch.device(config.device))
    if a.cls:
        start = plans.newest(a.plans, model_id, a.cls)
    else:
        start = (plans.nearest_class(a.plans, model_id, fp) or (None, False))[0]
    if start is not None and a.new_model:
        raise SystemExit(f"{model_id}: has a lock already; drop --new-model")
    if start is None:
        if not a.new_model:
            raise SystemExit(f"{model_id}: no lock {f'for class {a.cls}' if a.cls else 'yet'}; use --new-model "
                             "for a first one")
        return None, {}
    pins = _pins(start)
    if start.plan.fingerprint_class != fp.cls:
        # Another class's choices are only preferences here: keep the pins this machine can run.
        print(f"no lock for {fp.cls}: starting from {start.plan.fingerprint_class}'s")
        device = torch.device(config.device).type
        pins = {op: p for op, p in pins.items() if p in plugins.installed and plugins.available(p, device)}
    return start, pins


def _pins(lock: Lock) -> dict[str, str]:
    try:
        return plans.lock_overrides(lock)["kernels"]
    except ValueError as e:
        raise SystemExit(f"{lock.plan.model_id}: {e}") from None


def _run(a: argparse.Namespace, start: Lock | None, config: EngineConfig) -> int:
    """Check `config` against `start` and write the new lock if the outcome allows."""
    prompts = PromptSet.load(a.prompts)
    l2 = propagate.L2Inputs(prompts, a.golden, a.floor) if a.golden and a.floor else None
    checks = propagate.EngineChecks(config, trace(prompts), l2)
    stored = start and start.snapshot and propagate.snapshot_path(a.plans, start.plan, start.snapshot)
    previous = Snapshot.load(stored) if stored and stored.is_file() else None
    sm = platform.fingerprint(torch.device(config.device)).sm
    outcome = propagate.decide(start, checks, sm, previous, a.allow_uncovered_advance)
    print(propagate.report(outcome))
    if outcome.verdict == "advance" or (outcome.verdict == "approve" and a.approve):
        print(f"wrote {propagate.write(a.plans, outcome)}")
        return 0
    return {"unchanged": 0, "approve": 2}.get(outcome.verdict, 1)


def _propagate(a: argparse.Namespace) -> int:
    config = engine_config(str(checkpoint_dir(a.model)), a)
    start, pins = _start(a, config)
    return _run(a, start, dataclasses.replace(config, kernels=pins))


def _tune(a: argparse.Namespace) -> int:
    config = engine_config(str(checkpoint_dir(a.model)), a)
    start, pins = _start(a, config)
    base = dataclasses.replace(config, kernels=pins)
    engine = Engine(base)
    resolved = _pins(Lock(locks.plan_of(engine)))  # every tuned op starts pinned to what the base resolves to
    ops = tune.candidates(engine, tuple(a.ops))
    del engine

    def rate(trial: dict[str, str]) -> float:
        return tune.decode_rate(dataclasses.replace(base, kernels=trial), a.batch, tokens=a.tokens)

    pins, rates = tune.tune(resolved, ops, rate)
    for op, by in rates.items():
        print(f"{op}: " + ", ".join(f"{p} {r:.1f} tok/s" for p, r in sorted(by.items(), key=lambda x: -x[1])))
    return _run(a, start, dataclasses.replace(base, kernels=pins))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m holon_tools")
    sub = ap.add_subparsers(required=True)

    i = sub.add_parser("impact", help="the newest locks that pin a changed unit")
    i.add_argument("changes", nargs="+", help="unit names, paths, or a git range A..B")
    i.add_argument("--plans", default="plans")
    i.set_defaults(run=_impact)

    for name, run, text in (("propagate", _propagate, "carry a model's lock to the checked-out code"),
                            ("tune", _tune, "pin the fastest kernel provider per decode op, then propagate")):
        p = sub.add_parser(name, help=text)
        p.add_argument("model", help="checkpoint directory or hub id")
        p.add_argument("--plans", default="plans")
        p.add_argument("--class", dest="cls", help="start from this fingerprint class's newest lock")
        p.add_argument("--new-model", action="store_true", help="write the model's first lock")
        p.add_argument("--approve", action="store_true", help="write a lock that needs an approval")
        p.add_argument("--allow-uncovered-advance", action="store_true",
                       help="advance without approval on a byte-identical snapshot trace, although changed-line "
                            "coverage is not checked yet")
        p.add_argument("--prompts", default=str(DEFAULT_PROMPTS), help="prompt set of L2 and the snapshot trace")
        p.add_argument("--golden", type=pathlib.Path, help="fp32 reference recording (judge golden)")
        p.add_argument("--floor", type=pathlib.Path, help="judge floor recording for the activation dtype")
        engine_flags(p)
        p.set_defaults(run=run)
        if name == "tune":
            p.add_argument("--ops", nargs="+", default=list(tune.OPS))
            p.add_argument("--batch", type=int, default=8)
            p.add_argument("--tokens", type=int, default=64)

    a = ap.parse_args(argv)
    return a.run(a)


if __name__ == "__main__":
    sys.exit(main())
