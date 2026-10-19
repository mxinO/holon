"""Command line: the `holon impact | propagate | tune` commands of design §4.4 and §4.6, and `profile`
(docs/v1-offload.md F2).

    python -m holon_tools impact <unit | path | A..B>... [--plans plans]
    python -m holon_tools propagate <model> [--class C] [--new-model] [--approve] [--allow-approx] [--plans plans]
                                    --golden <fp32 golden> --floor <floor> [--prompts judge/prompts/v1.jsonl]
                                    [engine flags: --device --dtype --max-seqs ...]
    python -m holon_tools tune <model> [--ops op ...] [--batch 8] [--tokens 64] [--cpu-split] [--no-lock]
                                [--json record] [the flags of propagate]
    python -m holon_tools profile <model> [--prompts judge/prompts/v1.jsonl] [--max-tokens 64]
                                  [--name <stem>.profile.json] [engine flags: --host-memory ...]

`propagate` and `tune` exit 0 when nothing needs doing or the lock was written, 2 when the new lock needs an
approval (re-run with --approve to write it), 1 on failure — a lock binding an `approx` plugin (docs/v1-offload.md §6)
included, unless --allow-approx says a human gated it.

The lock's kernel pins carry over (they isolate a model from kernels installed since); engine settings come from the
flags, so changing one is a visible plan change that goes through the checks.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys

import torch
from holon.api import Lock, parse_versioned
from holon.core import plans, platform
from holon.core.assembly import resolve_model
from holon.core.cli import engine_config, engine_flags
from holon.core.engine import Engine, EngineConfig
from holon.core.plugins import PluginSet, discover
from holon_judge.golden import resolve_model as checkpoint_dir
from holon_judge.recording import PromptSet
from holon_judge.snapshot import Snapshot, trace

from . import impact, locks, profile, propagate, tune

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


def _start(a: argparse.Namespace, config: EngineConfig) -> tuple[Lock | None, dict[str, str], str]:
    """The lock to start from — the newest of `--class`, else of this machine's class, else of the nearest class
    (`--new-model` asserts there is none) —, the kernel pins it implies here, and the model id to write under: the
    history of the checkpoint's build if there is one (whatever name the file has here), else its identity."""
    plugins = PluginSet(discover())
    try:
        model_id = plans.history_id(a.plans, resolve_model(plugins, config)[0].identity())
    except ValueError as e:
        raise SystemExit(str(e)) from None
    fp = platform.fingerprint(torch.device(config.device))
    if a.cls:
        start = plans.newest(a.plans, model_id, a.cls)
    else:
        start = (plans.nearest_class(a.plans, model_id, fp) or (None,))[0]
    if start is not None and a.new_model:
        raise SystemExit(f"{model_id}: has a lock already; drop --new-model")
    if start is None:
        if not a.new_model:
            raise SystemExit(f"{model_id}: no lock {f'for class {a.cls}' if a.cls else 'yet'}; use --new-model "
                             "for a first one")
        return None, {}, model_id
    pins = _pins(start)
    if start.plan.fingerprint_class != fp.cls:
        # Another class's choices are only preferences here: keep the pins this machine can run.
        print(f"no lock for {fp.cls}: starting from {start.plan.fingerprint_class}'s")
        device = torch.device(config.device).type
        pins = {op: p for op, p in pins.items() if p in plugins.installed and plugins.available(p, device)}
    return start, pins, model_id


def _pins(lock: Lock) -> dict[str, str]:
    """The lock's kernel choices, which the tools rebuild as hard pins (an op bound to two providers stays unpinned)."""
    pins = plans.lock_overrides(lock)["kernel_preferences"]
    ops = {parse_versioned(key.rpartition("/")[2])[0] for key in lock.plan.sites}
    if split := sorted(ops - pins.keys()):
        print(f"{lock.plan.model_id}: {split} bound to several providers; left to default priority")
    return pins


def _run(a: argparse.Namespace, start: Lock | None, config: EngineConfig, model_id: str) -> int:
    """Check `config` against `start` and write the new lock if the outcome allows."""
    prompts = PromptSet.load(a.prompts)
    l2 = propagate.L2Inputs(prompts, a.golden, a.floor) if a.golden and a.floor else None
    stored = start and start.snapshot and propagate.snapshot_path(a.plans, start.plan, start.snapshot)
    previous = Snapshot.load(stored) if stored and stored.is_file() else None
    fp = platform.fingerprint(torch.device(config.device))
    # The stored trace's pools, so the new one is recorded on the same (from another process, another day) — on its
    # own class only: another class's card may not hold them (an RTX 4090's pools on an RTX 5080), so it measures.
    same = start is not None and start.plan.fingerprint_class == fp.cls
    pools = previous.meta.get("pool_bytes") if previous is not None and same else None
    checks = propagate.EngineChecks(config, trace(prompts), l2, pools, model_id)
    sm = fp.sm
    outcome = propagate.decide(start, checks, sm, previous, a.allow_uncovered_advance)
    print(propagate.report(outcome))
    if outcome.verdict == "advance" or (outcome.verdict == "approve" and a.approve):
        try:
            print(f"wrote {propagate.write(a.plans, outcome, a.allow_approx)}")
        except ValueError as e:
            raise SystemExit(str(e)) from None
        return 0
    return {"unchanged": 0, "approve": 2}.get(outcome.verdict, 1)


def _propagate(a: argparse.Namespace) -> int:
    config = engine_config(str(checkpoint_dir(a.model)), a)
    start, pins, model_id = _start(a, config)
    return _run(a, start, dataclasses.replace(config, kernels=pins), model_id)


def _tune(a: argparse.Namespace) -> int:
    config = engine_config(str(checkpoint_dir(a.model)), a)
    start, pins, model_id = (None, {}, None) if a.no_lock else _start(a, config)
    base = dataclasses.replace(config, kernels=pins)
    engine = Engine(base)
    resolved = _pins(Lock(locks.plan_of(engine)))  # every tuned op starts pinned to what the base resolves to
    ops = tune.candidates(engine, tuple(a.ops))
    store = engine.ctx.shared_objects.get("std.moe.offload")
    misses = store and store.top_k * base.max_seqs * (1 + base.max_draft_tokens)   # a graphed pass's most
    isa = engine.ctx.fingerprint.cpu_isa
    page_unit = max((p.unit_bytes for p in engine.ledger.pools.values() if p.tokens_per_unit and not p.host),
                    default=0)                       # the SSD level's I/O unit: a page, unit-major (D10)
    engine.close()
    del engine, store
    record = {}
    if a.cpu_split:                                  # F5b first: the kernels are then tuned with its decision in
        if not misses:
            raise SystemExit("--cpu-split: the engine flags must offload experts (--plugin-options moe.store=…)")
        prompts = [p.ids for p in PromptSet.load(a.prompts).prompts]

        def with_options(extra: dict[str, str]) -> float:
            trial = dataclasses.replace(base, plugin_options=base.plugin_options | extra)
            return tune.decode_rate(trial, a.batch, tokens=a.tokens, prompts=prompts)
        extra, record["cpu_split"] = tune.split(with_options, misses, isa)
        print(f"F5b: F5a {record['cpu_split']['f5a']:.1f} tok/s; threads at half the misses " + ", ".join(
            f"{t}: {r:.1f}" for t, r in record["cpu_split"]["threads_at_half"].items()) + "; shares " + ", ".join(
            f"{s}: {r:.1f}" for s, r in record["cpu_split"]["shares"].items()) + f"; gain "
            f"{record['cpu_split']['gain']:+.1%}: " + (f"taken ({extra})" if extra else "F5a stays"))
        base = dataclasses.replace(base, plugin_options=base.plugin_options | extra)

    if a.ssd_rates:                                  # this host's drive (D4): the level's clock rates
        threads = int(base.plugin_options.get("state_tier.io_threads", 3))
        got = record["ssd_rates"] = tune.ssd_rates(a.ssd_rates, a.ssd_unit or page_unit, threads)
        rates = {f"state_tier.ssd.{k}": f"{got[k]:.4g}" for k in ("read", "write", "mixed", "read_beside")}
        print("ssd rates: " + ", ".join(f"{k} {v}" for k, v in rates.items()) + f" (O_DIRECT {got['o_direct']})")
        if base.plugin_options.get("state_tier.levels") == "ssd":
            base = dataclasses.replace(base, plugin_options=base.plugin_options | rates)
    if a.clock:                                      # the tier's clock, on the settings the lock will run
        if not base.slots.get("tier"):
            raise SystemExit("--clock: the engine flags must bind a tier (--slots tier=std.state_tier)")
        samples = tune.step_times(base)
        cost = ",".join(f"{v:.6g}" for v in tune.clock(samples))
        record["clock"] = {"samples": samples, "step_cost": cost}
        print("clock: " + ", ".join(f"{t} tok × {r} rows {s * 1e3:.1f} ms" for t, r, s in samples)
              + f"; step_cost {cost}")
        base = dataclasses.replace(base, plugin_options=base.plugin_options | {"state_tier.step_cost": cost})

    def rate(trial: dict[str, str]) -> float:
        return tune.decode_rate(dataclasses.replace(base, kernels=trial), a.batch, tokens=a.tokens)

    pins, rates = tune.tune(resolved, ops, rate)
    for op, by in rates.items():
        print(f"{op}: " + ", ".join(f"{p} {r:.1f} tok/s" for p, r in sorted(by.items(), key=lambda x: -x[1])))
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(record | {"kernel_rates": rates}, indent=1) + "\n")
    if a.no_lock:
        return 0
    return _run(a, start, dataclasses.replace(base, kernels=pins), model_id)


def _profile(a: argparse.Namespace) -> int:
    model = checkpoint_dir(a.model)
    counts, directory = profile.profile(engine_config(str(model), a),
                                        [p.ids for p in PromptSet.load(a.prompts).prompts], a.max_tokens)
    name = a.name or profile.default_name(pathlib.Path(model))
    print(f"wrote {profile.write(directory, name, counts)}: moe.placement.profile={name}")
    return 0


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
        p.add_argument("--allow-approx", action="store_true",
                       help="write a lock that binds an `approx` plugin (a human gate, never a default lock)")
        p.add_argument("--prompts", default=str(DEFAULT_PROMPTS), help="prompt set of L2 and the snapshot trace")
        p.add_argument("--golden", type=pathlib.Path, help="fp32 reference recording (judge golden)")
        p.add_argument("--floor", type=pathlib.Path, help="judge floor recording for the activation dtype")
        engine_flags(p)
        p.set_defaults(run=run)
        if name == "tune":
            p.add_argument("--ops", nargs="+", default=list(tune.OPS))
            p.add_argument("--batch", type=int, default=8)
            p.add_argument("--tokens", type=int, default=64)
            p.add_argument("--cpu-split", action="store_true", help="decide F5b (an offloaded MoE's CPU share) first")
            p.add_argument("--clock", action="store_true",
                           help="fit a tier's state_tier.step_cost to this engine's timed steps first")
            p.add_argument("--ssd-rates", metavar="DIR",
                           help="measure the SSD level's rates on the drive under DIR (taken into the lock when the "
                                "engine flags bind the level, state_tier.levels=ssd)")
            p.add_argument("--ssd-unit", type=int, help="the I/O unit of --ssd-rates, bytes (default: the engine's "
                           "largest page unit)")
            p.add_argument("--no-lock", action="store_true", help="measure and report only; write no lock")
            p.add_argument("--json", help="write the measured rates here")

    f = sub.add_parser("profile", help="a routing profile for the offload store's warm start, beside the checkpoint")
    f.add_argument("model", help="checkpoint file or directory, or hub id")
    f.add_argument("--prompts", default=str(DEFAULT_PROMPTS), help="prompt set whose generations are counted")
    f.add_argument("--max-tokens", type=int, default=64, help="greedy tokens generated per prompt")
    f.add_argument("--name", help="the profile's file name, beside the checkpoint (default: <stem>.profile.json)")
    engine_flags(f)
    f.set_defaults(run=_profile)

    a = ap.parse_args(argv)
    return a.run(a)


if __name__ == "__main__":
    sys.exit(main())
