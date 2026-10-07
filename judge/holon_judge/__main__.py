"""Judge command line.

    python -m holon_judge golden   --model <hf dir or id> --prompts <jsonl> --out <file> [--threads N] [--k 20]
                                   [--dtype bfloat16 --device cuda]          # the floor: HF in the activation dtype
    python -m holon_judge compare  --golden <file> --candidate <file> [--floor <file>] [--json <file>]
    python -m holon_judge l2       --model <hf dir or id> --prompts <jsonl> --golden <file> --floor <file>
                                   [--dtype bfloat16] [--k 20] [--save <file>] [--json <file>]
    python -m holon_judge snapshot-diff <a.json> <b.json>
    python -m holon_judge snapshot --model <hf dir or id> --prompts <jsonl> [--runs 3] [--json <file>]
                                   # the snapshot trace on fresh engines: run-identical?
    python -m holon_judge bounded  --model <hf dir or id> --prompts <jsonl> [--json <file>] [--plant <bug>]
                                   # prefix cache on/off x overlap depth 1/0 on a multi-turn trace

`compare`, `l2`, `snapshot-diff`, `snapshot` and `bounded` exit with status 1 on a failed check.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import pathlib
import sys

import torch

from . import golden
from .compare import Result, compare
from .planted import PLANTS
from .recording import PromptSet, Recording


def _golden(a: argparse.Namespace) -> int:
    if a.threads:
        torch.set_num_threads(a.threads)
    rec = golden.record(a.model, PromptSet.load(a.prompts), golden.DTYPES[a.dtype], a.device, a.k)
    rec.save(a.out)
    print(json.dumps(rec.meta | {"bytes": pathlib.Path(a.out).stat().st_size}, indent=1))
    return 0


def _report(result: Result, json_out: str | None) -> int:
    print(result.table())
    if json_out:
        pathlib.Path(json_out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(json_out).write_text(json.dumps(result.to_json(), indent=1) + "\n")
    return 1 if result.passed is False else 0


def _compare(a: argparse.Namespace) -> int:
    floor = Recording.load(a.floor) if a.floor else None
    return _report(compare(Recording.load(a.golden), Recording.load(a.candidate), floor), a.json)


def _l2(a: argparse.Namespace) -> int:
    from .engine import score_model

    cand = score_model(a.model, PromptSet.load(a.prompts), golden.DTYPES[a.dtype], a.device, a.k)
    if a.save:
        cand.save(a.save)
    return _report(compare(Recording.load(a.golden), cand, Recording.load(a.floor)), a.json)


def _snapshot_diff(a: argparse.Namespace) -> int:
    from .snapshot import Snapshot, diff

    where = diff(Snapshot.load(a.a), Snapshot.load(a.b))
    print(where or "identical")
    return 1 if where else 0


def _snapshot(a: argparse.Namespace) -> int:
    import gc

    from holon.api import canonical_hash
    from holon.core.engine import Engine, EngineConfig

    from . import snapshot

    path = golden.resolve_model(a.model)
    requests = snapshot.trace(PromptSet.load(a.prompts))
    runs = []
    for _ in range(a.runs):
        engine = Engine(EngineConfig(model=str(path), dtype=golden.DTYPES[a.dtype]))
        runs.append(snapshot.record(engine, requests))
        engine.close()
        gc.collect()
    diffs = [snapshot.diff(runs[0], run) for run in runs[1:]]
    report = {"model": golden.model_meta(path)["model"], "dtype": a.dtype, "runs": a.runs,
              "hardware": golden.hardware(torch.device("cuda")), "torch": torch.__version__,
              "trace": canonical_hash([[r.ids, r.params, r.after] for r in requests]),
              "steps": len(runs[0].steps), "diffs": diffs, "run_identical": not any(diffs), **runs[0].meta}
    print(json.dumps(report, indent=1, default=str))
    if a.json:
        pathlib.Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(a.json).write_text(json.dumps(report, indent=1, default=str) + "\n")
    return 0 if report["run_identical"] else 1


def _bounded(a: argparse.Namespace) -> int:
    from holon.core.engine import Engine, EngineConfig

    from . import bounded

    path = golden.resolve_model(a.model)
    trace = bounded.Trace.from_prompts(PromptSet.load(a.prompts))

    def build(**overrides):
        return Engine(EngineConfig(model=str(path), dtype=golden.DTYPES[a.dtype], max_model_len=trace.max_len,
                                   **overrides))

    configs = {f"prefix {'on' if p else 'off'}, depth {int(o)}": {"prefix_cache": p, "overlap": o}
               for p, o in ((True, True), (True, False), (False, True))}       # the baseline is off, 0
    with PLANTS[a.plant]() if a.plant else contextlib.nullcontext():
        report = bounded.compare_configs(build, configs, trace, a.k)
    report |= {"model": golden.model_meta(path)["model"], "dtype": a.dtype,
               "hardware": golden.hardware(torch.device("cuda")), "torch": torch.__version__, "plant": a.plant}
    print(bounded.table(report))
    if a.json:
        pathlib.Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(a.json).write_text(json.dumps(report, indent=1) + "\n")
    return 0 if report["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m holon_judge")
    sub = ap.add_subparsers(required=True)

    g = sub.add_parser("golden", help="record HF teacher-forced top-k logprobs (golden or floor)")
    g.add_argument("--model", required=True, help="HF checkpoint directory or hub id")
    g.add_argument("--prompts", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--threads", type=int, default=0, help="CPU threads; pin it to re-record a golden bitwise")
    g.add_argument("--dtype", default="float32", choices=golden.DTYPES)
    g.add_argument("--device", default="cpu")
    g.add_argument("--k", type=int, default=20)
    g.set_defaults(run=_golden)

    c = sub.add_parser("compare", help="compare a recording with the golden; with --floor, pass/fail")
    c.add_argument("--golden", required=True)
    c.add_argument("--candidate", required=True)
    c.add_argument("--floor")
    c.add_argument("--json", help="write the machine-readable result here")
    c.set_defaults(run=_compare)

    e = sub.add_parser("l2", help="score the prompt set through a Holon engine and judge it")
    e.add_argument("--model", required=True, help="HF checkpoint directory or hub id")
    e.add_argument("--prompts", required=True)
    e.add_argument("--golden", required=True)
    e.add_argument("--floor", required=True)
    e.add_argument("--dtype", default="bfloat16", choices=golden.DTYPES)
    e.add_argument("--device", default="cuda")
    e.add_argument("--k", type=int, default=20)
    e.add_argument("--save", help="also save the candidate recording")
    e.add_argument("--json", help="write the machine-readable result here")
    e.set_defaults(run=_l2)

    s = sub.add_parser("snapshot-diff", help="check two snapshots for byte identity")
    s.add_argument("a")
    s.add_argument("b")
    s.set_defaults(run=_snapshot_diff)

    t = sub.add_parser("snapshot", help="run the snapshot trace on fresh engines and check it is run-identical")
    t.add_argument("--model", required=True, help="HF checkpoint directory or hub id")
    t.add_argument("--prompts", required=True)
    t.add_argument("--dtype", default="bfloat16", choices=golden.DTYPES)
    t.add_argument("--runs", type=int, default=3)
    t.add_argument("--json", help="write the machine-readable report here")
    t.set_defaults(run=_snapshot)

    b = sub.add_parser("bounded", help="judge engine configurations against the reference on a multi-turn trace")
    b.add_argument("--model", required=True, help="HF checkpoint directory or hub id")
    b.add_argument("--prompts", required=True)
    b.add_argument("--dtype", default="bfloat16", choices=golden.DTYPES)
    b.add_argument("--k", type=int, default=20)
    b.add_argument("--json", help="write the machine-readable report here")
    b.add_argument("--plant", choices=sorted(PLANTS), help="calibration: run with a planted bug (must fail)")
    b.set_defaults(run=_bounded)

    a = ap.parse_args(argv)
    return a.run(a)


if __name__ == "__main__":
    sys.exit(main())
