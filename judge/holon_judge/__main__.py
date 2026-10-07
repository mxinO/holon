"""Judge command line.

    python -m holon_judge golden   --model <hf dir or id> --prompts <jsonl> --out <file> [--threads N] [--k 20]
                                   [--dtype bfloat16 --device cuda]          # the floor: HF in the activation dtype
    python -m holon_judge compare  --golden <file> --candidate <file> [--floor <file>] [--json <file>]
    python -m holon_judge l2       --model <hf dir or id> --prompts <jsonl> --golden <file> --floor <file>
                                   [--dtype bfloat16] [--k 20] [--save <file>] [--json <file>]
    python -m holon_judge snapshot-diff <a.json> <b.json>

`compare`, `l2` and `snapshot-diff` exit with status 1 on a failed check.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

from . import golden
from .compare import Result, compare
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

    a = ap.parse_args(argv)
    return a.run(a)


if __name__ == "__main__":
    sys.exit(main())
