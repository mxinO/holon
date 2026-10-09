"""Judge command line.

    python -m holon_judge golden   --model <hf dir or id | .gguf> [--config <hf dir or id> [--q8-1 mmq|mmvq]]
                                   --prompts <jsonl> --out <file> [--threads N] [--k 20]
                                   [--dtype bfloat16 --device cuda]          # the floor: HF in the activation dtype
    python -m holon_judge compare  --golden <file> --candidate <file> [--floor <file>] [--json <file>]
    python -m holon_judge l2       --model <hf dir or id | .gguf> --prompts <jsonl> --golden <file> --floor <file>
                                   [--source gguf_dense] [--max-step-tokens N] [--plant <bug>]
                                   [--dtype bfloat16] [--k 20] [--save <file>] [--json <file>]
    python -m holon_judge retokenize --prompts <jsonl> --model <hf dir or id> --out <jsonl>
                                   # the same texts in another model's tokenizer (each must decode back exactly)
    python -m holon_judge snapshot-diff <a.json> <b.json>
    python -m holon_judge snapshot --model <hf dir or id> --prompts <jsonl> [--runs 3] [--json <file>]
                                   # the snapshot trace on fresh engines: run-identical?
    python -m holon_judge bounded  --model <hf dir or id | .gguf> --prompts <jsonl> [--json <file>] [--plant <bug>]
                                   [--sessions N] [--graphs-vs-eager]
                                   # prefix cache on/off x overlap depth 1/0 on a multi-turn trace; or decode
                                   # graphs on vs off at a batch of N sessions

`compare`, `l2`, `snapshot-diff`, `snapshot` and `bounded` exit with status 1 on a failed check.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import pathlib
import shlex
import sys

import torch

from . import golden
from .compare import Result, compare
from .planted import PLANTS
from .recording import PromptSet, Recording


def _golden(a: argparse.Namespace) -> int:
    if a.threads:
        torch.set_num_threads(a.threads)
    rec = golden.record(a.model, PromptSet.load(a.prompts), golden.DTYPES[a.dtype], a.device, a.k, a.config,
                        a.q8_1, a.pure_hf)
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

    if a.threads:
        torch.set_num_threads(a.threads)
    with PLANTS[a.plant]() if a.plant else contextlib.nullcontext():
        cand = score_model(a.model, PromptSet.load(a.prompts), golden.DTYPES[a.dtype], a.device, a.k, a.source,
                           a.max_step_tokens, a.state_memory_gib and int(a.state_memory_gib * 2**30), a.max_seqs)
    if a.save:
        cand.save(a.save)
    return _report(compare(Recording.load(a.golden), cand, Recording.load(a.floor)), a.json)


def _structure(a: argparse.Namespace) -> int:
    from holon.core.engine import Engine, EngineConfig

    from .structure import mismatches

    path = golden.resolve_model(a.model)
    with PLANTS[a.plant]() if a.plant else contextlib.nullcontext():
        engine = Engine(EngineConfig(model=str(path), dtype=golden.DTYPES[a.dtype], max_model_len=1024))
    bad = mismatches(engine, path)
    print(json.dumps({"model": golden.model_meta(path)["model"], "plant": a.plant, "mismatches": len(bad),
                      "first": bad[:8]}))
    return 1 if bad else 0


def _retokenize(a: argparse.Namespace) -> int:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(golden.resolve_model(a.model) / "tokenizer.json"))
    rows = [json.loads(line) for line in pathlib.Path(a.prompts).read_text().splitlines() if line]
    old = Tokenizer.from_file(str(golden.resolve_model(a.source_model) / "tokenizer.json")) if a.source_model else None
    for row in rows:
        ids = tok.encode(row["text"], add_special_tokens=False).ids
        if tok.decode(ids, skip_special_tokens=False) != row["text"]:
            raise ValueError(f"{row['id']}: the text does not round-trip through the tokenizer")
        if old is not None:          # where generation started: the same text prefix, in the new tokens
            prefix = old.decode(row["ids"][:row["gen_start"]], skip_special_tokens=False)
            row["gen_start"] = len(tok.encode(prefix, add_special_tokens=False).ids)
        row["ids"], row["n_tokens"] = ids, len(ids)
    out = pathlib.Path(a.out)
    out.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    print(f"{len(rows)} prompts, {sum(r['n_tokens'] for r in rows)} tokens -> {out}")
    return 0


def _snapshot_diff(a: argparse.Namespace) -> int:
    from .snapshot import Snapshot, diff

    where = diff(Snapshot.load(a.a), Snapshot.load(a.b))
    print(where or "identical")
    return 1 if where else 0


def _snapshot(a: argparse.Namespace) -> int:
    import gc

    from holon.api import canonical_hash
    from holon.core.cli import engine_config, engine_flags
    from holon.core.engine import Engine

    from . import snapshot

    path = golden.resolve_model(a.model)
    requests = snapshot.trace(PromptSet.load(a.prompts))
    runs = []
    flags = argparse.ArgumentParser()
    engine_flags(flags)
    config = engine_config(str(path), flags.parse_args(shlex.split(a.engine_args)))
    for _ in range(a.runs):
        engine = Engine(dataclasses.replace(config, dtype=golden.DTYPES[a.dtype]))
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

    from . import bounded, gguf

    path = golden.resolve_model(a.model)
    trace = bounded.Trace.from_prompts(PromptSet.load(a.prompts), sessions=a.sessions)

    def build(**overrides):
        return Engine(EngineConfig(model=str(path), dtype=golden.DTYPES[a.dtype], max_model_len=trace.max_len,
                                   **overrides))

    if a.graphs_vs_eager:                                # every decode step a batch of `sessions` rows, padded
        baseline = bounded.BASELINE | {"cuda_graphs": False}
        configs = {"graphs on": bounded.BASELINE | {"cuda_graphs": True}}
    else:
        baseline = bounded.BASELINE
        configs = {f"prefix {'on' if p else 'off'}, depth {int(o)}": {"prefix_cache": p, "overlap": o}
                   for p, o in ((True, True), (True, False), (False, True))}   # the baseline is off, 0
    with PLANTS[a.plant]() if a.plant else contextlib.nullcontext():
        report = bounded.compare_configs(build, configs, trace, a.k, baseline)
    meta = gguf.file_meta(path) if path.suffix == ".gguf" else golden.model_meta(path)
    report |= {"model": meta["model"], "dtype": a.dtype, "graphs_vs_eager": a.graphs_vs_eager,
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
    g.add_argument("--model", required=True, help="HF checkpoint directory or hub id, or a .gguf file")
    g.add_argument("--config", help="for a .gguf: the HF checkpoint (dir or hub id) it was made from")
    g.add_argument("--q8-1", choices=["mmq", "mmvq"], help="for a .gguf: Tier B's floor on that kernel path")
    g.add_argument("--pure-hf", action="store_true", help="an MoE floor with HF's own MoE block (information only)")
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
    e.add_argument("--model", required=True, help="HF checkpoint directory or hub id, or a .gguf file")
    e.add_argument("--source", help="weight source (default: detected); gguf_dense for GGUF Tier A")
    e.add_argument("--max-step-tokens", type=int, help="tokens per engine step (default: whole prompts)")
    e.add_argument("--max-seqs", type=int, default=64)
    e.add_argument("--state-memory-gib", type=float, help="state pools on the CPU (Tier A), which cannot be measured")
    e.add_argument("--threads", type=int, default=0, help="CPU threads (Tier A)")
    e.add_argument("--prompts", required=True)
    e.add_argument("--golden", required=True)
    e.add_argument("--floor", required=True)
    e.add_argument("--dtype", default="bfloat16", choices=golden.DTYPES)
    e.add_argument("--device", default="cuda")
    e.add_argument("--k", type=int, default=20)
    e.add_argument("--save", help="also save the candidate recording")
    e.add_argument("--json", help="write the machine-readable result here")
    e.add_argument("--plant", choices=sorted(PLANTS), help="calibration: run with a planted bug (must fail)")
    e.set_defaults(run=_l2)

    st = sub.add_parser("structure", help="every stacked expert of an HF MoE load equals the checkpoint's tensor")
    st.add_argument("--model", required=True, help="HF checkpoint directory or hub id")
    st.add_argument("--dtype", default="bfloat16", choices=golden.DTYPES)
    st.add_argument("--plant", choices=sorted(PLANTS), help="calibration: run with a planted bug (must fail)")
    st.set_defaults(run=_structure)

    rt = sub.add_parser("retokenize", help="a prompt set's texts in another model's tokenizer")
    rt.add_argument("--prompts", required=True)
    rt.add_argument("--model", required=True, help="HF checkpoint directory or hub id with tokenizer.json")
    rt.add_argument("--source-model", help="the tokenizer the set was built with (to carry gen_start over)")
    rt.add_argument("--out", required=True)
    rt.set_defaults(run=_retokenize)

    s = sub.add_parser("snapshot-diff", help="check two snapshots for byte identity")
    s.add_argument("a")
    s.add_argument("b")
    s.set_defaults(run=_snapshot_diff)

    t = sub.add_parser("snapshot", help="run the snapshot trace on fresh engines and check it is run-identical")
    t.add_argument("--model", required=True, help="HF checkpoint directory or hub id")
    t.add_argument("--prompts", required=True)
    t.add_argument("--dtype", default="bfloat16", choices=golden.DTYPES)
    t.add_argument("--runs", type=int, default=3)
    t.add_argument("--engine-args", default="", help="EngineConfig flags, as for `holon serve` (a class's settings)")
    t.add_argument("--json", help="write the machine-readable report here")
    t.set_defaults(run=_snapshot)

    b = sub.add_parser("bounded", help="judge engine configurations against the reference on a multi-turn trace")
    b.add_argument("--model", required=True, help="HF checkpoint directory or hub id, or a .gguf file")
    b.add_argument("--prompts", required=True)
    b.add_argument("--sessions", type=int, default=48, help="concurrent sessions: the decode batch")
    b.add_argument("--graphs-vs-eager", action="store_true", help="judge graphed decode against an eager baseline")
    b.add_argument("--dtype", default="bfloat16", choices=golden.DTYPES)
    b.add_argument("--k", type=int, default=20)
    b.add_argument("--json", help="write the machine-readable report here")
    b.add_argument("--plant", choices=sorted(PLANTS), help="calibration: run with a planted bug (must fail)")
    b.set_defaults(run=_bounded)

    a = ap.parse_args(argv)
    return a.run(a)


if __name__ == "__main__":
    sys.exit(main())
