"""What grammar.xgrammar's vetting and compiling cost (docs/m05-grammar.md D5, §6; evidence record): a vetting worker's
start — spawn to ready, which is also the respawn time after a kill — and its resident set; per constraint, the
worker's compile time, compiled size and peak resident growth, beside the engine process's own compile on a thread;
and the share of each subset a strict request would have accepted (D9's reject list). Constraints: a tool schema,
json_object, a 10-tool structural tag, a regex, and mid-size JSONSchemaBench schemas (never Github_hard, Snowplow or
Kubernetes: their compiles take seconds to minutes and gigabytes). CPU only, one worker at a time, per tokenizer.

Usage (from the repository root):
  python bench/grammar_vetting.py --model <checkpoint dir> [--model …] --jsb <JSONSchemaBench data dir>
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import statistics
import time
from importlib.metadata import version

from run import ROOT, git, gpu_class, model_id

SUBSETS = ("Glaiveai2K", "Github_easy", "Github_medium", "WashingtonPost")
TOOL = {"type": "object", "properties": {
    "command": {"type": "string"}, "timeout": {"type": "number"}, "description": {"type": "string"},
    "run_in_background": {"type": "boolean"}, "mode": {"enum": ["a", "b"]},
    "env": {"type": "array", "items": {"type": "object", "required": ["k", "v"], "additionalProperties": False,
                                       "properties": {"k": {"type": "string"}, "v": {"type": "string"}}}}},
    "required": ["command"], "additionalProperties": False}
FIXED = {
    "tool_schema": {"type": "json_schema", "schema": TOOL, "strict": True},
    "json_object": {"type": "json_object"},
    "tag_10_tools": {"type": "triggered_tags", "triggers": ["<tool_call>"], "tags": [
        {"type": "tag", "begin": f'<tool_call>\n{{"name": "tool_{i}", "arguments": ',
         "content": {"type": "json_schema", "schema": TOOL}, "end": "}\n</tool_call>"} for i in range(10)]},
    "regex": {"type": "regex", "pattern": r"\d{4}-\d{2}-\d{2}"},
}


def _dist(xs: list[float], digits: int = 1) -> dict | None:
    if not xs:
        return None
    xs = sorted(xs)
    return {"p50": round(statistics.median(xs), digits), "p90": round(xs[int(0.9 * (len(xs) - 1))], digits),
            "max": round(xs[-1], digits)}


def measure(model: str, jsb: pathlib.Path, per_subset: int) -> dict:
    import pandas as pd
    import xgrammar as xgr
    from holon_plugins.grammar.xgrammar.constraint import grammar, strict_violation
    from holon_plugins.grammar.xgrammar.vetting import Vetting
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    config = json.loads((pathlib.Path(model) / "config.json").read_text())
    vocab = config.get("vocab_size") or config["text_config"]["vocab_size"]
    stops = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    info = xgr.TokenizerInfo.from_huggingface(tok, vocab_size=vocab, stop_token_ids=stops)
    vetting = Vetting(info.serialize_json(), vocab, stops, workers=1)
    compiler = xgr.GrammarCompiler(info, max_threads=2, cache_enabled=False)

    def vet(c: dict):
        key = json.dumps(c)
        vetting.submit(key, key)
        while not (done := vetting.drain()):
            time.sleep(0.01)
        return done[0][1]

    def row(c: dict) -> dict:
        outcome = vet(c)
        out = {"ok": outcome.ok, "why": outcome.message[:60] if not outcome.ok else None}
        if outcome.ok:
            start = time.perf_counter()
            compiler.compile_grammar(grammar(c))
            out |= {"worker_ms": outcome.seconds * 1e3, "engine_ms": (time.perf_counter() - start) * 1e3,
                    "size_mb": outcome.size / 2**20, "growth_mb": outcome.growth / 2**20}
        return out

    fixed = {name: {k: round(v, 2) if isinstance(v, float) else v for k, v in row(c).items()}
             for name, c in FIXED.items()}
    vetting.timeout = 0.05                                    # three kills: the respawn time
    for _ in range(3):
        vet({"type": "json_schema", "schema": {"type": "string", "maxLength": 2000}})
    vetting.timeout = 10.0
    subsets = {}
    for subset in SUBSETS:
        frame = pd.read_parquet(jsb / subset)
        schemas = [json.loads(s) for s in frame.json_schema if len(s) < 8000][:per_subset]
        rows = [row({"type": "json_schema", "schema": s}) for s in schemas]
        ok = [r for r in rows if r["ok"]]
        subsets[subset] = {
            "n": len(rows), "vetted": len(ok), "refused": sorted({r["why"] for r in rows if not r["ok"]}),
            "strict_accepts": sum(strict_violation(s) is None for s in schemas),
            **{key: _dist([r[key] for r in ok], 2 if key.endswith("mb") else 1)
               for key in ("worker_ms", "engine_ms", "size_mb", "growth_mb")}}
    while len(vetting.starts) < 4:
        time.sleep(0.05)
    vetting.shutdown()
    starts = [{"ready_s": round(s, 2), "rss_mb": round(rss / 2**20)} for s, rss in vetting.starts]
    return {"model": model_id(model), "vocab": vocab, "worker_starts": starts, "constraints": fixed,
            "subsets": subsets}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, help="checkpoint directory (its tokenizer)")
    parser.add_argument("--jsb", required=True, type=pathlib.Path, help="JSONSchemaBench data directory")
    parser.add_argument("--per-subset", type=int, default=40)
    parser.add_argument("--gpu", type=int, default=0, help="only names the host's fingerprint class")
    args = parser.parse_args()
    results = []
    for model in args.model:
        results.append(measure(model, args.jsb, args.per_subset))
        print(json.dumps(results[-1], indent=1), flush=True)
    date, gpu = datetime.date.today().isoformat(), gpu_class(args.gpu)
    models = " ".join(f"--model {model_id(m)}" for m in args.model)
    record = {"workload": "grammar_vetting", "date": date, "gpu_class": gpu,
              "command": f"python bench/grammar_vetting.py {models} --jsb <JSONSchemaBench> "
                         f"--per-subset {args.per_subset}",
              "commit": git("rev-parse", "HEAD"), "xgrammar": version("xgrammar"), "results": results}
    out = ROOT / "evidence" / f"grammar_vetting_{gpu}_{date}.json"
    out.write_text(json.dumps(record, indent=1) + "\n")
    print(out)


if __name__ == "__main__":
    main()
