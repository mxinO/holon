"""JSONSchemaBench conformance of strict JSON-schema requests (docs/dev/m05-grammar.md D9, §1 acceptance 1; evidence).

The schemas: the test splits of JSONSchemaBench minus Github_trivial / Github_ultra and the 13 unsatisfiable Glaive
schemas (upstream issue 16), optionally a seeded sample. Three steps, each writing JSON lines into a work directory:

- `precompile` (no server, CPU): per schema, the server's own checks for a strict request — the constraint check with
  D9's reject list, then grammar.xgrammar's vetting worker (compile time, compiled size, peak growth; the 10 s
  timeout, the size and memory caps) — for one tokenizer. Run on an idle host.
- `run` (against `holon serve` with grammar.xgrammar): one pass of chat requests (the schema as the prompt, thinking
  off, no stop strings, `--max-tokens`), constrained by `response_format` (strict) or not (`--free`), greedy or
  sampled (`--temperature`, `--seed`); per request its status (a 400 / 503 is a rejection, with its reason), finish
  reason, tokens, TTFT, the largest gap between its stream's events, and whether its output validates
  (`validator_for(schema)`, format checks on; a request-time `error` finish counts invalid). The server's /metrics
  before and after, and its frontend process's CPU time, beside it. A 503 is retried twice after 2 s.
- `report`: the evidence record — declared coverage and rejections by reason per tokenizer, compile times, and per
  pass: valid / (accepted − truncated), truncated share (beside the unconstrained pass's), truncations ending in a
  whitespace run, TTFT cold (the server's first sight of a schema) vs warm, event gaps, frontend CPU per token.

Usage (from the repository root):
  python bench/conformance.py precompile --tokenizer <HF checkpoint> --jsb <data dir> --work <dir>
  python bench/conformance.py run --url http://127.0.0.1:8000 --jsb <data dir> --work <dir> --pass greedy \\
      [--sample 500] [--free] [--temperature 1 --seed 0] [--max-tokens 2048] [--concurrency 16] [--server-pid N]
      [--retry <pass>]       # only that pass's 503s, again (as pass '<pass>-retry'; the report merges it)
  python bench/conformance.py report --work <dir> --name <name> --gpu-class <class> --model <model id> \
      --serve "<flags>"
"""
from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import pathlib
import random
import re
import statistics
import time

SUBSETS = ("Github_easy", "Github_medium", "Github_hard", "Glaiveai2K", "JsonSchemaStore", "Kubernetes", "Snowplow",
           "WashingtonPost")
UNSATISFIABLE = {   # Glaive schemas no document satisfies (JSONSchemaBench issue 16: a oneOf over required names)
    "calculate_area_f88fb53c", "calculate_area_2f92f3ea", "calculate_area_d402e1cc", "calculate_area_95058385",
    "calculate_area_4493ae68", "calculate_area_f8e04f89", "calculate_area_8db9d7ff", "calculate_area_6fd20e8d",
    "calculate_area_e8f1513d", "calculate_area_43c11cd0", "calculate_area_3a8a9f78", "calculate_area_92ac029d",
    "calculate_area_e6818129"}
SYSTEM = "You need to generate a JSON object that matches the schema below."
INVALID_FORMATS = {"uri": "not a uri", "uuid": "1234-not-a-uuid", "duration": "P1Y2X"}   # one bad string per format
FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")
WHITESPACE_RUN = 32                    # a truncated output ending in at least this much whitespace ran away in it


def schemas(jsb: pathlib.Path, sample: int | None, seed: int, only: set[str] | None = None) -> list[dict]:
    import pandas as pd
    rows = [{"id": r.unique_id, "subset": s, "schema": r.json_schema} for s in SUBSETS
            for r in pd.read_parquet(jsb / s / "test.parquet").itertuples() if r.unique_id not in UNSATISFIABLE]
    rows = random.Random(seed).sample(rows, sample) if sample else rows
    return [r for r in rows if r["id"] in only] if only is not None else rows


def _write(path: pathlib.Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _read(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


# -- precompile --------------------------------------------------------------------------------------------------------
def precompile(args) -> None:
    import xgrammar as xgr
    from holon_plugins.grammar.xgrammar.constraint import check
    from holon_plugins.grammar.xgrammar.vetting import Vetting
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    config = json.loads((pathlib.Path(args.tokenizer) / "config.json").read_text())
    vocab = config.get("vocab_size") or config["text_config"]["vocab_size"]
    stops = sorted({tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")})
    info = xgr.TokenizerInfo.from_huggingface(tok, vocab_size=vocab, stop_token_ids=stops)
    vetting = Vetting(info.serialize_json(), vocab, stops)          # the plugin's defaults: two workers, 10 s, caps
    out, todo, pending = [], list(schemas(args.jsb, args.sample, args.sample_seed)), {}
    start = time.perf_counter()
    while todo or pending:
        while todo and len(pending) < vetting.queue:
            row = todo.pop()
            c = {"type": "json_schema", "schema": json.loads(row["schema"]), "strict": True}
            try:
                check(c)
            except ValueError as e:
                out.append({"id": row["id"], "subset": row["subset"], "accepted": False, "reason": str(e)[:300]})
                continue
            pending[row["id"]] = row
            vetting.submit(row["id"], json.dumps(c))
        for key, o in vetting.drain():
            row = pending.pop(key)
            out.append({"id": key, "subset": row["subset"], "accepted": True, "ok": o.ok, "kind": o.kind,
                        "message": o.message[:300], "ms": round(o.seconds * 1e3, 2),
                        "size_mb": round(o.size / 2**20, 3), "growth_mb": round(o.growth / 2**20, 1), "final": o.final})
        time.sleep(0.01)
    vetting.shutdown()
    print(f"{len(out)} schemas in {time.perf_counter() - start:.0f} s")
    _write(args.work / f"precompile_{vocab}.jsonl", out)


# -- run ---------------------------------------------------------------------------------------------------------------
def validate(schema: dict, text: str) -> tuple[bool | None, str]:
    """(valid, why not); None when the validator itself cannot evaluate the schema."""
    import jsonschema
    try:
        value = json.loads(text)
    except ValueError as e:
        return False, f"not JSON: {e}"
    try:
        cls = jsonschema.validators.validator_for(schema, default=jsonschema.Draft202012Validator)
        error = next(cls(schema, format_checker=cls.FORMAT_CHECKER).iter_errors(value), None)
    except Exception as e:  # noqa: BLE001 — e.g. a pattern Python's re cannot read
        return None, f"validator: {type(e).__name__}: {e}"[:300]
    return error is None, "" if error is None else f"{list(error.absolute_path)}: {error.message}"[:300]


def format_self_test() -> dict[str, bool]:
    """Every format the strict allowlist enforces is checked by the validator: an invalid string is refused."""
    import jsonschema
    from holon_plugins.grammar.xgrammar.constraint import FORMATS
    checker = jsonschema.Draft202012Validator.FORMAT_CHECKER
    result = {f: not checker.conforms(INVALID_FORMATS[f], f) for f in sorted(FORMATS)}
    if not all(result.values()):
        raise SystemExit(f"the validator does not check these formats: {result}")
    return result


def _cpu_s(pid: int | None) -> float | None:
    if pid is None:
        return None
    fields = pathlib.Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")


async def _one(client, args, row: dict, schema: dict, index: int) -> dict:
    body = {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": row["schema"]}],
            "temperature": args.temperature, "max_tokens": args.max_tokens, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    if args.seed is not None:
        body["seed"] = args.seed * 1_000_003 + index
    if not args.free:
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "output", "schema": schema,
                                                                          "strict": True}}
    out = {"id": row["id"], "subset": row["subset"]}
    for attempt in range(3):
        sent = time.perf_counter()
        async with client.stream("POST", "/v1/chat/completions", json=body) as r:
            if r.status_code != 200:
                text = (await r.aread()).decode()
                try:
                    message = json.loads(text)["error"]["message"]
                except (ValueError, KeyError, TypeError):
                    message = text
                out |= {"status": r.status_code, "error": message[:300], "attempts": attempt + 1}
                if r.status_code == 503 and attempt < 2:
                    await asyncio.sleep(2)
                    continue
                return out
            content, finish, tokens, first, last, gap = [], None, None, None, sent, 0.0
            async for line in r.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                now = time.perf_counter()
                first = first if first is not None else now
                gap, last = max(gap, now - last if first != now else 0.0), now
                event = json.loads(line[6:])
                for choice in event.get("choices", []):
                    content.append(choice.get("delta", {}).get("content") or "")
                    finish = choice.get("finish_reason") or finish
                if event.get("usage"):
                    tokens = event["usage"]["completion_tokens"]
            text = "".join(content)
            out |= {"status": 200, "finish": finish, "tokens": tokens, "ttft_ms": round((first - sent) * 1e3, 1),
                    "total_ms": round((last - sent) * 1e3, 1), "max_gap_ms": round(gap * 1e3, 1),
                    "trailing_ws": len(text) - len(text.rstrip()), "attempts": attempt + 1, "text": text}
            if finish == "stop":                     # unconstrained answers often come in a code fence
                out["valid"], out["why"] = validate(schema, FENCE.sub("", text.strip()) if args.free else text)
            elif finish == "error":
                out["valid"], out["why"] = False, "finish_reason error"
            return out
    return out


async def run_pass(args) -> None:
    import httpx
    only = None
    if args.retry:                       # the pass's requests the server turned away busy (503), again
        only = {r["id"] for r in _read(args.work / f"run_{args.retry}.jsonl") if r.get("status") == 503}
    rows = schemas(args.jsb, args.sample, args.sample_seed, only)
    tests = format_self_test()
    limits = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(base_url=args.url, timeout=None) as client:
        metrics_before, cpu_before = (await client.get("/metrics")).json(), _cpu_s(args.server_pid)

        async def one(i: int, row: dict) -> dict:
            async with limits:
                try:
                    schema = json.loads(row["schema"])
                except ValueError as e:
                    return {"id": row["id"], "subset": row["subset"], "status": None, "error": f"schema: {e}"}
                return await _one(client, args, row, schema, i)
        start = time.perf_counter()
        results = await asyncio.gather(*(one(i, row) for i, row in enumerate(rows)))
        wall = time.perf_counter() - start
        metrics_after, cpu_after = (await client.get("/metrics")).json(), _cpu_s(args.server_pid)
    _write(args.work / f"run_{args.name}.jsonl", results)
    meta = {"pass": args.name, "free": args.free, "temperature": args.temperature, "seed": args.seed,
            "max_tokens": args.max_tokens, "concurrency": args.concurrency, "sample": args.sample,
            "sample_seed": args.sample_seed, "wall_s": round(wall, 1), "format_self_test": tests,
            "metrics_before": metrics_before, "metrics_after": metrics_after,
            "frontend_cpu_s": None if cpu_before is None else round(cpu_after - cpu_before, 2)}
    (args.work / f"run_{args.name}.meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(summarize_pass(results, meta), indent=1))


# -- report ------------------------------------------------------------------------------------------------------------
def _dist(xs: list[float]) -> dict | None:
    if not xs:
        return None
    xs = sorted(xs)
    at = lambda q: xs[min(len(xs) - 1, int(q * len(xs)))]          # noqa: E731
    return {"n": len(xs), "p50": round(statistics.median(xs), 1), "p90": round(at(0.9), 1),
            "p99": round(at(0.99), 1), "max": round(xs[-1], 1)}


def reason(message: str) -> str:
    """A rejection's reason without the schema's own names: the JSON pointer and quoted names dropped."""
    message = re.sub(r"^grammar\.constraint@1: (strict schema: )?", "", message.split("\n")[0])
    message = re.sub(r"^[/#]\S*: ", "", message)
    message = re.sub(r"\(supported: .*\)", "", message)
    if "format" in message:
        return message.strip()[:120]
    message = re.sub(r"\[[^\]]*\]", "[…]", re.sub(r"'[^']*'", "'…'", message))
    return re.sub(r"\d+", "N", message).strip()[:120]


def _whitespace_run(r: dict) -> bool:
    """A truncated output that ran away in whitespace: it ends in a long run of it, or — as the frontend strips a text
    run's trailing whitespace — most of its tokens left no text (under one character per four tokens)."""
    return r["trailing_ws"] >= WHITESPACE_RUN or len(r.get("text", "").strip()) * 4 < (r.get("tokens") or 0)


def summarize_pass(rows: list[dict], meta: dict) -> dict:
    ok = [r for r in rows if r.get("status") == 200]
    rejected = [r for r in rows if r.get("status") != 200]
    truncated = [r for r in ok if r["finish"] == "length"]
    judged = [r for r in ok if r["finish"] != "length"]
    valid = [r for r in judged if r.get("valid") is True]
    invalid = [r for r in judged if r.get("valid") is False]
    unjudged = [r for r in judged if r.get("valid") is None]
    out = {"requests": len(rows), "accepted": len(ok),
           "rejected": {str(s): len([r for r in rejected if r.get("status") == s])
                        for s in sorted({r.get("status") for r in rejected}, key=str)},
           "finish": {f: sum(r["finish"] == f for r in ok) for f in sorted({r["finish"] for r in ok}, key=str)},
           "truncated": len(truncated), "truncated_share": round(len(truncated) / len(ok), 4) if ok else None,
           "truncated_whitespace_runs": [r["id"] for r in truncated if _whitespace_run(r)],
           "valid": len(valid), "invalid": len(invalid), "validator_could_not_judge": len(unjudged),
           "conformance": round(len(valid) / (len(valid) + len(invalid)), 6) if valid or invalid else None,
           "invalid_examples": [{k: r.get(k) for k in ("id", "finish", "why")} for r in invalid[:20]],
           "unjudged_examples": [{k: r.get(k) for k in ("id", "why")} for r in unjudged[:10]],
           "tokens": sum(r.get("tokens") or 0 for r in ok),
           "ttft_ms": _dist([r["ttft_ms"] for r in ok]), "max_gap_ms": _dist([r["max_gap_ms"] for r in ok]),
           "wall_s": meta["wall_s"], "attempts_over_1": sum(r.get("attempts", 1) > 1 for r in rows)}
    if meta.get("frontend_cpu_s") is not None and out["tokens"]:
        out["frontend_cpu_us_per_token"] = round(meta["frontend_cpu_s"] / out["tokens"] * 1e6, 1)
    before, after = meta["metrics_before"], meta["metrics_after"]
    out["engine_counters"] = {k: after[k] - before.get(k, 0) for k in after
                              if isinstance(after[k], int) and (k.startswith("spec_") or "." in k)}
    if out["engine_counters"].get("spec_drafted"):
        refused = out["engine_counters"].get("grammar.xgrammar.drafts_refused", 0)
        out["refused_draft_share"] = round(refused / out["engine_counters"]["spec_drafted"], 4)
    return out


def summarize_precompile(rows: list[dict]) -> dict:
    accepted = [r for r in rows if r["accepted"]]
    ok = [r for r in accepted if r["ok"]]
    failed = [r for r in accepted if not r["ok"]]
    by_reason: dict[str, int] = {}
    for r in rows:
        if not r["accepted"]:
            by_reason[reason(r["reason"])] = by_reason.get(reason(r["reason"]), 0) + 1
    subsets = {s: {"n": sum(r["subset"] == s for r in rows), "coverage": sum(r["subset"] == s for r in ok)}
               for s in SUBSETS}
    return {"schemas": len(rows), "strict_accepts": len(accepted), "compiled": len(ok),
            "declared_coverage": round(len(ok) / len(rows), 4) if rows else None, "per_subset": subsets,
            "strict_rejections": dict(sorted(by_reason.items(), key=lambda kv: -kv[1])),
            "vetting_failures": {f"{r['kind']}: {reason(r['message'])}": sum(
                f"{x['kind']}: {reason(x['message'])}" == f"{r['kind']}: {reason(r['message'])}" for x in failed)
                for r in failed},
            "compile_ms": _dist([r["ms"] for r in ok]), "size_mb": _dist([r["size_mb"] for r in ok]),
            "growth_mb_max": max((r["growth_mb"] for r in accepted), default=None),
            "compile_ms_per_schema": {r["id"]: r["ms"] for r in sorted(accepted, key=lambda r: r["id"])}}


def report(args) -> None:
    from run import ROOT, git
    work = args.work
    passes, merged = {}, {}
    for meta_path in sorted(work.glob("run_*.meta.json")):
        meta = json.loads(meta_path.read_text())
        if meta["pass"].endswith("-retry"):
            continue
        rows = _read(meta_path.with_name(meta_path.name.replace(".meta.json", ".jsonl")))
        retry = work / f"run_{meta['pass']}-retry.jsonl"
        if retry.exists():               # a 503's retry replaces it (the pass's counters are its first run's)
            again = {r["id"]: r for r in _read(retry)}
            rows = [again.get(r["id"], r) if r.get("status") == 503 else r for r in rows]
        merged[meta["pass"]] = rows
        passes[meta["pass"]] = {"settings": {k: meta[k] for k in ("free", "temperature", "seed", "max_tokens",
                                                                   "concurrency", "sample", "sample_seed")},
                                "format_self_test": meta["format_self_test"], **summarize_pass(rows, meta),
                                "rejection_reasons": _reasons(rows)}
    runs = {p: {r["id"]: r for r in rows} for p, rows in merged.items()}
    if "free" in runs:                      # the truncated shares over the schemas both passes accepted
        for p in [p for p in passes if passes[p]["settings"]["max_tokens"] == passes["free"]["settings"]["max_tokens"]]:
            both = [i for i, r in runs[p].items() if r.get("status") == 200 and runs["free"][i].get("status") == 200]
            share = {q: sum(runs[q][i]["finish"] == "length" for i in both) / len(both) for q in (p, "free")}
            passes[p]["truncated_vs_free"] = {"schemas": len(both), "share": round(share[p], 4),
                                              "free_share": round(share["free"], 4),
                                              "points": round((share[p] - share["free"]) * 100, 2)}
    precompiled = {p.stem.split("_")[1]: summarize_precompile(_read(p))
                   for p in sorted(work.glob("precompile_*.jsonl"))}
    date, gpu = datetime.date.today().isoformat(), args.gpu_class
    record = {"workload": f"conformance_{args.name}", "date": date, "gpu_class": gpu, "model": args.model,
              "serve": args.serve, "commit": git("rev-parse", "HEAD"), "method": __doc__.split("Usage")[0].strip(),
              "prompt": {"system": SYSTEM, "user": "<the schema's JSON text>"}, "precompile": precompiled,
              "passes": passes}
    out = ROOT / "evidence" / f"conformance_{args.name}_{gpu}_{date}.json"
    out.write_text(json.dumps(record, indent=1) + "\n")
    print(out)


def _reasons(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in rows:
        if r.get("status") != 200:
            key = f"{r.get('status')}: {reason(r.get('error', ''))}"
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("precompile", "run", "report"):
        p = sub.add_parser(name)
        p.add_argument("--work", type=pathlib.Path, required=True)
        if name != "report":
            p.add_argument("--jsb", type=pathlib.Path, required=True)
            p.add_argument("--sample", type=int)
            p.add_argument("--sample-seed", type=int, default=0)
    sub.choices["precompile"].add_argument("--tokenizer", required=True)
    run = sub.choices["run"]
    run.add_argument("--url", default="http://127.0.0.1:8000")
    run.add_argument("--pass", dest="name", required=True)
    run.add_argument("--free", action="store_true", help="no response_format: the unconstrained baseline")
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--seed", type=int)
    run.add_argument("--max-tokens", type=int, default=2048)
    run.add_argument("--concurrency", type=int, default=16)
    run.add_argument("--server-pid", type=int, help="the frontend process, for its CPU time")
    run.add_argument("--retry", help="a pass whose 503s to send again (this pass's name: '<pass>-retry')")
    rep = sub.choices["report"]
    for flag in ("--name", "--model", "--serve"):
        rep.add_argument(flag, required=True)
    rep.add_argument("--gpu-class", required=True, help="where the passes ran (e.g. rtx-5080)")
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    {"precompile": precompile, "run": lambda a: asyncio.run(run_pass(a)), "report": report}[args.command](args)


if __name__ == "__main__":
    main()
