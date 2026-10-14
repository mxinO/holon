"""What the host-token gate costs a constrained request (docs/m05-grammar.md D2; evidence record).

Identical tokens: every engine serves the same greedy requests (natural-text prompts cut from `--corpus`, 256 tokens
out, `ignore_eos`); configuration `none` selects no logits processor, `A` and `A′` select the test-only toy
(`testing.toy_mask`) with its `apply` a no-op, and open the constrained requests with a marker token that never
occurs, so their masks allow everything and only the gate changes their steps: under A′ the engine commits a
constrained row before scheduling the next step (`Engine.commit_for_constrained`), under A the sequence sits a step
out. Workloads: `single` (one request at a time, constrained unless `none`), `mixed` (four at once: one constrained,
three free). The state pools are pinned to the bytes the first engine with the processor measured, so every engine has
the same pool; `--rounds` interleaved rounds of fresh engines (none, A, A′), one at a time, CUDA waits sleeping (as
`bench/run.py --blocking-sync`), with the engine thread's CPU time per phase of each step (`schedule`, `fill`,
`launch` without the fill, `commit`, and the whole `step`).

Before the rounds, `--sizing-rounds` interleaved pairs of engines sized as deployed (`memory_fraction`), without and
with the processor selected, give what selecting it costs: KV tokens, graphs, capture time (`--rounds 0`: only
that). The toy has neither a real mask fill nor a kernel, so the record also measures grammar.xgrammar's: its `fill` per
constrained step over JSON-schema requests on the same engine flags (drafts included, refused drafts counted), and its
Triton `apply` per call for the row counts of a step, replayed from a CUDA graph.

Usage (from the repository root):
  python bench/gate_cost.py --gpu 0 --model <checkpoint> --corpus judge/prompts/v1.jsonl --name 8b \\
      --engine-args "--max-seqs 4 --max-model-len 768 --max-step-tokens 2048"
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import gc
import hashlib
import json
import multiprocessing
import os
import shlex
import statistics
import time
from unittest import mock

from run import ROOT, git, gpu_class, model_id, public

PROMPT, OUTPUT, SINGLE, MIXED = 256, 256, 4, 2        # tokens in and out; single-stream requests, mixed batches
CONFIGS = ("none", "A", "A′")
PHASES = ("schedule", "fill", "launch", "commit", "step")
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["name", "age", "email", "hobbies", "address"],
          "properties": {"name": {"type": "string"}, "age": {"type": "integer"}, "email": {"type": "string"},
                         "hobbies": {"type": "array", "items": {"type": "string"}},
                         "address": {"type": "object", "additionalProperties": False,
                                     "required": ["street", "city", "country"],
                                     "properties": {k: {"type": "string"} for k in ("street", "city", "country")}}}}
ASKS = ["a fictional software engineer from Lisbon", "a retired sailor who collects maps",
        "a young violinist living in Kyoto", "a beekeeper in rural Ontario"]


def _engine(args, cfg_name: str, state_memory: int | None, processor: str | None = None):
    from holon.core.cli import engine_config, engine_flags
    from holon.core.engine import Engine
    flags = argparse.ArgumentParser()
    engine_flags(flags)
    cfg = engine_config(args.model, flags.parse_args(shlex.split(args.engine_args)))
    if processor is None and cfg_name != "none":
        processor = "testing.toy_mask"
    slots = cfg.slots | ({"logits_processor": processor} if processor else {})
    engine = Engine(dataclasses.replace(cfg, slots=slots, state_memory=state_memory))
    engine.commit_for_constrained = cfg_name != "A"
    return engine


def _timed(engine) -> None:
    """Wrap the engine's phases: each call adds its engine-thread CPU time (µs) to `engine.phases` (the current
    step's), the launch without the mask fill inside it."""
    def wrap(obj, attr: str, phase: str) -> None:
        inner = getattr(obj, attr)

        def timed(*a, **kw):
            cpu, fill = time.thread_time(), engine.phases["fill"]
            try:
                return inner(*a, **kw)
            finally:
                spent = (time.thread_time() - cpu) * 1e6
                engine.phases[phase] += spent - (engine.phases["fill"] - fill if phase == "launch" else 0.0)
        setattr(obj, attr, timed)

    wrap(engine.sched, "schedule", "schedule")
    if engine.processor is not None:
        wrap(engine.processor, "fill", "fill")
    wrap(engine.executor, "launch", "launch")
    wrap(engine, "_commit", "commit")


def _serve(engine, requests: list[tuple[list[int], object]]) -> tuple[list[dict], list[dict]]:
    """Run `requests` at once: per request its tokens and when they arrived; per step its phases (µs)."""
    rids = [engine.add(p, params) for p, params in requests]
    out = {r: {"tokens": [], "times": []} for r in rids}
    steps = []
    while engine.has_work():
        engine.phases = dict.fromkeys(PHASES, 0.0)
        start, cpu = time.perf_counter(), time.thread_time()
        deltas = engine.step()
        now = time.perf_counter()
        engine.phases["step"] = (time.thread_time() - cpu) * 1e6
        steps.append({"wall_us": (now - start) * 1e6, **engine.phases})
        for d in deltas:
            out[d.rid]["tokens"] += d.token_ids
            out[d.rid]["times"] += [now] * len(d.token_ids)
            assert d.finish_reason in (None, "length"), d
    return [out[r] for r in rids], steps


def _digest(tokens: list[int]) -> str:
    return hashlib.sha256(json.dumps(tokens).encode()).hexdigest()[:16]


def _kv_tokens(engine) -> int:
    return min(p.capacity * p.tokens_per_unit for p in engine.ledger.pools.values() if p.tokens_per_unit)


def _tpot(times: list[float], until: float | None = None) -> float | None:
    times = [t for t in times if until is None or t <= until]
    return (times[-1] - times[0]) / (len(times) - 1) * 1e3 if len(times) > 1 else None


def gate_round(args, cfg_name: str, state_memory: int, corpus: list[int]) -> dict:
    from holon.api import SamplingParams
    from holon_testing.toy_mask.plugin import KEY
    engine = _engine(args, cfg_name, state_memory)
    engine.phases = dict.fromkeys(PHASES, 0.0)
    _timed(engine)
    marker = engine.spec.vocab - 1                   # a padding id: never in a prompt, never sampled
    free = SamplingParams(temperature=0.0, max_tokens=OUTPUT, ignore_eos=True)
    constrained = dataclasses.replace(free, ext={KEY: {"marker": marker}}) if cfg_name != "none" else free

    def prompt(i: int) -> list[int]:
        at = (i * 7919) % (len(corpus) - PROMPT)
        return corpus[at: at + PROMPT]

    _serve(engine, [(prompt(1000), free)])           # warmup (compiles, caches): other prompt tokens
    before = engine.stats()
    result = {"single": [], "mixed": []}
    for i in range(SINGLE):
        (req,), steps = _serve(engine, [(prompt(i), constrained)])
        decode = steps[2:]                           # past the prefill and the first decode step
        result["single"].append({"tokens": _digest(req["tokens"]), "tpot_ms": _tpot(req["times"]),
                                 "steps": len(steps), "phase_us_mean": {
                                     k: statistics.fmean(s[k] for s in decode) for k in (*PHASES, "wall_us")}})
    for b in range(MIXED):
        reqs, steps = _serve(engine, [(prompt(100 + 4 * b + j), constrained if j == 0 else free) for j in range(4)])
        free_end = max(r["times"][-1] for r in reqs[1:])
        result["mixed"].append({"tokens": [_digest(r["tokens"]) for r in reqs],
                                "tpot_ms": [_tpot(r["times"]) for r in reqs],
                                "constrained_tpot_ms_while_free": _tpot(reqs[0]["times"], free_end),
                                "constrained_tokens_while_free": sum(t <= free_end for t in reqs[0]["times"]),
                                "steps": len(steps)})
    after = engine.stats()
    result["stats"] = {k: v - before.get(k, 0) for k, v in after.items()
                       if k.startswith("spec_") or k.endswith("rows_filled")}
    result["kv_tokens"] = _kv_tokens(engine)
    engine.close()
    del engine
    gc.collect()                                     # the engine's cycles hold its weights until then
    return result


def grammar_costs(args, state_memory: int) -> dict:
    """grammar.xgrammar's mask fill per constrained step (engine thread, µs) over JSON-schema requests, and its
    `apply` per call (µs, CUDA graph replay) for the step's mask rows."""
    import torch
    from holon.api import SamplingParams
    from holon_plugins.grammar.xgrammar import kernel
    engine = _engine(args, "A′", state_memory, processor="grammar.xgrammar")
    tok = engine.source.tokenizer()
    fills, inner = [], engine.processor.fill

    def fill(rows, buf):
        start = time.perf_counter()
        inner(rows, buf)
        fills.append(((time.perf_counter() - start) * 1e6, sum(r.n for r in rows)))
    engine.processor.fill = fill
    params = SamplingParams(temperature=0.0, max_tokens=OUTPUT,
                            ext={"grammar.constraint@1": {"type": "json_schema", "schema": SCHEMA, "strict": True}})
    outputs = []
    for ask in ASKS:
        text = (f"<|im_start|>user\nDescribe {ask} as a JSON object with name, age, email, hobbies and address."
                "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        before = len(fills)
        out = engine.generate([tok.encode(text)], params)[0]
        outputs.append({"tokens": len(out), "steps": len(fills) - before, "text": tok.decode(out)[:200]})
    stats = engine.stats()
    vocab, width, k = engine.spec.vocab, engine.processor.mask_width, engine.cfg.max_draft_tokens
    engine.close()
    del engine
    gc.collect()
    apply_us = {}
    for rows in sorted({1, 1 + k}):
        logits = torch.zeros(rows, vocab, device="cuda")
        mask = torch.randint(-2**31, 2**31 - 1, (rows, width), dtype=torch.int32, device="cuda")
        index = torch.arange(rows, device="cuda")
        kernel.apply(logits, mask, index)            # compile
        graph, reps = torch.cuda.CUDAGraph(), 100
        with torch.cuda.graph(graph):
            for _ in range(reps):
                kernel.apply(logits, mask, index)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(7):
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end) * 1e3 / reps)
        apply_us[rows] = round(statistics.median(times), 2)
    per_step = [us for us, _ in fills]
    return {"fill_us_per_step": {"mean": round(statistics.fmean(per_step), 1),
                                 "p50": round(statistics.median(per_step), 1), "max": round(max(per_step), 1)},
            "columns_per_step_mean": round(statistics.fmean(n for _, n in fills), 2), "fills": len(fills),
            "drafts_refused": stats.get("grammar.xgrammar.drafts_refused"), "spec_drafted": stats.get("spec_drafted"),
            "apply_us_per_call": apply_us, "outputs": outputs}


def unpinned(args, cfg_name: str) -> dict:
    """An engine sized as deployed (`memory_fraction`), in a process of its own (`main` spawns one per engine: one
    process's earlier engines bias the free memory measured): its state bytes, KV tokens, graphs, capture time."""
    import torch
    from holon.core.executor import Executor
    from holon_testing.toy_mask.plugin import ToyMask
    mock.patch.object(ToyMask, "apply", lambda self, logits, mask, rows: None).start()
    seen, measure, capture = {}, Executor.state_memory, Executor.capture_graphs

    def state_memory(self, *a):
        seen["state_memory"] = measure(self, *a)
        return seen["state_memory"]

    def capture_graphs(self):
        torch.cuda.synchronize()
        start = time.perf_counter()
        capture(self)
        torch.cuda.synchronize()
        seen["capture_s"] = round(time.perf_counter() - start, 2)
    with mock.patch.object(Executor, "state_memory", state_memory), \
            mock.patch.object(Executor, "capture_graphs", capture_graphs):
        engine = _engine(args, cfg_name, None)
    seen |= {"graphs": len(engine.executor.graphs), "kv_tokens": _kv_tokens(engine)}
    return seen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", required=True, help="a JSONL prompt set (natural text)")
    parser.add_argument("--name", required=True)
    parser.add_argument("--engine-args", default="", help="engine flags, as for `holon serve`")
    parser.add_argument("--rounds", type=int, default=4, help="0: only the sizing (KV tokens, graphs, capture)")
    parser.add_argument("--sizing-rounds", type=int, default=1, help="interleaved engines sized as deployed")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    with multiprocessing.get_context("spawn").Pool(1, maxtasksperchild=1) as pool:   # before this process uses CUDA
        sizing = [{c: pool.apply(unpinned, (args, c)) for c in ("none", "A′")} for _ in range(args.sizing_rounds)]
    from worker import blocking_sync
    blocking_sync()
    import workloads
    from holon_testing.toy_mask.plugin import ToyMask
    corpus = workloads.corpus(args.corpus)
    mock.patch.object(ToyMask, "apply", lambda self, logits, mask, rows: None).start()    # the toy's kernel: none

    selected = {c: [r[c] for r in sizing] for c in ("none", "A′")}   # what selecting the processor costs
    state_memory = int(min(r["state_memory"] for r in selected["A′"]) * 0.98)   # every engine's pool
    print("sizing:", json.dumps(selected), flush=True)
    rounds = []
    for r in range(args.rounds):
        rounds.append({c: gate_round(args, c, state_memory, corpus) for c in CONFIGS})
        print(f"round {r}:", {c: [s["tpot_ms"] for s in rounds[-1][c]["single"]] for c in CONFIGS}, flush=True)
    costs = grammar_costs(args, state_memory) if rounds else None
    date, gpu = datetime.date.today().isoformat(), gpu_class(args.gpu)
    record = {"workload": f"gate_cost_{args.name}", "date": date, "gpu_class": gpu, "model": model_id(args.model),
              "command": " ".join(["python bench/gate_cost.py", *(public(a, args.model) for a in [
                  "--gpu", str(args.gpu), "--model", args.model, "--corpus", args.corpus, "--name", args.name,
                  "--rounds", str(args.rounds), "--sizing-rounds", str(args.sizing_rounds)])])
              + f" --engine-args {shlex.quote(args.engine_args)}",
              "commit": git("rev-parse", "HEAD"), "state_memory": state_memory, "processor_selected": selected,
              **({"summary": summarize(rounds, costs), "grammar": costs, "rounds": rounds} if rounds else {})}
    out = ROOT / "evidence" / f"gate_{'cost' if rounds else 'sizing'}_{args.name}_{gpu}_{date}.json"
    out.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record.get("summary"), indent=1))
    print(out)


def summarize(rounds: list[dict], costs: dict) -> dict:
    """Medians over the rounds after the first (with their spread), the identical-token precondition, and D2's
    numbers: the single-stream penalty of A′ (with grammar.xgrammar's fill and apply per step added) and the free
    rows' slowdown in the mixed batch under A and A′."""
    kept = rounds[1:] or rounds

    def med(values: list[float]) -> dict:
        return {"median": round(statistics.median(values), 3), "spread": round(max(values) - min(values), 3)}

    out = {"identical_tokens": all(
        r[c]["single"][i]["tokens"] == r["none"]["single"][i]["tokens"] and
        r[c]["mixed"][b]["tokens"] == r["none"]["mixed"][b]["tokens"]
        for r in rounds for c in CONFIGS for i in range(SINGLE) for b in range(MIXED)),
        "identical_spec_rows": all(r[c]["stats"].get(k) == r["none"]["stats"].get(k) for r in rounds for c in CONFIGS
                                   for k in {*r[c]["stats"], *r["none"]["stats"]} if k.startswith("spec_rows")),
        "spec_rows": rounds[0]["none"]["stats"]}
    for c in CONFIGS:
        single = [statistics.fmean(s["tpot_ms"] for s in r[c]["single"]) for r in kept]
        free = [statistics.fmean(t for m in r[c]["mixed"] for t in m["tpot_ms"][1:]) for r in kept]
        cons = [statistics.fmean(m["tpot_ms"][0] for m in r[c]["mixed"]) for r in kept]
        cons_w = [statistics.fmean(m["constrained_tpot_ms_while_free"] for m in r[c]["mixed"]) for r in kept]
        phases = {k: round(statistics.median(statistics.fmean(s["phase_us_mean"][k] for s in r[c]["single"])
                                             for r in kept), 1) for k in (*PHASES, "wall_us")}
        steps = statistics.fmean(s["steps"] for r in kept for s in r[c]["single"])
        out[c] = {"single_tpot_ms": med(single), "mixed_free_tpot_ms": med(free),
                  "mixed_constrained_tpot_ms": med(cons),
                  "mixed_constrained_tpot_ms_while_free": med(cons_w), "single_steps_per_request": steps,
                  "single_phase_us_mean": phases}
    none = out["none"]["single_tpot_ms"]["median"]
    tokens_per_step = OUTPUT / out["A′"]["single_steps_per_request"]
    added = (costs["fill_us_per_step"]["mean"] - out["A′"]["single_phase_us_mean"]["fill"]   # the toy's own fill
             + costs["apply_us_per_call"][max(costs["apply_us_per_call"])]) / 1e3
    out["d2"] = {
        "single_penalty_toy": round(out["A′"]["single_tpot_ms"]["median"] / none, 4),
        "single_penalty_with_fill_apply": round((out["A′"]["single_tpot_ms"]["median"] + added / tokens_per_step)
                                                / none, 4),
        "free_slowdown": {c: round(out[c]["mixed_free_tpot_ms"]["median"] / out["none"]["mixed_free_tpot_ms"]["median"],
                                   4) for c in ("A", "A′")},
        "constrained_vs_free_under": {c: round(out[c]["mixed_constrained_tpot_ms_while_free"]["median"]
                                               / out[c]["mixed_free_tpot_ms"]["median"], 4) for c in ("A", "A′")}}
    return out


if __name__ == "__main__":
    main()
