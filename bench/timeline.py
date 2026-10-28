"""FN step 7 (the CPU split's integration): where a verify pass's CPU share sits against the GPU's work, on the
deployment's graphed steps. `run` serves a `v1-long` prompt with MTP and F5b on (the flags given) and profiles
`--steps` decode/verify steps between cudaProfilerStart / Stop (`nsys profile --capture-range=cudaProfilerApi
--cuda-graph-trace=node --trace=cuda,nvtx`); kernels.cpu_task marks each CPU task with an NVTX range. `summarize`
pairs each layer's `k_signal` (the GPU hands the CPU its share) with the next `cpu_task` range and the next `k_wait`
(the GPU waits for the rows): per pair the wake latency (signal end → task start), the task's time, the GPU's own
work between the signal and the wait, and the wait (its stall, when the CPU is the later). Usage:

    nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop --cuda-graph-trace=node \\
        --trace=cuda,nvtx --sample=none -o <rep> python bench/timeline.py run --model <gguf> \\
        --flags '<holon serve flags>' --out <json>
    python bench/timeline.py summarize --rep <rep>.nsys-rep --run <json> --out <json>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shlex
import statistics
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]


def engine(model: str, flags: str, **base):
    """An engine from `holon serve`-style flags over `base` (the flags win; `plugin_options` given in `base` merge on
    top of theirs)."""
    from holon.core.cli import engine_flags, engine_settings, layered
    from holon.core.engine import Engine, EngineConfig
    parser = argparse.ArgumentParser()
    engine_flags(parser, given_only=True)
    extra = base.pop("plugin_options", None)
    cfg = layered(base, engine_settings(parser.parse_args(shlex.split(flags))))
    return Engine(EngineConfig(model, **(layered(cfg, {"plugin_options": extra}) if extra else cfg)))


def run(a) -> dict:
    import torch
    from holon.api import SamplingParams

    rows = [json.loads(x) for x in (ROOT / "judge/prompts/v1-long.jsonl").read_text().splitlines() if x]
    ids = max((r["ids"][:r["gen_start"]] for r in rows), key=len)[:a.ctx]
    e = engine(a.model, a.flags, max_seqs=1, max_model_len=a.ctx + a.steps * 8 + 64)
    e.generate([ids[::-1]], SamplingParams(temperature=0.0, max_tokens=16))        # warm-up (graphs, caches)
    e.add(ids, SamplingParams(temperature=0.0, max_tokens=a.steps * 6, ignore_eos=True))
    while e.has_work() and not any(d.token_ids for d in e.step()):
        pass
    for _ in range(4):
        e.step()
    walls, tokens = [], 0
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    for _ in range(a.steps):
        if not e.has_work():
            break
        t0 = time.perf_counter()
        out = e.step()
        if a.sync:                      # off: the next step launches while this one runs, as served (overlap)
            torch.cuda.synchronize()
        walls.append(time.perf_counter() - t0)
        tokens += sum(len(d.token_ids) for d in out)
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    stats = {k: v for k, v in e.stats().items() if k.startswith("spec")}
    e.close()
    return {"steps": len(walls), "wall_ms": [round(w * 1e3, 3) for w in walls], "tokens": tokens, "spec": stats,
            "flags": a.flags}


def summarize(a) -> dict:
    import sqlite3
    rep = pathlib.Path(a.rep)
    db = rep.with_suffix(".sqlite")
    if not db.exists():
        subprocess.run(["nsys", "export", "--type", "sqlite", "-o", str(db), str(rep)], check=True,
                       capture_output=True)
    c = sqlite3.connect(db)
    kernels = c.execute("SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s "
                        "ON s.id = k.shortName ORDER BY k.start").fetchall()
    names = dict(c.execute("SELECT id, value FROM StringIds"))
    tasks = sorted((s, e) for s, e, t, tid in c.execute(
        "SELECT start, end, text, textId FROM NVTX_EVENTS WHERE eventType = 59 AND end IS NOT NULL")
        if (t if t is not None else names.get(tid, "")) == "cpu_task")
    signals = [(s, e) for s, e, n in kernels if n == "k_signal"]
    waits = [(s, e) for s, e, n in kernels if n == "k_wait"]
    pairs, ti, wi = [], 0, 0
    for s0, s1 in signals:
        while ti < len(tasks) and tasks[ti][0] < s0:
            ti += 1
        while wi < len(waits) and waits[wi][0] < s1:
            wi += 1
        if ti >= len(tasks) or wi >= len(waits):
            break
        (c0, c1), (w0, w1) = tasks[ti], waits[wi]
        gpu = sum(min(e, w0) - max(s, s1) for s, e, n in kernels if e > s1 and s < w0 and n not in ("k_wait",))
        pairs.append({"wake_us": (c0 - s1) / 1e3, "task_us": (c1 - c0) / 1e3, "gpu_between_us": gpu / 1e3,
                      "span_us": (w0 - s1) / 1e3, "wait_us": (w1 - w0) / 1e3, "late_us": (c1 - w0) / 1e3})
        ti += 1
        wi += 1
    info = json.loads(pathlib.Path(a.run).read_text())

    def stat(k):
        xs = sorted(p[k] for p in pairs)
        return {"mean": round(statistics.mean(xs), 1), "p50": round(xs[len(xs) // 2], 1),
                "p90": round(xs[int(len(xs) * 0.9)], 1), "sum_ms_per_step": round(sum(xs) / 1e3 / info["steps"], 3)}
    gpu_ms = sum(e - s for s, e, _ in kernels) / 1e6 / info["steps"]
    return {"steps": info["steps"], "tokens_per_step": info["tokens"] / info["steps"],
            "wall_ms_p50": sorted(info["wall_ms"])[len(info["wall_ms"]) // 2],
            "gpu_kernel_ms_per_step": round(gpu_ms, 3),
            "signals_per_step": len(signals) / info["steps"], "tasks": len(tasks), "pairs": len(pairs),
            **({k: stat(k) for k in ("wake_us", "task_us", "gpu_between_us", "span_us", "wait_us", "late_us")}
               if pairs else {}), "flags": info["flags"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--flags", required=True)
    r.add_argument("--ctx", type=int, default=1024)
    r.add_argument("--steps", type=int, default=40)
    r.add_argument("--sync", action=argparse.BooleanOptionalAction, default=True,
                   help="a device sync after each step (its wall); --no-sync with --cuda-graph-trace=graph shows the "
                        "GPU's idle between graphed steps as served")
    r.add_argument("--out", required=True)
    s = sub.add_parser("summarize")
    s.add_argument("--rep", required=True)
    s.add_argument("--run", required=True)
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    out = run(a) if a.cmd == "run" else summarize(a)
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
