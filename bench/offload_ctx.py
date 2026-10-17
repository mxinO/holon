"""Batch-1 decode and prefill at fixed contexts, for the expert-offload comparisons (docs/v1-offload.md acceptance 3,
D14): every arm loads in a fresh process, runs one untimed warm-up request, then for each context C one request of
a C-token prompt generating 256 tokens, greedy, ignore_eos. Prefill tok/s = C / prompt time; decode tok/s =
(tokens - 1) / (last token - first token). Prompts are spans of natural text (the judge prompt set's token ids,
concatenated and cut at fixed, distinct offsets, wrapping), the same ids for every engine and round, so an expert
cache sees real routing and no prefix cache can reuse one prompt for another.

Engines: `llama` = llama-server (`--server`; its own `timings`: prompt_ms, and predicted_ms over the tokens after the
first); `freetoken` = FreeToken's offline `LLM` in its own venv (`--python`), timed in process as each token reaches
the frontend. An arm's flags are the engine's own command-line flags (FreeToken's parsed by its `parse_args`, so its
CLI defaults apply). Arms are interleaved: every round runs each arm once, in reverse order on odd rounds, each after
checking that the GPU is idle; peak VRAM is sampled with nvidia-smi.

    python bench/offload_ctx.py run --name <name> --gpu 0 --model <file.gguf> --arms arms.json --rounds 3 \\
        [--contexts 1024 4096 32768] [--ctx 33792] [--server <llama-server>] [--python <freetoken venv python>]
    python bench/offload_ctx.py worker --engine llama --server <llama-server> --model <file.gguf> --flags '-cmoe'

arms.json: [{"name": "cache47", "engine": "llama", "flags": "-cmoe --moe-cache-mib 7900"}, ...]
"""
from __future__ import annotations

import argparse
import datetime
import http.client as client
import json
import os
import pathlib
import re
import shlex
import socket
import statistics
import subprocess
import sys
import threading
import time

import run  # bench/run.py: record helpers (stdlib only)
import workloads

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS = ROOT / "judge" / "prompts" / "v1.jsonl"
WARMUP = (0, 512, 64)                                   # offset, prompt tokens, output tokens
OFFSETS = {1024: 4096, 4096: 8192, 32768: 20000}         # distinct starts: no prompt is a prefix of another
OUTPUT = 256
# log lines kept in the record: what each engine resolved (placement, cache size, strategy, hit rates)
KEEP = re.compile(r"(?i)moe.cache|moe strateg|auto-selected|offloaded \d+/\d+ layers|fit|n_cpu_moe|model buffer size|"
                  r"kv.?cache size|slots|hit rate|load_mode|expert")


def prompt(corpus: list[int], context: int) -> list[int]:
    start = OFFSETS.get(context, 0)
    return [corpus[(start + i) % len(corpus)] for i in range(context)]


# ---- workers: one arm, one process; print one JSON line ----

def llama_worker(a: argparse.Namespace, corpus: list[int]) -> dict:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    log = open(a.log, "w") if a.log else subprocess.DEVNULL
    cmd = [*shlex.split(a.wrap), a.server, "-m", a.model, "--host", "127.0.0.1", "--port", str(port), "-np", "1",
           "-c", str(a.ctx), *shlex.split(a.flags)]
    server = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)

    def post(path: str, body: dict | None = None) -> tuple[int, dict]:
        conn = client.HTTPConnection("127.0.0.1", port, timeout=3600)
        conn.request("POST" if body is not None else "GET", path, json.dumps(body) if body is not None else None,
                     {"Content-Type": "application/json"})
        r = conn.getresponse()
        return r.status, json.loads(r.read() or b"{}")

    try:
        deadline = time.monotonic() + 1200
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"llama-server exited with {server.returncode}")
            try:
                if post("/health")[0] == 200:
                    break
            except OSError:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError("llama-server not ready")
            time.sleep(1)

        def request(ids: list[int], n: int) -> dict:
            status, r = post("/completion", {"prompt": ids, "n_predict": n, "temperature": 0.0, "top_k": 1,
                                             "ignore_eos": True, "cache_prompt": False, "stream": False})
            if status != 200:
                raise RuntimeError(f"/completion {status}: {r}")
            t = r["timings"]
            assert t["prompt_n"] == len(ids) and t["predicted_n"] == n, t
            return {"prefill_tps": len(ids) / t["prompt_ms"] * 1e3, "decode_tps": (n - 1) / t["predicted_ms"] * 1e3}

        off, p, n = WARMUP
        request([corpus[(off + i) % len(corpus)] for i in range(p)], n)
        results = {str(c): {k: round(v, 3) for k, v in request(prompt(corpus, c), OUTPUT).items()}
                   for c in a.contexts}
    finally:
        server.send_signal(2)                           # SIGINT: a clean exit logs the MoE cache's hit rates
        try:
            server.wait(900 if a.wrap else 120)            # nsys writes its report on exit
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        if a.log:
            log.close()
    return {"results": results}


def freetoken_worker(a: argparse.Namespace, corpus: list[int]) -> dict:
    import dataclasses

    import torch
    from freetoken.core import SamplingParams
    from freetoken.llm import LLM
    from freetoken.message import DetokenizeMsg
    from freetoken.scheduler import SchedulerConfig
    from freetoken.server.args import parse_args

    cfg, _ = parse_args(["--model-path", a.model, "--max-seq-len-override", str(a.ctx), *shlex.split(a.flags)])
    own = ("model_path", "tp_info", "dtype", "offline_mode", "_unique_suffix")
    kwargs = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(SchedulerConfig) if f.name not in own}

    class Timed(LLM):
        stamps: list[float] = []

        def offline_send_result(self, reply):
            now = time.perf_counter()
            self.stamps += [now for m in reply if isinstance(m, DetokenizeMsg)]
            super().offline_send_result(reply)

    llm = Timed(a.model, dtype=cfg.dtype, **kwargs)

    def request(ids: list[int], n: int) -> dict:
        llm.stamps = []
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        llm.generate([ids], SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True))
        s = llm.stamps
        assert len(s) == n, (len(s), n)
        return {"prefill_tps": len(ids) / (s[0] - t0), "decode_tps": (n - 1) / (s[-1] - s[0])}

    off, p, n = WARMUP
    request([corpus[(off + i) % len(corpus)] for i in range(p)], n)
    return {"results": {str(c): {k: round(v, 3) for k, v in request(prompt(corpus, c), OUTPUT).items()}
                        for c in a.contexts}}


def worker(a: argparse.Namespace) -> None:
    corpus = workloads.corpus(CORPUS)
    out = (llama_worker if a.engine == "llama" else freetoken_worker)(a, corpus)
    print("RESULT " + json.dumps(out), flush=True)


# ---- orchestrator ----

def gpu_busy(gpu: int) -> str | None:
    """Memory in use or a compute process (utilization is a trailing average: it stays up after a process exits)."""
    smi = ["nvidia-smi", "-i", str(gpu), "--format=csv,noheader,nounits"]
    used = int(subprocess.run([*smi, "--query-gpu=memory.used"], capture_output=True, text=True, check=True).stdout)
    apps = subprocess.run([*smi, "--query-compute-apps=pid"], capture_output=True, text=True, check=True).stdout.split()
    return None if used < 1024 and not apps else f"{used} MiB used, {len(apps)} compute processes"


def peak_vram(gpu: int, stop: threading.Event, peak: list[int]) -> None:
    while not stop.wait(0.5):
        q = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True).stdout.strip()
        if q.isdigit():
            peak[0] = max(peak[0], int(q))


def run_arm(arm: dict, a: argparse.Namespace, log: pathlib.Path) -> dict:
    for _ in range(60):                                  # wait up to 10 min for an idle GPU
        if not (busy := gpu_busy(a.gpu)):
            break
        print(f"  GPU busy ({busy}), waiting", flush=True)
        time.sleep(10)
    else:
        raise SystemExit(f"GPU {a.gpu} stayed busy: {busy}")
    py = sys.executable if arm["engine"] == "llama" else a.python
    cmd = [py, str(pathlib.Path(__file__)), "worker", "--engine", arm["engine"], "--model", a.model, "--ctx",
           str(a.ctx), "--flags", arm["flags"], "--contexts", *map(str, arm.get("contexts", a.contexts))]
    if arm["engine"] == "llama":
        cmd += ["--server", a.server, "--log", str(log.with_suffix(".server.log")), "--wrap", arm.get("wrap", "")]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID")
    stop, peak = threading.Event(), [0]
    sampler = threading.Thread(target=peak_vram, args=(a.gpu, stop, peak), daemon=True)
    sampler.start()
    t = time.monotonic()
    with open(log, "w") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    stop.set()
    text = log.read_text(errors="replace")
    if arm["engine"] == "llama":
        text = log.with_suffix(".server.log").read_text(errors="replace") + text
    line = next((x for x in text.splitlines() if x.startswith("RESULT ")), None)
    if p.returncode or line is None:
        return {"error": run.reason(log)}
    out = json.loads(line[7:])
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)                         # terminal colours
    out["log"] = sorted({re.sub(r"\s+", " ", x.strip())[:240] for x in text.splitlines() if KEEP.search(x)
                         and not x.startswith("RESULT ")})
    return out | {"peak_vram_mib": peak[0], "wall_s": round(time.monotonic() - t, 1)}


def summary(rows: list[dict]) -> dict:
    out = {}
    for r in rows:
        for c, m in r.get("results", {}).items():
            for k, v in m.items():
                out.setdefault(r["arm"], {}).setdefault(c, {}).setdefault(k, []).append(v)
    return {arm: {c: {k: {"median": round(statistics.median(v), 2), "min": round(min(v), 2), "max": round(max(v), 2),
                          "n": len(v)} for k, v in ms.items()} for c, ms in cs.items()} for arm, cs in out.items()}


def orchestrate(a: argparse.Namespace) -> None:
    arms = json.loads(pathlib.Path(a.arms).read_text())
    outdir = ROOT / "bench" / ".out" / a.name
    outdir.mkdir(parents=True, exist_ok=True)
    raw = outdir / "raw.jsonl"
    rows = [json.loads(x) for x in raw.read_text().splitlines()] if raw.exists() and a.resume else []
    done = {(r["round"], r["arm"]) for r in rows if "error" not in r}
    for rnd in range(a.rounds):
        for arm in (arms if rnd % 2 == 0 else arms[::-1]):
            if (rnd, arm["name"]) in done:
                continue
            print(f"round {rnd} {arm['name']}", flush=True)
            r = {"round": rnd, "arm": arm["name"]} | run_arm(arm, a, outdir / f"r{rnd}_{arm['name']}.log")
            print("  " + json.dumps({k: r[k] for k in ("results", "error", "peak_vram_mib") if k in r}), flush=True)
            rows.append(r)
            with open(raw, "a") as f:
                f.write(json.dumps(r) + "\n")
    gpu = run.gpu_class(a.gpu)
    record = {
        "name": a.name, "date": datetime.date.today().isoformat(), "gpu": gpu, "model": run.model_id(a.model),
        "command": "python bench/offload_ctx.py " + run.public_flags(shlex.join(sys.argv[1:]), a.model),
        "commit": run.git("rev-parse", "HEAD"), "method": __doc__.split("\n\n")[0].replace("\n", " "),
        "contexts": a.contexts, "ctx": a.ctx, "output_tokens": OUTPUT, "warmup": WARMUP, "offsets": OFFSETS,
        "arms": [arm | {"flags": run.public_flags(arm["flags"], a.model)} for arm in arms],
        "summary": summary(rows), "rows": rows}
    out = pathlib.Path(a.out or outdir / "record.json")
    out.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record["summary"], indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("worker")
    w.add_argument("--engine", choices=("llama", "freetoken"), required=True)
    w.add_argument("--server")
    w.add_argument("--log")
    w.add_argument("--wrap", default="", help="a command prefix for the server (e.g. nsys profile ...)")
    o = sub.add_parser("run")
    o.add_argument("--name", required=True)
    o.add_argument("--gpu", type=int, default=0)
    o.add_argument("--arms", required=True)
    o.add_argument("--rounds", type=int, default=3)
    o.add_argument("--server", default=str(ROOT / "bench" / ".envs" / "llama.cpp-offload" / "bin" / "llama-server"))
    o.add_argument("--python", default=str(ROOT / "bench" / ".envs" / "freetoken-offload" / "bin" / "python"))
    o.add_argument("--out")
    o.add_argument("--resume", action="store_true", help="keep the rows of bench/.out/<name>/raw.jsonl")
    for p in (w, o):
        p.add_argument("--model", required=True)
        p.add_argument("--contexts", type=int, nargs="+", default=[1024, 4096, 32768])
        p.add_argument("--ctx", type=int, default=33792, help="context each engine allocates (KV) for every run")
        if p is w:
            p.add_argument("--flags", default="")
    a = ap.parse_args()
    worker(a) if a.cmd == "worker" else orchestrate(a)


if __name__ == "__main__":
    main()
