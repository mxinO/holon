"""Batch-1 decode and prefill at fixed contexts, for the expert-offload comparisons (docs/v1-offload.md acceptance 3,
D14; docs/v1-flash-next.md D11): every arm loads in a fresh process, runs one untimed warm-up request, then for each
context C its requests of a C-token prompt generating 256 tokens, greedy, ignore_eos. Prefill tok/s = C / prompt
time; decode tok/s = (tokens - 1) / (last token - first token). Prompts are spans of natural text, the same ids for
every engine and round, so an expert cache sees real routing and no prefix cache can reuse one prompt for another:
- `--corpus v1` (X): the judge prompt set's token ids, concatenated and cut at fixed, distinct offsets (wrapping),
  one request per context;
- `--corpus fn` (FN, D11): the text of v1-long, v1-qwen3_5 and v1, concatenated in that order and tokenized with
  Qwen3.8-Flash-Next's tokenizer (N tokens); context C (index c in `--contexts`) has the candidate starts
  (k * (N // 6) + c * (N // 18)) mod N, k = 0..5, a start inside the warm-up span (the corpus's last 2,048 tokens)
  moved past it, and `--spans` picks the candidates timed (default the first 3).
The orchestrator writes the warm-up and the timed requests' ids to `requests.json`; workers read only that file.

Engines: `llama` = llama-server (`--server`; its own `timings`: prompt_ms, and predicted_ms over the tokens after the
first); `freetoken` = FreeToken's offline `LLM` in its own venv (`--python`), timed in process as each token reaches
the frontend; `strata` = Strata's engine (`--strata`) in its `--serve` mode with raw token ids on stdin, timed in
this client as each verify window's tokens reach stdout (`--eos-ids 999999`, an id outside the vocabulary, so a
request ends only at its length; `--prompt-cache 0`; every request must report 0 reused tokens); `holon` = Holon's
`Engine` in process (`--holon-python`), stamped as each token reaches the host, prefix cache off. An arm's flags are
the engine's own command-line flags (FreeToken's parsed by its `parse_args`, so its CLI defaults apply; Holon's by
`holon serve`'s), and an arm's `env` its environment. Arms are interleaved: every round runs each arm once, in
reverse order on odd rounds, each after checking that the GPU is idle; peak VRAM is sampled with nvidia-smi.

    python bench/offload_ctx.py run --name <name> --gpu 0 --model <file.gguf> --arms arms.json --rounds 3 \\
        [--contexts 1024 4096 32768] [--ctx 33792] [--corpus v1|fn] [--spans 0 1 2] [--server <llama-server>] \\
        [--python <freetoken venv python>] [--strata <strata engine>] [--holon-python <python>]
    python bench/offload_ctx.py worker --engine llama --server <llama-server> --model <file.gguf> --flags '-cmoe' \\
        --requests <requests.json>

arms.json: [{"name": "cache47", "engine": "llama", "flags": "-cmoe --moe-cache-mib 7900"},
            {"name": "strata", "engine": "strata", "flags": "--pack <dir> --kv fp16", "env": {"STRATA_X": "0"}}, ...]
"""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
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
# D11's corpus (docs/v1-flash-next.md): these prompt sets' text, in this order, in Qwen3.8-Flash-Next's tokenizer
FN_SETS = ("v1-long.jsonl", "v1-qwen3_5.jsonl", "v1.jsonl")
FN_TOKENIZER = ("Qwen/Qwen3.8-Flash-Next", "de4b8e4d43b917e7706784d8bb445c9af86a3540")
FN_WARMUP = 2048                                        # the warm-up span: the corpus's last 2,048 tokens
STRATA_NO_EOS = 999999                                  # an id outside the vocabulary: Strata's ignore-EOS
# log lines kept in the record: what each engine resolved (placement, cache size, strategy, hit rates)
KEEP = re.compile(r"(?i)moe.cache|moe strateg|auto-selected|offloaded \d+/\d+ layers|fit|n_cpu_moe|model buffer size|"
                  r"kv.?cache size|slots|hit rate|load_mode|expert")


def prompt(corpus: list[int], context: int) -> list[int]:
    start = OFFSETS.get(context, 0)
    return [corpus[(start + i) % len(corpus)] for i in range(context)]


def span(corpus: list[int], start: int, n: int) -> list[int]:
    return [corpus[(start + i) % len(corpus)] for i in range(n)]


def fn_corpus() -> list[int]:
    """D11's corpus: the prompt sets' text concatenated and tokenized with the model's tokenizer (special tokens kept
    as the template wrote them)."""
    from tokenizers import Tokenizer
    repo, rev = FN_TOKENIZER
    hub = pathlib.Path(os.environ.get("HF_HUB_CACHE") or pathlib.Path(os.environ.get("HF_HOME") or
                                                                       "~/.cache/huggingface").expanduser() / "hub")
    tok = Tokenizer.from_file(str(hub / f"models--{repo.replace('/', '--')}" / "snapshots" / rev / "tokenizer.json"))
    text = "".join(json.loads(line)["text"] for name in FN_SETS
                   for line in (ROOT / "judge" / "prompts" / name).read_text().splitlines() if line)
    return tok.encode(text, add_special_tokens=False).ids


def fn_starts(n: int, contexts: list[int], count: int = 6) -> dict[int, list[int]]:
    """Each context's `count` candidate starts (D11: six, each a sixth of the corpus apart; past six, the midpoints
    between them), none inside the warm-up span [n - FN_WARMUP, n)."""
    out = {}
    for c, context in enumerate(contexts):
        starts = [(k % 6 * (n // 6) + k // 6 * (n // 12) + c * (n // 18)) % n for k in range(count)]
        out[context] = [0 if o >= n - FN_WARMUP else o for o in starts]
    return out


def plan(a: argparse.Namespace) -> dict:
    """The warm-up request and the timed ones (key "C" or "C/k" for candidate k) with their ids, for requests.json."""
    if a.corpus == "v1":
        corpus = workloads.corpus(CORPUS)
        off, p, n = WARMUP
        warm = {"ids": span(corpus, off, p), "output": n}
        reqs = [{"key": str(c), "context": c, "start": OFFSETS.get(c, 0), "ids": prompt(corpus, c)}
                for c in a.contexts]
    else:
        corpus = fn_corpus()
        warm = {"ids": corpus[-FN_WARMUP:], "output": 64}
        starts = fn_starts(len(corpus), a.contexts, max(6, max(a.spans) + 1))
        reqs = [{"key": f"{c}/{k}", "context": c, "start": starts[c][k], "ids": span(corpus, starts[c][k], c)}
                for c in a.contexts for k in a.spans]
    digest = hashlib.sha256(json.dumps(corpus).encode()).hexdigest()
    return {"corpus": a.corpus, "corpus_tokens": len(corpus), "corpus_sha256": digest, "warmup": warm,
            "requests": reqs}


def requests(a: argparse.Namespace, corpus: list[int] | None = None) -> tuple[dict, list[dict]]:
    """A worker's warm-up and timed requests: `--requests`' file, else (callers from spikes/) the v1 offsets."""
    if getattr(a, "requests", None):
        p = json.loads(pathlib.Path(a.requests).read_text())
        return p["warmup"], p["requests"]
    off, p, n = WARMUP
    return ({"ids": span(corpus, off, p), "output": n},
            [{"key": str(c), "ids": prompt(corpus, c)} for c in a.contexts])


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

        warm, reqs = requests(a, corpus)
        request(warm["ids"], warm["output"])
        results = {r["key"]: {k: round(v, 3) for k, v in request(r["ids"], OUTPUT).items()} for r in reqs}
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

    warm, reqs = requests(a, corpus)
    request(warm["ids"], warm["output"])
    return {"results": {r["key"]: {k: round(v, 3) for k, v in request(r["ids"], OUTPUT).items()} for r in reqs}}


def strata_worker(a: argparse.Namespace, corpus: list[int] | None) -> dict:
    """Strata's engine in `--serve` mode: `GEN <n> <ids>` on stdin; `T <id>` per token (flushed per verify window),
    then `DONE <generated> <prompt> <prompt_ms> <decode_ms> <reason> <accepted> <offered> <reused> <hits> <lookups>
    <RAM blobs> <file blobs> <file MB> <prompt read> <offloaded> ...` on stdout (St generate.cpp:10635-10732,
    12936-12942). The PLE shard is the model's second."""
    ple = a.model.replace("-00001-of-00002.gguf", "-00002-of-00002.gguf")
    cmd = [*shlex.split(a.wrap), a.server, "--serve", "--native", a.model, "--ple-gguf", ple, "--max-context",
           str(a.ctx), "--eos-ids", str(STRATA_NO_EOS), "--prompt-cache", "0", *shlex.split(a.flags)]
    log = open(a.log, "w") if a.log else subprocess.DEVNULL
    engine = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1)
    info = {}
    try:
        for line in engine.stdout:
            if line.startswith("INFO "):
                info = dict(x.split("=", 1) for x in line.split()[1:] if "=" in x)
            if line.startswith("READY"):
                break
        else:
            raise RuntimeError(f"strata exited with {engine.wait()} before READY")

        def request(ids: list[int], n: int) -> dict:
            t0 = time.perf_counter()
            engine.stdin.write(f"GEN {n} {','.join(map(str, ids))}\n")
            engine.stdin.flush()
            stamps, resume = [], None
            for line in engine.stdout:
                now = time.perf_counter()
                if line.startswith("T "):
                    stamps.append(now)
                elif line.startswith("RESUME "):
                    resume = int(line.split()[1])
                elif line.startswith("ERR"):
                    raise RuntimeError(line.strip())
                elif line.startswith("DONE "):
                    f = line.split()
                    break
            else:
                raise RuntimeError(f"strata exited with {engine.wait()} during a request")
            gen, acc, off = int(f[1]), int(f[6]), int(f[7])
            offloaded = int(f[15]) if len(f) > 15 else 0
            assert len(stamps) == gen == n and f[5] == "length", (len(stamps), f[:6])
            return {"prefill_tps": len(ids) / (stamps[0] - t0), "decode_tps": (n - 1) / (stamps[-1] - stamps[0]),
                    "reused": int(f[8]) + (resume or 0), "hit_rate": int(f[9]) / max(int(f[10]), 1),
                    # Strata's own rate leaves out the misses the GPU read over PCIe (`offloaded`, St #588); over
                    # every routed expert, as Holon counts: hits / (lookups + offloaded)
                    "hit_rate_all": int(f[9]) / max(int(f[10]) + offloaded, 1),
                    "pcie_share": offloaded / max(int(f[10]) + offloaded, 1),
                    "drafts_accepted": acc, "drafts_offered": off, "tokens_per_pass": gen / max(gen - acc, 1),
                    "engine_prompt_ms": float(f[3]), "engine_decode_ms": float(f[4])}

        warm, reqs = requests(a, corpus)
        request(warm["ids"], warm["output"])
        results = {r["key"]: {k: round(v, 4) if isinstance(v, float) else v
                              for k, v in request(r["ids"], OUTPUT).items()} for r in reqs}
    finally:
        try:
            engine.stdin.write("QUIT\n")
            engine.stdin.flush()
            engine.wait(120)
        except (OSError, subprocess.TimeoutExpired):
            engine.kill()
            engine.wait()
        if a.log:
            log.close()
    return {"results": results, "info": {k: info[k] for k in ("context", "expert_slots", "expert_cache_mib", "spec",
                                                              "mtp_max", "pcie_frac", "spec_min_p", "engine")
                                         if k in info}}


def holon_worker(a: argparse.Namespace, corpus: list[int] | None) -> dict:
    """Holon's `Engine` in process (as spikes/x4_gap/ctx.py), a stamp as each token reaches the host; prefix cache
    off, so no request reuses another's tokens."""
    import dataclasses

    import torch
    from holon.api import SamplingParams
    from holon.core.cli import engine_config, engine_flags
    from holon.core.engine import Engine

    flags = argparse.ArgumentParser()
    engine_flags(flags)
    cfg = engine_config(a.model, flags.parse_args(shlex.split(a.flags)))
    engine = Engine(dataclasses.replace(cfg, max_seqs=1, max_model_len=a.ctx, prefix_cache=False))
    store, seen = engine.ctx.shared_objects.get("std.moe.offload"), {"hits": 0, "routed": 0}
    if store is not None:                   # an offload deployment: the decode rows' routed experts found resident
        prev = {"map": store.residency.host.copy()}

        def tap(step, record) -> None:
            counts = record["counts"]                        # decode and verify rows' routes, [groups, experts]
            seen["routed"] += int(counts.sum())
            seen["hits"] += int(counts[(prev["map"] >= 0).all(-1)].sum())
            prev["map"] = store.residency.host.copy()
        store.taps.append(tap)

    def passes() -> tuple[int, int, dict[int, int]]:
        """Verify rows so far, the tokens they gave (each row: its accepted drafts and one token of its own), and the
        rows by drafts accepted."""
        rows = [(int(k.rsplit("_", 1)[1]), n) for k, n in engine.stats().items() if k.startswith("spec_rows_")]
        by = collections.Counter()
        for acc, n in rows:
            by[acc] += n
        return sum(n for _, n in rows), sum((acc + 1) * n for acc, n in rows), by

    def request(ids: list[int], n: int) -> dict:
        stamps, done = [], False
        torch.cuda.synchronize()
        seen.update(hits=0, routed=0)
        rows0, tokens0, by0 = passes()
        t0 = time.perf_counter()
        rid = engine.add(ids, SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True))
        while not done:
            for d in engine.step():
                if d.rid == rid:
                    stamps += [time.perf_counter()] * len(d.token_ids)
                    done = d.finish_reason is not None
        assert len(stamps) == n, (len(stamps), n)
        rows, tokens, by = passes()
        out = {"prefill_tps": len(ids) / (stamps[0] - t0), "decode_tps": (n - 1) / (stamps[-1] - stamps[0])}
        if rows > rows0:
            out["tokens_per_pass"] = (tokens - tokens0) / (rows - rows0)
            for d in range(1, 1 + max(by, default=0)):   # draft d accepted, given d - 1 were (Strata's spec depth)
                tested = sum(n - by0[a] for a, n in by.items() if a >= d - 1)
                out[f"accept_d{d}"] = sum(n - by0[a] for a, n in by.items() if a >= d) / max(tested, 1)
        if seen["routed"]:
            out["hit_rate_all"] = seen["hits"] / seen["routed"]
        return out

    warm, reqs = requests(a, corpus)
    request(warm["ids"], warm["output"])
    results = {r["key"]: {k: round(v, 4) for k, v in request(r["ids"], OUTPUT).items()} for r in reqs}
    engine.close()
    return {"results": results}


WORKERS = {"llama": llama_worker, "freetoken": freetoken_worker, "strata": strata_worker, "holon": holon_worker}


def worker(a: argparse.Namespace) -> None:
    corpus = None if a.requests else workloads.corpus(CORPUS)
    print("RESULT " + json.dumps(WORKERS[a.engine](a, corpus)), flush=True)


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
    py = {"freetoken": a.python, "holon": a.holon_python}.get(arm["engine"], sys.executable)
    cmd = [py, str(pathlib.Path(__file__)), "worker", "--engine", arm["engine"], "--model", a.model, "--ctx",
           str(a.ctx), f"--flags={arm['flags']}", "--requests", str(log.parent / "requests.json")]   # one '-'-led value
    if arm["engine"] in ("llama", "strata"):
        cmd += ["--server", a.server if arm["engine"] == "llama" else a.strata,
                "--log", str(log.with_suffix(".server.log")), "--wrap", arm.get("wrap", "")]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID", **arm.get("env", {}))
    stop, peak = threading.Event(), [0]
    sampler = threading.Thread(target=peak_vram, args=(a.gpu, stop, peak), daemon=True)
    sampler.start()
    t = time.monotonic()
    with open(log, "w") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    stop.set()
    text = log.read_text(errors="replace")
    if arm["engine"] in ("llama", "strata"):
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
    reqs = outdir / "requests.json"
    if a.requests:
        reqs.write_text(pathlib.Path(a.requests).read_text())
    elif not (a.resume and reqs.exists()):
        reqs.write_text(json.dumps(plan(a)))
    p = json.loads(reqs.read_text())
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
        "contexts": a.contexts, "ctx": a.ctx, "output_tokens": OUTPUT,
        "corpus": {k: p[k] for k in ("corpus", "corpus_tokens", "corpus_sha256")},
        "warmup": {"tokens": len(p["warmup"]["ids"]), "output": p["warmup"]["output"]},
        "requests": [{k: r[k] for k in ("key", "context", "start") if k in r} for r in p["requests"]],
        "arms": [arm | {"flags": run.public_flags(arm["flags"], a.model)} for arm in arms],
        "summary": summary(rows), "rows": rows}
    out = pathlib.Path(a.out or outdir / "record.json")
    out.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record["summary"], indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("worker")
    w.add_argument("--engine", choices=tuple(WORKERS), required=True)
    w.add_argument("--server", help="llama-server, or Strata's engine")
    w.add_argument("--requests", help="the orchestrator's requests.json (default: --contexts at the v1 offsets)")
    w.add_argument("--log")
    w.add_argument("--wrap", default="", help="a command prefix for the server (e.g. nsys profile ...)")
    o = sub.add_parser("run")
    o.add_argument("--name", required=True)
    o.add_argument("--gpu", type=int, default=0)
    o.add_argument("--arms", required=True)
    o.add_argument("--rounds", type=int, default=3)
    o.add_argument("--server", default=str(ROOT / "bench" / ".envs" / "llama.cpp-offload" / "bin" / "llama-server"))
    o.add_argument("--python", default=str(ROOT / "bench" / ".envs" / "freetoken-offload" / "bin" / "python"))
    o.add_argument("--strata", default=str(ROOT / "bench" / ".envs" / "strata" / "strata"))
    o.add_argument("--holon-python", default=sys.executable)
    o.add_argument("--requests", help="a plan written by `plan` (fn's corpus needs tokenizers, which a bench venv "
                   "may lack)")
    pl = sub.add_parser("plan", help="write requests.json")
    pl.add_argument("--out", required=True)
    for p in (o, pl):
        p.add_argument("--corpus", choices=("v1", "fn"), default="v1")
        p.add_argument("--spans", type=int, nargs="+", default=[0, 1, 2],
                       help="fn: the candidate starts timed (0-11; past 5 the midpoints between the six D11 starts)")
        p.add_argument("--contexts", type=int, nargs="+", default=[1024, 4096, 32768])
    o.add_argument("--out")
    o.add_argument("--resume", action="store_true", help="keep the rows of bench/.out/<name>/raw.jsonl")
    for p in (w, o):
        p.add_argument("--model", required=True)
        if p is w:
            p.add_argument("--contexts", type=int, nargs="+", default=[1024, 4096, 32768])
        p.add_argument("--ctx", type=int, default=33792, help="context each engine allocates (KV) for every run")
        if p is w:
            p.add_argument("--flags", default="")
    a = ap.parse_args()
    if a.cmd == "plan":
        pathlib.Path(a.out).write_text(json.dumps(plan(a)))
    else:
        worker(a) if a.cmd == "worker" else orchestrate(a)


if __name__ == "__main__":
    main()
