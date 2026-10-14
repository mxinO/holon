"""Compare engines on the same GPU and model; writes an evidence record.

Interleaved ABBA: every round runs each backend once (in order, reversed on odd rounds), each in a fresh process
with its own venv (`worker.py`), so slow drift of the machine (clocks, temperature, neighbours) hits every backend
alike and none always runs first. Each process loads the model once and replays every trace after a warmup
(`driver.warmup_sessions`).

With `--lock`, Holon runs the lock's engine settings and the baselines are matched to it (release design §5.2): the
lock's `max_seqs` and step budget for every backend, its KV format for the baselines, and Holon's measured KV tokens
(the first in-process `holon` run's, or `--kv-tokens`) as the baselines' KV cache size.

Arms: a backend named `<backend>@<label>` is another arm of the same engine, with `--arm <label>='<flags>'` added to
Holon's engine flags (over the lock's); a label without flags is an A/A arm. Every backend's TTFT is also paired per
(session, turn) with the first backend's of the same round (`paired_ttft`: the median of the per-request ratios).

Usage (from the repository root, inside Holon's .venv):
  python bench/run.py --name decode --gpu 5 --model <checkpoint dir> --backends holon minisgl \\
      --traces bench/traces/decode_b1.jsonl bench/traces/decode_b8.jsonl bench/traces/decode_b64.jsonl
"""
from __future__ import annotations

import argparse
import datetime
import getpass
import hashlib
import json
import os
import pathlib
import re
import shlex
import socket
import statistics
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
BENCH = ROOT / "bench"
ENVS = BENCH / ".envs"                                   # baseline venvs and sources (setup_baselines.sh)
SOURCES = {"minisgl": "mini-sglang", "nanovllm": "nano-vllm", "freetoken_http": "FreeToken"}
SERVERS = ("llamaserver", "freetoken_http", "sglang_http")   # a server binary or venv of its own, and our client
SET_BY_WORKER = ("max_seqs", "max_step_tokens", "max_model_len")   # from --max-batch, --max-step-tokens, the traces
METHOD = (
    "In-process engines driven by bench/driver.py, one scheduling step per call; client-side timestamps taken when "
    "a step returns a token. TTFT = first token - release (arrival, or previous turn's last token + gap); TPOT = "
    "(last - first) / (tokens - 1) per request; throughput = output tokens / (last token - first release). Decode "
    "steps = steps at peak batch that return no first token; decode tok/s = their tokens / their wall time. Max "
    "gap = a request's largest time between two deliveries after its first token. Paired TTFT = the median over "
    "(round, session, turn) of TTFT / the first backend's TTFT in the same round. CPU "
    "ms/step = time.process_time() delta around step() on those steps, measured with --blocking-sync (CUDA "
    "primary context in blocking-sync mode and torch events created blocking, so waiting for the GPU sleeps and "
    "is not counted). Greedy, ignore_eos, fixed output lengths; warmup on the same shapes with other prompt "
    "tokens, excluded. HTTP backends (llamaserver, holon_http; bench/backends/http.py): streamed requests, a "
    "token timestamped when its chunk arrives; their 'steps' are client polls, so decode tok/s and CPU ms/step do "
    "not apply to them.")
BACKENDS = ("holon", "holon_http", "holon_http_text", "holon_http_chat", "llamaserver", "freetoken_http",
            "sglang_http", "minisgl", "nanovllm")


def backend_name(name: str) -> str:
    """A backend, or an arm of a Holon backend (`holon@mixed`)."""
    base, at, label = name.partition("@")
    if base not in BACKENDS or at and not (base.startswith("holon") and re.fullmatch(r"[\w.-]+", label)):
        raise argparse.ArgumentTypeError(f"{name!r}: one of {', '.join(BACKENDS)}, or <holon backend>@<label>")
    return name


def git(*args: str, cwd: pathlib.Path = ROOT) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


def gpu_class(index: int) -> str:
    """Holon's fingerprint GPU class (product name without vendor words), e.g. "rtx-a5000"."""
    name = subprocess.run(["nvidia-smi", "-i", str(index), "--query-gpu=name", "--format=csv,noheader"],
                          capture_output=True, text=True, check=True).stdout.strip()
    return "-".join(w for w in re.split(r"[\s_]+", name.lower()) if w not in ("nvidia", "geforce"))


def model_id(path: str) -> str:
    """"Org/Name@snapshot" for a Hugging Face cache snapshot (plus "/<file>" for a file in it, e.g. a .gguf), else
    the directory or file name (no absolute paths)."""
    m = re.search(r"models--([^/]+)--([^/]+)/snapshots/([0-9a-f]+)(/[^/]+)?", path)
    return f"{m[1]}/{m[2]}@{m[3][:12]}{m[4] or ''}" if m else pathlib.Path(path).name


def public(arg: str, model: str) -> str:
    """A command-line argument fit for a published record: the model as its id, other paths repo-relative."""
    if arg == model:
        return model_id(model)
    if "=" in arg:                                       # --key=value and plugin options key=value
        key, value = arg.split("=", 1)
        return f"{key}={public(value, model)}"
    path = pathlib.Path(arg)
    if not path.is_absolute():                            # relative: published as is, unless it leaves the repo
        full = (pathlib.Path.cwd() / path).resolve()
        return str(full.relative_to(ROOT)) if full.exists() and full.is_relative_to(ROOT) else arg
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name


def public_flags(flags: str, model: str) -> str:
    """A flag string fit for a published record (each argument through `public`)."""
    return shlex.join(public(a, model) for a in shlex.split(flags))


def lock_flags(engine: dict, skip: tuple[str, ...] = (), extra: str = "", locked_kv: bool = True) -> list[str]:
    """`holon serve` flags that rebuild a lock's [engine] table (kernel pins are not flags: the default selection
    runs; release design D2 makes them soft preferences), then the `extra` flags given by hand: those override, a
    dict flag's items merged per key into the lock's (as D2's `holon serve` will). Under a lock (`locked_kv`) the KV
    format stays the lock's: the baselines are matched to it."""
    engine, rest, key = {k: dict(v) if isinstance(v, dict) else v for k, v in engine.items()}, [], None
    for arg in shlex.split(extra):
        if arg.startswith("--"):
            name, _, item = arg[2:].partition("=")              # --flag=item is --flag item
            key = name.replace("-", "_") if name in ("plugin-options", "slots", "kernels") else None
            if key is None:
                rest.append(arg)
            if key is None or not item:
                continue
        elif key is None:
            rest.append(arg)
            continue
        else:
            item = arg
        k, eq, v = item.partition("=")
        if not eq:
            raise SystemExit(f"--{key.replace('_', '-')}: expected KEY=VALUE items, got {item!r}")
        if locked_kv and (key, k) == ("plugin_options", "kv.format"):
            raise SystemExit("under --lock the KV format is the lock's (the baselines are matched to it)")
        engine.setdefault(key, {})[k] = v
    flags = []
    for key, value in sorted(engine.items()):
        if key in skip:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            flags.append(flag if value else "--no-" + flag[2:])
        elif isinstance(value, dict):
            flags += [flag, *(f"{k}={v}" for k, v in value.items())] if value else []
        else:
            flags += [flag, str(value)]
    return flags + rest


def read_lock(path: str) -> dict:
    """What a record says about the lock a run matched: where it is, its plan and class, and its [engine]."""
    from holon.core import plans

    plan = plans.read(path).plan
    return {"path": public(path, ""), "plan_id": plan.plan_id, "fingerprint_class": plan.fingerprint_class,
            "engine": plan.engine}


def reason(log: pathlib.Path) -> str:
    """A failed worker's reason for a record, a hint (the log stays local): its first exception line (a server's
    root cause comes before our client's), else its first error line, else its last line. Scrubbed for publishing:
    paths, host:port pairs and driver versions are replaced."""
    lines = [line.strip() for line in log.read_text(errors="replace").splitlines() if line.strip()]
    pick = next((line for line in lines if re.search(r"\b[\w.]*(Error|Exception): ", line)), None) or next(
        (line for line in lines if re.search(r"(?i)\berror\b", line)), None) or (lines or ["no output"])[-1]
    for name, mark in ((socket.getfqdn(), "<host>"), (socket.gethostname(), "<host>"), (getpass.getuser(), "<user>")):
        pick = pick.replace(name, mark) if len(name) > 2 else pick
    pick = re.sub(r"[^\s'\",;()\[\]{}]*[/~][^\s'\",;()\[\]{}]*", "<path>", pick)     # any token with / or ~
    pick = re.sub(r"\b\d+(?:\.\d+){3}:\d+|(?<![\w.-])(?!localhost:)(?=[\w.-]*[A-Za-z])[\w.-]+\]?:\d{2,5}\b",
                  lambda m: m[0] if m[0].startswith("127.0.0.1:") else "<host>:<port>", pick)
    return re.sub(r"(?i)(driver\D*)\d[\d.]*", r"\1<version>", pick)[:300]


def pct(xs: list[float], q: float) -> float | None:
    """Linear-interpolated percentile (q in [0, 100])."""
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * q / 100
    lo = int(k)
    return xs[lo] + (xs[min(lo + 1, len(xs) - 1)] - xs[lo]) * (k - lo)


def _dist_ms(xs: list[float], top: bool = False) -> dict:
    out = {f"p{q}": None if (v := pct(xs, q)) is None else round(v * 1e3, 2) for q in (50, 90, 99)}
    return out | ({"max": round(max(xs) * 1e3, 2) if xs else None} if top else {})


def summarize(cases: list[dict], columns: list[str], cpu_time: bool) -> dict:
    """Request percentiles pooled over `cases` (one per round); throughputs averaged over them. CPU time per step
    only when measured (`cpu_time`): with spinning CUDA waits it is just wall time."""
    ttft, tpot, gap, tput, wall, cpu = [], [], [], [], [], []
    batch = short = dec_steps = dec_tok = 0
    for case in cases:
        rows = [dict(zip(columns, r, strict=True)) for r in case["requests"]]
        ttft += [r["first"] - r["release"] for r in rows]
        tpot += [(r["last"] - r["first"]) / (r["generated"] - 1) for r in rows if r["generated"] > 1]
        gap += [r["max_gap"] for r in rows if r["generated"] > 1 and "max_gap" in r]   # older records lack it
        short += sum(r["generated"] != r["output"] for r in rows)
        span = max(r["last"] for r in rows) - min(r["release"] for r in rows)
        tput.append(sum(r["generated"] for r in rows) / span)
        peak = max(s[2] for s in case["steps"])
        batch = max(batch, peak)
        steady = [s for s in case["steps"] if s[2] == peak and s[3] == 0]
        dec_steps += len(steady)
        dec_tok += sum(s[2] for s in steady)
        wall += [s[0] for s in steady]
        cpu += [s[1] for s in steady] if cpu_time else []
    return {"requests": sum(len(c["requests"]) for c in cases), "ttft_ms": _dist_ms(ttft, top=True),
            "tpot_ms": _dist_ms(tpot), "max_gap_ms": _dist_ms(gap, top=True),
            "throughput_tok_s": round(sum(tput) / len(tput), 1), "throughput_per_round": [round(t, 1) for t in tput],
            "decode": {"batch": batch, "steps": dec_steps, "tok_s": round(dec_tok / sum(wall), 1) if wall else None,
                       "wall_ms_p50": None if not wall else round(pct(wall, 50) * 1e3, 3),
                       "cpu_ms_p50": None if not cpu else round(pct(cpu, 50) * 1e3, 3),
                       "cpu_ms_mean": None if not cpu else round(sum(cpu) / len(cpu) * 1e3, 3)},
            "wrong_length": short,
            "prompt_tokens": sum(r[columns.index("prompt")] for c in cases for r in c["requests"]),
            "stats": {k: sum(c["stats"].get(k, 0) for c in cases)
                      for k in sorted({k for c in cases for k in c.get("stats", {})})}}


def paired_ttft(runs: list[dict], ref: list[dict]) -> dict:
    """TTFT / the reference's per (session, turn) of the same round: the median ratio over every pair, and per round
    (by round)."""

    def ttfts(run: dict) -> dict[tuple[int, int], float]:
        c = run["columns"]
        return {(r[c.index("sid")], r[c.index("turn")]): r[c.index("first")] - r[c.index("release")]
                for r in run["requests"]}
    base = {run["round"]: ttfts(run) for run in ref}
    per_round, pooled = {}, []
    for run in runs:
        if run["round"] in base:
            mine, theirs = ttfts(run), base[run["round"]]
            ratios = [mine[k] / theirs[k] for k in mine.keys() & theirs.keys() if theirs[k] > 0]
            pooled += ratios
            per_round[run["round"]] = round(statistics.median(ratios), 4) if ratios else None
    return {"median": round(statistics.median(pooled), 4) if pooled else None, "per_round": per_round,
            "pairs": len(pooled)}


def run_worker(backend: str, args: argparse.Namespace, out: pathlib.Path,
               kv_tokens: int | None = None) -> tuple[dict | None, str]:
    # PCI order: --gpu is the nvidia-smi index, which also names the GPU class in the evidence.
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(args.gpu), HF_HUB_OFFLINE="1")
    backend, _, label = backend.partition("@")
    match: list[str] = []
    if backend.startswith("holon"):                  # this checkout, whatever copy the venv installed
        env["PYTHONPATH"] = os.pathsep.join(str(ROOT / d) for d in ("core", "std", "plugins"))
        python = sys.executable
        for var in ("HOLON_ENGINE_ARGS", "HOLON_SERVE_ARGS"):   # the lock's settings, the given flags, the arm's
            given = " ".join(filter(None, (env.get(var, ""), args.arms.get(label, ""))))
            if args.lock or given:                   # dict flags merge per key, as over a lock
                engine = args.lock["engine"] if args.lock else {}
                env[var] = shlex.join(lock_flags(engine, SET_BY_WORKER, given, locked_kv=bool(args.lock)))
    else:                                            # a server's client runs here; in-process baselines in their venv
        env.pop("PYTHONPATH", None)
        python = sys.executable if backend in SERVERS else str(ENVS / backend / "bin" / "python")
        match = [*(["--kv-tokens", str(kv_tokens)] if kv_tokens else []),
                 *(["--kv-format", args.kv_format] if args.kv_format else [])]
    cmd = [python, str(BENCH / "worker.py"), "--backend", backend, "--model", args.model, "--traces", *args.traces,
           "--max-batch", str(args.max_batch), "--max-step-tokens", str(args.max_step_tokens), "--out", str(out)]
    if args.blocking_sync:
        cmd.append("--blocking-sync")
    if args.corpus:
        cmd += ["--corpus", args.corpus]
    if args.temperature:
        cmd += ["--temperature", str(args.temperature)]
    cmd += match
    log = out.with_suffix(".log")
    with log.open("w") as f:
        rc = subprocess.run(cmd, env=env, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc:
        return None, f"exit code {rc}: {reason(log)}"
    return json.loads(out.read_text()), ""


def acceptance(stats: dict) -> float | None:
    """Accepted drafts over real drafts verified (speculating backends' stats: Holon's verify rows by k and drafts
    accepted, `spec_rows_k<k>_accepting_<n>`, or a server's accepted count)."""
    accepted = stats.get("spec_accepted", sum(int(k.rsplit("_", 1)[1]) * n for k, n in stats.items()
                                             if k.startswith("spec_rows_k")))
    return round(accepted / stats["spec_drafted"], 3) if stats.get("spec_drafted") else None


def table(summary: dict) -> str:
    lines = ["| trace | backend | TTFT p50 / p99 ms | paired TTFT | TPOT p50 / p99 ms | max gap p50 / p99 / max ms | "
             "out tok/s | decode tok/s (batch) | decode step: wall / CPU ms (p50) | draft acceptance |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for trace, by_backend in summary.items():
        for b, s in by_backend.items():
            d, a, g = s["decode"], acceptance(s["stats"]), s["max_gap_ms"]
            paired = s.get("paired_ttft", {}).get("median")
            lines.append(f"| {trace} | {b} | {s['ttft_ms']['p50']} / {s['ttft_ms']['p99']} | "
                         f"{'-' if paired is None else paired} | {s['tpot_ms']['p50']} / {s['tpot_ms']['p99']} | "
                         f"{g['p50']} / {g['p99']} / {g['max']} | {s['throughput_tok_s']} | "
                         f"{d['tok_s']} ({d['batch']}) | "
                         f"{d['wall_ms_p50']} / {"-" if d['cpu_ms_p50'] is None else d['cpu_ms_p50']} | "
                         f"{"-" if a is None else a} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="workload name for the evidence file")
    ap.add_argument("--gpu", type=int, required=True, help="CUDA device index to pin")
    ap.add_argument("--model", required=True, help="checkpoint directory or file (a .gguf)")
    ap.add_argument("--traces", nargs="+", required=True, help="JSONL specs from workloads.py")
    ap.add_argument("--backends", nargs="+", default=["holon", "minisgl"], type=backend_name,
                    help=f"from {', '.join(BACKENDS)}; <holon backend>@<label> for another arm")
    ap.add_argument("--arm", action="append", default=[], metavar="LABEL=FLAGS",
                    help="Holon engine flags of the arms named <holon backend>@LABEL (over the lock's)")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--lock", help="a lock file: Holon runs its engine settings, the baselines are matched to it")
    ap.add_argument("--max-batch", type=int, help="max running sequences, every backend (default: the lock's, or 64)")
    ap.add_argument("--max-step-tokens", type=int, help="max tokens per prefill step (default: the lock's, or 8192)")
    ap.add_argument("--kv-tokens", type=int, help="the baselines' KV cache size (default: Holon's measured)")
    ap.add_argument("--kv-format", help="the baselines' KV format as Holon's kv.format (default: the lock's)")
    ap.add_argument("--temperature", type=float, default=0.0, help="sample at this temperature (default: greedy)")
    ap.add_argument("--blocking-sync", action="store_true", help="CUDA waits sleep (for CPU time per step)")
    ap.add_argument("--corpus", help="JSONL prompt set whose token ids prompts are cut from (default: uniform ids)")
    ap.add_argument("--evidence-dir", default=str(ROOT / "evidence"))
    args = ap.parse_args()
    args.arms = dict(arm.split("=", 1) if "=" in arm else (arm, "") for arm in args.arm)
    for label, flags in args.arms.items():
        if set(SET_BY_WORKER) & {a[2:].split("=")[0].replace("-", "_") for a in shlex.split(flags) if a[:2] == "--"}:
            ap.error(f"--arm {label}: {', '.join(SET_BY_WORKER)} are set for every backend (--max-batch, ...)")
    if unknown := set(args.arms) - {b.partition("@")[2] for b in args.backends}:
        ap.error(f"--arm {', '.join(sorted(unknown))}: no backend of that label")
    if len(set(args.backends)) < len(args.backends):
        ap.error("--backends: each backend once (another arm of one: <backend>@<label>)")
    if args.lock:
        args.lock = read_lock(args.lock)
        engine = args.lock["engine"]
        args.max_batch = args.max_batch or engine["max_seqs"]
        args.max_step_tokens = args.max_step_tokens or engine["max_step_tokens"]
        if args.kv_format and args.kv_format != engine.get("plugin_options", {}).get("kv.format"):
            ap.error("under --lock the baselines' KV format is the lock's")
        args.kv_format = engine.get("plugin_options", {}).get("kv.format")
        for var in ("HOLON_ENGINE_ARGS", "HOLON_SERVE_ARGS"):   # refuse bad hand-given flags before any run
            for arm in ("", *args.arms.values()):
                lock_flags(engine, SET_BY_WORKER, f"{os.environ.get(var, '')} {arm}")
        if not args.kv_tokens and args.backends[0] != "holon" and not all(b.startswith("holon") for b in args.backends):
            ap.error("--lock matches the baselines to Holon's KV tokens: list holon first, or give --kv-tokens")
    args.max_batch, args.max_step_tokens = args.max_batch or 64, args.max_step_tokens or 8192

    date = datetime.date.today().isoformat()
    gpu = gpu_class(args.gpu)
    raw = BENCH / ".out" / f"{args.name}_{datetime.datetime.now():%Y%m%d-%H%M%S}"
    raw.mkdir(parents=True)
    runs, infos = [], {}
    kv_tokens = args.kv_tokens
    for r in range(args.rounds):
        for b in args.backends if r % 2 == 0 else reversed(args.backends):     # ABBA: no backend always first
            print(f"round {r} {b} ...", flush=True)
            if args.lock and not kv_tokens and not b.startswith("holon"):
                res, err = None, "not matched: Holon's KV tokens are unknown"
            else:
                res, err = run_worker(b, args, raw / f"r{r}-{b}.json", kv_tokens)
            if res is None:
                print(f"  failed:\n{err}", flush=True)
                runs.append({"round": r, "backend": b, "error": err})
                continue
            if args.lock and not kv_tokens:                 # Holon's, for the baselines after it
                kv_tokens = res["info"].get("kv_tokens")
            if b not in infos:
                info = res["info"]
                if "flags" in info.get("config", {}):        # server flags name local files (drafts, …)
                    flags = " ".join(public(a, args.model) for a in shlex.split(info["config"]["flags"]))
                    info = info | {"config": info["config"] | {"flags": flags}}
                base = b.partition("@")[0]
                infos[b] = info | ({"commit": git("rev-parse", "HEAD", cwd=ENVS / "src" / SOURCES[base])}
                                   if base in SOURCES else {})
            for case in res["cases"]:
                runs.append({"round": r, "backend": b, "trace": case["trace"], "columns": res["columns"],
                             "summary": summarize([case], res["columns"], args.blocking_sync), **case})

    ok = [run for run in runs if "error" not in run]
    grouped: dict[str, dict[str, list[dict]]] = {}
    for run in ok:
        grouped.setdefault(run["trace"], {}).setdefault(run["backend"], []).append(run)
    summary = {trace: {b: summarize(rs, rs[0]["columns"], args.blocking_sync) for b, rs in by_backend.items()}
               for trace, by_backend in grouped.items()}
    for trace, by_backend in grouped.items():               # each backend's TTFT against the first's, per request
        ref = by_backend.get(args.backends[0])
        for b, rs in by_backend.items():
            if ref and b != args.backends[0]:
                summary[trace][b]["paired_ttft"] = {"vs": args.backends[0], **paired_ttft(rs, ref)}
    for run in ok:
        del run["steps"]                                    # raw step timings stay in bench/.out
    argv, prev = [], ""
    for a in sys.argv[1:]:
        arm = a.removeprefix("--arm=") if a.startswith("--arm=") else a if prev == "--arm" else None
        if arm is not None and "=" in arm:                  # an arm's flags: each argument made public
            label, flags = arm.split("=", 1)
            a = ("--arm=" if a.startswith("--arm=") else "") + f"{label}={public_flags(flags, args.model)}"
        argv.append(a if arm is not None else public(a, args.model))
        prev = a
    traces = {pathlib.Path(p).stem: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()[:16]
              for p in args.traces}
    # Records written by earlier runs (evidence/) do not make the code dirty; anything else is listed.
    dirty = git("status", "--porcelain", "--untracked-files=no", "--", ".", ":!evidence").splitlines()
    commit = git("rev-parse", "HEAD") + ("-dirty" if dirty else "")
    record = {"workload": args.name, "date": date, "command": " ".join(["python bench/run.py", *argv]),
              "commit": commit, **({"dirty": dirty} if dirty else {}), "gpu_class": gpu,
              "model": model_id(args.model), "rounds": args.rounds,
              "order": args.backends, **({"arms": {k: public_flags(v, args.model) for k, v in args.arms.items()}}
                                     if args.arms else {}), "method": METHOD,
              "traces_sha256": traces,
              "matched": {"lock": args.lock and {k: v for k, v in args.lock.items() if k != "engine"},
                          "max_seqs": args.max_batch, "max_step_tokens": args.max_step_tokens,
                          "kv_tokens": kv_tokens, "kv_format": args.kv_format, "temperature": args.temperature,
                          "samplers": "top_k / top_p / min_p off on every engine"},
              **({"corpus": {pathlib.Path(args.corpus).name: hashlib.sha256(pathlib.Path(args.corpus).read_bytes())
                             .hexdigest()[:16]}} if args.corpus else {}), "backends": infos,
              "summary": summary, "runs": runs}
    out = pathlib.Path(args.evidence_dir) / f"bench_{args.name}_{gpu}_{date}.json"
    text = json.dumps(record, indent=1, default=str)
    text = re.sub(r"\[\s+([^\[\]{}]*?)\s+\]", lambda m: "[" + re.sub(r"\s+", " ", m[1]) + "]", text)  # rows on one line
    out.write_text(text + "\n")
    print(table(summary))
    print(f"wrote {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")


if __name__ == "__main__":
    main()
