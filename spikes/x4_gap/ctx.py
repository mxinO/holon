"""x4_gap's driver: `bench/offload_ctx.py`'s method (fresh process per arm, an untimed 512 + 64 warm-up, one request per
context of natural-text prompt ids generating 256 tokens, greedy, ignore_eos; interleaved rounds) with a Holon worker
beside its llama.cpp and FreeToken ones, and each engine's expert-cache counters:

- holon: `Engine.step()` in process, a stamp as each token reaches the host (as FreeToken's frontend stamps); with the
  offload store, every decode step's routed (layer, expert) pairs and the residency map are recorded from the store's
  own telemetry (`after_commit`'s `counts`), so misses and bytes per token are counted, not modelled. Nothing in
  core / std / plugins changes: the recorder wraps `TieredStore.after_commit` in this process only.
- freetoken: offload_ctx's worker plus `moe_collect_stats` (its in-graph decode miss counters), read per request.
- llama: offload_ctx's worker as is.

`chunk` / `kernel_chunk` (arm keys) set std.moe's expert-path pass size and kernels.ggml_cuda's mmq rows per call in
the Holon worker's process only (prefill probes for D8; both 512 as shipped).
`--nsys-context C` brackets context C's request with cudaProfilerStart/Stop (for `nsys --capture-range=cudaProfilerApi`)
and marks its decode with an NVTX range from the first token to the last (Holon and FreeToken).

    python spikes/x4_gap/ctx.py run --name <n> --arms <arms.json> --model <gguf> [--rounds 3] [--contexts 1024 4096] \\
        [--holon-python <venv python>] [--python <freetoken venv python>] [--server <llama-server>]
    python spikes/x4_gap/ctx.py worker --engine holon --model <gguf> --flags '<holon serve flags>' [--trace out.npz]
"""
from __future__ import annotations

import argparse
import contextlib
import json
import pathlib
import shlex
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
import offload_ctx  # noqa: E402  (bench/: the prompts, the llama worker, the orchestrator's pieces)
import workloads  # noqa: E402

OUTPUT = offload_ctx.OUTPUT


@contextlib.contextmanager
def profiled(on: bool):
    """cudaProfilerStart/Stop around a request (nsys `--capture-range=cudaProfilerApi`)."""
    import torch
    if not on:
        yield
        return
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    try:
        yield
    finally:
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()


def holon_worker(a: argparse.Namespace, corpus: list[int]) -> dict:
    import dataclasses

    import numpy as np
    import torch
    from holon.api import SamplingParams
    from holon.core.cli import engine_config, engine_flags
    from holon.core.engine import Engine

    rec = {"on": False, "steps": [], "store": None}
    try:
        from holon_std.tiered.store import TieredStore
    except ImportError:
        TieredStore = None
    if TieredStore is not None:
        commit = TieredStore.after_commit

        def recorded(self, step, telemetry):
            rec["store"] = self
            if rec["on"]:
                rec["steps"].append(np.flatnonzero(telemetry["counts"].numpy().reshape(-1)).astype(np.int32))
            return commit(self, step, telemetry)

        TieredStore.after_commit = recorded

    if a.chunk:                     # the expert path's pass size (std.moe's CHUNK), patched in this process only
        from holon_std.moe import units
        units.CHUNK = a.chunk
    if a.kernel_chunk:              # kernels.ggml_cuda's mmq rows per call (llama.cpp's ubatch, 512), likewise
        from holon_plugins.kernels.ggml_cuda import plugin as ggml
        ggml.CHUNK = a.kernel_chunk
    flags = argparse.ArgumentParser()
    engine_flags(flags)
    cfg = engine_config(a.model, flags.parse_args(shlex.split(a.flags)))
    engine = Engine(dataclasses.replace(cfg, max_seqs=1, max_model_len=a.ctx, prefix_cache=False))

    def request(ids: list[int], n: int, profile: bool = False) -> dict:
        params = SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True)
        stamps, done = [], False
        rec["steps"] = []
        with profiled(profile):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            rid = engine.add(ids, params)
            while not done:
                for d in engine.step():
                    if d.rid == rid:
                        now = time.perf_counter()
                        if not stamps and d.token_ids:
                            rec["on"] = True
                            if profile:
                                torch.cuda.nvtx.range_push("decode")
                        stamps += [now] * len(d.token_ids)
                        done = d.finish_reason is not None
            if profile:
                torch.cuda.nvtx.range_pop()
            rec["on"] = False
        assert len(stamps) == n, (len(stamps), n)
        out = {"prefill_tps": len(ids) / (stamps[0] - t0), "decode_tps": (n - 1) / (stamps[-1] - stamps[0])}
        if rec["store"] is not None and rec["steps"]:
            out |= cache_counts(rec["store"], rec["steps"])
        return out

    off, p, n = offload_ctx.WARMUP
    request([corpus[(off + i) % len(corpus)] for i in range(p)], n)
    results, traces = {}, {}
    for c in a.contexts:
        results[str(c)] = {k: round(v, 4) if isinstance(v, float) else v
                           for k, v in request(offload_ctx.prompt(corpus, c), OUTPUT, c == a.nsys_context).items()}
        traces[str(c)] = list(rec["steps"])
    out = {"results": results}
    if rec["store"] is not None:
        s = rec["store"]
        resident = s.residency.host >= 0                                     # [groups, items, roles]
        unit = np.array([[s.groups[g][r].unit_bytes for r in range(s.roles)] for g in sorted(s.groups)])
        out["store"] = {"cached_experts": int(resident.all(-1).sum()), "experts": int(resident[..., 0].size),
                        "classes": {c.name: {"unit_bytes": c.unit_bytes, "slots": c.size, "staging": c.staging,
                                             "pieces": len(c.pieces)} for c in s.classes.values()}}
        if a.trace:
            steps = {f"keys_{c}": np.concatenate(v) for c, v in traces.items() if v}       # layer * E + expert
            ptrs = {f"ptr_{c}": np.cumsum([0, *map(len, v)]) for c, v in traces.items() if v}
            np.savez_compressed(a.trace, resident=resident.all(-1), item_bytes=unit.sum(1), **steps, **ptrs)
    engine.close()
    return out


def cache_counts(store, steps: list) -> dict:
    """Per decode token (batch 1: one decode row per step): routed experts, misses and the bytes F5a gathers."""
    import numpy as np
    resident = (store.residency.host >= 0).all(-1).reshape(-1)
    unit = np.array([sum(store.groups[g][r].unit_bytes for r in range(store.roles)) for g in sorted(store.groups)])
    item_bytes = np.repeat(unit, store.items)
    routed = sum(len(x) for x in steps)
    misses = [x[~resident[x]] for x in steps]
    n = len(steps)
    return {"decode_steps": n, "routed_per_token": routed / n, "misses_per_token": sum(map(len, misses)) / n,
            "hit_rate": 1 - sum(map(len, misses)) / routed, "miss_mb_per_token": sum(item_bytes[m].sum()
                                                                                 for m in misses) / n / 1e6}


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
    kwargs["moe_collect_stats"] = True
    marks = {"profile": False}

    class Timed(LLM):
        stamps: list[float] = []

        def offline_send_result(self, reply):
            now = time.perf_counter()
            new = [now for m in reply if isinstance(m, DetokenizeMsg)]
            if new and not self.stamps and marks["profile"]:
                torch.cuda.nvtx.range_push("decode")
            self.stamps += new
            super().offline_send_result(reply)

    llm = Timed(a.model, dtype=cfg.dtype, **kwargs)
    cache = getattr(llm.engine, "moe_offload_cache", None)

    def request(ids: list[int], n: int, profile: bool = False) -> dict:
        llm.stamps = []
        if cache is not None:
            cache.reset_stats()
        marks["profile"] = profile
        with profiled(profile):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            llm.generate([ids], SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True))
            if profile:
                torch.cuda.nvtx.range_pop()
        s = llm.stamps
        assert len(s) == n, (len(s), n)
        out = {"prefill_tps": len(ids) / (s[0] - t0), "decode_tps": (n - 1) / (s[-1] - s[0])}
        if cache is not None:
            st = cache.decode_miss_stats()
            out |= {"ft_" + k: v for k, v in st.items() if isinstance(v, (int, float))}
        return out

    off, p, n = offload_ctx.WARMUP
    request([corpus[(off + i) % len(corpus)] for i in range(p)], n)
    out = {"results": {str(c): {k: round(v, 4) if isinstance(v, float) else v
                                for k, v in request(offload_ctx.prompt(corpus, c), OUTPUT,
                                                    c == a.nsys_context).items()} for c in a.contexts}}
    if cache is not None:
        out["cache"] = {"cache_size": getattr(cache, "cache_size", None), "decode_target": cache.decode_target}
    return out


def worker(a: argparse.Namespace) -> None:
    corpus = workloads.corpus(offload_ctx.CORPUS)
    fn = {"holon": holon_worker, "freetoken": freetoken_worker, "llama": offload_ctx.llama_worker}[a.engine]
    print("RESULT " + json.dumps(fn(a, corpus)), flush=True)


def orchestrate(a: argparse.Namespace) -> None:
    """offload_ctx's orchestrator, with each engine's python and this file's worker."""
    import os
    import subprocess
    import threading

    arms = json.loads(pathlib.Path(a.arms).read_text())
    outdir = pathlib.Path(a.out_dir) / a.name
    outdir.mkdir(parents=True, exist_ok=True)
    raw = outdir / "raw.jsonl"
    rows = [json.loads(x) for x in raw.read_text().splitlines()] if raw.exists() else []
    done = {(r["round"], r["arm"]) for r in rows if "error" not in r}
    py = {"holon": a.holon_python, "freetoken": a.python, "llama": sys.executable}
    for rnd in range(a.rounds):
        for arm in (arms if rnd % 2 == 0 else arms[::-1]):
            if (rnd, arm["name"]) in done or rnd >= arm.get("rounds", a.rounds):
                continue
            for _ in range(60):
                if not (busy := offload_ctx.gpu_busy(a.gpu)):
                    break
                time.sleep(10)
            else:
                raise SystemExit(f"GPU stayed busy: {busy}")
            log = outdir / f"r{rnd}_{arm['name']}.log"
            cmd = [*shlex.split(arm.get("wrap", "")), py[arm["engine"]], str(pathlib.Path(__file__)), "worker",
                   "--engine", arm["engine"], "--model", a.model, "--ctx", str(arm.get("ctx", a.ctx)),
                   "--flags", arm["flags"], "--contexts", *map(str, arm.get("contexts", a.contexts)),
                   "--nsys-context", str(arm.get("nsys_context", 0))]
            if arm["engine"] == "llama":
                cmd += ["--server", a.server, "--log", str(log.with_suffix(".server.log")),
                        "--wrap", arm.get("server_wrap", "")]
            if arm["engine"] == "holon":
                cmd += ["--trace", str(outdir / f"r{rnd}_{arm['name']}.npz"), "--chunk", str(arm.get("chunk", 0)),
                        "--kernel-chunk", str(arm.get("kernel_chunk", 0))]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID")
            stop, peak = threading.Event(), [0]
            threading.Thread(target=offload_ctx.peak_vram, args=(a.gpu, stop, peak), daemon=True).start()
            print(f"round {rnd} {arm['name']}", flush=True)
            t = time.monotonic()
            with open(log, "w") as f:
                p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
            stop.set()
            text = log.read_text(errors="replace")
            line = next((x for x in text.splitlines() if x.startswith("RESULT ")), None)
            r = {"round": rnd, "arm": arm["name"]}
            if p.returncode or line is None:
                r["error"] = text.strip().splitlines()[-1][:300] if text.strip() else f"exit {p.returncode}"
            else:
                r |= json.loads(line[7:]) | {"peak_vram_mib": peak[0], "wall_s": round(time.monotonic() - t, 1)}
            print("  " + json.dumps({k: r[k] for k in ("results", "error", "peak_vram_mib") if k in r}), flush=True)
            rows.append(r)
            with open(raw, "a") as f:
                f.write(json.dumps(r) + "\n")
    summary = offload_ctx.summary(rows)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("worker")
    w.add_argument("--engine", choices=("holon", "freetoken", "llama"), required=True)
    w.add_argument("--flags", default="")
    w.add_argument("--server")
    w.add_argument("--log")
    w.add_argument("--wrap", default="")
    w.add_argument("--trace", help="holon: the decode routing per context (.npz)")
    w.add_argument("--nsys-context", type=int, default=0)
    w.add_argument("--chunk", type=int, default=0, help="holon: tokens per expert-path pass (std.moe CHUNK; 0: 512)")
    w.add_argument("--kernel-chunk", type=int, default=0, help="holon: kernels.ggml_cuda's mmq rows per call (0: 512)")
    o = sub.add_parser("run")
    o.add_argument("--name", required=True)
    o.add_argument("--arms", required=True)
    o.add_argument("--gpu", type=int, default=0)
    o.add_argument("--rounds", type=int, default=3)
    o.add_argument("--out-dir", default=str(ROOT / "bench" / ".out"))
    o.add_argument("--holon-python", default=sys.executable)
    o.add_argument("--python", default=str(ROOT / "bench" / ".envs" / "freetoken-offload" / "bin" / "python"))
    o.add_argument("--server", default=str(ROOT / "bench" / ".envs" / "llama.cpp-offload" / "bin" / "llama-server"))
    for p in (w, o):
        p.add_argument("--model", required=True)
        p.add_argument("--contexts", type=int, nargs="+", default=[1024, 4096])
        p.add_argument("--ctx", type=int, default=33792)
    a = ap.parse_args()
    worker(a) if a.cmd == "worker" else orchestrate(a)


if __name__ == "__main__":
    main()
