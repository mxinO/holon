"""Milestone K step 2 (docs/dev/v1-kv-tiers.md, rev 8.2): `state_copy@1` beside prefill (a), its host cost per load (b),
and the mixed NVMe factor under "reads go first" (c).

    CUDA_VISIBLE_DEVICES=0 python bench.py --tag rtx-5080 --only a,b,c --model <27B gguf> --lock <lock> \
        --nvme-dir <node-local dir> [--l0 <l0 summary json>] [--clock-screen <clock_check json>]

Writes `evidence/k2_state_copy_<tag>_<date>.json` (L0 summary, a, b) and `evidence/k2_mixed_nvme_<tag>_<date>.json`
(c). The copies go through the engine's plugin set (`kernels.state_copy`'s `state_copy@1`), as std.state_tier binds
them; pools are the core's `memory.Pool` (host pools: `platform.host_memory`, mapped).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "bench"))
sys.path.insert(0, str(REPO / "spikes/k0_tiers/bench"))
import bench as k0  # noqa: E402  (K step 0's pools, I/O threads, files and helpers)

CHUNK, SEGMENTS, KV_PAGE = 34_816, 16, 34_816 * 16   # the 27B's q8_0 KV page: 16 layer chunks
GB = 1e9
PREDICTIONS = {
    "l0": "bitwise equal to the oracle, both directions and paths, chunks 34,816 / 65,536 / 61,440 / 3,145,728 B and "
          "n = 0; compute-sanitizer memcheck clean",
    "a_per_step": "one 0.57 / 1.29 GB gather per 512-token prefill step (~0.22 s): median +0.5-2%, p90 <= +3%",
    "a_back_to_back": "1.29 GB gathers back to back: median +3-8%, p90 <= +10%; gather 40-46 GB/s beside prefill",
    "b_build_us": {"16K": [30, 80], "32K": [60, 160]},
    "b_submit_sm_us": {"16K": [40, 100], "32K": [80, 200]},
    "b_per_chunk_ns": [4, 10],
    "b_submit_dma_ms": {"16K": 5, "32K": [21, 25]},
    "c_mixed_over_model": {"0.5": [0.80, 0.90], "0.75": [0.75, 0.85]},
}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def copier(direction: str):
    """`state_copy@1` as std.state_tier selects it."""
    from holon.core.build import BuildContext
    from holon.core.plugins import PluginSet, discover
    from holon_std.state_tier.kernels import STATE_COPY

    ctx = BuildContext(PluginSet(discover()), torch.device("cuda"))
    spec = ctx.select_kernel(STATE_COPY, site="k2", direction=direction, chunk=CHUNK)
    return spec.provider, spec.impl(ctx, direction=direction, chunk=CHUNK)


def rows(units: np.ndarray, capacity: int) -> torch.Tensor:
    """The tier's row list of `units` in a layer-major pool: row = s · capacity + u for each of the 16 segments."""
    return torch.from_numpy((np.arange(SEGMENTS, dtype=np.int64)[:, None] * capacity + units[None, :]).ravel())


class Pools:
    """A device KV pool and a mapped host KV pool, layer-major [16, capacity, 34,816], as rows of one chunk."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.dev = k0.pool("k2.dev", CHUNK, SEGMENTS, capacity, False)
        self.host = k0.pool("k2.host", CHUNK, SEGMENTS, capacity, True)
        for p in (self.dev, self.host):
            k0.device_bytes(p).view(torch.int64).random_()
        self.d = self.dev.buffer.view(-1, CHUNK)
        self.h = self.host.device_view(torch.uint8, (SEGMENTS * capacity, CHUNK))

    def load(self, n: int, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        return (rows(rng.choice(self.capacity, n, replace=False), self.capacity),
                rows(rng.choice(self.capacity, n, replace=False), self.capacity))


# ---------------------------------------------------------------------------------------------- b


def host_cost(pools: Pools, reps: int = 50) -> dict:
    """Host µs per load: the numpy row lists (both sides) and the provider's submit, with the GPU idle between calls,
    and back to back (calls queued behind each other, pinned blocks still in use)."""
    out = {}
    rng = np.random.default_rng(0)
    torch.cuda.set_stream(torch.cuda.Stream())               # the tier's copy stream (not the legacy default stream)
    for direction in ("h2d", "d2h"):
        provider, copy = copier(direction)
        src, dst = (pools.h, pools.d) if direction == "h2d" else (pools.d, pools.h)
        for load, units in (("16K", 1024), ("32K", 2048)):
            build, submit, queued = [], [], []
            for i in range(reps + 3):
                su, du = (rng.choice(pools.capacity, units, replace=False) for _ in range(2))
                torch.cuda.synchronize()
                t = time.perf_counter()
                s_rows, d_rows = rows(su, pools.capacity), rows(du, pools.capacity)
                t1 = time.perf_counter()
                copy(src, s_rows, dst, d_rows)
                t2 = time.perf_counter()
                if i >= 3:                                      # the first calls allocate pinned blocks
                    build.append((t1 - t) * 1e6)
                    submit.append((t2 - t1) * 1e6)
            torch.cuda.synchronize()
            lists = [pools.load(units, rng) for _ in range(12)]
            for s_rows, d_rows in lists:                        # back to back: the GPU behind the host
                t = time.perf_counter()
                copy(src, s_rows, dst, d_rows)
                queued.append((time.perf_counter() - t) * 1e6)
            torch.cuda.synchronize()
            chunks = units * SEGMENTS
            row = {"provider": provider, "chunks": chunks, "build_us": k0.summarize(build),
                   "submit_us": k0.summarize(submit), "submit_queued_us": k0.summarize(queued),
                   "total_us_p50": statistics.median(build) + statistics.median(submit)}
            row["ns_per_chunk"] = row["total_us_p50"] * 1e3 / chunks
            out[f"{direction}_{load}"] = row
            log(f"b {direction} {load}: build {row['build_us']['p50']:.0f} us + submit "
                f"{row['submit_us']['p50']:.0f} us (queued {row['submit_queued_us']['p50']:.0f}) = "
                f"{row['ns_per_chunk']:.1f} ns per chunk")
    torch.cuda.set_stream(torch.cuda.default_stream())
    return out


# ---------------------------------------------------------------------------------------------- a


def prefill_beside_gather(pools: Pools, args) -> dict:
    """The 27B prefilling in 512-token steps while the gather loads on the tier's copy stream: per-step wall time per
    arm, ABBA rounds. Arms: no copies; one 0.57 / 1.29 GB load enqueued before each step; 1.29 GB loads back to back."""
    from clock_check import STEADY, Sampler
    from holon.api import SamplingParams
    from holon.core.cli import engine_config, engine_flags
    from holon.core.engine import Engine
    from run import lock_flags, read_lock

    provider, copy = copier("h2d")
    lock = read_lock(args.lock)
    flags = argparse.ArgumentParser()
    engine_flags(flags)
    cfg = engine_config(args.model, flags.parse_args(lock_flags(lock["engine"])))
    # prefill-only 512-token steps; contexts up to 2 prompts (the load's 1.34 GB device pool leaves less KV than the
    # lock's max_model_len)
    engine = Engine(dataclasses.replace(cfg, max_step_tokens=512, chunk_with_decode=0, max_model_len=2 * args.prompt))
    stream = torch.cuda.Stream()
    rng = np.random.default_rng(1)
    loads = {"0.57GB": [pools.load(1024, rng) for _ in range(4)], "1.29GB": [pools.load(2324, rng) for _ in range(4)]}
    nbytes = {k: len(v[0][0]) * CHUNK for k, v in loads.items()}
    prompt_rng = np.random.default_rng(2)
    live: list[str] = []

    def feed():
        """Keep one long prompt prefilling (a new one when the last is done)."""
        if not engine.sched.waiting and not any(s.num_computed < s.prompt_len for s in engine.sched.running):
            for rid in live:
                engine.abort(rid)
            live.clear()
            live.append(engine.add(prompt_rng.integers(1000, 30000, args.prompt).tolist(),
                                   SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)))

    pending: list[tuple[torch.cuda.Event, torch.cuda.Event, int]] = []
    rates: dict[str, list[float]] = {}
    k = [0]

    def enqueue(label: str):
        s_rows, d_rows = loads[label][k[0] % 4]
        k[0] += 1
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            a.record()
            copy(pools.h, s_rows, pools.d, d_rows)
            b.record()
        pending.append((a, b, nbytes[label]))

    def drain(arm: str):
        stream.synchronize()
        rates.setdefault(arm, []).extend(n / a.elapsed_time(b) / 1e6 for a, b, n in pending)
        pending.clear()

    for _ in range(6):                                          # warm-up: graphs, autotuning, the first pinned blocks
        feed()
        engine.step()
    enqueue("1.29GB")
    drain("warm")
    arms = ["control", "per_step_0.57GB", "per_step_1.29GB", "back_to_back_1.29GB"]
    order = arms + arms[::-1]
    raw = {a: [] for a in arms}
    clocks = {a: [] for a in arms}
    sampler = Sampler()
    for rnd in range(args.rounds):
        for arm in order:
            ts, t0 = [], time.time()
            t = time.perf_counter()
            for i in range(args.warm + args.steps):
                feed()
                if arm.startswith("per_step"):
                    enqueue(arm.split("_")[-1])
                elif arm.startswith("back_to_back"):
                    while sum(not b.query() for _, b, _ in pending) < 12:   # ~12 loads (0.3 s) queued ahead
                        enqueue("1.29GB")
                engine.step()
                now = time.perf_counter()
                if i >= args.warm:
                    ts.append((now - t) * 1e3)
                t = now
            drain(arm)
            raw[arm].append(ts)
            clocks[arm].append(sampler.summary(t0, time.time()))
            log(f"a round {rnd} {arm}: step p50 {statistics.median(ts):.1f} ms p90 {sorted(ts)[len(ts) * 9 // 10]:.1f}"
                + (f", gather {statistics.median(rates[arm]):.1f} GB/s" if rates.get(arm) else ""))
    sampler.close()
    for rid in live:
        engine.abort(rid)
    engine.close()
    res = {"provider": provider, "lock": lock["path"], "plan_id": lock["plan_id"], "model": Path(args.model).name,
           "max_step_tokens": 512, "chunk_with_decode": 0, "max_model_len": 2 * args.prompt,
           "prompt_tokens": args.prompt,
           "steps_per_segment": args.steps, "warm_steps": args.warm, "rounds": args.rounds, "order": order,
           "load_bytes": nbytes, "arms": {}}
    base = k0.summarize([x for seg in raw["control"] for x in seg])
    for arm in arms:
        s = k0.summarize([x for seg in raw[arm] for x in seg])
        r = {"step_ms": s, "raw_ms": [[round(x, 2) for x in seg] for seg in raw[arm]], "clocks": clocks[arm]}
        if arm != "control":
            r.update(median_ratio=s["p50"] / base["p50"], p90_ratio=s["p90"] / base["p90"],
                     mean_ratio=s["mean"] / base["mean"], gather_gbps=statistics.median(rates[arm]))
        res["arms"][arm] = r
    allc = [c for arm in arms for c in clocks[arm] if c.get("samples")]
    res["clock_steady"] = bool(allc) and min(c["sm_mhz"]["p10"] for c in allc) >= STEADY * statistics.median(
        c["sm_mhz"]["p50"] for c in allc)
    return res


# ---------------------------------------------------------------------------------------------- c


def mixed(args) -> dict:
    """1 writer + 2 or 3 readers of 557,056 B units with O_DIRECT, each kind held at its byte share: rev 8.2's dispatch
    (while reads are queued at most one write in flight: one writer thread). Against D4's serial lane
    1 / (f / read_N + (1 - f) / write_1) from pure rates at the same thread counts taken just before."""
    root = Path(args.nvme_dir)
    d = root / f"k2_nvme_{os.getpid()}"
    d.mkdir()
    out = {"storage": args.storage, "rows": []}
    unit = (KV_PAGE, 1, False)
    read_path, write_path = d / "read_slots.bin", d / "write_slots.bin"
    try:
        st = os.statvfs(root)
        free = st.f_bavail * st.f_frsize
        size = min(16 << 30, free // 8) // KV_PAGE * KV_PAGE
        wsize = min(8 << 30, free // 8) // KV_PAGE * KV_PAGE
        out["read_file_how"], out["write_file_how"] = k0.create(read_path, size), k0.create(write_path, wsize)
        out["fill_gbps"] = k0.fill(read_path, size)
        secs = args.io_seconds

        def pure(kind: str, threads: int) -> float:
            path, fsize = (write_path, wsize) if kind == "write" else (read_path, size)
            return k0.io_window(secs, k0.IO(path, kind == "write", True, unit, KV_PAGE, fsize // KV_PAGE, threads,
                                            seed=4))[0]["gbps"]

        for readers_n in (2, 3):
            for f in (0.5, 0.75, None):
                r_pure, w_pure = pure("read", readers_n), pure("write", 1)

                def gate(io, n, f=f):
                    rb, wb = readers.bytes, writers.bytes                 # noqa: B023 (this cell's, set below)
                    tot = rb + wb + n
                    return (rb + n <= f * tot + n) if not io.write else (wb + n <= (1 - f) * tot + n)
                g = gate if f is not None else None
                readers = k0.IO(read_path, False, True, unit, KV_PAGE, size // KV_PAGE, readers_n, gate=g, seed=1)
                writers = k0.IO(write_path, True, True, unit, KV_PAGE, wsize // KV_PAGE, 1, gate=g, seed=2)
                rr, ww = k0.io_window(max(secs, 3.0), readers, writers)
                measured = (rr["bytes"] + ww["bytes"]) / max(rr["seconds"], ww["seconds"]) / GB
                share = rr["bytes"] / (rr["bytes"] + ww["bytes"])
                model = 1 / (share / r_pure + (1 - share) / w_pure)
                out["rows"].append({"readers": readers_n, "writers": 1, "read_share_target": f, "read_share": share,
                                    "read": rr, "write": ww, "measured_gbps": measured, "model_gbps": model,
                                    "mixed_factor": measured / model,
                                    "pure": {"read": r_pure, "write": w_pure}})
                log(f"c 1 writer + {readers_n} readers f={f}: share {share:.2f}, {measured:.2f} GB/s (reads "
                    f"{rr['gbps']:.2f}, writes {ww['gbps']:.2f}) vs model {model:.2f}: factor {measured / model:.3f}")
    finally:
        for p in d.glob("*"):
            p.unlink()
        d.rmdir()
        out["test_file_removed"] = not d.exists()
        log(f"c test files removed: {out['test_file_removed']}")
    return out


# ---------------------------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--only", default="a,b,c")
    ap.add_argument("--out-dir", default=str(REPO / "evidence"))
    ap.add_argument("--date", default=time.strftime("%Y-%m-%d"))
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--lock")
    ap.add_argument("--prompt", type=int, default=8192, help="tokens of each prefilling prompt (a)")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--l0", help="a JSON file of L0 results (pytest and memcheck) to record")
    ap.add_argument("--clock-screen")
    ap.add_argument("--nvme-dir")
    ap.add_argument("--storage", default="local NVMe")
    ap.add_argument("--io-seconds", type=float, default=3.0)
    args = ap.parse_args()
    only = set(args.only.split(","))
    torch.cuda.init()
    run = {"command": k0.public_command(sys.argv), "date": args.date, "env": k0.env_facts()}

    def record(kind: str):
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        path = Path(args.out_dir) / f"k2_{kind}_{args.tag}_{args.date}.json"
        res = json.loads(path.read_text()) if args.merge and path.exists() else {}
        res.setdefault("runs", []).append(run)
        res.update({"tag": args.tag, "env": run["env"], "predictions": PREDICTIONS})
        return path, res

    def save(path, res):
        path.write_text(json.dumps(res, indent=1, default=float) + "\n")
        log("wrote", path)

    if only & {"a", "b"} or args.l0:
        path, res = record("state_copy")
        if args.l0:
            res.setdefault("l0", {}).update(json.loads(Path(args.l0).read_text()))
        if args.clock_screen:
            res["clock_screen"] = json.loads(Path(args.clock_screen).read_text())
        if only & {"a", "b"}:
            pools = Pools(2400)                                 # before the engine: it sizes its KV around them
            if "b" in only:
                res["b"] = host_cost(pools)
                save(path, res)
            if "a" in only:
                res["a"] = prefill_beside_gather(pools, args)
        save(path, res)
    if "c" in only:
        path, res = record("mixed_nvme")
        res["c"] = mixed(args)
        save(path, res)


if __name__ == "__main__":
    main()
