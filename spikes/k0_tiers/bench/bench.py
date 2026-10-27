"""Milestone K step 0 (docs/dev/v1-kv-tiers.md §8): K-M2 copy paths (K-M3 folded in) and K-M4 NVMe.

    CUDA_VISIBLE_DEVICES=0 python bench.py --tag rtx-5080 [--only m2,m3,m4] [--quick] [--merge] \
        --model <27B gguf> --lock <lock> --nvme-dir <dir> --storage "local NVMe"

Writes `evidence/k0_m2_copy_<tag>_<date>.json` (K-M2 and K-M3) and `evidence/k0_m4_nvme_<tag>_<date>.json`. The
extension in csrc/ is built under $TORCH_EXTENSIONS_DIR (default local/tmp/k-build/ext) per tag.

K-M2 allocates its host pools with the core's own `holon.core.memory.Pool` (so `platform.host_memory`: an anonymous
mmap + `cudaHostRegister`, mapped for the SM path) and runs today's per-run loop as `memory.copy_units` itself.
"""
from __future__ import annotations

import argparse
import json
import mmap
import os
import random
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "bench"))              # run.lock_flags, clock_check.Sampler (K-M3)

MB, GB = 1e6, 1e9
KV_PAGE = 34_816 * 16                                # a 27B q8_0 KV page: 16 layers x 34,816 B = 557,056 B
CONV_SEG, SSM_SEG, GDN_LAYERS = 61_440, 3_145_728, 48  # the 27B's GDN checkpoint segments (std's GDN state specs)
PAYLOAD = GDN_LAYERS * (CONV_SEG + SSM_SEG)          # 153,944,064 B: one checkpoint (conv unit + SSM unit)
# chunk shapes: (bytes per segment, segments per unit)
SHAPES = {"kv_27b_q8_0": (34_816, 16), "kv_8b_bf16": (65_536, 36), "gdn_conv_27b": (CONV_SEG, GDN_LAYERS),
          "gdn_ssm_27b": (SSM_SEG, GDN_LAYERS)}
# loads in bytes: 1K / 16K tokens of the 27B's KV, and a 32K resume (KV + one checkpoint)
LOADS = {"1K": 64 * KV_PAGE, "16K": 1024 * KV_PAGE, "32K": 2048 * KV_PAGE + PAYLOAD}
FRAGS = {"random": 1, "runs4": 4, "runs16": 16}
TILE = 64 << 10                                      # the SM copy's largest work item
CTAS, UNROLLS = (2, 4, 8, 16, 32, 64, 128), (2, 4, 8)

PREDICTIONS = {  # fixed before measuring (docs/dev/v1-kv-tiers.md §8, the step-0 brief)
    "m2": {
        "blob_h2d_gbps": {"rtx-5080": [46.6, 48.4], "rtx-5090": [46.6, 48.4], "rtx-a5000": [22.0, 26.0],
                          "note": "one big copy from S2 host memory; X's M1 46.6-48.4 (47.5 nominal)"},
        "blob_d2h_gbps": {"rtx-5080": [45.0, 53.0], "rtx-5090": [45.0, 53.0], "rtx-a5000": [22.0, 26.0]},
        "loop_fragmented_34816": "<= 0.5x blob and host-bound; 5-10 us of host time per chunk (0.15-0.3 s for a "
                                 "fragmented 32K load of 32,768 chunks; the design says ~0.3 s)",
        "batch_fragmented_34816": "0.7-0.95x blob (uncertain); >= 0.9x at 65,536 B and >= 3 MiB; host time "
                                  "0.1-0.3 us per chunk",
        "sm_fragmented_34816": "gather H2D >= 0.9x blob with 8-32 CTAs; scatter D2H to mapped host >= 0.8x",
    },
    "m3": {
        "h2d_loads": "decode step median slowdown <= 5% with 0.5-2 GB H2D loads in flight (copy-engine DMA plus "
                     "~5% of DRAM bandwidth), p90 <= 15% (3b's bar: 1.1x median, 1.2x p90)",
        "d2h_stores": "154 MB D2H stores at the answer rate cost < 0.5% of decode (write-through stays default "
                      "unless stores cost > 2%)",
    },
    "m4": {
        "bar_desktops": "O_DIRECT reads of 557,056 B units 1.6-2.2 GB/s at 1 thread, 2.4-2.8 GB/s at 2, no gain "
                        "past 2-3; burst writes 1.5-3 GB/s; sustained past SLC 0.5-1.5 GB/s; buffered reads slower "
                        "or equal with DONTNEED; measured mixed throughput within 15% of the modelled shared lane; "
                        "per-segment checkpoint iovecs (48 segments) within 10% of the unit rate; engine-loop "
                        "slowdown beside 2 I/O threads < 3% (preadv releases the GIL)",
        "read_1t_gbps": [1.6, 2.2], "read_2t_gbps": [2.4, 2.8], "burst_write_gbps": [1.5, 3.0],
        "sustained_write_gbps": [0.5, 1.5], "mixed_vs_model": 0.15, "segments_vs_unit": 0.10,
        "loop_slowdown_2t": 0.03,
    },
}
DECISION_RULES = {
    "read_2t": "STOP if the 2-thread O_DIRECT read rate of 557,056 B units is < 1.75 GB/s (bar desktops)",
    "mixed": "STOP if the measured mixed rate is off the modelled shared-lane rate by > 15% (bar desktops)",
    "copy": "STOP if neither cudaMemcpyBatchAsync nor the SM gather reaches 0.8x of the blob rate on fragmented "
            "34,816 B chunks (bar desktops)",
    "stores": "STOP if 154 MB stores cost > 2% of decode at the answer rate (bar desktops)",
}
BAR = ("rtx-5080", "rtx-5090")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def summarize(xs) -> dict:
    xs = sorted(xs)
    return {"p50": statistics.median(xs), "mean": statistics.fmean(xs), "p10": xs[len(xs) // 10],
            "p90": xs[len(xs) * 9 // 10], "n": len(xs)}


def public_command(argv: list[str]) -> str:
    """The command line with no machine paths: paths inside the checkout as <repo>/..., others as <local>."""
    roots = [REPO, Path(sh(f"git -C {HERE} rev-parse --git-common-dir")).resolve().parent]
    out = [Path(argv[0]).name]
    for a in argv[1:]:
        if a.startswith("/"):
            root = next((r for r in roots if Path(a).resolve().is_relative_to(r)), None)
            a = f"<repo>/{Path(a).resolve().relative_to(root)}" if root else "<local>"
        out.append(a)
    return " ".join(out)


def build_ext(tag: str):
    major, minor = torch.cuda.get_device_capability()
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    from torch.utils.cpp_extension import load

    root = Path(os.environ.get("TORCH_EXTENSIONS_DIR", REPO / "local/tmp/k-build/ext"))
    out = root / f"k0-{tag}"
    out.mkdir(parents=True, exist_ok=True)
    return load(f"k0_{tag.replace('-', '_')}", [str(HERE / "csrc/k0.cpp"), str(HERE / "csrc/kernels.cu")],
                extra_cflags=["-O3", "-std=c++17"], extra_cuda_cflags=["-O3"], build_directory=str(out))


def env_facts() -> dict:
    p = torch.cuda.get_device_properties(0)
    return {"gpu": p.name, "sm": f"{p.major}.{p.minor}", "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "python": sys.version.split()[0], "commit": sh(f"git -C {HERE} rev-parse HEAD"),
            "dirty": bool(sh(f"git -C {HERE} status --porcelain -- .")),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}


# ---------------------------------------------------------------------------------------------- pools and lists


def pool(name: str, chunk: int, segments: int, cap: int, host: bool):
    """A core pool (`memory.Pool`) of `cap` units of `segments` x `chunk` bytes; host pools mapped (the SM path)."""
    from holon.core.memory import Pool

    p = Pool(name, unit_bytes=chunk * segments, segments=segments, min_units=0, pref_units=cap,
             tokens_per_unit=None, priority=0, device=torch.device("cuda"), host=host, mapped=host)
    p._set(cap)                                       # what Ledger.materialize does for a fixed pool
    return p


def device_bytes(p) -> torch.Tensor:
    """The pool's bytes as a device-addressable tensor (a mapped host pool's alias)."""
    return p.device_view(torch.uint8, (p.capacity * p.unit_bytes,)) if p.host else p.buffer


def picks(rng: np.random.Generator, cap: int, n: int, run: int) -> tuple[np.ndarray, np.ndarray]:
    """Unit ids in a source and a destination pool: n units at random distinct positions, in runs of `run` units
    consecutive in both pools (run 1: every unit its own run)."""
    def one():
        k = -(-n // run)
        starts = rng.choice(cap // run, k, replace=False) * run
        return (starts[:, None] + np.arange(run)[None, :]).ravel()[:n].astype(np.int64)
    return one(), one()


def addresses(p, units: np.ndarray, chunk: int, segments: int) -> np.ndarray:
    """[segments, n] chunk addresses of `units` in pool p: layer-major [segments, cap, chunk] when p has the segments,
    unit-major (one segment per unit) otherwise."""
    seg = np.arange(segments, dtype=np.int64)[:, None]
    if p.segments == segments:
        return p.base + (seg * p.capacity + units[None, :]) * chunk
    return p.base + units[None, :] * (segments * chunk) + seg * chunk


def chunk_lists(src, su, dst, du, chunk: int, segments: int):
    """(src, dst, bytes) int64 arrays of the copy, built vectorised. A run of units consecutive in both pools is one
    copy per segment when both pools are layer-major; a unit-major side has no such runs (its segments interleave)."""
    n = len(su)
    if src.segments == dst.segments == segments and n:
        brk = np.flatnonzero(np.r_[True, (np.diff(su) != 1) | (np.diff(du) != 1)])
        lens = np.diff(np.r_[brk, n])
        su, du = su[brk], du[brk]
    else:
        lens = np.ones(n, dtype=np.int64)
    s, d = addresses(src, su, chunk, segments), addresses(dst, du, chunk, segments)
    b = np.broadcast_to(lens[None, :] * chunk, s.shape)
    if src.segments != segments or dst.segments != segments:     # unit-major side: walk unit by unit
        s, d, b = s.T, d.T, b.T
    return np.ascontiguousarray(s).ravel(), np.ascontiguousarray(d).ravel(), np.ascontiguousarray(b).ravel()


def tiles(s: np.ndarray, d: np.ndarray, b: np.ndarray, tile: int = TILE) -> np.ndarray:
    """The SM copy's work items [n, 3] (src, dst, 16-byte words), chunks split into tiles of at most `tile` bytes."""
    k = -(-b // tile)
    idx = np.repeat(np.arange(len(b)), k)
    off = (np.arange(len(idx)) - np.repeat(np.cumsum(k) - k, k)) * tile
    size = np.minimum(b[idx] - off, tile)
    return np.stack([s[idx] + off, d[idx] + off, size // 16], axis=1)


class Staging:
    """A pinned host buffer the SM path's work items go through to a device table (as a copier would keep one)."""

    def __init__(self, rows: int):
        self.host = torch.empty((rows, 3), dtype=torch.int64, pin_memory=True)
        self.dev = torch.empty((rows, 3), dtype=torch.int64, device="cuda")

    def upload(self, t: np.ndarray) -> torch.Tensor:
        n = len(t)
        self.host[:n].numpy()[:] = t
        self.dev[:n].copy_(self.host[:n], non_blocking=True)
        return self.dev[:n]


# ---------------------------------------------------------------------------------------------- timing


def gpu_us(ext, fn, reps: int, gate_s: float) -> tuple[float, bool]:
    """Mean GPU time of fn() back to back behind a busy kernel (the host runs ahead); whether the host stayed ahead."""
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ext.busy(int(gate_s * 1e9))
    t = time.perf_counter()
    a.record()
    for _ in range(reps):
        fn()
    b.record()
    submit = time.perf_counter() - t
    b.synchronize()
    return a.elapsed_time(b) * 1000 / reps, submit < gate_s


def host_e2e(fn, reps: int = 3) -> tuple[float, float]:
    """Median host seconds of fn() (submission) and to its completion, from an idle GPU."""
    hs, es = [], []
    for _ in range(reps):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        hs.append(time.perf_counter() - t)
        torch.cuda.synchronize()
        es.append(time.perf_counter() - t)
    return statistics.median(hs), statistics.median(es)


def rate(ext, fn, nbytes: int, outer: int = 3) -> dict:
    """GB/s of fn() moving nbytes: GPU time behind a gate (median of `outer`), host time and end-to-end rate."""
    fn()
    torch.cuda.synchronize()
    host, e2e = host_e2e(fn)
    reps = max(2, min(30, int(0.12 / max(nbytes / 40e9, host))))
    gate = min(4.0, host * reps * 1.3 + 0.003)
    runs = [gpu_us(ext, fn, reps, gate) for _ in range(outer)]
    us = statistics.median(r[0] for r in runs)
    return {"gbps": nbytes / us / 1e3, "gpu_us": us, "host_ms": host * 1e3, "e2e_gbps": nbytes / e2e / 1e9,
            "host_ahead": all(r[1] for r in runs), "reps": reps}


def verify(src, su, dst, du, chunk: int, segments: int) -> bool:
    """Every destination chunk equals its source (compared on the device, segment by segment)."""
    def seg_view(p, s):
        t = device_bytes(p)
        if p.segments == segments:
            return t.view(segments, p.capacity, chunk)[s]
        return t.view(p.capacity, segments, chunk)[:, s]
    a_idx = torch.as_tensor(su, device="cuda")
    b_idx = torch.as_tensor(du, device="cuda")
    return all(torch.equal(seg_view(src, s).index_select(0, a_idx), seg_view(dst, s).index_select(0, b_idx))
               for s in range(segments))


# ---------------------------------------------------------------------------------------------- K-M2


def blob(ext, quick: bool) -> dict:
    """The reference blob rate: one big copy between S2 host memory and the device, H2D and D2H; torch's own pinned
    allocator (cudaHostAlloc) beside it; and the page-locking time of S2 host memory."""
    from holon.core import platform

    out = {"rows": []}
    dev = torch.device("cuda")
    t = time.perf_counter()
    probe = platform.host_memory(2 << 30, dev, False)
    out["pin_s_per_gib"] = (time.perf_counter() - t) / 2
    del probe
    n_max = 1 << 30
    host = platform.host_memory(n_max, dev, True)
    host.view(torch.int64)[: n_max // 8 : 4096].random_()
    d = torch.empty(n_max, dtype=torch.uint8, device=dev)
    pinned = torch.empty(256 << 20, dtype=torch.uint8, pin_memory=True)
    for mib in ((256,) if quick else (64, 256, 1024)):
        n = mib << 20
        for direction, (dst, src) in (("h2d", (d[:n], host[:n])), ("d2h", (host[:n], d[:n]))):
            r = rate(ext, lambda dst=dst, src=src: dst.copy_(src, non_blocking=True), n)
            out["rows"].append({"mib": mib, "direction": direction, "alloc": "s2_register", **r})
            log(f"blob {direction} {mib} MiB: {r['gbps']:.1f} GB/s")
    n = 256 << 20
    for direction, (dst, src) in (("h2d", (d[:n], pinned)), ("d2h", (pinned, d[:n]))):
        r = rate(ext, lambda dst=dst, src=src: dst.copy_(src, non_blocking=True), n)
        out["rows"].append({"mib": 256, "direction": direction, "alloc": "torch_pinned", **r})
    # the SM copy of one blob (tiled), a few configurations
    staging = Staging(n_max // TILE)
    for direction, (sp, dp) in (("h2d", (host.data_ptr(), d.data_ptr())), ("d2h", (d.data_ptr(), host.data_ptr()))):
        table = staging.upload(tiles(np.array([sp]), np.array([dp]), np.array([n_max])))
        best = max(((c, u, rate(ext, lambda c=c, u=u, t=table: ext.sm_copy(t, c, 256, u), n_max, 1)["gbps"])
                    for c in (8, 16, 32, 64) for u in (4, 8)), key=lambda x: x[2])
        out["rows"].append({"mib": 1024, "direction": direction, "alloc": "s2_register", "path": "sm",
                            "config": f"{best[0]}x256u{best[1]}", "gbps": best[2]})
        log(f"blob {direction} SM copy 1 GiB: {best[2]:.1f} GB/s ({best[0]}x256u{best[1]})")
    big = [r for r in out["rows"] if r["alloc"] == "s2_register" and "path" not in r]
    for direction in ("h2d", "d2h"):
        out[f"{direction}_gbps"] = max(r["gbps"] for r in big if r["direction"] == direction and r["mib"] >= 256)
    del host, d, pinned
    torch.cuda.empty_cache()
    return out


def m2(ext, quick: bool) -> dict:
    from holon.core.memory import copy_units

    out = {"blob": blob(ext, quick), "shapes": {}, "loads": LOADS, "tile_bytes": TILE}
    rng = np.random.default_rng(0)
    shapes = {k: SHAPES[k] for k in (("kv_27b_q8_0", "gdn_ssm_27b") if quick else SHAPES)}
    loads = {k: LOADS[k] for k in (("16K",) if quick else LOADS)}
    sweep = [(c, u) for c in ((8, 32) if quick else CTAS) for u in ((4,) if quick else UNROLLS)]
    for name, (chunk, segments) in shapes.items():
        unit = chunk * segments
        units = {k: max(1, round(v / unit)) for k, v in loads.items()}
        cap = -(-2 * max(units.values()) // 16) * 16
        dev = pool(f"{name}.dev", chunk, segments, cap, False)
        layer = pool(f"{name}.host_layer", chunk, segments, cap, True)
        unitm = pool(f"{name}.host_unit", chunk * segments, 1, cap, True)
        for p in (dev, layer, unitm):
            device_bytes(p).view(torch.int64).random_()
        staging = Staging(max(units.values()) * segments * max(1, chunk // TILE) + 16)
        rows = []
        log(f"M2 {name}: chunk {chunk} B x {segments} segments, unit {unit} B, pools of {cap} units")
        for layout, host in (("layer", layer), ("unit", unitm)):
            for direction in ("h2d", "d2h"):
                src, dst = (host, dev) if direction == "h2d" else (dev, host)
                best_sm = {}                      # load -> the random cell's best SM config, reused for runs
                for frag, run in FRAGS.items():
                    if quick and frag == "runs4":
                        continue
                    for load, n in units.items():
                        if run > n:
                            continue
                        su, du = picks(rng, cap, n, run)
                        nbytes = n * unit
                        row = {"layout": layout, "direction": direction, "frag": frag, "load": load, "units": n,
                               "bytes": nbytes}
                        t = time.perf_counter()
                        s, d, b = chunk_lists(src, su, dst, du, chunk, segments)
                        build = time.perf_counter() - t
                        row["chunks"] = len(b)
                        if layout == "layer":       # today's per-run loop (memory.copy_units) needs equal shapes
                            fn = lambda src=src, su=su, dst=dst, du=du: copy_units(src, su, dst, du)  # noqa: E731
                            row["loop"] = rate(ext, fn, nbytes)
                            row["loop"]["us_per_chunk"] = row["loop"]["host_ms"] * 1e3 / len(b)
                            dst_bytes = device_bytes(dst)
                            dst_bytes.zero_()
                            fn()
                            torch.cuda.synchronize()
                            row["loop"]["verified"] = verify(src, su, dst, du, chunk, segments)
                        st, dt, bt = (torch.from_numpy(x) for x in (s, d, b))
                        row["batch"] = rate(ext, lambda st=st, dt=dt, bt=bt: ext.memcpy_batch(st, dt, bt), nbytes)
                        row["batch"]["build_ms"] = build * 1e3
                        row["batch"]["us_per_chunk"] = (row["batch"]["host_ms"] + build * 1e3) * 1e3 / len(b)
                        device_bytes(dst).zero_()
                        ext.memcpy_batch(st, dt, bt)
                        torch.cuda.synchronize()
                        row["batch"]["verified"] = verify(src, su, dst, du, chunk, segments)
                        t = time.perf_counter()
                        items = tiles(s, d, b)
                        table = staging.upload(items)
                        sm_host = time.perf_counter() - t
                        if frag == "random":
                            cfgs = {f"{c}x256u{u}": rate(ext, lambda c=c, u=u, t=table: ext.sm_copy(t, c, 256, u),
                                                         nbytes, 1)["gbps"] for c, u in sweep}
                            top = max(cfgs, key=cfgs.get)
                            row["sm_sweep"] = cfgs
                            row["sm_saturating_ctas"] = min(int(k.split("x")[0]) for k, v in cfgs.items()
                                                            if v >= 0.95 * cfgs[top])
                            best_sm[load] = top
                        else:
                            top = best_sm[load]
                        c, u = int(top.split("x")[0]), int(top.split("u")[1])
                        row["sm"] = rate(ext, lambda c=c, u=u, t=table: ext.sm_copy(t, c, 256, u), nbytes)
                        row["sm"]["config"] = top
                        row["sm"]["build_upload_ms"] = sm_host * 1e3
                        row["sm"]["items"] = len(items)
                        device_bytes(dst).zero_()
                        ext.sm_copy(table, c, 256, u)
                        torch.cuda.synchronize()
                        row["sm"]["verified"] = verify(src, su, dst, du, chunk, segments)
                        rows.append(row)
                        log(f"M2 {name} {layout} {direction} {frag} {load} ({len(b)} chunks): "
                            + (f"loop {row['loop']['gbps']:.1f} GB/s ({row['loop']['host_ms']:.1f} ms host) "
                               if "loop" in row else "")
                            + f"batch {row['batch']['gbps']:.1f} ({row['batch']['host_ms']:.2f}+"
                              f"{build * 1e3:.2f} ms) sm {row['sm']['gbps']:.1f} ({top}); verified "
                            + str([row[k]["verified"] for k in ("loop", "batch", "sm") if k in row]))
        out["shapes"][name] = {"chunk": chunk, "segments": segments, "unit_bytes": unit, "capacity": cap, "rows": rows}
        del dev, layer, unitm, staging
        torch.cuda.empty_cache()
    return out


def m2_summary(m: dict) -> dict:
    """The decision cells: fragmented 34,816 B chunks against the blob rate, per path and direction."""
    out = {}
    for direction in ("h2d", "d2h"):
        blob_rate = m["blob"][f"{direction}_gbps"]
        cells = [r for r in m["shapes"]["kv_27b_q8_0"]["rows"]
                 if r["frag"] == "random" and r["direction"] == direction and r["load"] in ("16K", "32K")]
        for path in ("loop", "batch", "sm"):
            xs = [r[path]["gbps"] for r in cells if path in r]
            if xs:
                out[f"{direction}_{path}_frag34816_over_blob"] = [round(x / blob_rate, 3) for x in xs]
    best = max(min(out.get(f"h2d_{p}_frag34816_over_blob", [0])) for p in ("batch", "sm"))
    out["copy_rule_best_over_blob"] = best
    return out


# ---------------------------------------------------------------------------------------------- K-M3


def probe_latency(feeder, d2h: bool, sizes=(4096, 65536, 1 << 20), n: int = 30) -> dict:
    """Latency of a small copy on another stream (an engine step's inputs H2D, or its sampled tokens D2H) while the
    feeder's copies run: what a step that needs it waits. Copies up to 64 KiB H2D may go inline in the command stream
    instead of through a copy engine, so several sizes."""
    s = torch.cuda.Stream()
    if feeder is not None:
        feeder.start()
        time.sleep(0.05)
    out = {}
    for size in sizes:
        host = torch.zeros(size, dtype=torch.uint8, pin_memory=True)
        dev = torch.zeros(size, dtype=torch.uint8, device="cuda")
        xs = []
        for _ in range(n):
            t = time.perf_counter()
            with torch.cuda.stream(s):
                (host.copy_(dev, non_blocking=True) if d2h else dev.copy_(host, non_blocking=True))
            s.synchronize()
            xs.append((time.perf_counter() - t) * 1e3)
            time.sleep(0.003)
        r = summarize(xs)
        out[str(size)] = {"p50": r["p50"], "p90": r["p90"], "max": max(xs), "n": n}
    if feeder is not None:
        out["feeder"] = feeder.stop()
    return out


def m3(ext, args) -> dict:
    """The 27B decoding 4 sequences at its lock's settings while H2D loads or D2H stores run on other streams from
    or into S2 host memory: per-step wall time per arm, interleaved ABBA rounds."""
    from clock_check import STEADY, Sampler
    from holon.api import SamplingParams
    from holon.core.cli import engine_config, engine_flags
    from holon.core.engine import Engine
    from run import lock_flags, read_lock

    dev = torch.device("cuda")
    rng = np.random.default_rng(1)
    # copy resources first, so the engine sizes its KV around them
    ring = torch.empty(args.ring_mib << 20, dtype=torch.uint8, device=dev)
    conv_d = torch.empty(GDN_LAYERS, 1, CONV_SEG, dtype=torch.uint8, device=dev)
    ssm_d = torch.empty(GDN_LAYERS, 1, SSM_SEG, dtype=torch.uint8, device=dev)
    kv_units = 2048 + 300
    kv_h = pool("m3.kv_host_unit", KV_PAGE, 1, kv_units, True)
    conv_h = pool("m3.conv_host", CONV_SEG, GDN_LAYERS, 8, True)
    ssm_h = pool("m3.ssm_host", SSM_SEG, GDN_LAYERS, 8, True)
    for p in (kv_h, conv_h, ssm_h):
        device_bytes(p).view(torch.int64).random_()
    # the H2D load: a fragmented 32K resume = 2048 KV pages (16 x 34,816 B chunks each, unit-major host) + one
    # checkpoint (48 conv + 48 SSM segments), into a device ring (the bytes over PCIe are what matter)
    su = rng.choice(kv_units, 2048, replace=False)
    s_kv = (kv_h.base + su[:, None] * KV_PAGE + np.arange(16)[None, :] * 34_816).ravel()
    s_pl = np.r_[conv_h.base + np.arange(GDN_LAYERS) * 8 * CONV_SEG, ssm_h.base + np.arange(GDN_LAYERS) * 8 * SSM_SEG]
    sizes = np.r_[np.full(len(s_kv), 34_816), np.full(GDN_LAYERS, CONV_SEG), np.full(GDN_LAYERS, SSM_SEG)]
    offs = np.cumsum(np.r_[0, sizes[:-1]]) % (len(ring) - SSM_SEG) // 16 * 16
    load_s = np.r_[s_kv, s_pl].astype(np.int64)
    load_d = (ring.data_ptr() + offs).astype(np.int64)
    load_bytes = int(sizes.sum())
    load_t = [torch.from_numpy(x.astype(np.int64)) for x in (load_s, load_d, sizes)]
    load_table = torch.from_numpy(tiles(load_s, load_d, sizes.astype(np.int64))).cuda()
    # a checkpoint store: the payload's 96 segments, device -> a rotating host unit
    stores = []
    for k in range(8):
        sd = np.r_[conv_d.data_ptr() + np.arange(GDN_LAYERS) * CONV_SEG,
                   ssm_d.data_ptr() + np.arange(GDN_LAYERS) * SSM_SEG]
        dd = np.r_[conv_h.base + (np.arange(GDN_LAYERS) * 8 + k) * CONV_SEG,
                   ssm_h.base + (np.arange(GDN_LAYERS) * 8 + k) * SSM_SEG]
        bb = np.r_[np.full(GDN_LAYERS, CONV_SEG), np.full(GDN_LAYERS, SSM_SEG)]
        stores.append([torch.from_numpy(x.astype(np.int64)) for x in (sd, dd, bb)])
    store_stream = torch.cuda.Stream()
    sm_ctas, sm_unroll = args.m3_sm_config
    none = torch.empty(0, dtype=torch.int64)

    def feeder(kind, lists=load_t, nbytes=load_bytes, **kw):
        return ext.Feeder(kind, *(lists if kind != "sm" else (none, none, none)),
                          load_table if kind == "sm" else none, sm_ctas, sm_unroll, nbytes, **kw)
    pieces = [int(x) for x in args.m3_pieces.split(",")]
    feeders = {"h2d_dma": feeder("dma")}
    feeders.update({f"h2d_paced{p}x1": feeder("dma", per_call=p, depth=1) for p in pieces})
    feeders.update({"h2d_each": feeder("each"), "h2d_sm": feeder("sm"), "d2h_stress": feeder("dma", stores[0], PAYLOAD),
                    "d2h_stress_paced": feeder("dma", stores[0], PAYLOAD, per_call=8, depth=1)})
    extra = {"h2d_dma_noflag": feeder("dma", flags=0), f"h2d_paced{pieces[0]}x2": feeder("dma", per_call=pieces[0],
                                                                                          depth=2)}
    probe = {name: probe_latency(f, name.startswith("d2h"), (4096, 65536) if name.startswith("d2h") else
                                 (4096, 65536, 1 << 20)) for name, f in {**feeders, **extra}.items()}
    probe["idle"] = probe_latency(None, False)
    probe["idle_d2h"] = probe_latency(None, True, (4096, 65536))
    for name, r in probe.items():
        log(f"M3 probe {name}: " + ", ".join(f"{k} B p50 {v['p50']:.3f} p90 {v['p90']:.3f} ms" for k, v in r.items()
                                             if k != "feeder")
            + (f"; alone {r['feeder']['gbps']:.1f} GB/s" if "feeder" in r else ""))
    # the answer-rate store, paced (8 segments of the 96 at a time, one in flight) from its own thread
    once = [ext.Feeder("dma", *st, none, 0, 0, PAYLOAD, per_call=8, depth=1, loops=1) for st in stores]

    flags = argparse.ArgumentParser()
    engine_flags(flags)
    lock = read_lock(args.lock)
    cfg = engine_config(args.model, flags.parse_args(lock_flags(lock["engine"])))
    engine = Engine(cfg)
    pools = [{"name": p.name, "unit_bytes": p.unit_bytes, "segments": p.segments, "segment_bytes":
              p.unit_bytes // p.segments, "capacity": p.capacity, "host": p.host}
             for p in engine.ledger.pools.values() if p.capacity]
    ctx = args.m3_ctx
    prng = random.Random(0)
    rids = [engine.add([prng.randrange(1000, 30000) for _ in range(ctx)],
                       SamplingParams(temperature=0.0, max_tokens=1 << 20, ignore_eos=True)) for _ in range(4)]
    while engine.sched.waiting or any(s.num_computed < s.prompt_len for s in engine.sched.running):
        engine.step()
    for _ in range(32):                               # graphs captured, steady state
        engine.step()
    torch.cuda.synchronize()
    sampler = Sampler()
    arms = ["control", "h2d_dma", *(f"h2d_paced{p}x1" for p in pieces), "h2d_each", "h2d_sm", "d2h_answer",
            "d2h_answer_paced", "d2h_stress", "d2h_stress_paced"]
    every = 64 // 4                                   # 4 rows each crossing a 64-token boundary every 64 steps
    raw = {a: [] for a in arms}
    copies = {a: [] for a in arms}
    clocks = {a: [] for a in arms}
    order = arms + arms[::-1]
    n_store, started = 0, set()
    for rnd in range(args.m3_rounds):
        for arm in order:
            f = feeders.get(arm)
            if f is not None:
                f.start()
            ts, t, t0 = [], time.perf_counter(), time.time()
            for i in range(args.m3_warm + args.m3_steps):
                if arm.startswith("d2h_answer") and i % every == 0:
                    if arm == "d2h_answer":
                        ext.memcpy_batch(*stores[n_store % 8], stream=store_stream.cuda_stream)
                    else:
                        o = once[n_store % 8]
                        if o in started:
                            if o.busy():
                                raise RuntimeError("a paced store outlived 8 x 16 decode steps")
                            o.stop()
                        o.start()
                        started.add(o)
                    n_store += 1
                engine.step()
                now = time.perf_counter()
                if i >= args.m3_warm:
                    ts.append((now - t) * 1e3)
                t = now
            t1 = time.time()
            if f is not None:
                copies[arm].append(f.stop())
            store_stream.synchronize()
            for o in started:
                o.stop()
            started.clear()
            raw[arm].append(ts)
            clocks[arm].append(sampler.summary(t0, t1))
            log(f"M3 round {rnd} {arm}: step p50 {statistics.median(ts):.2f} ms p90 "
                f"{sorted(ts)[len(ts) * 9 // 10]:.2f}" + (f", copies {copies[arm][-1]['gbps']:.1f} GB/s"
                                                          if f is not None else ""))
    sampler.close()
    lens = [s.num_computed for s in engine.sched.running]
    for rid in rids:
        engine.abort(rid)
    engine.close()
    res = {"lock": lock["path"], "plan_id": lock["plan_id"], "model": Path(args.model).name, "ctx_start": ctx,
           "ctx_end": max(lens) if lens else None, "steps_per_segment": args.m3_steps, "warm_steps": args.m3_warm,
           "rounds": args.m3_rounds, "order": order, "store_every_steps": every, "load_bytes": load_bytes,
           "load_chunks": len(sizes), "store_bytes": PAYLOAD, "sm_config": f"{sm_ctas}x256u{sm_unroll}",
           "ring_mib": args.ring_mib, "piece_chunks": pieces, "piece_bytes": [p * 34_816 for p in pieces],
           "engine_pools": pools,
           "small_copy_probe_ms": probe, "arms": {}}
    base = [x for seg in raw["control"] for x in seg]
    for arm in arms:
        xs = [x for seg in raw[arm] for x in seg]
        s = summarize(xs)
        r = {"step_ms": s, "raw_ms": [[round(x, 3) for x in seg] for seg in raw[arm]], "clocks": clocks[arm]}
        if arm != "control":
            b = summarize(base)
            r["median_ratio"] = s["p50"] / b["p50"]
            r["p90_ratio"] = s["p90"] / b["p90"]
            r["mean_ratio"] = s["mean"] / b["mean"]
            # per round: this arm's segments against the control's in the same round (ABBA pairs)
            per = len(raw[arm]) // args.m3_rounds
            r["round_mean_ratios"] = [
                statistics.fmean(x for seg in raw[arm][i * per:(i + 1) * per] for x in seg)
                / statistics.fmean(x for seg in raw["control"][i * per:(i + 1) * per] for x in seg)
                for i in range(args.m3_rounds)]
        if copies[arm]:
            r["copy_gbps"] = statistics.median(c["gbps"] for c in copies[arm])
        res["arms"][arm] = r
    allc = [c for arm in arms for c in clocks[arm] if c.get("samples")]
    p10 = min(c["sm_mhz"]["p10"] for c in allc) if allc else None
    p50 = statistics.median(c["sm_mhz"]["p50"] for c in allc) if allc else None
    res["clock_steady"] = bool(allc) and p10 >= STEADY * p50
    return res


# ---------------------------------------------------------------------------------------------- K-M4


def aligned(n: int) -> dict:
    return {"bytes": n, "mod_512": n % 512, "mod_4096": n % 4096}


def region(nbytes: int) -> mmap.mmap:
    """Page-aligned anonymous memory (as an S2 host pool's map)."""
    m = mmap.mmap(-1, nbytes, flags=mmap.MAP_PRIVATE)
    m.write(np.random.default_rng(nbytes).bytes(nbytes))
    return m


def iovecs(m: mmap.mmap, chunk: int, segments: int, layer_major: bool, slot: int = 1, cap: int = 2):
    """One unit's iovec: its segments in an S2 pool laid out [segments, cap, chunk] (layer-major: segments apart) or
    the whole unit as one view."""
    mv = memoryview(m)
    if not layer_major:
        return [mv[slot * chunk * segments:(slot + 1) * chunk * segments]]
    return [mv[(s * cap + slot) * chunk:(s * cap + slot + 1) * chunk] for s in range(segments)]


class IO:
    """Python I/O threads as the SSD level will run them: os.preadv / os.pwritev of whole units at random slots."""

    def __init__(self, path: Path, write: bool, direct: bool, shape: tuple[int, int, bool], slot_bytes: int,
                 slots: int, threads: int, fadvise: bool = False, gate=None, seed: int = 0):
        self.path, self.write, self.direct, self.fadvise, self.gate = path, write, direct, fadvise, gate
        self.shape, self.slot_bytes, self.slots, self.threads, self.seed = shape, slot_bytes, slots, threads, seed
        self.bytes = 0
        self.lat: list[float] = []
        self.stop = threading.Event()
        self.lock = threading.Lock()

    def _run(self, w: int) -> None:
        chunk, segments, _ = self.shape
        m, iov = self.bufs[w]
        total = chunk * segments
        fd = os.open(self.path, (os.O_WRONLY if self.write else os.O_RDONLY) | (os.O_DIRECT if self.direct else 0))
        rng = random.Random(self.seed * 100 + w)
        try:
            while not self.stop.is_set():
                if self.gate is not None and not self.gate(self, total):
                    time.sleep(0.0001)
                    continue
                off = rng.randrange(self.slots) * self.slot_bytes
                t = time.perf_counter()
                n = os.pwritev(fd, iov, off) if self.write else os.preadv(fd, iov, off)
                if self.fadvise:
                    os.posix_fadvise(fd, off, total, os.POSIX_FADV_DONTNEED)
                dt = time.perf_counter() - t
                assert n == total, f"short I/O: {n} of {total}"
                with self.lock:
                    self.bytes += n
                    self.lat.append(dt)
            if self.write:
                os.fdatasync(fd)
        finally:
            os.close(fd)

    def start(self) -> None:
        chunk, segments, layer = self.shape          # buffers before the window opens
        self.bufs = []
        for _ in range(self.threads):
            m = region(chunk * segments * 2)
            self.bufs.append((m, iovecs(m, chunk, segments, layer)))
        self.t0 = time.perf_counter()
        self.ts = [threading.Thread(target=self._run, args=(w,)) for w in range(self.threads)]
        for t in self.ts:
            t.start()

    def finish(self) -> dict:
        self.stop.set()
        for t in self.ts:
            t.join()
        secs = time.perf_counter() - self.t0
        self.bufs = None
        lat = sorted(self.lat) or [0.0]
        return {"gbps": self.bytes / secs / GB, "bytes": self.bytes, "seconds": secs, "ops": len(self.lat),
                "lat_p50_ms": lat[len(lat) // 2] * 1e3, "lat_p99_ms": lat[int(len(lat) * 0.99)] * 1e3}


def io_window(seconds: float, *ios: IO) -> list[dict]:
    for io in ios:
        io.start()
    time.sleep(seconds)
    return [io.finish() for io in ios]


def fill(path: Path, nbytes: int) -> float:
    """Write nbytes of non-repeating-per-page data with O_DIRECT (the drive stores real data), fdatasync; GB/s."""
    chunk = 64 << 20
    m = region(chunk)
    words = np.frombuffer(m, dtype=np.uint64)
    fd = os.open(path, os.O_WRONLY | os.O_DIRECT)
    t = time.perf_counter()
    for k, off in enumerate(range(0, nbytes, chunk)):
        words[::512] += np.uint64(k + 1)                 # a fresh word per 4 KiB page per chunk
        os.pwritev(fd, [memoryview(m)[:min(chunk, nbytes - off)]], off)
    os.fdatasync(fd)
    secs = time.perf_counter() - t
    os.close(fd)
    del words
    m.close()
    return nbytes / secs / GB


def create(path: Path, nbytes: int) -> str:
    """A preallocated slot file: posix_fallocate, or a sparse file where the filesystem cannot preallocate."""
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        os.posix_fallocate(fd, 0, nbytes)
        how = "fallocate"
    except OSError:
        os.ftruncate(fd, nbytes)
        how = "ftruncate (no fallocate on this filesystem)"
    os.close(fd)
    return how


def direct_probe(path: Path) -> dict:
    """Which O_DIRECT iovec lengths and offsets the filesystem accepts: 512-aligned (34,816 B) and 4096-aligned."""
    out = {}
    m = region(1 << 20)
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    for label, length, off in (("len_4096k", 36_864, 0), ("len_512k", 34_816, 0), ("off_512k", 4096, 512)):
        try:
            os.preadv(fd, [memoryview(m)[:length]], off)
            out[label] = True
        except OSError as e:
            out[label] = f"refused ({e.strerror})"
    os.close(fd)
    m.close()
    return out


def cpu_body(n: int) -> int:
    """A fixed CPU-bound Python body (the engine loop's host work stand-in)."""
    acc = 0
    for i in range(n):
        acc = (acc * 31 + i) % 1_000_003
    return acc


def loop_times(ext, n: int, iters: int, period_s: float) -> list[float]:
    """Host ms of the CPU body per iteration; with period_s the loop then waits out the period in a GIL-free spin (a
    step's CUDA sync: the core stays busy, the GIL is released)."""
    out = []
    for _ in range(iters):
        t = time.perf_counter()
        cpu_body(n)
        dt = time.perf_counter() - t
        out.append(dt * 1e3)
        if period_s:
            ext.spin(max(0.0, period_s - dt))
    return out


def m4(ext, args) -> dict:
    parts = set(args.m4_parts.split(","))
    root = Path(args.nvme_dir)
    d = root / f"k0_nvme_{os.getpid()}"
    d.mkdir()
    out = {"storage": args.storage, "alignment": {}}
    unit = (34_816 * 16, 1, False)                   # a unit-major KV page, one iovec
    ssm = (SSM_SEG, GDN_LAYERS, True)                # a checkpoint's SSM unit as its 48 per-segment iovecs
    conv = (CONV_SEG, GDN_LAYERS, True)
    ssm_one = (SSM_SEG * GDN_LAYERS, 1, False)       # the same bytes as one iovec
    shapes = {"unit_557056": unit, "ssm_48x3MiB": ssm, "ssm_one_iovec": ssm_one, "conv_48x61440": conv}
    secs = 1.0 if args.quick else args.io_seconds
    read_path, write_path = d / "read_slots.bin", d / "write_slots.bin"
    try:
        out["alignment"] = {"sizes": {k: aligned(v) for k, v in {
            "kv_27b_q8_0_layer_chunk": 34_816, "kv_27b_unit_major_unit": KV_PAGE, "kv_8b_bf16_layer_chunk": 65_536,
            "gdn_conv_segment": CONV_SEG, "gdn_ssm_segment": SSM_SEG, "gdn_conv_unit": CONV_SEG * GDN_LAYERS,
            "gdn_ssm_unit": SSM_SEG * GDN_LAYERS}.items()}}
        st = os.statvfs(root)
        total, free = st.f_blocks * st.f_frsize, st.f_bavail * st.f_frsize
        size = min(args.read_gib << 30, free // 8) // (SSM_SEG * GDN_LAYERS) * (SSM_SEG * GDN_LAYERS)
        out["read_file_how"] = create(read_path, size)
        out["read_file_bytes"] = size
        out["fill_gbps"] = fill(read_path, size)
        log(f"M4 filled the read file ({size / GB:.1f} GB) at {out['fill_gbps']:.2f} GB/s")
        out["alignment"]["direct_probe"] = direct_probe(read_path)
        wsize = min(args.write_gib << 30, free // 8) // (SSM_SEG * GDN_LAYERS) * (SSM_SEG * GDN_LAYERS)
        out["write_file_how"] = create(write_path, wsize)

        def row(kind, mode, shape, threads, **kw):
            chunk, segments, _ = shapes[shape]
            slot = chunk * segments
            path = write_path if kind == "write" else read_path
            fsize = wsize if kind == "write" else size
            io = IO(path, kind == "write", mode == "direct", shapes[shape], slot, fsize // slot, threads,
                    fadvise=mode == "buffered_dontneed")
            r = io_window(secs, io)[0]
            out["rows"].append({"kind": kind, "mode": mode, "shape": shape, "threads": threads, **r, **kw})
            log(f"M4 {kind} {mode} {shape} x{threads}: {r['gbps']:.2f} GB/s (p50 {r['lat_p50_ms']:.2f} ms)")
            return r

        if "rw" in parts:
            out["rows"] = []
            # reads
            for mode in ("direct", "buffered_dontneed"):
                for threads in (1, 2, 3, 4):
                    row("read", mode, "unit_557056", threads)
                for shape in ("ssm_48x3MiB", "ssm_one_iovec", "conv_48x61440"):
                    for threads in (1, 2):
                        row("read", mode, shape, threads)
            # burst writes
            for threads in (1, 2, 3, 4):
                row("write", "direct", "unit_557056", threads)
            for shape in ("ssm_48x3MiB", "conv_48x61440"):
                row("write", "direct", shape, 2)
            for threads in (1, 2):
                row("write", "buffered_dontneed", "unit_557056", threads)
            # the C reference (blocking preadv / pwritev from C++ threads, no GIL)
            out["c_reference"] = []
            for kind, threads in (("read", 1), ("read", 2), ("write", 2)):
                path, fsize = (write_path, wsize) if kind == "write" else (read_path, size)
                r = ext.io_run(str(path), kind == "write", True, KV_PAGE, [KV_PAGE], fsize // KV_PAGE, threads, secs)
                out["c_reference"].append({"kind": kind, "threads": threads, **r})
                log(f"M4 C reference {kind} x{threads}: {r['gbps']:.2f} GB/s")
        # mixed reads and writes on one drive: r reader + r writer threads, each kind held at its byte share f (or
        # free-running), against the shared-lane model 1 / (f / read_r + (1 - f) / write_r) from pure rates taken
        # just before, at the same thread counts
        if "mixed" in parts:
            out["mixed"] = []
            pure = {}
            for r in (1, 2):
                for kind, path, fsize in (("read", read_path, size), ("write", write_path, wsize)):
                    io = IO(path, kind == "write", True, unit, KV_PAGE, fsize // KV_PAGE, r, seed=4)
                    pure[kind, r] = io_window(secs, io)[0]["gbps"]
            for f, r in ((0.5, 2), (0.75, 2), (None, 2), (0.5, 1)):
                def gate(io, n, f=f):
                    rb, wb = readers.bytes, writers.bytes      # noqa: B023 (this round's, set below)
                    tot = rb + wb + n
                    return (rb + n <= f * tot + n) if not io.write else (wb + n <= (1 - f) * tot + n)
                g = gate if f is not None else None
                readers = IO(read_path, False, True, unit, KV_PAGE, size // KV_PAGE, r, gate=g, seed=1)
                writers = IO(write_path, True, True, unit, KV_PAGE, wsize // KV_PAGE, r, gate=g, seed=2)
                rr, ww = io_window(max(secs, 3.0), readers, writers)
                measured = (rr["bytes"] + ww["bytes"]) / max(rr["seconds"], ww["seconds"]) / GB
                share = rr["bytes"] / (rr["bytes"] + ww["bytes"])
                model = 1 / (share / pure["read", r] + (1 - share) / pure["write", r])
                out["mixed"].append({"read_share_target": f, "threads_each": r, "read_share": share, "read": rr,
                                     "write": ww, "measured_gbps": measured, "model_gbps": model,
                                     "measured_over_model": measured / model,
                                     "model_inputs": {"read": pure["read", r], "write": pure["write", r]}})
                log(f"M4 mixed f={f} {r}+{r} threads: share {share:.2f}, measured {measured:.2f} GB/s (reads "
                    f"{rr['gbps']:.2f}, writes {ww['gbps']:.2f}) vs modelled {model:.2f} ({measured / model:.3f}x)")
        if "gil" in parts:
            # the engine loop beside the I/O threads (GIL): a fixed CPU-bound Python body; every row against the loop
            # alone just before it
            n = 2000
            while statistics.median(loop_times(ext, n, 5, 0)) < 3.0:
                n *= 2
            n = int(n * 3.0 / statistics.median(loop_times(ext, n, 9, 0)))
            out["gil"] = {"body_iters": n, "rows": []}
            for mode, period in (("busy", 0.0), ("step_15ms", 0.015)):
                iters = (150 if mode == "busy" else 100) // (3 if args.quick else 1)
                for threads, kind in ((1, "read"), (2, "read"), (3, "read"), (4, "read"), (2, "write"),
                                      (2, "read (C threads)")):
                    alone = summarize(loop_times(ext, n, iters, period))
                    path, fsize = (write_path, wsize) if kind == "write" else (read_path, size)
                    if kind.startswith("read (C"):
                        box = {}
                        window = 0.2 + iters * max(period, 0.003) * 1.5
                        io = threading.Thread(target=lambda w=window: box.update(ext.io_run(  # noqa: B023 (joined)
                            str(read_path), False, True, KV_PAGE, [KV_PAGE], size // KV_PAGE, 2, w)))
                        io.start()
                    else:
                        io = IO(path, kind == "write", True, unit, KV_PAGE, fsize // KV_PAGE, threads, seed=3)
                        io.start()
                    time.sleep(0.2)
                    body = summarize(loop_times(ext, n, iters, period))
                    if kind.startswith("read (C"):
                        io.join()
                        gbps = box.get("gbps")
                    else:
                        gbps = io.finish()["gbps"]
                    out["gil"]["rows"].append({"mode": mode, "io_threads": threads, "io_kind": kind, "alone_ms": alone,
                                               "body_ms": body, "slowdown_p50": body["p50"] / alone["p50"] - 1,
                                               "slowdown_mean": body["mean"] / alone["mean"] - 1,
                                               "io_gbps_during": gbps})
                    log(f"M4 GIL {mode} beside {threads} {kind} threads: body p50 {body['p50']:.2f} ms vs "
                        f"{alone['p50']:.2f} alone ({body['p50'] / alone['p50'] - 1:+.1%}), I/O {gbps:.2f} GB/s")
        write_path.unlink()
        if "sustained" in parts:
            # sustained writes past the SLC cache: 2 s windows over one pass of a big fresh slot file
            st = os.statvfs(root)
            total, free = st.f_blocks * st.f_frsize, st.f_bavail * st.f_frsize
            room = min(free - int(0.2 * total) - (1 << 30), args.sustain_gib << 30)
            if room < 20 << 30 or args.quick:
                out["sustained"] = {"skipped": "quick run" if args.quick else
                                    "less than 20 GiB writable while leaving 20% of the filesystem free"}
            else:
                burst = [r for r in out.get("rows", []) if r["kind"] == "write" and r["mode"] == "direct"
                         and r["shape"] == "unit_557056"]
                best = max(burst, key=lambda r: r["gbps"])["threads"] if burst else 2
                spath = d / "sustained.bin"
                slots = room // KV_PAGE
                how = create(spath, slots * KV_PAGE)
                order = np.random.default_rng(5).permutation(slots)
                nxt = [0]
                lk = threading.Lock()
                m_bytes = [0]

                def take():
                    with lk:
                        i = nxt[0]
                        nxt[0] += 1
                    return int(order[i]) if i < len(order) else None

                def writer(w):
                    m = region(KV_PAGE)
                    fd = os.open(spath, os.O_WRONLY | os.O_DIRECT)
                    try:
                        while (s := take()) is not None and time.perf_counter() < deadline:
                            os.pwritev(fd, [memoryview(m)], s * KV_PAGE)
                            with lk:
                                m_bytes[0] += KV_PAGE
                    finally:
                        os.close(fd)
                        m.close()

                deadline = time.perf_counter() + args.sustain_seconds
                ts = [threading.Thread(target=writer, args=(w,)) for w in range(best)]
                t0 = time.perf_counter()
                for t in ts:
                    t.start()
                windows, last = [], 0
                while any(t.is_alive() for t in ts):
                    time.sleep(2.0)
                    b = m_bytes[0]
                    windows.append({"t_s": round(time.perf_counter() - t0, 1), "gbps": (b - last) / 2.0 / GB,
                                    "written_gb": b / GB})
                    last = b
                for t in ts:
                    t.join()
                secs_all = time.perf_counter() - t0
                gb = [w["gbps"] for w in windows]
                peak = statistics.median(gb[: max(3, len(gb) // 10)])
                cliff = next((w for w in windows if w["gbps"] < 0.6 * peak), None)
                tail = gb[len(gb) // 2:] if cliff is None else [w["gbps"] for w in windows if w["t_s"] > cliff["t_s"]]
                out["sustained"] = {"file_how": how, "threads": best, "written_gb": m_bytes[0] / GB,
                                    "seconds": secs_all, "mean_gbps": m_bytes[0] / secs_all / GB, "start_gbps": peak,
                                    "cliff_at_gb": cliff["written_gb"] if cliff else None,
                                    "after_cliff_gbps": statistics.median(tail) if tail else None,
                                    "left_20pct_free": True, "windows": windows}
                log(f"M4 sustained: {m_bytes[0] / GB:.0f} GB in {secs_all:.0f}s; start {peak:.2f} GB/s, cliff at "
                    f"{cliff['written_gb'] if cliff else None} GB, after {out['sustained']['after_cliff_gbps']}")
                spath.unlink()
    finally:
        for p in d.glob("*"):
            p.unlink()
        d.rmdir()
        out["test_file_removed"] = not d.exists()
        log(f"M4 test files removed: {out['test_file_removed']}")
    return out


def m4_verdicts(m: dict, tag: str) -> dict:
    rows = {(r["kind"], r["mode"], r["shape"], r["threads"]): r["gbps"] for r in m["rows"]}
    bar = tag in BAR
    r2 = rows.get(("read", "direct", "unit_557056", 2))
    mixed = [x["measured_over_model"] for x in m.get("mixed", [])           # the rule's cells: 2 + 2, f held
             if x.get("threads_each", 2) == 2 and x["read_share_target"] is not None]
    seg = rows.get(("read", "direct", "ssm_48x3MiB", 2))
    one = rows.get(("read", "direct", "ssm_one_iovec", 2))
    out = {"read_2t_gbps": r2, "read_2t_rule": ("STOP" if r2 < 1.75 else "pass") if bar else "n/a (dev reference)",
           "mixed_over_model": mixed,
           "mixed_rule": ("STOP" if any(abs(x - 1) > 0.15 for x in mixed) else "pass") if bar else
           "n/a (dev reference)",
           "ssm_segments_over_one_iovec_2t": seg / one if seg and one else None,
           "ssm_segments_over_unit_2t": seg / r2 if seg and r2 else None}
    return out


# ---------------------------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="GPU class, e.g. rtx-5080")
    ap.add_argument("--only", default="m2,m3,m4")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--merge", action="store_true", help="update the records instead of replacing them")
    ap.add_argument("--out-dir", default=str(REPO / "evidence"))
    ap.add_argument("--date", default=time.strftime("%Y-%m-%d"))
    ap.add_argument("--model")
    ap.add_argument("--lock")
    ap.add_argument("--m3-ctx", type=int, default=4096)
    ap.add_argument("--m3-steps", type=int, default=64)
    ap.add_argument("--m3-warm", type=int, default=8)
    ap.add_argument("--m3-rounds", type=int, default=4)
    ap.add_argument("--m3-sm-config", default="8,4", help="CTAs,unroll of the SM arm (K-M2's pick)")
    ap.add_argument("--ring-mib", type=int, default=64)
    ap.add_argument("--m3-pieces", default="512,2048", help="chunks per piece of the paced H2D arms (18, 71 MB)")
    ap.add_argument("--clock-screen", help="bench/clock_check.py's JSON from this desktop, merged into the record")
    ap.add_argument("--nvme-dir")
    ap.add_argument("--storage", default="local NVMe", help="a generic label of the storage (no mount names)")
    ap.add_argument("--read-gib", type=int, default=16)
    ap.add_argument("--write-gib", type=int, default=8)
    ap.add_argument("--io-seconds", type=float, default=2.0)
    ap.add_argument("--sustain-gib", type=int, default=400)
    ap.add_argument("--sustain-seconds", type=float, default=300)
    ap.add_argument("--m4-parts", default="rw,mixed,gil,sustained", help="K-M4's parts; with --merge, the others "
                    "are kept from the record")
    args = ap.parse_args()
    args.m3_sm_config = tuple(int(x) for x in args.m3_sm_config.split(","))
    only = set(args.only.split(","))
    torch.cuda.init()
    ext = build_ext(args.tag)
    torch.set_num_threads(4)
    run = {"command": public_command(sys.argv), "date": args.date, "env": env_facts()}

    def record(kind: str):
        path = Path(args.out_dir) / f"k0_{kind}_{args.tag}_{args.date}.json"
        res = json.loads(path.read_text()) if args.merge and path.exists() else {}
        res.setdefault("runs", []).append(run)
        res.update({"tag": args.tag, "env": run["env"], "predictions": PREDICTIONS, "decision_rules": DECISION_RULES})
        return path, res

    def save(path, res):
        path.write_text(json.dumps(res, indent=1, default=float) + "\n")

    if only & {"m2", "m3"}:
        path, res = record("m2_copy")
        if "m2" in only:
            torch.cuda.set_stream(torch.cuda.Stream())   # cudaMemcpyBatchAsync refuses the legacy default stream
            res["m2"] = m2(ext, args.quick)
            res["m2_summary"] = m2_summary(res["m2"])
            torch.cuda.set_stream(torch.cuda.default_stream())
            save(path, res)
        if "m3" in only:
            if args.clock_screen:
                res["clock_screen"] = json.loads(Path(args.clock_screen).read_text())
            res["m3"] = m3(ext, args)
            save(path, res)
        v = res.setdefault("verdicts", {})
        if "m2_summary" in res:
            best = res["m2_summary"]["copy_rule_best_over_blob"]
            v["copy"] = {"best_h2d_frag34816_over_blob": best, "rule": (
                "STOP" if best < 0.8 else "pass") if args.tag in BAR else "n/a (dev reference)"}
        if "m3" in res:
            cost = res["m3"]["arms"]["d2h_answer"]["mean_ratio"] - 1
            v["stores"] = {"answer_rate_cost": cost, "rule": ("STOP" if cost > 0.02 else "pass")
                           if args.tag in BAR else "n/a (dev reference)"}
        save(path, res)
        log("wrote", path)
    if "m4" in only:
        path, res = record("m4_nvme")
        res["m4"] = {**(res.get("m4") or {}), **m4(ext, args)}
        res["verdicts"] = m4_verdicts(res["m4"], args.tag)
        save(path, res)
        log("wrote", path)


if __name__ == "__main__":
    main()
