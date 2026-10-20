"""Milestone K step 3 (docs/v1-kv-tiers.md rev 8.4, D4 / D11): the mixed factor under std.state_tier's own dispatch,
and the engine loop's host time beside its I/O threads (GIL).

The dispatch is the tier's (`holon_std.state_tier.iothreads.IoThreads` over `holon_std.state_tier.ssd.SsdLevel`, 557,056
B unit-major units, O_DIRECT): while a read is queued, no write starts except one a queued read depends on. The driver
decides jobs as the tier would:
- reads (`read=True`) of copies the level holds, kept `threads` deep while the read share of bytes is below its target;
- a share of those reads first needs a dependent write (an urgent write, `IoThreads.urgent`): the host units it fills;
- background writes (`read=False`), kept `threads` deep: they run only while no read is queued.

Per cell: bytes per kind over the window; the model is D4's serial lane 1 / (f / read + (1 - f) / write) at the share
actually reached, from pure rates at the same thread count measured just before; factor = measured / model. D4 charges
reads at the pure read rate and dependent writes at the write rate divided by the factor.

A comparison arm (`strict`, not the tier's) starts no background write while a read is queued or running.

GIL: a CPU body of ≈ 3 ms (the engine loop's host time per step) timed alone and beside the I/O threads running
reads and writes.

    python spikes/k3_tiers/bench.py --tag rtx-5080 --dir <node-local dir> [--seconds 6]
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

from holon_std.state_tier.iothreads import IoThreads, Job
from holon_std.state_tier.ssd import SsdLevel

UNIT = 557_056                                     # a 27B q8_0 page, unit-major (D10): 136 × 4 KiB
GB = 1e9

PREDICTIONS = {
    "written": "2026-10-19, before any run of this script",
    "mixed_factor": "0.85-0.95 at 50/50 and 75/25 with 2 and 3 threads: no write starts while a read is queued, so "
                    "the drive sees mostly one kind at a time; the rest is its own read/write turnaround (step 2 "
                    "measured 0.77-0.80 with one writer always in flight)",
    "reads_beside_writes": "the read rate within 10% of the pure read rate at the same thread count (reads never wait "
                           "behind background writes, only behind the one write in flight when they arrive)",
    "strict_arm": "added after the tier arm's runs, no prediction",
    "gil": "the 3 ms body slows < 3% beside 2 and 3 I/O threads (preadv / pwritev release the GIL; the dispatch is a "
           "deque and a condition)",
}


def log(*a):
    print(*a, file=sys.stderr, flush=True)


class StrictIo(IoThreads):
    """The comparison arm (not the tier's): no background write starts while a read is queued or running."""

    def __init__(self, n: int, name: str) -> None:
        self.running = 0
        super().__init__(n, name)

    def _take(self) -> Job | None:
        with self.cv:
            while not (self.reads or (self.writes and not self.running) or self.closed):
                self.cv.wait()
            if self.reads:
                job = self.reads.popleft()
                if job.read:
                    self.running += 1
                    fn = job.fn

                    def done(fn=fn):
                        try:
                            fn()
                        finally:
                            with self.cv:
                                self.running -= 1
                                self.cv.notify_all()
                    job.fn = done
                return job
            return self.writes.popleft() if self.writes else None


class Bench:
    def __init__(self, root: Path, threads: int, gib: int, strict: bool = False) -> None:
        self.level = SsdLevel({"dir": str(root), "bytes": gib << 30, "persist": "off"}, threads)
        self.slots = self.level.bind(None, {"kv": (UNIT, 1, 1.0)}, None)["kv"]
        self.io, self.threads = (StrictIo if strict else IoThreads)(threads, "bench.io"), threads
        self.bufs = [mmap.mmap(-1, UNIT) for _ in range(4 * threads + 8)]   # page-aligned host units
        self.key, self.held, self.rng = 0, [], random.Random(0)

    def unit(self, i: int) -> dict:
        return {"kv": [[memoryview(self.bufs[i % len(self.bufs)])]]}

    def fill(self, n: int) -> float:
        start = time.perf_counter()
        for _ in range(n):
            self._write(self.unit(self.key)).ticket.wait()
        return n * UNIT / (time.perf_counter() - start) / GB

    def _write(self, src, urgent: bool = False):
        key, self.key = self.key, self.key + 1
        while len(self.held) >= self.slots - 4 * self.threads - 8:         # room: drop the oldest copies
            old = self.held.pop(0)
            self.level.drop(old, None)
        self.held.append(key)
        return self.io.submit(lambda: self.level.write(key, None, src), read=urgent)

    def _read(self, i: int):
        key = self.rng.choice(self.held[len(self.held) // 2: max(len(self.held) // 2 + 1, len(self.held) - 8)])
        return self.io.submit(lambda: self.level.read(key, None, self.unit(i)), read=True)

    def window(self, seconds: float, share: float | None, dependent: float) -> dict:
        """Keep reads (while their byte share is below `share`; None: no reads) and background writes (share < 1)
        `threads` deep for `seconds`; a `dependent` share of reads first takes an urgent write."""
        done = {"read": 0, "write": 0, "urgent": 0}
        live: list = []
        start, i = time.perf_counter(), 0
        while time.perf_counter() - start < seconds:
            still = []
            for k, j in live:
                if j.ticket.done():
                    done[k] += UNIT
                else:
                    still.append((k, j))
            live = still
            reads = sum(k == "read" for k, _ in live)
            writes = sum(k == "write" for k, _ in live)
            total = sum(done.values()) or 1
            read_share = done["read"] / total
            if share is not None and reads < self.threads and (share >= 1 or read_share <= share):
                if self.rng.random() < dependent:
                    live.append(("urgent", self._write(self.unit(i), urgent=True)))
                    i += 1
                live.append(("read", self._read(i)))
                i += 1
            elif (share is None or share < 1) and writes < self.threads:
                live.append(("write", self._write(self.unit(i))))
                i += 1
            else:
                time.sleep(0.0002)
        self.io.drain()
        secs = time.perf_counter() - start
        for k, _ in live:
            done[k] += UNIT
        return {k: v / secs / GB for k, v in done.items()} | {"seconds": secs,
                                                               "bytes": {k: v for k, v in done.items()}}

    def close(self) -> None:
        self.io.close()
        self.level.close([])


def body(n: int) -> int:
    s = 0
    for i in range(n):
        s += i * i % 7
    return s


def gil(bench: Bench, iters: int) -> dict:
    n = 1
    while True:                                        # ≈ 3 ms of pure-Python host work
        t = time.perf_counter()
        body(n)
        if time.perf_counter() - t > 3e-3:
            break
        n *= 2

    def timed() -> list[float]:
        out = []
        for _ in range(iters):
            t = time.perf_counter()
            body(n)
            out.append((time.perf_counter() - t) * 1e3)
            time.sleep(0.002)
        return out
    alone = timed()
    stop, busy = threading.Event(), {"reads": 0}

    def load():
        i = 0
        while not stop.is_set():
            jobs = [bench._read(i + k) for k in range(bench.threads)] + [bench._write(bench.unit(i + 7))]
            for j in jobs:
                j.ticket.wait()
            busy["reads"] += bench.threads
            i += bench.threads + 1
    t = threading.Thread(target=load)
    t.start()
    beside = timed()
    stop.set()
    t.join()
    return {"body_iters": n, "alone_ms_p50": statistics.median(alone), "beside_ms_p50": statistics.median(beside),
            "slowdown_p50": statistics.median(beside) / statistics.median(alone) - 1, "reads_beside": busy["reads"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dir", required=True, help="a node-local directory on the drive under test")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--gib", type=int, default=16)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = Path(a.dir) / f"k3_level_{os.getpid()}"
    out = {"tag": a.tag, "predictions": PREDICTIONS, "unit_bytes": UNIT, "rows": [], "gil": {}}
    out["commit"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    out["dirty"] = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    try:
        for threads, strict in ((2, False), (3, False), (2, True), (3, True)):
            bench = Bench(root / f"{threads}{'s' if strict else ''}", threads, a.gib, strict)
            out.setdefault("o_direct", bench.level.dfd["kv"] is not None)
            out.setdefault("fill_gbps", {})[threads] = bench.fill(min(bench.slots // 2, 4000))
            pure_r = bench.window(a.seconds, 1.0, 0.0)["read"]
            pure_w = bench.window(a.seconds, None, 0.0)["write"]
            log(f"{threads} threads{' (strict)' if strict else ''}: pure read {pure_r:.2f} GB/s, pure write "
                f"{pure_w:.2f} GB/s")
            for share in (0.5, 0.75):
                for dependent in (0.0, 0.5):
                    got = bench.window(max(a.seconds, 4.0), share, dependent)
                    b = got["bytes"]
                    total = sum(b.values())
                    f = b["read"] / total
                    model = 1 / (f / pure_r + (1 - f) / pure_w)
                    measured = total / got["seconds"] / GB
                    row = {"threads": threads, "arm": "strict" if strict else "tier", "read_share_target": share,
                           "dependent_share": dependent,
                           "read_share": f, "measured_gbps": measured, "model_gbps": model,
                           "mixed_factor": measured / model, "pure": {"read": pure_r, "write": pure_w}, "window": got,
                           "reads_vs_pure": got["read"] / (pure_r * f) if f else None}
                    out["rows"].append(row)
                    log(f"  share {share} dependent {dependent}: reached {f:.2f}, {measured:.2f} GB/s vs model "
                        f"{model:.2f}: factor {measured / model:.3f}")
            if not strict:
                out["gil"][threads] = gil(bench, 150)
                log(f"  gil: {out['gil'][threads]}")
            bench.close()
    finally:
        for p in sorted(root.rglob("*"), reverse=True):
            p.unlink() if p.is_file() else p.rmdir()
        if root.exists():
            root.rmdir()
        out["test_files_removed"] = not root.exists()
    text = json.dumps(out, indent=1, default=str)
    if a.out:
        Path(a.out).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
