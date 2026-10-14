"""GPU clock screen (release design §13.5, lead decision 1): sustained decode on a model while nvidia-smi samples the
SM clock and the throttle reasons every 100 ms; a desktop's evidence is recorded only if its clocks hold steady. The
record keeps the GPU's name, its SM clock percentiles and the share of loaded samples under a thermal or hardware
slowdown (and, apart, under the power cap) — nothing about the host.

`Sampler` is also used by `step_profile.py` to read clocks over its graphed and eager windows (D1).

    python bench/clock_check.py --model <file.gguf | dir> [--lock <lock>] [--seconds 60] [--batch 4] [--json out]
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import os
import random
import shlex
import statistics
import subprocess
import threading
import time

import torch
from holon.api import SamplingParams
from holon.core.cli import engine_config, engine_flags
from holon.core.engine import Engine
from run import lock_flags, read_lock

SLOWDOWN = 0x08 | 0x20 | 0x40 | 0x80      # HW slowdown, SW thermal, HW thermal, HW power brake
POWER_CAP = 0x04                           # SW power cap: a card at its power limit
STEADY = 0.9                               # p10 / p50 of the SM clock under load at least this


class Sampler:
    """nvidia-smi sampling one GPU every `ms` ms in the background: (time.time(), SM MHz, memory MHz, W, reasons)."""

    def __init__(self, index: str | None = None, ms: int = 100) -> None:
        index = index or os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        self.samples: list[tuple[float, float, float, float, int]] = []
        self.proc = subprocess.Popen(
            ["nvidia-smi", "-i", index, "--query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,"
             "clocks_event_reasons.active", "--format=csv,noheader,nounits", f"--loop-ms={ms}"],
            stdout=subprocess.PIPE, text=True)
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self) -> None:
        for line in self.proc.stdout:
            try:
                ts, sm, mem, power, reasons = (x.strip() for x in line.split(","))
                t = datetime.datetime.strptime(ts, "%Y/%m/%d %H:%M:%S.%f").timestamp()
                self.samples.append((t, float(sm), float(mem), float(power), int(reasons, 16)))
            except ValueError:
                continue

    def close(self) -> None:
        self.proc.terminate()
        self.proc.wait()

    def summary(self, start: float, end: float) -> dict:
        """The window's samples (time.time() bounds): SM clock p10 / p50 / p90, memory clock, mean power, and the
        shares of samples in a slowdown and at the power cap."""
        xs = [s for s in self.samples if start <= s[0] <= end]
        if not xs:
            return {"samples": 0}
        sm = sorted(s[1] for s in xs)
        q = lambda p: sm[min(len(sm) - 1, int(p * len(sm)))]       # noqa: E731
        return {"samples": len(xs), "sm_mhz": {"p10": q(0.1), "p50": q(0.5), "p90": q(0.9)},
                "mem_mhz": statistics.median(s[2] for s in xs), "power_w": round(statistics.mean(s[3] for s in xs), 1),
                "slowdown_share": round(sum(bool(s[4] & SLOWDOWN) for s in xs) / len(xs), 3),
                "power_cap_share": round(sum(bool(s[4] & POWER_CAP) for s in xs) / len(xs), 3)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--lock")
    ap.add_argument("--engine-args", default="")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--json")
    args = ap.parse_args()
    flags = argparse.ArgumentParser()
    engine_flags(flags)
    given = lock_flags(read_lock(args.lock)["engine"], (), args.engine_args) if args.lock else shlex.split(
        args.engine_args)
    cfg = engine_config(args.model, flags.parse_args(given))
    engine = Engine(dataclasses.replace(cfg, max_seqs=max(cfg.max_seqs, args.batch), max_model_len=8192,
                                        prefix_cache=False))
    rng = random.Random(0)
    prompts = lambda: [[rng.randrange(1000, 30000) for _ in range(256)] for _ in range(args.batch)]   # noqa: E731
    engine.generate(prompts(), SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True))   # JIT builds, idle
    sampler = Sampler()
    tokens = 0
    start = time.time()
    while time.time() - start < args.seconds:
        out = engine.generate(prompts(), SamplingParams(temperature=0.0, max_tokens=1024, ignore_eos=True))
        tokens += sum(map(len, out))
    torch.cuda.synchronize()
    end = time.time()
    time.sleep(0.3)
    sampler.close()
    loaded = sampler.summary(start + 2.0, end)                # past the first prefill's ramp
    result = {"gpu": torch.cuda.get_device_name(), "seconds": round(end - start, 1), "batch": args.batch,
              "tok_s": round(tokens / (end - start), 1), **loaded,
              "steady": loaded.get("samples", 0) > 0 and loaded["sm_mhz"]["p10"] >= STEADY * loaded["sm_mhz"]["p50"]}
    print(json.dumps(result), flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
