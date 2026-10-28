"""Milestone K's resume bars (docs/dev/v1-kv-tiers.md §1 acceptance 3, §8): 3a, 3b and 3b-SSD, in process.

One engine at a lock's settings with std.state_tier and its SSD level. A parked conversation is a prompt of L tokens
run once (its blocks and its `prompt_end` checkpoint published); it is then left resident (VRAM), demoted to the host
level (RAM: the index evicts every unheld block, a pure free since the host level holds them), or demoted and written
down to the level (SSD: the host level writes every evictable copy back, and the bench waits for the writes). A resume
is the conversation plus a 256-token append; its TTFT is the time from `add` to its first token at the engine API.
Each conversation is parked once and resumed once (C18), the arms interleaved per repetition (A B C, then C B A). A
scattering prelude (short requests, half of them aborted) leaves the device units fragmented; every load's runs of
consecutive units are counted.

- `--cell 3a`: lengths 4K / 16K / 32K, `--reps` resumes per arm and length, idle or `--beside` three decoding
  sessions. Bars: RAM TTFT ≤ VRAM TTFT + 1.5 × KV bytes / 47.5 GB/s + one decode step (the checkpoint's host adopt
  is in both arms); SSD TTFT ≤ VRAM TTFT + 1.5 × (bytes / SSD rate + bytes / 47.5 GB/s) + one decode step, the bytes
  the KV and the checkpoint read from the level (§8: 1.29 GB at 32K).
- `--cell 3f`: the restart (`ssd.persist = close`). Engine A parks `--reps` 16K conversations (the rest demoted
  and written down to the level) and resumes the last; engine B, with the same parking history, closes; engine B2
  starts on B's directory (the restore's crc verification inside `attach`: startup time per persisted GB) and resumes
  it. Bar: B2's TTFT within 3a's SSD bar (`--vram-ms`: 3a's 16K VRAM median); its outputs and `cached_tokens` A's.
- `--cell 3b`: three sessions decoding while a fourth resumes 32K from RAM (`--ssd`: from SSD, 3b-SSD), over the
  conversations × `--rounds`. The decoders' step gaps are taken over the gated steps (from the resume's admission to
  the step before its append), against the same number of plain decode steps just before it (the control: nothing
  loading); the append step's gap apart. Bar: the gated-step median ≤ 1.1× the control's, p90 ≤ 1.2×. 3b-SSD reports,
  and the engine loop's host time per step during the read.

    python bench/tier_resume.py --cell 3a --model <file.gguf> --lock <lock> --ssd-dir <node-local dir> \\
        [--lengths 4096 16384 32768] [--reps 10] [--beside] [--step-ms 22.3] --json <file>
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import pathlib
import random
import statistics
import sys
import time

import torch
from clock_check import Sampler
from holon.api import SamplingParams
from holon.core.cli import engine_config, engine_flags
from holon.core.engine import Engine
from run import lock_flags, read_lock

LINK = 47.5e9                     # X's M1 blob rate (the bars' H2D link rate, C6)
SSD_RATE = 2.68e9                 # K-M4's large-block O_DIRECT read rate on the RTX 5080's drive (C6)
APPEND = 256
COUNTS = ("gated_steps", "gated_loads", "load_enqueue_wait_s", "level_reads", "level_read_bytes", "write_backs",
          "write_back_bytes")                    # the tier's counts per resume (3a)
GREEDY = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)


def log(*a) -> None:
    print(*a, file=sys.stderr, flush=True)


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


class Bench:
    def __init__(self, args, ssd_dir: str | None = None, persist: str = "off") -> None:
        lock = read_lock(args.lock)
        flags = argparse.ArgumentParser()
        engine_flags(flags)
        cfg = engine_config(args.model, flags.parse_args(lock_flags(lock["engine"])))
        opts = dict(cfg.plugin_options) | {
            "state_tier.host.bytes": args.host_bytes, "state_tier.host.pages": str(args.host_pages),
            "recurrent.checkpoints": str(args.staging), "recurrent.policy": "prompt_end,branch,answer:1",
            "state_tier.levels": "ssd", "state_tier.ssd.dir": ssd_dir or args.ssd_dir,
            "state_tier.ssd.bytes": args.ssd_bytes, "state_tier.ssd.persist": persist} | dict(
            o.split("=", 1) for o in args.options)
        self.cfg = dataclasses.replace(cfg, slots=dict(cfg.slots) | {"tier": "std.state_tier"}, plugin_options=opts,
                                       host_memory=args.host_memory)
        self.engine = Engine(self.cfg)
        self.tier, self.index = self.engine.tier, self.engine.state.index
        self.rng, self.lock = random.Random(args.seed), lock
        self.loads: list[dict] = []
        load = self.tier.load

        def counted(keys, units, payloads, after):          # runs of consecutive device units per load
            runs = sum(1 + sum(1 for a, b in zip(us, us[1:], strict=False) if b != a + 1)
                       for us in units.values() if us)
            self.loads.append({"blocks": len(keys), "payloads": len(payloads), "runs": runs})
            return load(keys, units, payloads, after)
        self.tier.load = counted
        self.decoders: list[str] = []

    def tokens(self, n: int) -> list[int]:
        return [self.rng.randrange(1000, 30000) for _ in range(n)]

    def run(self, prompt: list[int], params: SamplingParams = GREEDY) -> None:
        rid = self.engine.add(prompt, params)
        while any(r == rid for r in self.engine._live):
            self.step()

    def step(self) -> list:
        out = self.engine.step()
        for d in out:                                        # keep the decoders decoding
            if d.rid in self.decoders and d.finish_reason:
                self.decoders.remove(d.rid)
                self.start_decoder()
        return out

    def start_decoder(self) -> None:
        self.decoders.append(self.engine.add(self.tokens(256), SamplingParams(temperature=0.0, max_tokens=4096,
                                                                               ignore_eos=True)))

    def scatter(self) -> None:
        """Short requests run together, half aborted midway: the device units left scattered."""
        rids = [self.engine.add(self.tokens(self.rng.randrange(64, 1024)), GREEDY) for _ in range(8)]
        self.step()
        for rid in rids[::2]:
            self.engine.abort(rid)
        while any(r in self.engine._live for r in rids):
            self.step()

    def park(self, n: int) -> list[int]:
        prompt = self.tokens(n // 64 * 64)
        self.run(prompt)
        return prompt

    def demote(self) -> None:
        """Every unheld block off the device (pure frees: the host level holds them)."""
        for name, pool in self.index.pools.items():
            self.index.evict(name, pool.capacity, pure=True)

    def down(self) -> None:
        """Every evictable host copy written back to the level, and the writes done. With decoders running, the loop
        keeps stepping them while the writes run, as serving does (the clock's lanes move with launches only: a loop
        blocked in a drain leaves the write lane ahead of the clock, and the next read is charged beside writes that
        have finished, D4); with none, the drain."""
        for name, pool in self.tier.host.items():
            self.tier._room({name: pool.capacity}, pages=name in self.tier.pages, take=False)
        io = self.tier.io
        while self.decoders and (io.reads or io.writes or io.running):
            self.step()
        io.drain()

    def resume(self, prompt: list[int], append: list[int] | None = None,
               params: SamplingParams = GREEDY) -> tuple[float, int]:
        """TTFT (ms) of the conversation plus a 256-token append, and its cached tokens."""
        t0 = time.perf_counter()
        rid = self.engine.add(prompt + (append or self.tokens(APPEND)), params)
        cached, self.output = 0, []
        while True:
            for d in self.step():
                if d.rid == rid and d.token_ids:
                    ttft = (time.perf_counter() - t0) * 1e3
                    cached = d.cached_tokens or 0
                    self.output += d.token_ids
                    while rid in self.engine._live:
                        for e in self.step():
                            self.output += e.token_ids if e.rid == rid else []
                    return ttft, cached

    def pools(self) -> dict[str, int]:
        """The device pools' units (the staging pools are 0 if the engine measured too little free memory)."""
        return {p.name: p.capacity for p in self.engine.ledger.pools.values() if not p.host}

    def close(self) -> None:
        for rid in list(self.decoders):
            self.engine.abort(rid)
        self.engine.close()
        del self.engine, self.tier, self.index
        gc.collect()                                         # the next engine in this process measures what is free
        torch.cuda.empty_cache()


def cell_3a(b: Bench, args, sampler: Sampler) -> dict:
    out = {"lengths": {}}
    arms = ["vram", "ram", "ssd"]
    for n in args.lengths:
        rows = {a: [] for a in arms}
        for r in range(-1, args.reps):                       # rep -1: warm-up, untimed
            for arm in (arms if r % 2 == 0 else arms[::-1]):
                b.scatter()
                prompt = b.park(n)
                if arm != "vram":
                    b.demote()
                if arm == "ssd":
                    b.down()
                before, t0 = len(b.loads), time.time()
                counts0 = dict(b.tier.counts)
                ttft, cached = b.resume(prompt)
                tier = {k: b.tier.counts[k] - counts0.get(k, 0) for k in COUNTS}   # the resume's own (C-1)
                row = {"round": r, "order": "abc" if r % 2 == 0 else "cba", "ttft_ms": ttft, "cached": cached,
                       "loads": b.loads[before:], "level_read_bytes": tier["level_read_bytes"], "tier": tier,
                       "clocks": sampler.summary(t0, time.time())}
                if r >= 0:
                    rows[arm].append(row)
                log(f"3a {n} {arm} rep {r}: ttft {ttft:.1f} ms, cached {cached}, runs "
                    f"{[x['runs'] for x in row['loads']]}")
        kv = n // 64 * 64 * args.kv_bytes_per_token
        ssd = kv + sum(p.unit_bytes for k, p in b.tier.dev.items() if k not in b.tier.pages)   # and its checkpoint
        med = {a: statistics.median(x["ttft_ms"] for x in rows[a]) for a in arms}
        bars = {"ram": med["vram"] + 1.5 * kv / LINK * 1e3 + args.step_ms,
                "ssd": med["vram"] + 1.5 * (ssd / args.ssd_rate + ssd / LINK) * 1e3 + args.step_ms}
        out["lengths"][n] = {"kv_bytes": kv, "ssd_bytes": ssd, "median_ttft_ms": med, "p90_ttft_ms": {
            a: pct([x["ttft_ms"] for x in rows[a]], 0.9) for a in arms}, "bars_ms": bars,
            "pass": {a: med[a] <= bars[a] for a in bars},
            "tier_median": {a: {k: statistics.median(x["tier"][k] for x in rows[a]) for k in COUNTS} for a in arms},
            "rows": rows}
        log(f"3a {n}: median VRAM {med['vram']:.1f}, RAM {med['ram']:.1f} (bar {bars['ram']:.1f}), SSD "
            f"{med['ssd']:.1f} (bar {bars['ssd']:.1f})")
    return out


def cell_3f(args, sampler: Sampler) -> dict:
    """The two-engine restart (module docstring)."""
    root = pathlib.Path(args.ssd_dir)
    params = SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True)

    def history(b: Bench) -> list[tuple[list[int], list[int]]]:
        prompts = [b.park(16384) for _ in range(args.reps)]
        b.demote()
        b.down()
        return [(p, b.tokens(APPEND)) for p in prompts[-args.resumes:]]

    def resumes(b: Bench) -> tuple[list[float], list[int], list[list[int]]]:
        got = []
        for prompt, append in convs:                         # each from the level, in the same order
            ttft, cached = b.resume(prompt, append, params)
            got.append((ttft, cached, b.output))
        return [g[0] for g in got], [g[1] for g in got], [g[2] for g in got]
    a = Bench(args, str(root / "a"), "close")
    convs = history(a)
    ttft_a, cached_a, out_a = resumes(a)
    pools = [a.pools()]
    a.close()
    b = Bench(args, str(root / "b"), "close")
    assert history(b) == convs, "the parking history differs"
    persisted = b.tier.counts["write_back_bytes"]
    pools.append(b.pools())
    b.close()
    from holon_std.state_tier.ssd import SsdLevel
    from holon_std.state_tier.tier import HostTier
    spent: dict[str, float] = {}

    def timed(cls, name: str):                               # the restart's parts, apart from the engine build
        fn = getattr(cls, name)

        def wrapper(self, *a, **kw):
            t = time.perf_counter()
            try:
                return fn(self, *a, **kw)
            finally:
                spent[f"{cls.__name__}.{name}"] = time.perf_counter() - t
        return wrapper
    patched = [(SsdLevel, "restore"), (SsdLevel, "bind"), (HostTier, "attach")]
    saved = {(c, n): getattr(c, n) for c, n in patched}
    for c, n in patched:
        setattr(c, n, timed(c, n))
    t0 = time.perf_counter()
    try:
        b2 = Bench(args, str(root / "b"), "close")
    finally:
        for (c, n), fn in saved.items():
            setattr(c, n, fn)
    startup = time.perf_counter() - t0
    restored = b2.tier.counts["restored_blocks"]
    pools.append(b2.pools())
    restored_payloads = sum(s is not None for _, s in b2.tier.low)
    ttft_b, cached_b, out_b = resumes(b2)
    b2.level_reads = b2.tier.counts["level_read_bytes"]
    b2.engine.tier.level.persist = "off"                    # no snapshot of the measured run
    ssd = 16384 * args.kv_bytes_per_token + sum(p.unit_bytes for k, p in b2.tier.dev.items() if k not in b2.tier.pages)
    b2.close()
    bar = args.vram_ms + 1.5 * (ssd / args.ssd_rate + ssd / LINK) * 1e3 + args.step_ms
    med_b = statistics.median(ttft_b)
    return {"resumes": len(convs), "ttft_ms": {"no_restart": ttft_a, "restarted": ttft_b},
            "median_ttft_ms": {"no_restart": statistics.median(ttft_a), "restarted": med_b},
            "cached": {"no_restart": cached_a, "restarted": cached_b}, "outputs_identical": out_a == out_b,
            "restored_blocks": restored,
            "restored_payloads": restored_payloads, "device_pools": pools,
            "startup_s": startup, "restart_parts_s": spent, "persisted_bytes": persisted,
            "restore_s_per_persisted_gb": spent.get("SsdLevel.restore", 0) / (persisted / 1e9), "bar_ms": bar,
            "pass": med_b <= bar and out_a == out_b and cached_a == cached_b}


def cell_3b(b: Bench, args, sampler: Sampler) -> dict:
    """Decoders' step gaps while a fourth request resumes 32K from RAM (or SSD) vs plain decode steps."""
    n, gated, control, append_tier, append_ctrl, host_ms, loop_gated, loop_ctrl = 32768, [], [], [], [], [], [], []
    convs = [b.park(n) for _ in range(args.reps)]            # parked once, each resumed once per round
    for _ in range(3):
        b.start_decoder()
    for _ in range(8):
        b.step()

    def timed_steps(k: int) -> list[float]:
        gaps, t = [], time.perf_counter()
        for _ in range(k):
            c = time.thread_time()
            b.step()
            loop_ctrl.append((time.thread_time() - c) * 1e3)
            now = time.perf_counter()
            gaps.append((now - t) * 1e3)
            t = now
        return gaps
    rows = []
    for rnd in range(args.rounds):
        for i, prompt in enumerate(convs):
            b.demote()
            if args.ssd:
                b.down()
            rid = b.engine.add(prompt + b.tokens(APPEND), GREEDY)
            gaps, steps, t, t0 = [], 0, time.perf_counter(), time.time()
            cpu0 = time.process_time()
            seq = None
            while True:                                      # gated: until its first step runs
                c = time.thread_time()                       # the engine loop's own CPU time (this thread)
                b.step()
                loop = (time.thread_time() - c) * 1e3
                now = time.perf_counter()
                seq = seq or next(iter(b.engine._live.get(rid, ())), None)
                started = seq is not None and seq.num_computed > len(prompt) // 64 * 64
                if started:
                    append_tier.append((now - t) * 1e3)
                    break
                gaps.append((now - t) * 1e3)
                loop_gated.append(loop)
                steps += 1
                t = now
            host_ms.append((time.process_time() - cpu0) * 1e3 / max(1, steps + 1))
            while rid in b.engine._live:
                b.step()
            ctrl = timed_steps(max(1, steps))                 # plain decode steps, as many
            rid = b.engine.add(prompt + b.tokens(APPEND), GREEDY)     # the same conversation, from VRAM
            t = time.perf_counter()
            b.step()
            append_ctrl.append((time.perf_counter() - t) * 1e3)
            while rid in b.engine._live:
                b.step()
            gated += gaps
            control += ctrl
            rows.append({"round": rnd, "conversation": i, "gated_steps": steps, "gated_ms": gaps, "control_ms": ctrl,
                         "clocks": sampler.summary(t0, time.time())})
            log(f"3b{'-ssd' if args.ssd else ''} round {rnd} conv {i}: {steps} gated steps, median "
                f"{statistics.median(gaps) if gaps else float('nan'):.1f} vs control "
                f"{statistics.median(ctrl):.1f} ms")
    res = {"gaps": len(gated), "gated_median_ms": statistics.median(gated) if gated else None,
           "control_median_ms": statistics.median(control), "gated_p90_ms": pct(gated, 0.9) if gated else None,
           "control_p90_ms": pct(control, 0.9), "append_ratio": statistics.median(append_tier) / statistics.median(
               append_ctrl), "process_cpu_ms_per_step": statistics.median(host_ms),
           "loop_cpu_ms_per_step": {"gated": statistics.median(loop_gated) if loop_gated else None,
                                    "control": statistics.median(loop_ctrl),
                                    "gated_p90": pct(loop_gated, 0.9) if loop_gated else None},
           "blocking_sync": args.blocking_sync, "rows": rows}
    if gated:
        res |= {"median_ratio": res["gated_median_ms"] / res["control_median_ms"],
                "p90_ratio": res["gated_p90_ms"] / res["control_p90_ms"]}
        res["pass"] = res["median_ratio"] <= 1.1 and res["p90_ratio"] <= 1.2
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=("3a", "3b", "3f"), required=True)
    ap.add_argument("--vram-ms", type=float, help="3f: 3a's 16K VRAM median TTFT")
    ap.add_argument("--model", required=True)
    ap.add_argument("--lock", required=True)
    ap.add_argument("--ssd-dir", required=True)
    ap.add_argument("--ssd", action="store_true", help="3b: resume from the SSD level (3b-SSD)")
    ap.add_argument("--lengths", type=int, nargs="+", default=[4096, 16384, 32768])
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--resumes", type=int, default=5, help="3f: parked conversations each engine resumes (median)")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--beside", action="store_true", help="3a: three sessions decoding during the resumes")
    ap.add_argument("--step-ms", type=float, required=True, help="one decode step at the lock's settings (§8)")
    ap.add_argument("--kv-bytes-per-token", type=int, default=34816, help="the 27B q8_0's")
    ap.add_argument("--ssd-rate", type=float, default=SSD_RATE)
    ap.add_argument("--host-bytes", default="16GiB")
    ap.add_argument("--host-pages", type=float, default=0.85)
    ap.add_argument("--host-memory", type=int, default=20 << 30)
    ap.add_argument("--staging", type=int, default=5)
    ap.add_argument("--ssd-bytes", default="64GiB")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--blocking-sync", action="store_true", help="CUDA waits sleep, so the engine loop's CPU time is "
                    "host work (worker.blocking_sync)")
    ap.add_argument("--options", nargs="*", default=[], metavar="KEY=VALUE", help="more plugin options (e.g. the "
                    "tuned state_tier.step_cost)")
    ap.add_argument("--json")
    args = ap.parse_args()
    if args.blocking_sync:                                   # before the engine creates the CUDA context
        from worker import blocking_sync
        blocking_sync()
    sampler = Sampler()
    if args.cell == "3f":
        res = cell_3f(args, sampler)
        sampler.close()
        text = json.dumps(res, indent=1, default=str)
        if args.json:
            pathlib.Path(args.json).write_text(text + "\n")
        print(text)
        return
    b = Bench(args)
    try:
        if args.beside and args.cell == "3a":
            for _ in range(3):
                b.start_decoder()
        res = cell_3a(b, args, sampler) if args.cell == "3a" else cell_3b(b, args, sampler)
        res |= {"cell": args.cell + ("-ssd" if args.ssd else "") + (" beside" if args.beside else ""),
                "lock": b.lock["path"], "model": pathlib.Path(args.model).name, "gpu": torch.cuda.get_device_name(),
                "options": {k: v for k, v in b.cfg.plugin_options.items() if k != "state_tier.ssd.dir"},
                "tier_stats": b.tier.stats(), "step_ms": args.step_ms, "ssd_rate": args.ssd_rate, "link": LINK,
                "device_pools": b.pools(),
                "state_bytes": sum(p.capacity * p.unit_bytes for p in b.engine.ledger.pools.values() if not p.host)}
    finally:
        sampler.close()
        b.close()
    text = json.dumps(res, indent=1, default=str)
    if args.json:
        pathlib.Path(args.json).write_text(text + "\n")
    print(text[:2000])


if __name__ == "__main__":
    main()
