"""M0.7 D7: a discrete-event model of Holon's scheduler replaying a trace (agent, long_prompt), calibrated per class
from step-0 records, and checked against a measured `run.py` record of today's scheduler.

Costs (ms):
- decode step: the graphed wall per batch at two contexts (`step_profile.py` decode rows), linear in the mean
  context and in the batch;
- prefill step of N tokens: the eager wall of an N-token turn after a 4096-token history (`prefill_idle.py`'s
  aligned rows, one step each) — its host floor included — interpolated in N (flat below the smallest size), plus
  `alpha` x N x (history - 4096) when a second history is given (fitted on the sizes both measured);
- mixed step (D7): the prefill cost of chunk + decode rows (it runs eagerly) plus the decode rows' attention.
Scheduling, as `core/holon/core/sched.py`: `max_seqs` slots; while a waiting request can be admitted the step is
a prefill (FCFS, budget `max_step_tokens`), else every running request decodes. `mixed:<cap>`: decode rows first,
then prefill chunks up to `cap` tokens (at least min(64, cap)) while anything decodes. Prefix reuse is limited by
the recurrent checkpoints (`checkpoints` units, one per prompt at its last aligned position): a turn resumes from
the furthest checkpoint on its chain — its session's earlier prompt ends (while the session's KV blocks are cached:
a finished turn's blocks stay until the pool of `kv_tokens` needs them, least recently used first), or the shared
system prompt's end once a first turn appending < 64 tokens published it. Matching refreshes every checkpoint on
the chain; a publish into a full pool drops the least recently matched one. Checkpoints are
taken inside the step (on the 64-token grid); a budget-truncated step ends on the grid. Not modelled: overlap's
one-step lag, and admission as the prefill loop reaches a request (here every waiting request with a free slot is
admitted, and takes its hit, at the top of the step).

    python bench/mixed_sim.py --decode <step_profile json> [<step_profile json at another ctx>]
        --prefill-idle <prefill_idle json> [<prefill_idle json at another history>] --trace bench/traces/agent.jsonl
        [--measured <run.py record>] [--max-seqs 4] [--policies today mixed:256 mixed:512] [--json out]
"""
from __future__ import annotations

import argparse
import collections
import heapq
import json
import pathlib
import statistics

import numpy as np

ALIGN = 64                      # the recurrent checkpoint's alignment (lcm of the GDN chunk and the page size)
GRAPHED_AT = 86                 # step_profile.py's decode steps before the middle of its graphed window (old records)


class Costs:
    def __init__(self, decode_records: list[dict], idle_records: list[dict]) -> None:
        pts = collections.defaultdict(list)                  # batch -> [(ctx, wall)]
        for rec in decode_records:
            for row in rec["rows"].values():
                if row["kind"] == "decode":
                    pts[row["batch"]].append((row.get("graphed_ctx", row["ctx"] + GRAPHED_AT), row["graphed_wall_ms"]))
        self.batches = sorted(pts)
        slopes = {b: (max(pts[b])[1] - min(pts[b])[1]) / (max(pts[b])[0] - min(pts[b])[0])
                  for b in pts if len(pts[b]) > 1}
        self.per_row_ctx = statistics.mean(v / b for b, v in slopes.items()) if slopes else 0.0
        # batch -> its wall at no context, through the shortest context it was measured at, at the per-row slope
        self.dec = {b: min(pts[b])[1] - self.per_row_ctx * b * min(pts[b])[0] for b in self.batches}
        by_hist = {}
        for rec in idle_records:
            rows = next(v for k, v in rec.items() if k.startswith("aligned")) if "rows" not in rec else rec["rows"]
            by_hist[rec["history"]] = {r["new_tokens"]: r["wall_ms"] for r in rows if r["steps"] == 1}
        self.hist = min(by_hist, key=lambda h: abs(h - 4096))
        self.pre = sorted(by_hist[self.hist].items())
        self.alpha = 0.0                                       # ms per token per history token beyond self.hist
        for h, walls in by_hist.items():
            common = [n for n in walls if n in by_hist[self.hist] and h != self.hist]
            if common:
                self.alpha = statistics.mean((walls[n] - by_hist[self.hist][n]) / (n * (h - self.hist))
                                             for n in common)

    def decode(self, batch: int, ctx_sum: int) -> float:
        lo = max([b for b in self.batches if b <= batch], default=self.batches[0])
        hi = min([b for b in self.batches if b >= batch], default=self.batches[-1])
        at = self.dec[lo] if hi == lo else self.dec[lo] + (self.dec[hi] - self.dec[lo]) * (batch - lo) / (hi - lo)
        return at + self.per_row_ctx * ctx_sum

    def prefill(self, tokens: int, history: float = 4096) -> float:
        ns, walls = zip(*self.pre, strict=True)
        if tokens <= ns[-1]:
            t = float(np.interp(tokens, ns, walls))
        else:                                                   # beyond the largest: linear in the last segment
            t = walls[-1] + (walls[-1] - walls[-2]) / (ns[-1] - ns[-2]) * (tokens - ns[-1])
        return t + self.alpha * tokens * max(0.0, history - self.hist)


class Req:
    def __init__(self, sid: int, turn: int, release: float, prompt: int, out: int) -> None:
        self.sid, self.turn, self.release, self.prompt, self.out = sid, turn, release, prompt, out
        self.computed = self.hit = 0
        self.times: list[float] = []
        self.slot = False

    @property
    def ctx(self) -> int:
        return self.prompt + len(self.times)


def simulate(trace: list[dict], costs: Costs, policy: str, max_seqs: int, budget: int, checkpoints: int,
             kv_tokens: int | None = None) -> list[Req]:
    """Replay the trace; returns the finished requests with their token times (ms)."""
    cap = int(policy.split(":")[1]) if policy.startswith("mixed:") else None
    by_id = {s["id"]: s for s in trace}
    ready = [(s["arrival"] * 1e3, s["id"], 0, 0) for s in trace]           # (release, sid, turn, previous prompt + out)
    heapq.heapify(ready)
    # Checkpoints, least recently matched first: (owner, position); the owner is the session, or "system" for one at
    # or before the shared system prompt's end (a first turn appending < 64 tokens publishes it, for every session).
    lru: collections.OrderedDict[tuple, None] = collections.OrderedDict()
    cached: collections.OrderedDict[int, int] = collections.OrderedDict()   # sid -> its cached KV tokens (unshared)
    system = max(s["system"] for s in trace)
    waiting: list[Req] = []
    running: list[Req] = []
    done: list[Req] = []
    t = 0.0
    while ready or waiting or running:
        while ready and ready[0][0] <= t:
            release, sid, i, history = heapq.heappop(ready)
            s = by_id[sid]
            prompt = (history or s["system"]) + s["turns"][i]["append"]
            waiting.append(Req(sid, i, release, prompt, s["turns"][i]["output"]))
        if not waiting and not running:
            t = ready[0][0]
            continue
        for r in waiting:                                       # admission: a slot, and the checkpoint hit
            if not r.slot and sum(x.slot for x in running + waiting) < max_seqs:
                r.slot = True
                own = r.turn > 0 and (kv_tokens is None or r.sid in cached)
                on_chain = sorted((k for k in lru if k[0] == "system" or (own and k[0] == r.sid)), key=lambda k: k[1])
                for k in on_chain:                             # matching refreshes the chain's entries, in order
                    lru.move_to_end(k)
                r.computed = r.hit = max((k[1] for k in on_chain), default=0)
                cached.pop(r.sid, None)                        # its blocks are the request's now
        if kv_tokens is not None:                              # unused cached blocks go, least recently used first
            used = system + sum(r.ctx - system * (r.turn == 0 or r.hit > 0) for r in running + waiting if r.slot)
            while cached and used + sum(cached.values()) > kv_tokens:
                sid, _ = cached.popitem(last=False)
                for k in [k for k in lru if k[0] == sid]:
                    del lru[k]
        admitted = [r for r in waiting if r.slot]
        decode = list(running)
        if admitted and (cap is None or not decode):            # a prefill step (today; or nothing decodes)
            left, decode = budget, []
        elif admitted:                                         # mixed: decode rows first, then a capped chunk
            left = max(min(cap, budget - len(decode)), min(ALIGN, cap))
        else:
            left = 0
        rows = []
        for r in admitted:
            if left <= 0:
                break
            n = min(r.prompt - r.computed, left)
            at = r.prompt // ALIGN * ALIGN                      # the prompt-end checkpoint (Recurrent.cut)
            if r.computed < at < r.computed + n and (at - r.computed) % ALIGN:
                n = at - r.computed                             # off the grid: the step ends there
            elif checkpoints and r.computed + n < r.prompt and (r.computed + n) % ALIGN and \
                    (r.computed + n) // ALIGN * ALIGN > r.computed:
                n = (r.computed + n) // ALIGN * ALIGN - r.computed   # budget-truncated: down to the grid
            rows.append((r, n))
            left -= n
        chunk = sum(n for _, n in rows)
        hist = max((r.computed for r, _ in rows), default=0)
        if rows and decode:
            cost = costs.prefill(chunk + len(decode), hist) + costs.per_row_ctx * sum(r.ctx for r in decode)
        elif rows:
            cost = costs.prefill(chunk, hist)
        else:
            cost = costs.decode(len(decode), sum(r.ctx for r in decode))
        t += cost
        for r in decode:
            r.times.append(t)
        for r, n in rows:
            at = r.prompt // ALIGN * ALIGN
            r.computed += n
            if r.computed - n < at <= r.computed and checkpoints:   # published; the LRU evicts the oldest
                key = ("system" if r.turn == 0 and at <= system else r.sid, at)
                lru[key] = None
                lru.move_to_end(key)
                while len(lru) > checkpoints:
                    lru.popitem(last=False)
            if r.computed == r.prompt:
                r.times.append(t)
                waiting.remove(r)
                running.append(r)
        for r in [r for r in running if len(r.times) >= r.out]:
            running.remove(r)
            done.append(r)
            cached[r.sid] = r.prompt + r.out - system
            s = by_id[r.sid]
            if r.turn + 1 < len(s["turns"]):
                heapq.heappush(ready, (t + s["turns"][r.turn + 1]["gap"] * 1e3, r.sid, r.turn + 1, r.prompt + r.out))
    return done


def pct(xs: list[float], q: float) -> float:
    return round(float(np.percentile(xs, q)), 1)


def metrics(done: list[Req]) -> dict:
    ttft = [r.times[0] - r.release for r in done]
    tpot = [(r.times[-1] - r.times[0]) / (len(r.times) - 1) for r in done if len(r.times) > 1]
    gaps = [max(np.diff(r.times)) for r in done if len(r.times) > 1]
    span = max(r.times[-1] for r in done) - min(r.release for r in done)
    return {"ttft_ms": {"p50": pct(ttft, 50), "p99": pct(ttft, 99), "max": round(max(ttft), 1)},
            "tpot_ms": {"p50": pct(tpot, 50), "p99": pct(tpot, 99)},
            "max_gap_ms": {"p50": pct(gaps, 50), "p99": pct(gaps, 99), "max": round(max(gaps), 1)},
            "throughput_tok_s": round(sum(len(r.times) for r in done) / span * 1e3, 1),
            "prefix_tokens_saved": sum(r.hit for r in done)}


def paired_ttft(done: list[Req], base: list[Req]) -> float:
    ref = {(r.sid, r.turn): r.times[0] - r.release for r in base}
    return round(statistics.median((r.times[0] - r.release) / ref[(r.sid, r.turn)] for r in done
                                   if ref.get((r.sid, r.turn), 0) > 0), 3)


def measured(record: dict, trace: str) -> dict | None:
    """A run.py record's summary of `trace` for its first backend, in this module's metrics."""
    s = record["summary"].get(trace, {}).get(record["order"][0])
    if not s:
        return None
    rounds = len({run["round"] for run in record["runs"] if run.get("trace") == trace
                  and run["backend"] == record["order"][0]})
    return {"ttft_ms": s["ttft_ms"], "tpot_ms": s["tpot_ms"], "max_gap_ms": s.get("max_gap_ms"),
            "throughput_tok_s": s["throughput_tok_s"],
            "prefix_tokens_saved": s["stats"].get("prefix_tokens_saved", 0) // max(1, rounds)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decode", nargs="+", required=True, help="step_profile.py records (decode rows)")
    ap.add_argument("--prefill-idle", nargs="+", required=True, help="prefill_idle.py records")
    ap.add_argument("--trace", nargs="+", default=["bench/traces/agent.jsonl", "bench/traces/long_prompt.jsonl"])
    ap.add_argument("--measured", help="a run.py record of today's scheduler on the same traces")
    ap.add_argument("--max-seqs", type=int, default=4)
    ap.add_argument("--max-step-tokens", type=int, default=2048)
    ap.add_argument("--checkpoints", type=int, default=4, help="recurrent checkpoint units (recurrent.checkpoints)")
    ap.add_argument("--kv-tokens", type=int, help="the KV pool's tokens (default: the measured run's; else no limit)")
    ap.add_argument("--policies", nargs="+", default=["today", "mixed:256", "mixed:512"])
    ap.add_argument("--json")
    args = ap.parse_args()
    costs = Costs([json.loads(pathlib.Path(p).read_text()) for p in args.decode],
                  [json.loads(pathlib.Path(p).read_text()) for p in args.prefill_idle])
    record = json.loads(pathlib.Path(args.measured).read_text()) if args.measured else None
    if args.kv_tokens is None and record is not None:
        args.kv_tokens = next(iter(record["backends"].values())).get("kv_tokens")
    out = {"costs": {"decode": {b: round(a, 3) for b, a in costs.dec.items()},
                     "per_row_ctx_ms": round(costs.per_row_ctx, 7), "prefill_history": costs.hist,
                     "prefill": costs.pre, "alpha": costs.alpha},
           "settings": {"max_seqs": args.max_seqs, "max_step_tokens": args.max_step_tokens,
                        "checkpoints": args.checkpoints, "kv_tokens": args.kv_tokens}, "traces": {}}
    for path in args.trace:
        name = pathlib.Path(path).stem
        sessions = [json.loads(line) for line in pathlib.Path(path).read_text().splitlines() if line.strip()]
        runs = {p: simulate(sessions, costs, p, args.max_seqs, args.max_step_tokens, args.checkpoints,
                            args.kv_tokens) for p in args.policies}
        res = {p: metrics(done) for p, done in runs.items()}
        for p in args.policies[1:]:
            res[p]["paired_ttft"] = paired_ttft(runs[p], runs[args.policies[0]])
        if record is not None and (m := measured(record, name)):
            res["measured"] = m
        out["traces"][name] = res
        print(name)
        for p, m in res.items():
            print(f"  {p:10s} {json.dumps(m)}")
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
