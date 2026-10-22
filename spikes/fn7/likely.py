"""FN step 7, §21.7's measure-first gate: how well the placement profile's routing shares predict a prompt step's
routed misses. Serves `v1-long` prompts cut to `--ctx` tokens (one step each, after a warm-up and decode that refills
the cache) and, per prompt step, splits the pieces missing before it by the prediction "likely" — share p(g, e) with
1 − (1 − p)^T ≥ τ for the step's T prompt tokens — against what the step routed: streamed (missing and likely), caught
(and routed), wasted (likely, not routed), staged (routed, not likely), in GB, per τ. Usage:
likely.py <model> '<engine flags>' [--ctx 1024] [--prompts 3]"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
TAUS = (0.1, 0.3, 0.5, 0.7, 0.9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("flags")
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--prompts", type=int, default=3)
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT / "spikes/fn6"))
    from _engine import STORE, engine
    from holon.api import SamplingParams

    e = engine(a.model, a.flags, max_seqs=1, max_model_len=a.ctx + 256, prefix_cache=False)
    s = e.ctx.shared_objects[STORE]
    unit = np.array([[c.unit_bytes for _, c in sorted(s.groups[g].items())] for g in s._order])        # [G, R]
    counts = s._profile_counts().astype(np.float64)                                                  # [G, E]
    share = counts / np.maximum(counts.sum(1, keepdims=True), 1) * s.top_k
    prev, rows = {"map": s.residency.host.copy()}, []

    def tap(step, record) -> None:
        tokens = int(step.n_query.sum())
        if tokens >= a.ctx // 2 and record["lends"] is not None:                # a prompt step
            absent = prev["map"] < 0                                            # [G, E, R]
            routed = record["routed"][..., None]
            row = {"tokens": tokens, "absent_gb": float((absent * unit[:, None, :]).sum() / 1e9),
                   "routed_gb": float(((absent & routed) * unit[:, None, :]).sum() / 1e9)}
            for tau in TAUS:
                likely = (share >= 1 - (1 - tau) ** (1 / tokens))[..., None]
                gb = {k: float((m * unit[:, None, :]).sum() / 1e9) for k, m in (
                    ("streamed", absent & likely), ("caught", absent & likely & routed),
                    ("wasted", absent & likely & ~routed), ("staged", absent & routed & ~likely))}
                row[f"tau_{tau}"] = {k: round(v, 2) for k, v in gb.items()}
            rows.append(row)
        prev["map"] = s.residency.host.copy()

    s.taps.append(tap)
    prompts = [json.loads(x) for x in (ROOT / "judge/prompts/v1-long.jsonl").read_text().splitlines() if x]
    for p in prompts[:a.prompts]:
        e.generate([p["ids"][:a.ctx]], SamplingParams(temperature=0.0, max_tokens=128, ignore_eos=True))
    e.close()
    print(json.dumps({"profile": bool(counts.sum()), "ctx": a.ctx, "steps": rows}, indent=1))


if __name__ == "__main__":
    main()
