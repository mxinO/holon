"""The device cache's hit rate at decode (and verify), as Strata reports its own: the decode rows' routed lookups
that found their expert resident before the step (the host mirror as of the last commit), and the distinct experts
missed per step with their bytes (the CPU's share excluded).
Prompts the placement profile never saw: `v1-long`'s, cut to 4,096 tokens. Usage: hitrate.py <model> '<engine flags>'
<prompts.jsonl> <new tokens>"""
from __future__ import annotations

import json
import sys

import numpy as np
from _engine import STORE, engine
from holon.api import SamplingParams


def main() -> None:
    model, flags, prompts, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    e = engine(model, flags, max_seqs=1, max_model_len=33792)
    s = e.ctx.shared_objects[STORE]
    unit = np.array([[c.unit_bytes for _, c in sorted(s.groups[g].items())] for g in s._order])     # [G, R]
    prev, tot = {"map": s.residency.host.copy()}, dict.fromkeys(("lookups", "hits", "steps", "missed", "bytes"), 0)

    def tap(step, record) -> None:
        counts = record["counts"]
        if counts.sum():
            resident = (prev["map"] >= 0).all(-1)                         # [G, E] before the step
            tot["lookups"] += int(counts.sum())
            tot["hits"] += int(counts[resident].sum())
            tot["steps"] += 1
            missed = (counts > 0) & ~resident & ~record["cpu"]
            tot["missed"] += int(missed.sum())
            tot["bytes"] += int((missed[..., None] * unit[:, None, :]).sum())
        prev["map"] = s.residency.host.copy()

    s.taps.append(tap)
    for line in list(open(prompts))[:6]:
        e.generate([json.loads(line)["ids"][:4096]], SamplingParams(temperature=0.0, max_tokens=n))
    steps = tot["steps"]
    print(json.dumps(dict(tot, hit_rate=round(tot["hits"] / tot["lookups"], 4), miss_experts_per_step=round(
        tot["missed"] / steps, 2), miss_mb_per_step=round(tot["bytes"] / steps / 2**20, 2), flags=flags)))


if __name__ == "__main__":
    main()
