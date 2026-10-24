"""FN step 7 (budgeted admission, predict first): a routing trace of decode and verify steps for `admit_sim.py`. Serves
`v1-long` prompts with the deployment's flags (eager: hooks run) and records, from the first measured step, the
store's whole-expert residency [G, E] and the policy's scores, and each step's routed ids per target layer [G, T, k].
Usage: admit_trace.py <model> '<engine flags>' <out.npz> [--prompts 4] [--tokens 160] [--ballast-gib 16]"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("flags")
    ap.add_argument("out")
    ap.add_argument("--prompts", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=160)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--ballast-gib", type=float, default=0.0, help="emulate a card of this much VRAM (0: the card's)")
    a = ap.parse_args()
    import torch
    if a.ballast_gib:
        free, total = torch.cuda.mem_get_info()
        ballast = torch.empty(int(total - a.ballast_gib * 2**30), dtype=torch.uint8, device="cuda")  # noqa: F841
    sys.path.insert(0, str(ROOT / "spikes/fn6"))
    from _engine import STORE, engine
    from holon.api import SamplingParams
    from holon_std.moe.units import MoE

    e = engine(a.model, a.flags, max_seqs=1, max_model_len=a.ctx + a.tokens + 64, cuda_graphs=False)
    s = e.ctx.shared_objects[STORE]
    units = [m for m in e.executor.model.modules() if isinstance(m, MoE)]
    row = {id(u): s._order[u._store.layer] for u in units}
    groups, experts = s.residency.host.shape[:2]
    steps, cur, start = [], {}, {}
    route = MoE._route

    def hooked(self, x):
        out = route(self, x)
        g = row.get(id(self))
        if g is not None and x.shape[0] <= 8:                 # a decode or verify pass of the target
            if g == 0:
                if cur:
                    steps.append(dict(cur))
                    cur.clear()
                if not start:
                    start["resident"] = (s.residency.host >= 0).all(-1).copy()
                    start["score"] = s.state["score"].cpu().numpy().copy() if "score" in s.state else None
            cur[g] = out[1].detach().cpu().numpy().astype(np.int32)
        return out
    MoE._route = hooked
    prompts = [json.loads(x) for x in (ROOT / "judge/prompts/v1-long.jsonl").read_text().splitlines() if x]
    for p in prompts[:a.prompts]:
        e.generate([p["ids"][:a.ctx]], SamplingParams(temperature=0.0, max_tokens=a.tokens, ignore_eos=True))
    if cur:
        steps.append(dict(cur))
    steps = [st for st in steps if len(st) == len(units)]
    t_max = max(st[0].shape[0] for st in steps)
    ids = np.full((len(steps), len(units), t_max, steps[0][0].shape[1]), -1, dtype=np.int32)
    for i, st in enumerate(steps):
        for g, v in st.items():
            ids[i, g, :v.shape[0]] = v
    score = start["score"] if start["score"] is not None else np.zeros((groups, experts))
    np.savez_compressed(a.out, ids=ids, resident=start["resident"][:len(units)], score=score[:len(units)],
                        empty=np.array([len(x) for x in s.residency.empty][:len(units)]))
    print(json.dumps({"steps": len(steps), "groups": len(units), "experts": experts, "t_max": int(t_max),
                      "resident_share": float(start["resident"][:len(units)].mean())}))


if __name__ == "__main__":
    main()
