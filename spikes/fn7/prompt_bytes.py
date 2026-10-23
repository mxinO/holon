"""FN §21.14, the 5090's 1K prefill bytes: what a prompt fetches against the cache it meets, replayed. Serves the
bench's warm-up, then its 1K prompts (`--requests`, offload_ctx's plan) on the deployment; per prompt the experts each
layer's prompt rows route to (counted at the MoE stores' `route`), the cache before it and what it lent; then the
bytes those routes miss under: the cache it met, less what it lent (what the prompt fetched); the cache it met; a
cache seeded from the placement profile (F2's order) of the same bytes; and of `--share` of all experts' bytes
(Strata's). Usage: prompt_bytes.py <model> '<engine flags>' --requests <requests.json> [--share 0.54]"""
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
    ap.add_argument("--requests", required=True)
    ap.add_argument("--share", type=float, default=0.54)
    ap.add_argument("--ctx", default="1024")
    ap.add_argument("--tokens", type=int, default=8, help="decode tokens a request makes (the bench's: 256)")
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT / "spikes/fn6"))
    from _engine import STORE, engine
    from holon.api import SamplingParams
    from holon_std.moe.offload import Offload
    from holon_std.tiered.store import _Budget

    plan = json.loads(pathlib.Path(a.requests).read_text())
    e = engine(a.model, a.flags, max_seqs=1, max_model_len=33792, prefix_cache=False)
    s = e.ctx.shared_objects[STORE]
    groups, experts = s.residency.host.shape[:2]
    target = s.borrowing_groups
    nbytes = np.array([[sum(c.unit_bytes for c in s.groups[g].values()) for _ in range(experts)]
                       for g in s._order])                                                 # [G, E] an expert's bytes
    routed = np.zeros((groups, experts), dtype=bool)

    def counting(mod, name):
        inner = getattr(mod, name)

        def f(ids, first, x):
            if len(ids) > 64 and not mod._aux:
                got = ids.long().cpu().numpy()
                routed[s._order[mod.layer], np.unique(got[got >= 0])] = True
            return inner(ids, first, x)
        setattr(mod, name, f)
    for m in e.executor.model.modules():
        if isinstance(m, Offload):
            counting(m, "route")
            counting(m, "route_pipelined")
    state = {"pre": None, "lent": None}

    def tap(step, record) -> None:
        if state["pre"] is not None and state["lent"] is None and record["lends"] is not None:
            held = (s.residency.host >= 0).all(-1)
            state["lent"] = state["pre"] & ~held
    s.taps.append(tap)

    profile = s._profile_counts()
    order = _Budget.f2(profile)

    def seeded(budget: float, held: np.ndarray | None = None) -> np.ndarray:
        """F2's order of the profile, whole experts, until `budget` bytes (past `held`'s, kept)."""
        held = np.zeros((groups, experts), dtype=bool) if held is None else held.copy()
        left = budget - float((nbytes * held).sum())
        for g, i in order:
            if g < target and not held[g, i] and nbytes[g, i] <= left:
                held[g, i] = True
                left -= nbytes[g, i]
        return held

    def scored(budget: float, held: np.ndarray) -> np.ndarray:
        """F11's decayed counts (the device state, now), hottest first over every target layer, ties by (layer, id),
        whole experts until `budget` bytes (past `held`'s, kept)."""
        score = s.state["score"].float().cpu().numpy()[:target]
        held, left = held.copy(), budget - float((nbytes * held).sum())
        for flat in np.lexsort((np.arange(score.size), -score.ravel())):
            g, i = divmod(int(flat), experts)
            if not held[g, i] and nbytes[g, i] <= left:
                held[g, i] = True
                left -= nbytes[g, i]
        return held

    warm = plan["warmup"]
    e.generate([warm["ids"]], SamplingParams(temperature=0.0, max_tokens=warm["output"], ignore_eos=True))
    total = float(nbytes[:target].sum())
    capacity = float(sum((c.size - c.null - c.staging) * c.unit_bytes for c in s.classes.values()
                         if not c.name.startswith("aux.")))           # the target's cache slots, bytes
    out = []
    for r in plan["requests"]:
        if r["key"].split("/")[0] != a.ctx:
            continue
        routed[:] = False
        pre = (s.residency.host >= 0).all(-1).copy()
        state.update(pre=pre, lent=None, fill_lfu=scored(capacity, pre))
        e.generate([r["ids"]], SamplingParams(temperature=0.0, max_tokens=a.tokens, ignore_eos=True))
        lent = state["lent"] if state["lent"] is not None else np.zeros_like(pre)
        fill_lfu = state["fill_lfu"]
        rt = routed[:target]
        gb = lambda held: float((nbytes[:target] * (rt & ~held[:target])).sum()) / 1e9   # noqa: E731
        met = float((nbytes[:target] * pre[:target]).sum())
        out.append({"key": r["key"], "routed_gb": float((nbytes[:target] * rt).sum()) / 1e9,
                    "cache_met_share": round(met / total, 4), "lent_experts": int(lent.sum()),
                    "capacity_share": round(capacity / total, 4),
                    "missed_gb": {"met_less_lent": round(gb(pre & ~lent), 2), "met": round(gb(pre), 2),
                                  "met_filled": round(gb(seeded(capacity, pre)), 2),
                                  "met_filled_lfu": round(gb(fill_lfu), 2),
                                  "met_filled_less_lent": round(gb(seeded(capacity, pre) & ~lent), 2),
                                  "profile_same_bytes": round(gb(seeded(met)), 2),
                                  f"profile_{a.share}": round(gb(seeded(a.share * total)), 2)}})
        print(json.dumps(out[-1]), flush=True)
    e.close()
    print(json.dumps({"experts_gb": total / 1e9, "prompts": out}))


if __name__ == "__main__":
    main()
