"""FN step 7, fix 2's gate (docs/dev/v1-flash-next.md §21.10): what a refill of a prompt's lent slots could spare. Serves
two `v1-long` prompts cut to `--ctx` tokens, `--tokens` decode tokens each; per prompt, the experts it lent (resident
before it, absent after it) and the decode / verify rows' misses of the first `--window` steps after it: how many,
how many distinct experts, how many repeat an expert's earlier miss, and the share the lent experts cover — against
the prompt's own most-routed absent experts (its routes counted at the MoE stores' `route`). Usage:
after_prompt.py <model> '<engine flags>' [--ctx 1024] [--tokens 240] [--window 20]"""
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
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--tokens", type=int, default=240)
    ap.add_argument("--window", type=int, default=20)
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT / "spikes/fn6"))
    from _engine import STORE, engine
    from holon.api import SamplingParams
    from holon_std.moe.offload import Offload

    rows = [json.loads(x) for x in (ROOT / "judge/prompts/v1-long.jsonl").read_text().splitlines() if x]
    prompts = sorted((r["ids"][:r["gen_start"]] for r in rows), key=len)[-2:][::-1]
    e = engine(a.model, a.flags, max_seqs=1, max_model_len=a.ctx + a.tokens + 64, prefix_cache=False)
    s = e.ctx.shared_objects[STORE]
    groups, experts = s.residency.host.shape[:2]
    routed = np.zeros((groups, experts))                        # the prompt's routes, per expert
    state = {"map": None, "lent": None, "pre": None, "post": None}
    steps: list[np.ndarray] = []                                # per decode / verify step: its missed routes [G, E]

    def counting(mod, name):
        inner = getattr(mod, name)

        def f(ids, first, x):
            if len(ids) > 64 and not mod._aux and name == "route":
                np.add.at(routed[s._order[mod.layer]], ids.long().cpu().numpy().ravel(), 1)
            return inner(ids, first, x)
        setattr(mod, name, f)

    for m in e.executor.model.modules():
        if isinstance(m, Offload):
            counting(m, "route")

    def tap(step, record) -> None:
        held = (s.residency.host >= 0).all(-1)
        counts = record["counts"]
        if state["lent"] is None and counts.sum() == 0:          # the prompt's step
            state["lent"], state["post"] = state["pre"] & ~held, held
        elif state["map"] is not None and counts.sum():
            steps.append(counts * ~(state["map"] >= 0).all(-1))
        state["map"] = s.residency.host.copy()
    s.taps.append(tap)

    out = []
    for ids in prompts:
        ids = ids[:a.ctx]
        routed[:] = 0
        steps.clear()
        state.update(lent=None, pre=(s.residency.host >= 0).all(-1).copy())
        e.generate([ids], SamplingParams(temperature=0.0, max_tokens=a.tokens, ignore_eos=True))
        window = steps[:a.window]
        missed = sum(window)
        seen, repeat = np.zeros_like(missed, dtype=bool), 0.0
        for m in window:
            repeat += m[(m > 0) & seen].sum()
            seen |= m > 0
        lent = state["lent"]
        row = {"lent_experts": int(lent.sum()), "missed_routes": int(missed.sum()), "distinct": int(seen.sum()),
               "repeat_routes": int(repeat), "lent_share": round(float(missed[lent].sum() / missed.sum()), 3)}
        for n in (1000, 2000, 3000):
            score = np.where(state["post"], -1, routed)            # the prompt's most-routed absent experts
            pick = np.zeros(score.size, dtype=bool)
            pick[np.argsort(-score, axis=None)[:n]] = True
            pick = pick.reshape(score.shape) & (score > 0)
            row[f"prompt_top{n}_share"] = round(float(missed[pick].sum() / missed.sum()), 3)
        out.append(row)
    e.close()
    print(json.dumps({"ctx": a.ctx, "tokens": a.tokens, "window": a.window, "prompts": out}, indent=1))


if __name__ == "__main__":
    main()
