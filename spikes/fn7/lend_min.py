"""FN §21.16, lend the minimum: what a prompt's step lends, per purpose, against what each purpose needs; and what
lending only that would spare, replayed on the bench's exact workload (offload_ctx's plan: its warm-up, then its 1K
and 4K prompts in order, 256 decode tokens each).

Per prompt step that borrows (`Lent`): its mode (ring or stage), the empty slots it found, the experts it lent per
`lend` call — call 0 the tail (D11: the large passes' contiguous scratch, then the step-peak loan past it, split by
their bytes), calls 1… the ring's halves or the staging slots —, per class the slots it held against what the purpose
needs: staging, a layer's most distinct misses, or (known when the step lends) its most absent pieces — with
`moe.pipeline` parts, also the current part and the next of either —; the ring, twice a layer's most absent pieces
(both halves). The rounds are replayed with the policy's own `lend` (F11's coldest whole experts, the score before
the step) and checked against the experts the step lent; then the least need, iterated to its fixed point (lending
makes misses), picks its experts the same way.

The saving: the prompt's missed bytes (its routes against the cache it met, less what it lent) with the lent set
cut to the minimum (the tail kept whole), and the decode / verify rows' missed routes after it (the first `--window`
steps, and every step to the next prompt) that hit an expert lent now but not at the minimum, until it is back.
Usage: lend_min.py <model> '<engine flags>' --requests <requests.json> [--ctx 1024 4096] [--window 20]"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUTPUT = 256                                   # the bench's decode tokens a request (bench/offload_ctx.py)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("flags")
    ap.add_argument("--requests", required=True)
    ap.add_argument("--ctx", nargs="+", default=["1024", "4096"])
    ap.add_argument("--window", type=int, default=20)
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT / "spikes/fn6"))
    from _engine import STORE, engine
    from holon.api import SamplingParams
    from holon_plugins.placement.lfu import reference

    plan = json.loads(pathlib.Path(a.requests).read_text())
    e = engine(a.model, a.flags, max_seqs=1, max_model_len=33792, prefix_cache=False)
    s = e.ctx.shared_objects[STORE]
    groups, experts = s.residency.host.shape[:2]
    target = s.borrowing_groups
    classes = list(s.classes.values())
    table = s.class_table                                                     # [G, R]
    unit = np.array([c.unit_bytes for c in classes])
    nbytes = unit[table].sum(1)[:, None].repeat(experts, 1)                   # [G, E] an expert's bytes
    lending = s.lending_classes.cpu().numpy()
    pipeline = s.pipeline

    def slots(sel: np.ndarray) -> np.ndarray:
        """Per class, the slots of the experts `sel` [G, E] holds."""
        out = np.zeros(len(classes), dtype=np.int64)
        for r in range(table.shape[1]):
            np.add.at(out, table[:, r], sel.sum(1))
        return out

    def most(per_layer: np.ndarray) -> np.ndarray:
        """Per class, the most of `per_layer` [G] over the target's layers of it."""
        out = np.zeros(len(classes), dtype=np.int64)
        for r in range(table.shape[1]):
            np.maximum.at(out, table[:target, r], per_layer[:target])
        return out

    state = {"map": s.residency.host.copy(), "score": s.state["score"].cpu().clone(), "prompt": None}
    steps: list[tuple[np.ndarray, np.ndarray]] = []           # after the prompt: (missed routes [G, E], absent)
    found: dict = {}

    def tap(step, record) -> None:
        held = (state["map"] >= 0).all(-1)                    # the cache the step met
        if record["lends"] is not None and state["prompt"] is not None and not found:
            found.update(analyse(record, held, state["score"]))
        elif found and record["counts"].sum():
            steps.append((record["counts"] * ~held, (s.residency.host < 0).any(-1)))
        state["map"] = s.residency.host.copy()
        state["score"] = s.state["score"].cpu().clone()
    s.taps.append(tap)
    begin = s.begin

    def begun() -> None:                                       # the step's lent slots, while they exist
        begin()
        if s.lent is not None:
            state["lent"] = (s.lent.mode, s.lent.tail, int(s._step_loan))
    s.begin = begun

    def analyse(record: dict, pre: np.ndarray, score: torch.Tensor) -> dict:
        lent_call = s.lent_call.cpu().numpy()
        n_before = s.n_before.cpu().numpy().astype(np.int64)
        mode, tail, loan = state["lent"]
        routed = record["routed"][:, :]
        empties = np.zeros(len(classes), dtype=np.int64)
        for r in range(table.shape[1]):
            np.add.at(empties, table[:, r], n_before)
        tail_set = lent_call == 0
        res0 = torch.from_numpy(np.where((pre & ~tail_set)[..., None], 0, -1).repeat(table.shape[1], -1))
        cls_t = torch.from_numpy(table)
        tail_slots = np.zeros(len(classes), dtype=np.int64)
        if tail is not None:
            tail_slots[tail[0]] = tail[1]

        def rounds(needs: list[np.ndarray]) -> np.ndarray:
            """The policy's picks for these calls' needs, in turn, from the cache less the tail."""
            res, picked = res0.clone(), np.zeros((groups, experts), dtype=bool)
            for need in needs:
                p = reference.lend(need, res, cls_t, score).numpy()
                picked |= p
                res[torch.from_numpy(p)] = -1
            return picked

        actual = [lent_call == k for k in range(1, int(lent_call.max()) + 1)] if lent_call.max() >= 1 else []
        lent_rounds = np.zeros((groups, experts), dtype=bool)
        for x in actual:
            lent_rounds |= x
        replayed = rounds([record["lends"][k] for k in range(1, 1 + len(actual))])
        lent = tail_set | lent_rounds

        def held(sel: np.ndarray) -> np.ndarray:            # the slots serving ring / stage: empties, lent, less tail
            return empties + slots(tail_set | sel) - tail_slots

        def want(sel: np.ndarray, kind: str) -> np.ndarray:
            absent = ~(pre & ~tail_set & ~sel)
            if kind == "ring":                               # both halves: every absent piece streams
                return 2 * most(absent.sum(1))
            # staging: a layer's most distinct misses (its routes, known only as it runs), or its absent pieces (known
            # at the step's first resolve, where the lend is made); `parts`: the current part and the next
            misses = most((absent if kind.endswith("absent") else routed & absent).sum(1))
            return misses if kind.startswith("stage") else 2 * -(-misses // pipeline)
        out = {"mode": mode, "tail_class": None if tail is None else int(tail[0]), "tail_slots": int(tail_slots.sum()),
               "tail_unit_bytes": None if tail is None else int(unit[tail[0]]),
               "step_loan_bytes": loan, "empties": empties.tolist(),
               "lent_experts": {"tail": int(tail_set[:target].sum()),
                                **{f"call{k + 1}": int(x[:target].sum()) for k, x in enumerate(actual)}},
               "replay_matches": bool((replayed == lent_rounds).all()),
               "needs": [record["lends"][k].tolist() for k in range(len(record["lends"]))],
               "held": held(lent_rounds).tolist(), "lending_classes": lending.tolist(),
               "want_now": {k: want(lent_rounds, k).tolist() for k in ("ring", "stage", "stage_absent", "parts")}}
        if tail is not None and loan:
            out["tail_split"] = {"loan_share": round(loan / (tail[1] * unit[tail[0]]), 3)}
        sets = {"lent": lent}
        kinds = ["stage", "stage_absent"] + (["parts", "parts_absent"] if pipeline > 1 else [])
        for kind in (["ring"] if mode == "ring" else []) + kinds:
            sel = np.zeros((groups, experts), dtype=bool)
            for _ in range(8):                               # the least lend: its misses, then its picks, to a fix
                need = np.where(lending, np.maximum(want(sel, kind) - held(np.zeros_like(sel)), 0), 0)
                nxt = rounds([need])
                if (nxt == sel).all():
                    break
                sel = nxt
            sets[f"min_{kind}"] = tail_set | sel
            out[f"min_{kind}"] = {"lent_experts": int((tail_set | sel)[:target].sum()), "need": need.tolist()}
        gb = lambda held_: round(float((nbytes[:target] * (routed[:target] & ~held_[:target])).sum()) / 1e9, 2)  # noqa: E731
        out["missed_gb"] = {"met": gb(pre), **{k: gb(pre & ~v) for k, v in sets.items()}}
        out["_sets"] = sets
        return out

    def after(found: dict) -> dict:
        """The decode / verify rows' missed routes after the prompt; those an expert lent but not at a minimum took
        (until it is back), per minimum."""
        out = {}
        for k, v in found["_sets"].items():
            if k == "lent":
                continue
            spared = found["_sets"]["lent"] & ~v
            row = {}
            for name, n in (("window", a.window), ("all", len(steps))):
                gone, saved, missed = spared.copy(), 0, 0
                for m, absent in steps[:n]:
                    missed += int(m[:target].sum())
                    saved += int(m[:target][gone[:target]].sum())
                    gone &= absent
                row[name] = {"missed_routes": missed, "spared_routes": saved,
                             "share": round(saved / max(missed, 1), 3)}
            out[k] = row
        return out

    warm = plan["warmup"]
    e.generate([warm["ids"]], SamplingParams(temperature=0.0, max_tokens=warm["output"], ignore_eos=True))
    rows = []
    last = max(i for i, r in enumerate(plan["requests"]) if r["key"].split("/")[0] in a.ctx)
    for r in plan["requests"][:last + 1]:
        found.clear()
        steps.clear()
        state["prompt"] = r["key"]
        e.generate([r["ids"]], SamplingParams(temperature=0.0, max_tokens=OUTPUT, ignore_eos=True))
        if not found:
            rows.append({"key": r["key"], "borrowed": False})
            continue
        row = {"key": r["key"], **{k: v for k, v in found.items() if k != "_sets"}, "after": after(found)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    e.close()
    print(json.dumps({"pipeline": pipeline, "classes": [[c.name, int(c.unit_bytes)] for c in classes],
                      "prompts": rows}))


if __name__ == "__main__":
    main()
