"""Oracles of placement.lfu (F11, docs/v1-offload.md D4): the contract `moe_resolve.lfu@1`, call by call, and its
`lend`; and `replay`, the L1 policy-parity oracle over a whole trace, the M6 replay's `lfu`
(spikes/x0_offload/m6/replay.py) extended to the device's terms — f32, decay on steps with decode rows, the device's
tie-breaks, row kinds, lending and the lent slots' return.

`moe_resolve.lfu@1` is `moe_resolve@1` (std.moe.kernels) plus the layer's decayed counts `score [E] f32` (state, zero
at a seed). A call with a routed (not dropped) entry of a `DECODE` row first decays every score (× 0.915 in f32), then
adds 1.0 (in f32, one add per entry) per entry of a `DECODE` row; then it ranks the call's candidates — experts routed
by a `DECODE` row and missing in every role of the map (a ring piece included) — by score descending, ties by expert
id. The first ones take the layer's empty slots, unconditionally, from the top of its stack (an empty slot is the
layer's first victim, below every resident); the rest pair with its victims — experts resident in every role (in cache
slots) routed by no live entry of the call — by score ascending, ties by expert id, and are admitted while the
candidate's score is strictly greater than its victim's (the first failing pair stops the pairing: the layer keeps its
slots). An admitted expert takes its victim's (or empty) slot in every role (a victim's entries become −1: the RAM tier
holds every expert under this policy, D10), is logged in that order, and is gathered straight into its slots; the
call's other misses are fetched as @1's. A call without such an entry (a prefill chunk) changes nothing but what @1
does.

`lend(n, residency, classes, score)` picks whole resident experts, coldest by score first, ties by layer, then expert
id, until every class has min(n, its resident slots) lent slots (an expert's role slots go together)."""
from __future__ import annotations

import numpy as np
import torch

PADDING, DECODE = 0, 1           # row kinds (`moe_resolve@1`)
RANGE, OVERFLOW = 1, 2           # error bits (`moe_resolve@1` and its variants)
DECAY = np.float32(0.915)


def moe_resolve_lfu(ids: torch.Tensor, kinds: torch.Tensor, residency: torch.Tensor, host: torch.Tensor,
                    limits: torch.Tensor, staging: int, score: torch.Tensor, group: int,
                    ring: torch.Tensor | None = None, stage: torch.Tensor | None = None,
                    empty: torch.Tensor | None = None, n_empty: int = 0) -> dict:
    """One call: `slots [R, T, k]`, `fetch [R, 2, F]` (−1 past n_fetch), `n_fetch [R]`, `counts [E]` (this call's
    increments), `routed [E]` (the experts it saw routed), `errors` (bits), and the state after it: `residency`,
    `score`, `n_empty`, the delta `records`. `ring` defaults to none, `stage` to the staging slots k + i, `empty` to
    none."""
    t, k = ids.shape
    experts, roles = residency.shape
    ids, kinds = ids.long().cpu(), kinds.long().cpu()
    before, host = residency.long().cpu(), host.long().cpu()
    ring = torch.full((experts, roles), -1) if ring is None else ring.long().cpu()
    stage = (k + torch.arange(staging)).expand(roles, staging) if stage is None else stage.long().cpu()
    empty = torch.zeros(roles, 0, dtype=torch.long) if empty is None else empty.long().cpu()
    after, score = before.clone(), score.cpu().numpy().astype(np.float32).copy()
    live = (kinds[:, None] != PADDING) & (ids >= 0)
    routed = ids.clamp_min(0)
    counts = torch.bincount(routed[live & (kinds[:, None] == DECODE)], minlength=experts)
    needed = set(routed[live].tolist())
    seen = torch.zeros(experts, dtype=torch.int32)
    seen[sorted(needed)] = 1
    records = []
    if (counts > 0).any():
        seen[sorted(needed)] |= 2                                       # the victims' scope
        for e in range(experts):
            s = score[e] * DECAY
            for _ in range(int(counts[e])):
                s = s + np.float32(1.0)
            score[e] = s
        cache = [all(k + staging <= int(before[e, r]) < int(limits[r]) for r in range(roles)) for e in range(experts)]
        cands = sorted((e for e in range(experts) if counts[e] and (before[e] < 0).all()), key=lambda e: (-score[e], e))
        victims = sorted((e for e in range(experts) if cache[e] and e not in needed), key=lambda e: (score[e], e))
        spare = min(len(cands), n_empty)
        for i, a in enumerate(cands[:spare]):                          # the empty slots first, from the top
            after[a] = empty[:, n_empty - 1 - i]
            records.append((group, a, -2 - (n_empty - 1 - i)))
        n_empty -= spare
        for a, b in zip(cands[spare:], victims, strict=False):
            if not score[a] > score[b]:
                break
            after[a], after[b] = before[b], -1
            records.append((group, a, b))
    admitted = {a for _, a, _ in records}
    slots = torch.arange(k).expand(roles, t, k).clone()                 # null slot j by default
    width = stage.shape[1]
    fetch = torch.full((roles, 2, width), -1, dtype=torch.int64)
    n_fetch = torch.zeros(roles, dtype=torch.int64)
    errors = 0
    for r in range(roles):
        missing = sorted({e for e in routed[live].tolist()
                          if before[e, r] < 0 and (ring[e, r] < 0 or e in admitted)})
        staged = {e: i for i, e in enumerate(e for e in missing if e not in admitted)}
        n_fetch[r] = len(missing)
        for i, e in enumerate(missing[:width]):
            fetch[r, :, i] = torch.tensor([host[e, r], after[e, r] if e in admitted else stage[r, staged[e]]])
        for a, j in live.nonzero().tolist():
            e = int(routed[a, j])
            if e in staged:
                slots[r, a, j] = stage[r, min(staged[e], width - 1)]
                continue
            at = int(after[e, r]) if after[e, r] >= 0 else int(ring[e, r])
            if k + staging <= at < int(limits[r]):
                slots[r, a, j] = at
            else:                                                       # corrupt: null slot j, flagged
                errors |= RANGE
    return {"slots": slots.int(), "fetch": fetch.int(), "n_fetch": n_fetch.int(), "counts": counts.int(),
            "routed": seen, "errors": errors, "residency": after.int(), "score": torch.from_numpy(score),
            "n_empty": n_empty, "records": records}


def lend(n: int, residency: torch.Tensor, classes: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
    """`picked [G, E] bool` (see the module's text)."""
    groups, experts, roles = residency.shape
    whole = (residency >= 0).all(-1)
    order = sorted((float(score[g, e]), g, e) for g in range(groups) for e in range(experts) if whole[g, e])
    held = np.zeros(int(classes.max()) + 1, dtype=np.int64)
    for _, g, _e in order:
        for r in range(roles):
            held[int(classes[g, r])] += 1
    need, lent = np.minimum(held, n), np.zeros_like(held)
    picked = torch.zeros(groups, experts, dtype=torch.bool)
    for _, g, e in order:
        if (lent >= need).all():
            break
        picked[g, e] = True
        for r in range(roles):
            lent[int(classes[g, r])] += 1
    return picked


def replay(seed: np.ndarray, steps: list[dict], score: np.ndarray | None = None,
           classes: np.ndarray | None = None) -> list[dict]:
    """The L1 parity oracle: F11's residency [G, E] (whole experts) and empty slots per layer [G] after each step of
    a trace, from the seeded residency (and scores, zero at a seed). A step holds its routing `counts` [G, E] of decode
    rows, the experts any of its rows routed (`needed` [G, E]; default: the counts'), and `lends` (None: the step
    borrowed no slots): a borrowing step takes every empty slot at its start, then makes its `lend(n)` calls, each of
    the coldest whole experts by score (ties by layer, then expert id) until each class (`classes` [G, R]) has min(n,
    its resident slots) more; all those slots return empty to their layers at the step's end.
    The M6 replay's `lfu` (`simulate`, `_pair_swap` with gain 0), with the scores in f32 (× 0.915, then + 1.0 per
    entry), decayed only on steps with decode rows, empty slots filled first."""
    res = seed.astype(bool).copy()
    u = np.zeros(seed.shape, dtype=np.float32) if score is None else score.astype(np.float32).copy()
    empty = np.zeros(len(seed), dtype=np.int64)
    out = []
    for step in steps:
        c, lent = step["counts"], np.zeros(len(seed), dtype=np.int64)
        if step.get("lends") is not None:                              # the empty slots, borrowed first
            lent, empty = empty, np.zeros_like(empty)
        for n in step.get("lends") or ():
            whole = torch.from_numpy(np.where(res, 0, -1)[..., None].repeat(classes.shape[1], -1))
            picked = lend(n, whole, torch.from_numpy(classes), torch.from_numpy(u)).numpy()
            res &= ~picked
            lent += picked.sum(1)
        if c.any():
            u = u * DECAY
            for n in range(int(c.max())):
                u = np.where(c > n, u + np.float32(1.0), u)
            need = step.get("needed", c > 0)
            for g in range(len(res)):
                cands = sorted(np.flatnonzero((c[g] > 0) & ~res[g]), key=lambda e: (-u[g, e], e))
                spare = min(len(cands), empty[g])
                res[g, cands[:spare]] = True
                empty[g] -= spare
                victims = sorted(np.flatnonzero(res[g] & ~need[g]), key=lambda e: (u[g, e], e))   # not the new ones
                for a, b in zip(cands[spare:], victims, strict=False):
                    if not u[g, a] > u[g, b]:
                        break
                    res[g, a], res[g, b] = True, False
        empty += lent
        out.append({"resident": res.copy(), "empty": empty.copy()})
    return out
