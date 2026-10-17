"""Oracles of placement.lfu (F11, docs/v1-offload.md D4): the contract `moe_resolve.lfu@1`, call by call, and its
`lend`; and `replay`, the L1 policy-parity oracle over a whole trace, the M6 replay's `lfu`
(spikes/x0_offload/m6/replay.py) extended to the device's terms — f32, decay on steps with decode rows, the device's
tie-breaks.

`moe_resolve.lfu@1` is `moe_resolve@1` (std.moe.kernels) plus the layer's decayed counts `score [E] f32` (state, zero at
a seed). A call with a routed (not dropped) entry of a `DECODE` row first decays every score (× 0.915 in f32), then adds
1.0 (in f32, one add per entry) per entry of a `DECODE` row; then it pairs the call's candidates — experts routed by a
`DECODE` row and missing in every role — by score descending with its victims — experts resident in every role (in cache
slots) routed by no live entry of the call — by score ascending, ties by expert id; a candidate is admitted while its
score is strictly greater than its victim's, and the first failing pair stops the pairing (the layer keeps its slots).
An admitted expert takes its victim's slot in every role (the victim's entries become −1: the RAM tier holds every
expert under this policy, D10), is logged `(group, admitted, victim)` in pairing order, and is gathered straight into
those slots; the call's other misses are staged as @1's. A call without such an entry (a prefill chunk) changes nothing
but what @1 does.

`lend(n, residency, classes, score)` picks whole resident experts, coldest by score first, ties by layer, then expert
id, until every class has min(n, its resident slots) lent slots (an expert's role slots go together)."""
from __future__ import annotations

import numpy as np
import torch

PADDING, DECODE = 0, 1           # row kinds (`moe_resolve@1`)
RANGE, OVERFLOW = 1, 2           # error bits (`moe_resolve@1` and its variants)
DECAY = np.float32(0.915)


def moe_resolve_lfu(ids: torch.Tensor, kinds: torch.Tensor, residency: torch.Tensor, host: torch.Tensor,
                    limits: torch.Tensor, staging: int, score: torch.Tensor, group: int) -> dict:
    """One call: `slots [R, T, k]`, `fetch [R, 2, staging]` (−1 past n_fetch), `n_fetch [R]`, `counts [E]` (this call's
    increments), `errors` (bits), and the state after it: `residency`, `score`, the delta `records`."""
    t, k = ids.shape
    experts, roles = residency.shape
    ids, kinds = ids.long().cpu(), kinds.long().cpu()
    before, host = residency.long().cpu(), host.long().cpu()
    after, score = before.clone(), score.cpu().numpy().astype(np.float32).copy()
    live = (kinds[:, None] != PADDING) & (ids >= 0)
    routed = ids.clamp_min(0)
    counts = torch.bincount(routed[live & (kinds[:, None] == DECODE)], minlength=experts)
    needed = set(routed[live].tolist())
    records = []
    if (counts > 0).any():
        for e in range(experts):
            s = score[e] * DECAY
            for _ in range(int(counts[e])):
                s = s + np.float32(1.0)
            score[e] = s
        cache = [all(k + staging <= int(before[e, r]) < int(limits[r]) for r in range(roles)) for e in range(experts)]
        cands = sorted((e for e in range(experts) if counts[e] and (before[e] < 0).all()), key=lambda e: (-score[e], e))
        victims = sorted((e for e in range(experts) if cache[e] and e not in needed), key=lambda e: (score[e], e))
        for a, b in zip(cands, victims, strict=False):
            if not score[a] > score[b]:
                break
            after[a], after[b] = before[b], -1
            records.append((group, a, b))
    admitted = {a for _, a, _ in records}
    slots = torch.arange(k).expand(roles, t, k).clone()                 # null slot j by default
    fetch = torch.full((roles, 2, staging), -1, dtype=torch.int64)
    n_fetch = torch.zeros(roles, dtype=torch.int64)
    errors = 0
    for r in range(roles):
        missing = sorted({e for e in routed[live].tolist() if before[e, r] < 0})
        staged = {e: i for i, e in enumerate(e for e in missing if e not in admitted)}
        n_fetch[r] = len(missing)
        for i, e in enumerate(missing[:staging]):
            fetch[r, :, i] = torch.tensor([host[e, r], after[e, r] if e in admitted else k + staged[e]])
        for a, j in live.nonzero().tolist():
            e = int(routed[a, j])
            at = int(after[e, r])
            if at < 0:
                slots[r, a, j] = k + min(staged[e], staging - 1)
            elif k + staging <= at < int(limits[r]):
                slots[r, a, j] = at
            else:                                                       # corrupt: null slot j, flagged
                errors |= RANGE
    return {"slots": slots.int(), "fetch": fetch.int(), "n_fetch": n_fetch.int(), "counts": counts.int(),
            "errors": errors, "residency": after.int(), "score": torch.from_numpy(score), "records": records}


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


def replay(seed: np.ndarray, counts: list[np.ndarray], needed: list[np.ndarray] | None = None) -> list[np.ndarray]:
    """The L1 parity oracle: F11's residency [G, E] (whole experts) after each step of a trace, from the seeded
    residency and each step's routing counts [G, E] of decode rows (and the experts any of its rows routed to; by
    default those counts' — a decode-only trace). The M6 replay's `lfu` (`simulate`, `_pair_swap` with gain 0), with
    the scores in f32 (× 0.915, then + 1.0 per entry), decayed only on steps with decode rows."""
    res, u = seed.astype(bool).copy(), np.zeros(seed.shape, dtype=np.float32)
    out = []
    for s, c in enumerate(counts):
        if c.any():
            u = u * DECAY
            for n in range(int(c.max())):
                u = np.where(c > n, u + np.float32(1.0), u)
            need = (c > 0) if needed is None else needed[s]
            cand = np.where((c > 0) & ~res, -u, np.inf)                 # by count descending, ties by expert id
            corder = np.argsort(cand, axis=1, kind="stable")
            cvals = -np.take_along_axis(cand, corder, 1)
            vict = np.where(res & ~need, u, np.inf)                     # by count ascending, ties by expert id
            vorder = np.argsort(vict, axis=1, kind="stable")
            vvals = np.take_along_axis(vict, vorder, 1)
            ok = np.cumprod(np.isfinite(cvals) & np.isfinite(vvals) & (cvals > vvals), axis=1).astype(bool)
            rows = np.nonzero(ok)[0]
            res[rows, vorder[ok]], res[rows, corder[ok]] = False, True
        out.append(res.copy())
    return out
