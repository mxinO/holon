"""Oracles of placement.lru (F4, docs/v1-offload.md D1: FreeToken's and llama.cpp's cache, as the M6 replay's `lru`,
spikes/x0_offload/m6/replay.py): the contract `moe_resolve.lru@1`, call by call, and its `lend`.

`moe_resolve.lru@1` is `moe_resolve@1` (std.moe.kernels) plus the layer's logical clock `clock [1] int32` and the
step each expert was last routed `last [E] int32` (state, zero at a seed). A call with a `DECODE` row advances the
clock by one and stamps every expert a `DECODE` row routes with it; then its candidates — experts routed by a `DECODE`
row and missing in every role, in the order of their first `DECODE` entry (token, then rank) — take the slots of its
victims — experts resident in every role (in cache slots) routed by no live entry of the call, least recently routed
first, ties by expert id —, pair by pair while both last. An admitted expert takes its victim's slot in every role
(the victim's entries become −1: the RAM tier holds every expert), is logged `(group, admitted, victim)` in pairing
order and gathered straight into those slots; the call's other misses are staged as @1's. A call without `DECODE`
rows changes nothing but what @1 does.

`lend(n, residency, classes, last, clock)` picks whole resident experts, least recently routed first, ties by layer,
then expert id, until every class has min(n, its resident slots) lent slots."""
from __future__ import annotations

import numpy as np
import torch

PADDING, DECODE = 0, 1           # row kinds (`moe_resolve@1`)
RANGE, OVERFLOW = 1, 2           # error bits (`moe_resolve@1` and its variants)


def moe_resolve_lru(ids: torch.Tensor, kinds: torch.Tensor, residency: torch.Tensor, host: torch.Tensor,
                    limits: torch.Tensor, staging: int, last: torch.Tensor, clock: torch.Tensor, group: int) -> dict:
    """One call: `slots [R, T, k]`, `fetch [R, 2, staging]` (−1 past n_fetch), `n_fetch [R]`, `counts [E]` (this call's
    increments), `errors` (bits), and the state after it: `residency`, `last`, `clock`, the delta `records`."""
    t, k = ids.shape
    experts, roles = residency.shape
    ids, kinds = ids.long().cpu(), kinds.long().cpu()
    before, host = residency.long().cpu(), host.long().cpu()
    after, last, clock = before.clone(), last.cpu().clone(), clock.cpu().clone()
    live = (kinds[:, None] != PADDING) & (ids >= 0)
    decode = live & (kinds[:, None] == DECODE)
    routed = ids.clamp_min(0)
    counts = torch.bincount(routed[decode], minlength=experts)
    needed = set(routed[live].tolist())
    records = []
    if decode.any():
        clock += 1
        last[counts > 0] = clock
        first = {}
        for a, j in decode.nonzero().tolist():                         # entries in order: token, then rank
            first.setdefault(int(routed[a, j]), a * k + j)
        cache = [all(k + staging <= int(before[e, r]) < int(limits[r]) for r in range(roles)) for e in range(experts)]
        cands = sorted((e for e in first if (before[e] < 0).all()), key=first.get)
        victims = sorted((e for e in range(experts) if cache[e] and e not in needed), key=lambda e: (int(last[e]), e))
        for a, b in zip(cands, victims, strict=False):
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
            "errors": errors, "residency": after.int(), "last": last, "clock": clock, "records": records}


def lend(n: int, residency: torch.Tensor, classes: torch.Tensor, last: torch.Tensor,
         clock: torch.Tensor) -> torch.Tensor:
    """`picked [G, E] bool` (see the module's text)."""
    groups, experts, roles = residency.shape
    whole = (residency >= 0).all(-1)
    order = sorted((int(last[g, e]), g, e) for g in range(groups) for e in range(experts) if whole[g, e])
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
