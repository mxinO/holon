"""Oracles of placement.lru (F4, docs/v1-offload.md D1: FreeToken's and llama.cpp's cache, as the M6 replay's `lru`,
spikes/x0_offload/m6/replay.py): the contract `moe_resolve.lru@1`, call by call, and its `lend`.

`moe_resolve.lru@1` is `moe_resolve@1` (std.moe.kernels) plus the layer's logical clock `clock [1] int32` and the step
each expert was last routed `last [E] int32` (state, zero at a seed). A call with a routed (not dropped) entry of a
`DECODE` row advances the clock by one and stamps every expert a `DECODE` row routes with it; then its candidates —
experts routed by a `DECODE` row and missing in every role of the map (a ring piece included), in the order of their
first `DECODE` entry (token, then rank) — take first the layer's empty slots, from the top of its stack, then the slots
of its victims — experts resident in every role (in cache slots) routed by no live entry of the call, least recently
routed first, ties by expert id —, pair by pair while both last. An admitted expert takes its victim's (or empty) slot
in every role (a victim's entries become −1: the RAM tier holds every expert), is logged in that order and gathered
straight into those slots; the call's other misses are fetched as @1's. A call without such an entry (a prefill chunk)
changes nothing but what @1 does.

`lend(need, residency, classes, last, clock)` picks whole resident experts in their order — least recently routed
first, ties by expert id, then layer —: for each class c the first `need[c]` of them holding a slot of c (an expert's
role slots go together)."""
from __future__ import annotations

import numpy as np
import torch

PADDING, DECODE = 0, 1           # row kinds (`moe_resolve@1`)
RANGE, OVERFLOW = 1, 2           # error bits (`moe_resolve@1` and its variants)


def moe_resolve_lru(ids: torch.Tensor, kinds: torch.Tensor, residency: torch.Tensor, host: torch.Tensor,
                    limits: torch.Tensor, staging: int, last: torch.Tensor, clock: torch.Tensor, group: int,
                    ring: torch.Tensor | None = None, stage: torch.Tensor | None = None,
                    empty: torch.Tensor | None = None, n_empty: int = 0, plan: torch.Tensor | None = None) -> dict:
    """One call: `slots [R, T, k]`, `fetch [R, 2, F]` (−1 past n_fetch), `n_fetch [R]`, `counts [E]` (this call's
    increments), `routed [E]` (the experts it saw routed), `errors` (bits), and the state after it: `residency`,
    `last`, `clock`, `n_empty`, the delta `records`. `ring` defaults to none, `stage` to the staging slots k + i,
    `empty` to none; and the CPU's share (`plan`, default none; `cpu_share`), its `cpu [T, k]` and job rows `jobs`."""
    t, k = ids.shape
    experts, roles = residency.shape
    ids, kinds = ids.long().cpu(), kinds.long().cpu()
    before, host = residency.long().cpu(), host.long().cpu()
    ring = torch.full((experts, roles), -1) if ring is None else ring.long().cpu()
    stage = (k + torch.arange(staging)).expand(roles, staging) if stage is None else stage.long().cpu()
    empty = torch.zeros(roles, 0, dtype=torch.long) if empty is None else empty.long().cpu()
    after, last, clock = before.clone(), last.cpu().clone(), clock.cpu().clone()
    live = (kinds[:, None] != PADDING) & (ids >= 0)
    decode = live & (kinds[:, None] == DECODE)
    routed = ids.clamp_min(0)
    taken, cpu, jobs = cpu_share(ids, kinds, before, ring, host, plan)
    counts = torch.bincount(routed[decode], minlength=experts)
    needed = set(routed[live].tolist())
    seen = torch.zeros(experts, dtype=torch.int32)
    seen[sorted(needed)] = 1
    records = []
    if decode.any():
        seen[sorted(needed)] |= 2                                       # the victims' scope
        clock += 1
        last[counts > 0] = clock
        first = {}
        for a, j in decode.nonzero().tolist():                         # entries in order: token, then rank
            first.setdefault(int(routed[a, j]), a * k + j)
        cache = [all(k + staging <= int(before[e, r]) < int(limits[r]) for r in range(roles)) for e in range(experts)]
        cands = sorted((e for e in first if (before[e] < 0).all() and e not in taken), key=first.get)
        victims = sorted((e for e in range(experts) if cache[e] and e not in needed), key=lambda e: (int(last[e]), e))
        spare = min(len(cands), n_empty)
        for i, a in enumerate(cands[:spare]):                          # the empty slots first, from the top
            after[a] = empty[:, n_empty - 1 - i]
            records.append((group, a, -2 - (n_empty - 1 - i)))
        n_empty -= spare
        for a, b in zip(cands[spare:], victims, strict=False):
            after[a], after[b] = before[b], -1
            records.append((group, a, b))
    seen[sorted(taken)] |= 4                                            # the CPU's
    admitted = {a for _, a, _ in records}
    slots = torch.arange(k).expand(roles, t, k).clone()                 # null slot j by default
    width = stage.shape[1]
    fetch = torch.full((roles, 2, width), -1, dtype=torch.int64)
    n_fetch = torch.zeros(roles, dtype=torch.int64)
    errors = 0
    for r in range(roles):
        missing = sorted({e for e in routed[live].tolist()
                          if before[e, r] < 0 and (ring[e, r] < 0 or e in admitted)} - taken)
        staged = {e: i for i, e in enumerate(e for e in missing if e not in admitted)}
        n_fetch[r] = len(missing)
        for i, e in enumerate(missing[:width]):
            fetch[r, :, i] = torch.tensor([host[e, r], after[e, r] if e in admitted else stage[r, staged[e]]])
        for a, j in live.nonzero().tolist():
            e = int(routed[a, j])
            if e in taken:
                continue                                                # the CPU's: null slot j
            if e in staged:
                slots[r, a, j] = stage[r, min(staged[e], width - 1)]
                continue
            at = int(after[e, r]) if after[e, r] >= 0 else int(ring[e, r])
            if k + staging <= at < int(limits[r]):
                slots[r, a, j] = at
            else:                                                       # corrupt: null slot j, flagged
                errors |= RANGE
    return {"slots": slots.int(), "fetch": fetch.int(), "n_fetch": n_fetch.int(), "counts": counts.int(),
            "routed": seen, "errors": errors, "residency": after.int(), "last": last, "clock": clock,
            "n_empty": n_empty, "records": records, "cpu": cpu, "jobs": jobs}


def cpu_share(ids: torch.Tensor, kinds: torch.Tensor, residency: torch.Tensor, ring: torch.Tensor,
              host: torch.Tensor, plan: torch.Tensor | None) -> tuple[set[int], torch.Tensor, list[list[int]]]:
    """`moe_resolve@1`'s CPU share (F5b, std.moe.kernels), which this variant keeps: of the call's n candidates —
    routed, missing in every role, in the read host tier (−1) — the last min(plan[n], n) by (best routing rank, expert
    id); `cpu [T, k]` (t·k + j for their entries, else −1); the job rows by expert id."""
    t, k = ids.shape
    live = (kinds[:, None] != PADDING) & (ids >= 0)
    best: dict[int, int] = {}
    for a, j in live.nonzero().tolist():
        best[int(ids[a, j])] = min(best.get(int(ids[a, j]), k), j)
    held = ((residency == -1) & (ring < 0)).all(-1)
    cands = sorted((best[e], e) for e in best if held[e])
    c = 0 if plan is None or not cands else min(int(plan[len(cands)]), len(cands))
    taken = {e for _, e in cands[len(cands) - c:]}
    cpu, jobs = torch.full((t, k), -1, dtype=torch.int32), []
    for e in sorted(taken):
        entries = [a * k + j for a, j in live.nonzero().tolist() if int(ids[a, j]) == e]
        cpu.view(-1)[entries] = torch.tensor(entries, dtype=torch.int32)
        jobs.append([e, int(host[e, 0]), int(host[e, 1]), len(entries), *entries])
    return taken, cpu, jobs


def lend(need, residency: torch.Tensor, classes: torch.Tensor, last: torch.Tensor,
         clock: torch.Tensor) -> torch.Tensor:
    """`picked [G, E] bool` (see the module's text)."""
    groups, experts, _ = residency.shape
    whole = (residency >= 0).all(-1)
    order = sorted((int(last[g, e]), e, g) for g in range(groups) for e in range(experts) if whole[g, e])
    need = np.asarray(need)
    picked, taken = torch.zeros(groups, experts, dtype=torch.bool), np.zeros(len(need), np.int64)
    for _, e, g in order:
        for c in sorted({int(c) for c in classes[g]}):
            if taken[c] < need[c]:
                picked[g, e] = True
            taken[c] += 1
    return picked
