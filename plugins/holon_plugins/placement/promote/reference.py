"""Oracles of placement.promote (F10, docs/dev/v1-offload.md D1, §13: Project Maya's promote-on-fetch): the contract
`moe_resolve.promote@1`, call by call, and its `lend`; and `replay`, the L1 policy-parity oracle over a whole trace, the
M6 replay's `promote_on_fetch` (spikes/x0_offload/m6/replay.py) in the device's terms.

`moe_resolve.promote@1` is `moe_resolve@1` (std.moe.kernels) plus the layer's state: `last [E] int32` (the step each
expert was last routed), `clock [1] int32` (the layer's step) and `spare [S, 1 + R] int32` (the layer's spare slots:
per entry its owner + 1 — the expert whose slots they were, unmapped since — then its slot in every role; a row of
zeros is absent); all zero at a seed, so the spares start absent and the first decode call's boundary sets them up.
A call with a routed (not dropped) entry of a `DECODE` row, in this order:

1. Stamp: the clock advances by one; every expert a `DECODE` entry routes takes it as `last`.
2. Promote: candidates are experts routed by a `DECODE` row and missing in every role of the map (a ring piece counts
   as missing), in the order of their first `DECODE` entry (token, then rank); the first min(candidates, the layer's
   empty slots + its spares) are promoted. A promoted owner of a spare takes its own spare back (no record: the host
   still maps it there); the others, in order, take the layer's empty slots from the top (record `(group, a, −2 − j)`),
   then the spares no promoted expert owns, in entry order (record `(group, a, o)`, o the spare's owner: a took o's
   slots). A spare taken is absent.
3. Resolve: `moe_resolve@1` over the map after the promotions — a promoted expert is resident, so its entries read
   its new slot; the CPU's share (F5b) is taken from the misses left —; then each promoted expert but an owner is
   fetched straight into its slots, its fetch entries after the call's staged misses (staged misses by expert id, then
   the promoted in promotion order; `n_fetch` and `misses` count both). An owner's bytes are still in its spare: no
   slot of a spare is written while it is one (it is in no ring, stage or empty list), so taking it back costs no copy.
4. Refill (the step boundary): each absent spare, in entry order, takes the layer's least recently routed resident
   (every role in cache slots, by `last` then expert id; the call's own routes and promotions included, as in M6's
   replay) while more than one resident is left, and none while the layer holds every expert (M6: a layer keeps
   min(S, its slots − 1) spares, a complete one none): it is unmapped (no record) and its slots become the spare,
   owned by it. A layer whose residents were lent away refills later, as promotions fill its returned empty slots.

A call without a `DECODE` entry does what @1 does and nothing else. The host mirror maps a spare's owner at its spare's
slots until a record moves them (the delta log has no record for a slot the device keeps in its own region); `lend`
never picks an owner (it is not resident).

`lend(need, residency, classes, last, clock, spare)` picks whole resident experts least recently routed first,
ties by expert id, then layer: for each class c the first `need[c]` of them holding a slot of c (an expert's role
slots go together)."""
from __future__ import annotations

import numpy as np
import torch

PADDING, DECODE = 0, 1           # row kinds (`moe_resolve@1`)
RANGE, OVERFLOW = 1, 2           # error bits (`moe_resolve@1` and its variants)
SPARES = 3                       # spare slots per layer (Maya G1-G3, §13)


def moe_resolve_promote(ids: torch.Tensor, kinds: torch.Tensor, residency: torch.Tensor, host: torch.Tensor,
                        limits: torch.Tensor, staging: int, last: torch.Tensor, clock: torch.Tensor,
                        spare: torch.Tensor, group: int, ring: torch.Tensor | None = None,
                        stage: torch.Tensor | None = None, empty: torch.Tensor | None = None, n_empty: int = 0,
                        plan: torch.Tensor | None = None) -> dict:
    """One call: `slots [R, T, k]`, `fetch [R, 2, F]` (−1 past n_fetch), `n_fetch [R]`, `counts [E]` (this call's
    increments), `routed [E]` (the experts it saw routed), `errors` (bits), and the state after it: `residency`,
    `last`, `clock`, `spare`, `n_empty`, the delta `records`. `ring` defaults to none, `stage` to the staging
    slots k + i, `empty` to none; and the CPU's share (`plan`, default none; `cpu_share`), its `cpu [T, k]` and job
    rows `jobs`."""
    t, k = ids.shape
    experts, roles = residency.shape
    ids, kinds = ids.long().cpu(), kinds.long().cpu()
    after, host = residency.long().cpu().clone(), host.long().cpu()
    ring = torch.full((experts, roles), -1) if ring is None else ring.long().cpu()
    stage = (k + torch.arange(staging)).expand(roles, staging) if stage is None else stage.long().cpu()
    empty = torch.zeros(roles, 0, dtype=torch.long) if empty is None else empty.long().cpu()
    last, clock = last.cpu().clone(), clock.cpu().clone()
    spare = spare.cpu().clone()
    live = (kinds[:, None] != PADDING) & (ids >= 0)
    decode = live & (kinds[:, None] == DECODE)
    routed = ids.clamp_min(0)
    counts = torch.bincount(routed[decode], minlength=experts)
    needed = set(routed[live].tolist())
    seen = torch.zeros(experts, dtype=torch.int32)
    seen[sorted(needed)] = 1

    def cache(e: int) -> bool:
        return all(k + staging <= int(after[e, r]) < int(limits[r]) for r in range(roles))

    records, fetched = [], []
    if decode.any():
        seen[sorted(needed)] |= 2                                       # the victims' scope
        clock += 1                                                      # 1. stamp
        last[counts > 0] = clock
        first = {}
        for a, j in decode.nonzero().tolist():                          # entries in order: token, then rank
            first.setdefault(int(routed[a, j]), a * k + j)
        present = [s for s in range(len(spare)) if int(spare[s, 0]) > 0]   # 2. promote
        cands = sorted((e for e in first if (after[e] < 0).all()), key=first.get)
        promoted = cands[:n_empty + len(present)]
        owner = {int(spare[s, 0]) - 1: s for s in present}
        used = {owner[a] for a in promoted if a in owner}
        for a in promoted:
            if a in owner:
                after[a] = spare[owner[a], 1:]
        free = [s for s in present if s not in used]
        for a in (a for a in promoted if a not in owner):
            if n_empty:
                n_empty -= 1
                after[a] = empty[:, n_empty]
                records.append((group, a, -2 - n_empty))
            else:
                s = free.pop(0)
                used.add(s)
                after[a] = spare[s, 1:]
                records.append((group, a, int(spare[s, 0]) - 1))
        for s in used:
            spare[s] = 0
        fetched = [a for a in promoted if a not in owner]
    taken, cpu, jobs = cpu_share(ids, kinds, after, ring, host, plan)   # 3. @1 over the map after promotion
    seen[sorted(taken)] |= 4                                            # the CPU's
    slots = torch.arange(k).expand(roles, t, k).clone()                 # null slot j by default
    width = stage.shape[1]
    fetch = torch.full((roles, 2, width), -1, dtype=torch.int64)
    n_fetch = torch.zeros(roles, dtype=torch.int64)
    errors = 0
    for r in range(roles):
        missing = sorted({e for e in routed[live].tolist() if after[e, r] < 0 and ring[e, r] < 0} - taken)
        staged = {e: i for i, e in enumerate(missing)}
        listed = [(e, int(stage[r, min(i, width - 1)])) for i, e in enumerate(missing)]   # past F: @1's last slot
        listed += [(a, int(after[a, r])) for a in fetched]
        n_fetch[r] = len(listed)
        for i, (e, slot) in enumerate(listed[:width]):
            fetch[r, :, i] = torch.tensor([host[e, r] if host[e, r] >= 0 else -2 - e, slot])  # (−2 − e: on no
        for a, j in live.nonzero().tolist():                                                    # read tier)
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
    if decode.any():                                                    # 4. refill at the boundary
        absent = [s for s in range(len(spare)) if int(spare[s, 0]) == 0]
        victims = sorted((e for e in range(experts) if cache(e)), key=lambda e: (int(last[e]), e))
        victims = victims[:0 if len(victims) == experts else len(victims) - 1]
        for s, v in zip(absent, victims, strict=False):
            spare[s] = torch.tensor([v + 1, *after[v].tolist()])
            after[v] = -1
    return {"slots": slots.int(), "fetch": fetch.int(), "n_fetch": n_fetch.int(), "counts": counts.int(),
            "routed": seen, "errors": errors, "residency": after.int(), "last": last, "clock": clock,
            "spare": spare, "n_empty": n_empty, "records": records, "cpu": cpu, "jobs": jobs}


def cpu_share(ids: torch.Tensor, kinds: torch.Tensor, residency: torch.Tensor, ring: torch.Tensor,
              host: torch.Tensor, plan: torch.Tensor | None) -> tuple[set[int], torch.Tensor, list[list[int]]]:
    """`moe_resolve@1`'s CPU share (F5b, std.moe.kernels), here over the map after the promotions: of the call's n
    candidates — routed, missing in every role, in the read host tier (−1) — the last min(plan[n], n) by (best routing
    rank, expert id); `cpu [T, k]` (t·k + j for their entries, else −1); the job rows by expert id."""
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


def lend(need, residency: torch.Tensor, classes: torch.Tensor, last: torch.Tensor, clock: torch.Tensor,
         spare: torch.Tensor) -> torch.Tensor:
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


def replay(seed: np.ndarray, steps: list[dict], classes: np.ndarray | None = None, spares: int = SPARES) -> list[dict]:
    """The L1 parity oracle: F10's residency [G, E] (whole experts) and empty slots per layer [G] after each step of a
    trace, from the seeded residency. A step holds, per layer, the experts its `DECODE` entries route in the order of
    their first entry (`order`: G lists, empty for a layer without decode rows), and `lends` (None: the step borrowed
    no slots) as placement.lfu's `replay` takes them: a borrowing step takes every empty slot at its start, then makes
    its `lend(need)` calls (`classes` [G, R]); all those slots return empty to their layers at its end.
    The M6 replay's `promote_on_fetch` — misses in route order take the spares, the boundary evicts the least recently
    routed, one per spare taken, keeping a resident, none in a complete layer — with the device's terms: the spares set
    up at a layer's first decode step's boundary (M6 set them up at the seed from the profile's coldest), empty slots
    (taken before other spares), and spares returning to their owners."""
    res = seed.astype(bool).copy()
    groups, experts = res.shape
    last = np.zeros(res.shape, dtype=np.int32)
    clock = np.zeros(groups, dtype=np.int32)
    owners = [[-1] * spares for _ in range(groups)]       # per layer: each spare's owner (−1: absent)
    empty = np.zeros(groups, dtype=np.int64)
    out = []
    for step in steps:
        lent = np.zeros(groups, dtype=np.int64)
        if step.get("lends") is not None:                              # the empty slots, borrowed first
            lent, empty = empty, np.zeros_like(empty)
        for need in () if step.get("lends") is None else step["lends"]:
            whole = torch.from_numpy(np.where(res, 0, -1)[..., None].repeat(classes.shape[1], -1))
            picked = lend(need, whole, torch.from_numpy(classes), torch.from_numpy(last), None, None).numpy()
            res &= ~picked
            lent += picked.sum(1)
        for g, order in enumerate(step["order"]):
            if not len(order):
                continue
            clock[g] += 1
            last[g, list(order)] = clock[g]
            present = [s for s, o in enumerate(owners[g]) if o >= 0]
            promoted = [e for e in order if not res[g, e]][:empty[g] + len(present)]
            used = {owners[g].index(a) for a in promoted if a in owners[g]}
            free = [s for s in present if s not in used]
            for a in promoted:
                if a not in owners[g]:
                    if empty[g]:
                        empty[g] -= 1
                    else:
                        used.add(free.pop(0))
            res[g, promoted] = True
            for s in used:
                owners[g][s] = -1
            absent = [s for s, o in enumerate(owners[g]) if o < 0]
            victims = sorted(np.flatnonzero(res[g]), key=lambda e: (last[g, e], e))
            victims = victims[:0 if len(victims) == experts else len(victims) - 1]
            for s, v in zip(absent, victims, strict=False):
                owners[g][s] = int(v)
                res[g, v] = False
        empty += lent
        out.append({"resident": res.copy(), "empty": empty.copy()})
    return out
