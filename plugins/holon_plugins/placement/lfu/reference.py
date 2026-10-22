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

`lend(need, residency, classes, score)` picks whole resident experts in their order — coldest by score first, ties by
expert id, then layer —: for each class c the first `need[c]` of them holding a slot of c (an expert's role slots go
together, so one picked for a class lends its slots of the others too). `lend_tail(cls, n, residency, classes,
limits)` picks the experts holding any of class `cls`'s last n slots (docs/v1-moe-prefill.md D11; F11 keeps no spares
to release there)."""
from __future__ import annotations

import numpy as np
import torch

PADDING, DECODE = 0, 1           # row kinds (`moe_resolve@1`)
RANGE, OVERFLOW = 1, 2           # error bits (`moe_resolve@1` and its variants)
DECAY = np.float32(0.915)


def moe_resolve_lfu(ids: torch.Tensor, kinds: torch.Tensor, residency: torch.Tensor, host: torch.Tensor,
                    limits: torch.Tensor, staging: int, score: torch.Tensor, group: int,
                    ring: torch.Tensor | None = None, stage: torch.Tensor | None = None,
                    empty: torch.Tensor | None = None, n_empty: int = 0, plan: torch.Tensor | None = None,
                    defer: bool = False, budget: int | None = None, gain: float = 0.0) -> dict:
    """One call: `slots [R, T, k]`, `fetch [R, 2, F]` (−1 past n_fetch), `n_fetch [R]`, `counts [E]` (this call's
    increments), `routed [E]` (the experts it saw routed), `errors` (bits), and the state after it: `residency`,
    `score`, `n_empty`, the delta `records`. `ring` defaults to none, `stage` to the staging slots k + i, `empty` to
    none; and the CPU's share (`plan`, default none; `cpu_share`), its `cpu [T, k]` and job rows `jobs`. With `defer`,
    the CPU's experts are candidates too: an admitted one's pieces are listed apart, `later [R, 2, F]` and `n_later [R]`
    (fetched after the call: the CPU serves its entries), never in `fetch`. `budget` caps the call's admissions (the
    empty slots taken included; placement.budget's), and `gain` is the margin a candidate's score must pass its
    victim's by (f32; 0: strictly greater)."""
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
    taken, cpu, jobs = cpu_share(ids, kinds, before, ring, host, plan)
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
        cands = sorted((e for e in range(experts) if counts[e] and (before[e] < 0).all() and (defer or e not in taken)),
                       key=lambda e: (-score[e], e))
        victims = sorted((e for e in range(experts) if cache[e] and e not in needed), key=lambda e: (score[e], e))
        cap = len(cands) if budget is None else budget
        spare = min(len(cands), n_empty, cap)
        for i, a in enumerate(cands[:spare]):                          # the empty slots first, from the top
            after[a] = empty[:, n_empty - 1 - i]
            records.append((group, a, -2 - (n_empty - 1 - i)))
        n_empty -= spare
        for a, b in list(zip(cands[spare:], victims, strict=False))[:max(cap - spare, 0)]:
            if not score[a] > score[b] + np.float32(gain):
                break
            after[a], after[b] = before[b], -1
            records.append((group, a, b))
    seen[sorted(taken)] |= 4                                            # the CPU's
    admitted = {a for _, a, _ in records}
    slots = torch.arange(k).expand(roles, t, k).clone()                 # null slot j by default
    width = stage.shape[1]
    fetch = torch.full((roles, 2, width), -1, dtype=torch.int64)
    n_fetch = torch.zeros(roles, dtype=torch.int64)
    errors = 0
    later = torch.full((roles, 2, width), -1, dtype=torch.int64)
    n_later = torch.zeros(roles, dtype=torch.int64)
    for r in range(roles):
        deferred = sorted(taken & admitted)
        n_later[r] = len(deferred)
        for i, e in enumerate(deferred[:width]):
            later[r, :, i] = torch.tensor([host[e, r] if host[e, r] >= 0 else -2 - e, after[e, r]])
        missing = sorted({e for e in routed[live].tolist()
                          if before[e, r] < 0 and (ring[e, r] < 0 or e in admitted)} - taken)
        staged = {e: i for i, e in enumerate(e for e in missing if e not in admitted)}
        n_fetch[r] = len(missing)
        for i, e in enumerate(missing[:width]):
            fetch[r, :, i] = torch.tensor([host[e, r] if host[e, r] >= 0 else -2 - e,            # (−2 − e: on no
                                           after[e, r] if e in admitted else stage[r, staged[e]]])  # read tier)
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
            "routed": seen, "errors": errors, "residency": after.int(), "score": torch.from_numpy(score),
            "n_empty": n_empty, "records": records, "cpu": cpu, "jobs": jobs} | (
        {"later": later.int(), "n_later": n_later.int()} if defer else {})


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


def lend(need, residency: torch.Tensor, classes: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
    """`picked [G, E] bool` (see the module's text)."""
    groups, experts, _ = residency.shape
    whole = (residency >= 0).all(-1)
    order = sorted((float(score[g, e]), e, g) for g in range(groups) for e in range(experts) if whole[g, e])
    return _pick(order, np.asarray(need), classes, (groups, experts))


def lend_tail(cls: int, n: int, residency: torch.Tensor, classes: torch.Tensor, limits: torch.Tensor) -> torch.Tensor:
    """`picked [G, E] bool`: the experts with a role of class `cls` at one of its last n slots (`limits` [G, R]: each
    role's arena size)."""
    return ((classes[:, None, :] == cls) & (residency >= (limits - n)[:, None, :])).any(-1)


def _pick(order: list[tuple], need: np.ndarray, classes: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """The first need[c] experts of `order` ((key, expert, layer)) holding a slot of class c, for every c."""
    picked, taken = torch.zeros(shape, dtype=torch.bool), np.zeros(len(need), dtype=np.int64)
    for _, e, g in order:
        holds = sorted({int(c) for c in classes[g]})
        for c in holds:
            if taken[c] < need[c]:
                picked[g, e] = True
            taken[c] += 1
    return picked


def replay(seed: np.ndarray, steps: list[dict], score: np.ndarray | None = None,
           classes: np.ndarray | None = None, slots: np.ndarray | None = None, sizes: np.ndarray | None = None,
           stacks: list | None = None, budget: int | None = None, gain: float = 0.0) -> list[dict]:
    """The L1 parity oracle: F11's residency [G, E] (whole experts), empty slots per layer [G] and their stacks after
    each step of a trace, from the seeded residency (and scores, zero at a seed). A step holds its routing `counts`
    [G, E] of decode rows, the experts any of its rows routed (`needed` [G, E]; default: the counts'), those the CPU
    took (`cpu` [G, E], F5b: never candidates; default none), and `lends` (None: the step borrowed no slots): a
    borrowing step takes every empty slot at its start, then makes its calls — row 0, where non-zero (n at one class
    c), `lend_tail(c, n)` (D11), then each row's `lend(need)` (`classes` [G, R] the slot classes) —, as the module's
    text says; all those slots return empty to their layers at the step's end, on top of the empty slots it found, by
    call then expert id. The tail needs the slots: `slots` [G, E, R] each resident's arena slots at the trace's start
    (default: a stand-in slot per expert), `sizes` [C] the classes' arena slots, `stacks` each layer's empty slots then
    ([R] each, bottom first; default none).
    The M6 replay's `lfu` (`simulate`, `_pick_swap` with gain 0), with the scores in f32 (× 0.915, then + 1.0 per
    entry), decayed only on steps with decode rows, empty slots filled first (from the top of the stack)."""
    groups, experts = seed.shape
    roles = 1 if classes is None else classes.shape[1]
    at = np.where(seed[..., None], np.arange(experts)[:, None], -1).repeat(roles, -1) if slots is None else \
        np.asarray(slots, dtype=np.int64).copy()                       # [G, E, R]: each piece's slot, −1 none
    u = np.zeros(seed.shape, dtype=np.float32) if score is None else score.astype(np.float32).copy()
    stack = [[np.asarray(e) for e in s] for s in stacks] if stacks is not None else [[] for _ in range(groups)]
    out = []
    for step in steps:
        c, lends = step["counts"], step.get("lends")
        lent: list[list] = [[] for _ in range(groups)]                 # per layer: (call, expert, its slots)
        if lends is not None:                                          # the empty slots, borrowed first
            found, stack = stack, [[] for _ in range(groups)]
        for call, need in enumerate(() if lends is None else lends):
            if call == 0:
                if not np.any(need):
                    continue
                cls = int(np.flatnonzero(need)[0])
                picked = lend_tail(cls, int(need[cls]), torch.from_numpy(at), torch.from_numpy(classes),
                                   torch.from_numpy(np.asarray(sizes)[classes])).numpy()
            else:
                whole = torch.from_numpy(np.where((at >= 0).all(-1), 0, -1)[..., None].repeat(roles, -1))
                picked = lend(need, whole, torch.from_numpy(classes), torch.from_numpy(u)).numpy()
            for g, e in np.argwhere(picked):
                lent[g].append((call, e, at[g, e].copy()))
            at[picked] = -1
        if c.any():
            u = u * DECAY
            for n in range(int(c.max())):
                u = np.where(c > n, u + np.float32(1.0), u)
            need, cpu = step.get("needed", c > 0), step.get("cpu", np.zeros(seed.shape, dtype=bool))
            for g in range(groups):
                res = (at[g] >= 0).all(-1)
                cands = sorted(np.flatnonzero((c[g] > 0) & ~res & ~cpu[g]), key=lambda e: (-u[g, e], e))
                cap = len(cands) if budget is None else budget              # (placement.budget's)
                spare = min(len(cands), len(stack[g]), cap)
                for i, a in enumerate(cands[:spare]):                  # from the top of the stack
                    at[g, a] = stack[g][-1 - i]
                del stack[g][len(stack[g]) - spare:]
                res = (at[g] >= 0).all(-1)
                victims = sorted(np.flatnonzero(res & ~need[g]), key=lambda e: (u[g, e], e))   # not the new ones
                for a, b in list(zip(cands[spare:], victims, strict=False))[:max(cap - spare, 0)]:
                    if not u[g, a] > u[g, b] + np.float32(gain):
                        break
                    at[g, a], at[g, b] = at[g, b], -1
        if lends is not None:
            stack = [f + [s for *_, s in sorted(back, key=lambda x: x[:2])]
                     for f, back in zip(found, lent, strict=True)]
        out.append({"resident": (at >= 0).all(-1), "empty": np.array([len(s) for s in stack]),
                    "stacks": [np.array(s).reshape(-1, roles) for s in stack]})
    return out
