"""A group's in-graph resolve and fetch (docs/dev/v1-offload.md D5): its routed ids to arena slots per role through the
residency map (`moe_resolve@1`, or the policy's variant contract with the policy's device state), every miss gathered
from the host tier into its slot — a staging slot, or the slot a device-owned policy admitted it into
(`expert_fetch@1`, F5a). Under exclusive tiers (D10) `moe_tiers@2` then places the call's victims and SSD misses, the
I/O worker reads its SSD pieces (`cpu_task@1`'s copy task), and after the gather the victims are demoted into their RAM
units and the admitted moved from staging into their slots."""
from __future__ import annotations

import functools

import numpy as np
import torch

from . import ring
from .ring import PSEUDO, SHORT

RESOLVE, FETCH, EXPERT_PREFETCH = "moe_resolve@1", "expert_fetch@1", "expert_prefetch@1"
ACTIVATION_LOAN = "activation_loan@1"
TIERS, CPU_TASK = "moe_tiers@2", "cpu_task@1"


class DeviceResolver:
    def __init__(self, ctx, store, group_id: int, classes: list) -> None:
        self.store, self.group_id, self.classes = store, group_id, classes
        self.group = None                     # its row in the store's tables, once attached (`attach`)
        self._prepared = False                # the pass is prepared, not yet resolved
        self.io_slot = None                   # exclusive tiers: its SSD reads' slot on the I/O worker (D8g: one a
        self.io_slots: tuple = ()             # landing half, `io_slots`)
        facts = {"experts": store.items, "top_k": store.top_k, "roles": store.roles, "staging": store.staging}
        site = f"std.tiered[{store.name}]"
        op = store.policy.resolve_variant or RESOLVE
        self.resolve = ctx.select_kernel(op, site=site, **facts).impl(ctx, **facts)
        self._fetch = []
        for c in classes:
            facts = {"unit_bytes": c.unit_bytes, "staging": c.staging}
            self._fetch.append(ctx.select_kernel(FETCH, site=f"{site}.{c.name}", **facts).impl(ctx, **facts))

    def fetches(self) -> dict[str, object]:
        """Its classes' `expert_fetch@1` instances, by class name."""
        return {c.name: f for c, f in zip(self.classes, self._fetch, strict=True)}

    @property
    def fetches_parts(self) -> bool:
        """Its fetch fetches a list part by part (`expert_fetch@1`'s `parts`, every role's in one launch)."""
        fetch = self._fetch[0]
        return bool(getattr(fetch, "parts", False)) and getattr(fetch, "gather_roles", None) is not None

    def attach(self, group: int, state: dict[str, torch.Tensor]) -> None:
        """Its row `group` of the store's device tables (made when every group has its pieces)."""
        store = self.store
        self.group, self.state = group, state
        self.residency, self.host = store.residency.device[group], store.host_units[group]
        self.limits, self.counts = store.limits[group], store.counts[group]

    def prepare(self, first: int) -> None:
        """Before the first use of the pass starting at the step's token `first`: on the step's first, its counts and
        error words from zero, its edits applied and (a prefill step) its slots borrowed; on the layer's first, its ring
        and stage tables. Once per pass (a route transform's map and the resolve both need it)."""
        store = self.store
        if self._prepared:
            return
        self._prepared = True
        if not self.group and not first:
            store.start_step()
            store.wait_parked()
        if not first:
            self.ring, self.stage = store.sources(self.group)

    def present(self, first: int) -> torch.Tensor:
        """The layer's map with this step's ring pieces in their ring slots: what a route transform reads."""
        self.prepare(first)
        return torch.where(self.ring >= 0, self.ring, self.residency)

    def __call__(self, ids: torch.Tensor, first: int, plan: torch.Tensor | None = None,
                 jobs: torch.Tensor | None = None, before_fetch=None,
                 kinds: torch.Tensor | None = None, split: bool = False, parts: int = 1):
        """Slots [roles, T, k] for `ids` [T, k], the step's tokens `first`… (their row kinds, or `kinds`), the misses
        fetched; and
        `cpu` [T, k], the entries the CPU computes (F5b: its share by `plan`, its jobs in `jobs`; none without), handed
        to `before_fetch(cpu)` before the fetch (the CPU's jobs start, then the GPU's misses come over PCIe: D5)."""
        store = self.store
        kinds = store.kinds[first:first + len(ids)] if kinds is None else kinds     # given: a drafter sub-step's
        if len(kinds) != len(ids):
            raise ValueError(f"tokens {first}…{first + len(ids)} past the step's {len(store.kinds)} row kinds")
        self.prepare(first)
        if store.exclusive:
            store.call_start.copy_(store.n_delta[self.group:self.group + 1])
        split = split and getattr(self.resolve, "splits", False) and before_fetch is None
        lent = store.exclusive and store.lent is not None and self.group < store.borrowing_groups
        rounds = lent and store.lent.rounds
        beside = lent and store._beside and parts > 1 and before_fetch is None and not split     # D8g
        if rounds and (parts <= 1 or before_fetch is not None or split):
            raise RuntimeError(f"{store.name}: a pass of layer {self.group} did not pipeline in a step that stages in "
                               "rounds (docs/dev/v1-glm.md D8s: its fixed slots hold one part at a time)")
        parts = parts if self.fetches_parts and not store.exclusive and before_fetch is None else 1
        if store.parts > 1 and parts <= 1 and self.group < store.borrowing_groups:
            raise RuntimeError(f"{store.name}: a pass of layer {self.group} did not pipeline in a step whose staging "
                               "is a ring of parts (its slots hold two parts of a layer's misses)")
        out = self.resolve(
            ids, kinds, self.residency, self.host, self.limits, self.counts, store.routed[self.group], self.ring,
            self.stage, store.empty[self.group], store.n_empty[self.group:self.group + 1], store.misses, store.errors,
            store.delta[self.group], store.n_delta[self.group:self.group + 1], self.group,
            store.no_plan if plan is None else plan, store.no_jobs if jobs is None else jobs,
            **({"split": True} if split else {}), **self.state)
        slots, fetch, n, cpu = out[:4]
        self._prepared = False
        if beside:
            return self._beside(ids, kinds, slots, fetch, n)
        if rounds:
            return self._rounds(ids, kinds, slots, fetch, n)
        if store.exclusive:                      # exclusive tiers: RAM misses, then SSD rounds (docs/dev/v1-glm.md D8f)
            if before_fetch is not None:
                before_fetch(cpu)
            finish = self._tiers(ids, kinds, fetch, n, split)
            if split:                            # the hits' passes run while the RAM misses come and the SSD reads
                return slots, cpu, out[4], finish
            finish()
            return slots, cpu
        if split:                                # the fetch on a side stream; the hits' passes run meanwhile
            main = torch.accelerator.current_stream(ids.device)
            side = store.fetch_stream(split=True)
            side.wait_stream(main)
            with side:
                self._gather(fetch, n)
            fetch.record_stream(side)            # the main stream's allocations must not reuse them meanwhile
            n.record_stream(side)
            return slots, cpu, out[4], lambda: main.wait_stream(side)
        if parts > 1:                            # the misses part by part on a side stream, each part's entries known
            return self._pipelined(ids, slots, fetch, n, parts)
        if before_fetch is not None:
            before_fetch(cpu)
        self._gather(fetch, n)
        return slots, cpu

    def _tiers(self, ids: torch.Tensor, kinds: torch.Tensor, fetch: torch.Tensor, n: torch.Tensor, split: bool):
        """Exclusive tiers' call after its resolve (docs/dev/v1-glm.md D8f, D8c): `moe_tiers@2` places; round 0 of the
        SSD reads signalled at once; the RAM misses' gather beside those reads — on the high-priority side stream when
        the call splits, so a RAM miss never waits behind an SSD read. Returns `finish`, on the main stream: the
        read-ahead hits, the rounds (each waited, gathered, then the next signalled: rounds of A_od landing rows, as
        many as the call could need — no host sync; a round past the misses is empty; a graphed step's fit one), then
        the demotions and the admitted moves (after the side stream's gather)."""
        store, g = self.store, self.group
        self._tier_op(ids, kinds, fetch, n)
        # rounds of A landing rows for the most SSD misses a call can have: its distinct routed experts
        rounds = -(-min(store.items, store.top_k * len(ids)) // store.landing_rows)
        store.round_rows(0)                      # round 0's reads first: the RAM gather runs while they do (C S1-M1)
        store.io.signal(self.io_slot)
        main = torch.accelerator.current_stream(ids.device) if split else None
        side = store.fetch_stream(split=True) if split else None
        if side is not None:
            side.wait_stream(main)
            with side:
                self._gather(fetch, store.n_ram)
            fetch.record_stream(side)
        else:
            self._gather(fetch, store.n_ram)

        def finish() -> None:
            if store._ahead:                     # read ahead: its set signalled two layers back (or at step start)
                store.io.wait(store.ahead_slots[g])
                self._gather(store.hit, store.n_hit, landed=True, part=None)
            for r in range(rounds):
                if r:
                    store.round_rows(r)
                    store.io.signal(self.io_slot)
                store.io.wait(self.io_slot)
                self._gather(store.ssd, store.n_ssd, landed=True, part=r)
            if side is not None:
                main.wait_stream(side)
            self._moves(ids)
        return finish

    def _tier_op(self, ids: torch.Tensor, kinds: torch.Tensor, fetch: torch.Tensor, n: torch.Tensor) -> None:
        """`moe_tiers@2` on the call (the last calls' moves fenced first where they run beside it)."""
        store, g = self.store, self.group
        moves, n_moves = store.moves[g % 2], store.n_moves[g % 2]
        if store.moves_async:                    # the last call's admissions read the staging slots this one fills;
            main = torch.accelerator.current_stream(ids.device)          # the call before it, this table (`moves`)
            for key, event in (("admitted", store.admitted), (g % 2, store.moved[g % 2])):
                if key in store.recorded:        # (recorded in this step: the last step's moves joined before it)
                    main.wait_event(event)
        store.tier_op(ids, kinds, g, fetch, n, self.stage, store.delta[g], store.call_start,
                      store.n_delta[g:g + 1], store.residency.device, store.host_units, store.piece,
                      store.classes32, store.owner, store.capacity, store.heat, store.io_rows, moves,
                      n_moves, store.ssd, store.n_ssd, store.n_ram, store.landing_rows,
                      store.ahead_map[g] if store._ahead else None, store.hit, store.n_hit, **store.tier_stamps)

    def _moves(self, ids: torch.Tensor) -> None:
        """The call's demotions and admitted moves, after its gathers: on the main stream, or beside it on the store's
        moves stream (admissions first) under a policy with spares."""
        store, g = self.store, self.group
        moves, n_moves = store.moves[g % 2], store.n_moves[g % 2]
        moving = zip(self.classes, self._fetch, strict=True)
        if not store.moves_async:
            for r, (c, gather) in enumerate(moving):
                gather(moves[r, 0:2], n_moves[r, 0:1], c.slots(), c.units())      # demotions
                gather(moves[r, 2:4], n_moves[r, 1:2], c.slots(), c.slots())      # admitted, staged
            return
        moving = list(moving)                    # on the moves stream: admitted first (spares: never a demoted slot)
        stream = store.moves_stream()
        stream.wait_stream(torch.accelerator.current_stream(ids.device))
        beside = [{"beside": True} if getattr(gather, "beside", False) else {} for _, gather in moving]
        with stream:                             # (on few SMs where the copy can: beside the next calls' compute)
            for r, (c, gather) in enumerate(moving):
                gather(moves[r, 2:4], n_moves[r, 1:2], c.slots(), c.slots(), **beside[r])
            store.admitted.record()              # (the current stream: this one)
            store.recorded.add("admitted")
            for r, (c, gather) in enumerate(moving):
                gather(moves[r, 0:2], n_moves[r, 0:1], c.slots(), c.units(), **beside[r])
            store.moved[g % 2].record()
            store.recorded.add(g % 2)

    def _rounds(self, ids: torch.Tensor, kinds: torch.Tensor, slots: torch.Tensor, fetch: torch.Tensor,
                n: torch.Tensor):
        """D8s (docs/dev/v1-glm.md): a staging call under exclusive tiers in rounds through its classes' fixed slots
        (`Lent.fixed`: `rounds_q` for RAM misses and read-ahead hits, the landing rows' for SSD reads). After the tier
        op, each miss gets its part — read-ahead hits by `rounds_q`, then RAM misses by `rounds_q`, then SSD rounds
        by the landing rows; an expert's pieces share their lists' places (RAM and the SSD hold whole experts) — and
        that part's fixed slot, in the lists and in `slots` (its stage entry PSEUDO − i). Returns (slots, part [T, k],
        landed): `landed[p]()` gathers part p on the main stream (an SSD round after its read), the last one then the
        moves; part p + 1's gather follows part p's compute in stream order, so the slots are reused safely."""
        store, g = self.store, self.group
        self._tier_op(ids, kinds, fetch, n)
        store.round_rows(0)                      # round 0's reads first: the hits' and RAM's parts run meanwhile
        store.io.signal(self.io_slot)
        q, a = store.rounds_q, store.landing_rows
        most = min(store.items, store.top_k * len(ids))
        hits = -(-store.ahead_rows // q) if store._ahead else 0
        ram, ssd = -(-most // q), -(-most // a)
        lists = [(store.hit, store.n_hit, 0, q, 0)] if store._ahead else []
        lists += [(fetch, store.n_ram, hits, q, 0), (store.ssd, store.n_ssd, hits + ram, a, q)]
        slots, part = self._place_rounds(lists, slots)
        landed = []
        for p in range(hits):
            def hit(p=p):
                if p == 0:                       # read ahead: its set signalled two layers back (or at step start)
                    store.io.wait(store.ahead_slots[g])
                self._gather(store.hit, store.n_hit, landed=True, part=p, size=q)
            landed.append(hit)
        landed += [lambda p=p: self._gather(fetch, store.n_ram, part=p, size=q) for p in range(ram)]
        for r in range(ssd):
            def read(r=r):
                if r:
                    store.round_rows(r)
                    store.io.signal(self.io_slot)
                store.io.wait(self.io_slot)
                self._gather(store.ssd, store.n_ssd, landed=True, part=r, size=a)
                if r == ssd - 1:
                    self._moves(ids)
            landed.append(read)
        return slots, part, landed

    def _place_rounds(self, lists: list, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """D8s: each listed miss (`lists`: (table [R, 2, F], count [R], its first part, a part's size, its place in the
        fixed set)) its part and that part's fixed slot, in the lists and in `slots` (its stage entry PSEUDO − i).
        Returns (slots, part [T, k])."""
        store, g = self.store, self.group
        roles, width = len(self.classes), self.stage.shape[1]
        dev = slots.device
        part_of = torch.full((roles, width + 1), -1, dtype=torch.int64, device=dev)
        new_of = torch.full((roles, width + 1), -1, dtype=torch.int32, device=dev)
        fixed = store.lent.fixed
        for r in range(roles):
            own = fixed[int(store.class_table[g, r])]
            for table, count, first, size, at, *_ in lists:
                pos = torch.arange(table.shape[-1], device=dev)
                listed = pos < count[r]
                index = torch.where(listed, PSEUDO - table[r, 1].long(), width).clamp(0, width)
                slot = own[self._round_slot(at, pos, size)]
                part_of[r].scatter_(0, index, self._round_part(r, first, pos, size))
                new_of[r].scatter_(0, index, slot)
                table[r, 1].copy_(torch.where(listed, slot, table[r, 1]))
        self._whole(lists)
        staged = slots <= PSEUDO
        index = torch.where(staged, PSEUDO - slots.long(), width).clamp(0, width).view(roles, -1)
        slots = torch.where(staged, new_of.gather(1, index).view_as(slots), slots)
        return slots, torch.where(staged, part_of.gather(1, index).view_as(slots), -1).amax(0)

    def _whole(self, lists: list) -> None:
        """Every role's lists name the same experts in the same places (whole experts): else the error word's
        SHORT."""
        counts = torch.stack([c[:len(self.classes)] for _, c, *_ in lists])
        self.store.errors.bitwise_or_((counts != counts[:, :1]).any().int() * SHORT)

    def _stage_parts(self, lists: list, slots: torch.Tensor) -> torch.Tensor:
        """D8g's stage mode: each listed miss's part (`lists` as `_place_rounds`'), by its staging slot (each miss its
        own); part [T, k], the latest of an entry's pieces' (−1: resident or a null slot)."""
        part = torch.full(slots.shape[1:], -1, dtype=torch.int64, device=slots.device)
        for r, c in enumerate(self.classes):
            of = torch.full((c.size + 1,), -1, dtype=torch.int64, device=slots.device)
            for table, count, first, size, _, off in lists:
                pos = torch.arange(table.shape[-1], device=slots.device)
                at = torch.where(pos < count[r], table[r, 1].long(), c.size).clamp(0, c.size)
                of.scatter_(0, at, self._round_part(r, first, pos + off, size))
            of[c.size] = -1
            part = torch.maximum(part, of[slots[r].long().clamp(0, c.size)])
        self._whole(lists)
        return part

    def _beside(self, ids: torch.Tensor, kinds: torch.Tensor, slots: torch.Tensor, fetch: torch.Tensor,
                n: torch.Tensor):
        """D8g (docs/dev/v1-glm.md): an eager staging call's gathers by copy engines beside the compute. After the tier
        op round 0's reads are signalled (landing half 0), each miss gets its part — the read-ahead hits' and the RAM
        misses' (in rounds: by `rounds_q`, through the fixed set; else STAGE_PARTS of them, at least `rounds_q` each,
        each miss in its own staging slot), then the SSD rounds' (by the landing rows) —, the read-ahead set waited
        for (A G3-m1: before `tiered`, so every stream after it follows the read), and the lists read back (`tiered`).
        Returns (slots, part [T, k], landed): `landed` a `_Beside`, its length known once the lists are back — taken
        after part −1 is enqueued."""
        store, g = self.store, self.group
        main = torch.accelerator.current_stream(ids.device)
        self._tier_op(ids, kinds, fetch, n)
        store.round_rows(0)                      # round 0's reads first, on main before `tiered` (half 0)
        store.io.signal(self.io_slots[0])
        rounds, a, q = store.lent.rounds, store.landing_rows, store.rounds_q
        c_hit, c_ram = store.n_hit.max().long(), store.n_ram.max().long()
        if rounds:                               # the fixed set's front parts: hits, then RAM misses, by rounds_q
            size, first_ram, off_ram = q, -(-c_hit // q), 0
            first_ssd = first_ram + -(-c_ram // q)
        else:                                    # STAGE_PARTS front parts over the hits, then the RAM misses
            size = (-(-(c_hit + c_ram) // ring.STAGE_PARTS)).clamp(min=q)
            first_ram, off_ram, first_ssd = 0, c_hit, -(-(c_hit + c_ram) // size)
        lists = [(store.hit, store.n_hit, 0, size, 0, 0)] if store._ahead else []
        lists += [(fetch, store.n_ram, first_ram, size, 0, off_ram), (store.ssd, store.n_ssd, first_ssd, a, q, 0)]
        if rounds:
            slots, part = self._place_rounds(lists, slots)
        else:
            part = self._stage_parts(lists, slots)
        if store._ahead:                         # read ahead: its set signalled two layers back (or at step start)
            store.io.wait(store.ahead_slots[g])
        return slots, part, _Beside(self, ids, main, store.read_back(fetch), rounds)

    @staticmethod
    def _round_slot(at: int, pos: torch.Tensor, size: int) -> torch.Tensor:
        """D8s: a list's place `pos` → its round's fixed slot (the set's `at`…, `size` of them)."""
        return at + pos % size

    @staticmethod
    def _round_part(role: int, first: int, pos: torch.Tensor, size: int) -> torch.Tensor:
        """D8s: a list's place `pos` → its part (the list's first, then one per `size` places), every role alike."""
        return first + pos // size

    def _pipelined(self, ids: torch.Tensor, slots: torch.Tensor, fetch: torch.Tensor, n: torch.Tensor, parts: int):
        """(slots, part [T, k], landed): the misses' fetch in `parts` parts on the store's fetch stream — part i the
        entries [i·q, (i + 1)·q) of each role's list, q = ⌈n / parts⌉ —, an entry's part the latest of its pieces'
        (−1: resident or a null slot; a fetched piece's rank: its host unit's place in the role's list, staged or
        admitted). Parts 0 and 1 start now; `landed[i]()` — called in order, once the work of the parts before i is
        enqueued — starts part i + 1's fetch after that work (a staging ring of parts reuses part i − 1's slots:
        `Lent`), then makes the current stream wait for part i — in such a ring; else every part starts now."""
        store = self.store
        main = torch.accelerator.current_stream(slots.device)
        side = store.fetch_stream()
        side.wait_stream(main)
        events: list[torch.Event] = []
        ring = store.lent.width if store.lent is not None and store.lent.width is not None and \
            self.group < store.borrowing_groups else None          # a staging ring's slots per class

        def start(i: int) -> None:
            if i >= 2 and ring is not None:      # after the part two before it ran: its slots may be these
                side.wait_stream(main)
            with side:
                self._fetch[0].gather_roles(fetch, n, [c.units() for c in self.classes],
                                            [c.slots() for c in self.classes], part=i, parts=parts)
                events.append(torch.Event(device=slots.device))
                events[i].record(side)

        def landed(i: int) -> None:
            if i + 1 < parts and i >= 1 and ring is not None:
                start(i + 1)
            main.wait_event(events[i])
        for i in range(min(2, parts) if ring is not None else parts):
            start(i)
        fetch.record_stream(side)
        n.record_stream(side)
        width = self.stage.shape[1]
        part = torch.full(slots.shape[1:], -1, dtype=torch.int64, device=slots.device)
        live = ids >= 0
        for r, c in enumerate(self.classes):
            units = c.units().shape[0]
            listed = torch.arange(width, device=slots.device) < n[r]
            rank = torch.full((units + 1,), -1, dtype=torch.int64, device=slots.device)
            rank.scatter_(0, torch.where(listed, fetch[r, 0].long(), units), torch.arange(width, device=slots.device))
            rank[units] = -1
            at = torch.where(live, rank[self.host[ids.long().clamp(min=0), r].long()], -1)
            q = (n[r:r + 1].long().clamp(max=width) + parts - 1) // parts
            # an entry's part, the latest of its roles' (each role's list the same experts: residency is whole)
            part = torch.maximum(part, torch.where(at >= 0, at // q.clamp(min=1), -1))
            if ring is not None:                 # two parts past a ring the list wraps: never (a pass misses no more
                w = ring[int(store.class_table[self.group, r])]                # than it lacks)
                store.errors.bitwise_or_(((2 * q > w) & (n[r:r + 1].clamp(max=width) > w)).int() * SHORT)
        return slots, part, [lambda i=i: landed(i) for i in range(parts)]

    def _gather(self, fetch: torch.Tensor, n: torch.Tensor, landed: bool = False, part: int | None = 0,
                size: int | None = None) -> None:
        """The listed misses into their slots: from the RAM units, or (`landed`) round `part`'s from the landing rows by
        their runs (entries [part·A, (part + 1)·A), A the store's on-demand rows; `part` None: every entry); with
        `size`, entries [part·size, (part + 1)·size) of either (D8s's rounds)."""
        store = self.store
        srcs, runs = ([c.units() for c in self.classes], None) if not landed else \
            map(list, zip(*(store.landing_view(c.name) for c in self.classes), strict=True))
        a = size or (store.landing_rows if landed else 0)
        rounds = {"part": part, "size": a} if (landed or size) and part is not None else {}
        roles = getattr(self._fetch[0], "gather_roles", None)
        if roles is not None:                    # every role's misses in one launch
            roles(fetch, n, srcs, [c.slots() for c in self.classes], **({"runs": runs} if landed else {}), **rounds)
        else:
            for r, (c, gather) in enumerate(zip(self.classes, self._fetch, strict=True)):
                if rounds:                       # the round's slice of the list
                    gather(fetch[r][:, part * a:(part + 1) * a].contiguous(), (n[r:r + 1] - part * a).clamp(0, a),
                           srcs[r], c.slots(), *([runs[r]] if landed else []))
                else:
                    gather(fetch[r], n[r:r + 1], srcs[r], c.slots(), *([runs[r]] if landed else []))

    def done(self) -> None:
        """After the layer's expert passes (the step's ring moves on)."""
        self.store.layer_done(self.group)


class _Beside:
    """D8g (docs/dev/v1-glm.md): a call's gathers by copy engines beside the compute, as the host issues them; the
    sequence of its parts' `landed` callables. Its length is known once the call's lists are back (`load`, after the
    pass enqueued part −1); `self[i]()` makes the main stream wait for part i — where parts reuse one fixed set (a
    rounds step), after starting its copies with main current —, the last one then the call's moves.

    The streams: main, the store's control stream (the SSD rounds' device waits, copies and next signals) and each
    class's two `expert_prefetch@1` instances' (`parts`, `rounds`). The fences are `torch.Event`s (generic, recorded
    before they are waited): F1 `tiered` (main after the tier op, round 0's signal, the read-ahead's wait and the
    readback), which the control stream waits before anything; F2 a part's copies (its keys) or an SSD round's `ssd[r]`
    (control), waited on main before its pass; F3a a front part p ≥ 1 of a rounds step starts with main current, after
    part p − 1's pass (the provider waits the current stream); F3b `computed[r]` (main, after round r's pass), which
    the control stream waits before round r + 1's copies into the same fixed set (a rounds step); F5 round r + 2's
    rows and signal (the same landing half, its own request table) follow round r's copies on the control stream. Every
    round signalled is waited by main before the call ends (a REQ over an unseen DONE would answer an old wait)."""

    def __init__(self, resolver: DeviceResolver, ids: torch.Tensor, main, tiered, rounds: bool) -> None:
        self.resolver, self.store, self.ids, self.main, self.tiered = resolver, resolver.store, ids, main, tiered
        self.rounds, self.parts = rounds, None

    def __len__(self) -> int:
        return len(self.load())

    def __getitem__(self, i: int):
        return self.load()[i]

    def load(self) -> list:
        """The call's lists back (the host waits `tiered`), its first copies issued: front part 0 (stage: every front
        part, each miss its own slot) under the control stream after `tiered`, round 1's signal, round 0's copies."""
        if self.parts is not None:
            return self.parts
        s, res = self.store, self.resolver
        self.lists = s.read_lists(self.tiered)
        q, a = s.rounds_q, s.landing_rows
        n_hit, n_ram, n_ssd = (int(self.lists[k].max()) for k in ("n_hit", "n_ram", "n_ssd"))
        n_hit = n_hit if s._ahead else 0          # (kind, first entry, past its last) per front part
        if self.rounds:
            self.front = [[("hit", p, p + q)] for p in range(0, n_hit, q)] + \
                [[("ram", p, p + q)] for p in range(0, n_ram, q)]
        else:                                    # (`_beside`'s numbering: STAGE_PARTS over the hits, then RAM)
            size = max(q, -(-(n_hit + n_ram) // ring.STAGE_PARTS))
            self.front = [[(k, max(p - o, 0), p + size - o) for k, o, n in (("hit", 0, n_hit), ("ram", n_hit, n_ram))
                           if p + size > o and p < o + n] for p in range(0, n_hit + n_ram, size)]
        self.n_rounds = -(-n_ssd // a)
        self.keys = [[] for _ in self.front]
        self.events, self.computed, self.signalled, self.waited = {}, {}, {0}, set()
        control = s.control_stream()
        self.follow(control)
        with control:
            for p in range(len(self.front) if not self.rounds else min(1, len(self.front))):
                self.start(p)
            self.next_round(1)                   # round 1's reads into half 1 (round 0's went before `tiered`)
            if not self.n_rounds:                # round 0 asked nothing: its answer seen all the same
                s.io.wait(res.io_slots[0])
                self.events[0] = torch.Event(device=self.ids.device)
                self.events[0].record(control)
        if self.n_rounds:
            self.ssd(0)
        self.parts = [functools.partial(self.part, p) for p in range(len(self.front))] + \
            [functools.partial(self.round, r) for r in range(self.n_rounds)]
        if not self.parts:
            self.finish()
        return self.parts

    def follow(self, control) -> None:
        """F1: the control stream follows `tiered` (the tier op, round 0's request, the read-ahead's wait) — a device
        edge, though the host has waited `tiered` before issuing any control work (`read_lists`)."""
        control.wait_event(self.tiered)

    def start(self, p: int) -> None:
        """Front part p's copies on each class's `parts` instance, after the current stream's work."""
        s, res, g = self.store, self.resolver, self.resolver.group
        for r, c in enumerate(res.classes):
            for kind, lo, hi in self.front[p]:
                table, count = ("hit", "n_hit") if kind == "hit" else ("fetch", "n_ram")
                at = self.lists[table][r][:, lo:min(hi, int(self.lists[count][r]))]
                if not at.shape[1]:
                    continue
                if kind == "hit":                # landing rows: each row's piece from its read-ahead request
                    src, _ = s.landing_view(c.name, device=False)
                    runs = s.landing_runs(c.name, at[0], s.ahead_pieces(g, r, at[0]))
                else:
                    src, runs = c.host_view(), None
                self.keys[p].append(self.copy(s.copies[c.name][0], at, src, c, runs))

    def copy(self, inst, at, src, c, runs):
        """One `start` of `at` [2, n] (source row or unit, slot) into class c's slots; its (instance, key)."""
        key = self.store.copy_key()
        inst.start(key, torch.from_numpy(np.ascontiguousarray(at, dtype=np.int32)),
                   torch.tensor([at.shape[1]], dtype=torch.int32), src, c.slots(), runs)
        return inst, key

    def ssd(self, r: int) -> None:
        """SSD round r on the control stream: (a rounds step's round r ≥ 1: after main computed round r − 1, F3b) the
        device's wait for its half's request, its copies from the landing rows (the half's rows), their end
        (`ssd[r]`), then round r + 2's rows and signal on the same half (F5)."""
        s, res = self.store, self.resolver
        half, a = r % 2, s.landing_rows
        control = s.control_stream()
        with control:
            if (computed := self.computed.get(r - 1)) is not None:      # F3b (recorded in a rounds step only)
                control.wait_event(computed)
            s.io.wait(res.io_slots[half])
            keys = []
            for k, c in enumerate(res.classes):
                j = np.arange(r * a, min((r + 1) * a, int(self.lists["n_ssd"][k])))
                if not len(j):
                    continue
                at = self.lists["ssd"][k][:, j].copy()
                at[0] += half * a
                pieces = self.lists["io_rows"][k, 1 + j, 0]
                src, _ = s.landing_view(c.name, device=False)
                keys.append(self.copy(s.copies[c.name][1], at, src, c, s.landing_runs(c.name, at[0], pieces)))
            for inst, key in keys:
                inst.wait(key)
            self.events[r] = torch.Event(device=self.ids.device)
            self.events[r].record(control)
            self.next_round(r + 2)

    def next_round(self, r: int) -> None:
        """Round r's rows and signal on the current stream, into its half (`r % 2`), if the call has a round r."""
        if r < self.n_rounds:
            self.store.round_rows(r, half=r % 2)
            self.store.io.signal(self.resolver.io_slots[r % 2])
            self.signalled.add(r)

    def part(self, p: int) -> None:
        """Front part p landed on main (a rounds step's p ≥ 1 started here, with main current: F3a)."""
        if self.rounds and p:
            now = torch.accelerator.current_stream(self.ids.device)
            if (now.stream_id, now.device_index) != (self.main.stream_id, self.main.device_index):
                raise RuntimeError(f"{self.store.name}: a rounds part's copies started off the main stream (D8g F3a)")
            self.start(p)
        self.land(p)
        if p == len(self.parts) - 1:
            self.finish()

    def round(self, r: int) -> None:
        """SSD round r landed on main: its control work issued once main enqueued round r − 1's pass (F3b's record
        first), then main waits `ssd[r]`."""
        if r:
            if self.rounds:
                self.computed[r - 1] = torch.Event(device=self.ids.device)
                self.computed[r - 1].record(self.main)
            self.ssd(r)
        self.land(len(self.front) + r)
        if len(self.front) + r == len(self.parts) - 1:
            self.finish()

    def land(self, i: int) -> None:
        """Main waits for part i (F2): a front part's copies, or SSD round's `ssd[r]`."""
        if i < len(self.front):
            for inst, key in self.keys[i]:
                inst.wait(key)
        else:
            self.main.wait_event(self.events[i - len(self.front)])
            self.waited.add(i - len(self.front))

    def finish(self) -> None:
        """After the last part's wait: every round signalled waited by main, then the call's moves (F4)."""
        for r, event in self.events.items():
            if r not in self.waited:
                self.main.wait_event(event)
                self.waited.add(r)
        if self.waited != self.signalled:
            raise RuntimeError(f"{self.store.name}: SSD rounds {sorted(self.signalled)} signalled, "
                               f"{sorted(self.waited)} waited (D8g)")
        self.resolver._moves(self.ids)
        self.parts, self.ids = [], None          # (its parts' partials refer to it: no step tensor kept past the call)
