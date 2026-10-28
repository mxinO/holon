"""A group's in-graph resolve and fetch (docs/dev/v1-offload.md D5): its routed ids to arena slots per role through the
residency map (`moe_resolve@1`, or the policy's variant contract with the policy's device state), every miss gathered
from the host tier into its slot — a staging slot, or the slot a device-owned policy admitted it into
(`expert_fetch@1`, F5a). Under exclusive tiers (D10) `moe_tiers@2` then places the call's victims and SSD misses, the
I/O worker reads its SSD pieces (`cpu_task@1`'s copy task), and after the gather the victims are demoted into their RAM
units and the admitted moved from staging into their slots."""
from __future__ import annotations

import torch

from .ring import PSEUDO, SHORT

RESOLVE, FETCH, EXPERT_PREFETCH = "moe_resolve@1", "expert_fetch@1", "expert_prefetch@1"
ACTIVATION_LOAN = "activation_loan@1"
TIERS, CPU_TASK = "moe_tiers@2", "cpu_task@1"


class DeviceResolver:
    def __init__(self, ctx, store, group_id: int, classes: list) -> None:
        self.store, self.group_id, self.classes = store, group_id, classes
        self.group = None                     # its row in the store's tables, once attached (`attach`)
        self._prepared = False                # the pass is prepared, not yet resolved
        self.io_slot = None                   # exclusive tiers: its SSD reads' slot on the I/O worker
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
        rounds = store.exclusive and store.lent is not None and store.lent.rounds and \
            self.group < store.borrowing_groups
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
        roles, width = len(self.classes), self.stage.shape[1]
        dev = ids.device
        part_of = torch.full((roles, width + 1), -1, dtype=torch.int64, device=dev)
        new_of = torch.full((roles, width + 1), -1, dtype=torch.int32, device=dev)
        fixed = store.lent.fixed
        lists = [(store.hit, store.n_hit, 0, q, 0)] if store._ahead else []
        lists += [(fetch, store.n_ram, hits, q, 0), (store.ssd, store.n_ssd, hits + ram, a, q)]
        for r in range(roles):
            own = fixed[int(store.class_table[g, r])]
            for table, count, first, size, at in lists:
                pos = torch.arange(table.shape[-1], device=dev)
                listed = pos < count[r]
                index = torch.where(listed, PSEUDO - table[r, 1].long(), width).clamp(0, width)
                slot = own[self._round_slot(at, pos, size)]
                part_of[r].scatter_(0, index, self._round_part(r, first, pos, size))
                new_of[r].scatter_(0, index, slot)
                table[r, 1].copy_(torch.where(listed, slot, table[r, 1]))
        # every role's lists name the same experts in the same places (whole experts): else the error word's SHORT
        counts = torch.stack([c[:roles] for _, c, *_ in lists])
        store.errors.bitwise_or_((counts != counts[:, :1]).any().int() * SHORT)
        staged = slots <= PSEUDO
        index = torch.where(staged, PSEUDO - slots.long(), width).clamp(0, width).view(roles, -1)
        slots = torch.where(staged, new_of.gather(1, index).view_as(slots), slots)
        part = torch.where(staged, part_of.gather(1, index).view_as(slots), -1).amax(0)
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
