"""PrefixIndex: content-addressed reuse of per-request state across requests (design §6.3).

Keys are a hash chain over full token blocks — block i's key hashes block i-1's key with block i's tokens, rooted in
the engine's namespace — so a key identifies the whole prefix; a match also checks each entry's tokens and parent,
so a hash collision is a miss. An entry holds, per state spec, the pool units the spec published for that block,
and the index keeps one reference on each. Entries form a tree (a prefix shared by several continuations).

Invariant (kept by StateManager): a request holds exactly the units of the chain it follows, so whoever holds a
block also holds every block before it. Units only the index holds can therefore always be freed by evicting
least-recently-used leaves upward, which is what lets the scheduler count them as available.

An entry may also hold payloads: state a kind published for the position at the block's end (e.g. a copy of a
request's recurrent state), in the payload's own pool. A payload is independent of the tree — it can be dropped from
any entry, interior ones included, to make room in its pool, and never costs paged units; dropping an entry drops its
payloads.

With a tier (the slot `tier`; docs/v1-kv-tiers.md D3, D6) the index sees only the device boundary. An entry is
resident while it holds device units, and resident entries are prefix-closed; the others are entries whose pages
only the tier holds (`lower`). Eviction scans resident entries only and counts resident children (`kids`): a victim
whose pages the tier holds, or takes now, is demoted (its device units freed); any other is dropped with its subtree,
which is all non-resident. Payloads live in the tier from registration (no units in the index). The tier's copies run
on its own streams, so a unit freed while one may still read or write it is reused only after the next launch has
waited for that copy (`freed`), and a load into it waits for it too (D4's free rule). Every decision here depends on
the calls made, never on when a copy lands.
"""
from __future__ import annotations

import collections
import dataclasses

import numpy as np

from holon.api import BlockRecord, Fence, Load, Payload

from .memory import Pool


@dataclasses.dataclass(eq=False)
class Entry:
    key: int
    parent: int | None
    tokens: tuple[int, ...]
    units: dict[str, tuple[int, ...]]          # spec name -> device units holding this block (none once demoted)
    depth: int = 0                             # blocks before it in its chain
    children: set[int] = dataclasses.field(default_factory=set)     # their keys
    kids: int = 0                              # resident children
    hits: int = 0                              # matches that reached it
    payloads: dict[str, Payload] = dataclasses.field(default_factory=dict)   # spec name -> its payload here
    lower: bool = False                        # the tier holds its pages
    copy: Fence | None = None                  # the last copy on its device units (a store reading, a load writing)
    loading: Load | None = None                # the load writing its device units, until a step reads it


class PrefixIndex:
    def __init__(self, namespace: int, block: int, pools: dict[str, Pool],
                 payload_pools: dict[str, Pool] | None = None) -> None:
        """`pools`: the paged specs' pools by spec name; `payload_pools`: every pool a payload may name."""
        self.namespace, self.block, self.pools = namespace, block, pools
        self.payload_pools = payload_pools or {}
        self.entries: collections.OrderedDict[int, Entry] = collections.OrderedDict()   # resident, least recent first
        self.lower: dict[int, Entry] = {}                   # entries only the tier holds
        self._cached: dict[str, np.ndarray] = {}           # spec name -> bool mask over its pool's units
        self.tier = self.policy = None                     # the slots `tier` and `eviction_policy` (KS4, KS5)
        self.last: Fence | None = None                     # the end of the last launched step
        self.freed: dict[int, Fence] = {}                  # copies on device units freed since the last launch
        self.loads: dict[int, list] = {}   # a load no step read, by identity -> [load, keys of its entries still
                                           # loading, whether the tier was told that no request waits for it]

    def keys(self, tokens: list[int], start: int, n_blocks: int, prev: int | None) -> list[int]:
        """Chain keys of blocks start … n_blocks-1 of `tokens`, continuing from the key of block start-1."""
        out = []
        for i in range(start, n_blocks):
            prev = self.key(prev, tokens[i * self.block:(i + 1) * self.block])
            out.append(prev)
        return out

    def key(self, prev: int | None, block_tokens) -> int:
        """The chain key of one block after the block keyed `prev` (None: the first block)."""
        return hash((self.namespace if prev is None else prev, *block_tokens))

    def get(self, key: int) -> Entry | None:
        return self.entries.get(key) or self.lower.get(key)

    def match(self, keys: list[int], tokens: list[int]) -> list[Entry]:
        """The cached entries for the longest prefix of `keys` (resident ones first); marks them recently used."""
        found: list[Entry] = []
        for i, key in enumerate(keys):
            e = self.get(key)
            if (e is None or e.parent != (keys[i - 1] if i else None)
                    or e.tokens != tuple(tokens[i * self.block:(i + 1) * self.block])):
                break
            found.append(e)
        for e in found:
            e.hits += 1
            if e.key in self.entries:
                self.entries.move_to_end(e.key)
        return found

    def insert(self, key: int, parent: int | None, tokens: tuple[int, ...], units: dict[str, tuple[int, ...]]) -> Entry:
        """Publish one block (the key must be new and the parent cached and resident)."""
        self._hold(units)
        p = self.entries[parent] if parent is not None else None
        e = self.entries[key] = Entry(key, parent, tokens, units, depth=p.depth + 1 if p else 0)
        if p is not None:
            p.children.add(key)
            p.kids += 1
        return e

    def restore(self, records: list[BlockRecord]) -> None:
        """Enter the tier's restored blocks as non-resident entries, parents first (KS4); a record that is not its
        own key's block, whose parent is unknown or without its pages is skipped."""
        for r in records:
            p = self.lower.get(r.parent)
            if self.key(r.parent, r.tokens) == r.key and (p or r.parent is None) and set(r.units) == set(self.pools):
                self.lower[r.key] = Entry(r.key, r.parent, tuple(r.tokens), {}, p.depth + 1 if p else 0, lower=True,
                                          payloads={spec: Payload((), x.pool) for spec, x in r.payloads.items()})
                if p:
                    p.children.add(r.key)

    def cached(self, name: str) -> np.ndarray:
        """Which units of spec `name`'s pool the index references (a live mask, sized with the pool)."""
        mask = self._cached.get(name)
        if mask is None or len(mask) != self.pools[name].capacity:
            mask = self._cached[name] = np.zeros(self.pools[name].capacity, dtype=bool)
        return mask

    def _hold(self, units: dict[str, tuple[int, ...]]) -> None:
        for name, us in units.items():
            self.pools[name].incref(us)
            self.cached(name)[us[0] if len(us) == 1 else list(us)] = True

    def _evictable(self, e: Entry) -> bool:
        return e.kids == 0 and all(self.pools[n].held_once(us) for n, us in e.units.items())

    def evict(self, pool: str, need: int, pure: bool = False) -> None:
        """Evict least-recently-used unused resident leaves — and the ancestors that become unused leaves, while still
        short — until `pool` has `need` free units or nothing more is evictable. `pure`: only entries whose pages the
        tier holds, so every eviction is a free (a promotion's, D7). An `eviction_policy` picks each victim instead
        (none: the least recently used; liveness never rests on it)."""
        ok = lambda e: self._evictable(e) and (e.lower or not pure)        # noqa: E731
        while self.pools[pool].num_free < need:
            # The eligible entries, least recently used first: a lazy scan that stops at the victim, not a copy of the
            # index (this runs per request when a full pool makes every request in a step evict).
            e = self.policy.victim(e for e in self.entries.values() if ok(e)) if self.policy is not None else None
            if e is not None and (self.entries.get(e.key) is not e or not ok(e)):
                raise RuntimeError(f"eviction_policy: block {e.key} is not one of the eligible entries it was given")
            if (e := e or next((e for e in self.entries.values() if ok(e)), None)) is None:
                return
            while e is not None and ok(e):                                                 # up the chain, while short
                parent = self.entries.get(e.parent) if e.parent is not None else None
                self._evict(e)
                e = parent if self.pools[pool].num_free < need and self.policy is None else None

    def _evict(self, e: Entry) -> None:
        """Demote `e` if the tier holds its pages or takes them now (an eviction store), else drop it (D6)."""
        if not (e.lower or self.tier is not None and self.store([e], evicting=True)):
            return self._drop(e)
        self._free(e)
        self.lower[e.key] = self.entries.pop(e.key)
        if e.parent is not None:
            self.entries[e.parent].kids -= 1

    def promote(self, entries: list[Entry], units: dict[str, list[int]], after: list[Fence]) -> Load:
        """Make the non-resident chain `entries`, whose parent is resident, resident in `units` (per spec, in chain
        order; the index now references them too) and have the tier load their pages into them after `after` (D7)."""
        load = self.tier.load([e.key for e in entries], units, [], after)
        for i, e in enumerate(entries):
            us = {n: tuple(v[i * len(v) // len(entries):(i + 1) * len(v) // len(entries)]) for n, v in units.items()}
            self._hold(us)
            e.units, e.copy, e.loading = us, load, load
            self.entries[e.key] = self.lower.pop(e.key)
            if e.parent is not None:
                self.entries[e.parent].kids += 1
        self.loads[id(load)] = [load, {e.key for e in entries}, False]
        self.poll()                         # what the tier dropped to make room (never what it is loading)
        return load

    def after(self) -> list[Fence]:
        """What a copy into device units must follow: the last launched step and copies on units freed since."""
        return [f for f in (self.last, *self.freed.values()) if f is not None]

    def store(self, entries: list[Entry], evicting: bool = False) -> bool:
        """Offer the resident entries' pages to the tier (KS3): at insert, and at eviction (`evicting`), after the last
        launched step, which may still write them, and after the last copy on them; whether it holds them all."""
        after = [f for f in (self.last, *{id(e.copy): e.copy for e in entries}.values()) if f is not None]
        accepted, fence = self.tier.store([BlockRecord(e.key, e.parent, e.tokens, e.units, {}) for e in entries],
                                          after, evicting)
        for e in entries:
            e.lower, e.copy = (e.key, None) in accepted, fence
        self.poll()
        return all(e.lower for e in entries)

    def poll(self) -> None:
        """Apply the tier's own evictions (D6): a resident entry loses its flag, a non-resident one goes with its
        subtree, a payload copy goes from its entry."""
        for key, spec in self.tier.dropped() if self.tier is not None else ():
            if (e := self.get(key)) is None:
                continue
            if spec is not None:
                e.payloads.pop(spec, None)
            elif key in self.entries:
                e.lower = False
            else:
                self._drop(e)

    def attach(self, e: Entry, spec: str, payload: Payload, after: list[Fence] = ()) -> bool:
        """Give entry `e` `spec`'s payload, taking over the publisher's reference; False (the caller keeps the
        reference) if `e` already has one. With a tier the payload is stored (after `after`, the step that wrote it)
        and its unit freed: the tier keeps it (D7)."""
        if spec in e.payloads:
            return False
        if self.tier is not None:
            accepted, fence = self.tier.store([BlockRecord(e.key, e.parent, e.tokens, {}, {spec: payload})], after,
                                              False)
            if (e.key, spec) not in accepted:
                raise RuntimeError(f"the tier refused a payload store of {spec!r} (payload stores are mandatory)")
            self.payload_pools[payload.pool].decref(payload.units)
            self.freed[id(fence)] = fence
            payload = Payload((), payload.pool, payload.tentative)
        e.payloads[spec] = payload
        if self.tier is not None:
            self.poll()
        return True

    def settle(self, key: int, payload: Payload, spec: str, drop: bool) -> bool:
        """A request's tentative `payload`, entry `key`'s of `spec` if it still has it: dropped wherever it is
        (`drop`: a newer one superseded it), or made ordinary (its request was released)."""
        if (e := self.get(key)) is None or e.payloads.get(spec) is not payload:
            return False
        e.payloads[spec] = Payload(payload.units, payload.pool)
        if drop:
            del e.payloads[spec]
            if payload.units:
                self.payload_pools[payload.pool].decref(payload.units)
            else:
                self.tier.drop([key], spec)
        return True

    def evict_payloads(self, pool: str, need: int) -> None:
        """Drop least-recently-used payloads in `pool` that nobody else holds until it has `need` free units."""
        p = self.payload_pools[pool]
        if p.num_free >= need:
            return
        for e in list(self.entries.values()):
            if p.num_free >= need:
                return
            for spec, payload in list(e.payloads.items()):
                if payload.pool == pool and p.held_once(payload.units):
                    del e.payloads[spec]
                    p.decref(payload.units)

    def _free(self, e: Entry) -> None:
        """Release a resident entry's device units; a copy still on them is waited for before their reuse, and a load
        whose every entry goes before any step read it is abandoned (D6, D7)."""
        freed = all(self.pools[n].held_once(us) for n, us in e.units.items())    # not by `clear` under a request
        for name, us in e.units.items():
            self.cached(name)[us[0] if len(us) == 1 else list(us)] = False
            self.pools[name].decref(us)
        load, e.units, e.loading = e.loading, {}, None
        if e.copy is not None:
            self.freed[id(e.copy)], e.copy = e.copy, None
        if load is not None:
            (keys := self.loads[id(load)][1]).discard(e.key)
            if not keys and freed:
                del self.loads[id(load)]
                self.tier.loaded(load, None)

    def _unhold(self, entries: list[Entry]) -> None:
        for p in (p for e in entries for p in e.payloads.values() if p.units):     # (a tier's have no units)
            self.payload_pools[p.pool].decref(p.units)

    def _drop(self, e: Entry) -> None:
        """Drop `e` and its subtree (non-resident: resident entries are prefix-closed, and a resident victim has no
        resident child), telling the tier."""
        gone = [e]
        for x in gone:
            gone += [self.lower[c] for c in x.children]
        for x in gone:
            if self.entries.pop(x.key, None) is not None:
                self._free(x)
                if x.parent is not None:
                    self.entries[x.parent].kids -= 1
            else:
                del self.lower[x.key]
        self._unhold(gone)
        if (p := self.get(e.parent)) is not None:
            p.children.discard(e.key)
        if self.tier is not None:
            self.tier.drop([x.key for x in gone])

    def clear(self) -> None:
        """Drop every entry (`flush_prefix`: requests keep the units they hold, and the loads into them land) and the
        tier's copies."""
        for e in self.entries.values():
            self._free(e)
        self._unhold([*self.entries.values(), *self.lower.values()])
        self.entries, self.lower = collections.OrderedDict(), {}
        if self.tier is not None:
            self.tier.clear()
