"""PrefixIndex: content-addressed reuse of per-request state across requests (design §6.3).

Keys are a hash chain over full token blocks — block i's key hashes block i-1's key with block i's tokens, rooted in
the engine's namespace — so a key identifies the whole prefix; a match also checks each entry's tokens and parent,
so a hash collision is a miss. An entry holds, per state spec, the pool units the spec published for that block,
and the index keeps one reference on each. Entries form a tree (a prefix shared by several continuations).

Invariant (kept by StateManager): a request holds exactly the units of the chain it follows, so whoever holds a
block also holds every block before it. Units only the index holds can therefore always be freed by evicting
least-recently-used leaves upward, which is what lets the scheduler count them as available.
"""
from __future__ import annotations

import collections
import dataclasses

import numpy as np

from .memory import Pool


@dataclasses.dataclass(eq=False)
class Entry:
    key: int
    parent: int | None
    tokens: tuple[int, ...]
    units: dict[str, tuple[int, ...]]          # spec name -> units holding this block
    children: int = 0


class PrefixIndex:
    def __init__(self, namespace: int, block: int, pools: dict[str, Pool]) -> None:
        self.namespace, self.block, self.pools = namespace, block, pools
        self.entries: collections.OrderedDict[int, Entry] = collections.OrderedDict()   # least recently used first
        self._cached: dict[str, np.ndarray] = {}           # spec name -> bool mask over its pool's units

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

    def match(self, keys: list[int], tokens: list[int]) -> list[Entry]:
        """The cached entries for the longest prefix of `keys`; marks them recently used."""
        found: list[Entry] = []
        for i, key in enumerate(keys):
            e = self.entries.get(key)
            if (e is None or e.parent != (keys[i - 1] if i else None)
                    or e.tokens != tuple(tokens[i * self.block:(i + 1) * self.block])):
                break
            found.append(e)
        for e in found:
            self.entries.move_to_end(e.key)
        return found

    def insert(self, key: int, parent: int | None, tokens: tuple[int, ...], units: dict[str, tuple[int, ...]]) -> None:
        """Publish one block (the key must be new and the parent cached)."""
        for name, us in units.items():
            self.pools[name].incref(us)
            self.cached(name)[us[0] if len(us) == 1 else list(us)] = True
        self.entries[key] = Entry(key, parent, tokens, units)
        if parent is not None:
            self.entries[parent].children += 1

    def cached(self, name: str) -> np.ndarray:
        """Which units of spec `name`'s pool the index references (a live mask, sized with the pool)."""
        mask = self._cached.get(name)
        if mask is None or len(mask) != self.pools[name].capacity:
            mask = self._cached[name] = np.zeros(self.pools[name].capacity, dtype=bool)
        return mask

    def _evictable(self, e: Entry) -> bool:
        return e.children == 0 and all(self.pools[n].held_once(us) for n, us in e.units.items())

    def evict(self, pool: str, need: int) -> None:
        """Drop least-recently-used unused leaves — and the ancestors that become unused leaves, while still short —
        until `pool` has `need` free units or nothing more is evictable."""
        while self.pools[pool].num_free < need:
            # The least recently used unused leaf: a scan that stops there, not a copy of the index (this runs per
            # request when a full pool makes every request in a step evict).
            e = next((e for e in self.entries.values() if self._evictable(e)), None)
            if e is None:
                return
            while e is not None and self._evictable(e):           # up the chain, while still short
                parent = self.entries.get(e.parent) if e.parent is not None else None
                self._drop(e)
                e = parent if self.pools[pool].num_free < need else None

    def _drop(self, e: Entry) -> None:
        del self.entries[e.key]
        if e.parent is not None:
            self.entries[e.parent].children -= 1
        for name, us in e.units.items():
            self.cached(name)[us[0] if len(us) == 1 else list(us)] = False
            self.pools[name].decref(us)

    def clear(self) -> None:
        for e in reversed(list(self.entries.values())):
            self._drop(e)
