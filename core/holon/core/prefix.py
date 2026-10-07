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
        out, key = [], self.namespace if prev is None else prev
        for i in range(start, n_blocks):
            key = hash((key, *tokens[i * self.block:(i + 1) * self.block]))
            out.append(key)
        return out

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
            arr = np.asarray(us, dtype=np.int32)
            self.pools[name].incref(arr)
            self.cached(name)[arr] = True
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
        return e.children == 0 and all((self.pools[n].refcount(np.asarray(us)) == 1).all() for n, us in e.units.items())

    def evict(self, pool: str, need: int) -> None:
        """Drop least-recently-used unused leaves — and the ancestors that become unused leaves — until `pool` has
        `need` free units or nothing more is evictable."""
        for e in list(self.entries.values()):
            if self.pools[pool].num_free >= need:
                return
            while e is not None and e.key in self.entries and self._evictable(e):
                parent = self.entries.get(e.parent) if e.parent is not None else None
                self._drop(e)
                e = parent

    def _drop(self, e: Entry) -> None:
        del self.entries[e.key]
        if e.parent is not None:
            self.entries[e.parent].children -= 1
        for name, us in e.units.items():
            arr = np.asarray(us, dtype=np.int32)
            self.cached(name)[arr] = False
            self.pools[name].decref(arr)

    def clear(self) -> None:
        for e in reversed(list(self.entries.values())):
            self._drop(e)
