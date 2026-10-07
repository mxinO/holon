"""PrefixIndex: content-addressed reuse of per-request state across requests (design §6.3).

Keys are a hash chain over full token blocks — block i's key hashes block i-1's key with block i's tokens, rooted in
the engine's namespace — so a key identifies the whole prefix. An entry holds, per state spec, the pool units the
spec published for that block; the index keeps one reference on each. Entries form a tree (a prefix shared by
several continuations); eviction takes least-recently-used leaves whose units nobody but the index still uses, so
evicting never strands a reachable entry.
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

    def keys(self, tokens: list[int], start: int, n_blocks: int, prev: int | None) -> list[int]:
        """Chain keys of blocks start … n_blocks-1 of `tokens`, continuing from the key of block start-1."""
        out, key = [], self.namespace if prev is None else prev
        for i in range(start, n_blocks):
            key = hash((key, *tokens[i * self.block:(i + 1) * self.block]))
            out.append(key)
        return out

    def match(self, keys: list[int], tokens: list[int]) -> list[Entry]:
        """The cached entries for the longest prefix of `keys` (tokens are verified, so a hash collision is a miss);
        marks them recently used."""
        found = []
        for i, key in enumerate(keys):
            e = self.entries.get(key)
            if e is None or e.tokens != tuple(tokens[i * self.block:(i + 1) * self.block]):
                break
            found.append(e)
        for e in found:
            self.entries.move_to_end(e.key)
        return found

    def insert(self, key: int, parent: int | None, tokens: tuple[int, ...], units: dict[str, tuple[int, ...]]) -> bool:
        """Publish one block; False if it is already cached (the caller keeps its units private then)."""
        if key in self.entries or (parent is not None and parent not in self.entries):
            return False
        for name, us in units.items():
            self.pools[name].incref(np.asarray(us, dtype=np.int32))
        self.entries[key] = Entry(key, parent, tokens, units)
        if parent is not None:
            self.entries[parent].children += 1
        return True

    def _evictable(self, e: Entry) -> bool:
        return e.children == 0 and all((self.pools[n].refcount(np.asarray(us)) == 1).all() for n, us in e.units.items())

    def evict(self, pool: str, need: int) -> None:
        """Drop least-recently-used unused leaves until `pool` has `need` free units (or nothing is evictable)."""
        progress = True
        while self.pools[pool].num_free < need and progress:
            progress = False
            for e in list(self.entries.values()):
                if self._evictable(e):
                    self._drop(e)
                    progress = True
                    if self.pools[pool].num_free >= need:
                        break

    def _drop(self, e: Entry) -> None:
        del self.entries[e.key]
        if e.parent is not None:
            self.entries[e.parent].children -= 1
        for name, us in e.units.items():
            self.pools[name].decref(np.asarray(us, dtype=np.int32))

    def owned(self, pool: str) -> set[int]:
        """Units of `pool` the index holds a reference on."""
        return {u for e in self.entries.values() for u in e.units.get(pool, ())}

    def clear(self) -> None:
        for e in reversed(list(self.entries.values())):
            self._drop(e)
