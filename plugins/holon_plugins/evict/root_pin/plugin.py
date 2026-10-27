"""`evict.root_pin` (slot `eviction_policy`, docs/dev/v1-kv-tiers.md §5): Strata's "root pinned, others least recently
used". Among the eligible device entries the core offers (least recently used first), it skips those at depth ≤ d
whose blocks were matched at least h times, while any other is eligible: a shared system prompt's blocks stay on the
device under pressure, and go only when nothing else can. Options `root_pin.depth` (d) and `root_pin.hits` (h)."""
from __future__ import annotations


class RootPin:
    def __init__(self, depth: int, hits: int) -> None:
        self.depth, self.hits = depth, hits

    def victim(self, candidates):
        pinned = None
        for e in candidates:                         # the scan stops at the first unpinned entry
            if e.depth > self.depth or e.hits < self.hits:
                return e
            pinned = pinned or e
        return pinned


class Plugin:
    def register(self, regs) -> None:
        pass                                         # no kernels or units: the slot's factory is all it has

    def build(self, ctx) -> RootPin:
        opts = ctx.options.get("plugins", {})
        return RootPin(int(opts.get("root_pin.depth", 0)), int(opts.get("root_pin.hits", 2)))
