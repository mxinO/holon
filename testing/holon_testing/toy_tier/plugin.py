"""A toy state tier (slot `tier`) that proves the core's tier seams (docs/v1-kv-tiers.md §4) before std's does: host
copies of blocks in host pools shaped like the device pools it copies, its stores and loads on two streams the core
gives it (deferred under `holon_judge.deferred`, which its `before_launch` pumps), a clock of one per launch, seeded
page refusals and own drops (interior blocks included), persistence to a file at close keyed by the engine's identity.
It keeps the determinism contract and checks the core's side of it: `loaded` at most once with a fence and once with
None, nothing loaded or dropped that it must still hold.

Options: `toy_tier.scale` (host units per device unit), `toy_tier.refuse`, `toy_tier.drops`, `toy_tier.delay`,
`toy_tier.seed`, `toy_tier.path` (planned = false)."""
from __future__ import annotations

import collections
import os
import random
import types
from typing import Any

import torch
from holon.api import BlockRecord, Payload


class Plugin:
    def register(self, regs: Any) -> None:
        pass

    def build(self, ctx: Any) -> ToyTier:
        return ToyTier(ctx.options.get("plugins", {}))


def bind(index: Any, ledger: Any, options: dict, stream: Any, payloads: tuple[str, ...] = ()) -> ToyTier:
    """The toy tier on a bare prefix index (core tests without an engine): attached with `ledger`'s host memory, its
    copies on the streams `stream()` makes, the payload pools named `payloads`."""
    tier = ToyTier(options)

    def host_pool(name, *, unit_bytes, segments, units, mapped=False):
        return ledger.plugin_pool(name, unit_bytes=unit_bytes, segments=segments, min_units=units, pref_units=units,
                                  device="host", mapped=mapped, late=True)
    index.restore(tier.attach(types.SimpleNamespace(
        pools=dict(index.pools) | {k: index.payload_pools[k] for k in payloads}, identity=None, block=index.block,
        options=options, host_pool=host_pool, stream=stream)))
    index.tier = tier
    return tier


class ToyLoad:
    def __init__(self, fence: Any, due: float, payloads: dict, keys: list[int], read: list) -> None:
        self.fence, self.due, self.payloads, self.keys, self.read, self.told = fence, due, payloads, keys, read, set()

    def wait(self) -> None:
        self.fence.wait()

    def synchronize(self) -> None:
        self.fence.synchronize()


class ToyTier:
    def __init__(self, options: dict) -> None:
        get = lambda k, d: options.get(f"toy_tier.{k}", d)        # noqa: E731
        self.scale, self.refuse, self.drops = int(get("scale", 4)), float(get("refuse", 0)), float(get("drops", 0))
        self.delay, self.path, self.rng = int(get("delay", 0)), get("path", None), random.Random(int(get("seed", 0)))
        self.t = 0.0
        self.copies: dict[tuple[int, str | None], dict[str, list[int]]] = {}   # (key, None: pages | spec) -> units
        self.blocks: dict[int, tuple[int | None, tuple[int, ...]]] = {}       # key -> (parent, tokens)
        self.fences: dict[tuple[str, int], list] = collections.defaultdict(list)   # host unit -> before its reuse
        self.written: dict[tuple[str, int], Any] = {}                          # host unit -> the store that filled it
        self.loading: collections.Counter = collections.Counter()              # key -> page loads not yet `loaded`
        self.reading: collections.Counter = collections.Counter()              # host unit -> loads not yet `loaded`
        self.limbo: set[tuple[str, int]] = set()                               # released while still read
        self.pending: list[tuple[int, str | None]] = []
        self.counts: collections.Counter = collections.Counter()
        self.calls: list[tuple] = []                                           # what the core asked, in order
        self.tentative: set[tuple[int, str]] = set()                           # tentative payload copies
        self.loads: list[ToyLoad] = []                                         # every load it made

    # -- attach, persistence -----------------------------------------------------------------------------------------
    def attach(self, ctx: Any) -> list[BlockRecord]:
        self.ctx, self.dev, self.d2h, self.h2d = ctx, dict(ctx.pools), ctx.stream(), ctx.stream()   # as a GPU tier
        self.host = {k: ctx.host_pool(f"toy_tier.{k}", unit_bytes=p.unit_bytes, segments=p.segments,
                                      units=self.scale * p.capacity) for k, p in self.dev.items()}
        self.free = {k: list(range(p.capacity - 1, -1, -1)) for k, p in self.host.items()}
        if not self.path or not os.path.exists(self.path):
            return []
        saved = torch.load(self.path, weights_only=False)
        os.unlink(self.path)                          # read once: what it names may be rewritten from now on
        if saved["identity"] != ctx.identity or ctx.identity is None:
            return []
        for (key, spec), data in saved["copies"].items():
            units = self.copies[key, spec] = {k: self._take(k, v.shape[1]) for k, v in data.items()}
            for k, v in data.items():
                self._units(self.host[k])[:, units[k]] = v
        self.blocks = saved["blocks"]
        depth = lambda key: 0 if self.blocks[key][0] is None else 1 + depth(self.blocks[key][0])    # noqa: E731
        return [BlockRecord(key, *self.blocks[key], {spec: () for spec in self.copies.get((key, None), {})},
                            {s: Payload((), next(iter(us))) for (k, s), us in self.copies.items() if k == key and s})
                for key in sorted(self.blocks, key=depth)]

    def close(self) -> None:
        if self.path and self.ctx.identity is not None:
            for fences in self.fences.values():
                for f in fences:
                    f.synchronize()
            torch.save({"identity": self.ctx.identity, "blocks": {k: self.blocks[k] for k, _ in self.copies},
                        "copies": {key: {k: self._units(self.host[k])[:, us].clone() for k, us in units.items()}
                                   for key, units in self.copies.items()}}, self.path)

    # -- the protocol ----------------------------------------------------------------------------------------------
    def clock(self) -> float:
        return self.t

    def store(self, blocks, after, evicting):
        self.calls.append(("store", [b.key for b in blocks], evicting))
        accepted, jobs = set(), []
        for b in blocks:
            if b.units and (b.key, None) not in self.copies and self.rng.random() >= self.refuse:
                host = self._alloc({spec: len(us) for spec, us in b.units.items()})
                if host is not None:
                    self.copies[b.key, None] = host
                    jobs += [(spec, us, host[spec]) for spec, us in b.units.items()]
            accepted |= {(b.key, None)} if (b.key, None) in self.copies else set()
            for spec, p in b.payloads.items():
                while (host := self._alloc({p.pool: len(p.units)})) is None:     # payload stores are mandatory
                    self._evict(next(k for k in self.copies if k[1] is not None and not self.loading[k[0]]))
                self.copies[b.key, spec] = host
                (self.tentative.add if p.tentative else self.tentative.discard)((b.key, spec))
                jobs.append((p.pool, p.units, host[p.pool]))
                accepted.add((b.key, spec))
            self.blocks[b.key] = (b.parent, tuple(b.tokens))
        reuse = [f for k, _, hu in jobs for u in hu for f in self.fences.pop((k, u), [])]
        fence = self.d2h.run(lambda: [self._copy(self.dev[k], su, self.host[k], hu) for k, su, hu in jobs],
                                [*after, *reuse])
        for k, _, hu in jobs:
            for u in hu:
                self.written[k, u] = fence
                self.fences[k, u].append(fence)
        self.counts["stores"] += 1
        return accepted, fence

    def load(self, keys, units, payloads, after):
        self.calls.append(("load", list(keys), list(payloads), list(after)))
        jobs, read, out = [], [], {}
        for i, key in enumerate(keys):
            assert (key, None) in self.copies, "the core loads a page copy the tier dropped"
            host = self.copies[key, None]
            for spec, us in units.items():
                per = len(us) // len(keys)
                jobs.append((spec, host[spec], us[i * per:(i + 1) * per]))
                read += [(spec, u) for u in host[spec]]
            self.loading[key] += 1
        for key, spec in payloads:
            assert (key, spec) in self.copies, "the core loads a payload copy the tier dropped"
            (pool, hu), = self.copies[key, spec].items()
            out[key, spec] = Payload(tuple(hu), self.host[pool].name)
            read += [(pool, u) for u in hu]
        self.reading.update(read)
        stream = self.h2d if jobs else self.d2h      # a payload-only load copies nothing: after the stores it reads
        fence = stream.run(lambda: [self._copy(self.host[k], hu, self.dev[k], du) for k, hu, du in jobs],
                           [*after, *{id(f): f for r in read if (f := self.written.get(r)) is not None}.values()])
        self.counts["loads"] += 1
        self.counts["loaded_blocks"] += len(keys)
        self.loads.append(ToyLoad(fence, self.t + self.delay, out, list(keys), read))
        return self.loads[-1]

    def loaded(self, load, done):
        kind = "none" if done is None else "fence"
        assert kind not in load.told, f"loaded(…, {kind}) twice"
        if not load.told:
            for key in load.keys:
                self.loading[key] -= 1
            for r in load.read:                                 # read: may go once the fences below are done
                self.reading[r] -= 1
                if not self.reading[r] and r in self.limbo:
                    self.limbo.discard(r)
                    self.free[r[0]].append(r[1])
        load.told.add(kind)
        for r in load.read:                                     # its readers end before the units' reuse
            self.fences[r].append(done if done is not None else load.fence)
        self.t = max(self.t, load.due)
        self.counts["abandoned" if done is None else "waiterless" if done is load else "loaded"] += 1

    def hit(self, keys, adopted) -> None:
        self.calls.append(("hit", list(keys), list(adopted)))
        self.counts["hits"] += 1

    def settled(self, payloads) -> None:
        assert all(k in self.tentative or k not in self.copies for k in payloads), "settled an ordinary payload"
        self.tentative -= set(payloads)

    def untold(self) -> list[ToyLoad]:
        """The loads the core never told it about (`loaded`): there should be none once no request runs."""
        return [load for load in self.loads if not load.told]

    def drop(self, keys, spec=None) -> None:
        """A payload being adopted may go (its host units stay until `loaded`); pages being loaded may not."""
        for key in keys:
            assert spec is not None or not self.loading[key], "the core dropped a block it is loading"
            for k in [k for k in self.copies if k[0] == key and (spec is None or k[1] == spec)]:
                self._release(k)

    def dropped(self):
        out, self.pending = self.pending, []
        return out

    def clear(self) -> None:
        for k in list(self.copies):
            self._release(k)

    def before_launch(self, plan):
        getattr(self.d2h.__dict__.get("owner"), "pump", lambda: None)()      # a deferred stream: a launch went by
        self.t += 1
        if self.copies and self.rng.random() < self.drops:       # its own eviction, interior blocks included
            held = [k for k in self.copies if not self.loading[k[0]]]
            if held:
                self._evict(self.rng.choice(held))
        return [], []

    def stats(self) -> dict[str, int]:
        return dict(self.counts)

    # -- storage ---------------------------------------------------------------------------------------------------
    def _evict(self, k) -> None:
        self._release(k)
        self.pending.append(k)
        self.counts["dropped"] += 1

    def _release(self, k) -> None:
        self.tentative.discard(k)
        for pool, us in self.copies.pop(k).items():
            for u in us:
                if self.reading[pool, u]:                       # a load not yet `loaded` reads it: kept till then
                    self.limbo.add((pool, u))
                else:
                    self.free[pool].append(u)

    def _take(self, pool: str, n: int) -> list[int]:
        """The least recently freed units: their fences are the likeliest to be done, so a missing wait shows."""
        units = self.free[pool][:n]
        del self.free[pool][:n]
        return units

    def _alloc(self, need: dict[str, int]) -> dict[str, list[int]] | None:
        if any(len(self.free[k]) < n for k, n in need.items()):
            return None
        return {k: self._take(k, n) for k, n in need.items()}

    @staticmethod
    def _units(pool) -> torch.Tensor:
        return pool.view(torch.uint8, (pool.segments, pool.capacity, pool.unit_bytes // pool.segments))

    def _copy(self, src, src_units, dst, dst_units) -> None:
        self._units(dst)[:, list(dst_units)] = self._units(src)[:, list(src_units)]
