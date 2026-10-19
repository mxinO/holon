"""State tiers below the device (docs/v1-kv-tiers.md §4): the slot `tier` keeps copies of prefix-index blocks — pages
and payloads — that device memory cannot hold, and the slot `eviction_policy` picks the device's victims.

The core owns the device boundary: whether an entry is resident, device eviction (a victim the tier holds is demoted,
any other is offered as an eviction store or dropped), promotion into device units, and gating a request on its loads.
The tier owns everything below: its levels, their capacity and eviction, its copies (on its own streams), its clock.

**The determinism contract** (the tier's obligation; the judge checks it):
- every return value and every `dropped()` list is a function of the calls received and the step hooks
  (`before_launch`, `after_commit`, which the core calls on the tier directly), never of copy or I/O completion;
- payload stores are accepted;
- reuse of its own storage waits on its own tickets; reuse of a host unit, or CPU I/O on it, waits on all the unit's
  outstanding fences — stores' ends, and every `done` given to `loaded`, which add and never replace;
- I/O jobs run first-in first-out in the order they were decided; abandoning a load (`loaded(load, None)`) completes
  an enqueue it still owes as a no-op and wakes its waiters;
- every device read of a host unit follows the copy that wrote it: in the `after` of the tier's own copy, and inside
  a `Load`'s fence for the host payloads a kind's adopt reads;
- a copy being loaded is never dropped before `loaded`, and `loaded` advances the clock to at least the load's due
  time.
Loads and fences are compared by identity.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .memory import Pool
    from .state import Payload


@dataclasses.dataclass(frozen=True, slots=True)
class BlockRecord:
    """One prefix-index entry as a tier sees it. In `store`: `units` maps each paged spec to the device units of its
    pages (empty: payloads only) and `payloads` each spec to its payload (device units); in what `attach` restores:
    `units` maps each spec whose pages the tier holds to (), `payloads` each payload it holds to `Payload((), pool)`."""

    key: int
    parent: int | None
    tokens: tuple[int, ...]
    units: Mapping[str, tuple[int, ...]]
    payloads: Mapping[str, Payload]


class Fence(Protocol):
    """A point in device order: the core's step ends, the tier's copies and loads."""

    def wait(self) -> None:
        """Make the work enqueued next on the current stream wait for it (the host first waits until it is enqueued)."""

    def synchronize(self) -> None:
        """Wait for it on the host."""

    def done(self) -> bool:
        """Whether it has passed, without waiting (for statistics: no decision may read it)."""


class Stream(Protocol):
    """A copy stream the core makes for a tier (`TierCtx.stream`); off the GPU its work runs at once, in order."""

    def run(self, fn: Callable[[], object], after: Sequence[Fence] = ()) -> Fence:
        """Enqueue `fn()`'s device work on this stream after `after`; the fence it returns follows that work."""


class Load(Fence, Protocol):
    """A tier's copy of blocks into device units or payloads to host units; its fence follows the copy."""

    due: float                                    # when the request may run, on the tier's clock
    payloads: Mapping[tuple[int, str], Payload]   # (key, spec) -> the host copy a kind's `adopt` reads


class TierCtx(Protocol):
    """What `StateTier.attach` gets."""

    pools: Mapping[str, Pool]     # the device pools it copies: each paged spec's by its spec's name (never its
                                  # pool's), each payload pool by its own name; a key equal to its pool's `name` is
                                  # a payload pool
    identity: str | None          # the engine's state identity (`plans.state_identity`); None if it has no plan
    max_seqs: int
    max_model_len: int
    block: int                    # tokens per prefix-index block
    options: Mapping[str, object]   # the deployment's plugin options

    def host_pool(self, name: str, *, unit_bytes: int, segments: int, units: int, mapped: bool = False) -> Pool:
        """A host pool of exactly `units` units, allocated now within `EngineConfig.host_memory` (after the device's
        pools: the one exception to requesting host pools by `before_materialize`)."""

    def stream(self) -> Stream: ...


class StateTier(Protocol):
    """The slot `tier` (single, opt-in: `EngineConfig.slots["tier"]`), built with the engine by its plugin
    (`hooks.StateTierPlugin`). Optional hooks (`holon.api.hooks`): EditsBeforeLaunch, ObservesCommits, Reports."""

    def attach(self, ctx: TierCtx) -> list[BlockRecord]:
        """Once, after the device pools' final materialize and before graph capture: allocate its storage; the blocks
        it restored, parents first."""

    def clock(self) -> float:
        """Its virtual time at the step being scheduled: a request runs once it reaches its loads' `due`."""

    def store(self, blocks: Sequence[BlockRecord], after: Sequence[Fence], evicting: bool
              ) -> tuple[set[tuple[int, str | None]], Fence]:
        """Copy the blocks' pages and payloads from their device units after `after`: what it accepted, as (key,
        spec) with None for pages, and the fence after which their device units may be reused. A page store may be
        refused, also at eviction (`evicting`: the block is dropped then); a payload store may not."""

    def load(self, keys: Sequence[int], units: Mapping[str, Sequence[int]], payloads: Sequence[tuple[int, str]],
             after: Sequence[Fence]) -> Load:
        """Copy the pages of `keys` into the device `units` (per spec, in key order) after `after`, and bring the
        payloads (key, spec) to host units."""

    def loaded(self, load: Load, done: Fence | None) -> None:
        """At most once with a fence — the end of the first step that read `load`, or the load's own fence once no
        request waits for it and no step read it — and at most once with None: every block of `load` left the device
        before any step read it, so a copy it has not enqueued yet is skipped."""

    def hit(self, keys: Sequence[int], adopted: Sequence[tuple[int, str]]) -> None:
        """A request matched `keys` and adopted these payloads (recency)."""

    def settled(self, payloads: Sequence[tuple[int, str]]) -> None:
        """These tentative payloads (key, spec) are ordinary from now on: their request was released."""

    def drop(self, keys: Sequence[int], spec: str | None = None) -> None:
        """The index dropped these blocks (spec None: every copy) or their payload of `spec`."""

    def dropped(self) -> list[tuple[int, str | None]]:
        """Its own evictions since the last call: (key, None) for pages, (key, spec) for a payload. The core polls it
        at each commit, after each store and after each load, so a block it drops is never loaded — provided it
        evicts only within `store`, `load` and its step hooks."""

    def clear(self) -> None:
        """Drop every copy (`Engine.flush_prefix`)."""

    def close(self) -> None:
        """At `Engine.close`: persist what it keeps across restarts, release its storage."""


class EntryView(Protocol):
    """A resident prefix-index entry as an eviction policy sees it."""

    key: int
    depth: int                    # blocks before it in its chain
    hits: int                     # matches that reached it


class EvictionPolicy(Protocol):
    """The slot `eviction_policy` (single, opt-in), built by its plugin (`hooks.EvictionPolicyPlugin`): which device
    entry to evict next. Without one, the least recently used eligible entry."""

    def victim(self, candidates: Iterator[EntryView]) -> EntryView | None:
        """One of `candidates` (the eligible entries, least recently used first, produced lazily: bound the scan)."""
