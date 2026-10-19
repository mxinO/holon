"""The `Level` protocol: a storage level below std.state_tier's host level (docs/v1-kv-tiers.md D3). `state_tier.levels`
binds one: the built-in `ssd`, or a `Ref` to a level unit. Plugins implement it structurally (they cannot import std),
with `holon.api` types and builtins only.

The tier decides every placement, refusal and eviction, and chooses the slots of every write (`slots`: per shape, the
level's unit indices `0 … capacity - 1`), so a level never waits for room; a level moves bytes and reports rates.
`write`, `read` and `drop` are blocking calls the tier makes on its own I/O threads (`state_tier.io_threads`: a level's
concurrency is that thread count); the tier never runs two calls on one copy at once, writes a slot only once every
call on its previous copy has returned, and calls `drop` once none on the copy is queued or running. `bind`, `rates`,
`restore` and `close` are called on the engine thread.

- `ctx` is the `holon.api.TierCtx` the tier's `attach` got.
- A copy is `(key, spec)`: `spec` None for the block's pages (every paged spec's), else a payload spec.
- `shapes`: host pool name -> (unit bytes, segments, share): the level's share of its storage for that shape (the
  tier's split of `host.bytes`, so every paged pool, and every payload pool, gets as many units). `bind` returns the
  units it holds per shape.
- `src` and `dst` map a shape to the copy's units of it, one entry per unit: a sequence of that unit's writable
  segment buffers (`memoryview`s), an iovec for `preadv` / `pwritev`. Unit-major page units have one segment; payload
  units (segment-major, in the kind's own shape) have one per segment. A level without slots ignores `slots`.
- `rates()`: `{"read": bytes/s, "write": bytes/s, "mixed": factor, "read_beside": factor, "lane": str | None}`.
  `mixed` divides the write rate of a write a queued read waits for; `read_beside` multiplies the read rate of a read
  decided while background writes are running (D4); levels naming the same lane share one virtual lane.
- `restore()` (persistent levels): the copies it kept, parents first, as `BlockRecord`s whose `units` map each paged
  spec to the copy's slots and whose `payloads` map each spec to `Payload(slots, shape)`. `close(records)`: persist
  the copies the records name (`units`: the paged specs; `payloads`: each payload's shape and unit count), or nothing.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol


class Level(Protocol):
    def bind(self, ctx: Any, shapes: Mapping[str, tuple[int, int, float]], identity: str | None) -> Mapping[str, int]:
        """Allocate storage; the units it holds per shape."""

    def write(self, key: int, spec: str | None, src: Mapping[str, Sequence[Sequence[memoryview]]],
              slots: Mapping[str, Sequence[int]]) -> None: ...

    def read(self, key: int, spec: str | None, dst: Mapping[str, Sequence[Sequence[memoryview]]]) -> None: ...

    def drop(self, key: int, spec: str | None) -> None: ...

    def rates(self) -> Mapping[str, object]: ...

    def restore(self) -> list: ...

    def close(self, records: Sequence) -> None: ...
