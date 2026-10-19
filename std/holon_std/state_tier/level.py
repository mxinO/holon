"""The `Level` protocol: a storage level below std.state_tier's host level (docs/v1-kv-tiers.md D3). Built-in names or
`Ref`s in `state_tier.levels` bind them, slowest last. Plugins implement it structurally (they cannot import std),
with `holon.api` types and builtins only.

The tier decides every placement, refusal and eviction on logical unit counts; a level moves bytes and reports rates.
Its methods are called on the tier's I/O threads (`iothreads.py`) and block until done, except `bind`, `restore` and
`close`, called on the engine thread. The tier never runs two calls on one copy at once, and calls `drop` for a copy
only once no call on it is queued or running.

- A copy is `(key, spec)`: `spec` None for the block's pages (every paged spec's), else a payload spec.
- `shapes`: host pool name -> (unit bytes, segments, share): the level's share of its storage for that shape. The tier
  gives every paged pool, and every payload pool, as many units (std.state_tier's split of `host.bytes`).
- `src` and `dst` map a shape to the copy's units of it, one entry per unit: a sequence of that unit's writable
  segment buffers (`memoryview`s), an iovec for `preadv` / `pwritev`. Unit-major page units have one segment; payload
  units (segment-major, in the kind's own shape) have one per segment.
- `rates()`: `{"read": bytes/s, "write": bytes/s, "mixed": factor, "lane": str | None}`; levels naming the same lane
  share one virtual lane of the tier's clock.
- `restore()` (persistent levels): the copies it kept, as `BlockRecord`s with `units` mapping each paged spec to ()
  and `payloads` each spec to `Payload((0,) * n, shape)`, n the payload's unit count; parents first. `close(records)`:
  persist the copies the records name (the same encoding), or nothing.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol


class Level(Protocol):
    def bind(self, ctx: Any, shapes: Mapping[str, tuple[int, int, float]], identity: str | None) -> Mapping[str, int]:
        """Allocate storage; the units it holds per shape."""

    def write(self, key: int, spec: str | None, src: Mapping[str, Sequence[Sequence[memoryview]]]) -> None: ...

    def read(self, key: int, spec: str | None, dst: Mapping[str, Sequence[Sequence[memoryview]]]) -> None: ...

    def drop(self, key: int, spec: str | None) -> None: ...

    def rates(self) -> Mapping[str, object]: ...

    def restore(self) -> list: ...

    def close(self, records: Sequence) -> None: ...
