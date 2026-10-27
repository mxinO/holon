"""std.tiered.gguf_direct@1 (F9, docs/dev/v1-offload.md D10): the SSD tier — every piece where the checkpoint's file
holds
it, read with `O_DIRECT` (buffered where the filesystem has none) by `cpu_task@1`'s copy task on the dedicated I/O
workers into pinned staging, so the page cache never holds a second copy. Always valid: a piece leaving RAM needs no
write. Its descriptors are the pieces' extents."""
from __future__ import annotations

import os

from . import files


class GgufDirect:
    device_readable = False

    def __init__(self, ctx, **site) -> None:
        self.fds: dict[str, tuple[int, bool]] = {}
        self.extents: dict[str, list] = {}

    def holds(self, cls: str, pieces: int) -> int:
        return pieces

    def prepare(self, ctx, cls: str, unit_bytes: int, extents, staging: int = 0, granule: int = 0) -> None:
        for parts in extents:
            for e in parts:
                if e.path not in self.fds:
                    self.fds[e.path] = files._open(e.path)
        self.extents[cls] = extents

    def units(self, cls: str):
        raise TypeError("std.tiered.gguf_direct@1 is not device-readable")

    def read(self, cls: str, piece: int) -> tuple:
        return "pread", tuple((*self.fds[e.path], e.offset, e.nbytes) for e in self.extents[cls][piece])

    def shutdown(self) -> None:
        for fd, _ in self.fds.values():
            os.close(fd)
        self.fds.clear()
