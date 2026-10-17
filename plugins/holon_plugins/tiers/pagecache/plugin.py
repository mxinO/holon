"""tiers.pagecache@1: a file tier whose reads go through the OS page cache — the low-RAM tier docs/v1-offload.md
acceptance 1 holds out ("mmap", in effect), a `Tier` (std.tiered's protocol) the design does not design.

As the store's slower tier (`moe.tiers = "std.tiered.pinned@1,tiers.pagecache@1"`, RAM capped by `moe.ram`), it
answers every piece no RAM unit holds with `pread` extents read without O_DIRECT, which `cpu_task@1`'s copy task reads
into pinned staging on the I/O workers. Its own storage decisions: it holds every piece (the checkpoint's file has
them all, and a piece leaving RAM needs no write), takes no pinned or device memory of its own, and leaves residency to
the kernel's page cache — whatever free memory the host has beyond the pinned budget keeps the hot experts' file
pages, so a repeated miss is a memory copy rather than a disk read. Each file is opened buffered and advised random
access (an expert's extent is read whole; readahead past it would only fetch neighbours no step asked for)."""
from __future__ import annotations

import os


class PageCache:
    device_readable = False

    def __init__(self, ctx, **site) -> None:
        self.fds: dict[str, int] = {}
        self.extents: dict[str, list] = {}

    def holds(self, cls: str, pieces: int) -> int:
        return pieces

    def prepare(self, ctx, cls: str, unit_bytes: int, extents, staging: int = 0, granule: int = 0) -> None:
        for parts in extents:
            for e in parts:
                if e.path not in self.fds:
                    fd = self.fds[e.path] = os.open(e.path, os.O_RDONLY)
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
        self.extents[cls] = extents

    def units(self, cls: str):
        raise TypeError("tiers.pagecache@1 is not device-readable")

    def read(self, cls: str, piece: int) -> tuple:
        return "pread", tuple((self.fds[e.path], 0, e.offset, e.nbytes) for e in self.extents[cls][piece])

    def shutdown(self) -> None:
        for fd in self.fds.values():
            os.close(fd)
        self.fds.clear()


class Plugin:
    def register(self, regs) -> None:
        regs.units.add("tiers.pagecache", PageCache)
