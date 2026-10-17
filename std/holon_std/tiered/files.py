"""Reading checkpoint extents into host memory: each file once, in large aligned chunks, with `O_DIRECT` where the
filesystem allows it, so the page cache never holds a second copy of the experts (docs/v1-offload.md D10)."""
from __future__ import annotations

import concurrent.futures
import functools
import mmap
import os

import numpy as np
from holon.api import Extent

ALIGN = 4096                  # O_DIRECT's offset, length and buffer alignment (the largest logical block size)
WINDOW = 64 << 20             # bytes per read
WORKERS = 8


def read(jobs: list[tuple[Extent, np.ndarray]]) -> None:
    """Fill each destination (a uint8 array of its extent's size) with its extent's bytes."""
    by_path: dict[str, list[tuple[Extent, np.ndarray]]] = {}
    for extent, dst in jobs:
        if dst.nbytes != extent.nbytes:
            raise ValueError(f"{extent}: a destination of {dst.nbytes} bytes")
        by_path.setdefault(extent.path, []).append((extent, dst))
    for path, pieces in by_path.items():
        pieces.sort(key=lambda p: p[0].offset)
        windows, lo, run = [], 0, []
        for extent, dst in pieces:          # consecutive pieces share a window while it stays within WINDOW
            if run and _up(extent.offset + extent.nbytes) - lo > WINDOW:
                windows.append((lo, run))
                run = []
            if not run:
                lo = extent.offset // ALIGN * ALIGN
            run.append((extent, dst))
        windows.append((lo, run))
        fd = _open(path)
        try:
            with concurrent.futures.ThreadPoolExecutor(WORKERS) as pool:
                list(pool.map(functools.partial(_read_window, fd, path), *zip(*windows, strict=True)))
        finally:
            os.close(fd)


def _up(n: int) -> int:
    return -(-n // ALIGN) * ALIGN


def _open(path: str) -> int:
    try:
        return os.open(path, os.O_RDONLY | os.O_DIRECT)
    except OSError:                              # e.g. tmpfs: buffered reads
        return os.open(path, os.O_RDONLY)


def _read_window(fd: int, path: str, lo: int, pieces: list[tuple[Extent, np.ndarray]]) -> None:
    hi = _up(max(e.offset + e.nbytes for e, _ in pieces))
    with mmap.mmap(-1, hi - lo) as buf:          # page-aligned, as O_DIRECT needs
        view, got = memoryview(buf), 0
        while got < hi - lo:
            n = os.preadv(fd, [view[got:]], lo + got)
            if n <= 0:
                break
            got += n
        data = np.frombuffer(buf, dtype=np.uint8)
        for extent, dst in pieces:
            if extent.offset + extent.nbytes > lo + got:
                raise OSError(f"{path}: {extent} lies past the file's end")
            dst[...] = data[extent.offset - lo:extent.offset - lo + extent.nbytes]
        del data, view
