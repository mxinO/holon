"""`state_tiers.zlib`: a `Level` below std.state_tier's host level (docs/v1-kv-tiers.md §1's held-out proof, D3) that
the protocol was not shaped around. It keeps lossless zlib blobs of variable size in plain memory instead of fixed
slots, compressing and decompressing on the tier's I/O threads (CPU threads) between its blobs and the host units.
It takes pages and payloads alike when the host level evicts them, and reports CPU rates for the tier's clock. Its
capacity counts uncompressed units, so the tier decides its refusals before any compression (deterministic);
compression only saves memory. Bound by `state_tier.levels = state_tiers.zlib@1`.

Options: `zlib.bytes` (the uncompressed capacity, split over the host pools as the tier asks), `zlib.level` (zlib's,
default 1), `zlib.read` and `zlib.write` (the bytes per second the clock charges: decompress, compress)."""
from __future__ import annotations

import re
import threading
import zlib

SIZES = {"": 1, "k": 1 << 10, "kib": 1 << 10, "m": 1 << 20, "mib": 1 << 20, "g": 1 << 30, "gib": 1 << 30}


def _bytes(text: object) -> int:
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([a-zA-Z]*)\s*", str(text))
    if m is None or m[2].lower() not in SIZES:
        raise ValueError(f"zlib.bytes: {text!r} is not a byte size (e.g. 4GiB)")
    return int(float(m[1]) * SIZES[m[2].lower()])


class ZlibLevel:
    def __init__(self, ctx, options) -> None:
        self.nbytes = _bytes(options.get("zlib.bytes", "4GiB"))
        self.level = int(options.get("zlib.level", 1))
        self.speed = {"read": float(options.get("zlib.read", 1.0e9)), "write": float(options.get("zlib.write", 3.0e8))}
        self.blobs: dict = {}
        self.lock = threading.Lock()

    def bind(self, ctx, shapes, identity) -> dict[str, int]:
        return {name: int(self.nbytes * share) // unit_bytes for name, (unit_bytes, _, share) in shapes.items()}

    def write(self, key, spec, src) -> None:
        blobs = {name: [zlib.compress(b"".join(unit), self.level) for unit in units] for name, units in src.items()}
        with self.lock:
            self.blobs[key, spec] = blobs

    def read(self, key, spec, dst) -> None:
        with self.lock:
            blobs = self.blobs[key, spec]
        for name, units in dst.items():
            for unit, blob in zip(units, blobs[name], strict=True):
                data, at = zlib.decompress(blob), 0
                for mv in unit:
                    mv[:] = data[at:at + len(mv)]
                    at += len(mv)

    def drop(self, key, spec) -> None:
        with self.lock:
            self.blobs.pop((key, spec), None)

    def rates(self) -> dict[str, object]:
        return self.speed | {"mixed": 1.0, "lane": "zlib:cpu"}

    def stored(self) -> int:
        """Bytes its blobs take (what compression saves against the uncompressed capacity)."""
        with self.lock:
            return sum(len(b) for copy in self.blobs.values() for units in copy.values() for b in units)

    def restore(self) -> list:
        return []

    def close(self, records) -> None:
        self.blobs.clear()


class Plugin:
    def register(self, regs) -> None:
        regs.units.add("state_tiers.zlib", ZlibLevel)
