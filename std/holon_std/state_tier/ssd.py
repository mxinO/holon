"""std.state_tier's SSD level, the built-in `Level` `ssd` (docs/v1-kv-tiers.md D11).

Layout. One preallocated slot file per host pool shape in `ssd.dir` (`posix_fallocate`), its slots
`unit bytes` rounded up to 4 KiB apart, sized by the shape's share of `ssd.bytes`. `bind` takes an exclusive `flock`
on the directory and refuses one another engine holds (runtime reads are not crc-checked, so two engines must never
share slots), and one without the bytes.

I/O. `os.preadv` / `os.pwritev` of whole units, on the tier's I/O threads. With `ssd.direct` (the default) a unit
whose segments and offset are 4 KiB-aligned goes through an `O_DIRECT` descriptor; otherwise, or where the filesystem
refuses `O_DIRECT`, buffered, with `POSIX_FADV_DONTNEED` after. A write records the crc32 of the bytes it wrote.

Persistence (`ssd.persist = close`). `close` writes a snapshot: the slot files `fdatasync`ed, then the manifest (the
identity, the geometry and one record per block: key, parent, tokens, slot and crc32 per unit of each copy) written
to a temporary file, `fsync`ed, renamed into place and the directory `fsync`ed. `bind` reads the manifest, then
unlinks it and `fsync`s the directory before any slot can be rewritten, so a record never names a slot reused after it
was written. `restore` keeps a record only if its identity matches, every slot lies inside the current geometry and
holds its crc (verified on threads), and its parent is kept (parents first); a crash leaves no manifest: nothing is
restored."""
from __future__ import annotations

import collections
import concurrent.futures
import fcntl
import json
import os
import pathlib
import threading
import zlib
from collections.abc import Mapping, Sequence

import numpy as np
from holon.api import BlockRecord, Payload

ALIGN = 4096
MANIFEST = "manifest.json"
RATES = {"read": 2.17e9, "write": 1.77e9, "mixed": 0.79}      # K-M4 on the RTX 5080's drive (tune measures them)


def _aligned(unit: Sequence[memoryview], offset: int) -> bool:
    return offset % ALIGN == 0 and all(len(mv) % ALIGN == 0 and np.frombuffer(mv, np.uint8).ctypes.data % ALIGN == 0
                                       for mv in unit)


def _fsync_dir(path: pathlib.Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class SsdLevel:
    def __init__(self, options: Mapping[str, object], threads: int) -> None:
        if not options.get("dir"):
            raise ValueError("state_tier.ssd.dir: the directory of the SSD level's slot files")
        self.root, self.threads = pathlib.Path(str(options["dir"])), threads
        self.nbytes = int(options["bytes"])
        self.direct = str(options.get("direct") or "true").lower() not in ("0", "false", "off")
        self.persist = str(options.get("persist") or "off")
        if self.persist not in ("off", "close"):
            raise ValueError(f"state_tier.ssd.persist {self.persist!r}: off or close")
        self.overrides = {k: float(options[k]) for k in ("read", "write", "mixed") if options.get(k)}
        self.cv = threading.Condition()
        self.held: dict[tuple[int, str | None], dict[str, list[int]]] = {}     # copy -> shape -> slots
        self.crcs: dict[tuple[int, str | None], dict[str, list[int]]] = {}
        self.manifest: dict | None = None

    # -- bind, rates ---------------------------------------------------------------------------------------------
    def bind(self, ctx, shapes: Mapping[str, tuple[int, int, float]], identity: str | None) -> dict[str, int]:
        if self.persist == "close" and identity is None:
            raise RuntimeError("state_tier.ssd.persist = close needs the engine's state identity (its plan cannot be "
                               "computed)")
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = open(self.root / "lock", "a+b")                 # noqa: SIM115 — held until close
        try:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise RuntimeError(f"state_tier.ssd.dir {self.root}: another engine holds it") from None
        self.identity, self.geometry = identity, {}
        for name, (unit_bytes, _, share) in shapes.items():
            stride = -(-unit_bytes // ALIGN) * ALIGN
            self.geometry[name] = (unit_bytes, int(self.nbytes * share) // stride, stride)
        files = {name: self.root / f"{name}.slots" for name in shapes}
        need = sum(n * stride - (files[k].stat().st_size if files[k].exists() else 0)
                   for k, (_, n, stride) in self.geometry.items())
        st = os.statvfs(self.root)
        if need > st.f_bavail * st.f_frsize:
            self.lock.close()
            raise RuntimeError(f"state_tier.ssd.dir {self.root}: needs {need} more bytes for state_tier.ssd.bytes, "
                               f"has {st.f_bavail * st.f_frsize}")
        path = self.root / MANIFEST                                  # read, then gone before any slot is rewritten
        if path.exists():
            self.manifest = json.loads(path.read_text())
            path.unlink()
            _fsync_dir(self.root)
        self.fd, self.dfd, self.free = {}, {}, {}
        for name, (_, n, stride) in self.geometry.items():
            fd = self.fd[name] = os.open(files[name], os.O_RDWR | os.O_CREAT, 0o600)
            os.ftruncate(fd, n * stride)
            try:
                os.posix_fallocate(fd, 0, n * stride)
            except OSError:
                pass                                                 # a filesystem that cannot preallocate
            self.dfd[name] = None
            if self.direct:
                try:
                    self.dfd[name] = os.open(files[name], os.O_RDWR | os.O_DIRECT)
                except OSError:
                    pass                                             # no O_DIRECT here: buffered
            self.free[name] = collections.deque(range(n))
        return {name: n for name, (_, n, _) in self.geometry.items()}

    def rates(self) -> dict[str, object]:
        return RATES | self.overrides | {"lane": f"ssd:{os.stat(self.root).st_dev}"}

    # -- I/O ---------------------------------------------------------------------------------------------------
    def write(self, key: int, spec: str | None, src: Mapping[str, Sequence[Sequence[memoryview]]]) -> None:
        with self.cv:
            while any(len(self.free[name]) < len(units) for name, units in src.items()):
                self.cv.wait()                                       # a dropped copy's last read is still running
            slots = {name: [self.free[name].popleft() for _ in units] for name, units in src.items()}
        crcs = {}
        for name, units in src.items():
            crcs[name] = []
            for unit, slot in zip(units, slots[name], strict=True):
                crc = 0
                for mv in unit:
                    crc = zlib.crc32(mv, crc)
                crcs[name].append(crc)
                self._io(name, slot, unit, write=True)
        with self.cv:
            self.held[key, spec], self.crcs[key, spec] = slots, crcs

    def read(self, key: int, spec: str | None, dst: Mapping[str, Sequence[Sequence[memoryview]]]) -> None:
        with self.cv:
            slots = self.held[key, spec]
        for name, units in dst.items():
            for unit, slot in zip(units, slots[name], strict=True):
                self._io(name, slot, unit, write=False)

    def drop(self, key: int, spec: str | None) -> None:
        with self.cv:
            for name, slots in self.held.pop((key, spec), {}).items():
                self.free[name].extend(slots)
            self.crcs.pop((key, spec), None)
            self.cv.notify_all()

    def _io(self, name: str, slot: int, unit: Sequence[memoryview], write: bool) -> None:
        offset, total = slot * self.geometry[name][2], sum(len(mv) for mv in unit)
        direct = self.dfd[name] is not None and _aligned(unit, offset)
        fd = self.dfd[name] if direct else self.fd[name]
        n = os.pwritev(fd, unit, offset) if write else os.preadv(fd, unit, offset)
        if n != total:
            raise OSError(f"state_tier.ssd: {'wrote' if write else 'read'} {n} of {total} bytes at slot {slot} of "
                          f"{name}")
        if not direct:
            os.posix_fadvise(fd, offset, total, os.POSIX_FADV_DONTNEED)

    # -- persistence ---------------------------------------------------------------------------------------------
    def restore(self) -> list[BlockRecord]:
        m, self.manifest = self.manifest, None
        if m is None or self.persist != "close" or m.get("identity") != self.identity or self.identity is None:
            return []
        geometry, claimed = m.get("geometry", {}), set()
        good = []
        for r in m["records"]:
            slots = [(name, s, c) for copy in r["copies"].values() for name, pairs in copy.items() for s, c in pairs]
            ok = all(name in self.geometry and geometry.get(name, [None])[0] == self.geometry[name][0]
                     and s < self.geometry[name][1] and (name, s) not in claimed for name, s, _ in slots)
            if ok:
                claimed |= {(name, s) for name, s, _ in slots}
                good.append((r, slots))
        with concurrent.futures.ThreadPoolExecutor(self.threads) as pool:
            checks = list(pool.map(lambda rs: all(self._verify(name, s, c) for name, s, c in rs[1]), good))
        kept, out = {}, []
        for (r, _), ok in zip(good, checks, strict=True):
            if ok:
                kept[r["key"]] = r
        done: set[int] = set()
        while True:                                                  # parents first; orphans never come
            batch = [r for k, r in kept.items() if k not in done and (r["parent"] is None or r["parent"] in done)]
            if not batch:
                break
            for r in batch:
                done.add(r["key"])
                out.append(r)
        records = []
        for r in out:
            for spec, copy in r["copies"].items():
                key = (r["key"], None if spec == "" else spec)
                self.held[key] = {name: [s for s, _ in pairs] for name, pairs in copy.items()}
                self.crcs[key] = {name: [c for _, c in pairs] for name, pairs in copy.items()}
                for name, pairs in copy.items():
                    for s, _ in pairs:
                        self.free[name].remove(s)
            records.append(BlockRecord(r["key"], r["parent"], tuple(r["tokens"]), {s: () for s in r["pages"]}, {
                spec: Payload((0,) * n, shape) for spec, (shape, n) in r["payloads"].items()}))
        return records

    def _verify(self, name: str, slot: int, crc: int) -> bool:
        """Whether the slot holds the bytes whose crc32 the manifest recorded."""
        unit_bytes, _, stride = self.geometry[name]
        buf = bytearray(unit_bytes)
        return os.preadv(self.fd[name], [buf], slot * stride) == unit_bytes and zlib.crc32(buf) == crc

    def close(self, records: Sequence[BlockRecord]) -> None:
        try:
            if self.persist == "close":
                self._snapshot(records)
        finally:
            for fd in [*self.fd.values(), *(d for d in self.dfd.values() if d is not None)]:
                os.close(fd)
            self.lock.close()

    def _snapshot(self, records: Sequence[BlockRecord]) -> None:
        for fd in self.fd.values():
            os.fdatasync(fd)
        out = []
        with self.cv:
            for r in records:
                copies = {"": (r.key, None)} if r.units else {}
                copies |= {spec: (r.key, spec) for spec in r.payloads}
                if any(c not in self.held for c in copies.values()):
                    continue                                         # a copy it does not hold: the block is not kept
                out.append({"key": r.key, "parent": r.parent, "tokens": list(r.tokens), "pages": sorted(r.units),
                            "payloads": {spec: [p.pool, len(p.units)] for spec, p in r.payloads.items()},
                            "copies": {name: {shape: [[s, c] for s, c in zip(self.held[c][shape], self.crcs[c][shape],
                                                                            strict=True)]
                                              for shape in self.held[c]} for name, c in copies.items()}})
        tmp = self.root / (MANIFEST + ".tmp")
        with open(tmp, "w") as f:
            json.dump({"identity": self.identity, "geometry": {k: list(v) for k, v in self.geometry.items()},
                       "records": out}, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.root / MANIFEST)
        _fsync_dir(self.root)
