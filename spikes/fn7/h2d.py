"""FN step 7 (the 5090's 1K prefill): host-to-device traffic in an nsys trace, either engine's. The trace's memcpys
(H2D, and every kernel) are merged into busy intervals; segments are runs of H2D copies with gaps under `--gap` ms.
Per segment: its wall time, H2D bytes and copy-busy time, kernel-busy time, and the time both are busy (the overlap).
Only segments of at least `--min-mb` MB are shown. Usage: h2d.py <rep.nsys-rep | .sqlite> [--gap 30] [--min-mb 500]"""
from __future__ import annotations

import argparse
import json
import pathlib
import sqlite3
import subprocess


def union(iv: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def clip(iv: list[tuple[int, int]], a: int, b: int) -> list[tuple[int, int]]:
    return [(max(x, a), min(y, b)) for x, y in iv if y > a and x < b]


def both(p: list[tuple[int, int]], q: list[tuple[int, int]]) -> int:
    i = j = t = 0
    while i < len(p) and j < len(q):
        t += max(0, min(p[i][1], q[j][1]) - max(p[i][0], q[j][0]))
        if p[i][1] < q[j][1]:
            i += 1
        else:
            j += 1
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--gap", type=float, default=30.0)
    ap.add_argument("--min-mb", type=float, default=500.0)
    a = ap.parse_args()
    db = pathlib.Path(a.trace)
    if db.suffix != ".sqlite":
        out = db.with_suffix(".sqlite")
        if not out.exists():
            subprocess.run(["nsys", "export", "--type", "sqlite", "-o", str(out), str(db)], check=True,
                           capture_output=True)
        db = out
    c = sqlite3.connect(db)
    copies = c.execute("SELECT start, end, bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE copyKind = 1 "
                       "ORDER BY start").fetchall()
    kern = union([(s, e) for s, e in c.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL")])
    segs, gap = [], a.gap * 1e6
    for s, e, n in copies:
        if segs and s - segs[-1][1] < gap:
            segs[-1][1] = max(segs[-1][1], e)
            segs[-1][2].append((s, e, n))
        else:
            segs.append([s, e, [(s, e, n)]])
    t0 = copies[0][0] if copies else 0
    rows = []
    for s, e, cs in segs:
        nbytes = sum(n for *_, n in cs)
        if nbytes < a.min_mb * 1e6:
            continue
        cp = union([(x, y) for x, y, _ in cs])
        kk = clip(kern, s, e)
        busy = sum(y - x for x, y in cp)
        rows.append({"at_s": round((s - t0) / 1e9, 3), "wall_ms": round((e - s) / 1e6, 1), "copies": len(cs),
                     "h2d_mb": round(nbytes / 1e6, 1), "copy_busy_ms": round(busy / 1e6, 1),
                     "gbs_busy": round(nbytes / busy, 1) if busy else None,
                     "kernel_busy_ms": round(sum(y - x for x, y in kk) / 1e6, 1),
                     "overlap_ms": round(both(cp, kk) / 1e6, 1),
                     "mean_copy_kb": round(nbytes / len(cs) / 1e3, 1)})
    print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
