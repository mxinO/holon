"""Replay placement policies on the decode routing Holon recorded on the RTX 5080 (`ctx.py --trace`), with M6's
simulator (`spikes/x0_offload/m6/replay.py`): each policy starts from the residency Holon's static seed actually had
(same per-layer slot counts), so `static` must reproduce the measured misses, and `lru` / `lfu` / `belady` say what the
same cache would have fetched under a dynamic policy. The prompt's own routing is not in the trace (prefill rows are
never counted), so a dynamic policy here starts cold for the prompt's experts; FreeToken's cache has seen the prompt.

    python spikes/x4_gap/replay.py <trace.npz> [<trace.npz> ...]     # prints JSON
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "x0_offload" / "m6"))
import replay as m6  # noqa: E402

POLICIES = ("static", "lru", "lfu", "belady")


def decode(keys: np.ndarray, ptr: np.ndarray, L: int, E: int) -> m6.Decode:
    st = np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))
    by_key = np.lexsort((st, keys))
    nxt = np.full(len(keys), m6.BIG)
    same = keys[by_key[1:]] == keys[by_key[:-1]]
    nxt[by_key[:-1][same]] = st[by_key[1:][same]]
    return m6.Decode("holon", L, E, ptr, keys.astype(np.int64), np.ones(len(keys)), nxt,
                     np.ones(len(ptr) - 1, np.int64), np.bincount(keys, minlength=L * E).astype(float))


def main() -> None:
    out = {}
    for path in sys.argv[1:]:
        z = np.load(path)
        resident, item_bytes = z["resident"], z["item_bytes"]
        L, E = resident.shape
        ebytes = np.repeat(item_bytes, E).astype(float)
        seed = resident.reshape(-1)
        quota = resident.sum(1)
        rank = np.zeros(L * E, np.int64)
        for name in z.files:
            if not name.startswith("keys_"):
                continue
            ctx = name[5:]
            tr = decode(z[name], z[f"ptr_{ctx}"], L, E)
            row = {}
            for p in POLICIES:
                r = m6.simulate(tr, p, quota, seed, rank, ebytes, 0.0)
                row[p] = {"hit_rate": round(r["hit_rate"], 4),
                          "fetch_mb_per_token": round(r["bytes_per_token"]["fetch"] / 1e6, 2)}
            out[f"{pathlib.Path(path).stem}@{ctx}"] = {"cached": int(seed.sum()), "experts": L * E, "policies": row}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
