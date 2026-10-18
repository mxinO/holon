"""One evidence record from a run directory of `exact.py`, `bench/holon_tg.py` and `gather_alloc.py` outputs
(`gather_alloc*.json`: one process each, as host placement moves rates by a few percent between processes):

    python spikes/x2_store/record.py <runs dir> <evidence json> <commit measured> <commands…>
"""
from __future__ import annotations

import json
import pathlib
import statistics
import sys

from holon_judge.snapshot import Snapshot, diff


def main() -> None:
    runs, out, commit, commands = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3], sys.argv[4:]
    resident = Snapshot.load(runs / "resident_0.json")
    exactness = []
    for path in sorted(runs.glob("offload_*.json")):
        snap = Snapshot.load(path)
        meta = {k: snap.meta[k] for k in ("fit", "run", "pools", "peak_misses")}
        exactness.append({**meta, "name": path.stem, "vs_resident": diff(resident, snap) or "identical"})
    tg = {arm: [json.loads(p.read_text())["results"]["tg128"] for p in sorted(runs.glob(f"tg_{arm}_*.json"))]
          for arm in ("resident", "offload")}
    gathers = [json.loads(p.read_text()) for p in sorted(runs.glob("gather_alloc*.json"))]   # one process each
    ms = {arm: statistics.median(1000 / r["mean"] for r in rounds) for arm, rounds in tg.items()}
    record = {
        "gpu": "rtx-a5000", "model": "unsloth/Qwen3-30B-A3B-GGUF Q4_K_M",
        "commit": commit, "commands": commands,
        "exactness": {"resident": {"steps": len(resident.steps), "pools": resident.meta["pools"]},
                      "offload": exactness},
        "overhead": {"tg128_batch1": tg, "ms_per_token_median": {k: round(v, 4) for k, v in ms.items()},
                     "offload_minus_resident_ms": round(ms["offload"] - ms["resident"], 4)},
        "gather_alloc": {"runs": gathers, "vs_cudaHostAlloc": {blob: {
            alloc: {"median": round(statistics.median(ratios), 4), "range": [min(ratios), max(ratios)]}
            for alloc, ratios in ((a, [g["results"][blob][a]["vs_cudaHostAlloc"] for g in gathers])
                                  for a in gathers[0]["results"][blob])} for blob in gathers[0]["results"]}},
    }
    out.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({"exactness": [(e["name"], e["vs_resident"]) for e in exactness],
                      **record["overhead"]["ms_per_token_median"]}))


if __name__ == "__main__":
    main()
