"""The experts a deployment caches on the device: per class its cache slots and pieces, and the cached share of the
experts' bytes (C-M1's target: >= 23% with the MTP pack resident). Usage: cached.py <model> '<engine flags>'"""
from __future__ import annotations

import json
import sys

import torch
from _engine import STORE, engine


def main() -> None:
    model, flags = sys.argv[1:3]
    e = engine(model, flags, max_seqs=1, max_model_len=33792)
    s = e.ctx.shared_objects[STORE]
    rows = {n: {"cache": c.size - c.null - c.staging, "pieces": len(c.pieces), "unit_bytes": c.unit_bytes}
            for n, c in s.classes.items()}
    cached = sum(r["cache"] * r["unit_bytes"] for r in rows.values())
    total = sum(r["pieces"] * r["unit_bytes"] for r in rows.values())
    print(json.dumps({"flags": flags, "lending": s.lending, "classes": rows, "cached_bytes": cached,
                      "cached_fraction": round(cached / total, 4), "vram_reserved": torch.cuda.memory_reserved(),
                      "vram_total": torch.cuda.get_device_properties(0).total_memory}))


if __name__ == "__main__":
    main()
