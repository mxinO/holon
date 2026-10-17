"""A routing profile for the static policy (`moe.placement.profile`): decode routing counts per (layer, expert) from
M6's captured traces (`spikes/x0_offload/m6/capture.py`; the agent and mixed traces, which share no prompt with the
judge-corpus spans x4_gap decodes), as `{"layers": {"<layer>": [count per expert]}}`.

    python spikes/x4_gap/routing_profile.py <out.json> <trace.npz> [<trace.npz> ...]
"""
from __future__ import annotations

import json
import sys

import numpy as np


def main() -> None:
    out, paths = sys.argv[1], sys.argv[2:]
    counts = None
    for p in paths:
        with np.load(p) as z:
            ids = z["ids"][z["decode"]].astype(np.int64)                 # [rows, layers, k]
        _, layers, _ = ids.shape
        c = np.stack([np.bincount(ids[:, layer].ravel(), minlength=128) for layer in range(layers)])
        counts = c if counts is None else counts + c
    with open(out, "w") as f:
        json.dump({"layers": {str(layer): row.tolist() for layer, row in enumerate(counts)}}, f)
    top = np.sort(counts, axis=1)[:, ::-1]
    print(f"{int(counts.sum())} routes; the top 47% of each layer's experts take "
          f"{top[:, :60].sum() / counts.sum():.1%}")


if __name__ == "__main__":
    main()
