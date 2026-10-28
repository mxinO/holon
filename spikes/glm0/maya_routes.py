"""GLM step 0, N2 (docs/dev/v1-glm.md §8): Project Maya's decode route log as an M6 trace.

Maya writes one line per decode route with `STRATA_GLM_ROUTE_LOG=<prefix>` (My src/core/glm_fast_path.cu:4148-4161):
`<layer> e0 .. e7 <fetch> <miss> <cpu> | n0 .. n15` — the route's experts in selection order, then bit masks over
e0..e7 (fetched over PCIe, read from disk, computed by the CPU lane; none set: a VRAM hit), then the near misses.
A token's routes come in ascending layer order (one conversation, `--parallel 1`), so a token ends where the layer
does not increase. Prompt routes are not logged (the prompt path does not go through the service thread).

Writes `spikes/x0_offload/m6/replay.py`'s trace form (`load` / `decode_steps`): `ids` [R, L, 8] uint16 (R decode
tokens, L = the MoE layers present, as indices from 0), `step` [R] (one token per step), `seq` [R] (0: one conversation), `decode` [R] (all true),
`tier` [R, L, 3] uint8 (fetch, miss, cpu masks: the live tier Maya saw, for validating a replay of its own policy),
`info` (JSON: the layers, Maya's live hit rate).

    python spikes/glm0/maya_routes.py <route log> <out.npz> [--skip-tokens N]
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def parse(path: str) -> tuple[list[int], np.ndarray, np.ndarray]:
    rows, tiers, layers = [], [], []          # per token: {layer: ids}, {layer: masks}
    prev = None
    with open(path) as f:
        for line in f:
            head = line.split("|", 1)[0].split()
            if len(head) != 12:
                continue                      # a torn last line
            il, ids, masks = int(head[0]), [int(x) for x in head[1:9]], [int(x) for x in head[9:12]]
            if prev is None or il <= prev:
                rows.append({})
                tiers.append({})
            rows[-1][il], tiers[-1][il] = ids, masks
            prev = il
            if il not in layers:
                layers.append(il)
    layers.sort()
    full = [i for i, r in enumerate(rows) if len(r) == len(layers)]     # a token cut off at the log's end is dropped
    ids = np.array([[rows[i][l] for l in layers] for i in full], dtype=np.uint16)
    tier = np.array([[tiers[i][l] for l in layers] for i in full], dtype=np.uint8)
    return layers, ids, tier


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("out")
    ap.add_argument("--skip-tokens", type=int, default=0, help="leading tokens dropped (e.g. a warm-up request's)")
    a = ap.parse_args()
    layers, ids, tier = parse(a.log)
    ids, tier = ids[a.skip_tokens:], tier[a.skip_tokens:]
    R = len(ids)
    hits = int(sum(bin(~(f | m | c) & 0xFF).count("1") for f, m, c in tier.reshape(-1, 3).tolist()))
    info = {"source": "Project Maya v1.1.27 STRATA_GLM_ROUTE_LOG", "layers": layers, "tokens": R,
            "live_hit_rate": hits / max(ids.size, 1)}
    np.savez_compressed(a.out, ids=ids, step=np.arange(R), seq=np.zeros(R, np.int32), decode=np.ones(R, bool),
                        tier=tier, info=json.dumps(info))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
