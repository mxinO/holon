"""FN step 0, N3's routing input: Strata's `--dump-routing` trace (St generate.cpp:1007-1046: records of int32 layer,
int32 k, k x int32 expert ids, k x float32 weights; one record per layer per decode position, a verify window's
positions one record each) read into decode tokens [n, 48, 10], and the decode hit rate a per-layer cache of S slots
gets on it under Holon's `placement.lfu@1` (plugins/holon_plugins/placement/lfu/reference.py: per decode step every
score decays x0.915 and each routed expert adds 1; a missing routed expert takes an empty slot, else replaces the
coldest resident not routed this step while its score is strictly higher; ties by expert id). The cache starts as a
profile seeds it (Strata's `expert-profile.bin`: every layer's first S experts in the profile's ranking), as Holon's
bundle profile does (D12f).

    python spikes/fn0/routing.py <routing.bin> --slots 97 155 [--profile <expert-profile.bin>]
"""
from __future__ import annotations

import argparse
import json
import struct

import numpy as np

LAYERS, EXPERTS, DECAY = 48, 512, np.float32(0.915)


def read(path: str) -> np.ndarray:
    """The trace's decode tokens [n, LAYERS, k] (int32 expert ids)."""
    raw = np.fromfile(path, dtype=np.int32)
    layers, ids, i = [], [], 0
    while i < len(raw):
        layer, k = int(raw[i]), int(raw[i + 1])
        layers.append(layer)
        ids.append(raw[i + 2:i + 2 + k])
        i += 2 + 2 * k
    layers = np.array(layers)
    starts = np.flatnonzero(layers == 0)
    assert len(layers) == len(starts) * LAYERS and all((layers[s:s + LAYERS] == np.arange(LAYERS)).all()
                                                       for s in starts), "records not in whole tokens"
    return np.stack(ids).reshape(len(starts), LAYERS, -1)


def profile(path: str) -> list[list[int]]:
    """Each layer's experts in a Strata profile's ranking (St tools/make_profile.py: magic, 5 u32, (u16, u16) pairs)."""
    blob = open(path, "rb").read()
    n = struct.unpack_from("<5I", blob, 4)[4]
    order: list[list[int]] = [[] for _ in range(LAYERS)]
    for i in range(n):
        layer, e = struct.unpack_from("<HH", blob, 24 + 4 * i)
        order[layer].append(e)
    return order


def lfu_hits(tokens: np.ndarray, slots: int, seed: list[list[int]] | None = None) -> float:
    """Decode hit rate of `placement.lfu@1` with `slots` per layer, empty at the start or seeded with each layer's first
    `slots` experts of `seed` (scores zero)."""
    hits = total = 0
    for layer in range(tokens.shape[1]):
        score = np.zeros(EXPERTS, np.float32)
        resident = np.zeros(EXPERTS, bool)
        if seed is not None:
            resident[seed[layer][:slots]] = True
        free = slots - int(resident.sum())
        for routed in tokens[:, layer]:
            hit = resident[routed]
            hits += int(hit.sum())
            total += len(routed)
            score *= DECAY
            np.add.at(score, routed, np.float32(1.0))
            cands = sorted((int(e) for e in routed[~hit]), key=lambda e: (-score[e], e))
            take = min(free, len(cands))
            resident[cands[:take]] = True
            free -= take
            if take == len(cands):
                continue
            victims = [int(e) for e in np.flatnonzero(resident) if e not in set(routed.tolist())]
            victims.sort(key=lambda e: (score[e], e))
            for a, b in zip(cands[take:], victims, strict=False):
                if not score[a] > score[b]:
                    break
                resident[a], resident[b] = True, False
    return hits / total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace")
    ap.add_argument("--slots", type=int, nargs="+", required=True, help="cache slots per layer")
    ap.add_argument("--profile", help="a Strata expert profile seeding the cache")
    a = ap.parse_args()
    tokens, seed = read(a.trace), profile(a.profile) if a.profile else None
    print(json.dumps({"tokens": len(tokens), "hit_rate": {s: round(lfu_hits(tokens, s, seed), 4) for s in a.slots}}))


if __name__ == "__main__":
    main()
