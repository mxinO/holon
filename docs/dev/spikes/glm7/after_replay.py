"""GLM step 7, mode (ii) (a): `after_record.py`'s recording replayed piece by piece (a role's tier: VRAM, RAM, SSD
only). Per prompt size (means over its requests):
- the prompt: its routed pieces that were SSD-only before it (the least it must read) against the bytes it read, and
  the excess; the pieces it displaced (VRAM or RAM before, SSD only after), from VRAM and from RAM;
- the decode: the SSD bytes a token the map predicts (routed pieces SSD-only at the step's start) against the bytes
  read (the check of the replay), over tokens 1-32 and 33-256, and the share of them re-reading pieces the prompt
  displaced.
    python docs/dev/spikes/glm7/after_replay.py <after.npz>"""
import collections
import sys

import numpy as np

z = np.load(sys.argv[1])
mib = z["piece_mib"]                                                    # [G, R]
keys = sorted({k.rsplit(".", 1)[0] for k in z.files if "." in k}, key=lambda k: tuple(map(int, k.split("_"))))
agg = collections.defaultdict(lambda: collections.defaultdict(list))
for k in keys:
    before, maps, read = z[f"{k}.before"], z[f"{k}.maps"], z[f"{k}.bytes"] / 2**20
    n_prompt, parked, n_steps, _, experts = z[f"{k}.meta"]
    n_prompt, n_steps = int(n_prompt), int(n_steps)
    routed = np.unpackbits(z[f"{k}.routed"], axis=-1)[..., :int(experts)].astype(bool)     # [S, G, E]
    size = mib[:, None, :]                                              # [G, 1, R]
    a = agg[int(k.split("_")[0])]
    prompt = routed[:n_prompt].any(0)[..., None]
    least = ((prompt & (before == 2)) * size).sum() / 1024
    a["prompt GiB read"].append(read[n_prompt - 1] / 1024)
    a["prompt GiB least (routed SSD-only before)"].append(least)
    a["prompt GiB excess"].append(read[n_prompt - 1] / 1024 - least)
    after = maps[n_prompt - 1]
    displaced = (before < 2) & (after == 2)
    a["displaced pieces"].append(displaced.sum())
    a["  from VRAM"].append((displaced & (before == 0)).sum())
    a["  from RAM"].append((displaced & (before == 1)).sum())
    a["displaced GiB"].append((displaced * size).sum() / 1024)
    a["parked experts"].append(int(parked))
    pred, mine = [], []
    for t in range(n_prompt, n_steps):                                  # decode step t reads its SSD-only routes
        miss = routed[t][..., None] & (maps[t - 1] == 2)
        pred.append((miss * size).sum())
        mine.append(((miss & displaced) * size).sum())
    pred, mine = np.array(pred), np.array(mine)
    got = np.diff(read[n_prompt - 1:])
    for name, sl in (("1-32", slice(0, 32)), ("33-256", slice(32, None))):
        a[f"decode MB/token read, {name}"].append(got[sl].mean())
        a[f"decode MB/token predicted, {name}"].append(pred[sl].mean())
        a[f"decode share re-reading displaced, {name}"].append(mine[sl].sum() / max(pred[sl].sum(), 1e-9))
for size, a in sorted(agg.items()):
    print(f"== {size}-token prompts ({len(a['prompt GiB read'])})")
    for m, v in a.items():
        print(f"  {m:46s} {np.mean(v):9.3f}  ({np.min(v):.3f} – {np.max(v):.3f})")
