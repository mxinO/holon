"""GLM step 7, mode (ii) (a): after_record.py's recording of the bench's workload, replayed for the after-prompt levers.
Per prompt size: the experts the prompt displaced (VRAM or RAM before it, SSD after), the decode's SSD reads a token
over tokens 1-32 and 33-256 and the share of them displaced ones
a background refill of the K hottest displaced
experts (RAM's heat at the prompt's end) before the decode: the reads it would save (an expert's first decode read
only) against the reads it costs (K), and where the saved ones fall (tokens 1-32 or later).
    python after_replay.py <after_bench.npz>"""
import collections
import sys

import numpy as np

z = np.load(sys.argv[1])
keys = sorted({k.rsplit(".", 1)[0] for k in z.files}, key=lambda k: (int(k.split("_")[0]), int(k.split("_")[1])))
agg = collections.defaultdict(lambda: collections.defaultdict(list))
for k in keys:
    size = int(k.split("_")[0])
    before, maps, bts = z[f"{k}.before"], z[f"{k}.maps"], z[f"{k}.bytes"]
    meta = z[f"{k}.meta"]
    n_prompt, parked, n_steps, wall, E = int(meta[0]), int(meta[1]), int(meta[2]), float(meta[3]), int(meta[4])
    routed = np.unpackbits(z[f"{k}.routed"], axis=-1)[..., :E].astype(bool)
    after = maps[n_prompt - 1]
    disp = (before < 2) & (after == 2)
    reads = [(routed[t] & (maps[t - 1] == 2)) for t in range(n_prompt, n_steps)]      # decode step t's SSD reads
    early, late = reads[:32], reads[32:]
    a = agg[size]
    a["displaced"].append(disp.sum())
    a["displaced_vram"].append((disp & (before == 0)).sum())
    a["parked"].append(parked)
    a["prompt_routed_ssd_before"].append((routed[:n_prompt].any(0) & (before == 2)).sum())
    a["prompt_gib"].append(bts[n_prompt - 1] / 2**30)
    a["reads_1_32"].append(sum(r.sum() for r in early) / 32)
    a["reads_33_256"].append(sum(r.sum() for r in late) / max(len(late), 1))
    a["disp_share_1_32"].append(sum((r & disp).sum() for r in early) / max(sum(r.sum() for r in early), 1))
    a["disp_share_33_256"].append(sum((r & disp).sum() for r in late) / max(sum(r.sum() for r in late), 1))
    first = np.full(disp.shape, -1)                         # each displaced expert's first decode read (token)
    for t, r in enumerate(reads):
        m = r & disp & (first < 0)
        first[m] = t
    heat = z[f"{k}.heat_end"] if f"{k}.heat_end" in z.files else np.zeros(disp.shape)
    order = np.argsort(-np.where(disp, heat, -np.inf), axis=None, kind="stable")
    for K in (250, 500, 1000, 2000):
        take = np.zeros(disp.size, bool)
        take[order[:min(K, int(disp.sum()))]] = True
        f = first.ravel()[take]
        a[f"refill{K}_saved"].append((f >= 0).sum())
        a[f"refill{K}_saved_1_32"].append(((f >= 0) & (f < 32)).sum())
for size, a in sorted(agg.items()):
    print(f"== {size}-token prompts ({len(a['displaced'])})")
    for m, v in a.items():
        print(f"  {m:28s} {np.mean(v):9.2f}  (min {np.min(v):.2f}, max {np.max(v):.2f})")
