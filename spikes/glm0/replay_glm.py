"""GLM step 0, N2 (docs/dev/v1-glm.md §8, D7, D8): X's exclusive-tier replay (`spikes/x0_offload/m6/exclusive.py`) on
GLM-5.3-Flash Maya-S traces (`maya_routes.py`), at the RTX 5090's and the RTX 5080's fits.

Only the model and the rates change; the policies, lags and the time model are exclusive.py's:
- experts: Maya-S, gate / up IQ2_XXS 2 x 2,162,688 B; down IQ2_S 2,686,976 B (layers 7-40) or IQ3_XXS 3,211,264 B
  (3-6, 41-44) — G-B §1; E = 288, the trace's 42 MoE layers;
- dense bytes a decode token reads: 7.1 GB (§7: non-expert tensors but the token table);
- the all-resident GPU time per batch-1 token `--t-b1` (default 7.0 ms est.: 7.1 GB + 336 hit experts at 1.55 TB/s,
  + 1 ms fixed; N4 replaces it);
- H2D / D2H 50 GB/s (FN §23.6, THP); SSD `--ssd` GB/s and its depth-1 read time `--ssd-lat-ms` per 7 MB expert (N3);
- fits: VRAM 7.3% / 27% (§7: the 5080 / the 5090); RAM holding all, or a 50 GiB tier (0.854 of the rest at 27%).
Held-out splits: each trace name is tested with every other trace as the profile's training set.

    python spikes/glm0/replay_glm.py --traces <dir of .npz> --out <json> [--ssd 3.0] [--ssd-lat-ms 2.5] [--jobs 16]
    python spikes/glm0/replay_glm.py --windows --traces <dir>      # D7: distinct experts per 1..4-token window
"""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import pathlib
import sys
import time

import numpy as np

M6 = pathlib.Path(__file__).resolve().parents[1] / "x0_offload" / "m6"
sys.path.insert(0, str(M6))
import exclusive as X  # noqa: E402
import replay as R  # noqa: E402

E = 288
LAYERS = list(range(3, 45))
EBYTES = np.array([2 * 2_162_688.0 + (3_211_264.0 if (layer < 7 or layer > 40) else 2_686_976.0) for layer in LAYERS])
DENSE = 7.1e9
FITS = {"rtx-5080": 0.073, "rtx-5090": 0.27}
RAM_OF_REST = {"all": 1.0, "50GiB": 0.854}


def configure(t_b1: float, ssd: float, lat_ms: float) -> None:
    """exclusive.py reads these module globals at call time."""
    X.EBYTES, X.DENSE, X.T_B1, X.PCIE, X.SSD = EBYTES, DENSE, t_b1, 50e9, ssd
    X.GPU_EFF = (DENSE + 8 * EBYTES.sum()) / t_b1
    X.LAT = lambda b: lat_ms * 1e-3 * b / EBYTES.mean()                    # noqa: E731  depth-1, scaled by size
    X.EXTENT_WAIT = X.LAT(EBYTES.mean() / 3) / 3


def run(traces: str, jobs: int, variants: tuple[str, ...]) -> dict:
    for p in sorted(pathlib.Path(traces).glob("*.npz")):
        X._TR[p.stem] = R.decode_steps(R.load(p), E)
    names = sorted(X._TR)
    todo = []
    for test in names:
        train = tuple(n for n in names if n != test) or (test,)
        for (card, vf), (rk, rf) in itertools.product(FITS.items(), RAM_OF_REST.items()):
            todo += [(test, train, vf, rf, v) for v in (("lfu_allram",) if rk == "all" else variants)]
    with mp.get_context("fork").Pool(jobs) as pool:
        done = pool.map(X._job, todo, chunksize=1)
    cells: dict = {}
    vcard = {vf: card for card, vf in FITS.items()}
    rname = {rf: rk for rk, rf in RAM_OF_REST.items()}
    for (test, train, vf, rf, v), r in done:
        cells.setdefault(test, {}).setdefault(vcard[vf], {}).setdefault(rname[rf], {})[v] = r
    return {"cells": cells}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--out")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--t-b1", type=float, default=7.0e-3)
    ap.add_argument("--ssd", type=float, default=3.0, help="GB/s")
    ap.add_argument("--ssd-lat-ms", type=float, default=2.5, help="depth-1 read of one 7 MB expert")
    ap.add_argument("--variants", nargs="+", default=["f11_deferred", "f11_deferred_keep", "f10_spares",
                                                       "f10_spares_keep", "f11_ssd"])
    ap.add_argument("--windows", action="store_true")
    a = ap.parse_args()
    configure(a.t_b1, a.ssd * 1e9, a.ssd_lat_ms)
    if a.windows:
        for p in sorted(pathlib.Path(a.traces).glob("*.npz")):
            raw = R.load(p)
            print(p.stem, json.dumps(R.windows(raw, raw["ids"].shape[1], E)))
        return
    t0 = time.time()
    rec = run(a.traces, a.jobs, tuple(a.variants))
    rec |= {"replay_s": round(time.time() - t0, 1), "model": "GLM-5.3-Flash Maya-S", "fits": FITS,
            "ram_of_rest": RAM_OF_REST, "rates": {"t_b1_ms": a.t_b1 * 1e3, "h2d_gbps": 50, "ssd_gbps": a.ssd,
                                                   "ssd_depth1_ms": a.ssd_lat_ms}, "dense_bytes": DENSE}
    text = json.dumps(rec, indent=1)
    if a.out:
        pathlib.Path(a.out).write_text(text)
    for test, by in rec["cells"].items():
        for card, by_r in by.items():
            for rk, by_v in by_r.items():
                print(test, card, rk, {v: (round(float(r["hit_rate"]), 3), round(float(r["ms_per_token"]), 2)) for v, r in by_v.items()})


if __name__ == "__main__":
    main()
