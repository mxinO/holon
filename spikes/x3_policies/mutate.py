"""How sharp the L1 policy-parity check is on a recorded trace (`run.py`'s `<name>_routes.npz`): placement.lfu's
replay oracle reproduces the device's residency at every step; each mutation of the oracle below (a wrong rule the
device might have had) is run on the same trace, reporting the first step it differs at (None: this trace cannot tell
it from the right rule).

    python spikes/x3_policies/mutate.py <routes.npz>
"""
import sys

import numpy as np
from holon_plugins.placement.lfu import reference as lfu

MUTATIONS = {"decay": "x0.9 instead of x0.915", "order": "increments before the decay",
             "bulk": "one f32 add of the step's count", "f64": "the decay in f64", "needs": "victims excluding only "
             "decode-routed experts", "tie": "candidate ties by highest id", "vtie": "victim ties by highest id",
             "strict": "admit on >= instead of >", "nostop": "no stop at the first failing pair"}


def replay(seed, counts, needed, mut):
    res, u, out = seed.astype(bool).copy(), np.zeros(seed.shape, dtype=np.float32), []
    for s, c in enumerate(counts):
        if c.any():
            u = (u.astype(np.float64) * 0.915).astype(np.float32) if mut == "f64" else \
                u * (np.float32(0.9) if mut == "decay" else lfu.DECAY)
            if mut == "order":
                u = (u / lfu.DECAY + c.astype(np.float32)) * lfu.DECAY
            elif mut == "bulk":
                u = u + c.astype(np.float32)
            else:
                for n in range(int(c.max())):
                    u = np.where(c > n, u + np.float32(1.0), u)
            need = (c > 0) if mut == "needs" else needed[s]
            cand = np.where((c > 0) & ~res, -u, np.inf)
            corder = np.argsort(cand, axis=1, kind="stable")
            if mut == "tie":
                corder = cand.shape[1] - 1 - np.argsort(cand[:, ::-1], axis=1, kind="stable")
            cvals = -np.take_along_axis(cand, corder, 1)
            vict = np.where(res & ~need, u, np.inf)
            vorder = np.argsort(vict, axis=1, kind="stable")
            if mut == "vtie":
                vorder = vict.shape[1] - 1 - np.argsort(vict[:, ::-1], axis=1, kind="stable")
            vvals = np.take_along_axis(vict, vorder, 1)
            ok = ((cvals >= vvals) if mut == "strict" else (cvals > vvals)) & np.isfinite(cvals) & np.isfinite(vvals)
            ok = ok if mut == "nostop" else np.cumprod(ok, axis=1).astype(bool)
            rows = np.nonzero(ok)[0]
            res[rows, vorder[ok]], res[rows, corder[ok]] = False, True
        out.append(res.copy())
    return out


def main() -> None:
    z = np.load(sys.argv[1])
    seed, counts, needed, resident = z["seed"], list(z["counts"]), list(z["needed"]), z["resident"]
    first = lambda got: next((i for i, (a, b) in enumerate(zip(got, resident, strict=True))  # noqa: E731
                              if not (a == b).all()), None)
    steps = [{"counts": c, "needed": n} for c, n in zip(counts, needed, strict=True)]
    print("oracle:", first([s["resident"] for s in lfu.replay(seed, steps)]))
    for mut, text in MUTATIONS.items():
        print(f"{mut} ({text}):", first(replay(seed, counts, needed, mut)))


if __name__ == "__main__":
    main()
