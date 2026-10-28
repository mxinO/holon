"""placement.profile_decay@1 (F2–F3, Strata X2; docs/dev/v1-offload.md D1, D4): a profile-seeded cache, then adaptive
swaps
by decayed routing counts on a fixed cadence — the Strata-parity arm, as the M6 replay's `profile_decay`
(bench/placement_replay.py; Strata generate.cpp:8829-8987). A host policy: the cache is the host's, edited
between steps, and the store applies each swap in two phases (the victim unmapped at launch c + 2, the incoming
expert copied into its slots after that step, mapped at c + 2 + d).

Counts are the decode and verify rows' routes (+1 per entry). Every `EVERY` steps with decode rows a round: per
layer, non-resident experts counted at least `U_MIN` (by count, descending) pair with residents (ascending) while the
candidate's count is at least its victim's + `GAIN`; of all pairs the `SWAPS` with the largest gain swap (ties in
layer order); the counts then decay × `DECAY`. A round is skipped while a swap of the last one is not yet mapped."""
from __future__ import annotations

import numpy as np

DECAY, EVERY, SWAPS, GAIN, U_MIN = 0.7, 4, 96, 1.5, 2.0       # Strata's (the replay's defaults)


class ProfileDecay:
    resolve_variant = None
    device_owned = False

    def __init__(self, ctx, **site) -> None:
        self.rounds = 0                       # rounds that swapped (since the last seed)

    def device_state(self) -> dict:
        return {}

    def seed(self, profile: np.ndarray, budget) -> list[tuple[int, int]]:
        seeded = [(g, i) for g, i in budget.warm_start(profile) if budget.take(g, i)]
        self.u, self.steps, self.rounds = np.zeros(profile.shape), 0, 0
        self.want = np.zeros(profile.shape, dtype=bool)          # the residency its edits make
        for g, i in seeded:
            self.want[g, i] = True
        return seeded

    def after_commit(self, counts: np.ndarray, view: np.ndarray) -> list[tuple[int, int, bool]]:
        if not counts.any():                                      # a step without decode rows
            return []
        self.u += counts
        self.steps += 1
        edits = []
        if self.steps % EVERY == 0:
            if ((view >= 0).all(-1) == self.want).all():          # every swap of the last round mapped
                for g, a, b in _pairs(self.want, self.u):
                    self.want[g, a], self.want[g, b] = True, False
                    edits += [(g, a, True), (g, b, False)]
                self.rounds += bool(edits)
            self.u *= DECAY
        return edits


def _pairs(resident: np.ndarray, u: np.ndarray) -> list[tuple[int, int, int]]:
    """Strata's pairing (the replay's `_pair_swap`): (group, incoming, outgoing) by gain, the largest `SWAPS`."""
    cand = np.where(~resident & (u >= U_MIN), -u, np.inf)
    corder = np.argsort(cand, axis=1, kind="stable")
    cvals = -np.take_along_axis(cand, corder, 1)
    vict = np.where(resident, u, np.inf)
    vorder = np.argsort(vict, axis=1, kind="stable")
    vvals = np.take_along_axis(vict, vorder, 1)
    ok = np.cumprod(np.isfinite(cvals) & np.isfinite(vvals) & (cvals >= vvals + GAIN), axis=1).astype(bool)
    groups = np.nonzero(ok)[0]
    a, b, gain = corder[ok], vorder[ok], (cvals - vvals)[ok]
    top = np.argsort(-gain, kind="stable")[:SWAPS]
    return [(int(groups[x]), int(a[x]), int(b[x])) for x in top]


class Plugin:
    def register(self, regs) -> None:
        regs.units.add("placement.profile_decay", ProfileDecay)
