"""std.tiered.static@1 (F1): a frozen residency map — F2's warm start (a routing profile's hottest items, 4 per group
first; without one every group's items in id order, taken round-robin across groups), as far as the cache slots go;
never an edit after it."""
from __future__ import annotations

import numpy as np


class Static:
    resolve_variant = None
    device_owned = False
    heat = None                   # under exclusive RAM: no displacement (keeps take free units only)

    def __init__(self, ctx, **site) -> None:
        pass

    def device_state(self) -> dict:
        return {}

    def seed(self, profile: np.ndarray, budget) -> list[tuple[int, int]]:
        return [(g, i) for g, i in budget.warm_start(profile) if budget.take(g, i)]

    def after_commit(self, counts: np.ndarray, view: np.ndarray) -> list:
        return []
