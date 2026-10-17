"""std.tiered.static@1 (F1): a frozen residency map — the hottest items of a routing profile, else every group's
items in id order taken round-robin across groups, as far as the cache slots go; never an edit after it."""
from __future__ import annotations

import numpy as np


class Static:
    resolve_variant = None

    def __init__(self, ctx, **site) -> None:
        pass

    def device_state(self) -> dict:
        return {}

    def seed(self, profile: np.ndarray, budget) -> list[tuple[int, int]]:
        return [(g, i) for g, i in ranking(profile) if budget.take(g, i)]

    def after_commit(self, counts: np.ndarray, view: np.ndarray) -> list:
        return []


def ranking(profile: np.ndarray) -> list[tuple[int, int]]:
    """(group, item) pairs by count descending, ties by item, then group: under a flat profile (no profile: zeros),
    item 0 of every group, then item 1, …, so every group gets the same share."""
    group, item = np.divmod(np.arange(profile.size), profile.shape[1])
    return [(int(group[x]), int(item[x])) for x in np.lexsort((group, item, -profile.ravel()))]
