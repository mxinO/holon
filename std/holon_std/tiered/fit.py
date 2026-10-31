"""`moe.ram=fit` (docs/dev/v1-glm-ram.md D1, D2, D4): the RAM tier sized from the host budget, once, at the store's
first `before_materialize` — one share f of every class's pieces (an expert serves from RAM only with all its pieces
there), against `BuildCtx.host_left()` after every fixed host pool, the store's own included, less a reserve. f ≥ 1:
every piece fits, and the store is the lock's (inclusive); below, exclusive tiers hold `held` units a class.

The store owns the host remainder alone: `fit` refuses beside a state tier (its host level would race the RAM tier),
and without a host budget."""
from __future__ import annotations

import math
from collections.abc import Mapping

from holon.api import OutOfUnits

from . import files

STEP = 256                  # f rounded down to 1/STEP: placement does not drift with a few MB of free RAM (A-RAM-m1)
BOUNCE = files.WORKERS * files.WINDOW     # the seed reads' bounce windows
PINNED = 64 << 20           # the pinned buffers outside the ledger (D8g's read-back lists, D8x's move table, the
#                             ring's copy list: KBs each)
GROWTH = 3 << 30            # R, the process's non-pinned growth to a 32K step's peak (D4; B m4: std's constant, the
#                             worst case measured: GLM on an RTX 5090, 2.52 GiB at the lock, 2.62 at mode (ii))
RESERVE = BOUNCE + PINNED + GROWTH
GiB = 1 << 30


def budget(ctx, name: str) -> int:
    """`ctx.host_left()`, the host memory left to pin; refused where `fit` cannot own it: beside a state tier (the
    engine's `tier` slot), or without a host budget."""
    if "tier" in ctx.options.get("slots", {}):
        raise ValueError(f"{name}: moe.ram=fit sizes the RAM tier to the host memory left, which the state tier "
                         f"({ctx.options['slots']['tier']}) would take too: give moe.ram a share, or drop the tier")
    if (left := ctx.host_left()) is None:
        raise ValueError(f"{name}: moe.ram=fit needs a host budget: `holon serve` derives one; elsewhere, "
                         "--host-memory")
    return left


def whole(left: int, total: int) -> bool:
    """Every piece (`total` bytes) fits in `left` beside the reserve: f ≥ 1, the lock's inclusive store (no I/O
    pools)."""
    return left - RESERVE >= total


def floor(classes: Mapping[str, tuple[int, int]], keeps: int) -> float:
    """The working floor's f: the least multiple of 1/STEP giving every class (name → (pieces, unit bytes)) a call's
    keeps — `keeps` units, the graphed steps' misses, at most its pieces — as `held`."""
    return math.ceil(max(min(n, keeps) / n for n, _ in classes.values()) * STEP) / STEP


def share(name: str, left: int, classes: Mapping[str, tuple[int, int]], keeps: int,
          own: Mapping[str, int]) -> tuple[float, dict[str, int]]:
    """f, and each class's RAM units `max(1, ⌊f × pieces⌋)`, for `left` bytes (the host memory left after the store's
    own host pools, `own`: name → bytes): f = (left − RESERVE) / every piece's bytes, rounded down to 1/STEP, below 1
    (exclusive tiers). `classes`: name → (pieces, unit bytes). Refused (OutOfUnits) below the working floor (`floor`),
    each term named."""
    total, least = sum(n * unit for n, unit in classes.values()), floor(classes, keeps)
    f = min(math.floor(max(0, left - RESERVE) * STEP / total), STEP - 1) / STEP
    if f < least:
        tier, mine = math.ceil(least * total), sum(own.values())
        need = mine + RESERVE + tier
        raise OutOfUnits(
            f"{name}: moe.ram=fit: below the working floor: the store needs {need / GiB:.2f} GiB past the host pools "
            f"pinned before it — its own I/O pools {mine / GiB:.2f} GiB ("
            + ", ".join(f"{k} {v / GiB:.3f}" for k, v in own.items()) + f"), the reserve {RESERVE / GiB:.2f} GiB (the "
            f"seed reads' bounce windows {BOUNCE / GiB:.2f}, the pinned buffers outside the ledger {PINNED / GiB:.2f}, "
            f"the process's growth {GROWTH / GiB:.2f}), a minimal RAM tier {tier / GiB:.2f} GiB (f {least:.4f} of the "
            f"experts' {total / GiB:.2f} GiB: every class a call's keeps, {keeps} units) — and "
            f"{(left + mine) / GiB:.2f} GiB was left (the host budget less those pools, or the RAM available less the "
            "headroom: the lesser). "
            f"Short by {(need - left - mine) / GiB:.2f} GiB: free that much RAM")
    return f, {c: max(1, math.floor(f * n)) for c, (n, _) in classes.items()}


def recheck(ctx, name: str, solved: int, need: int) -> None:
    """Just before the RAM tier's pool: refused (OutOfUnits) when the host memory left (`ctx.host_left()`) no longer
    holds `need` bytes (its pool and the reserve) — less than f was solved from (`solved`): never solved again, so no
    pool is sized twice (B-R6)."""
    if (now := ctx.host_left()) is None or now < need:
        raise OutOfUnits(f"{name}: moe.ram=fit: the host memory left fell from {solved / GiB:.2f} GiB, which its RAM "
                         f"share was solved from, to {(now or 0) / GiB:.2f} GiB before the RAM tier's pool, short of "
                         f"its {need / GiB:.2f} GiB with the reserve: something else took host memory meanwhile")
