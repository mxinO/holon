"""Warns, never fails: the committed newest locks a range outdates (base..HEAD changes the `src` of a unit they pin;
`holon_tools impact`). Merged, they read as outdated, and `holon serve` falls back to another class's current lock or
the defaults until the next propagation round, so such a branch merges at a release or a milestone's close, with that
round (docs/dev/release.md §22). A change to `core/holon` outdates every lock. Usage: check_lock_impact.py [base]"""
from __future__ import annotations

import sys

from _common import ROOT
from holon.core import plans
from holon.core.plugins import discover
from holon_tools import impact, locks

base = sys.argv[1] if len(sys.argv) > 1 else "origin/main"
try:
    units = impact.changed_units([f"{base}..HEAD"], plans.unit_dirs(discover()), ROOT)
except ValueError as e:
    sys.exit(print(f"skip lock impact: {e}"))
if hit := impact.impact(locks.heads(ROOT / "plans"), units):
    pinned = sorted({u for _, us in hit for u in us})
    print(f"WARN {base}..HEAD outdates {len(hit)} committed locks (through {', '.join(pinned)}): merge it at a release "
          "or a milestone's close, with the propagation round (docs/dev/release.md §22)")
else:
    print(f"ok   {base}..HEAD outdates no committed lock")
