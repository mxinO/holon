"""The committed lock histories (`plans/`; design §4.4, release design §7):

- every lock reads back: its `digest` and `plan_id` recomputed by `canonical_hash` match (the committed locks are its
  golden vectors), so neither a hand edit nor a change to the hashing goes unseen;
- one model id per checkpoint build (`holon serve` and the tools find a lock by its build alone), named so that
  `holon serve <name>` fetches it (`<org>/<repo>` or `<org>/<repo>/<stem>`: what `holon models` prints);
- locks never pin test-only plugins (the `testing.*` distribution). The tools plan and write locks for any engine —
  their own tests write CPU locks on the reference kernels into temporary directories — so this is enforced here;
- every plugin option a lock sets is declared by an installed plugin (manifest `[options]`), so `holon serve` serves
  a committed lock without the undeclared-option warning.

`--release` (before a release tag) also checks that every newest lock pins the unit sources of this tree. Run in the
project environment."""
from __future__ import annotations

import collections
import sys

from _common import ROOT, fail
from holon.core import cli, plans
from holon.core.plugins import discover

PLANS = ROOT / "plans"
declared = cli.declared_options(discover())
msgs = []
for path in sorted(PLANS.rglob("*.lock")):
    try:
        lock = plans.read(path)
    except ValueError as e:
        msgs.append(str(e).replace(str(ROOT) + "/", ""))
        continue
    msgs += [f"{path.relative_to(ROOT)}: pins {unit}" for unit in lock.plan.units if unit.startswith("testing.")]
    msgs += [f"{path.relative_to(ROOT)}: no installed plugin declares the option {key}"
             for key in lock.plan.engine.get("plugin_options", {}) if key not in declared]
builds = collections.defaultdict(list)
for model_id in plans.model_ids(PLANS):
    builds[model_id.rpartition("@")[2]].append(model_id)
msgs += [f"build {b}: lock histories under several model ids {ids}" for b, ids in builds.items() if len(ids) > 1]
msgs += [f"{model_id}: not servable by name (<org>/<repo>[/<stem>])" for model_id in plans.model_ids(PLANS)
         if not cli.SERVABLE.fullmatch(model_id.rpartition("@")[0])]
if "--release" in sys.argv:
    current = plans.current_srcs(discover())
    for directory in sorted({p.parent for p in PLANS.rglob("*.lock")}):
        if found := plans.numbered(directory):
            lock = plans.read(found[-1])
            msgs += [f"{found[-1].relative_to(ROOT)}: {name} differs from this tree" for name, pin in
                     lock.plan.units.items() if current.get(name) != pin.src]
fail(msgs, "locks read back, one servable model id per build, no test-only plugins, declared options"
     + (", newest locks pin this tree" if "--release" in sys.argv else ""))
