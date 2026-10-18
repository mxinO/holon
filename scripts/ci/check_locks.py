"""The committed lock histories (`plans/`; design §4.4, release design §7):

- every lock reads back: its `digest` and `plan_id` recomputed by `canonical_hash` match (the committed locks are its
  golden vectors), so neither a hand edit nor a change to the hashing goes unseen;
- one model id per checkpoint build (`holon serve` and the tools find a lock by its build alone), named so that
  `holon serve <name>` fetches it (`<org>/<repo>` or `<org>/<repo>/<stem>`: what `holon models` prints);
- locks never pin test-only plugins (the `testing.*` distribution). The tools plan and write locks for any engine —
  their own tests write CPU locks on the reference kernels into temporary directories — so this is enforced here;
- every plugin option a lock sets is declared by an installed plugin (manifest `[options]`), with a value in its
  `choices`, so `holon serve` serves a committed lock without a warning and never drops it;
- no lock carries a CPU tier (a class `<gpu>+<cpu tier>/…`, F5b's split) until the lead decides the policy for
  benchmark-only and tier locks: the lookup ranks a lock of the host's tier first, so a committed one would become the
  default of every host of that tier (docs/plan.md §10, "F5b locks").

`--release` (before a release tag) also checks that every newest lock pins the unit sources of this tree.
`--plans <dir>` checks another tree of lock histories (default: the repository's `plans/`). Run in the project
environment."""
from __future__ import annotations

import collections
import pathlib
import sys

from _common import ROOT, fail
from holon.core import cli, plans
from holon.core.plugins import discover

PLANS = pathlib.Path(sys.argv[sys.argv.index("--plans") + 1]) if "--plans" in sys.argv else ROOT / "plans"
declared = cli.declared_options(discover())
msgs = []
for path in sorted(PLANS.rglob("*.lock")):
    try:
        lock = plans.read(path)
    except ValueError as e:
        msgs.append(str(e).replace(str(PLANS.parent) + "/", ""))
        continue
    name = path.relative_to(PLANS.parent)
    msgs += [f"{name}: pins {unit}" for unit in lock.plan.units if unit.startswith("testing.")]
    if "+" in path.parents[1].name or "+" in lock.plan.fingerprint_class:
        msgs.append(f"{name}: carries a CPU tier ({lock.plan.fingerprint_class}): no tier locks until their policy is "
                    "decided (docs/plan.md §10, F5b locks)")
    options = lock.plan.engine.get("plugin_options", {})
    msgs += [f"{name}: no installed plugin declares the option {key}" for key in options if key not in declared]
    msgs += [f"{name}: {bad}" for bad in cli.outside_choices(options, declared)]
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
            msgs += [f"{found[-1].relative_to(PLANS.parent)}: {name} differs from this tree" for name, pin in
                     lock.plan.units.items() if current.get(name) != pin.src]
fail(msgs, "locks read back, one servable model id per build, no test-only plugins, declared options, no CPU tier"
     + (", newest locks pin this tree" if "--release" in sys.argv else ""))
