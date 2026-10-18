"""Milestone K step 1's core plants (docs/v1-kv-tiers.md §13): each a one-line bug planted in a copy of `core/holon`,
paired with the core test it must fail. Run from the repository root:

    python spikes/k1_core/plants.py [--discover] [--only NAME ...] [--out evidence/k1_core_plants_cpu_<date>.json]

Default: run each plant's paired tests against the planted copy (`PYTHONPATH` first) and check they fail (a plant
whose tests pass, or time out, is reported). `--discover`: run the tier tests against each plant and list the tests
that fail (how the pairs were chosen). The stream-timing plant `store_no_after` (a store after the committed step,
not the last launched one) needs the GPU and stays with step 2's GPU plants."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

T = "core/tests/"
TIER_TESTS = [T + f for f in ("test_tiers.py", "test_payloads.py", "test_state_liveness.py", "test_prefix.py",
                              "test_sched.py")]
PAY, LIVE, TIERS, PREFIX = T + "test_payloads.py::", T + "test_state_liveness.py", T + "test_tiers.py::", \
    T + "test_prefix.py::"
# name: (file under core/holon/core, text, replacement, the tests it must fail)
PLANTS = {
    "abandon_under_request": ("prefix.py", "if not keys and freed:", "if not keys:",
                              [PAY + "test_clear_leaves_requests_their_units_and_loads_with_any_lru_order"]),
    "no_waiterless": ("state.py", "                idx.tier.loaded(load, load)\n", "                pass\n", [LIVE]),
    "no_rerun_fixed_point": ("state.py", "n = self._hit_blocks(seq, entries[:n]) if n < len(entries) else n", "pass",
                             [PAY + "test_a_hit_is_trimmed_to_its_units_and_to_payloads_the_tier_still_holds"]),
    "promote_after_last_only": ("state.py", "{name: us.tolist() for name, us in units.items()}, idx.after())",
                                "{name: us.tolist() for name, us in units.items()}, [f for f in [idx.last] if f])",
                                [PAY + "test_a_promotion_into_units_freed_under_a_store_waits_for_it"]),
    "branch_from_kids": ("state.py", "entries and entries[-1].children else 0", "entries and entries[-1].kids else 0",
                         [PAY + "test_a_staging_unit_is_reused_only_after_its_payload_is_stored"]),
    "promote_not_pure": ("prefix.py", "ok = lambda e: self._evictable(e) and (e.lower or not pure)",
                         "ok = lambda e: self._evictable(e)",
                         [PAY + "test_a_hit_is_trimmed_to_its_units_and_to_payloads_the_tier_still_holds"]),
    "staging_not_freed": ("prefix.py", "            self.freed[id(fence)] = fence\n", "",
                          [PAY + "test_a_staging_unit_is_reused_only_after_its_payload_is_stored"]),
    "store_no_copy": ("prefix.py", "after = [f for f in (self.last, *{id(e.copy): e.copy for e in entries}.values()) if "
                      "f is not None]", "after = [f for f in (self.last,) if f is not None]",
                      [PAY + "test_a_joining_request_waits_for_the_load_and_an_eviction_store_after_it"]),
    "double_fence": ("state.py", "                if not told:\n                    idx.tier.loaded(load, done)",
                     "                if True:\n                    idx.tier.loaded(load, done)", [LIVE]),
    "climb_with_policy": ("prefix.py", "need and self.policy is None else None", "need else None",
                          [PREFIX + "test_an_eviction_policy_keeps_the_roots_under_pressure"]),
    "drop_top_only": ("prefix.py", "self.tier.drop([x.key for x in gone])", "self.tier.drop([e.key])", [LIVE]),
    "promote_in_place_no_after": ("state.py", "idx.promote([e], {name: list(us) for name, us in mine.items()}, "
                                  "idx.after())", "idx.promote([e], {name: list(us) for name, us in mine.items()}, [])",
                                  [PAY + "test_a_recomputed_block_only_the_tier_holds_is_promoted_into_place"]),
    "swap_onto_lower": ("state.py", "                if key in idx.lower:           # only the tier holds it: into "
                        "this request's own units (D7)\n                    idx.promote([e], {name: list(us) for "
                        "name, us in mine.items()}, idx.after())\n", "",
                        [TIERS + "test_steps_and_outputs_never_depend_on_when_copies_land"]),
    "joiner_no_wait": ("state.py", "{id(e.loading): e.loading for e in entries if e.loading is not None}",
                       "{id(e.loading): e.loading for e in entries[len(resident):] if e.loading is not None}",
                       [PAY + "test_a_joining_request_waits_for_the_load_and_an_eviction_store_after_it"]),
    "publish_no_wait": ("state.py", "                if e.loading is not None:\n                    self.waits."
                        "setdefault(seq.slot, []).append(e.loading)\n            else:", "            else:",
                        [PAY + "test_a_recomputed_block_only_the_tier_holds_is_promoted_into_place"]),
    "payload_load_after_empty": ("state.py", "load = idx.tier.load([], {}, tier, idx.after())",
                                 "load = idx.tier.load([], {}, tier, [])",
                                 [PAY + "test_a_recomputed_block_only_the_tier_holds_is_promoted_into_place"]),
    "gate_off": ("sched.py", "if self.state.due.get(seq.slot, now) > now:", "if False:",
                 [TIERS + "test_steps_and_outputs_never_depend_on_when_copies_land"]),
    "pull_latest": ("sched.py", "del self.state.due[min(self.state.due, key=self.state.due.get)]",
                    "del self.state.due[max(self.state.due, key=self.state.due.get)]",
                    [PAY + "test_when_only_gated_requests_are_left_the_earliest_due_runs"]),
    "resident_lower_kept": ("prefix.py", "            elif key in self.entries:\n                e.lower = False\n",
                            "            elif key in self.entries:\n                pass\n",
                            [TIERS + "test_steps_and_outputs_never_depend_on_when_copies_land"]),
    "no_poll_at_commit": ("engine.py", "            self.state.index.poll()                      # what the tier "
                          "dropped in its hooks, before any publish", "            pass",
                          [TIERS + "test_steps_and_outputs_never_depend_on_when_copies_land"]),
    "hit_none": ("state.py", "idx.tier.hit(list(self.chain[seq.slot]),", "idx.tier.hit([],",
                 [PAY + "test_a_unit_freed_under_a_store_is_reused_only_after_it_and_a_hit_reads_the_tier"]),
    "release_keeps_due": ("state.py", "for d in (self.waits, self.due):", "for d in (self.waits,):", [LIVE]),
    "no_poll_after_promote": ("prefix.py", "        self.poll()                         # what the tier dropped to make "
                              "room (never what it is loading)\n", "",
                              [PAY + "test_a_hit_is_trimmed_to_its_units_and_to_payloads_the_tier_still_holds"]),
    "no_refit_after_poll": ("state.py", "                entries = entries[:self._hit_blocks(seq, entries)]\n",
                            "                pass\n",
                            [PAY + "test_a_hit_is_trimmed_to_its_units_and_to_payloads_the_tier_still_holds"]),
    "waits_from_trimmed": ("state.py", "{id(e.loading): e.loading for e in entries if e.loading is not None}",
                           "{id(e.loading): e.loading for e in entries + [e for e in idx.entries.values() if "
                           "e.loading] if e.loading is not None}",
                           [PAY + "test_a_joining_request_waits_for_the_load_and_an_eviction_store_after_it"]),
    "no_settle": ("state.py", "if self.index.settle(key, p, spec, drop=False)]", "if False]",
                  [PAY + "test_a_request_keeps_its_last_tentative_payloads"]),
    "policy_unguarded": ("prefix.py", "            if e is not None and (self.entries.get(e.key) is not e or not "
                         "ok(e)):\n", "            if False:\n",
                         [PREFIX + "test_an_eviction_policy_keeps_the_roots_under_pressure"]),
    "clear_via_drop": ("prefix.py", "        for e in self.entries.values():\n            self._free(e)\n        "
                       "self._unhold", "        for e in reversed([*self.lower.values(), *self.entries.values()]):\n"
                       "            if self.get(e.key) is e:\n                self._drop(e)\n        self._unhold",
                       [PAY + "test_clear_leaves_requests_their_units_and_loads_with_any_lru_order"]),
}


def run(name: str, tests: list[str], scratch: pathlib.Path, timeout: int) -> tuple[str, list[str]]:
    f, old, new, _ = PLANTS[name]
    copy = scratch / name
    shutil.copytree("core/holon", copy / "holon")
    path = copy / "holon" / "core" / f
    src = path.read_text()
    if src.count(old) != 1:
        return f"stale ({src.count(old)} matches)", []
    path.write_text(src.replace(old, new))
    env = dict(os.environ, PYTHONPATH=str(copy), CUDA_VISIBLE_DEVICES="")
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rf", *tests],
                           env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "timeout", []
    failed = [line.split()[1] for line in r.stdout.splitlines() if line.startswith("FAILED ")]
    return ("fails" if r.returncode else "passes"), failed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--discover", action="store_true")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--out")
    args = ap.parse_args()
    rows = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name in args.only or PLANTS:
            verdict, failed = run(name, TIER_TESTS if args.discover else PLANTS[name][3], pathlib.Path(tmp),
                                  args.timeout)
            rows[name] = {"file": PLANTS[name][0], "tests": PLANTS[name][3], "verdict": verdict, "failed": failed}
            print(f"{name}: {verdict} {failed if args.discover else ''}", flush=True)
    caught = sum(r["verdict"] == "fails" for r in rows.values())
    print(f"{caught} / {len(rows)} plants fail their tests")
    if args.out:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        pathlib.Path(args.out).write_text(json.dumps({
            "measurement": "K step 1's core plants, each against the core test it must fail (CPU)",
            "command": "python spikes/k1_core/plants.py" + (" --discover" if args.discover else ""), "commit": commit,
            "dirty": bool(subprocess.run(["git", "status", "--porcelain", "core"], capture_output=True,
                                         text=True).stdout.strip()),
            "gpu_only": {"store_no_after": "a store issued after the committed step, not the last launched one: "
                                           "a stream-timing plant for step 2 (GPU)"},
            "plants": rows, "caught": f"{caught} / {len(rows)}"}, indent=1) + "\n")


if __name__ == "__main__":
    main()
