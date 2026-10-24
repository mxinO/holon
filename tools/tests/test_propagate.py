"""`propagate`: the outcomes of design §4.6 with scripted checks, and the command line end to end on a CPU model."""
import dataclasses

import pytest
import torch
from holon.api import Binding, Lock, Plan, UnitPin
from holon.core import plans
from holon.core.engine import EngineConfig
from holon.core.memory import Ledger
from holon_judge import golden
from holon_judge.snapshot import Snapshot
from holon_tools import locks, propagate
from holon_tools.__main__ import main

SITE = "kv/attention.paged_decode@1"


def _plan(src="aaaa", commit="c1", props=("deterministic",)) -> Plan:
    return Plan("org/m@1", "rtx-a5000/cu130-torch2.11", "sha256:r", commit, "none",
                {"holon": UnitPin("0.0.1", "core"), "kern": UnitPin("1.0.0", src)},
                {SITE: Binding("kern", {"facts": {"heads": 4}, "props": list(props)})})


def _snap(token=7) -> Snapshot:
    return Snapshot(streams=[[[token, 8]]], steps=[[], [[0, 0, f"{token:064x}"]]])


@dataclasses.dataclass
class Scripted:
    """Checks that return given results and count how often each ran."""

    new: Plan
    snaps: list
    l2_passed: bool = True
    calls: list = dataclasses.field(default_factory=list)

    def plan(self):
        self.calls.append("plan")
        return self.new

    def snapshot(self):
        self.calls.append("snapshot")
        return self.snaps.pop(0)

    def l2(self):
        self.calls.append("l2")
        return self.l2_passed, "L2 " + ("PASS" if self.l2_passed else "FAIL")


@pytest.fixture
def current():
    return Lock(_plan(), propagate.trace_hash(_snap()), (), sm=86)


def test_unchanged_closure_writes_nothing(current):
    checks = Scripted(_plan(commit="c2"), [])
    out = propagate.decide(current, checks, 86)
    assert out.verdict == "unchanged" and out.lock is None and checks.calls == ["plan"]


def test_identical_snapshot_advances_without_l2(current):
    checks = Scripted(_plan(src="bbbb", commit="c2"), [_snap(), _snap()])
    out = propagate.decide(current, checks, 86)
    assert out.verdict == "advance" and checks.calls == ["plan", "snapshot", "snapshot"]
    assert out.lock.snapshot == current.snapshot and out.lock.plan.units["kern"].src == "bbbb"
    assert list(out.changes) == ["commit", "unit:kern"]


def test_identical_snapshot_with_a_settings_change_needs_approval_without_l2(current):
    # e.g. a longer max_model_len: the trace cannot show the pools' margin at it
    plan = dataclasses.replace(_plan(src="bbbb"), engine={"max_model_len": 49152})
    checks = Scripted(plan, [_snap(), _snap()])
    out = propagate.decide(current, checks, 86)
    assert out.verdict == "approve" and "l2" not in checks.calls and out.lock.snapshot == current.snapshot
    assert out.evidence[0].endswith("not only code changed: ['engine:max_model_len'] (an approval; no L2)")


def test_bounded_change_needs_approval(current):
    out = propagate.decide(current, Scripted(_plan(src="bbbb"), [_snap(9), _snap(9)]), 86, previous=_snap())
    assert out.verdict == "approve" and out.lock.snapshot == propagate.trace_hash(_snap(9))
    assert out.evidence == ["snapshot trace differs from the lock's: step 1: deltas differ for (request, sequence) "
                            "[(0, 0)]", "trace covered step widths -, prefix tokens saved -", "L2 PASS"]


def test_l2_failure_fails(current):
    out = propagate.decide(current, Scripted(_plan(src="bbbb"), [_snap(9), _snap(9)], l2_passed=False), 86)
    assert out.verdict == "fail" and out.lock is None and out.evidence[-1] == "L2 FAIL"


@pytest.mark.parametrize("runs", [2, 3])
def test_a_snapshot_that_is_not_run_identical_fails(current, runs):
    checks = Scripted(_plan(src="bbbb"), [_snap(9)] * (runs - 1) + [_snap(10)])     # the opt-in third run decides
    out = propagate.decide(current, checks, 86, runs=runs)
    assert out.verdict == "fail" and "not run-identical" in out.evidence[0] and "l2" not in checks.calls
    assert checks.calls.count("snapshot") == runs


def test_changed_uncovered_site_never_advances():
    racy = _plan(props=())
    current = Lock(racy, propagate.trace_hash(_snap()), locks.uncovered(racy), sm=86)
    out = propagate.decide(current, Scripted(_plan(src="bbbb", props=()), [_snap(), _snap()]), 86)
    assert out.verdict == "approve" and "uncovered sites changed" in out.evidence[0]


def test_newly_nondeterministic_site_never_advances(current):
    out = propagate.decide(current, Scripted(_plan(props=()), [_snap(), _snap()]), 86)
    assert out.verdict == "approve" and out.evidence[0].endswith(f"uncovered sites changed: ['{SITE}']")


def test_unstable_snapshot_with_uncovered_sites_goes_to_l2(current, tmp_path):
    out = propagate.decide(current, Scripted(_plan(props=()), [_snap(9), _snap(10)]), 86)
    assert out.verdict == "approve" and out.lock.snapshot is None and out.snapshot is None
    assert "not run-identical" in out.evidence[0] and SITE in out.evidence[0]
    propagate.write(tmp_path, out)
    assert locks.heads(tmp_path)[0].snapshot is None and not list(tmp_path.rglob("snapshots"))


def test_a_changed_trace_is_named(current):
    new = dataclasses.replace(_snap(9), meta={"trace": "b"})
    previous = dataclasses.replace(_snap(), meta={"trace": "a"})
    out = propagate.decide(current, Scripted(_plan(src="bbbb"), [new, new]), 86, previous=previous)
    assert "the trace's requests changed" in out.evidence[0]


def test_new_model_always_needs_approval():
    out = propagate.decide(None, Scripted(_plan(), [_snap(), _snap()]), 86)
    assert out.verdict == "approve" and out.evidence[0] == "new model: first lock" and out.evidence[-1] == "L2 PASS"


def test_write_stores_lock_and_snapshot(tmp_path, current):
    out = propagate.decide(None, Scripted(_plan(), [_snap(), _snap()]), 86)
    path = propagate.write(tmp_path, out)
    assert path.name == "1.lock" and plans.read(path).plan.plan_id == out.lock.plan.plan_id
    stored = Snapshot.load(propagate.snapshot_path(tmp_path, out.lock.plan, out.lock.snapshot))
    assert propagate.trace_hash(stored) == out.lock.snapshot
    with pytest.raises(ValueError, match="nothing to write"):
        propagate.write(tmp_path, propagate.Outcome("unchanged", {}, []))
    # A plan binding an `approx` plugin (A1's tail skip, as installed) is a human gate, never a default lock: refused
    # whatever its verdict, written only with the explicit override.
    plan = _plan()
    approx = dataclasses.replace(plan, units=plan.units | {"transforms.tail_skip": UnitPin("1.0.0", "bbbb")})
    out = propagate.decide(None, Scripted(approx, [_snap(), _snap()]), 86)
    assert out.verdict == "approve" and propagate.approx(approx) == ["transforms.tail_skip"]
    with pytest.raises(ValueError, match=r"binds approx plugins \['transforms.tail_skip'\]"):
        propagate.write(tmp_path, out)
    assert propagate.write(tmp_path, out, allow_approx=True).name == "2.lock"


def test_snapshot_engines_share_one_pool_size(monkeypatch):
    # Pool sizes set the schedule a trace records: every snapshot engine gets the first one's pools (a later build would
    # measure less free memory), or those of the lock's stored snapshot (recorded in another process). A host pool has
    # its own budget: it is no device memory to pass on.
    built = []

    class Measuring:                                             # each build would measure 10 bytes less
        def __init__(self, config):
            capacity = config.state_memory or 1000 - 10 * len(built)
            self.ledger = Ledger(torch.device("cpu"), host_memory=64)
            self.ledger.request("kv", unit_bytes=1, min_units=capacity, pref_units=capacity)
            self.ledger.plugin_pool("tier", unit_bytes=64, min_units=1, device="host")
            self.ledger.materialize(None)
            built.append(config.state_memory)

        def has_work(self):
            return False

    monkeypatch.setattr(propagate, "Engine", Measuring)
    checks = propagate.EngineChecks(EngineConfig(model="m"), [], None)
    first, again = checks.snapshot(), checks.snapshot()
    assert built == [None, 1000] and first.meta["pool_bytes"] == again.meta["pool_bytes"] == 1000
    propagate.EngineChecks(EngineConfig(model="m"), [], None, pool_bytes=990).snapshot()
    assert built[-1] == 990


@pytest.mark.slow
def test_cli_on_cpu(tiny_checkpoint, tiny_prompts, tiny_prompts_file, cpu_flags, tmp_path, capsys, monkeypatch):
    gold, floor = tmp_path / "gold.safetensors", tmp_path / "floor.safetensors"
    pools, checks = [], propagate.EngineChecks          # the stored pools each run's checks start from
    monkeypatch.setattr(propagate, "EngineChecks", lambda *a: pools.append(a[3]) or checks(*a))
    golden.record(str(tiny_checkpoint), tiny_prompts).save(gold)
    golden.record(str(tiny_checkpoint), tiny_prompts, dtype=torch.bfloat16).save(floor)

    def run(plans_dir, *extra, **engine):
        # Room for the snapshot trace: 62 short requests of up to 32 + 65 tokens, queued four at a time.
        engine = cpu_flags | {"max_model_len": 128, "state_memory": 4 << 20} | engine
        flags = [f"--{k.replace('_', '-')}={str(v).removeprefix('torch.')}" for k, v in engine.items()]
        return main(["propagate", str(tiny_checkpoint), "--plans", str(plans_dir), "--prompts", str(tiny_prompts_file),
                     "--golden", str(gold), "--floor", str(floor), *flags, *extra])

    home = tmp_path / "plans"
    with pytest.raises(SystemExit, match="no lock yet"):
        run(home)
    assert run(home, "--new-model") == 2                       # passes L2, waits for approval
    assert run(home, "--new-model", "--approve") == 0
    assert "verdict: approve" in capsys.readouterr().out
    (first,) = home.rglob("*.lock")
    assert first.name == "1.lock" and len(list(home.rglob("snapshots/*.json"))) == 1
    assert run(home) == 0                                       # same code and settings: nothing to do
    assert "verdict: unchanged" in capsys.readouterr().out
    assert run(home, max_step_tokens=8) == 2                    # a settings change is a checked change
    assert "engine:max_step_tokens: 16 -> 8" in capsys.readouterr().out
    assert pools[-1] == 4 << 20                                 # on the lock's own class: its stored pools
    assert len(list(home.rglob("*.lock"))) == 1

    # Another class's lock, verified here: the trace is byte-identical, so no L2, but a new class needs an approval.
    lock = plans.read(first)
    other = dataclasses.replace(lock.plan, fingerprint_class="cpu/other-toolchain")
    away = tmp_path / "away"
    locks.append(away, dataclasses.replace(lock, plan=other))
    stored = propagate.snapshot_path(away, other, lock.snapshot)
    stored.parent.mkdir(parents=True)
    stored.write_bytes(propagate.snapshot_path(home, lock.plan, lock.snapshot).read_bytes())
    assert run(away) == 2
    out = capsys.readouterr().out
    assert "starting from cpu/other-toolchain's" in out and "verdict: approve" in out
    assert "snapshot trace byte-identical" in out and "L2 " not in out and "prefix tokens saved 0" not in out
    assert pools[-1] is None                                    # another class's pools may not fit here: measured
    assert run(away, "--approve") == 0
    assert plans.newest(away, lock.plan.model_id, lock.plan.fingerprint_class).snapshot == lock.snapshot
