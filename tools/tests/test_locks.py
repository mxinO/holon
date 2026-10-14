"""Lock files and lock history (design §4.4): format round trip, append/newest/heads, nearest-class fallback."""
import dataclasses

import pytest
import torch
from holon.api import Binding, Fingerprint, Lock, Plan, UnitPin
from holon.core import plans
from holon_tools import locks

MODEL = "org/model@0123456789abcdef"


def _plan(cls="rtx-a5000/cu130-torch2.11", decode="kernels.fast", src="aaaa", commit="c1", step=4096) -> Plan:
    facts = {"heads": 4, "eps": 1e-6, "theta": 1e6, "dtype": "bfloat16", "quoted \"key\"": True}
    return Plan(model_id=MODEL, fingerprint_class=cls, recipe="sha256:r", commit=commit, uv_lock="none",
                units={"holon": UnitPin("0.0.1", "core"), "std.ops": UnitPin("1.0.0", src, "sha256:sem"),
                       "kernels.fast": UnitPin("1.0.0", "ffff")},
                sites={"kv/attention.paged_decode@1": Binding(decode, {"facts": facts, "props": ["deterministic"]}),
                       "std.ops.rmsnorm[dim=64]/rmsnorm@1": Binding("std.ops", {"facts": facts, "props": []})},
                engine={"dtype": "bfloat16", "max_step_tokens": step, "cuda_graphs": True})


def test_lock_round_trip_is_exact():
    lock = Lock(_plan(), "sha256:snap", ("std.ops.rmsnorm[dim=64]/rmsnorm@1",), sm=86)
    text = locks.dumps(lock)
    back = plans.loads(text)
    assert back.plan.plan_id == lock.plan.plan_id and locks.dumps(back) == text
    assert (back.snapshot, back.uncovered, back.sm) == (lock.snapshot, lock.uncovered, 86)
    assert plans.loads(locks.dumps(Lock(_plan()))).snapshot is None


def test_round_trip_of_nested_tuples_and_unicode():
    plan = _plan()
    odd = Binding("kernels.fast", {"facts": {"shape": (1, (2, 3)), "név": "ünï\tcode", "neg": -1.5e-300},
                                   "launch": {"tiles": [64, 128], "split": {"k": 2}}}, {"warps": 4})
    plan = dataclasses.replace(plan, sites=plan.sites | {"ü site/rmsnorm@1": odd})
    back = plans.loads(locks.dumps(Lock(plan)))
    assert back.plan.plan_id == plan.plan_id
    assert back.plan.sites["ü site/rmsnorm@1"].tunables == {"warps": 4}


@pytest.mark.parametrize("edit", [('"kernels.fast" }', '"std.mixers.gqa" }'),     # a binding (in the plan)
                                  ("sm = 86", "sm = 89"),                           # verification results
                                  ('trace = "sha256:snap"', 'trace = "sha256:other"'),
                                  ("uncovered = [", 'uncovered = ["kv/attention.paged_decode@1", ')])
def test_hand_edits_are_rejected(tmp_path, edit):
    text = locks.dumps(Lock(_plan(), "sha256:snap", ("std.ops.rmsnorm[dim=64]/rmsnorm@1",), sm=86))
    assert edit[0] in text
    path = tmp_path / "1.lock"
    path.write_text(text.replace(*edit))
    with pytest.raises(ValueError, match="edited by hand"):
        plans.read(path)
    path.write_text(text.replace("# Written", "# A comment edit is harmless. Written"))
    assert plans.read(path).sm == 86


def test_unwritable_values_fail_loudly():
    bad = dataclasses.replace(_plan(), sites={"s/rmsnorm@1": Binding("x", {"facts": {"none": None}})})
    with pytest.raises(TypeError, match="NoneType"):
        locks.dumps(Lock(bad))


def test_history_append_newest_heads(tmp_path):
    cls = "rtx-a5000/cu130-torch2.11"
    assert plans.newest(tmp_path, MODEL, cls) is None
    first = locks.append(tmp_path, Lock(_plan()))
    second = locks.append(tmp_path, Lock(_plan(decode="std.mixers.gqa")))
    assert (first.name, second.name) == ("1.lock", "2.lock")
    assert first.relative_to(tmp_path).as_posix() == f"{MODEL}/{cls}/1.lock"
    assert plans.newest(tmp_path, MODEL, cls).plan.sites["kv/attention.paged_decode@1"].impl == "std.mixers.gqa"
    assert plans.history(tmp_path, MODEL, cls) == [first, second]      # rollback = the one before the newest
    locks.append(tmp_path, Lock(_plan(cls="rtx-4090/cu130-torch2.11")))
    assert sorted(lk.plan.fingerprint_class for lk in locks.heads(tmp_path)) == ["rtx-4090/cu130-torch2.11", cls]
    with pytest.raises(ValueError, match="not a model id"):
        plans.history(tmp_path, "../escape", cls)


def test_nearest_class_prefers_own_then_same_gpu_then_same_sm(tmp_path):
    fp = Fingerprint(gpu="rtx-3090", sm=86, vram_gib=24, toolchain="cu130-torch2.11")
    assert plans.nearest_class(tmp_path, MODEL, fp) is None
    locks.append(tmp_path, Lock(_plan(cls="rtx-5090/cu130-torch2.11"), sm=120))
    assert plans.nearest_class(tmp_path, MODEL, fp)[0].plan.fingerprint_class == "rtx-5090/cu130-torch2.11"
    locks.append(tmp_path, Lock(_plan(cls="rtx-a5000/cu128-torch2.10"), sm=86))
    lock, unverified, path = plans.nearest_class(tmp_path, MODEL, fp)
    assert path.parent.name == "cu128-torch2.10" and path.name == "1.lock"
    assert lock.plan.fingerprint_class == "rtx-a5000/cu128-torch2.10" and unverified       # same SM beats toolchain
    locks.append(tmp_path, Lock(_plan(cls="rtx-3090/cu128-torch2.10"), sm=86))
    assert plans.nearest_class(tmp_path, MODEL, fp)[0].plan.fingerprint_class == "rtx-3090/cu128-torch2.10"
    locks.append(tmp_path, Lock(_plan(cls=fp.cls), sm=86))
    lock, unverified, _ = plans.nearest_class(tmp_path, MODEL, fp)
    assert lock.plan.fingerprint_class == fp.cls and not unverified


def test_outdated_locks_are_treated_as_missing(tmp_path):
    fp = Fingerprint(gpu="rtx-a5000", sm=86, vram_gib=24, toolchain="cu130-torch2.11")
    locks.append(tmp_path, Lock(_plan(src="old"), sm=86))
    locks.append(tmp_path, Lock(_plan(cls="rtx-3090/cu130-torch2.11", src="new"), sm=86))
    current = {"holon": "core", "std.ops": "new", "kernels.fast": "ffff"}
    lock, unverified, _ = plans.nearest_class(tmp_path, MODEL, fp, current)
    assert lock.plan.fingerprint_class == "rtx-3090/cu130-torch2.11" and unverified
    assert plans.nearest_class(tmp_path, MODEL, fp, current | {"std.ops": "newer"}) is None


def test_diff_lists_changed_units_and_sites():
    a, b = _plan(), _plan(decode="std.mixers.gqa", src="bbbb", commit="c2", step=512)
    d = plans.diff(a, b)
    assert sorted(d) == ["commit", "engine:max_step_tokens", "site:kv/attention.paged_decode@1", "unit:std.ops"]
    assert d["unit:std.ops"][1].src == "bbbb" and d["engine:max_step_tokens"] == (4096, 512)
    smaller = dataclasses.replace(a, units={k: v for k, v in a.units.items() if k != "kernels.fast"})
    assert plans.diff(a, smaller) == {"unit:kernels.fast": (a.units["kernels.fast"], None)}
    assert plans.diff(a, a) == {}


def test_lock_overrides_rebuild_settings_and_prefer_each_op():
    assert plans.lock_overrides(Lock(_plan())) == {
        "dtype": torch.bfloat16, "max_step_tokens": 4096, "cuda_graphs": True,
        "kernel_preferences": {"attention.paged_decode": "kernels.fast", "rmsnorm": "std.ops"}}
    plan = _plan()       # an op bound to two providers (preferences are per op) is left to default priority
    split = dataclasses.replace(plan, sites=plan.sites | {"other/rmsnorm@1": Binding("kernels.fast")})
    assert plans.lock_overrides(Lock(split))["kernel_preferences"] == {"attention.paged_decode": "kernels.fast"}


def test_uncovered_sites_lack_determinism():
    assert locks.uncovered(_plan()) == ("std.ops.rmsnorm[dim=64]/rmsnorm@1",)
