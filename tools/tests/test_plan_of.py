"""The plan of a built engine (design §4.3): stable across builds, and it moves with kernel pins."""
import dataclasses
import textwrap

import pytest
from holon.api import Lock, Manifest
from holon.core import plans
from holon.core.plugins import Installed, PluginSet, discover

FAKE = "tests.fake_rmsnorm"


@pytest.fixture
def with_fake_kernel(tmp_path, monkeypatch) -> PluginSet:
    """The installed plugins plus a second CPU rmsnorm provider (lower priority than std.ops)."""
    pkg = tmp_path / "fake_rmsnorm"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "plugin.toml").write_text(textwrap.dedent(f"""
        [plugin]
        name = "{FAKE}"
        version = "0.1.0"
        entry = "plugin:Plugin"
        slot = "kernel"
        license = "Apache-2.0"

        [resources]
        provides = ["kernel.rmsnorm@1"]
        """))
    (pkg / "plugin.py").write_text(textwrap.dedent(f"""
        import torch
        from holon.api import Constraint, KernelSpec

        class RMSNorm:
            def __init__(self, ctx, *, dim, eps, offset, dtype):
                self.eps, self.offset = eps, offset

            def __call__(self, x, weight):
                v = x.float()
                y = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + self.eps)
                return (y * (weight.float() + self.offset)).to(x.dtype)

        class Plugin:
            def register(self, regs):
                regs.kernels.add("rmsnorm", KernelSpec(
                    op="rmsnorm", provider="{FAKE}", impl=RMSNorm, requires=Constraint(devices=frozenset({{"cpu"}})),
                    props=frozenset({{"deterministic"}}), priority=-200))
        """))
    monkeypatch.syspath_prepend(str(tmp_path))
    return PluginSet(discover() | {FAKE: Installed(Manifest.load(pkg / "plugin.toml"), "fake_rmsnorm", pkg)})


def test_plan_is_stable_across_builds(tiny_engine, tiny_checkpoint):
    engine = tiny_engine()
    plan = plans.plan_of(engine)
    assert plans.plan_of(tiny_engine()).plan_id == plan.plan_id
    assert plan.model_id.startswith(f"{tiny_checkpoint.name}@") and plan.fingerprint_class.startswith("cpu/")
    assert {plans.CORE_UNIT, "models.qwen3", "std.ops", "std.mixers.gqa", "std.weights.hf"} <= plan.units.keys()
    assert plan.units["std.ops"].sem and plan.units["std.blocks"].sem is None      # only numeric units have oracles
    assert plan.sites.keys() == engine.ctx.kernels.keys()
    assert plans.uncovered(plan) == ()
    assert plan.engine["max_step_tokens"] == 16 and plan.engine["dtype"] == "float32"
    assert "state_memory" not in plan.engine and "kernels" not in plan.engine        # per-host budget; sites


def test_plan_moves_with_engine_settings(tiny_engine):
    changed = plans.diff(plans.plan_of(tiny_engine()), plans.plan_of(tiny_engine(max_step_tokens=8)))
    assert changed == {"engine:max_step_tokens": (16, 8)}


def test_plan_moves_with_a_kernel_pin(tiny_engine, with_fake_kernel):
    default = plans.plan_of(tiny_engine(with_fake_kernel))
    pinned = plans.plan_of(tiny_engine(with_fake_kernel, kernels={"rmsnorm": FAKE}))
    assert FAKE in default.units                    # loaded as a candidate, so it is in the closure
    changed = plans.diff(default, pinned)
    assert pinned.plan_id != default.plan_id and changed
    assert all(key.startswith("site:") and key.endswith("/rmsnorm@1") for key in changed)
    assert {new.impl for _, new in changed.values()} == {FAKE}


def test_lock_overrides_rebuild_the_same_plan(tiny_engine, with_fake_kernel):
    pinned = plans.plan_of(tiny_engine(with_fake_kernel, kernels={"rmsnorm": FAKE}))
    lock = plans.loads(plans.dumps(Lock(pinned)))
    rebuilt = plans.plan_of(tiny_engine(with_fake_kernel, **plans.lock_overrides(lock)))
    assert rebuilt.plan_id == pinned.plan_id


def test_weight_changes_change_the_model_id(tiny_engine, tiny_checkpoint, tmp_path):
    copy = tmp_path / tiny_checkpoint.name
    copy.mkdir()
    for f in tiny_checkpoint.iterdir():
        data = f.read_bytes()
        (copy / f.name).write_bytes(data[:-1] + bytes([data[-1] ^ 1]) if f.suffix == ".safetensors" else data)
    a, b = plans.plan_of(tiny_engine()), plans.plan_of(tiny_engine(model=str(copy)))
    assert a.model_id != b.model_id and dataclasses.replace(b, model_id=a.model_id).plan_id == a.plan_id
