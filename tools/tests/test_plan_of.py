"""The plan of a built engine (design §4.3): stable across builds, and it moves with kernel pins."""
import dataclasses
import json
import os
import textwrap
import time

import pytest
from holon.api import Lock, Manifest
from holon.core import plans
from holon.core.plugins import Installed, PluginSet, discover
from holon_tools import locks

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
    plan = locks.plan_of(engine)
    assert locks.plan_of(tiny_engine()).plan_id == plan.plan_id
    assert plan.model_id.startswith(f"{tiny_checkpoint.name}@") and plan.fingerprint_class.startswith("cpu/")
    assert {plans.CORE_UNIT, "models.qwen3", "std.ops", "std.mixers.gqa", "std.weights.hf"} <= plan.units.keys()
    assert plan.units["std.ops"].sem and plan.units["std.blocks"].sem is None      # only numeric units have oracles
    assert plan.sites.keys() == engine.ctx.kernels.keys()
    assert locks.uncovered(plan) == ()
    assert plan.engine["max_step_tokens"] == 16 and plan.engine["dtype"] == "float32"
    assert "state_memory" not in plan.engine and "kernels" not in plan.engine        # per-host budget; sites


def test_plan_moves_with_engine_settings(tiny_engine):
    changed = locks.diff(locks.plan_of(tiny_engine()), locks.plan_of(tiny_engine(max_step_tokens=8)))
    assert changed == {"engine:max_step_tokens": (16, 8)}


def test_plan_moves_with_a_kernel_pin(tiny_engine, with_fake_kernel):
    default = locks.plan_of(tiny_engine(with_fake_kernel))
    pinned = locks.plan_of(tiny_engine(with_fake_kernel, kernels={"rmsnorm": FAKE}))
    assert FAKE not in default.units                # loaded only as a candidate: not code that runs
    changed = locks.diff(default, pinned)
    assert pinned.plan_id != default.plan_id and changed.pop(f"unit:{FAKE}") == (None, pinned.units[FAKE])
    assert changed and all(key.startswith("site:") and key.endswith("/rmsnorm@1") for key in changed)
    assert {new.impl for _, new in changed.values()} == {FAKE}


def test_lock_overrides_rebuild_the_same_plan(tiny_engine, with_fake_kernel):
    pinned = locks.plan_of(tiny_engine(with_fake_kernel, kernels={"rmsnorm": FAKE}))
    lock = plans.loads(locks.dumps(Lock(pinned)))
    rebuilt = locks.plan_of(tiny_engine(with_fake_kernel, **plans.lock_overrides(lock)))
    assert rebuilt.plan_id == pinned.plan_id


def test_identity_remembers_weight_digests(tiny_checkpoint, tmp_path, monkeypatch):
    from holon_std.weights.hf import source

    monkeypatch.setattr(source, "_CACHE", tmp_path / "cache")
    copy = tmp_path / "ckpt"
    copy.mkdir()
    for f in tiny_checkpoint.glob("*.*"):
        (copy / f.name).write_bytes(f.read_bytes())
    before = sorted(p.name for p in copy.iterdir())
    first = source.HFSafetensors(copy).identity()
    assert sorted(p.name for p in copy.iterdir()) == before                     # the checkpoint is never written
    (cache,) = (tmp_path / "cache").iterdir()
    known = json.loads(cache.read_text())
    assert set(known) == {"config.json", "model.safetensors"}
    # a restart trusts the cache while the file is unchanged, and rehashes once it is replaced, even with its size
    # and mtime kept (ctime moves)
    known["model.safetensors"][-1] = "0" * 64
    cache.write_text(json.dumps(known))
    assert source.HFSafetensors(copy).identity() != first
    weights = copy / "model.safetensors"
    st = weights.stat()
    while weights.stat().st_ctime_ns == st.st_ctime_ns:     # some filesystems keep ctime in coarse ticks
        time.sleep(0.05)
        weights.write_bytes(weights.read_bytes())
        os.utime(weights, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert source.HFSafetensors(copy).identity() == first


def test_weight_changes_change_the_model_id(tiny_engine, tiny_checkpoint, tmp_path):
    copy = tmp_path / tiny_checkpoint.name
    copy.mkdir()
    for f in tiny_checkpoint.iterdir():
        data = f.read_bytes()
        (copy / f.name).write_bytes(data[:-1] + bytes([data[-1] ^ 1]) if f.suffix == ".safetensors" else data)
    a, b = locks.plan_of(tiny_engine()), locks.plan_of(tiny_engine(model=str(copy)))
    assert a.model_id != b.model_id and dataclasses.replace(b, model_id=a.model_id).plan_id == a.plan_id


def test_tier_a_source_plans_as_gguf_with_its_test_only_pin(tiny_engine, tiny_checkpoint, tmp_path):
    """The judge's Tier A source (a GGUF read densely) keeps the GGUF kind, and its plan pins it: CI refuses any
    tracked lock pinning a `testing.*` plugin (scripts/ci/check_locks.py)."""
    import gguf
    from holon_testing.gguf_files import write

    file = tmp_path / "tiny.gguf"
    write(tiny_checkpoint, file, gguf.GGMLQuantizationType.Q8_0)
    plan = locks.plan_of(tiny_engine(model=str(file), source="gguf_dense"))
    assert plan.engine["source"] == "gguf" and "testing.gguf_dense" in plan.units


def test_a_drafters_checkpoint_is_planned_by_identity(tiny_checkpoint, cpu_flags, tmp_path):
    # The DFlash2 checkpoint the drafter opens (BuildCtx.weight_source) is part of the plan as its identity, under the
    # option naming it: another draft is another plan; the host path stays a deployment input, out of the lock.
    from holon.core.assembly import open_source
    from holon.core.engine import Engine, EngineConfig
    from holon_testing import dflash_files

    drafts = [dflash_files.save(tmp_path / f"draft{i}", 64, 512, (0, 1), mask=7, seed=i) for i in range(2)]
    engine = Engine(EngineConfig(model=str(tiny_checkpoint), **cpu_flags, max_draft_tokens=7,
                                 plugin_options={"spec.k": "7", "dflash.path": str(drafts[0])},
                                 slots={"drafter": "drafters.dflash", "draft_policy": "std.spec.fixed_k",
                                        "verifier": "std.spec.rejection"}))
    plan = locks.plan_of(engine)
    identity = open_source(engine.plugins, str(drafts[0])).identity()
    assert plan.sources == {"dflash.path": identity} and plan.engine["plugin_options"] == {"spec.k": "7"}
    text = locks.dumps(Lock(plan))
    assert str(tmp_path) not in text and plans.loads(text).plan.plan_id == plan.plan_id
    other = dataclasses.replace(plan, sources={"dflash.path": open_source(engine.plugins, str(drafts[1])).identity()})
    assert other.plan_id != plan.plan_id and "source:dflash.path" in locks.diff(plan, other)
