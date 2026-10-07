"""Reading side of the lock history: unit source hashes and engine overrides (writing is tested with the tools)."""
import shutil

import torch
from holon.api import Binding, Lock, Plan
from holon.core import plans


def test_src_hash_is_content_only(tmp_path):
    unit = tmp_path / "unit"
    (unit / "sub").mkdir(parents=True)
    (unit / "a.py").write_text("x = 1\n")
    (unit / "sub" / "b.py").write_text("y = 2\n")
    first = plans.src_hash(unit)
    assert first.startswith("sha256:")
    # caches, build products and hidden files are not source
    for junk in ("__pycache__/a.cpython-312.pyc", "build/lib/a.py", "x.egg-info/PKG-INFO", ".hidden/f", "k.so",
                 "sub/c.pyc"):
        (unit / junk).parent.mkdir(parents=True, exist_ok=True)
        (unit / junk).write_text("junk")
    assert plans.src_hash(unit) == first
    copy = tmp_path / "installed" / "unit"                 # an installed copy elsewhere hashes the same
    shutil.copytree(unit, copy)
    assert plans.src_hash(copy) == first
    (unit / "sub" / "b.py").write_text("y = 3\n")
    assert plans.src_hash(unit) != first
    (unit / "sub" / "b.py").write_text("y = 2\n")
    assert plans.src_hash(unit) == first
    (unit / "sub" / "b.py").rename(unit / "b.py")           # paths count, not only contents
    assert plans.src_hash(unit) != first


def test_lock_overrides_rebuild_settings_and_pin_each_op():
    plan = Plan("org/m@1", "rtx-a5000/cu130-torch2.11", "sha256:r", "c", "none", {},
                {"kv/attention.paged_decode@1": Binding("kernels.fast"), "s/rmsnorm@1": Binding("std.ops")},
                engine={"dtype": "bfloat16", "max_step_tokens": 512})
    assert plans.lock_overrides(Lock(plan)) == {
        "dtype": torch.bfloat16, "max_step_tokens": 512,
        "kernels": {"attention.paged_decode": "kernels.fast", "rmsnorm": "std.ops"}}
