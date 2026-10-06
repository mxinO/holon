"""Invariant 6: no torch.cuda outside holon.core.platform and kernel plugins."""
from __future__ import annotations

from _common import CORE, ROOT, fail, py_files

ALLOWED = [CORE / "core" / "platform", ROOT / "plugins" / "kernels"]
msgs = []
for p in py_files(CORE, ROOT / "std", ROOT / "plugins"):
    if any(a in p.parents for a in ALLOWED) or "tests" in p.parts:
        continue
    for i, line in enumerate(p.read_text().splitlines(), 1):
        if "torch.cuda" in line:
            msgs.append(f"{p.relative_to(ROOT)}:{i}")
fail(msgs, "no torch.cuda outside platform/kernels")
