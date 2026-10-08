"""Locks never pin test-only plugins (the `testing.*` distribution: reference kernels, judge fixtures such as GGUF
Tier A's dense source). The tools plan and write locks for any engine — their own tests write CPU locks on the
reference kernels into temporary directories — so the published `plans/` is where this is enforced."""
from __future__ import annotations

import tomllib

from _common import ROOT, fail

msgs = [f"{p.relative_to(ROOT)}: pins {unit}"
        for p in sorted((ROOT / "plans").rglob("*.lock"))
        for unit in tomllib.loads(p.read_text()).get("units", {})
        if unit.startswith("testing.")]
fail(msgs, "no test-only plugins in locks")
