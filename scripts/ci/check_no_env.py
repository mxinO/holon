"""Invariant 5: no environment-variable configuration in core or std."""
from __future__ import annotations

import re

from _common import CORE, ROOT, fail, py_files

PAT = re.compile(r"os\.environ|os\.getenv|getenv\(")
msgs = [
    f"{p.relative_to(ROOT)}:{i}"
    for p in py_files(CORE, ROOT / "std")
    for i, line in enumerate(p.read_text().splitlines(), 1)
    if PAT.search(line)
]
fail(msgs, "no env-var configuration in core/std")
