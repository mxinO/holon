"""Invariant 8: core (holon.api + holon.core) stays within 6000 code lines."""
from __future__ import annotations

from _common import CORE, fail, py_files

CAP = 6000
total = 0
for p in py_files(CORE):
    in_doc = False
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if line.startswith(('"""', "'''")):
            if not (len(line) > 3 and line.endswith(('"""', "'''"))):
                in_doc = not in_doc
            continue
        if in_doc or not line or line.startswith("#"):
            continue
        total += 1
print(f"     core code lines: {total} / {CAP}")
fail([f"core has {total} code lines > {CAP}"] if total > CAP else [], "core LOC cap")
