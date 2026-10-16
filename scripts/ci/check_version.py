"""One version: every distribution's `pyproject.toml` and `holon.__version__` agree (`scripts/bump_version.py` sets
them together)."""
from __future__ import annotations

import re

from _common import ROOT, fail

FILES = {f: r'(?m)^version = "([^"]+)"$' for f in
         ["pyproject.toml", *(f"{d}/pyproject.toml" for d in ("core", "std", "plugins", "judge", "testing", "tools"))]}
FILES["core/holon/__init__.py"] = r'(?m)^__version__ = "([^"]+)"$'
found = {f: (m.group(1) if (m := re.search(p, (ROOT / f).read_text())) else None) for f, p in FILES.items()}
fail([] if len(set(found.values())) == 1 and None not in found.values()
     else [f"{f}: {v}" for f, v in found.items()], "one version")
