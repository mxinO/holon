"""Bump Holon's version everywhere at once: `python scripts/bump_version.py 0.1.2`.

The seven distributions release together, so they share one version: each `pyproject.toml`'s `version`, and
`holon.__version__`; then `uv lock` refreshes the lock file. `scripts/ci/check_version.py` fails if they ever disagree.
(v1 moves to one `VERSION` file read at build time, with `holon.__version__` from the installed metadata — plan §10.)
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DISTS = ("core", "std", "plugins", "judge", "testing", "tools")
PYPROJECTS = ["pyproject.toml", *(f"{d}/pyproject.toml" for d in DISTS)]
CORE = "core/holon/__init__.py"


def main() -> None:
    if len(sys.argv) != 2 or not re.fullmatch(r"\d+\.\d+\.\d+", sys.argv[1]):
        sys.exit("usage: python scripts/bump_version.py X.Y.Z")
    new = sys.argv[1]
    for name in PYPROJECTS:
        path = ROOT / name
        text, n = re.subn(r'(?m)^version = "[^"]+"$', f'version = "{new}"', path.read_text(), count=1)
        if n != 1:
            sys.exit(f"{name}: no `version = \"…\"` line")
        path.write_text(text)
    path = ROOT / CORE
    text, n = re.subn(r'(?m)^__version__ = "[^"]+"$', f'__version__ = "{new}"', path.read_text(), count=1)
    if n != 1:
        sys.exit(f"{CORE}: no __version__ line")
    path.write_text(text)
    subprocess.run(["uv", "lock"], cwd=ROOT, check=True)
    print(f"version {new}: {len(PYPROJECTS)} pyprojects, {CORE}, uv.lock")


if __name__ == "__main__":
    main()
