"""Shared helpers for CI checks (stdlib only)."""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
CORE = ROOT / "core" / "holon"


def py_files(*dirs: pathlib.Path) -> list[pathlib.Path]:
    out: list[pathlib.Path] = []
    for d in dirs:
        if d.exists():
            out.extend(p for p in d.rglob("*.py") if "__pycache__" not in p.parts)
    return sorted(out)


def fail(msgs: list[str], what: str) -> None:
    if msgs:
        print(f"FAIL {what}:")
        for m in msgs:
            print("  " + m)
        sys.exit(1)
    print(f"ok   {what}")
