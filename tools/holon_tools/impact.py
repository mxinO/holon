"""`impact`: which locks pin a changed unit (design §4.6).

A change is named as a unit, a path (every unit whose directory holds it) or a git range `A..B` (the units whose
source tree hash — their `src` pin — differs between A and B). The core is the unit `holon`.
"""
from __future__ import annotations

import pathlib
import subprocess
from collections.abc import Iterable, Mapping

from holon.api import Lock


def changed_units(changes: Iterable[str], dirs: Mapping[str, pathlib.Path], repo: pathlib.Path) -> set[str]:
    """Units touched by `changes`, given each unit's source directory. A unit's `src` covers its whole directory,
    so a path or range inside a nested unit touches the enclosing one too."""
    dirs, repo = {u: d.resolve() for u, d in dirs.items()}, repo.resolve()
    out: set[str] = set()
    for change in changes:
        if change in dirs:
            out.add(change)
        elif pathlib.Path(change).exists():
            path = pathlib.Path(change).resolve()
            out |= {u for u, d in dirs.items() if path == d or d in path.parents}
        elif ".." in change:
            a, b = (rev or "HEAD" for rev in change.split("..", 1))
            if bad := [r for r in (a, b) if _git(repo, "rev-parse", "--verify", "--quiet", f"{r}^{{commit}}") is None]:
                raise ValueError(f"unknown revision {bad[0]!r}")
            out |= {u for u, d in dirs.items() if _tree(repo, a, d) != _tree(repo, b, d)}
        else:
            raise ValueError(f"{change!r} is not an installed unit, an existing path or a git range A..B")
    return out


def impact(heads: Iterable[Lock], units: set[str]) -> list[tuple[Lock, list[str]]]:
    """The newest locks pinning any of `units`, each with the units it pins."""
    return [(lock, hit) for lock in heads if (hit := sorted(units & lock.plan.units.keys()))]


def repo_root(cwd: pathlib.Path | None = None) -> pathlib.Path:
    """The git work tree holding `cwd` (default: the current directory)."""
    cwd = cwd or pathlib.Path.cwd()
    return pathlib.Path(_git(cwd, "rev-parse", "--show-toplevel") or cwd).resolve()


def _tree(repo: pathlib.Path, rev: str, d: pathlib.Path) -> str | None:
    """The tree hash of directory `d` at `rev`; None where it does not exist (or lies outside `repo`)."""
    if not d.is_relative_to(repo):
        return None
    return _git(repo, "rev-parse", "--verify", "--quiet", f"{rev}:{d.relative_to(repo).as_posix()}")


def _git(cwd: pathlib.Path, *args: str) -> str | None:
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None
