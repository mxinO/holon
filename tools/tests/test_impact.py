"""`impact`: changed units from names, paths and git ranges, and the locks that pin them."""
import subprocess

import pytest
from holon.api import Binding, Lock, Plan, UnitPin
from holon.core import plans
from holon_tools import impact, locks
from holon_tools.__main__ import main


def _git(repo, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", *args],
                   cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    """A repository with units `holon` (core/), `std.a`, `kern.b` and `outside` (std/, holding the other two)."""
    for d in ("core", "std/a", "std/b"):
        (tmp_path / d).mkdir(parents=True)
        (tmp_path / d / "x.py").write_text("x = 1\n")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "one")
    (tmp_path / "std/b/x.py").write_text("x = 2\n")
    _git(tmp_path, "commit", "-q", "-am", "two")
    return tmp_path


def _dirs(repo):
    return {"holon": repo / "core", "std.a": repo / "std/a", "kern.b": repo / "std/b", "outside": repo / "std"}


def test_names_paths_and_ranges(repo):
    dirs = _dirs(repo)
    assert impact.changed_units(["std.a"], dirs, repo) == {"std.a"}
    # A unit's src covers its whole directory: a change in a nested unit is one of the enclosing unit too.
    assert impact.changed_units([str(repo / "std/b/x.py")], dirs, repo) == {"kern.b", "outside"}
    assert impact.changed_units(["HEAD~1..HEAD"], dirs, repo) == {"kern.b", "outside"}
    assert impact.changed_units([str(repo / "core")], dirs, repo) == {"holon"}
    assert impact.changed_units([str(repo / ".git")], dirs, repo) == set()               # exists, in no unit
    assert impact.changed_units(["HEAD.."], dirs, repo) == set()
    # a range compares `src` pins, read from git as from a checkout: comments and docs touch no unit
    (repo / "std/a/x.py").write_text("x = 1  # why\n")
    (repo / "std/a/README.md").write_text("doc\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "three")
    assert impact.changed_units(["HEAD~1..HEAD"], dirs, repo) == set()
    assert impact.changed_units(["HEAD~2..HEAD"], dirs, repo) == {"kern.b", "outside"}
    assert impact._src(repo, "HEAD", repo / "std") == plans.src_hash(repo / "std")
    with pytest.raises(ValueError, match="unknown revision"):
        impact.changed_units(["nope..HEAD"], dirs, repo)
    with pytest.raises(ValueError, match="not an installed unit"):
        impact.changed_units(["std.opz"], dirs, repo)


def _lock(model, units):
    return Lock(Plan(model, "rtx-a5000/cu130-torch2.11", "sha256:r", "c", "none",
                     {u: UnitPin("1.0.0", "src") for u in units}, {"kv/attention.paged_decode@1": Binding("kern.b")}))


def test_impact_lists_the_newest_locks_pinning_a_unit(tmp_path, capsys):
    locks.append(tmp_path, _lock("org/a@1", ["holon", "std.a"]))
    locks.append(tmp_path, _lock("org/b@2", ["holon", "kern.b"]))
    hits = impact.impact(locks.heads(tmp_path), {"kern.b", "unrelated"})
    assert [(lock.plan.model_id, units) for lock, units in hits] == [("org/b@2", ["kern.b"])]
    assert len(impact.impact(locks.heads(tmp_path), {"holon"})) == 2
    # the command line, on the units installed here
    assert main(["impact", "std.ops", "--plans", str(tmp_path)]) == 0
    assert capsys.readouterr().out.splitlines() == ["changed units: std.ops"]
