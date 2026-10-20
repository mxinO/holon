"""No private infrastructure details in commit messages: every subject and body in `base..HEAD` (default
`origin/main..HEAD`) against `check_private.py`'s patterns, theirs and the untracked `.private-patterns`, which also
lists the user branch prefix (`<user>/`), so branch names stay out of history: merges take a neutral message. The
prefix lives there, not here, for the same reason. Usage: `check_commit_messages.py [base-ref]`; skipped when the
base is unknown (a shallow clone)."""
from __future__ import annotations

import subprocess
import sys

from _common import ROOT, fail
from check_private import patterns


def main() -> None:
    base = sys.argv[1] if len(sys.argv) > 1 else "origin/main"
    if subprocess.run(["git", "rev-parse", "--verify", "-q", base], cwd=ROOT, capture_output=True).returncode:
        print(f"skip commit messages: base {base!r} not found")
        return
    log = subprocess.run(["git", "log", "--format=%H%x00%B%x01", f"{base}..HEAD"], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout
    pat, msgs = patterns(), []
    for entry in filter(str.strip, log.split("\x01")):
        sha, _, body = entry.strip().partition("\x00")
        msgs += [f"{sha[:12]}:{i}: {m.group(0)!r}" for i, line in enumerate(body.splitlines(), 1)
                 if (m := pat.search(line))]
    fail(msgs[:40] + ([f"… {len(msgs) - 40} more"] if len(msgs) > 40 else []),
         f"no private details in commit messages ({base}..HEAD)")


if __name__ == "__main__":
    main()
