"""No private infrastructure details in files that would be published.

Checks every tracked or trackable (non-ignored) file for absolute user paths, plus the host/cluster/user patterns
listed one regex per line in the untracked `.private-patterns` file, so the names themselves never enter the repo.
"""
from __future__ import annotations

import re
import subprocess

from _common import ROOT, fail

GENERIC = [r"/home/[\w.-]+", r"/Users/[\w.-]+", r"/scratch[\w.-]*/"]


def _patterns() -> re.Pattern[str]:
    local = ROOT / ".private-patterns"
    extra = [] if not local.is_file() else [
        line.strip() for line in local.read_text().splitlines() if line.strip() and not line.startswith("#")]
    return re.compile("|".join(f"(?:{p})" for p in GENERIC + extra), re.IGNORECASE)


def _files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=ROOT, capture_output=True,
                         check=True).stdout
    return [f for f in out.decode().split("\0") if f and (ROOT / f).is_file()]


pat = _patterns()
msgs = []
for name in _files():
    if pat.search(name):
        msgs.append(f"{name}: private name in the file name")
    try:
        text = (ROOT / name).read_text()
    except (UnicodeDecodeError, OSError):
        continue
    msgs += [f"{name}:{i}: {m.group(0)!r}" for i, line in enumerate(text.splitlines(), 1) if (m := pat.search(line))]
fail(msgs[:40] + ([f"… {len(msgs) - 40} more"] if len(msgs) > 40 else []), "no private infrastructure details")
