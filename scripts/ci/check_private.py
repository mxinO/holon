"""No private infrastructure details in files that would be published.

Checks every tracked or trackable (non-ignored) file — recordings' safetensors metadata included — for absolute user
paths, machine details (CPU models, memory generations, driver versions: records name the GPU and nothing else of
the host), plus the host/cluster/user patterns listed one regex per line in the untracked `.private-patterns` file,
so the names themselves never enter the repo.
"""
from __future__ import annotations

import json
import re
import struct
import subprocess

from _common import ROOT, fail

GENERIC = [r"/home/[\w.-]+", r"/Users/[\w.-]+", r"/scratch[\w.-]*/"]
MACHINE = [r"\b(?:EPYC|Xeon|Threadripper|Ryzen)\b", r"\bCore\(TM\)", r"\bZen ?[1-6]\b", r"\bDDR[3-6]\b",
           r"\bdriver[ _:=\"]{1,4}\d{3}\.\d+"]
# Upstream files (public, pinned by SHA256SUMS or a commit) can match the patterns; our patches on them are checked.
VENDORED = ("plugins/holon_plugins/kernels/ggml_cuda/vendor/", "spikes/s2_gguf/third_party/")
SELF = "scripts/ci/check_private.py"         # its patterns name what they look for


def patterns() -> re.Pattern[str]:
    local = ROOT / ".private-patterns"
    extra = [] if not local.is_file() else [
        line.strip() for line in local.read_text().splitlines() if line.strip() and not line.startswith("#")]
    return re.compile("|".join(f"(?:{p})" for p in GENERIC + MACHINE + extra), re.IGNORECASE)


def _files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=ROOT, capture_output=True,
                         check=True).stdout
    return [f for f in out.decode().split("\0")
            if f and (ROOT / f).is_file() and not f.startswith(VENDORED) and f != SELF]


def _safetensors_meta(path) -> str:
    """The header's metadata, one entry per line (the tensors are numbers)."""
    with open(path, "rb") as f:
        header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
    return "\n".join(f"{k}: {v}" for k, v in header.get("__metadata__", {}).items())


def main() -> None:
    pat = patterns()
    msgs = []
    for name in _files():
        if pat.search(name):
            msgs.append(f"{name}: private name in the file name")
        try:
            text = _safetensors_meta(ROOT / name) if name.endswith(".safetensors") else (ROOT / name).read_text()
        except (UnicodeDecodeError, OSError, ValueError):
            continue
        msgs += [f"{name}:{i}: {m.group(0)!r}" for i, line in enumerate(text.splitlines(), 1)
                 if (m := pat.search(line))]
    fail(msgs[:40] + ([f"… {len(msgs) - 40} more"] if len(msgs) > 40 else []), "no private infrastructure details")


if __name__ == "__main__":
    main()
