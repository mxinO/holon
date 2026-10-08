"""Third-party code is declared (AGENTS.md, "Third-party code"): the repository is Apache-2.0 (LICENSE, NOTICE); every
path THIRD_PARTY_NOTICES.md names exists; every code file whose header marks it as taken from elsewhere is named
there; every plugin's `[provenance]` has an allowed license, its source in the notices, its files present and named
there, and vendored directories carry their upstream LICENSE."""
from __future__ import annotations

import hashlib
import re
import subprocess
import tomllib

from _common import ROOT, fail

APACHE_SHA256 = "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"   # apache.org LICENSE-2.0.txt
ALLOWED = {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause"}
MARKER = re.compile(r"\b(adapted|copied) from\b", re.IGNORECASE)
CODE = re.compile(r"\.(py|cu|cuh|cpp|c|h|sh)$")

msgs = []
license_file, notices_file = ROOT / "LICENSE", ROOT / "THIRD_PARTY_NOTICES.md"
if not license_file.is_file() or hashlib.sha256(license_file.read_bytes()).hexdigest() != APACHE_SHA256:
    msgs.append("LICENSE is not the Apache License 2.0 text")
if not (ROOT / "NOTICE").is_file():
    msgs.append("NOTICE is missing")
notices = notices_file.read_text() if notices_file.is_file() else ""
named = set(re.findall(r"`((?:plugins|std|core|judge|tools|testing|bench|spikes)/[^`\s]+)`", notices))
msgs += [f"THIRD_PARTY_NOTICES.md names {p}, which does not exist" for p in sorted(named) if not (ROOT / p).exists()]

tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
vendored: list[str] = []
for manifest in sorted(p for p in tracked if p.endswith("plugin.toml")):
    prov = tomllib.loads((ROOT / manifest).read_text()).get("provenance")
    if not prov:
        continue
    base = manifest.removesuffix("plugin.toml")
    if prov.get("license") not in ALLOWED:
        msgs.append(f"{manifest}: provenance license {prov.get('license')!r} not in {sorted(ALLOWED)}")
    for f in prov.get("files", []):
        path = base + f
        if not (ROOT / path).exists():
            msgs.append(f"{manifest}: provenance file {f} does not exist")
        elif f.endswith("/"):
            vendored.append(path)
            if not (ROOT / path / "LICENSE").is_file():
                msgs.append(f"{manifest}: vendored {f} has no LICENSE")
        if path not in named:
            msgs.append(f"{manifest}: provenance file {f} is not in THIRD_PARTY_NOTICES.md")
    if prov.get("files") and prov["source"] not in notices:
        msgs.append(f"{manifest}: provenance source {prov['source']} is not in THIRD_PARTY_NOTICES.md")

for path in tracked:
    if CODE.search(path) and not any(path.startswith(v) for v in vendored) and path not in named \
            and MARKER.search((ROOT / path).read_text(errors="replace")):
        msgs.append(f"{path}: marked as third-party code but not in THIRD_PARTY_NOTICES.md")
fail(msgs, "third-party code declared (license, notices, provenance)")
