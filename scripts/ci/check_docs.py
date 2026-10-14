"""The user and contributor docs name only what exists (release design D8): every repository path in their code spans
and links (anchors included), and every `holon …` / `python -m holon_… …` command with the flags it gives. Paths in
code are relative to the root, links to the document; placeholders (`<…>`) are skipped. Run in the project
environment: commands are checked against their `--help`."""
from __future__ import annotations

import re
import subprocess
import sys

from _common import ROOT, fail

DOCS = ["README.md", "docs/adding-a-model.md"]
TOP = {p.name for p in ROOT.iterdir() if not p.name.startswith(".")}
CODE = re.compile(r"```[^\n]*\n(.*?)```|`([^`\n]+)`", re.DOTALL)
LINK = re.compile(r"\]\(([^)\s]+)\)")
COMMAND = re.compile(r"(?:^|[\s`(])(holon|python -m holon_\w+) ([a-z][\w-]*)([^|;&#\n]*)")
FLAG = re.compile(r"(?<![\w-])--[a-z][\w-]*")


def slug(heading: str) -> str:
    """GitHub's anchor of a heading."""
    return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


def anchors(path) -> set[str]:
    return {slug(m) for m in re.findall(r"^#+ (.+)$", path.read_text(), re.MULTILINE)}


_help: dict[tuple[str, str], str | None] = {}


def help_text(program: str, sub: str) -> str | None:
    """`<program> <sub> --help`, or None if the subcommand does not exist."""
    if (program, sub) not in _help:
        module = "holon" if program == "holon" else program.split()[-1]
        out = subprocess.run([sys.executable, "-m", module, sub, "--help"], cwd=ROOT, capture_output=True, text=True)
        _help[program, sub] = out.stdout if out.returncode == 0 else None
    return _help[program, sub]


msgs = []
for doc in DOCS:
    path = ROOT / doc
    text = path.read_text()
    for target in LINK.findall(text):
        if re.match(r"[a-z]+:", target):                         # http(s), mailto
            continue
        file, _, anchor = target.partition("#")
        dest = (path.parent / file).resolve() if file else path
        if not dest.exists():
            msgs.append(f"{doc}: link to a missing {target}")
        elif anchor and dest.suffix == ".md" and anchor not in anchors(dest):
            msgs.append(f"{doc}: link to a missing anchor {target}")
    for block, span in CODE.findall(text):
        code = (block or span).replace("\\\n", " ")
        for token in re.split(r"[\s\"'()\[\],=]+", code):
            token = token.rstrip(".:;")
            if token.split("/")[0] in TOP and not re.search("[<…]", token) and not any(ROOT.glob(token)):
                msgs.append(f"{doc}: missing path {token}")
        for program, sub, rest in COMMAND.findall(code):
            usage = help_text(program, sub)
            if usage is None:
                msgs.append(f"{doc}: `{program} {sub}` is not a command")
                continue
            msgs += [f"{doc}: `{program} {sub}` has no flag {f}" for f in FLAG.findall(rest)
                     if not re.search(rf"(?<![\w-]){re.escape(f)}(?![\w-])", usage)]
fail(msgs, "docs name existing paths and commands")
