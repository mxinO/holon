"""`manifest-deps` (release design D1): every distribution declares what it imports. Each third-party top-level
module a distribution's package imports, and each module its plugin manifests list (`[dependencies] python`), must
be provided (`importlib.metadata.packages_distributions`) by a distribution in its declared dependency closure — so
an install of the distribution alone has it. Run in the project environment (`uv sync --group dev`)."""
from __future__ import annotations

import ast
import importlib.metadata as md
import re
import sys
import tomllib

from _common import ROOT, fail, py_files

DISTS = {"holonlm": "core/holon", "holonlm-std": "std/holon_std", "holonlm-plugins": "plugins/holon_plugins",
         "holonlm-testing": "testing/holon_testing", "holonlm-judge": "judge/holon_judge",
         "holonlm-tools": "tools/holon_tools"}
OWN = {path.rsplit("/", 1)[1] for path in DISTS.values()}


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def closure(dist: str) -> set[str]:
    """`dist` and every distribution its requirements reach (extras left out)."""
    seen, todo = set(), [dist]
    while todo:
        name = norm(todo.pop())
        if name in seen:
            continue
        seen.add(name)
        try:
            requires = md.requires(name) or []
        except md.PackageNotFoundError:
            continue
        todo += [re.match(r"[A-Za-z0-9._-]+", r).group() for r in requires if "extra ==" not in r]
    return seen


def imports(path) -> set[str]:
    out = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            out |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.add(node.module.split(".")[0])
    return out - set(sys.stdlib_module_names) - OWN


providers = {mod: {norm(d) for d in dists} for mod, dists in md.packages_distributions().items()}
msgs = []
for dist, package in DISTS.items():
    have = closure(dist)
    needs = {(mod, str(p.relative_to(ROOT))) for p in py_files(ROOT / package) for mod in imports(p)}
    for manifest in sorted((ROOT / package).rglob("plugin.toml")):
        listed = tomllib.loads(manifest.read_text()).get("dependencies", {}).get("python", [])
        needs |= {(mod, str(manifest.relative_to(ROOT))) for mod in listed}
    for mod, where in sorted(needs):
        if not providers.get(mod, set()) & have:
            msgs.append(f"{where}: {mod} ({sorted(providers.get(mod, [])) or 'not installed'}) is not in "
                        f"{dist}'s declared dependencies")
fail(msgs, "manifest-deps: imports covered by each distribution's declared dependencies")
