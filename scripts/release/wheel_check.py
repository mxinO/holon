"""The GPU acceptance's install checks (docs/dev/v1-pip.md D12, steps 0-2 and 4), run by a pip install's Python on
the card, in the wheel-based image:

    <venv>/bin/python scripts/release/wheel_check.py <the wheels' directory>

- no CUDA toolkit: no `nvcc` on PATH, no CUDA_HOME / CUDA_PATH, no /usr/local/cuda*, and Holon finds none;
- origins: every installed file of every `holonlm*` distribution has the hash the wheels' RECORDs give it;
- locks: this card's class has locks, every newest one current (its units' sources are the installed ones);
- load: every module shipped for this SM loads through its plugin's loader (the five extensions, and FlashInfer's
  modules from the AOT directory), and no cache gains a file (`prebuilt/`, FlashInfer's JIT directory, Holon's).

Prints one line per check and a JSON summary last; exits 1 on any problem.
"""
from __future__ import annotations

import csv
import glob
import importlib
import importlib.metadata
import io
import json
import os
import pathlib
import shutil
import sys
import zipfile

EXTENSIONS = ("ggml_cuda", "expert_io", "promote", "state_copy", "cpu_task")


def toolkit() -> tuple[list[str], str]:
    from holon_plugins import host

    nvcc = shutil.which("nvcc")
    msgs = ([f"nvcc on PATH: {nvcc}"] * bool(nvcc)
            + [f"{v} is set" for v in ("CUDA_HOME", "CUDA_PATH") if os.environ.get(v)]
            + [f"{p} exists" for p in glob.glob("/usr/local/cuda*")]
            + ["Holon finds a CUDA toolkit"] * host.toolkit())
    return msgs, "present" if msgs else "absent"


def origins(wheels: pathlib.Path) -> tuple[list[str], str]:
    """Each holonlm* wheel's RECORD against its installed distribution's (same version, every hashed file equal)."""
    msgs, count = [], 0
    for whl in sorted(wheels.glob("holonlm*.whl")):
        name, version = whl.name.split("-")[:2]
        dist = importlib.metadata.distribution(name.replace("_", "-"))
        if dist.version != version:
            msgs.append(f"{whl.name}: installed {dist.metadata['Name']} {dist.version}")
            continue
        with zipfile.ZipFile(whl) as z:
            record = next(n for n in z.namelist() if n.endswith(".dist-info/RECORD"))
            want = {row[0]: row[1] for row in csv.reader(io.StringIO(z.read(record).decode())) if row and row[1]}
        installed = {str(f): f"{f.hash.mode}={f.hash.value}" for f in dist.files or () if f.hash}
        msgs += [f"{whl.name}: {path} differs or is missing" for path, h in want.items() if installed.get(path) != h]
        count += len(want)
    if not count:
        msgs.append(f"no holonlm* wheels in {wheels}")
    return msgs, f"{count} files as the wheels' RECORDs hash them"


def locks() -> tuple[list[str], str]:
    import torch
    from holon.core import plans, platform, plugins

    cls = platform.fingerprint(torch.device("cuda")).cls
    current = plans.current_srcs(plugins.discover())
    dirs = sorted({p.parent for p in (plans.TREE / "plans").rglob("*.lock")})
    heads = [plans.read(plans.numbered(d)[-1]) for d in dirs]
    mine = [lock for lock in heads if lock.plan.fingerprint_class == cls]
    msgs = [f"{lock.plan.model_id}: unit {u} outdated" for lock in mine for u, pin in lock.plan.units.items()
            if current.get(u) != pin.src]
    return msgs + [f"no locks for {cls}"] * (not mine), f"{len(mine)} newest locks of {cls}"


def files(*roots: pathlib.Path) -> set[str]:
    return {str(p) for r in roots if r.is_dir() for p in r.rglob("*") if p.is_file()}


def load() -> tuple[list[str], str]:
    import holon_plugins
    import torch
    from flashinfer.jit import env
    from holon_plugins.kernels.flashinfer import aot, prebuild

    major, minor = torch.cuda.get_device_capability()
    sm = 10 * major + minor
    xdg = pathlib.Path(os.environ.get("XDG_CACHE_HOME", pathlib.Path.home() / ".cache"), "holon")
    watched = (pathlib.Path(holon_plugins.__path__[0]) / "kernels", env.FLASHINFER_JIT_DIR, xdg)
    before, loaded, msgs = files(*watched), [], []
    for k in EXTENSIONS:
        ext = importlib.import_module(f"holon_plugins.kernels.{k}.extension")
        ext.load(sm)
        loaded.append(k)
    aot.bind(torch.device("cuda"))
    for spec in prebuild.specs(sm):
        spec.build_and_load()
        loaded.append(f"flashinfer/{spec.name}")
    msgs += [f"a cache gained {p}" for p in sorted(files(*watched) - before)]
    return msgs, f"{len(loaded)} modules for sm {sm}: {', '.join(loaded[:len(EXTENSIONS)])} and FlashInfer's"


def main() -> int:
    wheels = pathlib.Path(sys.argv[1])
    problems, summary = [], {}
    for name, check in (("toolkit", toolkit), ("origins", lambda: origins(wheels)), ("locks", locks), ("load", load)):
        msgs, what = check()
        problems += msgs
        summary[name] = what
        print(f"{'ok  ' if not msgs else 'FAIL'} {name}: {what}" + "".join(f"\n  {m}" for m in msgs), flush=True)
    print(json.dumps(summary | {"problems": problems}))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
