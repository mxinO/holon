"""The GPU acceptance's install checks (docs/dev/v1-pip.md D12, steps 0, 1 and 4), run by a pip install's Python on
the card, in the wheel-based image:

    <venv>/bin/python scripts/release/wheel_check.py <the wheels' directory>

- no CUDA toolkit: no `nvcc` on PATH, no CUDA_HOME / CUDA_PATH, no /usr/local/cuda*, and Holon finds none;
- origins: every file the `holonlm*` wheels' RECORDs hash is installed, its bytes hashing as the RECORD says;
- load: every module shipped for this SM loads through its plugin's loader from `prebuilt/` (the five extensions
  from their shipped `module_path`, FlashInfer's modules from the AOT directory), and no cache gains a file
  (`prebuilt/`, FlashInfer's JIT directory, Holon's).

That this class's locks are current is `holon models` (and the CPU check's, for every class).

Prints one line per check and a JSON summary last; exits 1 on any problem.
"""
from __future__ import annotations

import base64
import csv
import glob
import hashlib
import importlib
import importlib.metadata
import io
import json
import os
import pathlib
import shutil
import sys
import zipfile

from build_wheels import EXTENSIONS  # beside this script (standard library only at its top)


def toolkit() -> tuple[list[str], str]:
    from holon_plugins import host

    nvcc = shutil.which("nvcc")
    msgs = ([f"nvcc on PATH: {nvcc}"] * bool(nvcc)
            + [f"{v} is set" for v in ("CUDA_HOME", "CUDA_PATH") if os.environ.get(v)]
            + [f"{p} exists" for p in glob.glob("/usr/local/cuda*")]
            + ["Holon finds a CUDA toolkit"] * host.toolkit())
    return msgs, "present" if msgs else "absent"


def digest(path: pathlib.Path) -> str:
    """A RECORD hash: sha256, urlsafe base64 without padding."""
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest()).rstrip(b"=").decode()


def origins(wheels: pathlib.Path) -> tuple[list[str], str]:
    """Each holonlm* wheel's RECORD against the installed files themselves (the same version, every hashed file
    present with those bytes)."""
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
        for rel, h in want.items():
            path = pathlib.Path(dist.locate_file(rel))
            if not path.is_file() or digest(path) != h:
                msgs.append(f"{whl.name}: {rel} {'differs' if path.is_file() else 'is missing'}")
        count += len(want)
    if not count:
        msgs.append(f"no holonlm* wheels in {wheels}")
    return msgs, f"{count} installed files hash as the wheels' RECORDs say"


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
        got, shipped = pathlib.Path(ext.load(sm).__file__), ext.module_path(sm)
        if got != shipped or not shipped.is_relative_to(ext.SHIPPED):
            msgs.append(f"{k}: loaded {got}, not the shipped {shipped}")
        loaded.append(k)
    aot.bind(torch.device("cuda"))
    for spec in prebuild.specs(sm):
        if not (spec.is_aot and spec.aot_path.is_relative_to(aot.directory(sm))):
            msgs.append(f"flashinfer/{spec.name}: not in the AOT directory {aot.directory(sm)}")
        spec.build_and_load()
        loaded.append(f"flashinfer/{spec.name}")
    msgs += [f"a cache gained {p}" for p in sorted(files(*watched) - before)]
    return msgs, f"{len(loaded)} modules for sm {sm} from prebuilt/: {', '.join(EXTENSIONS)} and FlashInfer's"


def main() -> int:
    wheels = pathlib.Path(sys.argv[1])
    problems, summary = [], {}
    for name, check in (("toolkit", toolkit), ("origins", lambda: origins(wheels)), ("load", load)):
        msgs, what = check()
        problems += msgs
        summary[name] = what
        print(f"{'ok  ' if not msgs else 'FAIL'} {name}: {what}" + "".join(f"\n  {m}" for m in msgs), flush=True)
    print(json.dumps(summary | {"problems": problems}))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
