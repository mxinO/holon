"""Build and check Holon's wheels (docs/dev/v1-pip.md D10): one script for a local build and for CI.

    build_wheels.py pure      --out DIR [--ref REF]   holonlm-core/-std/-plugins and the holonlm umbrella, from
                                                      `git archive REF` (any host)
    build_wheels.py overlay   --sm N --out DIR        holonlm-cuda-smN: inside overlay.Dockerfile's image, at /holon
    build_wheels.py reference --sm N --out DIR        the same modules built in the locks' image (reference.Dockerfile)
    build_wheels.py sass      --sm N DIR              the SASS gate: the overlay's modules against the reference's
    build_wheels.py check     DIR                     the CPU check: the wheels' contents, then a fresh venv's install

`in_image.sh` runs the image steps from `git archive` of a commit, unpacked at /holon (never a bind-mounted checkout:
whatever a binary embeds then names /holon, D4). The privacy scan of the wheels is `scripts/ci/check_private.py
--wheels DIR`. The script imports only the standard library at the top, so its hidden steps (`_modules`, `_installed`)
run under another environment's Python: the build's venv, a fresh install's.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
SMS = (86, 89, 120)
ARCH_LIST = {86: "8.6", 89: "8.9", 120: "12.0f"}       # FlashInfer's arch list entries (its prebuild's)
LIMIT = 100 << 20                       # PyPI's default per-file limit
EXTENSIONS = ("ggml_cuda", "expert_io", "promote", "state_copy", "cpu_task")     # prebuilt per SM (D2)
EXTERNAL = {"libc10.so", "libc10_cuda.so", "libtorch_cpu.so", "libtorch_python.so", "libtorch.so",
            "libtorch_cuda.so", "libcudart.so.13",                  # torch and its CUDA runtime provide them (D4)
            "libcuda.so.1"}                                          # the driver itself (any GPU host has it)
SYSTEM = {"libc.so.6", "libm.so.6", "libstdc++.so.6", "libgcc_s.so.1", "libdl.so.2", "librt.so.1", "libpthread.so.0",
          "ld-linux-x86-64.so.2"}                                    # manylinux's allowed system libraries
LICENSES = ("LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md")
OWN = {"holonlm-core", "holonlm-std", "holonlm-plugins"}
TAG = "cp312-cp312-manylinux_2_28_x86_64"
NVCC = "V13.0.48"                       # the locks' nvcc (CUDA 13.0.0, F5)
TOOLS = {"hatchling": "1.33.0", "wheel": "0.48.0", "auditwheel": "6.8.2"}         # the build's own tools, pinned
CLASSIFIERS = ["Programming Language :: Python :: 3.12", "Operating System :: POSIX :: Linux",
               "Environment :: GPU :: NVIDIA CUDA :: 13"]
URLS = '[project.urls]\nRepository = "https://github.com/mxinO/holon"\n'


def sh(*cmd: str | pathlib.Path, **kw) -> str:
    return subprocess.run([str(c) for c in cmd], check=True, text=True, capture_output=True, **kw).stdout


def version(root: pathlib.Path = ROOT) -> str:
    return tomllib.loads((root / "core" / "pyproject.toml").read_text())["project"]["version"]


def project(where: pathlib.Path, name: str, description: str, deps: list[str], wheel: str, root: pathlib.Path,
            readme: str = "README.component.md", extra: str = "") -> None:
    """A generated hatchling project at `where` (outside any checkout: the root .gitignore's `*.so` must never apply),
    with the license files and its PyPI page (`readme`, from scripts/release) copied in."""
    where.mkdir(parents=True, exist_ok=True)
    for f in LICENSES:
        shutil.copy(root / f, where / f)
    shutil.copy(root / "scripts" / "release" / readme, where / "README.md")
    (where / "pyproject.toml").write_text(
        f'[project]\nname = "{name}"\nversion = "{version(root)}"\nlicense = "Apache-2.0"\n'
        f"license-files = {json.dumps(list(LICENSES))}\nrequires-python = \">=3.12,<3.13\"\n"
        f'readme = "README.md"\nclassifiers = {json.dumps(CLASSIFIERS)}\n'
        f"description = {json.dumps(description)}\ndependencies = {json.dumps(deps, indent=4)}\n{extra}\n{URLS}\n"
        f'[build-system]\nrequires = ["hatchling=={TOOLS["hatchling"]}"]\nbuild-backend = "hatchling.build"\n\n'
        f"[tool.hatch.build]\nignore-vcs = true\n\n[tool.hatch.build.targets.wheel]\n{wheel}\n")


def pins(root: pathlib.Path) -> list[str]:
    """The runtime closure's third-party packages at uv.lock's versions, for Linux x86-64 CPython 3.12 (the only
    platform the overlays exist for): an edge whose marker does not hold there is not followed (pyzmq's PyPy-only
    cffi). One version per name, or the lock forked and the pins would be ambiguous."""
    from packaging.markers import Marker

    env = {"sys_platform": "linux", "platform_system": "Linux", "platform_machine": "x86_64",
           "implementation_name": "cpython", "platform_python_implementation": "CPython", "python_version": "3.12",
           "python_full_version": "3.12.12", "os_name": "posix"}
    by_name = collections.defaultdict(list)
    for p in tomllib.loads((root / "uv.lock").read_text())["package"]:
        by_name[p["name"]].append(p)
    if forked := sorted(n for n, ps in by_name.items() if len(ps) > 1):
        raise SystemExit(f"uv.lock holds several versions of {forked}: no single pin")
    seen, todo = set(), [(name, ()) for name in sorted(OWN)]
    while todo:
        name, extras = todo.pop()
        if (name, extras) in seen or name not in by_name:
            continue
        seen.add((name, extras))
        p = by_name[name][0]
        edges = p.get("dependencies", []) + [d for e in extras for d in p.get("optional-dependencies", {}).get(e, [])]
        for d in edges:
            if "marker" not in d or Marker(d["marker"]).evaluate(env | {"extra": ""}):
                todo.append((d["name"], tuple(d.get("extra", ()))))
    names = sorted({n for n, _ in seen} - OWN)
    return [f"{n}=={by_name[n][0]['version'].split('+')[0] if n == 'torch' else by_name[n][0]['version']}"
            for n in names]


def pure(out: pathlib.Path, ref: str) -> None:
    """The pure wheels and the umbrella, from `git archive ref`: never the working tree's untracked or edited files."""
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as d:
        root = pathlib.Path(d, "tree")
        root.mkdir()
        archive = subprocess.run(["git", "-C", str(ROOT), "archive", "--format=tar", ref], capture_output=True,
                                 check=True).stdout
        subprocess.run(["tar", "-x", "-C", str(root)], input=archive, check=True)
        for member in ("core", "std", "plugins"):
            sh("uv", "build", "--wheel", "-q", "-o", out.resolve(), root / member)
        v = version(root)
        where = pathlib.Path(d, "holonlm")
        locks = where / "holon_locks"
        for f in sorted((root / "plans").rglob("*")):
            if f.is_file() and (f.suffix == ".lock" or f.name == "defaults.toml"):
                (locks / "plans" / f.relative_to(root / "plans")).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(f, locks / "plans" / f.relative_to(root / "plans"))
        shutil.copy(root / "uv.lock", locks / "uv.lock")
        (locks / "__init__.py").write_text('"""The lock histories (`plans/`) and the environment lock (`uv.lock`) of '
                                           'this Holon release, found by `holon.core.plans.TREE`."""\n')
        deps = [f"holonlm-{d}=={v}" for d in ("core", "std", "plugins")]
        deps += [f"holonlm-cuda-sm{sm}=={v}" for sm in SMS] + pins(root)
        project(where, "holonlm", "Holon, an LLM inference engine for one consumer NVIDIA GPU: `holon serve <model>`",
                deps, 'packages = ["holon_locks"]', root, "README.pypi.md",
                '[project.scripts]\nholon = "holon.core.cli:main"\n')
        sh("uv", "build", "--wheel", "-q", "-o", out.resolve(), where)


def _toolchain(image: str) -> dict[str, str]:
    """The compilers this image builds with, asserted and recorded: nvcc is the locks', and gcc is 13 (in the overlay's
    image gcc-toolset-13; otherwise Rocky's gcc 8.5 would be used silently, D4)."""
    if pathlib.Path.cwd() != pathlib.Path("/holon"):
        raise SystemExit("build at /holon, from `git archive` (in_image.sh): binaries embed their build paths")
    nvcc = next(line for line in sh("nvcc", "--version").splitlines() if "release" in line)
    gcc, gxx = (sh(c, "--version").splitlines()[0] for c in ("gcc", "c++"))
    if NVCC not in sh("nvcc", "--version") or not all(re.search(r" 13\.\d+\.\d+", c) for c in (gcc, gxx)):
        raise SystemExit(f"{image}: nvcc {nvcc}, gcc {gcc}, c++ {gxx}: want nvcc {NVCC} and gcc 13")
    return {"commit": os.environ.get("HOLON_COMMIT", "unknown"), "image": os.environ.get("HOLON_IMAGE", "unknown"),
            "nvcc": nvcc, "gcc": gcc, "nvcc_append_flags": os.environ.get("NVCC_APPEND_FLAGS", "")}


def _modules(sm: int, stage: pathlib.Path) -> tuple[dict[str, pathlib.Path], dict[str, str]]:
    """`holon prebuild --sm N` into `stage`, then every module it built, by its overlay path, and the versions the
    modules were built against."""
    env = os.environ | {"XDG_CACHE_HOME": str(stage / "xdg"), "FLASHINFER_WORKSPACE_BASE": str(stage / "fi"),
                        "MAX_JOBS": os.environ.get("MAX_JOBS", "3"), "FLASHINFER_CUDA_ARCH_LIST": ARCH_LIST[sm],
                        "PATH": f"{pathlib.Path('.venv/bin').resolve()}:{os.environ['PATH']}"}   # the venv's ninja
    sh("uv", "sync", "--frozen", "-q")
    subprocess.run([".venv/bin/holon", "prebuild", "--sm", str(sm)], check=True, env=env)
    found = json.loads(sh(".venv/bin/python", __file__, "_modules", "--sm", str(sm), env=env))
    if missing := sorted(p for p, src in found["modules"].items() if not pathlib.Path(src).is_file()):
        raise SystemExit(f"prebuild left these modules unbuilt: {missing}")
    return {p: pathlib.Path(src) for p, src in found["modules"].items()}, found["versions"]


def probe_modules(sm: int) -> None:
    """(`_modules`, in the build's venv.) The modules `holon prebuild --sm N` built here, by the path an overlay ships
    each one at, and the versions they were built against, as JSON."""
    import importlib.metadata

    from holon_plugins.kernels.flashinfer import aot, prebuild

    out = {}
    for k in EXTENSIONS:
        ext = __import__(f"holon_plugins.kernels.{k}.extension", fromlist=["x"])
        out[f"holon_plugins/kernels/{k}/prebuilt/{ext.key(sm)}/{ext.NAME}.so"] = str(ext.module_path(sm))
    for spec in prebuild.specs(sm):
        out[f"holon_plugins/kernels/flashinfer/prebuilt/{aot.directory(sm).name}/{spec.name}/{spec.name}.so"] = str(
            spec.jit_library_path)
    versions = {d: importlib.metadata.version(d) for d in ("torch", "flashinfer-python", "apache-tvm-ffi")}
    print(json.dumps({"modules": out, "versions": versions}))


def overlay(sm: int, out: pathlib.Path) -> None:
    built = _toolchain("overlay")
    with tempfile.TemporaryDirectory(dir="/tmp") as d:
        where = pathlib.Path(d, f"holonlm-cuda-sm{sm}")
        modules, versions = _modules(sm, pathlib.Path("/holon/stage"))
        for rel, src in modules.items():
            (where / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, where / rel)
        (where / "BUILDINFO").write_text(json.dumps(built | versions, indent=1) + "\n")
        project(where, f"holonlm-cuda-sm{sm}", f"Holon's prebuilt CUDA modules for SM {sm} (no toolkit needed)",
                [f"holonlm-plugins=={version()}", "torch==2.11.0"],
                'only-include = ["holon_plugins"]\nartifacts = ["*.so"]\n\n'
                '[tool.hatch.build.targets.wheel.extra-metadata]\n"BUILDINFO" = "BUILDINFO"', ROOT)
        sh("uv", "build", "--wheel", "-q", "-o", where / "dist", where)
        # Retagged for the platform, but still Root-Is-Purelib: true, on purpose. Its modules belong beside the pure
        # plugins wheel's code; where purelib and platlib differ (lib against lib64), platlib would put them elsewhere.
        sh("uvx", "--from", f"wheel=={TOOLS['wheel']}", "wheel", "tags", "--remove", "--python-tag", "cp312",
           "--abi-tag", "cp312", "--platform-tag", "manylinux_2_28_x86_64", next((where / "dist").glob("*.whl")))
        wheel = next((where / "dist").glob(f"*{TAG}.whl"))
        report = sh("uvx", f"auditwheel=={TOOLS['auditwheel']}", "show", wheel)   # glibc symbols: manylinux_2_28
        policy = re.search(r'constrains the platform tag to "manylinux_2_(\d+)_x86_64"', report.replace("\n", " "))
        needed = {n for so in where.rglob("*.so")
                  for n in re.findall(r"\(NEEDED\)\s+Shared library: \[([^]]+)\]", sh("readelf", "-d", so))}
        if not policy or int(policy.group(1)) > 28 or not needed <= EXTERNAL | SYSTEM:
            raise SystemExit(f"not manylinux_2_28, or libraries beyond torch's: {sorted(needed - EXTERNAL - SYSTEM)}"
                             f"\n{report}")
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy(wheel, out)


def reference(sm: int, out: pathlib.Path) -> None:
    """The same modules from the locks' image, as a tree the SASS gate reads (not a wheel), with its BUILDINFO."""
    built = _toolchain("reference")
    tree = out / f"reference-sm{sm}"
    modules, versions = _modules(sm, pathlib.Path("/holon/stage"))
    for rel, src in modules.items():
        (tree / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, tree / rel)
    (tree / "BUILDINFO").write_text(json.dumps(built | versions, indent=1) + "\n")


def anon(text: str) -> str:
    """`text` with each anonymous namespace's mangled name read as `_ANON_`: its length prefix and the characters it
    counts (`38_GLOBAL__N__a473880c_6_api_cu_383b0930`). nvcc's id hashes the translation unit, which the host
    compiler's headers move."""
    out, at = [], 0
    for m in re.finditer(r"(\d+)(?=_GLOBAL__N__)", text):
        if m.start() >= at:
            out.append(text[at:m.start()] + "_ANON_")
            at = m.end() + int(m.group(1))
    return "".join(out) + text[at:]


def sass_hashes(so: pathlib.Path) -> collections.Counter[tuple[str, str]]:
    """(function, hash of its SASS) for every function of every cubin in `so`, streamed from `cuobjdump -sass` with
    the address columns stripped (the module's own load addresses are not code). Only instruction lines count: a
    cubin's header (its arch, and "compressed" in a compressed fatbin) would otherwise join the function before it.
    Anonymous namespaces are read as one name (`anon`)."""
    out: collections.Counter[tuple[str, str]] = collections.Counter()
    name, h = None, None
    proc = subprocess.Popen(["cuobjdump", "-sass", str(so)], stdout=subprocess.PIPE, text=True)
    for line in proc.stdout:
        line = anon(line)
        if (m := re.match(r"\s*Function : (\S+)", line)):
            if name:
                out[(name, h.hexdigest())] += 1
            name, h = m.group(1), hashlib.sha256()
        elif name and line.lstrip().startswith("/*"):      # instructions and encodings, never a cubin's header
            h.update(re.sub(r"/\*[0-9a-f]{4,}\*/", "", line).strip().encode() + b"\n")
    if name:
        out[(name, h.hexdigest())] += 1
    if proc.wait():
        raise SystemExit(f"cuobjdump failed on {so}")
    return out


def _u(data: bytes, at: int, n: int) -> int:
    return int.from_bytes(data[at:at + n], "little")


def data_hashes(so: pathlib.Path) -> collections.Counter[tuple[str, str]]:
    """(section, hash of its bytes) for the device data of every cubin in `so`: initialized constant and global tables
    (`.nv.constant*`, `.nv.global.init`: ggml's IQ grids, …), which `cuobjdump -sass` does not show."""
    out: collections.Counter[tuple[str, str]] = collections.Counter()
    with tempfile.TemporaryDirectory() as d:
        sh("cuobjdump", "-xelf", "all", pathlib.Path(so).resolve(), cwd=d)
        for cubin in pathlib.Path(d).glob("*.cubin"):
            data = cubin.read_bytes()
            shoff, size, count, names = _u(data, 0x28, 8), _u(data, 0x3A, 2), _u(data, 0x3C, 2), _u(data, 0x3E, 2)
            header = [(_u(data, at, 4), _u(data, at + 4, 4), _u(data, at + 24, 8), _u(data, at + 32, 8))
                      for at in range(shoff, shoff + count * size, size)]       # ELF64 section headers
            strings = header[names][2]
            for name_at, kind, offset, length in header:
                name = data[strings + name_at:data.index(b"\0", strings + name_at)].decode()
                if name.startswith((".nv.constant", ".nv.global.init")):
                    body = b"" if kind == 8 else data[offset:offset + length]          # SHT_NOBITS: no bytes
                    out[(anon(name), hashlib.sha256(body).hexdigest())] += 1
    return out


def sass(sm: int, root: pathlib.Path) -> None:
    """Every module the overlay ships has the reference's SASS, function by function, and its device data tables,
    section by section: names and counts 1:1, so a renamed function cannot leave the compared set (D10.5)."""
    wheel = next(root.glob(f"holonlm_cuda_sm{sm}-*.whl"))
    ref = root / f"reference-sm{sm}"
    bad, compared, tables = [], 0, 0
    with zipfile.ZipFile(wheel) as z, tempfile.TemporaryDirectory() as d:
        for rel in (n for n in z.namelist() if n.endswith(".so")):
            if not (ref / rel).is_file():
                bad.append(f"{rel}: not in the reference build (another key: other sources or flags?)")
                continue
            shipped = pathlib.Path(z.extract(rel, d))
            a, b = sass_hashes(shipped), sass_hashes(ref / rel)
            compared += sum(a.values())
            only = sorted({n for n, _ in (a - b) + (b - a)})
            bad += [f"{rel}: {len(only)} functions differ, e.g. {only[:3]}"] * bool(only)
            bad += [f"{rel}: no SASS read"] * (not a)
            x, y = data_hashes(shipped), data_hashes(ref / rel)
            tables += sum(x.values())
            only = sorted({n for n, _ in (x - y) + (y - x)})
            bad += [f"{rel}: {len(only)} data sections differ, e.g. {only[:3]}"] * bool(only)
    print(f"sm {sm}: {compared} functions and {tables} data sections compared", flush=True)
    if bad:
        raise SystemExit("SASS gate failed:\n  " + "\n  ".join(bad))
    print(f"ok   sm {sm}: every shipped module's SASS and device data equal the reference build's", flush=True)


def _contents(root: pathlib.Path) -> list[pathlib.Path]:
    """The wheels' own checks; returns the wheels."""
    wheels = sorted(root.glob("*.whl"))
    msgs, owner = [], {}
    for w in wheels:
        with zipfile.ZipFile(w) as z:
            members = z.namelist()
            wheel_meta = next((z.read(m).decode() for m in members if m.endswith(".dist-info/WHEEL")), "")
            metadata = next((z.read(m).decode() for m in members if m.endswith(".dist-info/METADATA")), "")
        msgs += [f"{w.name}: {w.stat().st_size >> 20} MiB, over PyPI's 100 MB"] * (w.stat().st_size > LIMIT)
        msgs += [f"{w.name}: no {f} in its licenses" for f in LICENSES if not any(m.endswith(f"licenses/{f}")
                                                                                  for m in members)]
        msgs += [f"{w.name}: no PyPI page (long description)"] * ("Description-Content-Type: text/markdown"
                                                                    not in metadata)
        files = [m for m in members if ".dist-info/" not in m]
        if w.name.startswith("holonlm_cuda_sm"):
            msgs += [f"{w.name}: {m} is not a module under a plugin's prebuilt/" for m in files
                     if not re.fullmatch(r"holon_plugins/kernels/\w+/prebuilt/.+\.so", m)]
            msgs += [f"{w.name}: not Root-Is-Purelib (its modules would leave the plugins' directory)"] * (
                "Root-Is-Purelib: true" not in wheel_meta)
            if missing := {*EXTENSIONS, "flashinfer"} - {m.split("/")[2] for m in files}:
                msgs.append(f"{w.name}: no modules of {sorted(missing)}")
            for m in files:
                if m in owner:
                    msgs.append(f"{m}: in {w.name} and {owner[m]}")
                owner[m] = w.name
        elif w.name.startswith("holonlm_plugins"):
            msgs += [f"{w.name}: ships {m}" for m in files if "/prebuilt/" in m]
        elif w.name.startswith(f"holonlm-{version()}-"):
            msgs += [f"{w.name}: no holon_locks/__init__.py"] * ("holon_locks/__init__.py" not in files)
    if len(wheels) != 4 + len(SMS):
        msgs.append(f"{len(wheels)} wheels, not {4 + len(SMS)}: {[w.name for w in wheels]}")
    if msgs:
        raise SystemExit("wheel contents:\n  " + "\n  ".join(msgs))
    print(f"ok   {len(wheels)} wheels: sizes, licenses, PyPI pages, overlay layouts disjoint", flush=True)
    return wheels


def probe_installed() -> None:
    """(`_installed`, in a fresh install's venv.) Its locks are the installed holon_locks and current, its environment
    is uv.lock's, and every plugin finds a shipped module for every SM, as JSON."""
    from holon.core import plans, plugins
    from holon_plugins.kernels.flashinfer import aot, prebuild

    msgs = [f"the locks are not the installed holon_locks: {plans.TREE}"] * (
        plans.INSTALLED is None or plans.TREE != plans.INSTALLED)
    msgs += [f"drift: {d}" for d in plans.drifted(plans.TREE / "uv.lock")]
    current = plans.current_srcs(plugins.discover())
    histories = sorted({p.parent for p in (plans.TREE / "plans").rglob("*.lock")})
    for d in histories:
        pinned = plans.read(plans.numbered(d)[-1]).plan.units
        msgs += [f"outdated: {d.relative_to(plans.TREE)} ({u})" for u, pin in pinned.items()
                 if current.get(u) != pin.src]
    for sm in SMS:
        for k in EXTENSIONS:
            ext = __import__(f"holon_plugins.kernels.{k}.extension", fromlist=["x"])
            if not (p := ext.module_path(sm)).is_relative_to(ext.SHIPPED) or not p.is_file():
                msgs.append(f"{k} sm {sm}: {p} is not a shipped module")
        msgs += [f"flashinfer sm {sm}: no shipped {s.name}" for s in prebuild.specs(sm)
                 if not (aot.directory(sm) / s.name / f"{s.name}.so").is_file()]
    print(json.dumps({"locks": len(histories), "problems": msgs}))


def check(root: pathlib.Path) -> None:
    """The CPU check (D10.6): the wheels' contents, then a fresh `python3.12 -m venv` installs them by path (the
    third-party pins from PyPI, which proves PyPI's torch agrees with them) and finds its locks current, its
    environment uv.lock's (torch included) and every module shipped."""
    wheels = _contents(root)
    with tempfile.TemporaryDirectory(dir=root) as d:
        venv = pathlib.Path(d, "venv")
        sh(sh("uv", "python", "find", "3.12").strip(), "-m", "venv", venv)
        subprocess.run([venv / "bin/pip", "install", "-q", "--no-cache-dir", *wheels], check=True)
        sh(venv / "bin/holon", "--help")
        found = json.loads(sh(venv / "bin/python", pathlib.Path(__file__).resolve(), "_installed", cwd=d,
                              env=os.environ | {"FLASHINFER_CUDA_ARCH_LIST": "8.6"}).splitlines()[-1])
        if found["problems"]:
            raise SystemExit("fresh install:\n  " + "\n  ".join(found["problems"]))
        print(f"ok   a fresh venv installs the wheels: {found['locks']} lock histories current, no drift, every "
              f"module shipped", flush=True)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="step", required=True)
    for step in ("pure", "overlay", "reference"):
        s = sub.add_parser(step)
        s.add_argument("--out", type=pathlib.Path, required=True)
        if step == "pure":
            s.add_argument("--ref", default="HEAD", help="the commit to archive (default: HEAD)")
        else:
            s.add_argument("--sm", type=int, required=True, choices=SMS)
    s = sub.add_parser("sass")
    s.add_argument("--sm", type=int, required=True, choices=SMS)
    s.add_argument("dir", type=pathlib.Path)
    sub.add_parser("check").add_argument("dir", type=pathlib.Path)
    sub.add_parser("_modules").add_argument("--sm", type=int, required=True, choices=SMS)
    sub.add_parser("_installed")
    a = p.parse_args(argv)
    {"pure": lambda: pure(a.out, a.ref), "overlay": lambda: overlay(a.sm, a.out),
     "reference": lambda: reference(a.sm, a.out), "sass": lambda: sass(a.sm, a.dir), "check": lambda: check(a.dir),
     "_modules": lambda: probe_modules(a.sm), "_installed": probe_installed}[a.step]()


if __name__ == "__main__":
    sys.exit(main())
