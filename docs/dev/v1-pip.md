# v1 — `pip install holonlm`: wheels with prebuilt CUDA extensions

Status: **design rev 1 (2026-11-03), for the three-reviewer panel. Nothing is built or published.**

**The owner's goal:** `pip install holonlm`, then `holon serve <model>`, just works on a supported consumer GPU, with
no git clone, no uv and no local CUDA toolkit. "Easy to use is important."

**Decided before this design:** gguf-py's reader is vendored into Holon at llama.cpp `27c7a769` (D7). Publishing is a
public action: the owner approves the first upload. This milestone stops at a local-wheel install (TestPyPI on the
owner's go, Q6).

## 1. Findings (2026-11-03, on the development host; no GPU run)

- **F1. A wheel install hashes like the checkout; it does not find the locks.**
  - `uv build --wheel` of core, std and plugins from a clean `main` (4b5baf77), unpacked: all 54 units, the core
    included, have the same `src2` hash as in the checkout. All 27 newest locks are current against the unpacked
    wheels. Scheme 2 hashes relative paths and content only. Hatchling ships every tracked file under the package
    directories (`csrc/`, `vendor/`, `plugin.toml`, profiles). The hash leaves out `.so`, `__pycache__`, `*.md` and
    tests, so binaries added to a unit directory do not move it.
  - **The blocker:** `plans.TREE` is `CORE_DIR.parents[1]`. In a wheel install that is `<venv>/lib/python3.12`, so
    `holon serve` finds no `plans/` (every checkpoint serves with defaults) and no `uv.lock` (`uv_lock = "none"`). D6
    fixes it.
- **F2. PyPI's torch 2.11.0 is the cu130 build.** Its `manylinux_2_28_x86_64` cp312 wheel and the PyTorch cu130
  index's `2.11.0+cu130` wheel have the same RECORD, file for file (12,482 entries), except `METADATA`. That includes
  `torch/version.py`, so `torch.__version__` is `2.11.0+cu130` in both. Both depend on `cuda-toolkit==13.0.2`. The
  comparison read both RECORDs by HTTP range requests. So pip needs no extra index, and the toolchain class
  (`cu130-torch2.11`) and every extension cache key are the same under either wheel.
- **F3. Sizes.**
  - **ggml_cuda, built from `main`'s sources for each SM** (deflated is the wheel's compression):

    | SM | module | device code | deflated | with `--compress-mode=size`: module / deflated |
    |---|---|---|---|---|
    | 86 | 144 MB | 120 MB | 49 MB | 53 / 31 MB |
    | 89 | 145 MB | 120 MB | 49 MB | 53 / 31 MB |
    | 120a | 179 MB | 154 MB | 60 MB | 60 / 38 MB |

    nvcc's default mode stores this fatbin uncompressed.
  - **`size` mode leaves the SASS unchanged.** On one mmq instance per SM (sm_86, sm_120a), `cuobjdump -sass`
    differs only by the "compressed" tag.
  - **FlashInfer's modules, deflated:** 8.3 MB for sm_86's 8 and 22 MB for sm_120f's 14 (measured); sm_89's 14,
    ≈ 22 MB est.
  - **The rest:** the other extensions are under 1 MB per SM together. The locks and `defaults.toml` are 1.5 MB
    deflated; the snapshot JSONs, which serving does not need, are 15 MB more.
- **F4. What runs without a toolkit, from the code.**
  - **Holon's own extensions** (ggml_cuda, expert_io, promote, state_copy, cpu_task) link libtorch and
    `libcudart.so.13`. torch loads the CUDA runtime first, from its pip CUDA packages. They need nothing else.
  - **FlashInfer** loads a module with no compiler when its AOT path holds it:
    `jit_env.FLASHINFER_AOT_DIR/<name>/<name>.so`, read when the module is fetched. Otherwise its JIT runs nvcc.
    Its modules link `libcudart.so.13` and TVM FFI, not libtorch.
  - **CuTe DSL** (FlashInfer's norms) compiles with its bundled toolchain.
  - **Triton 3.6** compiles its `cuda_utils` and per-kernel launchers at first use with `gcc`/`clang` against
    `Python.h`. **So a pip host needs a C compiler and Python's headers** (Q5).
  - **Not verified:** no Holon serve has run without a toolkit. Every GPU run so far was inside a CUDA devel
    container. Step 0 checks it first.
- **F5. The locks' nvcc is CUDA 13.0.0 (V13.0.48)**, the release environment's. The `Dockerfile` builds with
  13.0.3, so its image's binaries are not the ones the locks were verified with (noted for the Docker item, Q9).
- **F6. What Holon uses from gguf-py.**
  - **std:** the type tables in `constants.py` (`GGML_QUANT_SIZES`, `GGMLQuantizationType`) in `weights.gguf`, and
    `quants.dequantize` in `quants.py`. qsa uses `dequantize` at load (GLM's Q6_K `k_b`/`v_b`); moe uses it in its
    reference. `quants.py` imports `constants` and `lazy`. Those three upstream files (≈ 8,200 lines, numpy only)
    cover std.
  - **judge, testing and the tests:** `GGUFReader`, `GGUFWriter`, `Keys`, `quants.quantize`, that is, the whole
    package.
- **F7. Names.** `holonlm`, `holonlm-core`, `holonlm-std`, `holonlm-plugins`, `holonlm-judge` and `holonlm-tools` are
  free on PyPI, and nothing reserves them before a first upload. `uv.lock` has one git source (gguf) and one extra
  index (torch). The runtime closure is 91 packages, 3 of them Holon's.
- **F8. A pre-existing bug, found while measuring.** `python -m holon_plugins.kernels.ggml_cuda.prebuild --sm 89 120`
  builds the second SM's module as `holon_ggml_cuda_v1.so`. torch's extension loader versions a name already
  loaded in the process, and `load` then never finds the module. The same applies to every plugin `prebuild` module
  that loops over SMs in one process. `holon prebuild` is not affected: it runs one process per (plugin, SM). Fixed
  in step 1, with its test.

## 2. Distributions (D1)

| distribution | content | wheel | depends on |
|---|---|---|---|
| `holonlm` | **what users install**: no code; the locks (D6) | `py3-none-any` | every dist below `==X.Y.Z`, plus the runtime closure pinned exactly (D9) |
| `holonlm-core` (renamed from `holonlm`) | `holon` (ABI and core), the `holon` command | `py3-none-any` | torch `==2.11.0`, numpy, msgspec, pyzmq, nvidia-ml-py (as today) |
| `holonlm-std` | `holon_std`, with the vendored gguf-py subset (D7) | `py3-none-any` | as today, without `gguf` |
| `holonlm-plugins` | `holon_plugins`, sources included (`csrc/`, `vendor/`: the cache keys hash them; the fallback build needs them) | `py3-none-any` | as today |
| `holonlm-cuda-sm86`, `-sm89`, `-sm120` | the prebuilt native modules for one SM (D2), installed into the kernel plugins' `prebuilt/` directories | `cp312-cp312-manylinux_2_28_x86_64` | `holonlm-plugins==X.Y.Z`, torch `==2.11.0` |

- **Not published:** `holonlm-judge`, `holonlm-tools` and `holonlm-testing`. The judge and the tools work on the
  repository's goldens, snapshots and lock history; testing is never installed with Holon. The workspace root
  (`holon-workspace`) stays the development project. There are **no extras**: everything a lock can select is in the
  base install, as release D1 left it.
- **Why the core is renamed:** `pip install holonlm` has to bring everything. Today `holonlm` is the core alone, which
  std and plugins depend on, so it cannot also be the umbrella. Nothing is published yet, so the rename costs nothing
  outside the repository (Q1). In the repository it touches the members' dependencies, the root `[tool.uv.sources]`,
  `plans.RUNTIME`, `check_manifest_deps.DISTS`, a test fixture and the docs.
- **Why one overlay per SM:** each stays under PyPI's 100 MB default (§7), so neither PyPI nor TestPyPI needs a
  size-limit request. An overlay holds only `.so` files at paths no other distribution writes, so the three install
  side by side, and uninstalling one removes only its own files. The plugins look only in their own directory (D5),
  so merging the overlays into one later, after a limit request (Q3), changes no plugin code.
- **One version:** every published distribution is `X.Y.Z`, and they pin each other exactly.
  - `scripts/bump_version.py` and `check_version.py` stay the mechanism. The build script reads the version from
    `core/pyproject.toml` for the distributions it generates (`holonlm`, the overlays), so they cannot disagree.
  - The plan's "one version file" (a `VERSION` file read through hatchling's dynamic version) stays deferred. It
    buys nothing for pip, and dynamic workspace versions change what `uv.lock` records, and with it `plans.uv_lock`.
- **Plugin discovery:** unchanged. pip installs std's and plugins' `holon.plugins` entry points as uv does. The
  locks use a new entry-point group, `holon.locks` (D6).

## 3. Native code (D2–D5)

- **D2. What is compiled at build time:**

  | plugin | kind | in the wheels |
  |---|---|---|
  | `kernels.ggml_cuda` | torch extension, one per SM (86, 89, 120a) | prebuilt, fatbin compressed (`--compress-mode=size`) |
  | `kernels.expert_io`, `.promote`, `.state_copy`, `.cpu_task` | torch extensions, one per SM | prebuilt |
  | `kernels.flashinfer` | FlashInfer JIT modules: `holon prebuild`'s set, 8 on sm_86, 14 on sm_89 and sm_120 | prebuilt, per SM, used as FlashInfer's AOT directory |
  | `kernels.ggml_cpu` | torch extension per CPU tier, C/C++ only | **not prebuilt**: an opt-in no lock uses; `holon prebuild` builds it with gcc/g++, no toolkit (Q8) |
  | Triton kernels (`triton_*`, `gdn_triton`, `hc_triton`, `ple_triton`, `qsa_triton`, `fla`) | Triton JIT | none: compiled and cached at first use (F4) |
  | CuTe DSL norms (through FlashInfer) | in-memory JIT | none (bundled compiler) |

  - **`--compress-mode=size`** joins ggml_cuda's `NVCC_FLAGS` for every build, local prebuilds included, so there
    is one binary per key. It leaves the SASS unchanged (F3) and cuts ggml_cuda's deflated size by 37% (sm_86,
    sm_89) and 36% (sm_120). The cost is decompression when the module loads, measured in step 2. If it adds more
    than 1 s to a start, it is dropped: the overlays fit without it (§7).
  - The flag changes ggml_cuda's cache key, so repository installs run `holon prebuild` once more.
- **D3. SMs:** 86, 89 and 120: RTX 3090; RTX 4090; RTX 5080, RTX 5090 and RTX PRO 6000 Blackwell. These are the
  locks' classes and the SMs the code already targets (`ARCHS`). Any other SM finds no binary, and the plugin reports
  it with the existing messages (GGUF refused at startup, as today).
- **D4. ABI.**
  - torch is pinned `==2.11.0`, the core's pin today. The specifier matches PyPI's `2.11.0` and the index's
    `2.11.0+cu130`, which are the same build (F2), so there is no extra index URL.
  - **The overlays are built in `nvidia/cuda:13.0.0-devel-rockylinux8`, pinned by digest.** It has glibc 2.28,
    the same nvcc as the locks (F5), gcc-toolset-13 (the major version of the lock environment's Ubuntu 24.04
    gcc), and uv-managed CPython 3.12 for the headers.
  - `auditwheel show` must find `manylinux_2_28` with only torch's libraries and `libcudart.so.13` external. These
    are excluded, never bundled: torch's own dependencies provide them. The wheel is then tagged
    `cp312-cp312-manylinux_2_28_x86_64` (`wheel tags`).
  - **No new compatibility check.** Each extension's existing cache key (sources, flags, SM, CUDA, CPython tag, torch
    version) names its shipped directory. A different torch or Python does not find the directory and falls back
    (D5). FlashInfer's directory is named by `flashinfer.__version__` and the SM (D5), so the module set always
    matches the installed FlashInfer. Its modules do not link libtorch.
- **D5. Lookup and fallback, in the plugins only.**
  - **Holon's extensions:** `module_path` looks in `HERE / "prebuilt" / key(sm)`, then in the user cache as today.
    `build()` returns at once when the shipped module exists, so `holon prebuild` in a pip install builds only what
    is missing. FlashInfer's prebuild module likewise skips an SM whose AOT directory is shipped.
  - **FlashInfer:** the plugin, when it first binds on a device, sets `flashinfer.jit.env.FLASHINFER_AOT_DIR` to
    `HERE / "prebuilt" / f"{flashinfer.__version__}-sm{sm}"` if that directory exists. FlashInfer reads the attribute
    on each module fetch; its `jit/env.py` says its attributes exist to be overridden. An engine process serves one
    device, so one SM; this is documented.
  - **When nothing matches:** the user cache, then today's `NotBuilt` error. It now says what the install ships
    ("prebuilt for sm 86/89/120, torch 2.11.0+cu130, CPython 3.12; found …") and that `holon prebuild` needs CUDA
    13's nvcc. FlashInfer looks in the AOT directory, then in its JIT cache (where a repository install's
    `holon prebuild` puts the modules, as today). A module in neither starts FlashInfer's JIT, which needs nvcc.
    That failure is re-raised with the same message.

## 4. Locks, data and third-party code (D6–D9)

- **D6. Locks in the wheel.**
  - `holonlm` ships the package `holon_locks`: `plans/**/<n>.lock`, `plans/defaults.toml` and `uv.lock`. The
    snapshot JSONs stay out (judge data; 15 MB).
  - It declares the entry point `[project.entry-points."holon.locks"] holon = "holon_locks"`.
  - **Core** (`plans.py`, ≈ 10 lines): `TREE` is the checkout when `CORE_DIR.parents[1] / "plans"` is a directory,
    else the directory of the one installed `holon.locks` entry point. More than one is an error, for now. Then
    `TREE / "plans"` and `TREE / "uv.lock"` work unchanged, and `plan_of`'s `uv_lock` equals the repository's hash.
    That is honest because the closure is pinned exactly (D9).
  - The group is general: any distribution may ship lock histories. Merging several roots is additive and left for
    later.
  - **`src` in a pip install:** F1 shows it matches. Three things keep it matching:
    - wheels are built only from `git archive` of a commit;
    - the overlays add only `.so` files, which the hash leaves out;
    - the build's CPU check (D10) installs every wheel in a fresh venv and fails unless every newest lock is current.
  - **Data already inside the wheels:** std's routing profiles (`std/holon_std/tiered/profiles`, part of `std.tiered`).
    Holon ships no tokenizer or template data; both come from the checkpoint. The Qwen pre-tokenizer regex is code.
- **D7. gguf-py vendored.** `std/holon_std/vendor/gguf/`, laid out like the other vendored trees:
  - **Upstream files, unmodified:** `constants.py`, `quants.py` and `lazy.py`, plus upstream's `LICENSE` (MIT, the
    ggml authors), at llama.cpp `27c7a769`.
  - **Our files:** an empty `__init__.py`; `vendor.sh` (`--check` joins `run_all.sh`); `SHA256SUMS`. There is no
    patch: the files' relative imports work as they are.
  - **std's three import sites** become `holon_std.vendor.gguf.constants` / `.quants`. std drops `gguf>=0.17`.
  - **Where:** a library of the std distribution, outside every unit directory, because three units use it and units
    never import each other. So, like gguf-py today, it is in no unit's `src`. A re-vendor is a dependency bump
    followed by a propagation, as a `uv.lock` change is.
  - **judge, testing and the tests** keep gguf-py from the git source (F6: they need the reader and writer). Those
    distributions are not published, so PyPI's rule does not apply. `vendor.sh --check` asserts that its commit is
    the root's `gguf` source rev, so the oracle and std read the same tables.
  - **The MTP pack hint is rewritten.** `models.qwen4_exp`'s `source_hint` (`deploy.py`) and the README's
    Qwen3.8-Flash-Next section tell users how to make the MTP pack with `uv run` and the workspace's `gguf`. A pip
    user has neither. The new recipe runs Strata's tools in a venv of their own:
    `pip install "gguf @ git+https://github.com/ggml-org/llama.cpp@27c7a769…#subdirectory=gguf-py"`. pip accepts a
    git URL on the command line; PyPI forbids one only in a package's metadata.
  - **`THIRD_PARTY_NOTICES.md`:** an entry under llama.cpp / ggml (commit `27c7a769`, MIT, the paths).
  - **`check_provenance`** learns non-plugin vendored trees: every tracked `vendor/` directory must have a `LICENSE`
    and be named in the notices.
- **D8. Notices inside the wheels.**
  - **Every published wheel** carries `LICENSE`, `NOTICE` and `THIRD_PARTY_NOTICES.md` in `dist-info/licenses`
    (`license-files`). `core/`, `std/` and `plugins/` get tracked symlinks to the root files. The generated
    distributions copy them.
  - **Vendored trees** ship with their upstream `LICENSE`: ggml_cuda, ggml_cpu, fla and the new gguf.
  - **`THIRD_PARTY_NOTICES.md` gains "In the binary wheels"**, the compiled code the overlays redistribute, with the
    license texts not yet in the file:
    - llama.cpp / ggml (MIT);
    - FlashInfer 0.6.12 (Apache-2.0) and the headers its modules compile: CUTLASS (BSD-3-Clause), CCCL (Apache-2.0
      with LLVM exception), spdlog (MIT), TVM FFI (Apache-2.0);
    - the PyTorch and pybind11 headers inlined into the torch extensions (BSD-3-Clause).
  - **Device code** compiled with the CUDA toolkit is redistributable under its EULA's terms for object code, as for
    every CUDA wheel on PyPI.
  - **The build script** fails if a wheel lacks the three files.
  - **FlashInfer itself** is a plain dependency (`flashinfer-python==0.6.12` from PyPI). Neither
    `flashinfer-jit-cache` nor `flashinfer-cubin` is used: Holon's modules come from its own overlays.
- **D9. Sources and pins.**
  - **torch** comes from PyPI (F2). The workspace keeps the cu130 index in `uv.lock`; the files are identical.
  - **gguf:** see D7. Nothing else in the runtime closure comes from outside PyPI (F7).
  - **`holonlm` pins every third-party package of the runtime closure to `uv.lock`'s version** (87 once std drops
    gguf; torch as `==2.11.0`).
    - The build script generates them with `plans.uv_lock`'s closure walk, so `uv.lock` stays the only source.
    - A pip install then resolves the environment the locks were verified in. tokenizers, jinja2 (chat templates),
      xgrammar, nvidia-cutlass-dsl and the CUDA libraries all shape outputs.
    - It also keeps the shipped `uv.lock` hash honest.
  - The cost: those versions are fixed in the venv, so the README recommends a dedicated one (`python3.12 -m venv`,
    or `pipx install holonlm`) (Q2).

## 5. Build and CI (D10–D11)

- **D10. One script, `scripts/release/build_wheels.py`**, used both locally (`docker`, output under
  `local/tmp/pip/`) and in CI. It works from `git archive <ref>`:
  1. `uv build --wheel` for core, std and plugins.
  2. `holonlm` from a generated `pyproject.toml`: the pins, `holon_locks`, the entry point and the license files.
  3. Per SM, inside the build image (D4):
     - `uv sync --frozen` (runtime only);
     - `holon prebuild --sm N`, with `XDG_CACHE_HOME` and `FLASHINFER_WORKSPACE_BASE` pointed at a staging directory;
     - the `.so` files copied into the overlay layout;
     - the overlay built from a generated `pyproject.toml`, checked by `auditwheel show`, and retagged.
  4. **The CPU check:**
     - each wheel is under 100 MB and has its license files;
     - a fresh `python3.12 -m venv` installs every wheel, with the third-party pins from PyPI;
     - `holon --help` runs;
     - every newest lock is current against the installed units (`plans.current_srcs`, as `holon models` computes
       it).
- **D11. CI: `.github/workflows/wheels.yml`**, run on `workflow_dispatch` and on `v*` tags.
  - **`pure`** (ubuntu-24.04): steps 1, 2 and 4.
  - **`cuda`:** a matrix over sm 86 / 89 / 120 in the build image, step 3, on the 4-vCPU runner. Estimate: 20–40
    min per SM, from the measured build times scaled by cores. It frees runner disk first; the image and the
    environment are ≈ 15 GB est.
  - Artifacts: the wheels.
  - **`publish`:** a manual job behind a GitHub environment (`testpypi`, `pypi`) whose required reviewer is the
    owner. It uses trusted publishing (OIDC, no tokens) and uploads exactly the artifacts the GPU matrix tested.
  - **GPU tests are not in GitHub** (no GPU runners). `scripts/release/wheel_check.sh` (D12) runs them on each card
    class against a run's artifacts, and the results are evidence records.

## 6. Acceptance (D12–D13)

- **D12. GPU matrix: RTX 3090, RTX 4090, RTX 5080, RTX 5090, each on the desktop its class's locks were propagated
  on.** Each class gets one fresh container from the public `ubuntu:24.04`, with `python3.12-venv`, `python3.12-dev`,
  `gcc` and `g++` only. It is checked to have no `nvcc`, no `CUDA_HOME` and no `/usr/local/cuda*`. In it:
  1. `python3.12 -m venv`, then `pip install --find-links <artifacts> holonlm`.
  2. `holon models`: this card's locks are listed as current.
  3. **Smoke:**
     - `holon serve` the class's Qwen3-8B Q4_K_M lock;
     - one chat request answers;
     - the served report's `diff` against the lock is empty (`plans.served`).
  4. **Load:** every shipped module loads on the card through its plugin's `load` (ggml_cuda, expert_io, promote,
     state_copy, cpu_task, FlashInfer's modules). This covers the binaries no lock binds today (promote, cpu_task).
  5. **Traces:** each snapshot trace must be byte-identical to the lock's recorded trace. The locks run on each class
     cover the shipped binaries the locks bind:
     - Qwen3-8B Q4_K_M everywhere (ggml_cuda, FlashInfer, Triton);
     - Qwen3.8-27B everywhere (fla; state_copy on the RTX 5080 and RTX 5090);
     - Qwen3-30B-A3B Q4_K_M on the RTX 5080 (expert_io, offload).

     The judge comes from the tag's checkout, installed on top with `--no-deps`. `pip show` confirms the runtime is
     still the wheels'.
  6. **Cold start:** Triton's first compile is within `Policy.start_timeout`.
- **D13. Acceptance (the owner's list):**
  - in a fresh venv, `pip install` from local wheels serves on each card class with no toolkit (D12 1–4);
  - outputs are identical to the repository install (D12 5: the locks were propagated from it);
  - every wheel is under 100 MB (§7);
  - `check_provenance` and the notices are green (D7, D8);
  - the README's install section is rewritten:
    - `pip install holonlm` and `holon serve`;
    - the host needs: Linux x86-64, a driver for CUDA 13.0 (R580 or newer), CPython 3.12 with its headers, gcc/g++;
    - the source install moves below it, for contributors.

## 7. Wheel sizes (deflated, est.)

The ggml_cuda and FlashInfer columns come from F3. Totals add < 1 MB for the other extensions.

| wheel | ggml_cuda | FlashInfer | **total, with `size` mode** | total, without |
|---|---|---|---|---|
| `holonlm-cuda-sm86` | 31 | 8.3 | **≈ 40 MB** | ≈ 58 MB |
| `holonlm-cuda-sm89` | 31 | ≈ 22 | **≈ 54 MB** | ≈ 72 MB |
| `holonlm-cuda-sm120` | 38 | 22 | **≈ 61 MB** | ≈ 83 MB |
| one `holonlm-cuda` (Q3) | | | ≈ 155 MB | ≈ 213 MB |
| `holonlm-plugins` | | | 0.8 MB (measured) | |
| `holonlm-std` | | | 0.36 MB (measured), + ≈ 0.1 MB of gguf | |
| `holonlm-core` | | | 0.15 MB (measured) | |
| `holonlm` | | | ≈ 1.6 MB (the locks, `uv.lock`) | |

- Every wheel is under 100 MB with or without `size` mode. The largest leaves 39 MB of headroom, or 17 MB without.
- If more is needed, FlashInfer's modules can take the same flag (`FLASHINFER_EXTRA_CUDAFLAGS` at prebuild).
- A user downloads ≈ 158 MB of Holon, beside torch's 531 MB wheel and its dependencies. The development venv,
  for scale, is 6.1 GB installed, 3.1 GB of it NVIDIA's CUDA packages.

## 8. Core lines, ABI and locks (D14)

- **Core:**
  - `plans.py`: `TREE` from the checkout or `holon.locks` (≈ 10 lines);
  - `RUNTIME` renamed (`holonlm-core`).
  - Below 5,300 of 6,000 lines.
- **ABI:** no `holon.api` change. The new entry-point group `holon.locks` is additive and goes into design §4.1.
  The `[build]` module CLI is unchanged.
- **std:** three imports and the vendored package.
- **plugins:** six loaders (shipped directory first), the FlashInfer AOT directory, ggml_cuda's flag, the MTP pack
  hint.
- **Every lock is outdated by this change** (the core, `std.weights.gguf`, `std.mixers.qsa`, `std.moe`, the kernel
  plugins; the rename also moves `uv_lock`).
  - **One propagation round** on the four classes closes it: 2-run traces byte-identical, no L2 expected, because no
    numerics change.
  - This is the "next relock" the plan anticipated.
  - The wheels are then built from the propagated commit, so `holon_locks` holds locks that pin the wheels' code.

## 9. Steps

0. **No-toolkit probe (first, ≈ ½ day, one RTX 5080):** today's environment, with the FlashInfer AOT override
   prototyped under `local/`, in an Ubuntu 24.04 container without a toolkit. `holon serve` Qwen3-8B Q4_K_M and
   bf16, then Qwen3.8-27B. This settles F4's unknowns (Triton's headers, CuTe DSL, FlashInfer loading) before any
   build code.
1. **Code:** D7 (gguf), the rename, D6 (core), D5 (loaders, FlashInfer), D2's flag, D8 (symlinks, notices,
   provenance), D10's script; CPU tests.
2. **Local build:** the wheels in docker on the development host; sizes, the load-time cost of `size` mode, the CPU
   check.
3. **Propagation** of the 27 locks (repository install, four classes), committed.
4. **Merge;** then the `wheels` workflow, dispatched on `main`, builds the artifacts.
5. **D12 on the artifacts**, four classes; evidence records.
6. **README;** then the owner's go for TestPyPI and PyPI (Q6).

## 10. Risks

- **A trace moves between the wheel and the repository install.** The binaries come from the same nvcc. The host
  compiler preprocesses device code with its own libstdc++ headers, though: gcc-toolset-13 on Rocky 8 against the
  locks' Ubuntu gcc 13. If a trace moves, the overlays' compiler moves to the locks' exact gcc. Failing that, the
  propagation runs on the overlay binaries, so the wheels become the reference.
- **Step 0 finds another runtime compile** (e.g. CuTe DSL wanting `ptxas`). Then its pip package becomes a pin, or a
  requirement is documented.
- **GitHub runner limits:** disk and the 6 h job limit. Fallback: build on the development host with the same
  script, and upload the artifacts from a CI job that only publishes.
- **Python other than 3.12:** pip reports no matching distribution. The README says so (Q4).
- **Names taken before the first upload** (F7; Q6).
- **The RTX 3090 27B trace differs by desktop** (plan §10, open): D12 runs on the desktop that propagated.

## 11. Owner questions

1. **Rename the core distribution to `holonlm-core`**, so that `holonlm` is the one-line install (D1)? Default: yes.
2. **Pin the whole runtime closure exactly in `holonlm`** (87 pins; D9)? This gives reproducible locks and an honest
   `uv_lock`, but fixes numpy, fastapi and the others in the venv. The alternative pins only what shapes outputs
   (torch, triton, flashinfer, cutlass-dsl, xgrammar, transformers, tokenizers, jinja2) and puts floors on the rest.
   Default: all.
3. **Three per-SM overlays**, each under 100 MB with no PyPI request? Or one `holonlm-cuda` (≈ 155 MB) after a
   file-size-limit request to PyPI (and TestPyPI), whose wait is out of our hands? Default: per SM.
4. **Python 3.12 only** in v1 (cp312 binaries, Ubuntu 24.04's Python)? 3.13 would need cp313 overlays and a trace
   check. Default: 3.12, 3.13 later.
5. **A C compiler and Python's headers as host requirements** (Triton's launcher)? Default: yes, documented; the
   alternative is no small change.
6. **Publishing:**
   - trusted publishing from GitHub Actions, with you as the required reviewer of the `testpypi` and `pypi`
     environments;
   - a TestPyPI upload counts as public: your go there too?
   - register PyPI "pending publishers" now? They do not reserve the names.
7. **Leave `holonlm-judge`, `holonlm-tools` and `holonlm-testing` unpublished?** Default: yes.
8. **`kernels.ggml_cpu` not prebuilt** (an opt-in no lock uses; `holon prebuild` builds it with gcc/g++)? Default:
   yes.
9. **Docker:** rebase the image on the wheels (a runtime base, no devel toolkit, nvcc 13.0.0 binaries; F5) in a
   later step? Default: after this milestone.
