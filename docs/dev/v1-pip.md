# v1 — `pip install holonlm`: wheels with prebuilt CUDA extensions

Status: **design rev 2.1 (2026-11-03): cleared for build.** The panel's findings on rev 1 (1030b0d1) and its notes on
rev 2 (e7dd2210) are answered in §13. Nothing is built or published yet. It builds to the coordinator's defaults for
the owner questions (§12).

**The owner's goal:** `pip install holonlm`, then `holon serve <model>`, just works on a supported consumer GPU, with
no git clone, no uv and no local CUDA toolkit. "Easy to use is important."

**Decided before this design:** gguf-py's reader is vendored into Holon at llama.cpp `27c7a769` (D7). Publishing is a
public action: the owner approves the first upload, and the TestPyPI upload too. This milestone stops at a local-wheel
install.

## 1. Findings (2026-11-03, on the development host; no GPU run)

- **F1. A wheel install hashes like the checkout; it does not find the locks.**
  - `uv build --wheel` of core, std and plugins from a clean `main` (4b5baf77), unpacked: all 54 units, the core
    included, have the same `src2` hash as in the checkout. All 27 newest locks are current against the unpacked
    wheels.
  - Why it matches: scheme 2 hashes relative paths and content only, and the wheels hold exactly the tracked files.
    A counted that for each package: plugins 291/291, std 125/125, core 35/35. The hash leaves out `.so`,
    `__pycache__`, `*.md` and tests.
  - The extension cache keys hash every file under `vendor/` and `csrc/`, so they match too.
  - **The blocker:** `plans.TREE` is `CORE_DIR.parents[1]`. In a wheel install that is `<venv>/lib/python3.12`, so
    `holon serve` finds no `plans/` (every checkpoint serves with defaults) and no `uv.lock` (`uv_lock = "none"`). D6
    fixes it.
- **F2. PyPI's torch 2.11.0 is the cu130 build.**
  - Its `manylinux_2_28_x86_64` cp312 wheel and the PyTorch cu130 index's `2.11.0+cu130` wheel have the same RECORD,
    file for file (12,482 entries), except `METADATA`. Their `METADATA` differ only in the `Version` line;
    `Requires-Dist` is identical (`cuda-toolkit==13.0.2`, …). Both were read by HTTP range requests.
  - `torch/version.py` is among the identical files, so `torch.__version__` is `2.11.0+cu130` under either wheel.
  - So pip needs no extra index, and the toolchain class (`cu130-torch2.11`) and every cache key are the same.
- **F3. Sizes.**
  - **ggml_cuda, built from `main`'s sources for each SM** (deflated is the wheel's compression):

    | SM | module | device code | deflated | with `--compress-mode=size`: module / deflated |
    |---|---|---|---|---|
    | 86 | 144 MB | 120 MB | 49 MB | 53 / 31 MB |
    | 89 | 145 MB | 120 MB | 49 MB | 53 / 31 MB |
    | 120a | 179 MB | 154 MB | 60 MB | 60 / 38 MB |

    nvcc's default mode stores this fatbin uncompressed.
  - **`size` mode on one mmq instance per SM** (sm_86, sm_120a): `cuobjdump -sass` differs only by the "compressed"
    tag. One instance proves little; D10's gate compares every function of every module.
  - **FlashInfer's modules, deflated:** 8.3 MB for sm_86's 8 and 22 MB for sm_120f's 14 (measured); sm_89's 14,
    ≈ 22 MB est.
  - **The rest:** the other extensions are under 1 MB per SM together. The locks and `defaults.toml` are 1.5 MB
    deflated; the snapshot JSONs, which serving does not need, are 15 MB more.
  - **A whole pip install** is ≈ 3.2 GB downloaded (C, from `uv.lock`'s wheel sizes) and ≈ 6 GB installed. pip's
    cache keeps another ≈ 3 GB unless `--no-cache-dir` is passed.
- **F4. What runs without a toolkit, from the code.** A and C checked these independently.
  - **Holon's own extensions** (ggml_cuda, expert_io, promote, state_copy, cpu_task) link libtorch and
    `libcudart.so.13`. torch loads the CUDA runtime first, from its pip CUDA packages.
  - **FlashInfer** loads a module with no ninja or nvcc when its AOT path holds it:
    `jit_env.FLASHINFER_AOT_DIR/<name>/<name>.so`, read on every fetch. Without nvcc, its CUDA version comes from
    `torch.version.cuda`. Its modules link `libcudart.so.13` and TVM FFI, not libtorch.
  - **CuTe DSL** (FlashInfer's norms) compiles with its embedded compiler. It looks for a toolkit only to dump SASS.
  - **No module carries PTX:** ggml_cuda, the torch extensions and FlashInfer's modules are SASS only. So no driver
    JIT happens, and an unsupported SM fails cleanly.
  - **Triton 3.6** uses its bundled `ptxas` in both installs. It compiles its `cuda_utils` and per-kernel launchers
    at first use with `$CC`, `gcc` or `clang` against `sysconfig`'s `Python.h`. **So a pip host needs a C compiler
    and Python's headers** (D-H).
  - **A pip install contains an nvcc anyway:** `nvidia-cuda-nvcc` 13.2.86, through `flashinfer-python →
    cuda-tile[tileiras]`, under `site-packages/nvidia/cu13/bin`. It is not on `PATH`, and torch, FlashInfer and CuTe
    DSL never find it.
  - **Not verified:** no Holon serve has run without a toolkit. Step 0 checks it first.
- **F5. The locks' nvcc is CUDA 13.0.0 (V13.0.48)**, the release environment's: the public
  `nvidia/cuda:13.0.0-devel-ubuntu24.04` image, with its gcc 13. The `Dockerfile` builds with 13.0.3, so its image's
  binaries are not the ones the locks were verified with (D12 replaces it).
- **F6. What Holon uses from gguf-py.**
  - **std:** five import statements in four files.
    - `weights/gguf/reader.py:16` and `source.py:21`: the type tables of `constants.py`.
    - `mixers/qsa/units.py:312`: `quants.dequantize` at load, for GLM's Q6_K `k_b`/`v_b`.
    - `moe/reference.py:328` and `:354`: `quants.dequantize`, in the reference.
    - `quants.py` imports `constants` and `lazy`. Those three upstream files (≈ 8,200 lines, numpy only) cover std.
  - **judge, testing and the tests:** `GGUFReader`, `GGUFWriter`, `Keys`, `quants.quantize`, that is, the whole
    package.
- **F7. Names and sources.** `holonlm`, `holonlm-core`, `holonlm-std`, `holonlm-plugins`, `holonlm-judge` and
  `holonlm-tools` are free on PyPI, and nothing reserves them before a first upload. `uv.lock` has one git source
  (gguf) and one extra index (torch). The runtime closure is 91 packages, 3 of them Holon's.
- **F8. Pre-existing bug:** a plugin `prebuild` module given several SMs builds the second as `<name>_v1.so`.
  - Example: `python -m holon_plugins.kernels.ggml_cuda.prebuild --sm 89 120`.
  - torch's extension versioner renames a name already built in the process. The module also exports
    `PyInit_<name>_v1`, so renaming the file would not help, and `load` never finds it.
  - It affects ggml_cuda, expert_io, promote, state_copy and cpu_task.
  - `holon prebuild` is not affected: it runs one process per (plugin, SM).
- **F9. Today's modules embed the build host's absolute paths.**
  - The current sm_86 ggml_cuda module holds 51 strings with the development host's absolute paths, the user
    directory and worktree names included. They come from ggml's `__FILE__` asserts and from torch's
    `INTERNAL ASSERT … at "<venv>/…/c10/util/intrusive_ptr.h"`.
  - The expert_io module holds 7. FlashInfer's `rope.so` holds 5.
  - A wheel built from such a tree would publish them (B1).
- **F10. CI sizing.** A full `holon prebuild --sm 120`, the largest SM, on the development host at `MAX_JOBS=16`:
  - 12.5 min wall and 72 CPU-minutes (4,010 s user and 336 s system);
  - FlashInfer's 14 modules took 191 s of it, built one after another;
  - the largest single compile peaked at 3.4 GB RSS.

## 2. Distributions (D1)

| distribution | content | wheel | depends on |
|---|---|---|---|
| `holonlm` | **what users install**: the `holon` command; the locks (D6) | `py3-none-any`, `requires-python >=3.12,<3.13` | every dist below `==X.Y.Z`; the runtime closure pinned exactly, with markers (D9) |
| `holonlm-core` (renamed from `holonlm`) | `holon` (ABI and core); `python -m holon` | `py3-none-any` | torch `==2.11.0`, numpy, msgspec, pyzmq, nvidia-ml-py (minimums, as today) |
| `holonlm-std` | `holon_std`, with the vendored gguf-py subset (D7) | `py3-none-any` | as today, without `gguf` |
| `holonlm-plugins` | `holon_plugins`, sources included (`csrc/`, `vendor/`: the keys hash them; the fallback build needs them) | `py3-none-any` | as today |
| `holonlm-cuda-sm86`, `-sm89`, `-sm120` | prebuilt native modules for one SM, as `.so` files in the kernel plugins' `prebuilt/` directories (D2); `BUILDINFO` in dist-info | `cp312-cp312-manylinux_2_28_x86_64` | `holonlm-plugins==X.Y.Z`, torch `==2.11.0` |

- **Not published:** `holonlm-judge`, `holonlm-tools` and `holonlm-testing`. The judge and the tools work on the
  repository's goldens, snapshots and lock history; testing is never installed with Holon. The workspace root
  (`holon-workspace`) stays the development project.
- **Wheels only.** There are no sdists: an sdist of an overlay or of `holonlm` would make pip try a hopeless build on
  aarch64 or macOS. There are no extras.
- **Why the core is renamed:** `pip install holonlm` has to bring everything. Today `holonlm` is the core alone, which
  std and plugins depend on. Nothing is published, so the rename costs nothing outside the repository. In the
  repository it touches:
  - the members' dependencies and the root `[tool.uv.sources]`;
  - `plans.RUNTIME` and `check_manifest_deps.DISTS`;
  - a test fixture and the docs.
- **The `holon` command** moves from `holonlm-core` to `holonlm` and to the workspace root, so exactly one installed
  distribution owns `bin/holon`.
  - This makes `pipx install holonlm`, `uv tool install holonlm` and `uvx --from holonlm holon …` work. pipx and uv
    expose only the named package's commands.
  - A components-only install runs `python -m holon`.
- **Why one overlay per SM:** a user downloads all three either way, ≈ 155 MB, because pip has no GPU markers. The
  split is purely a size-limit shard: each stays under PyPI's 100 MB default (§8), so neither PyPI nor TestPyPI needs
  a request.
  - A size-limit request is filed in parallel (Q3). If PyPI grants it before the first upload, the binaries fold into
    `holonlm-plugins` as one platform wheel: a packaging-only change, since the plugins look only in their own
    directory (D5).
  - Overlays write only `.so` files at paths no other distribution writes (asserted, D10). So the three install side
    by side, an uninstall removes only its own files, and an upgrade replaces them.
- **One version:** every published distribution is `X.Y.Z`, pinned to each other exactly.
  - `scripts/bump_version.py` and `check_version.py` stay the mechanism. The build script takes the version from
    `core/pyproject.toml` for the distributions it generates.
  - The plan's "one version file" stays deferred: it buys nothing for pip, and dynamic workspace versions change what
    `uv.lock` records.
- **Plugin discovery:** unchanged; pip installs std's and plugins' `holon.plugins` entry points as uv does.

## 3. Native code (D2–D5)

- **D2. What is compiled at build time:**

  | plugin | kind | in the wheels |
  |---|---|---|
  | `kernels.ggml_cuda` | torch extension, one per SM (86, 89, 120a) | prebuilt, fatbin compressed (`--compress-mode=size`) |
  | `kernels.expert_io`, `.promote`, `.state_copy`, `.cpu_task` | torch extensions, one per SM | prebuilt |
  | `kernels.flashinfer` | `holon prebuild`'s FlashInfer set: 8 modules on sm_86, 14 on sm_89 and sm_120 | prebuilt, per SM: Holon's AOT directory for FlashInfer |
  | `kernels.ggml_cpu` | torch extension per CPU tier, C/C++ only | **not prebuilt** (an opt-in no lock uses). `holon prebuild` builds it with gcc/g++, and its `NotBuilt` names gcc/g++, not nvcc |
  | Triton kernels (`triton_*`, `gdn_triton`, `hc_triton`, `ple_triton`, `qsa_triton`, `fla`) | Triton JIT | none: compiled and cached at first use |
  | CuTe DSL norms (through FlashInfer) | in-memory JIT | none (embedded compiler) |

  - **`--compress-mode=size`** joins ggml_cuda's `NVCC_FLAGS` for every build, local prebuilds included.
    - It cuts ggml_cuda's deflated size by 37% (sm_86, sm_89) and 36% (sm_120).
    - It is kept only if D10's full SASS comparison (compressed against uncompressed, every function, every SM)
      finds no difference.
    - It is also kept only if D12's warm start shows less than 1 s of decompression. The decompressed image (≈ 120–154
      MB) becomes anonymous memory per engine process; its RSS is recorded.
    - Otherwise it is dropped: the overlays fit without it (§8).
- **D3. SMs:** 86, 89 and 120: RTX 3090; RTX 4090; RTX 5080, RTX 5090 and RTX PRO 6000 Blackwell. These are the
  locks' classes and the SMs the code already targets (`ARCHS`). Any other SM installs, then fails at start with the
  existing messages (GGUF refused). The README lists the supported SMs before the install command.
- **D4. ABI and the build image.**
  - **torch** is pinned `==2.11.0`, the core's pin today. That matches PyPI's `2.11.0` (F2). Another torch is refused
    (D5).
  - **The overlays are built in `nvidia/cuda:13.0.0-devel-rockylinux8`, pinned by digest.** It has glibc 2.28 and
    the locks' nvcc (F5). Its compiler is gcc-toolset-13, set for nvcc (`-ccbin`), torch's `cpp_extension` (`CC`,
    `CXX`) and FlashInfer's ninja. The script asserts `gcc 13.*` and nvcc `V13.0.48` before compiling, because
    otherwise Rocky's gcc 8.5 is used silently. uv-managed CPython 3.12 provides the headers.
  - **The build happens only at the fixed neutral path `/holon`**, from the `git archive` unpacked there: venv at
    `/holon/.venv`, staging at `/holon/stage`, uv's Python under `/opt/python`. Whatever `__FILE__` or a header
    path embeds is then `/holon/…`, which names nothing private; D10's scan proves it. There is no
    `-ffile-prefix-map` (rev 2.1): it would have to reach nvcc, torch's `cpp_extension`, FlashInfer's ninja and the
    reference build alike, and the neutral path already is the fix.
  - **The wheel tag:** `auditwheel show` must find `manylinux_2_28`, with only torch's libraries and
    `libcudart.so.13` external. These are never bundled, because torch's dependencies provide them. The wheel is
    then tagged `cp312-cp312-manylinux_2_28_x86_64` (`wheel tags`).
  - **No new compatibility record.** Each extension's cache key names its shipped directory: sources, flags, SM, CUDA,
    CPython tag, torch version, and now its own `extension.py` (the build recipe: A-m4, as ggml_cpu already does).
    FlashInfer's directory is named by everything its modules depend on (D5).
- **D5. Lookup, refusal and fallback, in the plugins.**
  - **Holon's extensions:**
    - `module_path` looks in `HERE / "prebuilt" / key(sm)`, then in the user cache as today.
    - `build()` returns at once when the shipped module exists, so `holon prebuild` in a pip install builds only what
      is missing.
    - **F8's fix:** each plugin `prebuild` module re-executes itself once per SM, as ggml_cpu does per tier. The test
      mocks `subprocess.run` and asserts one process per SM.
  - **FlashInfer:**
    - Holon's AOT directory for FlashInfer is always
      `HERE / "prebuilt" / f"{flashinfer}-tvmffi{tvm_ffi}-cu{torch.version.cuda}-sm{sm}"`. The plugin sets
      `flashinfer.jit.env.FLASHINFER_AOT_DIR` to it when it first binds, even when the directory is missing, so an
      installed `flashinfer-jit-cache` can never supply foreign modules.
    - Without an AOT module, FlashInfer falls to its JIT cache, where a repository install's `holon prebuild` puts
      them, as today. The prebuild skip tests each spec's `aot_path`.
    - Setting a third-party module global is acceptable for a kernel plugin: invariant 3 is about Holon's own state.
      The plugin says so in a comment.
    - A process that binds a second SM is refused (`HostError`, since plugins raise only `holon.api` types): the
      same-major binary would otherwise run sm_86 code on an sm_89 device without error.
    - At bind, the plugin refuses (`HostError`) FlashInfer's CUDA norm in place of its CuTe DSL one: if
      `nvidia-cutlass-dsl` fails to import, or `FLASHINFER_USE_CUDA_NORM=1` is set, FlashInfer silently switches,
      which changes numerics.
  - **Refusal of another torch:** a `+cu128` or `+cpu` torch satisfies `torch==2.11.0`. Such a torch finds no
    shipped module (the key holds `cu12` / the version).
    - Holon's loaders and the FlashInfer plugin then raise `HostError` (D-H), when the overlay ships modules but none
      for this torch. They parse the shipped key (`sm…-cu…-cpython-312-torch<v>-<hash>`) to tell the cases apart.
    - The message names `pip install --force-reinstall "torch==2.11.0+cu130" --index-url
      https://download.pytorch.org/whl/cu130`: the exact local label matches only the cu130 build (PyPI's files, F2)
      whatever extra index `pip.conf` adds.
    - A CPU-only torch is refused by the CLI before it reads the GPU, naming the same install.
  - **An overlay from another Holon version** (rev 2.1, A-R5): the shipped key's torch matches but its source hash
    does not, for example after `pip install -U holonlm-plugins` alone. The message is its own: "the prebuilt
    modules are from another Holon version: `pip install --force-reinstall holonlm==X.Y.Z`". This is a `NotBuilt`,
    not a `HostError`.
  - **When nothing matches and torch is right** (a missing add-on, another SM, a components-only install), and no
    CUDA toolkit could build the module: `HostError`, refused at once (code review, A-C2 and C-C1). The message names
    the add-on to install (`pip install holonlm-cuda-sm<N>==<v>`, or `pip install holonlm`), or, for an SM no add-on
    covers, that it needs a source install with CUDA 13's nvcc. One helper words it for every extension and
    FlashInfer (`holon_plugins/host.py`: `missing`, `no_add_on`).
    - **With a toolkit** (a source install): today's `NotBuilt`, naming `holon prebuild`.
    - FlashInfer counts the user's own modules in its JIT cache as built, as the extensions count their cache.

## 4. Host prerequisites, failing fast (D-H)

- **The requirement:** a C compiler and Python 3.12's headers, both for Triton's launchers (F4). A default Ubuntu
  24.04 desktop usually lacks `python3.12-dev` and often gcc.
- **What fails today:** the first Triton compile raises inside the engine's start, as a plain `RuntimeError` or as
  gcc's stderr above a bare `CalledProcessError`.
  - That is an `"error"` start failure, so the supervisor retries it four times, rolls back to the defaults, and
    retries four more (`supervisor.py:362-371`). Each attempt reloads the weights.
  - If a kernel first compiles after "ready", the failure is a crash-rollback onto a configuration that is not the
    lock.
- **The fix has three parts:**
  1. **`holon.api.HostError(RuntimeError)`**: the host lacks something a plugin needs, so retrying cannot help.
     - **Two causes only:** a missing C toolchain, and a torch build the shipped modules do not fit (D5). A missing
       module, another Holon version's overlay and a second SM stay `NotBuilt` or `PluginError`.
     - **ABI 1.9, a minor bump.** It adds the name `HostError` and the kind `"host"` to the listed failed-kinds
       vocabulary (`core/abi.txt`). Every plugin that imports it declares `core_abi = ">=1.9,<2.0"`: `host.py`'s ten
       callers, the five extension loaders and FlashInfer.
     - **The kind:** `_kind` walks the exception chain outermost-first. Only the generic `"error"` kind
       (`PluginLoadFailed`, plain exceptions) is transparent; the first other kind wins. So an outer `OutOfUnits`
       (`"memory"`) or `PluginError` (`"plugin"`) keeps its kind, and `"host"` is found under a generic wrapper.
     - **The supervisor** treats `"host"` like `"plugin"` and `"memory"`, through the existing path (B's form,
       ≈ 3 core lines): no retry, roll back now. Each remaining configuration then fails once, and the supervisor
       gives up with the message. A host requirement that only the lock's configuration has is what rollback is
       for.
     - **Tests:**
       - `PluginLoadFailed` wrapping `HostError` gives `"host"`;
       - an outer `OutOfUnits` or `PluginError` keeps its kind;
       - the supervisor makes one attempt per configuration on `"host"`;
       - the `--no-supervised` path prints the message (a 5-line message path, not tested: corrected in §15).
  2. **`holon_plugins/host.py`**, a shared library module outside every plugin family; the independence contract
     lists families.
     - **What it checks:** `require_c_toolchain()` finds `$CC`, `gcc` or `clang`, and `Python.h` under
       `sysconfig.get_paths()["include"]` (with Triton's `posix_local` → `posix_prefix` correction). Then it compiles
       a one-line C file that includes `Python.h`, in Triton's command form (≈ 50 ms, once per process). That also
       catches a gcc without `libc6-dev`.
     - **Its docstring states the rule** for code outside a unit directory: it must not shape outputs, or the units
       that use it pin it. That is the vendored gguf's rule too, through the stamps (D7).
     - **Its message:**
     > Triton compiles a small C launcher at first use: install a C compiler and Python 3.12's headers (Ubuntu:
     > `sudo apt install build-essential python3.12-dev`), or use a uv-managed Python (`uv tool install --python
     > 3.12 holonlm`), which has its headers.

     **Every plugin that imports Triton calls it at load.** That is ten plugins, found by grepping `import triton`:
     `kernels.triton_attn`, `.triton_experts`, `.triton_fp8`, `.gdn_triton`, `.hc_triton`, `.ple_triton`,
     `.qsa_triton`, `.fla`, `grammar.xgrammar` and `transforms.tail_skip`. A step-1 test asserts that the grep and the
     callers agree.

     **The guarantee, restated** (A-R2): the check fails the first start, before ready, with no retry.
     - Plugins loaded as candidates at `build_model` are checked before any weight is read.
     - Kernels chosen after the weights load are checked after one weight load: the FP8, NVFP4 and MXFP4 qlinear and
       experts, `tail_skip`, and xgrammar's bitmask.
     - Since the check runs at candidate load, a host without a C toolchain is refused even for a deployment that
       would never run Triton. That matches the requirement being global; today every lock binds Triton.
  3. **The README**:
     - the apt one-liner as the first command;
     - the uv-managed Python as the alternative, which also serves distros whose Python is not 3.12;
     - a negative container in D12 checks the message and the time to fail.
- **Later, not v1 (C):** Triton's launcher key depends only on the C source and the platform, and Triton 3.6 has a
  `knobs.build.impl` hook. Shipping `cuda_utils` and the locks' launchers would remove gcc and `Python.h`.

## 5. Locks, data and third-party code (D6–D9)

- **D6. Locks in the wheel.**
  - `holonlm` ships the data package `holon_locks`: `plans/**/<n>.lock`, `plans/defaults.toml` and `uv.lock`. The
    snapshot JSONs stay out (judge data; 15 MB).
  - **Core** (`plans.py`, ≈ 4 lines):
    - **The checkout wins** when the core runs from one: `CORE_DIR.parent.name == "core"` and `parents[1] / "plans"`
      is a directory.
    - **Else the locks package:** `TREE` is `holon_locks`'s directory, from `importlib.util.find_spec("holon_locks")`'s
      `submodule_search_locations[0]`. That works for a package without `__init__.py` too. The generated package has
      one, which D10.6 asserts.
    - **Both present** (an editable checkout beside a stale `holonlm`): the checkout is used, with a warning (rev 2.1,
      A-R6).
    - `find_spec` of a top-level package imports nothing, so it is safe at import, where argparse defaults read it.
      There is no entry-point group.
    - Naming Holon's own data package names no model, layer, format or feature (invariant 1).
  - **No locks package** (a components-only install, or an overlay alone): `TREE` is `parents[1]`. The existing
    warning "no lock directory …: serving defaults" (`cli.py:254-256`) gains "install `holonlm` for the locks".
  - **`uv_lock` reflects what is installed** (A-M2, corrected in rev 2.1 by A-R1 and B; ≈ 12 core lines).
    - `plans.uv_lock` walks the lock's runtime closure as today, ignoring markers, and computes the lock's hash
      exactly as today, so no lock moves.
    - **Drift:** a closure package that is installed at a version other than the lock's. A closure package that is not
      installed is not drift. The walk reaches PyPy-only `cffi` and `pycparser` on CPython, and a package that
      should be present and is absent fails at import anyway.
    - **Equal versions:** an equal version, or the public part of a local-labelled lock version (F2's `2.11.0`
      against `2.11.0+cu130`).
    - **No drift:** the lock's hash. **Drift:** `installed:<hash of the installed (name, version) set>`, and `holon
      serve` warns once at start, naming each package that differs. A drifted environment then shows `uv_lock` in
      `served`'s diff.
    - `holon_tools` refuses to write a lock whose `uv_lock` starts with `installed:`, so propagation never records
      drift.
    - Its test: the development venv (where `cffi` is absent) gives the lock's hash.
    - One code path for the checkout and pip, so a uv venv that drifted is caught too.
  - **`src` in a pip install:** F1 shows it matches. Three things keep it matching:
    - wheels are built only from `git archive` of a commit;
    - the overlays add only `.so` files, which the hash leaves out;
    - D10's CPU check installs every wheel in a fresh venv and fails unless every newest lock is current.
  - **`src_hash` filters before it reads** (B-M2; one core line): `_is_source(rel)` is tested before
    `read_bytes()`. Otherwise every `holon serve` and `plan_of` would read and hold the overlays' hundreds of MB. It
    moves no hash. Its test: an excluded large file is never opened.
  - **Data already inside the wheels:** std's routing profiles (`std/holon_std/tiered/profiles`, part of `std.tiered`).
    Holon ships no tokenizer or template data; both come from the checkpoint.
- **D7. gguf-py vendored.** In `std/holon_std/vendor/gguf/`, laid out like the other vendored trees:
  - **Upstream files, unmodified:** `constants.py`, `quants.py` and `lazy.py`, with upstream's `LICENSE` (MIT, the
    ggml authors), at llama.cpp `27c7a769`.
  - **Our files:** an empty `__init__.py`; `vendor.sh` (`--check` joins `run_all.sh`); `SHA256SUMS`. There is no
    patch.
  - **std's five imports in four files** (F6) become `holon_std.vendor.gguf.constants` / `.quants`. std drops
    `gguf>=0.17`. `std/holon_std/vendor` joins ruff's `extend-exclude`, as fla's vendor does. Its `__init__.py`
    states the rule for code outside a unit (D-H 2): the stamps below pin it.
  - **The using units are stamped** (A-M3, B-M1).
    - `std.weights.gguf`, `std.mixers.qsa` and `std.moe` each get a `[provenance]` in their `plugin.toml`, as AGENTS.md
      already asks of a plugin that uses vendored code: source llama.cpp, commit `27c7a769…`, MIT, and the vendored
      path relative to the unit (`../../vendor/gguf/`; `../vendor/gguf/` for moe).
    - `plugin.toml` is in each unit's `src`, so a re-vendor, which must change the commit, moves exactly those units'
      hashes and outdates their locks.
    - `vendor.sh --check` asserts that the three manifests name its `COMMIT`. `check_provenance` normalizes `..` in
      `files`.
  - **judge, testing and the tests** keep gguf-py from the git source (F6: they need the reader and writer). They are
    not published.
    - `vendor.sh --check` also asserts that its commit is the root's `gguf` source rev, so the oracle and std read the
      same tables.
    - The two copies coexist in the development venv. Their `GGMLQuantizationType` classes are distinct: equality,
      hashing and dict lookups work (IntEnum), but `isinstance`, `is` and class patterns across them do not. A test
      or judge module never mixes them. The rule goes in the vendored `__init__.py`'s comment.
  - **The MTP pack hint is rewritten.** `models.qwen4_exp`'s `source_hint` (`deploy.py`) and the README's
    Qwen3.8-Flash-Next section use `uv run` and the workspace's `gguf`; a pip user has neither.
    - The new recipe is a side venv with numpy and
      `"gguf @ https://github.com/ggml-org/llama.cpp/archive/27c7a769….zip#subdirectory=gguf-py"`, an archive URL that
      needs no git.
    - Strata at `296b59d` comes as an archive too.
  - **`THIRD_PARTY_NOTICES.md`** gets an entry under llama.cpp / ggml (commit `27c7a769`, MIT, the paths).
    `check_provenance` learns non-plugin vendored trees: every tracked `vendor/` directory needs a `LICENSE` and a
    mention in the notices.
- **D8. Notices inside the wheels.**
  - **Every published wheel** carries `LICENSE`, `NOTICE` and `THIRD_PARTY_NOTICES.md` in `dist-info/licenses`
    (`license-files`). `core/`, `std/` and `plugins/` get tracked symlinks to the root files; the generated
    distributions copy them. `[project.urls]` names the public repository.
  - **Vendored trees** ship with their upstream `LICENSE`: ggml_cuda, ggml_cpu, fla and gguf.
  - **`THIRD_PARTY_NOTICES.md` gains "In the binary wheels"**, the compiled code the overlays redistribute, with the
    license texts not yet in it:
    - llama.cpp / ggml (MIT);
    - FlashInfer 0.6.12 (Apache-2.0) and the headers its modules compile: CUTLASS (BSD-3-Clause), CCCL (Apache-2.0
      with LLVM exception), spdlog (MIT), TVM FFI (Apache-2.0);
    - the PyTorch and pybind11 headers inlined into the torch extensions (BSD-3-Clause).
  - **Device code** compiled with the CUDA toolkit is redistributable as object code under its EULA, as for every
    CUDA wheel on PyPI.
  - The build fails if a wheel lacks the three files. FlashInfer itself stays a dependency (`flashinfer-python` from
    PyPI); `flashinfer-jit-cache` and `flashinfer-cubin` are not used.
- **D9. Sources and pins.**
  - **torch** comes from PyPI (F2). The workspace keeps the cu130 index in `uv.lock`; the files are identical.
    **gguf:** see D7. Nothing else comes from outside PyPI.
  - **`holonlm` pins every third-party package of the runtime closure to `uv.lock`'s version** (87 once std drops
    gguf; torch as `==2.11.0`). **The components keep their minimums**, so library use and third-party plugins can
    compose `holonlm-core`, `-std`, `-plugins` and an overlay without the pins.
  - **How the pins are generated:** by `plans.uv_lock`'s closure walk at build time, so `uv.lock` stays the only
    source.
    - Each dependency's marker is carried over (A-m7: `cffi` is PyPy-only).
    - The generator asserts one version per name: a forked `uv.lock` fails.
  - **Why:** a pip install then resolves the environment the locks were verified in. tokenizers, jinja2 (chat
    templates), xgrammar, nvidia-cutlass-dsl and the CUDA libraries all shape outputs. D6's `uv_lock` check catches
    later drift.
  - **The cost:** security fixes in pinned packages (certifi, urllib3, starlette, h11, jinja2) need a patch release.
    That brings a propagation round, which is automatic under release §22 when traces are byte-identical.
  - **The README** recommends `uv tool install --python 3.12 holonlm` or `pipx install holonlm`, or a dedicated venv
    without extra indexes.

## 6. Build and CI (D10–D11)

- **D10. One script, `scripts/release/build_wheels.py`**, the same locally (docker on the development host) and in
  CI. It always runs inside its image, at `/holon` (D4), from `git archive <ref>`. It never runs on a bind-mounted
  checkout.
  1. **Pure wheels:** `uv build --wheel` for core, std and plugins.
  2. **`holonlm`,** from a generated `pyproject.toml`: the pins, `holon_locks`, the `holon` script, the license
     files and `requires-python`.
  3. **Per SM, in the build image:**
     - `uv sync --frozen`, runtime only;
     - `holon prebuild --sm N`, with `XDG_CACHE_HOME` and `FLASHINFER_WORKSPACE_BASE` under `/holon/stage`;
     - the `.so` files copied into the overlay layout;
     - the overlay built from a generated project outside any checkout, with `[tool.hatch.build] ignore-vcs = true`
       and `artifacts = ["*.so"]`. The root `.gitignore`'s `*.so` must never empty it (C-M4).
     - `BUILDINFO` in dist-info: commit, image digest, nvcc and gcc versions, torch, FlashInfer and TVM FFI;
     - `auditwheel show`, then the retag.
  4. **The reference build** (the exactness gate's other half; A-M1, C-M2): the same `holon prebuild --sm N` of the
     same commit in the lock environment's image, `nvidia/cuda:13.0.0-devel-ubuntu24.04` by digest with its gcc 13,
     also at `/holon`.
  5. **The SASS gate,** per SM, CPU-only. Every shipped module must have the same SASS as the reference's, function
     by function: ggml_cuda, the four small extensions, and every FlashInfer module.
     - **Method:** `cuobjdump -sass` (CUDA 13.0's, so the job runs in the devel image, not with Triton's bundled
       12.8 copy), streamed and hashed per function with the address columns stripped, never written out whole.
     - **It fails on any function present in one build and not the other:** names and counts must match 1:1, so a
       naming difference cannot shrink the compared set. Both builds at `/holon` should make names and assert strings
       equal; step 2's first run confirms it.
     - **Device data too** (code review, A-m7): the initialized constant and global tables of every cubin
       (`.nv.constant*`, `.nv.global.init`, read from `cuobjdump -xelf`), section by section. They compare
       deterministically (§15).
     - This covers Rocky's gcc-toolset-13 against Ubuntu's gcc 13, kernel by kernel, with no GPU.
     - **Once, in step 2,** the same comparison runs between compressed and uncompressed builds of every module on
       every SM. That proves `size` mode before the flag is kept, and is repeated whenever nvcc changes.
  6. **The CPU check,** in its own job after all wheels exist:
     - **Wheel contents:**
       - each wheel is under 100 MB and has its license files;
       - the plugins wheel has no `prebuilt/` path;
       - `holonlm` holds `holon_locks/__init__.py`;
       - each overlay holds exactly the expected modules (ggml_cuda, expert_io, promote, state_copy, cpu_task, and
         FlashInfer's `specs(sm)` names), nothing but `.so` under `holon_plugins/kernels/*/prebuilt/` besides its
         dist-info, and no path shared with another overlay.
     - **Privacy (B1), a hard gate before any upload:** `check_private.py --wheels <dir>`, a new mode, scans every
       member of every wheel: text members as text, and binaries through `strings`, `RECORD` and `METADATA` included.
       Any hit fails.
       - CI runs the generic and machine patterns.
       - The lead runs the full patterns (`.private-patterns`, which stays untracked) on the exact artifacts before
         asking the owner to approve an upload, and records the result.
       - A vendored upstream string that matches is allowed by its exact text, never by skipping a file.
     - **Fresh install:** a `python3.12 -m venv` installs every wheel by path, with the third-party pins from PyPI.
       This proves that PyPI torch's requirements agree with the pins (B-m7). Then:
       - `holon --help` runs;
       - `torch.__version__` equals `uv.lock`'s;
       - every newest lock is current (`plans.current_srcs`);
       - `uv_lock` equals the lock's hash (D6);
       - for every shipped SM, each extension's `module_path(sm)` is a shipped file under `prebuilt/`, and every
         `specs(sm)` name exists in the AOT directory. `key()` needs only torch, not a GPU.
     - `uv pip install` of the same set prints no file conflict.
- **D11. CI: `.github/workflows/wheels.yml`** on GitHub-hosted runners (the repository is public, and Actions runs on
  every push). It runs on `workflow_dispatch` and on `v*` tags.
  - **Jobs:**
    - `pure`: steps 1 and 2;
    - `cuda`: a matrix over the three SMs, step 3, in the build image;
    - `reference`: a matrix over the three SMs, step 4;
    - `check`: steps 5 and 6, needing all of them.
  - **Artifacts:** the wheels, `BUILDINFO` and the gate's reports.
  - **Runner sizing (F10):** the standard public runner is 4 vCPU, 16 GB RAM and 14 GB free disk; it can free about
    30 GB by removing preinstalled SDKs.
    - Per SM, a `cuda` or `reference` job needs ≈ 72 CPU-minutes. The vCPUs are SMT threads, FlashInfer's modules build
      one after another, and the image pull, `uv sync` and disk freeing add minutes, so ≈ 25–45 min per job (est., C).
    - **Memory: `MAX_JOBS=3` by default** (rev 2.1, C). ninja schedules the heaviest mmq instances together, and
      4 × 3.4 GB is too close to 16 GB beside the runner's own processes. 4 is used only if step 2 proves an sm_120
      build under `systemd-run -p MemoryMax=14G`.
    - **Disk:** the image (≈ 8–10 GB extracted) plus the venv (≈ 6 GB, `UV_NO_CACHE`) plus the build objects (≈ 1 GB)
      need the freeing step.
    - Six build jobs run in parallel, far inside the 6 h job limit.
  - **Publishing is `.github/workflows/publish.yml`, never a rebuild** (code review, A-C1, B-M3, C-C2). The rule:
    - D12 runs on one `wheels.yml` run's artifacts, and its evidence record lists each wheel's sha256;
    - the publish dispatch names that run and that record;
    - it downloads that run's `wheels` artifact and uploads it only if the files are exactly the record's, behind the
      GitHub environment (`testpypi`, `pypi`) whose required reviewer is the owner;
    - trusted publishing (OIDC), no tokens. Pending publishers are registered only when the go is close.
  - **GPU tests are not in GitHub** (no GPU runners): `scripts/release/wheel_check.sh` runs D12 on each card class
    against a run's artifacts, and the results become evidence records.

## 7. Acceptance (D12–D13)

- **D12. The image and the GPU matrix.**
  - **The image (Q9):** the `Dockerfile` becomes the wheel-based runtime image.
    - `ubuntu:24.04` by digest, plus `python3.12-venv python3.12-dev build-essential`.
    - A venv at `/opt/holon/venv` gets Holon's wheels by path from the build context's `wheels/` (or
      `holonlm==X.Y.Z` from PyPI once published).
    - No CUDA toolkit and no prebuild. It is ≈ 6–7 GB est., against 14.8 GB.
    - This image is D12's container; the desktops run it as an imported container image.
    - The source-install image and its nvcc 13.0.3 (F5) go. If step 2 shows the new image is not cheap, the current
      base is instead pinned to 13.0.0.
  - **Per card class** (RTX 3090, RTX 4090, RTX 5080, RTX 5090), each on the desktop its class's locks were propagated
    on. In the image, checked first:
    - no `nvcc` on `PATH`;
    - no `CUDA_HOME` / `CUDA_PATH`;
    - no `/usr/local/cuda*`.

    The venv's pip nvcc 13.2 (F4) is present. That nothing native compiles is shown by watching the caches, as
    `bench/cold_start.py` does: `prebuilt/`, the AOT directory, FlashInfer's JIT directory and `~/.cache/holon/*` gain
    nothing; only Triton's cache may.
  1. **Origins:** `pip inspect` shows every `holonlm*` distribution with the artifacts' RECORD hashes. Installing by
     path already means a same-version upload by someone else on PyPI cannot be chosen (F7).
  2. **Locks:** `holon models` lists this card's locks as current.
  3. **Smoke:**
     - `holon serve` the class's Qwen3-8B Q4_K_M lock;
     - one chat request answers;
     - `served`'s diff against the lock is empty, `uv_lock` included.
  4. **Load:** every shipped module loads on the card through its plugin's `load`. This covers what no lock binds
     today (promote, cpu_task).
  5. **Traces:** every newest lock of the class (27 over the four classes; the step-3 propagation's set, on the
     same desktops) runs its snapshot trace from the pip install. Each must be byte-identical to the lock's recorded
     trace.
     - **The tools,** installed for this in a second venv layer:
       - the tag's checkout, `pip install -e ./judge -e ./tools`, editable because `holon_tools` finds its prompts
         from `__file__`;
       - their dependencies under a constraints file generated from `uv.lock` (accelerate, pandas, pyarrow,
         compressed-tensors, xgrammar), plus gguf from the git rev;
       - `--plans` on a scratch copy, since an advance writes a lock.
     - **Unchanged runtime:** `pip freeze` before and after shows only additions; the 91 runtime packages are
       unchanged.
  6. **Speed against the repository install on the same desktop** (C-M1):
     - **Serving:** an ABBA run (3 rounds, `perf_smoke`'s rule and clock screen), with the pip venv as one arm.
       - **Workloads:** Qwen3-8B Q4_K_M batch-1 decode and one prefill depth. On the RTX 5080 also its Qwen3-30B-A3B
         Q4_K_M offload lock, batch-1 decode. That covers expert_io's and cpu_task's host code, which a different gcc
         built and the SASS gate cannot see.
       - **Parity rule (rev 2.1, C):** the verdict is not "slower" **and** the median per-round ratio is ≥ 0.98. A
         "void" (the clock screen) means a rerun.
       - **Both arms in the same container context:** the harness runs inside the image, with the checkout, its
         `.venv` and uv's Python directory bind-mounted at their own paths. If parity fails, a third arm (the same
         wheels in a uv-Python venv) separates the interpreter from the binaries.
       - `bench/backends/`' interpreter-override pattern (as for `FREETOKEN_PYTHON`) gives the Holon arm the pip
         venv's `python -m holon`.
     - **First start:** `bench/cold_start.py` from the pip venv, with empty Triton and CuTe caches, plus one warm
       start. Each is recorded against the repository install's, along with the engine process's RSS. They must pass
       two rules:
       - **cold:** only Triton compiles, with the same kernel count as the repository install;
       - **warm:** within 1 s of the repository's. That is D2's rule for `size` mode, applied directly.
     - **Install** (recorded with the network noted; no threshold):
       - wall time of `pip install` into a fresh venv with an empty pip cache, with `uv pip install` for reference;
       - the bytes downloaded and the installed size.
     - **Time to first answer:** install + first `holon serve` + one chat, against the repository's clone +
       `uv sync` + `holon prebuild` + serve.
     - All of these are evidence records. The README states the measured numbers.
  7. **Negative and restricted runs (RTX 5080 only):**
     - the image without `python3.12-dev`: `holon serve` fails with D-H's message on the first start, with no retry.
       Its time to fail is recorded twice: on the Qwen3-8B Q4_K_M lock (before any weight) and on an FP8 or NVFP4
       lock, whose Triton binds after one weight load;
     - one run as a non-root user with site-packages read-only: nothing writes into `prebuilt/` or the AOT directory;
     - a `pip install --upgrade` from the previous release's artifacts, once one exists: old overlay files go.
- **D13. Acceptance (the owner's list):**
  - in a fresh venv, a local-wheel install serves on each card class with no toolkit (D12 1–4);
  - outputs are identical to the repository install: D10's SASS gate on every module, and D12.5's every-lock traces;
  - speed matches the repository install, and the install-to-first-answer time is measured (D12.6);
  - every wheel is under 100 MB (§8);
  - `check_provenance`, the notices and the wheel privacy scan are green (D7, D8, D10);
  - the README's install section is rewritten:
    - supported SMs first;
    - the Ubuntu apt line, `python3.12 -m venv`, `pip install holonlm`, `holon serve`;
    - the uv-managed alternative (`uv tool install --python 3.12 holonlm`);
    - the measured download and disk;
    - where it does not work: no Python 3.12, aarch64, macOS, Windows, drivers older than R580, other SMs, PEP 668
      outside a venv; WSL2 untested;
    - the source install, for contributors, below it.
  - The README's sentence on run-identical traces says that identity holds per machine. Elsewhere it is "within the
    judge's budget", since glibc's libm and the interpreter differ. D12 itself changes the interpreter (Ubuntu's
    CPython against uv's).

## 8. Wheel sizes (deflated, est.)

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
  FlashInfer's modules could take the same flag (`FLASHINFER_EXTRA_CUDAFLAGS`) if more is ever needed.
- The 10 GB per-project quota allows ≈ 160 releases of the largest overlay.
- A user downloads ≈ 158 MB of Holon within F3's ≈ 3.2 GB.

## 9. Core lines, ABI and locks (D14)

- **Core:** ≈ 22 lines on today's 5,281, so ≈ 5,303 of 6,000.
  - `plans.py`: `TREE` (≈ 4 lines);
  - `uv_lock` against the installed versions (≈ 12);
  - `src_hash` filtering before it reads (1);
  - `RUNTIME` renamed;
  - the `"host"` kind, its chain walk and the supervisor's mapping (≈ 5).
- **ABI 1.9, a minor bump:** `holon.api.HostError`, and `"host"` in the failed-kinds vocabulary; `core/abi.txt` is
  regenerated. Its users declare `core_abi = ">=1.9,<2.0"` (D-H). There is no entry-point group, and the `[build]`
  CLI is unchanged.
- **std:** five imports, the vendored package and three `[provenance]` sections.
- **plugins:**
  - `holon_plugins/host.py`, about 15 lines of host checks;
  - six loaders (shipped directory first, `extension.py` in the key, the refusal);
  - the FlashInfer AOT directory, the second-SM refusal and the norm assert;
  - F8's re-exec;
  - ggml_cuda's flag;
  - the Triton plugins' host check;
  - the MTP pack hint.
- **Every lock is outdated** (the core; `std.weights.gguf`, `std.mixers.qsa`, `std.moe`; the kernel plugins; the
  rename also moves `uv_lock`).
  - **One propagation round** on the four classes closes it: 2-run traces byte-identical, no L2 expected, because no
    numerics change.
  - The wheels are built from the propagated commit, so `holon_locks` pins the wheels' code.

## 10. Steps

0. **No-toolkit probe** (first, ≈ ½ day, one RTX 5080). Today's environment, with the FlashInfer AOT override
   prototyped under `local/`, runs in an Ubuntu 24.04 container without a toolkit. `holon serve` Qwen3-8B Q4_K_M and
   bf16, then Qwen3.8-27B. This settles F4's unknowns before any build code.
1. **Code:**
   - D7 (gguf, stamps), the rename and the `holon` script;
   - D6 (core: `TREE`, `uv_lock`, `src_hash`) and D-H (`HostError`, `host.py`);
   - D5 (loaders, keys, refusal, FlashInfer, F8);
   - D2's flag and D8 (symlinks, notices, provenance);
   - D10's script, the privacy `--wheels` mode, the Dockerfile;
   - CPU tests.
2. **Local build** of all wheels in docker on the development host:
   - sizes;
   - the full SASS comparisons (compressed against uncompressed; Rocky against Ubuntu);
   - the CPU check and the full-pattern privacy scan.
3. **Propagation** of the 27 locks (repository install, four classes), committed.
4. **Merge;** then `wheels.yml`, dispatched on `main`, builds the artifacts.
5. **D12 on the artifacts,** four classes; evidence records.
6. **README;** then the owner's go for TestPyPI, then PyPI.

## 11. Risks

- **The SASS gate finds a difference** (Rocky's gcc-toolset-13 against Ubuntu's gcc 13). The host compiler
  preprocesses device code with its own libstdc++ headers. The gate names the kernels first, before any GPU time.
  The fallbacks, in order, with their costs:
  1. **Match Ubuntu's gcc point release in the build image** (gcc-toolset-13's 13.x against Ubuntu's 13.3, est.). The
     cost: a custom compiler in the image.
  2. **Build the overlays on Ubuntu 24.04.** The wheels become `manylinux_2_39`. They drop hosts with older glibc:
     Ubuntu 22.04 (2.35), Debian 12 (2.36), RHEL/Rocky 9 (2.34). They keep Ubuntu 24.04+, Debian 13 and Fedora 40+.
     torch itself would still install on the dropped hosts; Holon would not. This is the owner's call.
  3. **Propagate from the overlay binaries.** Then repository installs become the unverified side. They already
     build with whatever gcc a contributor has, so trace identity across host compilers was never guaranteed for
     them either.
- **Step 0 finds another runtime compile** (e.g. CuTe DSL wanting `ptxas`). Then its pip package becomes a pin, or a
  requirement is documented.
- **The privacy scan's false positives in vendored strings.** They are allowed by exact text; a real hit stops the
  upload.
- **Python other than 3.12:** `holonlm`'s `requires-python` makes pip say so plainly. The README points to uv.
- **Names taken before the first upload** (F7): the size-limit request and the pending publishers are filed only when
  the go is close.
- **The RTX 3090 27B trace differs by desktop** (plan §10, open): D12 runs on the desktop that propagated.

## 12. Owner questions: the defaults this design builds to

The coordinator set these defaults; the owner has not answered.

1. **Rename the core to `holonlm-core`:** yes.
2. **Exact pins:** in `holonlm` only; the components keep minimums. The README recommends `uv tool install` or
   `pipx`.
3. **Per-SM overlays:** yes. A PyPI size-limit request is filed in parallel; if granted before the first upload, the
   binaries fold into `holonlm-plugins`.
4. **Python 3.12 only:** yes.
5. **gcc and Python's headers as requirements:** yes, with D-H's fail-fast check.
6. **Publishing:**
   - trusted publishing, with the owner as approver;
   - TestPyPI also needs the owner's go;
   - publishers are registered only when the go is close.
7. **Leave `holonlm-judge`, `holonlm-tools` and `holonlm-testing` unpublished:** yes.
8. **Don't prebuild `kernels.ggml_cpu`:** yes; its message names gcc/g++.
9. **Docker:** the wheel-based image in this milestone, as D12's container; if that is not cheap, pin the current base
   to 13.0.0.

## 13. The panel's findings on rev 1, answered

A: correctness and robustness. B: architecture and simplicity. C: build, performance and user experience. The panel's
reports are not in the repository; each row says what changed.

| finding | answer | where |
|---|---|---|
| **B-B1** binaries embed build-host paths; nothing scans wheels | **Fixed.** Measured as F9 (51 paths in today's ggml_cuda module). Build only inside the image at `/holon`, plus `-ffile-prefix-map`. `check_private --wheels` covers members, `strings` of binaries, RECORD and METADATA; CI runs the generic patterns, the lead the full ones on the exact artifacts. A hard gate before upload | D4, D10.6 |
| **A-M1** three models cannot back "identical outputs" | **Fixed.** Per-function SASS gate on every shipped module against the lock environment's build, plus every newest lock's trace per class on the pip install | D10.4–5, D12.5 |
| **A-M2** `uv_lock` is a constant in a pip install | **Fixed.** `uv_lock` compares installed versions with the lock's (public part for local labels), else `installed:<hash>`, plus a start warning naming the packages | D6 |
| **A-M3** a gguf re-vendor moves no lock | **Fixed** in B-M1's form: `[provenance]` with the commit in the three using units' manifests, enforced by `vendor.sh --check`. All five imports named | D7, F6 |
| **A-M4** `pipx install holonlm` cannot work | **Fixed.** `holon` is declared by `holonlm` and the workspace root, not by `holonlm-core` | D1 |
| **A-M5** missing compiler or headers fail slowly through retries | **Fixed.** `HostError`, a final `"host"` failure kind, `require_c_toolchain()` at the Triton plugins' load, the apt line, uv-managed Python, a negative container | D-H, D12.7 |
| **A-M6** another torch is accepted; FlashInfer is not keyed on CUDA | **Fixed.** The refusal with the reinstall command; FlashInfer's directory names CUDA (and TVM FFI, B-m6); the README says a venv without extra indexes | D5, D13 |
| **A-m1** lock-root specification | **Changed** to B-m1's `find_spec`. (a) no package: the checkout path and an amended warning; (b) `find_spec` imports nothing, so it is import-safe; (c) the heuristic is gone, since the installed package wins; (d) moot | D6 |
| **A-m2** the CPU check cannot see the overlays | **Fixed.** `check` is its own job after every wheel; the overlay namelists, the lookups resolving to shipped files, and `torch.__version__` against `uv.lock` | D10.6, D11 |
| **A-m3** pin the build image's host compiler | **Fixed.** gcc-toolset-13 for `-ccbin`, `CC`/`CXX` and ninja; `gcc 13.*` and `V13.0.48` asserted; recorded in `BUILDINFO` | D4, D10.3 |
| **A-m4** keys leave out the recipe | **Fixed.** `extension.py` joins the five keys | D4 |
| **A-m5** FlashInfer edge cases | **Fixed:** (a) the AOT directory is always set; (b) the skip tests each `aot_path`; (c) a second SM raises; (d) the CuTe-norm assert | D5 |
| **A-m6** the pip closure holds nvcc 13.2 | **Fixed.** Noted in F4; D12 checks `PATH`, `CUDA_HOME` and `/usr/local/cuda*`, and proves "nothing compiles" by watching caches | F4, D12 |
| **A-m7** markers in the pins; torch's METADATA | **Fixed.** Markers carried, one version per name; the METADATA `Requires-Dist` diffed: identical | D9, F2 |
| **A-m8** the judge setup for traces | **Fixed.** Editable judge and tools, dependencies under `uv.lock` constraints, gguf from git, a scratch `--plans`, a `pip freeze` diff | D12.5 |
| **A-m9** dependency confusion with `--find-links` | **Fixed.** Install by path; `pip inspect` against the artifacts' RECORD hashes | D12.1 |
| **A-m10** F8's fix | **Fixed** as one process per SM, with a mocked-subprocess test | D5 |
| **B-M1** gguf outside every `src` | **Fixed** (A-M3's row) | D7 |
| **B-M2** `src_hash` reads the overlays | **Fixed.** Filter before reading; one core line, no hash moves, with its test | D6 |
| **B-M3** CI on a private repository | **Rejected; the premise is outdated.** The repository is public, and Actions runs on every push, so `wheels.yml` runs on GitHub-hosted 4-vCPU runners. Runner time and size are measured (F10, D11). A PyPI upload does not change the source's visibility. `[project.urls]` names the public repository | D11, D8 |
| **B-m1** `find_spec("holon_locks")` instead of an entry-point group | **Taken** | D6 |
| **B-m2** `uv_lock` reports the shipped file | **Fixed** by A-M2's check rather than a note | D6 |
| **B-m3** `--no-deps` judge cannot import `gguf` | **Fixed** by A-m8's install | D12.5 |
| **B-m4** two gguf copies; ruff | **Taken.** No `isinstance`/`is` across the copies (said in the vendored `__init__.py`); `std/holon_std/vendor` in ruff's exclusions | D7 |
| **B-m5** assert overlay ownership | **Taken.** No `prebuilt/` in the plugins wheel, `.so` only under `prebuilt/` in overlays, disjoint paths, no `uv pip` conflict. `prebuilt/` is gitignored in the plugin directories | D10.6 |
| **B-m6** TVM FFI in FlashInfer's directory | **Taken**, with the comment on the third-party global | D5 |
| **B-m7** pin asserts | **Taken.** One version per name; the fresh PyPI install stays mandatory | D9, D10.6 |
| **B-m8** sdist policy | **Taken.** Wheels only | D1 |
| **C-M1** no speed, first-start or install acceptance | **Fixed.** ABBA with the pip venv as an arm, cold and warm start, install time, download and disk, and install-to-first-answer, each against the repository install on the same desktop | D12.6 |
| **C-M2** SASS evidence too thin; the costs of §10's fallback | **Fixed.** The full per-function gates (compressed against uncompressed, once; Rocky against Ubuntu, every build); every lock's trace; the `manylinux_2_39` cost and the "wheels become the reference" inversion spelled out | D10.5, D12.5, §11 |
| **C-M3** pipx / uv tool | **Fixed** (A-M4's row); `uvx --from holonlm holon` too | D1 |
| **C-M4** the CPU check does not prove that binaries are found; `.gitignore` can empty an overlay | **Fixed.** Overlay projects generated outside any checkout, `ignore-vcs` and `artifacts`; namelists; lookups resolved in a fresh venv | D10.3, D10.6 |
| **C-M5** host prerequisites | **Fixed** (A-M5's row), with the README's apt line and the uv path | D-H, D13 |
| **C-m1** `size` mode costs | **Taken.** Warm `start_s`, RSS, and the D12 desktops' driver versions recorded (the README's R580 floor checked against them) | D2, D12.6 |
| **C-m2** build provenance | **Taken.** Compiler asserts; `BUILDINFO` | D4, D10.3 |
| **C-m3** torch from another index | **Fixed** (A-M6's row) | D5 |
| **C-m4** README flow and where it fails | **Taken**, as D13's list | D13 |
| **C-m5** the judge install | **Fixed** (A-m8's row) | D12.5 |
| **C-m6** no sdists; origins | **Taken** | D1, D12.1 |
| **C-m7** MTP recipe without git | **Taken.** Archive URLs for gguf-py and Strata | D7 |
| **C-m8** read-only and upgrade runs | **Taken** (the upgrade once a previous release exists) | D12.7 |
| **C-m9** CI estimate | **Measured** (F10); disk freeing kept | D11 |
| **C-m10** cross-host traces | **Taken** as a README sentence | D13 |
| **C on Q5** shipping Triton launchers later | **Recorded** as the way out, not v1 | D-H |
| **C on Q9** the wheel-based Docker image now | **Taken** as D12's container, with the 13.0.0 pin as the fallback | D12 |

**Rev 2.1: the panel's notes on rev 2 (e7dd2210).** All three cleared it for build; B withdrew M3.

| finding | answer | where |
|---|---|---|
| **A-R1, B r2-M1** the installed-version check reports `cffi` and `pycparser` as drift everywhere | **Fixed** in B's form, with no new core dependency: a closure package that is not installed is not drift; only an installed one at another version is. The hash is computed as today, so no lock moves. `holon_tools` refuses to write an `installed:` value. The test covers the development venv | D6 |
| **B r2-M2** `HostError` is a minor ABI bump | **Fixed.** ABI 1.9 (the name, and `"host"` in the failed kinds); its users declare `core_abi >= 1.9`. The core count is stated as ≈ 5,303 | D-H, §9 |
| **A-R2** two more Triton importers; the timing guarantee | **Fixed.** The ten plugins are found by grepping `import triton` (`grammar.xgrammar` and `transforms.tail_skip` added), with a step-1 test. The guarantee is restated: before ready, no retry, after one weight load for kernels chosen after the weights. D12.7 times both cases | D-H, D12.7 |
| **A-R3** `_kind`'s precedence; `HostError`'s scope | **Fixed.** Outermost-first, with only `"error"` transparent. `HostError` has two causes (C toolchain, torch); FlashInfer's missing module is a `NotBuilt`. Tests | D-H, D5 |
| **A-R4, B r2-m2** the supervisor and `"host"` | **Changed** to B's form: `"host"` takes the existing no-retry, roll-back path, with one attempt per configuration, then the give-up. The `--no-supervised` path's message is not tested (§15, B m7) | D-H |
| **A-R5** an overlay from another Holon version | **Fixed.** Its own `NotBuilt` message, from parsing the shipped key | D5 |
| **A-R6** a mixed venv | **Fixed.** The checkout wins over an installed `holon_locks`, with a warning | D6 |
| **B r2-m1** `origin` is `None` for a namespace package | **Fixed.** `submodule_search_locations[0]`; the generated package's `__init__.py` is asserted | D6 |
| **A on M1, B r2-m3** where `-ffile-prefix-map` lives | **Dropped.** The neutral `/holon` path is the fix, and the scan proves it. No key moves for it | D4 |
| **B r2-m4** non-unit modules in plugin distributions | **Taken.** `host.py`'s and the vendored gguf's docstrings state the rule: code outside a unit must not shape outputs, or the units that use it pin it | D-H, D7 |
| **B r2-m5** the check refuses Triton-free deployments | **Accepted and stated** | D-H |
| **B r2-m6** the core count | **Fixed:** ≈ 5,303 | §9 |
| **C S1** a weak parity rule | **Fixed.** Not "slower" and median ratio ≥ 0.98; "void" means a rerun | D12.6 |
| **C S2** one GPU-resident workload | **Fixed.** The RTX 5080's 30B offload lock is a second workload | D12.6 |
| **C S3** where the arms run | **Fixed.** Same container context; a third arm if parity fails | D12.6 |
| **C** start pass rules | **Fixed.** Cold: Triton only, equal kernel count; warm: within 1 s | D12.6 |
| **C** CI memory | **Fixed.** `MAX_JOBS=3` by default; 4 only if proven under a 14 GB cap in step 2; ≈ 25–45 min per job; `UV_NO_CACHE` | D11 |
| **C G1** the SASS gate | **Fixed.** In the devel image (`cuobjdump` 13.0), streamed per-function hashes, failing on unmatched functions | D10.5 |
| **C H1** gcc without `libc6-dev` | **Taken.** The check compiles a one-line C file that includes `Python.h` | D-H |

## 14. Step 0, as run (2026-11-03): serving with no toolkit

Record: `docs/dev/evidence/pip_step0_no_toolkit_rtx-5080_2026-11-03.txt`.

- **The setup.**
  - **Container:** the public `ubuntu:24.04` with `gcc`, `libc6-dev` and `curl` added, on one RTX 5080. The check
    found no `nvcc` on `PATH`, no `CUDA_HOME` or `CUDA_PATH`, and no `/usr/local/cuda*`.
  - **Python:** a uv-managed CPython 3.12.12 venv from `uv.lock`, so the repository's code at this branch.
  - **Native modules:** `holon prebuild --sm 120`'s, built on another host.
  - **FlashInfer:** its modules reached through a stand-in for D5's AOT override. The stand-in is a
    `flashinfer_jit_cache` shim, kept in `local/` and never shipped.
- **Result: all three start from this card's lock and answer.**

  | lock | what it exercises | ready after | answer |
  |---|---|---|---|
  | Qwen3-8B Q4_K_M | ggml_cuda, FlashInfer, Triton | 90 s | still reasoning at the 200-token cap |
  | Qwen3-8B FP8 | `triton_fp8`, bound after the weights | 72 s | still reasoning at the 200-token cap |
  | Qwen3.8-27B UD-IQ3_S | fla, state_copy, the 16 GiB KV tier | 113 s | 391, correct |

  - Each plan was reported ready from its lock, with no diff.
  - No native module was built during serving.
  - Triton compiled 68 kernels and 26 launchers with the container's gcc.
  - CuTe DSL's norms ran without a toolkit.
- **F4's unknowns are settled.**
  - Nothing besides Triton needs a compiler.
  - FlashInfer still writes Python-generated `.cu` sources (92 KB) into its workspace, though it compiles nothing. So
    D12.7's read-only run keeps FlashInfer's workspace (under `HOME`) writable, and makes only site-packages read-only.
  - The rest of the design stands as written. Step 1 (the code) starts from here.
