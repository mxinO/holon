# AGENTS.md — rules for anyone (human or coding agent) changing Holon

## Where to start, by task

| task | read |
|---|---|
| using or serving Holon | `README.md` |
| adding a model | `docs/adding-a-model.md` |
| core or ABI work (`core/`, `holon.api`, manifests, locks) | `docs/design.md`: §1 invariants, §3 vocabulary, §4 plugin system |
| a kernel plugin | `docs/design.md` §5.3–§5.4, §5.6 (borrowing kernels) |
| a std unit or another plugin (state, sampling, speculation, frontends) | `docs/design.md` §5–§7 and §9 (feature → plugin map) |
| the judge, goldens, locks and their tools | `docs/design.md` §4.4, §4.6, §10.3; `judge/README.md`; `tools/README.md` |
| benchmarks | `bench/README.md` |

`docs/plan.md` (roadmap and milestone status) and the development records in `docs/dev/` (one per milestone) hold
the history and decisions; read them only for the item at hand, never as required reading.

## Code style

- **Well structured, elegant and simple.** When two designs are equally robust, choose the simpler one. Small
  modules with one job, explicit data flow, no clever indirection, no speculative abstraction. Delete code rather
  than add flags.
- Match the surrounding code: naming, comment density, idioms. Comments explain *why*, not *what*.
- Type hints on public functions; dataclasses/Protocols for contracts (`holon.api`).

## Invariants (CI-enforced: `scripts/ci/run_all.sh`)

1. The core (`core/holon`) never names a model, layer type, quant format or feature.
2. Plugins import only `holon.api`, their own package and declared third-party libraries; other units are
   referenced with `Ref` and injected, never imported. Every distribution declares what it imports (`manifest-deps`);
   production code never imports the test-only plugins (import-linter contracts in `pyproject.toml`).
3. No module-level mutable state; no environment-variable configuration in core/std.
4. No `torch.cuda` outside `holon.core.platform` and kernel plugins.
5. Core stays ≤ 6000 code lines.
6. The plugin ABI is listed in `core/abi.txt` and each distribution's provided contracts in `std/contracts.txt` and
   `plugins/contracts.txt` (`scripts/ci/check_abi.py`); a change regenerates them with `--write` under the version
   rule: a removal or change needs a bump, a new contract is free. Every manifest declares `core_abi`.
7. Every committed lock re-reads with its digest (`scripts/ci/check_locks.py`); `uv.lock` matches the pyprojects;
   vendored trees equal upstream plus their patch.

## Implementation policy

- References (mini-sglang, nano-vllm, vLLM, SGLang, FreeToken, Strata, kvmem) are **studied, not copied**. Write
  our own plugin-native code. Only kernels may be borrowed, under design §5.6, as "Third-party code" below says.
- No claim without a measurement. Every performance claim gets an evidence record (command, fingerprint class,
  plan_id or commit, raw numbers) under `evidence/` or the spike's `results/`.
- Every bug fix comes with a test that fails before and passes after.
- **Lean tests.** A new test catches something no existing test catches; otherwise extend or parametrize an existing
  one. Share expensive setup (module-scoped engines, one run checked by several assertions), prefer tiny models on
  the CPU, and use the GPU only where it is the point. Watch `pytest --durations`. Mark a test `slow` (over ≈ 8 s) or
  `model` (a real checkpoint): `pytest` leaves them out, the pre-merge full run takes them with `-m ""`, sharded over
  idle GPUs (`docs/dev/release.md` §22).
- Never edit `judge/` goldens/snapshots or `plans/` locks by hand; they are written by the tools.

## Commits

- Sign off every commit and sign it: `git commit -S -s`.
- Name no coding agent as an author or co-author.
- Bump the version with `python scripts/bump_version.py X.Y.Z` (every `pyproject.toml`, `holon.__version__`,
  `uv.lock`); CI's `check_version.py` fails if they disagree.

## Third-party code

Holon is Apache-2.0 (`LICENSE`, `NOTICE`). Code from elsewhere — vendored, copied or adapted, even a few lines — is
allowed only under MIT, BSD or Apache-2.0, and is declared where it lands (`scripts/ci/check_provenance.py`):

- **The file says so**, in one line near its top: `Adapted from <project> <commit> (<upstream path>), <license>`
  (or "copied from"). Vendored trees keep their upstream `LICENSE` and are fetched by a script at a pinned commit,
  with our changes in a patch, never edited by hand.
- **The plugin declares it**: its manifest's `[provenance]` gives `source`, `commit`, `license` and every vendored or
  adapted file.
- **`THIRD_PARTY_NOTICES.md` lists it**: project, commit, license and copyright, the repository paths, and the
  license text for every non-Apache license.

Studying code to write our own needs none of this; a comment naming what we followed is enough.

## Environment

- Python env: `uv sync --group dev` makes `.venv` from the committed `uv.lock` — uv's own Python 3.12, torch 2.11
  from the PyTorch CUDA 13.0 index, every pin, the test-only plugins, the judge and the tools; no system
  site-packages. Activate with `. .venv/bin/activate`. A user's `uv sync` installs the runtime distributions only.
  Never install into it ad hoc: declare what a distribution imports in its `pyproject.toml` (CI's `manifest-deps`
  checks it), then `uv lock` and commit `uv.lock`. Host needs: an NVIDIA driver for CUDA 13, the CUDA 13 toolkit
  (nvcc), gcc/g++; `holon prebuild` builds the GGUF kernels and FlashInfer's modules for the visible GPUs.
- Machine-specific notes (hosts, clusters, paths) live in `AGENTS.local.md`, which is not tracked; read it if
  present. Never commit private infrastructure details — host or cluster names, usernames, absolute paths, machine
  details beyond the GPU name — to tracked files, results included (`scripts/ci/check_private.py`), or to commit
  messages, branch names included (`scripts/ci/check_commit_messages.py`: merges take a neutral `-m`).
- Consumer performance evidence comes from consumer desktops (RTX 5080/5090/4090/3090); on shared test machines, run
  one persistent container and build/test inside it rather than many short batch jobs.
- Always pin GPUs explicitly (`CUDA_VISIBLE_DEVICES`) in scripts and spikes so parallel work does not collide.
