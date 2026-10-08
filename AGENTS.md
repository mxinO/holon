# AGENTS.md — rules for anyone (human or coding agent) changing Holon

Read `docs/design.md` (architecture) and `docs/plan.md` (milestones) first.

## Code style

- **Well structured, elegant and simple.** When two designs are equally robust, choose the simpler one. Small
  modules with one job, explicit data flow, no clever indirection, no speculative abstraction. Delete code rather
  than add flags.
- Match the surrounding code: naming, comment density, idioms. Comments explain *why*, not *what*.
- Type hints on public functions; dataclasses/Protocols for contracts (`holon.api`).

## Invariants (CI-enforced: `scripts/ci/run_all.sh`)

1. The core (`core/holon`) never names a model, layer type, quant format or feature.
2. Plugins import only `holon.api`, their own package and declared third-party libraries; other units are
   referenced with `Ref` and injected, never imported.
3. No module-level mutable state; no environment-variable configuration in core/std.
4. No `torch.cuda` outside `holon.core.platform` and kernel plugins.
5. Core stays ≤ 6000 code lines.

## Implementation policy

- References (mini-sglang, nano-vllm, vLLM, SGLang, FreeToken, Strata, kvmem) are **studied, not copied**. Write
  our own plugin-native code. Only kernels may be borrowed, under design §5.6, as "Third-party code" below says.
- No claim without a measurement. Every performance claim gets an evidence record (command, fingerprint class,
  plan_id or commit, raw numbers) under `evidence/` or the spike's `results/`.
- Every bug fix comes with a test that fails before and passes after.
- Never edit `judge/` goldens/snapshots or `plans/` locks by hand; they are written by the tools.

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

- Python env: `.venv` (uv, `--system-site-packages` over the conda torch 2.11+cu130 stack). Activate with
  `. .venv/bin/activate`. Do not install into it ad hoc — add dependencies to the relevant `pyproject.toml`.
- Machine-specific notes (hosts, clusters, paths) live in `AGENTS.local.md`, which is not tracked; read it if
  present. Never commit private infrastructure details — host or cluster names, usernames, absolute paths — to
  tracked files, results included (`scripts/ci/check_private.py`).
- Consumer performance evidence comes from consumer desktops (RTX 5080/5090/4090); on shared test machines, run
  one persistent container and build/test inside it rather than many short batch jobs.
- Always pin GPUs explicitly (`CUDA_VISIBLE_DEVICES`) in scripts and spikes so parallel work does not collide.
