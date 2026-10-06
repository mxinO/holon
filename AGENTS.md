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
  our own plugin-native code. Only kernels may be borrowed, under design §5.6, with license and provenance recorded.
- No claim without a measurement. Every performance claim gets an evidence record (command, fingerprint class,
  plan_id or commit, raw numbers) under `evidence/` or the spike's `results/`.
- Every bug fix comes with a test that fails before and passes after.
- Never edit `judge/` goldens/snapshots or `plans/` locks by hand; they are written by the tools.

## Environment

- Python env: `.venv` (uv, `--system-site-packages` over the conda torch 2.11+cu130 stack). Activate with
  `. .venv/bin/activate`. Do not install into it ad hoc — add dependencies to the relevant `pyproject.toml`.
- Machine-specific notes (hosts, clusters, paths) live in `AGENTS.local.md`, which is not tracked; read it if
  present. Never commit private infrastructure details — host or cluster names, usernames, absolute paths — to
  tracked files, results included (`scripts/ci/check_private.py`).
- Consumer performance evidence comes from consumer desktops (RTX 5080/5090/4090); on shared test machines, run
  one persistent container and build/test inside it rather than many short batch jobs.
- Always pin GPUs explicitly (`CUDA_VISIBLE_DEVICES`) in scripts and spikes so parallel work does not collide.
