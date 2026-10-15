# Release (v0) — packaging, locks, legibility benchmark, benchmark suite, ABI freeze: design and plan

Status: **revision 3.2, reviewed clean** (2026-10-14). Revs 1 (97c39ff), 2 (9534861) and 3 (41431b5) were reviewed by a panel of three
(A: serving, packaging, ABI; B: methodology; C: architecture, scope); every finding is resolved in §12. Scope: plan
§2's Release row ("every v0 model: packaging, quickstart, rtx-5090 / rtx-3090 locks, an agent demo; the legibility
benchmark (§9) before the ABI freezes; the benchmark suite (decode, prefill, agent trace vs llama.cpp, FreeToken and
SGLang where each runs) on the release classes"), plan §9's legibility row (design D9 in `docs/m03-qwen38.md`, moved
before the freeze by the owner, 2026-10-14), design §4.4 / §4.6 (release classes; "a release commit requires every
development and release class refreshed"). Builds on M0.5 (`holon serve`, the HTTP frontend, the Claude Code session),
M0.6 (`docs/m06-lowp.md`, in progress) and M0.7 (`docs/m07-perf.md`, rev 3.1; starts at M0.6 Gate A's merge,
343f3a1), whose step 0 builds the benchmark runner reused here. **Publishing — pushing, making a repository public —
needs the owner's go; this design only prepares it (§8).**

## 1. Goal and acceptance

A user with an RTX 3090, 4090, 5080 or 5090 installs Holon at the release tag, serves a model at its tuned lock with
one command, and points an OpenAI or Anthropic client, or Claude Code, at it; every published number comes from an
evidence record. Acceptance (each a gate, §9):

1. **Legibility — retired as a gate by the owner (2026-10-14; §4.0).** The exploratory pilot is reported as measured;
   no counted runs, no verdict, no legibility claim in the README.
2. **ABI 1.0** (§7): declared, enforced at load, listed and checked in CI.
3. **Locks** (§6): every cell of the matrix written and approved at the release commit, in the release environment;
   `holon serve` reports a plan equal to the lock's on each class, or whose only diff is a reported `max_model_len`
   clamp (D2).
4. **Packaging** (D1): a clean install by the README alone, in a fresh environment, on each class serves a chat
   completion and an Anthropic message; a cold start compiles nothing but Triton kernels; the Dockerfile builds.
5. **Benchmarks** (§5): recorded on all four classes; every GGUF cell's gate statistic ≥ 0.8× llama.cpp at the vendored
   pin (plan §9); all cells published, losses included.
6. **Agent demo** (D7): the pre-registered task, 3 runs each on the RTX 5080 and 4090; published either way.
7. **The public root built and scanned clean, and the owner's go**, then publishing (§8).

Non-goals: wheels and a published image; merging the legibility runs' Mixtral packages; DFlash2 in locks or the suite;
new families or formats; v1's offload, VMM, sandbox and self-improvement; GPUs outside sm_86 / 89 / 120.

## 2. Starting point (blockers found)

| # | Finding | Lands in |
|---|---|---|
| B1 | `holon serve` never reads a lock: tuned pins, KV format, checkpoints and speculation reach a user only as hand-typed flags (M0.5's 27B command has nine). `nearest_class` / `lock_overrides` are called only by the tools | D2 |
| B2 | Locks are found by `model_id` = display name (from the HF cache layout) + build hash: the same file downloaded with `--local-dir` never finds its lock | D2 |
| B3 | The root `pyproject.toml` is virtual: `uv sync --no-dev` installs nothing. The core imports torch, numpy, msgspec and zmq but declares `dependencies = []`; manifests list third-party imports no dist declares; the dev venv hides this through system site-packages. No `uv.lock` (locks record `uv_lock = "none"`), no console script, a transformers dev build | D1 |
| B4 | `kernels.ggml_cuda` compiles on the user's machine (nvcc of torch's CUDA major, ninja, gcc) into a per-user cache and links libtorch; FlashInfer JIT-compiles at first use (S1: ≈ 100–120 s for all specs per SM) | D1 |
| B5 | `core_abi` / `slot_abi` are parsed, declared by no manifest, checked nowhere; the core calls 23 plugin hooks by `hasattr` / `getattr` string that `holon.api` does not declare | §7 |
| B6 | No root README; no contributor entry point short of design.md + plan.md (≈ 167 KB, which AGENTS.md says to read first); AGENTS.md's Environment section describes the system-site-packages venv | D8 |
| B7 | Speculation is on in no lock (M0.4 D8 left it open) | D3 |
| B8 | No SGLang backend or pin; FreeToken and llama.cpp found only through environment variables; FreeToken reads GGUF for one other family and three types, not our files | §5 |
| B9 | rtx-5090's Qwen3-8B bf16 lock is outdated; rtx-4090 has no 8B GGUF lock; rtx-3090 has none | §6 |
| B10 | **The history leaks** (reviewer C's scan of every blob reachable from the main and release branches (4952) and every commit message with `check_private.py`'s full pattern set, vendored trees skipped): 182 historical blob versions match the machine patterns — spike results and READMEs, goldens (22), judge results (54), three evidence files, an old `docs/plan.md` — mainly from 4fb5dc4, 8a9eab1, 8f46bd9, 544ab9a, dcf8902 and 4d2cfa2; four merge-commit messages (ace5d5e1d0, 28e9835588, fa915af5e1, 9c3a7fe0a4) match a private name through their branch names. Today's tree passes. The private remote already holds this history | §8 |

## 3. Decisions

- **D1 — Install from source at the release tag with uv; a Dockerfile for hosts without a toolkit; no wheels.**
  - Why not wheels: the extension links libtorch (one binary per torch build and Python), FlashInfer needs nvcc at first
    use, the lock classes pin `cu130-torch2.11`, and PyPI cannot pin torch's CUDA index. Considered: FlashInfer's
    `flashinfer-jit-cache` wheels (they remove its nvcc need, not ggml's).
  - **Root project:** `dependencies = ["holonlm", "holonlm-std", "holonlm-plugins"]`; `[tool.uv]` with
    `default-groups = []` (no `holonlm-testing`: references never serve), `python-preference = "only-managed"`,
    `environments = ["sys_platform == 'linux' and platform_machine == 'x86_64'"]`; `.python-version` 3.12;
    `requires-python = ">=3.12,<3.13"` in every dist; a committed `uv.lock`. Contributors use `uv sync --group dev`
    (AGENTS.md).
  - **Pins:** torch 2.11.0 from the PyTorch cu130 index (`[tool.uv.sources]`), Triton 3.6, FlashInfer at M0.7's pin
    (0.6.12 today), a **released** transformers on which the bundle tests pass, xgrammar 0.2.8.
  - **Declared dependencies:** the core declares torch, numpy, msgspec, pyzmq; std and plugins declare what they
    import — transformers is a runtime dependency (`std/holon_std/weights/gguf/tokenizer.py` builds the GGUF tokenizer
    with it; plan §1's "judge/test only" is corrected in step 6); FastAPI, uvicorn and xgrammar become plain
    dependencies of `holonlm-plugins` (the `http` and `grammar` extras go: locks select grammar, D3). **`manifest-deps`**
    is an import scan per dist: every third-party top-level import maps (`importlib.metadata.packages_distributions`)
    into the dist's declared closure.
  - **Host requirements:** Linux x86-64, a driver supporting CUDA 13, the CUDA 13 toolkit (nvcc), ninja, gcc / g++.
  - **Architectures:** ggml for sm_86 / 89 / 120a; FlashInfer for 8.6 / 8.9 / 12.0f. Other SMs: GGUF refuses at startup
    with the existing message; bf16 untested, said in the README.
  - **`holon prebuild`** (a core CLI command; no plugin named): kernel plugins declare their build module in a new
    manifest section `[build] module = "…"`; its CLI, `python -m <module> --sm N`, is part of the ABI. The command lists
    the visible GPUs' SMs through `holon.core.platform` and runs one subprocess per (plugin, SM); each module resolves
    its own cache (FlashInfer's sets `FLASHINFER_CUDA_ARCH_LIST` to that SM in its process: plugins may read the
    environment, core and std may not). The core never resolves cache variables. Declaring cache directories for
    read-only mounts (design §5.6 rule 7) waits for the sandbox (v1) and is additive.
  - **Dockerfile** (shipped, not published): `FROM` the public CUDA 13.0 devel Ubuntu 24.04 image **pinned by digest**,
    plus build-essential; the source from `git archive` of the tag; `uv sync --frozen`; `holon prebuild --sm 86 89 120`;
    every cache directory pinned by environment variables to paths independent of the user; entry point `holon`. The
    **environment identity** is the plan's `uv_lock` hash (D2) plus the base image digest.
  - **Measured in step 1:** a cold start compiles nothing but Triton kernels; first-start time (Triton + capture) per
    model against `Policy.start_timeout` (600 s), raised or documented if a model comes close.
- **D2 — `holon serve` starts from the lock** (design §4.3 step 4, §4.4; ≈ 100 core lines, all general).
  - **One lookup by build hash**, in `holon.core.plans`, shared by the core and the tools: the weight source's
    identity → histories whose `model_id` ends in `@<build>` → `nearest_class` with the installed units' `src`. The
    tools write under the existing history's `model_id`.
  - **Plan check:** `plan_of`, `_NOT_PLANNED` and the lock-diff function move into `holon.core.plans`, without git.
    The supervisor's history entries become (config, lock path), and the `--no-supervised` path gets the lock path too;
    `EngineProcess.start` passes it to the engine process, which reads the lock, computes its plan with the lock's `model_id` and `commit`, and returns `plan_id` and the diff against the lock
    in its `ready` message (an internal IPC frame); the log and `/metrics` show them. `uv_lock` hashes only the runtime
    closure: the sorted (name, version, source) of the packages reachable from the three runtime dists, parsed from
    `uv.lock` with `tomllib`, so dev-only pins never move a plan.
  - **Precedence:** user flags > lock > defaults. Dict flags merge per key; `key=` (empty value) deletes a lock setting;
    `--no-lock` serves defaults (both in the README). Lock kernel pins go to `EngineConfig.kernel_preferences` (soft:
    a provider that does not support the site or device falls back to default priority, logged; not planned, like
    `kernels`); `--kernels` pins stay hard. `lock_overrides` raises when one op has two providers; step 2 checks the MTP
    locks, and such an op is left unpinned. `plugin_options` apply wholesale.
  - **Memory:** `max_seqs` and `max_step_tokens` stay exactly the lock's; a lock that does not fit fails to start and
    falls back (below). Only `max_model_len` adapts, through `EngineConfig.clamp_context` (default false, so tools,
    judge and bench are unchanged; not planned), which `holon serve` sets:
    - the **requested** `max_model_len` is the lock's value; with no lock, `--max-model-len auto` (the serve default)
      requests min(the model's trained context, 32768), so every build-time reader stays bounded (the n-gram
      drafter's buffer, rope's `max_positions` fact);
    - today a paged pool requests units for one full-length sequence (`assembly.py` `_bind`), so a request that does
      not fit raises `OutOfUnits` before anything could clamp. With `clamp_context`, paged pools request a floor of
      Σ ceil(r / page) units (each pool its own page size) over the worst-step profile's rows r (= ceil(`max_step_tokens` / page) when the requested
      `max_model_len` ≥ `max_step_tokens`) instead; `max_model_len` = min(requested, the smallest paged pool's capacity
      in tokens), computed after the pools are materialized (`engine.py` L183, which needs the executor's profile) and
      before graph capture (L187: captured graphs bake in the position clamp, `executor.py:480`), and pushed into the
      engine's config copy (shared with the executor and admission), `Scheduler.max_model_len` and the state
      manager's token limit; block tables keep the requested width (already bounded);
    - build-time readers keep the requested value, so the plan's only diff is `engine.max_model_len`, reported;
    - the trained context is a new optional **model-plugin** hook (Qwen3.5 / 3.8 HF configs nest it under
      `text_config`, so the weight source cannot read it generically), an ABI addition before the freeze; None means
      budget only, under the 32K cap.
  - **Unverified** (borrowed from another class): the same, pins soft. **No lock for the file:** defaults, logged.
  - **Fallback:** the supervisor's history is `[defaults + flags, lock + flags]`. The `failed` IPC frame gains a kind
    field (today it carries only text; no message matching); a start failing with `PluginError` or `OutOfUnits`
    (deterministic: a missing provider, an ABI mismatch, a lock that does not fit) rolls back at once; other start failures roll back after
    `rollback_crashes` (4) tries — a start that hangs costs up to 4 × `start_timeout` (40 min) before the fallback, so
    the README names `--no-lock`. In an install, older locks pin other code; design §4.4's "rollback = restart on the
    previous lock" is corrected for installs in step 6.
  - **Where locks come from:** `--plans DIR`, default `plans/` of the source tree holding the installed core; absent → a
    loud warning, defaults. A lock naming a second checkpoint (DFlash2's drafter) is not servable from the lock alone:
    its plan records the drafter by identity, the user passes the path.
  - Tests: one parametrized case in `core/tests/test_plans.py` on toy locks (build-hash match, precedence, merge and
    delete, clamp, soft pins).
- **D3 — Release locks select grammar and decide MTP.**
  - Every release lock selects `grammar.xgrammar` (`slots.logits_processor`): one graph set with a processor selected
    since M0.5's panel round (KV −0.12%, TPOT ≤ 0.05%; `docs/m05-grammar.md` §11, panel item 1). Costs: two vetting
    workers, +1.6–2.2 GB of host RAM and a 3–4 s worker start (`docs/m05-grammar.md` §6, step 2); the plugin's `src`
    is pinned in every lock, so a change to it re-propagates every model; the development-class cells gain it at
    M0.7's propagation.
  - **MTP** (Qwen3.8-27B, per class), on a Claude-Code-shaped trace (M0.7's runner: the agent workload with a ≈ 16K
    shared prefix, several sessions, sampled at the temperature Claude Code sends, read with `bench/capture_proxy.py`),
    ABAB rounds of ≥ 200 requests each, MTP vs plain. **On** if median TPOT improves by ≥ 5% **and** ≥ 2 SE, **and**
    for p95 per-request max inter-token gap and p95 TTFT the one-sided 90% upper bound of the paired per-round ratio
    MTP / plain is ≤ 1.10 (the pooled p99s are reported as point estimates). A power estimate after 4 rounds sets how
    many more to run, up to 8; still undecided → plain, the expected outcome on a noisy trace. Stated beside it:
    KV on the RTX 5080 falls from 58K to 39.6K tokens (`docs/m04-spec.md` §10); the lock keeps M0.7's
    `chunk_with_decode`, which alternates with a drafter. The README says how to turn speculation off
    (`--max-draft-tokens 0`). DFlash2 stays a documented opt-in.
- **D4 — Legibility benchmark: Mixtral against mini-sglang** (§4).
- **D5 — Benchmarks gated at the vendored llama.cpp pin; newest upstream published beside** (§5).
- **D6 — ABI 1.0 at the release** (§7); versions 0.1.0, tag `v0.1.0`.
- **D7 — Agent demo, pre-registered.** `bench/demo/`: a small Python project (three modules, tests failing on a seeded
  bug and a missing function; task: make `pytest` pass without editing tests), committed in step 6. `holon serve` the
  class's 27B lock (5080: UD-IQ3_S; 4090: UD-Q4_K_M); Claude Code at a pinned version, headless, the M0.5 environment;
  3 runs per class, after one uncounted smoke run of M0.5's Claude Code session task (`docs/m05-grammar.md`) at the
  pinned version against the same `holon serve`. Success: the harness scores the agent's `shop/` against its own
  copy of the tests and `pytest` passes. A run that changes the scoring environment is invalid: it counts as a miss
  and is published (as built: §15.5, §15.9). Recorded: turns,
  wall time, served and prefix-saved tokens, TTFT / TPOT, the sampling parameters Claude Code sends, the tool-call
  sequence (no file text). Every miss is published; a smaller task, if any, is pre-registered as a new task and
  reported beside it. Claude Code's auto-compaction assumes a far larger window than the 32K served; the README says so.
- **D8 — Documentation: README, AGENTS.md and a contributor guide**, written in step 4 and frozen at a recorded commit
  before the reference Mixtral ports and the pilot (they are part of what §4 measures).
  - **`docs/adding-a-model.md`** (lean, generic, by a task-independent rule: it covers what every existing model
    package has): one table walking through all three packages (qwen3, qwen3_moe, qwen3_5); one worked example, the
    most recent addition — qwen3_5 from qwen3 via `derived_from`; the manifest, the entry point and the reinstall; `Ref` / `ModelSpec`, `WeightMap` and `stack`,
    the bundle; the static checks and how a model is judged; links into design.md. Nothing the tree does not already
    say; no Mixtral, no benchmark. Reviewer C reviews the frozen text for task tailoring before the freeze. A lean CI
    check (`run_all.sh`) asserts that every path and command it names exists.
  - **AGENTS.md:** short rules plus routing by task — adding a model → the guide; core or ABI work → design.md.
    plan.md is never required reading. The Environment section describes D1's environment and `uv sync --group dev`.
  - **README:** what Holon is; supported GPUs and the recommended file per card (§6), M0.6's files on the 5090 / 3090
    marked "borrowed, unverified"; install at the release tag (both paths); quickstart; `CUDA_VISIBLE_DEVICES` on
    mixed-GPU hosts; Claude Code; turning speculation off, `--no-lock`, `key=`; models served without locks; the
    performance tables (rendered, §5.4); the legibility result, pass or fail; that SHAs in older docs, evidence and
    lock histories refer to the private development history; known limitations (single GPU, no offload, M0.5's recorded
    truncation shortfall and deferred runaway guards, first-start Triton compile, the context limit, a `git pull`
    while serving makes a supervisor restart import new code — stop the server first). Quickstart (fixed by step 10):

  ```bash
  git clone --branch v0.1.0 <repository> holon && cd holon
  uv sync --frozen                     # Python 3.12, torch 2.11 + CUDA 13 and every pin from uv.lock
  uv run holon prebuild                # GGUF kernels and FlashInfer modules for the visible GPUs, once
  uv run holon models                  # the models locked for this card, by name
  uv run holon serve unsloth/Qwen3.8-27B-GGUF   # this card's locked file, into the HF cache; prints the Claude Code line
  ANTHROPIC_BASE_URL=http://127.0.0.1:8000 ANTHROPIC_AUTH_TOKEN=holon ANTHROPIC_MODEL=<the printed id> \
  ANTHROPIC_DEFAULT_HAIKU_MODEL=<same> CLAUDE_CODE_SUBAGENT_MODEL=<same> claude   # (§14)
  ```
- **D9 — Lean tests.** No new suite: the `test_plans.py` case (D2); one `test_plugins.py` case (a missing or
  out-of-range `core_abi`); static scripts in `run_all.sh`: the `manifest-deps` import scan, `check_abi.py`, the guide
  check, `uv lock --check`, `check_locks.py` (§7). The clean install (step 10) is a scripted run per class, not CI. A
  CPU-only GitHub Actions run of `run_all.sh` and the CPU tests comes with publishing.

## 4. The legibility benchmark (plan §9, design D9)

### 4.0 Status: retired as a gate (owner, 2026-10-14)

The harness was built and validated (on the legibility branch, its §13.8: fixtures, δ, reference ports within 0.5δ,
plants failing, `$STATIC` at both branch points, a sandboxed run harness). The owner approved an exploratory 1 + 1
pilot (Claude Sonnet 5.5): **both sides passed the Mixtral check on their first try.** Cost (costing 2): Holon $0.36
(21 turns, 8 files, +144 lines), mini-sglang $0.20 (13 turns, 4 files, +98 / −3). Up to the first passing check,
Holon $0.24 and mini-sglang $0.17 (≈ 1.4×); after it, Holon's agent spent $0.12 on work the prompt did not ask for (a
CPU test, a README line, the static checks three times), prompted by AGENTS.md's rules. Documentation reading was
$0.055 of Holon's cost; neither agent opened design.md. One run per side is within noise (×1.8); no verdict.

**What the pilot did show is isolation.** Holon's change was a new `models/mixtral/` package plus its entry point
(and the unrequested test and README line): no shared code changed. mini-sglang's agent edited three shared files
that every model uses — the MoE detection rule in `config.py` (from the model-type name to "has experts"), a
Mixtral-specific default in the shared config, and a key-rename rule in the shared weight loader applied to every
model's weights — checked only against two Qwen3 regression fixtures. Holon's per-model locks and byte-identical
propagation (e.g. M0.6 Gate A: four new locks, all seven existing snapshot traces unchanged) are the evidence for that
property.

**Decision (owner, 2026-10-14):** Holon's advantage is isolation — adding or changing a model does not touch or affect
other models — not a lower agent cost per model. The cost-based legibility gate (plan §9's row, acceptance 1, step 5)
is retired; no counted runs. Recorded honestly: it is retired after an exploratory pilot that leaned against Holon on
cost (≈ 1.4× to the first pass, ≈ 1.8× overall, one run each). Frictions the pilot surfaced (the bundle's tool /
reasoning formats for a model without them, `derived_from`'s commit pin, AGENTS.md rules prompting unrequested work)
are documentation and small-structure follow-ups; the frozen-docs rule (§13.6) lapses with the gate. The remainder of
§4 is kept as the design of the retired benchmark.

### 4.1 Model

**Mixtral** (`MixtralForCausalLM`, Mixtral-8x7B-Instruct-v0.1): GQA 32 / 8, head_dim 128, plain RoPE (θ 1e6), RMSNorm,
no attention bias, untied head, 8 experts top-2 (softmax → top-k → renormalise, as Qwen3-MoE's `norm_topk_prob`),
per-expert `w1` / `w3` / `w2`, a `gate` router.
- **Holon** has every unit and no package: `std.mixers.gqa` with `qk_norm=None` (a path no test covers: a bug there is
  fixed in std and charged to Holon), `std.moe@1` (`renormalize=True`), `std.ops`, `std.blocks.prenorm`,
  `WeightRule.stack`.
- **mini-sglang** at `9a91cfa` registers Llama, Qwen2, Qwen3, Qwen3-MoE, Mistral and Mistral3, not Mixtral; its layers
  (`RopeAttn` without qk-norm, `MoEMLP` with renormalisation) cover it; its loader hard-codes `gate_proj` / `up_proj`.
- **Biases, both stated:** against Holon, Mixtral is probably memorized on the baseline's side — mini-sglang shares
  SGLang's and vLLM's vocabulary (`RopeAttn`, `qkv_proj`, `MoELayer`), whose Mixtral files are widely published; for
  Holon, a guide written for adding models (D8) where mini-sglang ships none — its equivalent is its upstream history,
  which it keeps (§4.2). The guide-free runs (§4.5) size the second.

Rejected: Llama / Mistral / Qwen2 (mini-sglang has them); Phi-3 / Phi-4 (fused `qkv_proj`: a `WeightRule` maps one
tensor to one target, so a core change); Gemma, OLMo 2, Granite, GLM-4 (missing units on both sides); Qwen3.5-MoE,
Qwen3-Next (no GDN in mini-sglang: kernel porting on one side); SmolLM3 (NoPE: a std knob).

### 4.2 Branch points and environment

- **Holon:** main after M0.6 Gate A merges, with steps 1, 2 and 4 (packaging, D2, the documentation of D8), as **one
  squashed commit**, its tree and log grepped for Mixtral and the benchmark. Stripped from the shipped tree at the
  branch point: `docs/release.md`, `bench/legibility/`, design.md goal 5's "(measured: plan M0.3 legibility
  benchmark)", `docs/m03-qwen38.md` D9, plan.md's Release-row clause "the legibility benchmark (§9) before the ABI
  freezes" and its §9 legibility row, and `docs/m07-perf.md`'s non-goal sentence on the benchmark if that file is on
  main. Any later documentation edit voids the branch point. M0.7 changes no model-facing surface; if that surface
  (recipe terms, `ModelSpec` / `WeightMap`, the manifest schema, the model plugin protocol, std knobs) changes before
  the freeze, the protocol re-runs.
- **mini-sglang:** a clone at `9a91cfa` with its full upstream history (its shipped equivalent of a guide, including
  the Mistral addition 7b670a2). Holon's squash stays: its history holds the benchmark's own design.
- Both clones sit under a root with no CLAUDE.md or AGENTS.md in any parent; the memory files Claude Code loads are
  recorded per run.
- **Per run:** a fresh clone, its own venv installed offline from the clone with the dev group (Holon) or its pinned
  requirements (mini-sglang), JIT caches warmed in it, one RTX A5000 of the development host. `$STATIC`, verified
  passing at the branch point: Holon `scripts/ci/run_all.sh`; mini-sglang `pre-commit run --all-files` (hooks
  pre-installed).
- **Reference:** one file outside both clones, `$REFERENCE` = transformers 4.57.3's `modeling_mixtral.py`.
- **Agent:** Claude Code at a pinned version, one dated model id, pinned thinking / effort and auto-compaction settings;
  headless (`claude -p … --output-format json --allowedTools Read,Write,Edit,Bash,Glob,Grep --max-turns 150`), a
  configuration directory holding credentials only, no web tools, the same Bash timeout on both sides. Wall cap 2 h,
  check time excluded.
- **Prompt** (identical; N is the larger side's check time):

  > This repository is an LLM inference engine. Add support for Mixtral (`MixtralForCausalLM`). `$MIXTRAL` is
  > Mixtral-8x7B-Instruct-v0.1 cut to its first four layers (bf16 safetensors, original key names); the engine must load
  > it and generate from token ids. Chat templates and tool-call parsing are out of scope. `$REFERENCE` is the reference
  > implementation. You are done when `$CHECK $MIXTRAL`, `$CHECK $QWEN3` and `$CHECK $QWEN3_MOE` pass (each takes up to
  > N minutes and prints a short summary; full logs go to a file) and `$STATIC` passes. Do not modify the checker or
  > the fixtures; do not use the network.

### 4.3 Correctness (one check, both sides)

- **Fixtures**, made by slicing the safetensors directly (original key names; the key list recorded): Mixtral's first
  4 decoder layers with embeddings, final norm, head and tokenizer (≈ 12 GB bf16; refused if the config enables a
  sliding window); regressions Qwen3-0.6B and Qwen3-30B-A3B cut to 4 layers.
- **`$CHECK`:** the engine (bench worker and backend, `holon` or `minisgl`, in the run's venv) greedily generates 64
  tokens for 16 prompts of judge prompt set v1 (≤ 512 tokens, batched at 16); HF transformers fp32 on the CPU scores
  the sequences teacher-forced. Pass: every generated token within δ of the fp32 maximum logit; δ per fixture =
  max(0.1, 2 × the largest shortfall of HF bf16's greedy tokens on 8 disjoint calibration prompts). Output: a fixed
  summary of ≤ 30 lines, logs to a file.
- **Validated before the pilot:** a reference Mixtral port per engine, written after the documentation freeze and kept
  outside the clones, passes all three fixtures with its worst shortfall ≤ 0.5 δ (else δ for that fixture is raised on
  both sides before any run; values recorded); both unmodified engines pass the regression fixtures; plants fail on
  both — `w1` / `w3` swapped, renormalisation off, experts off by one, RoPE θ 1e4, a wrong KV-head mapping. A plant
  that passes strengthens the check first.
- A run succeeds only if the harness, after the agent stops, re-runs the three checks and `$STATIC` and they pass.

### 4.4 Cost

- **Costing 1:** `total_cost_usd` from Claude Code's JSON result. **Costing 2:** recomputed per call from the run's
  session JSONL (or `stream-json`) usage fields — input, cache writes (their 5-minute / 1-hour split from the usage
  fields), cache reads, output — with cache rebuilds caused by tool waits longer than the cache lifetime priced as
  reads (they penalize the side with slower checks). A pass needs both ≤ 0.70.
- Reported: turns, wall time, diff size, and tokens and turns per concept through a file-path → concept map fixed
  before the pilot (manifest, entry points, `Ref` / `ModelSpec`, `WeightMap`, bundle, CI rules, and documentation —
  the guide, AGENTS.md, design.md — counted separately, so a fail is attributed to its cause).
- **Documentation cost** (pre-registered, from per-call usage): each documentation token's first cache write plus one
  cache read on every later call of the run, priced as in each costing.
- A run failing for infrastructure (API outage, host fault) is re-run; a run failing its checks or a cap counts as +∞.

### 4.5 Protocol, rule and prediction (fixed before the pilot)

- **Counted:** **n = 10 per side**, fixed; launched in pairs (one per side, concurrently, on two GPUs). Runs 1–2 per
  side are the **pilot**: they validate the harness and estimate σ of log cost for the record, and count if the
  protocol is unchanged after them. A failed pilot run triggers a harness review; if no harness fault is found, it
  counts as +∞.
- **Guide-free runs** (report only): 3 Holon runs on the branch point without `docs/adding-a-model.md` and with
  AGENTS.md's routing line to it removed, to separate documentation from structure.
- **Rule:** r = median cost (Holon) / median cost (mini-sglang), failures +∞, under each costing. **Both r ≤ 0.70 →
  pass; else fail.** Reported on their own lines: the 80% bootstrap interval of r and each side's success rate.
  **Inconclusive:** if most runs on both sides fail although the reference ports passed, the harness (not the
  engines) is in doubt: stop, review it, re-run the protocol; no verdict is drawn from those runs and the release
  waits.
- **Prediction** (cost ∝ turns × mean context, both sides on one agent and system prompt S ≈ 15K tokens):
  - mini-sglang: ≈ 15K tokens read (model files, layers, loader, config), ≈ 45 turns including the loader debugging
    for `w1` / `w3` → mean context ≈ S + 25K = 40K → ≈ 1.8M token-turns.
  - Holon: ≈ 4K of guide and AGENTS.md, ≈ 12K of recipes, units and manifests, ≈ 40 turns (five files, a reinstall,
    `$STATIC`) → mean context ≈ S + 20K = 35K → ≈ 1.4M token-turns.
  - The token-turn ratio is ≈ 0.78. Priced (cache reads ≈ 0.1×, writes ≈ 1.25×, output ≈ 5× the input rate),
    Holon's extra output (five new files) and the cache rewrites after its reinstall and `$STATIC` waits raise it:
    **r ≈ 0.85 (0.6–1.1)**, a fail more likely than a pass. If the agent also reads design.md (≈ 27K tokens), r rises
    by ≈ 0.25–0.6.
- **Fail branches**, pre-registered:
  - **documentation-attributed** if Holon's median with its documentation cost (§4.4) removed gives r ≤ 0.70 under
    both costings: a documentation fix that may only remove or re-route reading (no new content), its diff reviewed and
    grepped like the branch point, then a full re-run (≈ 3–4 days, ≈ $100–500) if the owner decides; the re-run is
    labelled "after a transcript-informed documentation fix" and published beside the first result.
  - **concept-attributed** otherwise: plan §9's pivot (collapse slots and nouns), §4.7.
- **Budget:** 20 counted runs plus 3 guide-free, ≈ 1–2 h each; estimated $5–25 per run → ≈ $100–500 for the
  counted runs and ≈ $50–150 for the guide-free ones (owner question 2); the pilot's actual cost re-estimates it. fp32 scoring needs ≈ 24 GB of host memory per check (two at a time,
  capped).

### 4.6 After the runs

The agents' Mixtral packages stay on local branches, not merged. Frictions costing a run ≥ 10% of its tokens are fixed
before the freeze if non-breaking. The result, pass or fail, goes into the README.

### 4.7 If it fails on concepts

A pivot trigger: the release stops before step 6; I escalate to the owner with the record, the concept attribution and
a design note on which slots and nouns to collapse, reviewed by the panel before code (≈ 2–4 weeks if taken). After any
pivot the full protocol re-runs. The ABI does not freeze until the owner decides.

## 5. Benchmark suite

### 5.1 Runner and baselines

- `bench/run.py` with M0.7 step 0's additions (max inter-token gap, long prompt, paired TTFT). Added:
  `backends/sglang_http.py` (SGLang's `/generate` with `input_ids`, streamed, ≈ 40 lines), a temperature option for
  D3's sampled trace, `bench/report.py` (README tables from records, no hand-copied numbers).
- **llama.cpp:** the **gate** runs at the vendored pin (M0.6's `cbe28bb3`: the same kernels); the newest upstream at
  suite time is a second, ungated column (speculation cells use it if the pin lacks MTP). FreeToken at a pinned commit;
  SGLang at a pinned release; all pinned in `setup_baselines.sh`.
- **Early (step 3, on current main):** feasibility smoke per (engine, model, class) — M0.6's files included on the
  3090 — and tier-1 bars on the RTX 3090 and 5090 against `llama-bench` at `cbe28bb3` and upstream, each record naming
  the ggml commit Holon ran (51ce9c11 on main until M0.6 merges).
- **Where each runs** (step 3 confirms): llama.cpp on every GGUF cell and on M0.6's formats by M0.6's bar; FreeToken
  and SGLang on HF checkpoints only (8B bf16 on 24 / 32 GB cards; M0.6's FP8 / NVFP4 where they load). No GGUF
  comparison with SGLang (spike S2: IQ3_S mis-decoded, 0.05–0.8× llama.cpp) or FreeToken (B8). A cell that does not
  run is published as "—" with the reason.

### 5.2 Cells and settings

| Tier | (file, classes) | Workloads |
|---|---|---|
| 1 | 8B Q4_K_M (all four); 27B UD-IQ3_S (5080), UD-Q4_K_M (4090, 5090, 3090), plain and with MTP; 8B bf16 (4090, 5090, 3090) | bars; `decode_b1`, `decode_b8`; agent trace; long prompt |
| 2 | every other locked file (§6) | bars |

- **Holon serves at its lock**; baselines match the lock's `max_seqs`, step budget, KV tokens and KV format. Flags
  recorded per baseline: `llama-server -ngl 99 -fa on -kvu -np <max_seqs> -c <KV tokens> -b <step budget> -ub 512
  -ctk/-ctv <format>`; `llama-bench` with the same `-ctk/-ctv`, `-b`, `-ub 512`; SGLang `--max-running-requests`,
  `--chunked-prefill-size`, `--max-total-tokens`, `--kv-cache-dtype`, `--mem-fraction-static 0.9`; FreeToken's as in
  `freetoken_http.py`. Greedy, `ignore_eos`, each engine's prefix caching on.
- Prompts are cut from the natural-text corpus (`--corpus`, judge prompt set v1) for every engine (M0.4: uniform ids
  inflate acceptance ≈ 2×).
- **Rounds:** ≥ 3 ABAB, ≥ 4 for the agent trace and the long prompt; per-round values published; one A/A per class on
  the agent trace; a difference under 2 sd of the rounds is printed as "≈".

### 5.3 Metrics and gate

- **Gate statistic** (plan §9), per GGUF cell and per metric — in-process pp512 and tg128 at depth 0 and 8192: per
  round, the median of repetitions 2–5 (`llama_bench.py` vs `llama-bench`, `-ub 512`); the statistic is the median
  over rounds of those medians, ≥ 0.8× the pin. Below it, the cell is fixed and re-measured at ≥ 0.8× before release.
- **Reported:** TTFT, TPOT p50 / p99; per-request max inter-token gap p50 / p99 / max; throughput; KV tokens.
  Speculation tables are labelled with their trace and sampling and add tokens per verify step, the depth-1 share, the
  speedup over Holon plain and the share of repeated output.
- **Ratios are the claim;** absolute tokens/s are per board.
- **Predictions** (compared, not gated): GGUF decode 0.97–1.05× and pp512 1.04–1.20× the pin; agent TTFT p50 well below
  `llama-server`'s (57 vs 228 ms, 8B, M0.1); bf16 8B vs FreeToken 0.98–1.0× decode (M0.3, A5000); SGLang no prior
  record; 27B + MTP 1.5–1.9× single stream.

### 5.4 Publication

Records `evidence/bench_release-<cell>_<class>_<date>.json` (`run.py`: public command, Holon commit, the lock's
`plan_id`, the environment identity, baseline commits and flags, KV capacities, per-request rows, per-round summaries;
GPU name only). README tables by `bench/report.py`, each linking its record; losses stay in.

## 6. Locks

One lock per model's recommended file per card (✓ on main, **R** new, — borrows or does not fit):

| Model file | rtx-5080 (16 GB) | rtx-4090 (24 GB) | rtx-5090 (32 GB) | rtx-3090 (24 GB, sm_86) |
|---|---|---|---|---|
| Qwen3-8B bf16 | — (does not fit) | ✓ | **R** (refresh) | **R** |
| Qwen3-8B Q4_K_M | ✓ | **R** | **R** | **R** |
| Qwen3-8B UD-IQ3_XXS | ✓ | — | — | — |
| Qwen3-30B-A3B UD-IQ3_XXS | ✓ | — | — | — |
| Qwen3-30B-A3B Q4_K_M | — (17.3 GiB) | ✓ | **R** | **R** |
| Qwen3.8-27B UD-IQ3_S | ✓ | — | — | — |
| Qwen3.8-27B UD-Q4_K_M | — (15.3 GiB) | ✓ | **R** | **R** |
| M0.6's files (8B NVFP4 / FP8, 30B NVFP4, 30B MXFP4_MOE) | M0.6's locks | M0.6's locks | borrow | borrow (FP8: none, v1) |

- 9 new locks; every lock selects grammar, the 27B locks carry D3's MTP decision. Served with defaults, no lock: Qwen3.5
  (4B / 9B), the bf16 27B and 30B (large-memory correctness paths), M0.6's fixtures and the `amd-quark` file.
- **Environment:** steps 7–9 run in the step-1 environment (the Dockerfile's image built on each class's desktop, one
  persistent container), its identity recorded; the development-class locks are re-propagated in it.
- **Process:** `propagate` every existing lock, then `tune` each **R** cell from the nearest class's lock, L2 against
  the existing goldens and floors (per kernel path, not per card), snapshot run-identical ×3. Approvals delegated for
  passing verdicts, recorded here, counted per change (design §4.6), and the sweep's wall time per class measured, both
  against plan §9's triggers (> 30 min, > 1 / day). A trigger that fires schedules async propagation for v1 and is
  recorded; it does not block the release.
- **Release commit:** every class refreshed at it, on the public root (§8); any later code change propagates before
  the tag. If M0.6 Gate B merges after step 7, it adds one more propagation.

## 7. The ABI freeze

**Frozen at ABI 1.0:**
- `holon.api`: every public name and signature, each Protocol member marked **required** or **optional**. The core's
  plugin hooks called by `hasattr` / `getattr` string today — `admit`, `adopt`, `bind_kernels`, `build`, `capacity`,
  `cut`, `draft_head`, `finalize_weights`, `graph_key`, `hit_length`, `ks`, `lookahead`, `needs_host_lens`,
  `next_publish`, `payload_pool`, `planner`, `q_support`, `rewind`, `shutdown`, `state_specs`, `stats`, `substeps`
  — become small separate optional Protocols (the core keeps its `hasattr` checks, now against declared names); so
  does the model plugin's trained-context hook (D2). Also `EngineConfig`'s fields, the `BuildCtx.options` keys the
  core provides, and the IPC records (`EngineRequest`, `TokenDelta`, `Abort`).
- The manifest schema (sections, keys, `[build] module` and its `--sm N` CLI), `SLOTS`, the entry-point group
  `holon.plugins` (value: the plugin's package; design §4.1's "path to manifest" is corrected in step 6).
- String vocabularies: `KernelSpec` props, finish reasons, event and message kinds.
- The `op.*@major` contracts (`attention.paged_decode@1`, `qlinear@2`, …) and std unit interfaces (`name@major`,
  knobs whose defaults never change).
- The lock format, the fingerprint class format, `canonical_hash` and `src_hash`.
- `holon serve`'s model-name grammar: a path, `<org>/<repo>`, `<org>/<repo>:<TAG>`, `<org>/<repo>/<file>` (design
  §5.5; the weight sources own it beyond `<org>/<repo>`, §14.3).
- Not frozen: `holon.core` internals, the judge, the tools.

**Mechanism:**
- `holon.api.ABI`; `core_abi` **required** in every manifest (first-party and `holonlm-testing`) from step 2, checked
  before the strict key parse, so a manifest from another ABI gets the version message; out of range → startup error
  (invariant 7). `slot_abi` is removed for v0; per-slot versions return, additively, when a slot first needs one.
- `scripts/ci/check_abi.py` renders the frozen surface — names, signatures, dataclass fields in order with types and
  defaults, Protocol members with their required / optional mark and whether plugins or the core implement the
  Protocol, `SLOTS`, manifest keys, `EngineConfig`, the options keys, the string vocabularies — and compares it with
  `core/abi.txt`. It also collects every `hasattr` / `getattr` string literal in `holon.core` and fails on one
  `holon.api` does not declare, except an allow-list for the core's own attributes (`parent`, a core object's field). A difference fails unless the listing is regenerated with the version bumped: a new
  optional Protocol or member, a new name, field or key → minor; a required member added to a plugin-implemented
  Protocol, or anything removed or changed → major (a reviewed core ABI change, invariant 9, with a codemod for
  first-party plugins).
- **`check_locks.py`** in `run_all.sh` re-reads every committed lock (digest and `plan_id` recomputed by
  `canonical_hash`: golden vectors for it) and rejects two `model_id`s for one build. Its `--release` mode also checks
  that every newest lock's unit `src`s match the tree (`src_hash`); it runs in step 10 and before the tag.
- **Timing:** the mechanism lands in step 2 with the ABI at 0.1, so M0.7's one propagation covers it and D2; step 6
  sets 1.0 and versions 0.1.0 before the locks.

**Known pending additions** (checked against the rule): v1's offload / resolver hooks (a TieredStore, an in-graph
resolver, placement policies — new optional Protocols and an existing slot: minor), per-site kernel pins (a new
`EngineConfig` field and lock section: minor), changed-line coverage and L2 metrics in locks (new lock keys: minor),
the VMM ledger's live resize (an optional state-kind hook: minor). If any of them needs a required member on a
plugin-implemented Protocol, an early 2.0 is expected and accepted.

**What the legibility benchmark can still change**, until the freeze: anything above — collapsing slots or nouns, the
recipe terms, the model plugin protocol, manifest keys, how plugins register.

## 8. Publishing (prepared, not done)

- **History:** the history as it stands is not published (B10). Right after the freeze (end of step 6) a **squashed
  public root** of the release tree is created locally and steps 7–10 commit on top of it, so the release locks and
  records name public SHAs. The public line is pushed nowhere before the go, the private remote included. At step 6
  local main moves to the public line and in-flight branches (e.g. M0.6 Gate B) are cherry-picked onto it; development
  continues there. Owner question 1 (needed by step 6; default: squash) chooses this
  (recommended: pushed later to a fresh public repository) or `git filter-repo` over every leaking blob and message
  with a force-push of the private remote — which already holds the early history, and GitHub keeps force-pushed-over
  commits reachable by SHA until it purges them, so that path also needs GitHub's purge.
- **Step 10's scan:** `check_private.py`'s full pattern set over every blob of the public line, goldens' and results'
  safetensors metadata included, and every commit message; zero hits. No matched value is printed into a tracked file.
- Also before asking: `check_provenance.py` green, `THIRD_PARTY_NOTICES.md` complete, release notes (what v0 serves,
  the §6 matrix, results, the legibility result, known limitations, recorded shortfalls), the tag.
- **On the owner's go only:** push the public line and the tag; make the repository public; enable the CPU-only GitHub
  Actions run. The Dockerfile ships; a published image and PyPI wheels do not (v1).

## 9. Steps

| # | Work | Checks | Days |
|---|---|---|---|
| 1 | Packaging (D1): root project, declared deps, `manifest-deps`, pins, `uv.lock`, `.python-version`, `holon prebuild` with `[build]`, Dockerfile | fresh venv without system site-packages: `holonlm-testing` absent, CPU tests pass; cold start compiles only Triton; first start vs `start_timeout` | 2.5 |
| 2 | D2 (serve from the lock, the trained-context hook, the `failed` frame's kind) and the ABI mechanism at 0.1 (§7), `core_abi` required | `test_plans.py`, `test_plugins.py` cases; `check_locks`; `check_abi` | 3 |
| 3 | Baselines (§5.1): pins, SGLang backend, temperature option, `report.py`; smoke and tier-1 bars on the 3090 / 5090 — on current main | smoke matrix and bars recorded | 2 |
| — | M0.6 Gate A merges (Gate B is off the critical path); M0.7 starts | — | — |
| 4 | Documentation (D8), frozen at a recorded commit; then the legibility harness: fixtures, `$CHECK`, calibration, reference ports, plants, `$STATIC`, concept map, the squashed branch point | guide check; δ and port margins recorded; plants fail | 3 |
| 5 | ~~Legibility runs~~ — retired by the owner after the exploratory pilot (§4.0) | pilot recorded | — |
| — | M0.7 closes: one propagation covering D1's environment, D2, the mechanism and grammar on the development classes | its gates; served plan = lock on rtx-5080 / 4090 (acceptance 3's form) | — |
| 6 | ABI 1.0, versions 0.1.0 — first cut the names nothing uses (M0.7 §17: the `SLOTS` with no plugin and that the core never loads, the dead `SchedulePolicy` Protocol, manifest `[tunables]` / `Binding.tunables`; re-adding one later is a minor bump); design §4.1 / §4.4 / §4.5 and plan §1 corrections; demo task committed (D7); panel review of steps 1–6; the squashed public root created locally | review clean | 2 |
| 7 | Locks (§6) in the release environment on the public root: propagate all, tune the R cells, MTP decision (D3) | L2, snapshots ×3; approvals and sweep time recorded | 2 + 3 machine |
| 8 | Benchmark suite (§5), after step 7 | records; gate | 1.5 + 3 machine |
| 9 | Agent demo (D7) | record | 0.5 |
| 10 | Clean install by the README on each class; `check_locks --release`; history scan (§8); notices; release notes; release status in plan.md closing or deferring each carried CI item (`no-arch-branches`, `reference-provenance`, `isolation-check`, the IPC IDL); panel code review | review clean; zero scan hits | 2 |
| 11 | **Owner's go**, then publishing (§8) | — | 0.5 |

**Schedule, conditional on a legibility pass:** M0.6 Gate A's rest (A4–A6, ≈ 7–9 days) → M0.7 (14–18 days; steps 1–5
run beside it) → steps 6–11 (≈ 12–14 days with 7 and 8 sequential, machine days included) ≈ 33–41 days, **≈ 5–7
weeks**. A documentation-attributed fail adds ≈ 3–4 days and ≈ $100–500 if the owner re-runs; a concept-attributed fail
adds the pivot (≈ 2–4 weeks) and a full re-run. Core diff: D2 (≈ 100 lines), `holon prebuild` and the ABI check
(≈ 50), all general.

## 10. Risks

- **Legibility fails** (predicted r ≈ 0.85, a fail more likely than a pass): pre-registered branches (§4.5, §4.7).
- **Context:** Claude Code assumes a far larger window than 32K; long sessions hit the limit (README, D7).
- **Baselines on sm_120 / sm_86** may not run: "—", no patches beyond a documented one-liner; step 3 finds out early.
- **Late fixes** after step 7 re-propagate every class; the tag waits.
- **Grammar in every lock** couples every model to the xgrammar plugin's `src` (D3).
- **First start:** Triton compiles per model shape; if it nears `start_timeout`, step 1 raises the limit.
- **Desktop availability** (four classes, ≈ 6 machine days): parallel per class; the 3090 is the least exercised.

## 11. Open questions for the owner

1. **Publishing** (needed by step 6; default: squash): the squashed public root created locally after the freeze,
   local main moved onto it, steps 7–10 on top, pushed nowhere before your go and then to a fresh public repository
   (recommended) — or `filter-repo`, a force-push of the private remote and GitHub's purge? Reserve the PyPI names
   `holonlm*` now?
2. **Legibility agent and spend:** proposed default — Claude Code with the current Claude Opus model at its dated id,
   default thinking / effort, on your account; 20 counted runs ≈ $100–500 and 3 guide-free ≈ $50–150, plus
   ≈ $100–500 for a documentation-fail re-run.
   Agreed, or another model / cap?
3. **FP8 without locks:** if M0.6 Gate B is late, does FP8 ship served with defaults (a scope change), or wait?

**Lead decisions recorded:** versions 0.1.0 / ABI 1.0 (D6); Mixtral, n = 10 and both costings (§4); MTP per D3;
grammar selected in every lock (D3); llama.cpp's vendored pin as the gate, upstream beside (D5); 3-bit files locked
only on the 16 GB card; Qwen3.5 without locks; M0.6's files borrow on the 5090 / 3090, are smoke-tested on the 3090 in
step 3 and marked "borrowed, unverified" in the README (§6); Dockerfile shipped, image not published (§8).

## 12. Review findings, resolved

**Rev 3.1** (A, the clamp's insertion point): the pool capacity exists only after `ledger.materialize` (the profile needs
the executor), so the clamp is applied after it and before graph capture, pushed into the engine config, scheduler and
state manager; the floor is Σ ceil(r / page) over the profile's rows (D2).

**Rev 3** (A: serving; B: methodology; C: scope).

| Finding | Resolution |
|---|---|
| A-MAJOR: the clamp can never run — paged pools request one full-length sequence (`assembly.py` `_bind`), so `OutOfUnits` fires first | requested `max_model_len` = the lock's, else min(trained context, 32768); with `clamp_context` paged pools request ceil(`max_step_tokens` / page) units and `max_model_len` = min(requested, the smallest paged pool's tokens) before the state manager and scheduler; build-time readers keep the requested value, so the only plan diff is `engine.max_model_len` (D2) |
| A-m1: start failures told apart by text | the `failed` frame gains a kind; `PluginError` and `OutOfUnits` roll back at once (D2) |
| A-m2: `WeightSource` cannot read Qwen3.5 / 3.8's nested context | an optional model-plugin hook; None → budget only under the 32K cap (D2, §7) |
| A-m3: `parent` is the core's own attribute | allow-listed in `check_abi`, not declared (§7) |
| A-m4: the diff function and `--no-supervised` | the lock-diff function moves into the core with `plan_of`; `--no-supervised` gets the lock path (D2) |
| A-m5 | covered by A-MAJOR |
| B-MAJOR 1: the guide could be tailored to the task; mini-sglang's history removed | task-independent rule: all three packages in one table, qwen3_5 from qwen3 as the worked example; reviewer C reviews the frozen text; listed as a bias in Holon's favour; mini-sglang keeps its full history; 3 report-only guide-free runs (D8, §4.1, §4.2, §4.5) |
| B-MAJOR 2: the documentation-fail branch is undefined | pre-registered documentation cost (first write + later reads per call) and its 0.70 test under both costings; the fix only removes or re-routes reading, reviewed and grepped; the re-run labelled and published beside the first (§4.4, §4.5) |
| B-m1: prediction unpriced | r ≈ 0.85 (0.6–1.1) priced with writes, reads and output; design.md adds 0.25–0.6 (§4.5) |
| B-m2: MTP rule on p99s of a short trace | paired per-round ratios of p95 gap and p95 TTFT with ≥ 200 requests per round; pooled p99 reported; power estimate after 4 rounds; "undecided → plain" expected (D3) |
| B-m3: pilot runs outside n | runs 1–2 of the 10; a failed pilot run without a harness fault counts as +∞ (§4.5) |
| B-m4: TPOT "or" | ≥ 5% **and** ≥ 2 SE (D3) |
| C-m1: ordering of the public line | owner question 1 needed by step 6, default squash; pushed nowhere before the go, private remote included; local main moves at step 6, in-flight branches cherry-picked (§8, §11) |
| C-m2: a documentation fail could hide | covered by B-MAJOR 2: both results published, numeric definition |

**Rev 2** (A: serving, packaging, ABI; B: methodology; C: architecture, scope).

| Finding | Resolution |
|---|---|
| A-N1: clamping `max_seqs` / `max_step_tokens` changes plans silently; budget sizing as a default | only `max_model_len` adapts, inside the engine after pool sizing, capped by `WeightSource.max_context`, reported as a plan diff; `clamp_context` and `--max-model-len auto` in `holon serve` only; acceptance 3 allows that diff (D2, §1). Supersedes C2-m3 |
| A-N2: source-match check in every `run_all` breaks between propagations | `run_all` re-reads digests / `plan_id` and checks one `model_id` per build; source match is `--release`, in step 10 and before the tag (§7) |
| A-N3: `uv_lock` moves on dev-only pins | hash of the runtime closure's sorted (name, version, source), parsed with `tomllib` (D2) |
| A-N4: served-plan check before propagation | after M0.7's single propagation (§9) |
| A-N5: hook list incomplete | 23 literals listed; `check_abi` fails on any undeclared `hasattr` / `getattr` literal in `holon.core` (§7) |
| A-N6: Protocol additions ambiguous | members marked required / optional; optional hooks as small Protocols; new optional → minor, required on plugin-implemented → major (§7) |
| A-N7: `kernel_preferences` would enter plans; how the engine learns the lock | not planned (with `_NOT_PLANNED` moved to `holon.core.plans`); the lock path travels with the history entry to the engine process (D2) |
| A-N8: no way to drop a lock setting | `key=` deletes; `--no-lock` documented (D2, D8) |
| A-N9: grammar's costs unstated | host RAM, worker start, `src` coupling, development-class timing stated (D3, §10) |
| A-N10: prebuild reads devices and caches itself; `caches` unused | GPUs through `holon.core.platform`; caches resolved by each module; `[build].caches` dropped; `--sm N` is ABI (D1, §7) |
| A-N11: mixed-GPU hosts | `CUDA_VISIBLE_DEVICES` in the README (D8) |
| A-N12: deterministic start failures wait for four tries | roll back on the first `PluginError`; a hanging start costs up to 40 min, `--no-lock` named (D2) |
| A-N13: bare `hf` | `uv run hf download` (D8) |
| A-N14: dev tools for contributors and `$STATIC` | `uv sync --group dev` in AGENTS.md; run venvs include the dev group (D1, §4.2) |
| A-N15: `core_abi` required late | required from step 2 (§7) |
| B-M-1: n from a pilot is a forking path | n = 10 fixed; pilot validates and estimates σ, counts if the protocol is unchanged; failed pilot run → harness review (§4.5) |
| B-M-2 / C2-M2: Holon's entry point is ≈ 167 KB of design and plan | lean `docs/adding-a-model.md` and routing AGENTS.md, frozen before the ports and pilot; guide check in CI; prediction re-derived with arithmetic; documentation- and concept-attributed fail branches; schedule conditional on a pass; stripped lines listed (D8, §4.2, §4.5) |
| B-M-3: MTP rule's margins and statistics | fixed 10% margin, one-sided 90% bound of the paired per-round ratio on pooled p99s, TPOT ≥ 5% or 2 SE, rounds up to 8, else plain (D3) |
| B minor: usage source, both costings | per-call usage from the session JSONL; cache-write split; pass needs both ≤ 0.70, in §1.1 too (§4.4) |
| B minor: both sides failing | stop, review the harness, re-run; no extension; inconclusive defined (§4.5) |
| B minor: two check times in the prompt | one N, the larger (§4.2) |
| B minor: port margin | worst shortfall ≤ 0.5 δ, else δ raised on both sides first (§4.3) |
| B minor: early bars' versions | `llama-bench` at `cbe28bb3`, Holon's ggml commit recorded (§5.1) |
| B minor: gate statistic | median over rounds of per-round medians of repetitions 2–5, `-ub 512` (§5.3) |
| B-m7: stripped lines unlisted | listed (§4.2) |
| C2-M1: release locks and records would name private SHAs | squashed public root created locally after step 6; steps 7–10 on it; README note on older SHAs; in owner question 1 (§8, D8, §11) |
| C2-m1: unrecorded scope choices | lead decisions: Qwen3.5 without locks, M0.6 files borrowed and smoke-tested on the 3090, grammar in every lock (§11) |
| C2-m2: a fired propagation trigger | async propagation scheduled for v1, recorded, not blocking (§6) |
| C2-m3: budget-sized `max_model_len` without a lock | superseded by A-N1 |
| C2-m4: base image floats | `FROM` by digest; environment identity = runtime-closure hash + base digest (D1) |
| C2-m5: ABI 1.0 vs v1's additions | pending additions listed and checked as minor; an early 2.0 accepted otherwise (§7) |
| C2-m6: schedule arithmetic | 33–41 days, ≈ 5–7 weeks on a pass; fail paths budgeted; Gate B's late merge adds one propagation (§9, §6) |

**Rev 1.**

| Finding | Resolution |
|---|---|
| C-B1: history leaks (182 blob versions, 4 merge messages); §2's scan too narrow | B10 corrected; full-pattern scan of every blob and message in step 10; squashed public root; image not published (§8, Q1) |
| A-M1: root project virtual; `uv sync --no-dev` installs nothing | root dependencies, `default-groups = []`; step 1 asserts `holonlm-testing` absent (D1) |
| A-M2: core imports undeclared, hidden by system site-packages | declared; `manifest-deps` as an import scan per dist; step 1 in a fresh venv; AGENTS.md updated in step 4 (D1, D8) |
| A-M3: Python and compiler unpinned | `.python-version`, `<3.13`, `only-managed`, gcc / g++, build-essential (D1) |
| A-M4: prebuild hard-codes plugins; one arch list for all SMs; per-user caches | `holon prebuild` reads the manifests' `[build]`, one subprocess per (plugin, SM); caches pinned in the Dockerfile; cold start checked; `flashinfer-jit-cache` noted (D1) |
| A-M5: "served plan = lock" not computable | `plan_of` in `holon.core.plans` without git; plan with the lock's `model_id` / `commit`; diff reported (D2) |
| A-M6: lock pins hard; feature keys classified by the core | `kernel_preferences` (soft), `--kernels` hard; `plugin_options` wholesale; memory per A-N1 (D2) |
| A-M7: no fallback when a lock cannot start | supervisor history `[defaults + flags, lock + flags]`; design §4.4 corrected for installs (D2, step 6) |
| A-M8: two lookups could diverge; tools could fork a history | one build-hash lookup shared; tools write under the existing `model_id`; `check_locks` rejects two ids per build (D2, §7) |
| A-M9: version / ABI edits could invalidate fresh locks | ABI 1.0 and 0.1.0 in step 6 before locks; mechanism in step 2; `check_locks` (§7, §9) |
| A-M10: listing misses duck-typed hooks, config, vocabularies; Protocol additions | hooks declared; listing covers `EngineConfig`, options keys, field order, vocabularies, required / optional; locks as golden vectors; `op.*@major` frozen (§7) |
| A-M11: grammar off in locks | release locks select `grammar.xgrammar`; xgrammar a plain dependency (D3) |
| A-m1: `core_abi` optional, parsed after strict keys | required, checked first, testing manifests included (§7) |
| A-m2: `slot_abi` unused | dropped for v0 (§7) |
| A-m3: core estimate low | ≈ 100 lines (D2) |
| A-m4: dict flags replace the lock's | merged per key (D2) |
| A-m5: silent missing plans dir; image from a dirty tree | loud warning; `git archive` (D1, D2) |
| A-m6: first start may exceed `start_timeout` | measured in step 1 (D1) |
| A-m7: a lock naming a second checkpoint | not servable from the lock alone (D2) |
| A-m8: README installs `main` | clones the release tag (D8) |
| A-m9: lock resolves for any platform | `environments` restricted (D1) |
| A-m10: design §4.1's entry-point text wrong | corrected in step 6 (§7) |
| A-m11: `lock_overrides` raises on an op with two providers | checked on MTP locks in step 2; such an op left unpinned (D2) |
| A-m12: `git pull` under a running server | documented (D8) |
| B-B1: n = 3 underpowered; extension band ad hoc | single 0.70 rule, no band (n fixed at 10 in rev 3, B-M-1) (§4.5) |
| B-M1: branch point leaks the benchmark; mini-sglang history has a Mistral diff | squashed commit, tree and log grepped; depth-1 clone; parent-directory memory files excluded and recorded (§4.2) |
| B-M2: reference and fixtures differ by side | one `$REFERENCE`; fixtures sliced with original key names, key list recorded (§4.2, §4.3) |
| B-M3: check output, JIT, timeouts and cache rebuilds bias cost | ≤ 30-line summary; warmed caches; equal Bash timeouts and stated check time; second costing; check time off the wall cap (§4.2–§4.4) |
| B-M4: check not shown to discriminate or to admit a correct port | reference ports per engine with recorded margin; regressions pass unmodified; five plants (§4.3) |
| B-M5: shared venvs; static checks unnamed | per-run offline venvs; `$STATIC` per side verified at the branch point (§4.2) |
| B-M6: llama.cpp commit choice moves the gate | gate at the vendored pin; upstream a second column; KV format and batch flags recorded (§5.1, §5.2) |
| B-M7: gated metric unspecified; "fixed" undefined | gate statistic defined; fixed = re-measured ≥ 0.8× (§5.3) |
| B-m1: memorization favours the baseline | stated as a bias against Holon (§4.1) |
| B-m2: AGENTS.md's reading list | replaced by routing and a lean guide; documentation attributed separately (D8, §4.4) |
| B-m3: concept attribution post hoc; pivot re-test | path → concept map fixed before the pilot; full protocol after any pivot (§4.4, §4.7) |
| B-m4: infra failures, unpinned agent | re-run; dated model id, Claude Code version, thinking / effort, auto-compaction pinned (§4.2, §4.4) |
| B-m5: rounds, noise | ≥ 3 rounds, per-round values, A/A per class, "≈" under 2 sd (§5.2) |
| B-m6: speculation tables unlabeled; repetition | labelled by trace and sampling; repeated-output share recorded (§5.3) |
| B-m7: baseline settings unmatched | Holon at its lock; baselines matched to it, flags listed (§5.2) |
| B-m8: demo grading and reporting | pre-registered; tests restored; sampling recorded; misses published (D7) |
| B-m9: absolute tok/s compared across boards | ratios are the claim (§5.3) |
| C-M1: MTP rule ignores tail latency and KV cost | tail guards on a Claude-Code-shaped sampled trace; KV cost and `chunk_with_decode` stated; off switch documented (D3) |
| C-M2: docs written after the benchmark | documentation in step 4, inside the branch point (D8, §4.2) |
| C-M3: locks made in a different environment than users get | steps 7–9 in the step-1 environment, identity recorded; dev classes re-propagated in it (§6) |
| C-M4: verified locks can over-commit memory | `max_model_len` clamp; a lock that does not fit falls back (D2, A-N1) |
| C-M5: critical path, upstream risk | M0.7 starts at M0.6 Gate A; early smoke on 3090 / 5090 with upstream vs pin; mechanism in step 2 (§9) |
| C-M6: lock count and approvals | one recommended file per model per card (9 new); M0.6 files borrow on 5090 / 3090; approvals and sweep time measured (§6) |
| C-m1: rollback to an outdated lock | history of defaults + lock (D2) |
| C-m2: JIT build steps undeclared before the freeze | `[build] module` before ABI 1.0 (D1, §7) |
| C-m3: transformers a runtime dependency | declared; plan §1 corrected in step 6 (D1) |
| C-m4: DFlash2 tier | dropped (§5.2) |
| C-m5: demo version, compaction | Claude Code pinned; compaction noted (D7) |
| C-m6: README omissions | legibility result, models without locks, speculation off (D8) |
| C-m7: carried CI items | closed or deferred in the release status; CPU-only Actions with publishing (§8, step 10) |
| C-m8: core classifying option keys | `plugin_options` wholesale (D2) |

## 13. As built (steps 1–2)

Branch commits 55e30e5 (step 1), 290f7ae (step 2), 7eeb46a (review round 1), and the commits after them. The design
above is unchanged; what follows records results, deviations and decisions.

### 13.1 Step 1 — packaging (D1)

- **Environment.** The root project depends on the three runtime distributions and builds an empty wheel
  (`bypass-selection`). `[tool.uv]` has `default-groups = []`, `python-preference = "only-managed"` and one
  environment (Linux x86-64). `.python-version` is 3.12 and every distribution requires `>=3.12,<3.13`. `uv.lock` holds
  113 packages.
  - **Pins.** torch 2.11.0+cu130 (from the cu130 index, `explicit`), Triton 3.6.0, FlashInfer 0.6.12, xgrammar
    0.2.8 and **transformers 5.6.2** (released; the old venv ran 5.6.0.dev0).
  - **Version changes from the old venv.** These were floating, now locked: numpy 2.5.4, msgspec 0.22.0, fastapi
    0.143.0, uvicorn 0.54.0, safetensors 0.8.0, huggingface-hub 1.33.0 and nvidia-cutlass-dsl 4.8.0 (was 4.5.2).
    xgrammar was 0.2.3 in the old venv even though plugins declared 0.2.8.
  - **Extra nvcc.** FlashInfer's dependency `cuda-tile[tileiras]` pulls a pip `nvidia-cuda-nvcc` 13.2. It is unused:
    FlashInfer's JIT runs the toolkit's nvcc 13.0.
  - **The GDN L0 yardstick.** `plugins/tests/test_gdn_kernels.py` was pinned to 5.6.0.dev0, because its 2× bound is
    set against HF's torch path. 5.6.2 keeps that path: `l2norm` and the recurrent function are identical, and the
    chunked function only drops a mask the tril'd decay mask already applies (diffed). The L0 file passes at 5.6.2
    (56 passed), so its pin now reads 5.6.2. The milestone skill's "pinned to 5.6.0.dev0" line is outside this tree.
  - **The `dev` group** adds the test-only plugins, the judge, the tools, scipy (statistical tests) and jsonschema
    (`bench/conformance.py`).
- **Declared dependencies.** `manifest-deps` (`scripts/ci/check_manifest_deps.py`, in `run_all.sh`) scans each
  distribution's imports and each manifest's `[dependencies] python`. Against the old metadata it fails on every core
  import, so it bites. It found these undeclared imports, now declared:
  - core: torch, numpy, msgspec, pyzmq;
  - std: gguf, safetensors, **transformers** (the GGUF tokenizer — plan §1's "judge/test only" is the drift, corrected
    in step 6);
  - plugins: FlashInfer, Triton, transformers (the models' manifests list it), fastapi/uvicorn and xgrammar (the
    `http` / `grammar` extras are gone);
  - testing: safetensors, transformers;
  - judge: transformers, safetensors, huggingface-hub, accelerate, xgrammar, pandas and pyarrow (the constrained
    check).

  `run_all.sh` also runs `uv lock --check`.
- **`holon` console script** (`holon.core.cli:main`) with `serve` and `prebuild`.
- **`holon prebuild [--sm N …]`.** It reads every installed manifest's new `[build] module` and runs
  `python -m <package>.<module> --sm N` once per (plugin, SM). The SMs come from `holon.core.platform.visible_sms()`
  unless given. The core resolves no cache variable.
  - **The module is relative to the plugin's package**, like `entry`.
  - **Deviation from the brief's wording:** `FLASHINFER_CUDA_ARCH_LIST` is set by FlashInfer's build module in its
    own process (`kernels/flashinfer/prebuild.py`: 8.6 / 8.9 / 12.0f), as D1 specifies. It is not set by the core
    command, which names no plugin's variables.
  - **FlashInfer's module** builds every module the plugin can select: FA2 prefill for q ∈ {bf16, fp16}, KV ∈ {q, e4m3
    from sm_89}, head dims {64, 128, 256}, plus rope and silu_and_mul. That is 8 modules on sm_86 and 14 on sm_89 and
    sm_120.
  - **ggml_cuda's existing `prebuild` is its build module.** Its cache now follows `XDG_CACHE_HOME`, and `MAX_JOBS`
    caps its jobs.
  - **Bug found by the image build:** torch's extension loader read the GPUs for an arch list, so a host without a GPU
    could not build. The fix sets `TORCH_CUDA_ARCH_LIST` to the target. It changes no binary: the torch module is C++
    only.
  - **The `--sm N` CLI of build modules is ABI** (design §4.1's manifest example now shows `[build]`).
  - **On this host:** `holon prebuild` for sm_86 took 2 min 13 s (FlashInfer 47 s, ggml ≈ 85 s at `MAX_JOBS=24`).
- **Dockerfile.** `FROM nvidia/cuda:13.0.3-devel-ubuntu24.04` pinned by digest, build-essential, uv 0.11.1 by digest,
  the source from `git archive`, `uv sync --frozen` and `holon prebuild --sm 86 89 120`.
  - Every cache is pinned to a user-independent path: uv's Python and cache, XDG (ggml), FlashInfer, Triton, CuTe DSL
    and HF.
  - **It builds** (`git archive … | docker build --build-arg MAX_JOBS=16 -`, no GPU used). The prebuild for three SMs
    took 458 s, and the image is 14.5 GB. In the image, `holon prebuild --sm 86` finds everything built (0 s), and its
    ggml cache key equals the host's. Nothing was pushed.
- **AGENTS.md** Environment section now describes `uv sync --group dev` and the host requirements. The task-routed
  AGENTS.md is deferred to step 4.
- **Checks.**
  - **Fresh venv, no system site-packages** (`uv sync --group dev`): the CPU suite passes (586 passed, 482 GPU-only
    skipped).
  - **A user's `uv sync --frozen`** installs holon-workspace, holonlm, holonlm-std and holonlm-plugins only: no
    holonlm-testing, pytest or scipy. Its plugin set has no `testing.*`, and `hf` and `holon` are on its path.
  - **Cold start** (`bench/cold_start.py`, `evidence/cold_start_rtx-a5000_2026-10-14.json`, RTX A5000). This was
    `holon serve` after `holon prebuild`, with empty Triton and CuTe caches; no lock is current (§13.2), so defaults
    applied, and the 27B ran with its lock's settings as flags. FlashInfer and ggml compiled **nothing**. Triton
    compiled 12–40 kernels.

    | model | serving after | engine start | first completion |
    |---|---|---|---|
    | Qwen3-8B Q4_K_M | 20 s (86 s with a cold page cache, an earlier run) | 14 s | 0.34 s |
    | Qwen3-8B bf16 | 20 s | 14 s | 0.83 s |
    | Qwen3.8-27B UD-IQ3_S | 31 s | 24 s | 1.47 s |
    | Qwen3-30B-A3B Q4_K_M | 69 s (17 GB read) | 61 s | 0.30 s |

    Every start is ≤ 12% of `Policy.start_timeout` (600 s), which stays. Consumer cards are measured in step 10.
  - **Finding: CuTe DSL compiles too.** FlashInfer 0.6.12's `rmsnorm` / `fused_add_rmsnorm` run CuTe DSL kernels
    whenever `nvidia-cutlass-dsl` imports, and it is a hard dependency. They compile in memory at first use, about
    0.3 s per norm shape, with no disk cache (measured; `CUTE_DSL_CACHE_DIR` stays empty).

    So the strict reading of "a cold start compiles nothing but Triton" fails, by about 1 s per start. Making it hold
    means FlashInfer's CUDA-JIT norm module (prebuildable), and that changes the norm kernels' numerics: an L0 check
    and a propagation. This is left to the lead (§13.3).

### 13.2 Step 2 — D2 and the ABI mechanism at 0.1

- **Lookup.** `holon.core.plans.history_id(plans, identity)` finds the model id whose histories end in `@<build>`
  (one at most), else the identity. `find` → `nearest_class` (newest per class, own class first, then the same GPU,
  SM or toolchain; locks pinning other code are outdated). It returns the lock, `unverified` and its path.
  - `holon serve` and the tools' `_start` share it. The tools write under the existing history's model id: `plan_of`
    takes `model_id`.
  - `check_locks.py` re-reads every lock (digest and `plan_id` recomputed by `canonical_hash`; the committed locks
    are its golden vectors), rejects two model ids for one build, and keeps the no-`testing.*` rule. `--release` checks
    that every newest lock pins this tree's sources; today every lock fails it, as expected until M0.7's propagation.
- **Precedence** (`cli.serve_history`, `cli.layered`).
  - Flags parse with no defaults (`given_only`), so only flags given override the lock. Dict flags merge per key, and
    `key=` deletes a key. `--no-lock` serves defaults and flags; `--plans DIR` defaults to `plans/` of the tree holding
    the core, with a loud warning if absent.
  - The history is `[defaults + flags, lock + flags]`, and each entry carries its lock path (None for defaults).
  - Lock pins become `EngineConfig.kernel_preferences` (`lock_overrides`). They are soft: a preferred provider that
    cannot serve the site falls back to default priority, logged. `--kernels` stays hard. Both are unplanned
    (`plans.NOT_PLANNED`).
  - `lock_overrides` now leaves an op bound to two providers unpinned instead of raising. No committed lock has such
    an op and no MTP lock exists yet; the tools print the ops they leave unpinned.
  - `plugin_options` apply wholesale. A borrowed lock is used the same way.
- **Memory.**
  - **What stays the lock's.** `max_seqs` and `max_step_tokens` are taken as they are.
  - **Requested length.** `EngineConfig.clamp_context` defaults to false and only `holon serve` sets it (it is no
    longer a flag of the tools). The requested `max_model_len` is a flag's, else the lock's, else `auto`:
    min(the model plugin's new optional `trained_context` hook, 32768). Qwen3 / Qwen3-MoE / Qwen3.5 read
    `max_position_embeddings`, nested under `text_config` for Qwen3.5 HF configs: 40960 for Qwen3, 262144 for
    Qwen3.8-27B, so 32768.
  - **Pool floor.** With `clamp_context`, paged pools request Σ ceil(r / page) over `executor.profile_rows(cfg)`, the
    profile's rows.
  - **The clamp.** `Engine._clamp` sets `max_model_len` = min(requested, the smallest paged pool's tokens) after
    `ledger.materialize` and before `reset_padding` / `capture_graphs`. It pushes the value into the engine's own
    config copy (`Engine` now copies its config), `Scheduler.max_model_len` and `StateManager.limit` (block-table
    columns past the clamp hold only the null unit). Build-time readers keep the requested value.
  - **Tests.** The toy test shows a 64-token request that does not fit raising `OutOfUnits` without the clamp, and
    being served at 32 with it. The served plan's only diff is `engine:max_model_len: 64 -> 32`. Mutations of the
    clamp, the floor and the limit each fail it.
- **Plan check.** `plan_of` (minus git), `unit_pin`, `diff`, `NOT_PLANNED` and `uv_lock` moved to `holon.core.plans`.
  - **`uv_lock`** hashes the sorted (name, version, source) of the runtime distributions' closure in `uv.lock`, extras
    included: torch reaches the CUDA libraries through them, a review finding. Dev-only pins move nothing.
  - **What the engine process reports.** It computes its plan under the lock's model id and commit (`plans.served`,
    which reports an error instead of failing a start). It sends `plan_id`, the lock path and the diff in its `ready`
    frame. Each start logs them, and `/metrics` shows them (`Client.stats` adds the ready report).
  - **Lock paths.** `--no-supervised` gets the lock path too. The bench's HTTP backends record what was served.
- **Fallback.**
  - The `failed` frame is (kind, error), kinds "plugin" / "memory" / "error" (`ipc.FAILED_KINDS`). A start failing
    with `PluginError` or `OutOfUnits` rolls back (or gives up) at once; others after `rollback_crashes` tries.
  - Manifest errors at discovery (another ABI) are `PluginError`.
  - **Tests.** The rollback test checks the starts: 4 transient failures, then the good config, and one plugin-error
    start before giving up.
  - **Kept as it is:** the frontend keeps the newest entry's `plugin_options` after a rollback.
- **ABI 0.1 mechanism.** `holon.api.ABI = "0.1"`.
  - **`core_abi`** (e.g. `">=0.1,<0.2"`, `abi_allows`) is required in all 33 manifests, the test fixtures' included.
    It is checked before the strict key parse. `slot_abi` is removed (design §4.1 / §4.5 updated, and §4.1's
    entry-point text fixed).
  - **Optional hooks.** The 22 hooks the core looks up by name, plus the new trained-context hook, are 23 small
    optional Protocols in `holon.api.hooks`; the 23rd literal of the 23 in §7, `parent`, is allow-listed. The optional
    members of `Mixer`, `Drafter`, `StateKind` and `DraftPolicy` moved there with their docs. Only `Rewindable` stays
    exported from `holon.api` itself.
  - **String vocabularies** are now declared: `FINISH_REASONS`, `ERROR_KINDS` and `KERNEL_PROPS` (now enforced by
    `KernelSpec`).
  - **`scripts/ci/check_abi.py`** (in `run_all.sh`) renders `core/abi.txt`, which covers:
    - names and signatures, dataclass fields by index with types and defaults;
    - Protocol members, required or optional, and who implements each Protocol;
    - `EngineConfig`, the `BuildCtx.options` keys and the manifest schema;
    - the vocabularies, the IPC frame and failed kinds.

    The `op.*` / `unit.*` contracts first-party plugins provide are listed per distribution
    (`std/contracts.txt`, `plugins/contracts.txt`; §13.4).

    It fails on drift and on undeclared `hasattr` / `getattr` literals (a planted one fails). `--write` requires the
    bump: an addition is minor; a removal, a change, a member added to a required plugin-implemented Protocol or a
    field added without a default is major (below 1.0, the minor number). The rule is tested, and CI holds it against
    the base ref (§13.4). The listing was regenerated inside steps 1–2, before any lock.
- **End to end on the GPU.** A current lock for Qwen3-8B Q4_K_M on rtx-a5000 was written by the tools into a
  temporary directory only. The file was then served under another path and name (a `--local-dir`-style copy, so the
  identity is `Qwen3-8B-Q4_K_M@120307ba529eb243`, not the hub's).
  - `holon serve --plans <tmp> --plugin-options x.test=1` started from that lock, as newest history entry (config 1).
  - The engine reported its `plan_id` with the only diff `engine:plugin_options: {} -> {'x.test': '1'}`, the flag
    given. `/metrics` shows the same.
- **Review.** One reviewer (correctness, architecture, tests) found 3 major, 4 minor and 3 nit issues in round 1, all
  fixed in 7eeb46a:
  - **Majors:** `uv_lock` ignored extras; `check_abi` blocked minor bumps after 1.0; a required plugin member passed
    as minor.
  - **Minors:** untested plan report and ABI discovery; a plan report could fail a start; bench backends now serving
    from locks unrecorded; the tools' lookup duplicated.
  - **Nits:** stray `.lock` files, clamp/preference flags in the tools, the parent's CUDA context (none, checked on
    a GPU).

  Round 2 reported one nit (the cold-start record's command omitted `--flags`), fixed. No findings remain.
- **Tests added or changed.**
  - `core/tests/test_plans.py`: the lookup / precedence / merge-delete / `auto` case (6 parameters), the clamp and
    soft-pin engine test, and the `uv_lock` closure.
  - `core/tests/test_api.py`: `core_abi` cases and the bump rule.
  - `core/tests/test_plugins.py`: another ABI at discovery.
  - `plugins/tests/test_supervisor.py`: the rollback test counts starts and covers the plugin-error kind; history
    entries became tuples.
  - The tools' tests moved to `plans.diff` and preferences. The duplicate `lock_overrides` test in core was removed.
- **Suite at the end of step 2** (RTX A5000 ×2): `pytest core std plugins judge testing tools` gave 1031 passed and
  55 skipped (models or GPUs absent here) in 14 min 50 s, in the uv venv; `scripts/ci/run_all.sh` is green. The
  slowest test is still the rollback test (41 s, from 66 s: the plugin-error case gives up after one start).
- **Core size:** 4134 code lines (3864 before: +270 against the ≈ 150 budgeted).
  - About 40 of them moved from `holon_tools.locks` (`plan_of`, `diff`, `unit_pin`).
  - About 55 are the hook Protocols §7 asks for.
  - The rest are D2 (about 110), `holon prebuild` (about 15) and the ABI check in manifests (about 15).
- **Not done here, as planned:** grammar in locks (M0.7's propagation); propagating or writing any lock; the plan §1 and
  design §4.4 corrections (step 6).
- **Every lock is outdated by this branch** (the core's and every manifest's `src` changed). Until M0.7's
  propagation, `holon serve` finds no current lock and serves defaults, logged as "no current lock … serving defaults".
  - **With defaults, the 27B does not start on a 24 GB card:** 64 sequences' recurrent state leave no room for KV, so
    `OutOfUnits` gives up at once. It needs its lock's flags (`--max-seqs 4 --max-step-tokens 2048 --plugin-options
    kv.format=q8_0 recurrent.checkpoints=4`).

### 13.3 For the lead

1. **CuTe DSL norms at a cold start (§13.1):** accept and document ≈ 0.3 s per norm shape as the one non-Triton
   compile, or move FlashInfer's norms to the prebuildable CUDA-JIT module (a numerics change: L0, then M0.7's
   propagation).
2. **Core size:** +270 lines against ≈ 150 (breakdown above). The cap of 6000 is far off.
3. **Lockless serving until M0.7:** models whose defaults do not fit (the 27B at `max_seqs` 64) need flags until
   their locks are propagated. The supervisor reports it at once rather than after 4 × `start_timeout`.

**Lead decisions (2026-10-14):**
1. **Escalated to the owner (acceptance changed after measuring), batched by the lead.** Pending the owner: **the
   CuTe DSL norms' cold-start compile is accepted and documented** (≈ 1 s per start, about 0.3 s per norm shape, in memory). Switching to FlashInfer's prebuildable CUDA norm module would change numerics (L0 plus a propagation) for a second per start. Acceptance 4's "compiles nothing but Triton" reads "compiles nothing but Triton and FlashInfer's CuTe DSL norms".
2. **The core's +270 lines over the ≈ 150 budget is accepted** (4134 / 6000). About 40 of them moved from the tools, and about 55 are the hook Protocols §7 asks for.
3. **Revised after the panel's code review (reverses the first decision 3): `holon serve` without a lock serves at
   `max_seqs` 8.** The defaults entry of the history (no lock found, `--no-lock`, and the fallback) gets a serve-only
   default `max_seqs` of 8 (`cli.SERVE_MAX_SEQS`), logged and overridden by `--max-seqs`, applied like
   `--max-model-len auto`. Tools, judge and bench keep `EngineConfig`'s 64. So design §4.4's "never refuses to serve"
   holds for the 27B on a 24 GB card: checked on the RTX A5000, `holon serve` of Qwen3.8-27B UD-IQ3_S with no lock and
   no flags starts in 76 s at `max_seqs` 8 and the full 32768 context, and answers a chat completion. A lock's
   `max_seqs` stays exact (D2).

### 13.4 Panel code review of steps 1–2 (two reviewers, scope and serving)

Every finding is fixed, each with a test that fails before it where code changed.

- **C-S1 (major): plugin contracts were under the core's ABI.** `core/abi.txt` listed every `op.*` / `unit.*`
  contract, so a new std unit would have forced a `holon.api.ABI` bump: a core edit that re-propagates every lock
  (design §4.5, invariant 9).
  - Each distribution now lists its own contracts (`std/contracts.txt`, `plugins/contracts.txt`), rendered and
    checked by `check_abi.py` under their own rule. A new contract is free; one removed or moved to another provider
    needs the same contract at a higher major (`contract_error`, tested).
  - `core/abi.txt` keeps `holon.api`, the hooks, `EngineConfig`, the options keys, the manifest schema and the
    vocabularies.
  - **Every provided resource is a contract** except implementations and what a model plugin builds (`kernel.*`,
    `arch.*`, `model.*`; a last-round minor). Besides `op.*` and `unit.*`, that is `state_kind.*` (`paged@1`,
    `recurrent@1`), `grammar.constraint@1`, the weight `transform.*`, and the slot interfaces (`drafter.*`,
    `draft_policy.*`, `frontend.*`, `sampler.*`, `verifier.*`, `weight_source.*`). The listings were regenerated
    (additions only) and the core ABI stays 0.1; the test asserts the coverage.
- **C-S2 (major): the no-lock 27B start.** The defaults entry now serves at `max_seqs` 8 (§13.3, lead decision 3
  revised). Tested in the `serve_history` case and checked on the A5000.
- **Scope minors.**
  - Lead decision 1 is marked escalated.
  - A toy CPU test for `holon prebuild` (`core/tests/test_plugins.py`): fixture plugins whose package-relative build
    modules record their `--sm`, one subprocess per (plugin, SM), a plugin without `[build]` not run, and a non-zero
    exit naming the failed (plugin, SM) after the rest ran.
  - The units' `adopt` hook is renamed `adopt_weights` (`holon.api.hooks.AdoptsWeights`; std's ops and MoE units,
    `build.load_weights`, design §5.5), so `adopt` is the state kinds' alone.
- **Serving minors.**
  1. **CI enforces the bump.** `run_all.sh` runs `check_abi.py --base <its base ref>`, which holds each listing
     (`core/abi.txt` and both contracts listings) to its rule against the base's committed one. Main has none yet, so
     the first one lands with this branch; against this branch's own earlier commit, the reformatted listing fails as
     it should.
  2. **Dataclass fields are rendered with their index**, so an insertion is a change (major). A field added without a
     default is major; frozenset defaults render sorted. Tested.
  3. **`cli.find_lock` catches `ValueError`** (two model ids for one build, a lock edited by hand), warns naming the
     history or file, and serves defaults. Two cases were added to the `serve_history` test.
  4. **The failed kind "plugin" is the resolver's alone.** A plugin's import or registration failure is now
     `PluginLoadFailed` (a `PluginError`), sent as "error" and retried, since it may be transient. Tested through
     `ipc`'s encoding.
  5. **Dockerfile.** After the prebuild, the cache directory (XDG, FlashInfer, Triton, CuTe) and `HF_HOME` are made
     writable by any user (`chmod -R a+rwX`, in the same layer), so `docker run --user` works. Checked: the image
     rebuilds. As uid 12345, `holon prebuild --sm 86` (which takes the GGUF build lock in the cache) succeeds, and so
     does a write into each cache directory and `HF_HOME`.

After the fixes the full GPU suite (RTX A5000 ×2) gave 1037 passed and 55 skipped in 15 min 42 s, and
`run_all.sh` is green. The core is 4144 code lines.

The listings changed again inside the branch (reformatted, contracts split). The ABI stays 0.1: no lock, release or
main commit refers to it yet.


**ABI bump policy (lead, 2026-10-14; scope reviewer's round-2 minor):** M0.6's Gate A merges into `main` first. Steps 1–2 then rebase onto it: M0.6's four unit `adopt(self, ctx, name, q, index=None)` definitions are renamed `adopt_weights`, `AdoptsWeights` takes the `index` parameter, and the listings land on `main` at ABI 0.1 for the first time, so nothing is compared yet. M0.7 builds on that `main`; its removal of `SchedulePolicy.mix_phases` is one bump to 0.2 inside M0.7, and its single propagation covers the manifests' `core_abi` edit. Step 6 sets 1.0.

**Notes for later steps (lead, last round on steps 1–2):**
- **Merges that bring units (M0.6, M0.7) rename unit `adopt` to `adopt_weights`**: in the units, the core's call and
  the tests. `check_abi`'s literal scan cannot catch a reintroduced `getattr(unit, "adopt")`, because the state kinds
  still declare `adopt` (`RestoresPayloads`).
- **The README's `docker run --user` example passes `-e HOME=<a writable path>`**, so the weight sources' file-hash
  cache (under the home directory) persists across starts.
- **The GitHub Actions job that comes with publishing fetches the base ref** (e.g. `fetch-depth: 0`, or a fetch of
  `main`). Otherwise `check_abi.py --base` finds no base and skips the bump check.


### 13.5 Step 3 — baselines, smoke and tier-1 bars on main's engine code (2026-10-14)

**Built** (`bench/`, no core, std or plugin change; the engine code is local main's, 2ef170d):
- `setup_baselines.sh` pins and builds every baseline under `bench/.envs/`:

  | baseline | pin | build |
  |---|---|---|
  | llama.cpp, gate | `cbe28bb3` (M0.6 Gate A's vendored pin) | `llama-bench` and `llama-server`, one build for sm_86 / 89 / 120a, CUDA 13.0 |
  | llama.cpp, upstream | `df18bee1` (HEAD on 2026-10-14; re-pinned at step 8) | the same |
  | SGLang | release 0.5.21 | own venv: torch 2.13.0 + cu130; it pins pre-release kernels, so `--prerelease=allow` |
  | FreeToken | `74a38777` | own venv: torch 2.14.1 + cu130, `[accel]`, extension built against that torch |

- `backends/sglang_http.py`: SGLang's `/generate` with `input_ids` (`--skip-tokenizer-init`), streamed.
- A sampling temperature (`run.py --temperature`) reaches every backend that samples. The other samplers (top-k, top-p, min-p) are sent as "off" in each engine's own spelling, because their defaults differ: llama.cpp uses 40 / 0.95 / 0.05, and FreeToken and SGLang take theirs from `generation_config.json`.
- **Baselines matched to a lock** (`run.py --lock`):
  - Holon runs the lock's `[engine]` settings. Hand-given flags override them, and dict flags merge per key as D2's `holon serve` will. Kernel pins are not applied, so the default selection runs (D2 makes pins soft in step 2).
  - Every backend gets the lock's `max_seqs` and step budget.
  - The baselines get the lock's KV format and Holon's measured KV tokens, which come from the first in-process `holon` run. A baseline is skipped, with a recorded reason, until those tokens are known.
  - A baseline without the lock's KV format refuses to run: llama.cpp has no FP8 KV, SGLang has no q8_0, and FreeToken matches only the model dtype.
  - Flags:
    - `llama-server -ngl 99 -fa on -kvu -np <max_seqs> -c <KV tokens> -b <budget> -ub 512 -ctk/-ctv <format>`;
    - SGLang `--max-running-requests --chunked-prefill-size --max-total-tokens --kv-cache-dtype --mem-fraction-static 0.9`;
    - FreeToken adds `--num-tokens`.
  - The record's `matched` block lists what was matched. A failed backend's record carries its reason: the log's first exception, with paths cut to file names.
- `bars.py` produces the gate's bars (§5.3):
  - Holon's in-process `llama_bench.py` runs against `llama-bench` at the gate pin and at upstream, ABAB in fresh processes, at the lock's settings: `-ctk/-ctv`, `-b` = the lock's budget, `-ub 512`.
  - A round's value is the median of repetitions 2–5. The statistic is the median over rounds.
  - Each record names the ggml commit Holon ran (`kernels.ggml_cuda`'s vendored tree) and each llama.cpp build's commit.
  - It refuses locks with speculation, since llama-bench drafts nothing.
- `report.py` renders the tables below from the records, each table linking its record.
- Code review: one reviewer, six rounds, clean at 632eeaa. Its findings were:
  - FreeToken was never matched.
  - KV-token matching depended on backend order.
  - Hand-given dict flags replaced the lock's.
  - Sampler defaults differed between engines.
  - Error strings leaked command lines; failure reasons are now scrubbed of paths, host names and driver versions.
  - The gate column could pass silently.

**Settings.**
- Neither the RTX 5090 nor the RTX 3090 has a GGUF lock yet; they are the **R** cells of step 7. So every cell runs the nearest class's lock, marked "borrowed" in the records:
  - 8B Q4_K_M: rtx-5080 lock 6;
  - 30B-A3B Q4_K_M and 27B UD-Q4_K_M: rtx-4090 locks 4 and 3;
  - 8B bf16: the rtx-5090 lock 6 (outdated, B9) on the 5090 and the rtx-4090 lock 9 on the 3090.
- None of these locks selects grammar yet (D3 lands at M0.7's propagation).
- Bars: 3 rounds per cell. Smoke: one round of `decode_b1` and `decode_b8` on prompts cut from judge prompt set v1 (`v1-qwen3_5` for the 27B).
- MTP smoke: Holon runs `drafters.mtp` with k ≤ 3; llama.cpp runs `--spec-type draft-mtp --spec-draft-n-max 3`.
- Holon at commit 8bf7bca (bench code only beyond main). The M0.6-file smoke and the 3090 27B re-measure ran at c9a6f0c, whose only change is that failure reasons are recorded.

**Tier-1 bars** (Holon / llama.cpp; gate ≥ 0.8× the pin; "≈": within 2 sd of the per-round ratios):

NVIDIA GeForce RTX 3090; Holon's ggml at 51ce9c11

| file (record) | lock | metric | Holon t/s | llama.cpp gate (cbe28bb3c) t/s | Holon / gate | llama.cpp upstream (df18bee11) t/s | Holon / upstream |
|---|---|---|---|---|---|---|---|
| [qwen3-30b-a3b-q4_k_m](../evidence/bars_release-qwen3-30b-a3b-q4_k_m_rtx-3090_2026-10-14.json) | rtx-4090 (borrowed) | pp512 | 5083.3 | 4637.5 | 1.10 | 4650.4 | 1.09 |
| [qwen3-30b-a3b-q4_k_m](../evidence/bars_release-qwen3-30b-a3b-q4_k_m_rtx-3090_2026-10-14.json) | rtx-4090 (borrowed) | pp512 @ d8192 | 3258.9 | 3288.7 | 0.99 | 3303.1 | 0.99 |
| [qwen3-30b-a3b-q4_k_m](../evidence/bars_release-qwen3-30b-a3b-q4_k_m_rtx-3090_2026-10-14.json) | rtx-4090 (borrowed) | tg128 | 209.2 | 212.2 | 0.99 | 212.4 | 0.98 |
| [qwen3-30b-a3b-q4_k_m](../evidence/bars_release-qwen3-30b-a3b-q4_k_m_rtx-3090_2026-10-14.json) | rtx-4090 (borrowed) | tg128 @ d8192 | 175.5 | 176.3 | 1.00 | 176.5 | 0.99 |
| [qwen3.8-27b-ud-q4_k_m-r2](../evidence/bars_release-qwen3.8-27b-ud-q4_k_m-r2_rtx-3090_2026-10-14.json) | rtx-4090 (borrowed) | pp512 | 886.5 | 1567.4 | **≈ 0.57 (< 0.8)** | 1576.9 | ≈ 0.56 |
| [qwen3.8-27b-ud-q4_k_m-r2](../evidence/bars_release-qwen3.8-27b-ud-q4_k_m-r2_rtx-3090_2026-10-14.json) | rtx-4090 (borrowed) | pp512 @ d8192 | 1513.1 | 1432.5 | 1.06 | 1434.8 | 1.05 |
| [qwen3.8-27b-ud-q4_k_m-r2](../evidence/bars_release-qwen3.8-27b-ud-q4_k_m-r2_rtx-3090_2026-10-14.json) | rtx-4090 (borrowed) | tg128 | 41.9 | 23.4 | 1.79 | 29.5 | 1.42 |
| [qwen3.8-27b-ud-q4_k_m-r2](../evidence/bars_release-qwen3.8-27b-ud-q4_k_m-r2_rtx-3090_2026-10-14.json) | rtx-4090 (borrowed) | tg128 @ d8192 | 39.2 | 26.8 | 1.47 | 27.3 | 1.44 |
| [qwen3.8-27b-ud-q4_k_m-r3](../evidence/bars_release-qwen3.8-27b-ud-q4_k_m-r3_rtx-3090_2026-10-14.json) | rtx-4090 (borrowed) | pp512 | 1447.9 | 1387.9 | 1.04 | 1369.1 | 1.06 |
| [qwen3.8-27b-ud-q4_k_m-r3](../evidence/bars_release-qwen3.8-27b-ud-q4_k_m-r3_rtx-3090_2026-10-14.json) | rtx-4090 (borrowed) | pp512 @ d8192 | 1276.9 | 1251.8 | ≈ 1.02 | 1251.0 | ≈ 1.02 |
| [qwen3.8-27b-ud-q4_k_m-r3](../evidence/bars_release-qwen3.8-27b-ud-q4_k_m-r3_rtx-3090_2026-10-14.json) | rtx-4090 (borrowed) | tg128 | 44.2 | 42.5 | 1.04 | 42.4 | 1.04 |
| [qwen3.8-27b-ud-q4_k_m-r3](../evidence/bars_release-qwen3.8-27b-ud-q4_k_m-r3_rtx-3090_2026-10-14.json) | rtx-4090 (borrowed) | tg128 @ d8192 | 43.0 | 41.5 | 1.04 | 41.5 | 1.04 |
| [qwen3.8-27b-ud-q4_k_m](../evidence/bars_release-qwen3.8-27b-ud-q4_k_m_rtx-3090_2026-10-14.json) | rtx-4090 (borrowed) | pp512 | 685.6 | 1565.0 | **0.44 (< 0.8)** | 1572.5 | 0.44 |
| [qwen3.8-27b-ud-q4_k_m](../evidence/bars_release-qwen3.8-27b-ud-q4_k_m_rtx-3090_2026-10-14.json) | rtx-4090 (borrowed) | pp512 @ d8192 | 1510.2 | 1434.3 | 1.05 | 1435.1 | 1.05 |
| [qwen3.8-27b-ud-q4_k_m](../evidence/bars_release-qwen3.8-27b-ud-q4_k_m_rtx-3090_2026-10-14.json) | rtx-4090 (borrowed) | tg128 | 38.5 | 21.9 | 1.75 | 28.3 | 1.36 |
| [qwen3.8-27b-ud-q4_k_m](../evidence/bars_release-qwen3.8-27b-ud-q4_k_m_rtx-3090_2026-10-14.json) | rtx-4090 (borrowed) | tg128 @ d8192 | 36.2 | 24.1 | 1.50 | 24.7 | 1.47 |
| [qwen3-8b-q4_k_m](../evidence/bars_release-qwen3-8b-q4_k_m_rtx-3090_2026-10-14.json) | rtx-5080 (borrowed) | pp512 | 5421.6 | 5236.0 | 1.03 | 5236.8 | 1.03 |
| [qwen3-8b-q4_k_m](../evidence/bars_release-qwen3-8b-q4_k_m_rtx-3090_2026-10-14.json) | rtx-5080 (borrowed) | pp512 @ d8192 | 3653.4 | 3823.7 | 0.95 | 3826.9 | 0.95 |
| [qwen3-8b-q4_k_m](../evidence/bars_release-qwen3-8b-q4_k_m_rtx-3090_2026-10-14.json) | rtx-5080 (borrowed) | tg128 | 143.7 | 141.1 | ≈ 1.02 | 140.2 | ≈ 1.02 |
| [qwen3-8b-q4_k_m](../evidence/bars_release-qwen3-8b-q4_k_m_rtx-3090_2026-10-14.json) | rtx-5080 (borrowed) | tg128 @ d8192 | 120.7 | 118.1 | ≈ 1.02 | 117.2 | 1.03 |

NVIDIA GeForce RTX 5090; Holon's ggml at 51ce9c11

| file (record) | lock | metric | Holon t/s | llama.cpp gate (cbe28bb3c) t/s | Holon / gate | llama.cpp upstream (df18bee11) t/s | Holon / upstream |
|---|---|---|---|---|---|---|---|
| [qwen3-30b-a3b-q4_k_m](../evidence/bars_release-qwen3-30b-a3b-q4_k_m_rtx-5090_2026-10-14.json) | rtx-4090 (borrowed) | pp512 | 12544.5 | 11553.3 | 1.09 | 11541.1 | 1.09 |
| [qwen3-30b-a3b-q4_k_m](../evidence/bars_release-qwen3-30b-a3b-q4_k_m_rtx-5090_2026-10-14.json) | rtx-4090 (borrowed) | pp512 @ d8192 | 8428.7 | 8972.9 | 0.94 | 8961.6 | 0.94 |
| [qwen3-30b-a3b-q4_k_m](../evidence/bars_release-qwen3-30b-a3b-q4_k_m_rtx-5090_2026-10-14.json) | rtx-4090 (borrowed) | tg128 | 338.0 | 361.7 | 0.93 | 361.8 | 0.93 |
| [qwen3-30b-a3b-q4_k_m](../evidence/bars_release-qwen3-30b-a3b-q4_k_m_rtx-5090_2026-10-14.json) | rtx-4090 (borrowed) | tg128 @ d8192 | 289.9 | 294.7 | 0.98 | 295.0 | 0.98 |
| [qwen3.8-27b-ud-q4_k_m](../evidence/bars_release-qwen3.8-27b-ud-q4_k_m_rtx-5090_2026-10-14.json) | rtx-4090 (borrowed) | pp512 | 4625.6 | 4138.1 | 1.12 | 4146.7 | 1.12 |
| [qwen3.8-27b-ud-q4_k_m](../evidence/bars_release-qwen3.8-27b-ud-q4_k_m_rtx-5090_2026-10-14.json) | rtx-4090 (borrowed) | pp512 @ d8192 | 3783.3 | 3937.8 | ≈ 0.96 | 3928.9 | ≈ 0.96 |
| [qwen3.8-27b-ud-q4_k_m](../evidence/bars_release-qwen3.8-27b-ud-q4_k_m_rtx-5090_2026-10-14.json) | rtx-4090 (borrowed) | tg128 | 84.8 | 82.3 | 1.03 | 82.3 | 1.03 |
| [qwen3.8-27b-ud-q4_k_m](../evidence/bars_release-qwen3.8-27b-ud-q4_k_m_rtx-5090_2026-10-14.json) | rtx-4090 (borrowed) | tg128 @ d8192 | 82.5 | 78.6 | 1.05 | 78.6 | 1.05 |
| [qwen3-8b-q4_k_m](../evidence/bars_release-qwen3-8b-q4_k_m_rtx-5090_2026-10-14.json) | rtx-5080 (borrowed) | pp512 | 14717.0 | 14185.3 | 1.04 | 14140.0 | 1.04 |
| [qwen3-8b-q4_k_m](../evidence/bars_release-qwen3-8b-q4_k_m_rtx-5090_2026-10-14.json) | rtx-5080 (borrowed) | pp512 @ d8192 | 9964.5 | 10951.2 | 0.91 | 10924.4 | 0.91 |
| [qwen3-8b-q4_k_m](../evidence/bars_release-qwen3-8b-q4_k_m_rtx-5090_2026-10-14.json) | rtx-5080 (borrowed) | tg128 | 244.3 | 253.2 | 0.96 | 253.2 | 0.96 |
| [qwen3-8b-q4_k_m](../evidence/bars_release-qwen3-8b-q4_k_m_rtx-5090_2026-10-14.json) | rtx-5080 (borrowed) | tg128 @ d8192 | 206.0 | 207.8 | 0.99 | 207.8 | 0.99 |

**Cells under 0.8× (flagged, not fixed):**

- **Qwen3.8-27B UD-Q4_K_M, pp512 at depth 0, RTX 3090: 0.44× the pin, and 0.57× on a re-measure on the same desktop (B).** This is a measurement artifact, not a kernel gap.
  - **Desktop B.** During the re-measure, the clock log (nvidia-smi, every 500 ms, kept local) shows its GPU under software thermal slowdown in 43% of loaded samples, with SM clocks swinging between ≈ 300 and 1.9 GHz.
  - **Both engines were hit.** Within each process, Holon's pp512 fell from ≈ 1,690 to ≈ 550 t/s. llama-bench's tg128 fell from 44 to ≈ 22 t/s, and that fall is what turned the tg ratios into 1.5–1.8×.
  - **Desktop C.** The same cell on another RTX 3090 desktop (C) measured pp512 1.04×, pp512 at 8K 1.02×, tg128 1.04× and tg128 at 8K 1.04×, with every per-round ratio within 0.01 of the median. That GPU also throttled thermally (70% of samples) but held steady clocks (p10 / p50 1.40 / 1.49 GHz).
  - **A5000 cross-check.** On an RTX A5000 (sm_86, development host), Holon's own pp512 is flat across repetitions (1,211 → 1,183 t/s).
  - All three 27B records on the 3090 are kept above (`-r2` is the re-measure on B, `-r3` the run on C). The 8B and 30B cells on B are steady, with per-round ratios within 0.01. The one exception is the 8B's third llama-bench round of tg128 (1.06× at depth 0, 1.32× at 8K), which shows the same swings; the median discards it.
- No other cell is under 0.8×. Below the predicted bands (§5.3) but above 0.8×:
  - pp512 at 8K: 0.91–0.96× on the 5090 and on the 3090's 8B;
  - tg128 on the 5090: 8B 0.96×, 30B-A3B 0.93×.
  - These are M0.7's step-0 rows to re-measure at the `cbe28bb3` pin. Note that Holon still ran ggml `51ce9c11` here.

**Smoke matrix** (feasibility: one round, so the numbers are not claims; "—" with the recorded reason):

| record | model | GPU | backend | decode_b1: TTFT / TPOT p50 ms | decode_b8: TTFT / TPOT p50 ms |
|---|---|---|---|---|---|
| [release-smoke-qwen3-30b-a3b-mxfp4_moe](../evidence/bench_release-smoke-qwen3-30b-a3b-mxfp4_moe_rtx-3090_2026-10-14.json) | Face314/Qwen3-30B-A3B-Instruct-2507-MXFP4_MOE@68d644f2fca0/Qwen3-30B-A3B-Instruct-2507-MXFP4_MOE.gguf | rtx-3090 | holon | — (exit code 1: holon.core.plugins.PluginError: no kernel for linear_experts@1 on cuda sm 86 with {'layout': 'ggml.mxfp4/gguf.native', 'experts': 128, 'in': 768, 'out': 2048, 'top_k': 8, 'rows_per_token': 8, 'glu': 'none', 'act_dtype': 'float32', 'out_dtype': 'float32'}) |  |
| [release-smoke-qwen3-30b-a3b-mxfp4_moe](../evidence/bench_release-smoke-qwen3-30b-a3b-mxfp4_moe_rtx-3090_2026-10-14.json) | Face314/Qwen3-30B-A3B-Instruct-2507-MXFP4_MOE@68d644f2fca0/Qwen3-30B-A3B-Instruct-2507-MXFP4_MOE.gguf | rtx-3090 | llamaserver | 90.83 / 5.54 | 641.16 / 20.25 |
| [release-smoke-qwen3-30b-a3b-mxfp4_moe](../evidence/bench_release-smoke-qwen3-30b-a3b-mxfp4_moe_rtx-5090_2026-10-14.json) | Face314/Qwen3-30B-A3B-Instruct-2507-MXFP4_MOE@68d644f2fca0/Qwen3-30B-A3B-Instruct-2507-MXFP4_MOE.gguf | rtx-5090 | holon | — (exit code 1: holon.core.plugins.PluginError: no kernel for linear_experts@1 on cuda sm 120 with {'layout': 'ggml.mxfp4/gguf.native', 'experts': 128, 'in': 768, 'out': 2048, 'top_k': 8, 'rows_per_token': 8, 'glu': 'none', 'act_dtype': 'float32', 'out_dtype': 'float32'}) |  |
| [release-smoke-qwen3-30b-a3b-mxfp4_moe](../evidence/bench_release-smoke-qwen3-30b-a3b-mxfp4_moe_rtx-5090_2026-10-14.json) | Face314/Qwen3-30B-A3B-Instruct-2507-MXFP4_MOE@68d644f2fca0/Qwen3-30B-A3B-Instruct-2507-MXFP4_MOE.gguf | rtx-5090 | llamaserver | 37.91 / 3.33 | 288.87 / 10.11 |
| [release-smoke-qwen3-30b-a3b-nvfp4](../evidence/bench_release-smoke-qwen3-30b-a3b-nvfp4_rtx-3090_2026-10-14.json) | nvidia/Qwen3-30B-A3B-NVFP4@2538ded2a4ed | rtx-3090 | holon | — (exit code 1: holon.core.plugins.PluginError: no weight rule maps checkpoint tensor 'model.layers.0.mlp.experts.0.down_proj.input_scale') |  |
| [release-smoke-qwen3-30b-a3b-nvfp4](../evidence/bench_release-smoke-qwen3-30b-a3b-nvfp4_rtx-3090_2026-10-14.json) | nvidia/Qwen3-30B-A3B-NVFP4@2538ded2a4ed | rtx-3090 | sglang_http | 53.82 / 5.29 | 235.79 / 12.41 |
| [release-smoke-qwen3-30b-a3b-nvfp4](../evidence/bench_release-smoke-qwen3-30b-a3b-nvfp4_rtx-3090_2026-10-14.json) | nvidia/Qwen3-30B-A3B-NVFP4@2538ded2a4ed | rtx-3090 | freetoken_http | — (exit code 1: RuntimeError: chunk expects at least a 1-dimensional tensor) |  |
| [release-smoke-qwen3-30b-a3b-nvfp4](../evidence/bench_release-smoke-qwen3-30b-a3b-nvfp4_rtx-5090_2026-10-14.json) | nvidia/Qwen3-30B-A3B-NVFP4@2538ded2a4ed | rtx-5090 | holon | — (exit code 1: holon.core.plugins.PluginError: no weight rule maps checkpoint tensor 'model.layers.0.mlp.experts.0.down_proj.input_scale') |  |
| [release-smoke-qwen3-30b-a3b-nvfp4](../evidence/bench_release-smoke-qwen3-30b-a3b-nvfp4_rtx-5090_2026-10-14.json) | nvidia/Qwen3-30B-A3B-NVFP4@2538ded2a4ed | rtx-5090 | sglang_http | — (exit code 1: AttributeError: 'FusedMoE' object has no attribute 'g1_scale_c') |  |
| [release-smoke-qwen3-30b-a3b-nvfp4](../evidence/bench_release-smoke-qwen3-30b-a3b-nvfp4_rtx-5090_2026-10-14.json) | nvidia/Qwen3-30B-A3B-NVFP4@2538ded2a4ed | rtx-5090 | freetoken_http | — (exit code 1: RuntimeError: chunk expects at least a 1-dimensional tensor) |  |
| [release-smoke-qwen3-30b-a3b-q4_k_m](../evidence/bench_release-smoke-qwen3-30b-a3b-q4_k_m_rtx-3090_2026-10-14.json) | unsloth/Qwen3-30B-A3B-GGUF@d5b1d57bd0b5/Qwen3-30B-A3B-Q4_K_M.gguf | rtx-3090 | holon | 76.09 / 4.71 | 426.84 / 15.88 |
| [release-smoke-qwen3-30b-a3b-q4_k_m](../evidence/bench_release-smoke-qwen3-30b-a3b-q4_k_m_rtx-3090_2026-10-14.json) | unsloth/Qwen3-30B-A3B-GGUF@d5b1d57bd0b5/Qwen3-30B-A3B-Q4_K_M.gguf | rtx-3090 | llamaserver | 97.16 / 5.01 | 684.93 / 19.61 |
| [release-smoke-qwen3-30b-a3b-q4_k_m](../evidence/bench_release-smoke-qwen3-30b-a3b-q4_k_m_rtx-5090_2026-10-14.json) | unsloth/Qwen3-30B-A3B-GGUF@d5b1d57bd0b5/Qwen3-30B-A3B-Q4_K_M.gguf | rtx-5090 | holon | 32.29 / 3.01 | 162.39 / 8.54 |
| [release-smoke-qwen3-30b-a3b-q4_k_m](../evidence/bench_release-smoke-qwen3-30b-a3b-q4_k_m_rtx-5090_2026-10-14.json) | unsloth/Qwen3-30B-A3B-GGUF@d5b1d57bd0b5/Qwen3-30B-A3B-Q4_K_M.gguf | rtx-5090 | llamaserver | 50.84 / 3.11 | 361.51 / 11.09 |
| [release-smoke-qwen3.8-27b-ud-q4_k_m-mtp](../evidence/bench_release-smoke-qwen3.8-27b-ud-q4_k_m-mtp_rtx-3090_2026-10-14.json) | unsloth/Qwen3.8-27B-GGUF@4ca720788d1e/Qwen3.8-27B-UD-Q4_K_M.gguf | rtx-3090 | holon | 169.49 / 13.61 | 2771.39 / 19.14 |
| [release-smoke-qwen3.8-27b-ud-q4_k_m-mtp](../evidence/bench_release-smoke-qwen3.8-27b-ud-q4_k_m-mtp_rtx-3090_2026-10-14.json) | unsloth/Qwen3.8-27B-GGUF@4ca720788d1e/Qwen3.8-27B-UD-Q4_K_M.gguf | rtx-3090 | llamaserver | 484.06 / 14.58 | 4527.17 / 22.63 |
| [release-smoke-qwen3.8-27b-ud-q4_k_m-mtp](../evidence/bench_release-smoke-qwen3.8-27b-ud-q4_k_m-mtp_rtx-5090_2026-10-14.json) | unsloth/Qwen3.8-27B-GGUF@4ca720788d1e/Qwen3.8-27B-UD-Q4_K_M.gguf | rtx-5090 | holon | 67.59 / 6.92 | 1317.73 / 9.52 |
| [release-smoke-qwen3.8-27b-ud-q4_k_m-mtp](../evidence/bench_release-smoke-qwen3.8-27b-ud-q4_k_m-mtp_rtx-5090_2026-10-14.json) | unsloth/Qwen3.8-27B-GGUF@4ca720788d1e/Qwen3.8-27B-UD-Q4_K_M.gguf | rtx-5090 | llamaserver | 296.8 / 7.41 | 2406.54 / 12.31 |
| [release-smoke-qwen3.8-27b-ud-q4_k_m](../evidence/bench_release-smoke-qwen3.8-27b-ud-q4_k_m_rtx-3090_2026-10-14.json) | unsloth/Qwen3.8-27B-GGUF@4ca720788d1e/Qwen3.8-27B-UD-Q4_K_M.gguf | rtx-3090 | holon | 160.35 / 21.71 | 4859.08 / 35.12 |
| [release-smoke-qwen3.8-27b-ud-q4_k_m](../evidence/bench_release-smoke-qwen3.8-27b-ud-q4_k_m_rtx-3090_2026-10-14.json) | unsloth/Qwen3.8-27B-GGUF@4ca720788d1e/Qwen3.8-27B-UD-Q4_K_M.gguf | rtx-3090 | llamaserver | 406.85 / 23.16 | 7786.95 / 43.96 |
| [release-smoke-qwen3.8-27b-ud-q4_k_m](../evidence/bench_release-smoke-qwen3.8-27b-ud-q4_k_m_rtx-5090_2026-10-14.json) | unsloth/Qwen3.8-27B-GGUF@4ca720788d1e/Qwen3.8-27B-UD-Q4_K_M.gguf | rtx-5090 | holon | 63.0 / 11.74 | 2059.1 / 13.99 |
| [release-smoke-qwen3.8-27b-ud-q4_k_m](../evidence/bench_release-smoke-qwen3.8-27b-ud-q4_k_m_rtx-5090_2026-10-14.json) | unsloth/Qwen3.8-27B-GGUF@4ca720788d1e/Qwen3.8-27B-UD-Q4_K_M.gguf | rtx-5090 | llamaserver | 239.66 / 12.6 | 3654.62 / 16.68 |
| [release-smoke-qwen3-8b-bf16](../evidence/bench_release-smoke-qwen3-8b-bf16_rtx-3090_2026-10-14.json) | Qwen/Qwen3-8B@b968826d9c46 | rtx-3090 | holon | 68.14 / 20.08 | 416.61 / 20.82 |
| [release-smoke-qwen3-8b-bf16](../evidence/bench_release-smoke-qwen3-8b-bf16_rtx-3090_2026-10-14.json) | Qwen/Qwen3-8B@b968826d9c46 | rtx-3090 | sglang_http | 87.35 / 19.78 | 431.79 / 20.41 |
| [release-smoke-qwen3-8b-bf16](../evidence/bench_release-smoke-qwen3-8b-bf16_rtx-3090_2026-10-14.json) | Qwen/Qwen3-8B@b968826d9c46 | rtx-3090 | freetoken_http | 69.9 / 19.91 | 438.86 / 20.45 |
| [release-smoke-qwen3-8b-bf16](../evidence/bench_release-smoke-qwen3-8b-bf16_rtx-5090_2026-10-14.json) | Qwen/Qwen3-8B@b968826d9c46 | rtx-5090 | holon | 17.9 / 10.05 | 104.42 / 11.21 |
| [release-smoke-qwen3-8b-bf16](../evidence/bench_release-smoke-qwen3-8b-bf16_rtx-5090_2026-10-14.json) | Qwen/Qwen3-8B@b968826d9c46 | rtx-5090 | sglang_http | 27.24 / 9.93 | 113.24 / 10.92 |
| [release-smoke-qwen3-8b-bf16](../evidence/bench_release-smoke-qwen3-8b-bf16_rtx-5090_2026-10-14.json) | Qwen/Qwen3-8B@b968826d9c46 | rtx-5090 | freetoken_http | 19.94 / 10.03 | 100.97 / 10.97 |
| [release-smoke-qwen3-8b-fp8](../evidence/bench_release-smoke-qwen3-8b-fp8_rtx-3090_2026-10-14.json) | Qwen/Qwen3-8B-FP8@220b46e3b218 | rtx-3090 | holon | — (exit code 1: holon.core.plugins.PluginError: 'model.layers.0.mlp.down_proj.weight_scale_inv' maps to 'layers.0.ffn.down_proj.weight_scale_inv', which the model does not have) |  |
| [release-smoke-qwen3-8b-fp8](../evidence/bench_release-smoke-qwen3-8b-fp8_rtx-3090_2026-10-14.json) | Qwen/Qwen3-8B-FP8@220b46e3b218 | rtx-3090 | sglang_http | 69.93 / 11.18 | 452.0 / 11.96 |
| [release-smoke-qwen3-8b-fp8](../evidence/bench_release-smoke-qwen3-8b-fp8_rtx-3090_2026-10-14.json) | Qwen/Qwen3-8B-FP8@220b46e3b218 | rtx-3090 | freetoken_http | 91.32 / 12.63 | 656.76 / 21.59 |
| [release-smoke-qwen3-8b-fp8](../evidence/bench_release-smoke-qwen3-8b-fp8_rtx-5090_2026-10-14.json) | Qwen/Qwen3-8B-FP8@220b46e3b218 | rtx-5090 | holon | — (exit code 1: holon.core.plugins.PluginError: 'model.layers.0.mlp.down_proj.weight_scale_inv' maps to 'layers.0.ffn.down_proj.weight_scale_inv', which the model does not have) |  |
| [release-smoke-qwen3-8b-fp8](../evidence/bench_release-smoke-qwen3-8b-fp8_rtx-5090_2026-10-14.json) | Qwen/Qwen3-8B-FP8@220b46e3b218 | rtx-5090 | sglang_http | 17.98 / 6.44 | 59.23 / 6.64 |
| [release-smoke-qwen3-8b-fp8](../evidence/bench_release-smoke-qwen3-8b-fp8_rtx-5090_2026-10-14.json) | Qwen/Qwen3-8B-FP8@220b46e3b218 | rtx-5090 | freetoken_http | 15.17 / 6.06 | 73.92 / 6.97 |
| [release-smoke-qwen3-8b-nvfp4](../evidence/bench_release-smoke-qwen3-8b-nvfp4_rtx-3090_2026-10-14.json) | nvidia/Qwen3-8B-NVFP4@ccd10a893cbc | rtx-3090 | holon | — (exit code 1: holon.core.plugins.PluginError: 'model.layers.0.mlp.down_proj.input_scale' maps to 'layers.0.ffn.down_proj.input_scale', which the model does not have) |  |
| [release-smoke-qwen3-8b-nvfp4](../evidence/bench_release-smoke-qwen3-8b-nvfp4_rtx-3090_2026-10-14.json) | nvidia/Qwen3-8B-NVFP4@ccd10a893cbc | rtx-3090 | sglang_http | 65.34 / 7.63 | 428.39 / 8.28 |
| [release-smoke-qwen3-8b-nvfp4](../evidence/bench_release-smoke-qwen3-8b-nvfp4_rtx-3090_2026-10-14.json) | nvidia/Qwen3-8B-NVFP4@ccd10a893cbc | rtx-3090 | freetoken_http | — (exit code 1: RuntimeError: chunk expects at least a 1-dimensional tensor) |  |
| [release-smoke-qwen3-8b-nvfp4](../evidence/bench_release-smoke-qwen3-8b-nvfp4_rtx-5090_2026-10-14.json) | nvidia/Qwen3-8B-NVFP4@ccd10a893cbc | rtx-5090 | holon | — (exit code 1: holon.core.plugins.PluginError: 'model.layers.0.mlp.down_proj.input_scale' maps to 'layers.0.ffn.down_proj.input_scale', which the model does not have) |  |
| [release-smoke-qwen3-8b-nvfp4](../evidence/bench_release-smoke-qwen3-8b-nvfp4_rtx-5090_2026-10-14.json) | nvidia/Qwen3-8B-NVFP4@ccd10a893cbc | rtx-5090 | sglang_http | 12.93 / 4.26 | 34.46 / 4.61 |
| [release-smoke-qwen3-8b-nvfp4](../evidence/bench_release-smoke-qwen3-8b-nvfp4_rtx-5090_2026-10-14.json) | nvidia/Qwen3-8B-NVFP4@ccd10a893cbc | rtx-5090 | freetoken_http | — (exit code 1: RuntimeError: chunk expects at least a 1-dimensional tensor) |  |
| [release-smoke-qwen3-8b-q4_k_m](../evidence/bench_release-smoke-qwen3-8b-q4_k_m_rtx-3090_2026-10-14.json) | unsloth/Qwen3-8B-GGUF@a6adef130ffb/Qwen3-8B-Q4_K_M.gguf | rtx-3090 | holon | 50.36 / 6.96 | 364.9 / 15.08 |
| [release-smoke-qwen3-8b-q4_k_m](../evidence/bench_release-smoke-qwen3-8b-q4_k_m_rtx-3090_2026-10-14.json) | unsloth/Qwen3-8B-GGUF@a6adef130ffb/Qwen3-8B-Q4_K_M.gguf | rtx-3090 | llamaserver | 79.06 / 7.41 | 649.81 / 18.1 |
| [release-smoke-qwen3-8b-q4_k_m](../evidence/bench_release-smoke-qwen3-8b-q4_k_m_rtx-5090_2026-10-14.json) | unsloth/Qwen3-8B-GGUF@a6adef130ffb/Qwen3-8B-Q4_K_M.gguf | rtx-5090 | holon | 19.83 / 4.03 | 125.21 / 5.53 |
| [release-smoke-qwen3-8b-q4_k_m](../evidence/bench_release-smoke-qwen3-8b-q4_k_m_rtx-5090_2026-10-14.json) | unsloth/Qwen3-8B-GGUF@a6adef130ffb/Qwen3-8B-Q4_K_M.gguf | rtx-5090 | llamaserver | 43.14 / 4.26 | 346.5 / 8.15 |

**M0.6's files** (borrowed on the 5090 and 3090):
- They are not available on main: M0.6 is unmerged, and main's ggml is still at 51ce9c11. Holon on main refuses each one at load, and the records name the reason:
  - FP8 and NVFP4: no weight rule for the scale tensors;
  - the MXFP4_MOE GGUF: no `linear_experts` kernel for `ggml.mxfp4`.
- So the gap is recorded rather than measured. These files' Holon smoke waits for M0.6 Gate A on main.
- The baselines did run them: 
  - **llama.cpp** serves the MXFP4_MOE GGUF on both cards.
  - **SGLang** serves the 8B FP8 and 8B NVFP4 on both cards. It serves the 30B NVFP4 on the 3090, but fails it on the 5090 (`'FusedMoE' object has no attribute 'g1_scale_c'`).
  - **FreeToken** serves the 8B FP8 on both cards. It fails both NVFP4 checkpoints at weight load on both cards (`chunk expects at least a 1-dimensional tensor`).
  - The design's "M0.6's files smoke-tested on the 3090" (§11) is therefore met for the baselines only.

**Deviations and decisions:**
- **Locks.** The 3090 and 5090 cells run other classes' locks (above). Step 8 reruns at the step-7 locks.
- **Smoke engine.** Holon is smoked as the in-process engine, because it reports the KV tokens the baselines are matched to. The served path (`holon_http`) is unchanged since M0.5's records.
- **Lost GPU.** The first RTX 3090 desktop (A) lost its GPU mid-run (it fell off the bus) during the first bar cell, and its partial record was discarded. Every 3090 record comes from desktop B, except the 27B `-r3` run on C.
- **Concurrent writers.** Two allocations of one class shared a Triton cache directory, which gave "Stale file handle". Each allocation now gets its own tag, and the affected 5090 runs were redone.
- **Venvs.** The cluster venv's and the development venv's editable installs predate M0.5's plugin renames (`frontends.openai`). The runs used the M0.5 cluster venv, whose entry points match main.
- **Commits.** The records come from two fixed CI worktrees, at 8bf7bca and c9a6f0c (above).

**For the lead:**
1. **The 3090 27B flag** (above) is an artifact of desktop B's thermal clock swings. On desktop C the cell is 1.02–1.04×. Proposal: before step 8, the 3090 class's release evidence (locks and suite) runs on desktops that hold steady clocks under the 27B. A short clock check (sustained 27B decode with nvidia-smi logging) would screen a desktop, and that check's result would be recorded beside each 3090 record. Your call whether to make it a rule.
2. **No 5090 / 3090 GGUF locks exist yet.** Every cell here ran a borrowed lock's settings (marked in the tables). Step 8 re-runs the bars at the step-7 locks, after M0.6 Gate A moves ggml to `cbe28bb3`; until then the gate compares Holon's `51ce9c11` kernels with llama.cpp at `cbe28bb3`.
3. **FreeToken and SGLang on NVFP4** fail at load as listed. These will publish as "—" with those reasons unless a later pin fixes them. Nothing was patched (§10: "no patches beyond a documented one-liner").
4. **Upstream llama.cpp** (`df18bee1`) is within 1.5% of the pin on every steady cell, so no upstream regression or gain is in sight.

**Lead decisions on step 3 (2026-10-14):**
1. **Every desktop that produces release evidence (locks, the suite, the demo) is screened first** with a short GPU clock check under the 27B: SM clocks steady and a thermal-slowdown share recorded. The check's result is written beside each record (GPU clock and slowdown share only, no host details). The RTX 3090 desktop whose clocks swung (pp512 0.44× / 0.57×) is excluded.
2. **Step 8 re-runs the bars at the step-7 locks after M0.6's Gate A merge**, so Holon's ggml is at the gate pin `cbe28bb3`. The cells below their predicted bands but above 0.8× (pp512 at 8K 0.91–0.96, the 5090's tg128 0.93–0.96) are re-measured in M0.7's step 0.
3. **Baselines that fail on a file are published as "—" with their recorded reason;** they are not patched (SGLang's NVFP4 30B on the 5090, FreeToken's NVFP4 files).

### 13.6 Step 4, documentation (drafted 2026-10-14; frozen after M0.6's Gate A merge)

`README.md`, `docs/adding-a-model.md` (≈ 3.3K tokens) and the task-routed `AGENTS.md`, together with `scripts/ci/check_docs.py` in `run_all.sh`, landed in bdabd9e. The writer was told not to read §4 or `bench/legibility/`. **Deviation, recorded for the experiment:** two of the writer's repository searches matched four single lines of §4: a planned guide size (≈ 4K), a mention of a reinstall, and the pinned agent settings. The lines it saw are those of §4 near lines 222, 227, 286 and 289 of this file at 7a65d28. It read nothing else of §4 and never saw the target model. The guide's length and reinstall section follow the writer's brief, which predates those lines. The tailoring review (reviewer C) and the freeze happen before any reference port or pilot.

**Tailoring review (reviewer C, 2026-10-14): the guide and AGENTS.md's routing pass all four tests** (derivable from the
tree, no emphasis driven by the task, proportion, no mention). The deviation compromises nothing, because the control is
the review of the text. The README's benchmark placeholder failed test 4, so it was removed; the result section is
added after the verdict. A drift check of the frozen text follows M0.6's Gate A merge.

### 13.7 Rebase onto M0.6 Gate A (2026-10-14)

The release branch was rebased onto `main` at 936c1ab (M0.6 Gate A merged). The rebase is linear, so the step-3
merge (ad2919f before the rebase) is gone. Its only content of its own, the lead's step-3 decisions in this document,
came in with the step-3 record. Outside the files `main` changed, the tree equals the merge's. The commit hashes in
§13.1–§13.6 name the commits before the rebase.

- **Conflicts.**
  - `kernels/ggml_cuda/plugin.toml`: kept M0.6's vendored commit and `plugin.py` provenance, and step 1's `[build]`.
  - `holon.api.__all__`: the release's list without `Converter`, `Dialect` and `SchemeImpl`.
  - `holon_tools.__main__`: M0.6's rule that a lock's pools seed checks on its own class only, with step 2's model id
    passed to `EngineChecks`.
  - `build.py`, std's `ops/units.py` and `moe/units.py`: M0.6's per-slice loading, with the hook renamed (below). The
    release's `set(q.parts) != {"blocks"}` guard on `ExpertLinear` is dropped: M0.6's parts carry scales. The soft
    kernel preference in `_choose` merged cleanly.
  - `hooks.py`: `AdoptsWeights` takes `index` and keeps the docstring from the last pre-rebase commit (False is a load error).
  - `bench/backends/holon_http.py`: `served()` and the temperature option are both kept.
  - `docs/release.md`: replaced by the step-3 merge's text.
  - Merged without conflict: `manifest.py`, `ops.py`, `plugins.py`, the judge, std's weights and `design.md` (§4.1's
    entry-point fix, with M0.6's §5.5 and no dialect or converter registry).
- **Rename, `adopt` → `adopt_weights` for units.** It is applied in the core's `_adopt`, std's `Linear`, `Embedding`,
  `LMHead` and `ExpertLinear`, the toy `_Rows` in `core/tests/test_stacking.py`, `plugins/tests/test_quantized_units.py`
  (`test_ggml_cuda_moe.py` already had it), and design §5.5 and the feature map. Each takes `index: int | None = None`,
  as does `hooks.AdoptsWeights`. The state kinds' `adopt(seq, payload)` is unchanged.
- **Listings, still ABI 0.1.** `main` has none, so they were written fresh and `check_abi.py --base main` passes.
  - `core/abi.txt` loses `Converter`, `Dialect`, `SchemeImpl`, `Registries.{schemes, converters, dialects}` and the
    `dialect` / `converter` slots, and `AdoptsWeights.adopt_weights` gains `index`.
  - `std/contracts.txt`: `op.qlinear@1` → `@2`, `op.linear_experts@1` → `@2`.
  - `plugins/contracts.txt` is unchanged.
- **Manifests and dependencies.**
  - Every manifest M0.6 changed already carried `core_abi`. The new test fixture `tests.toy_quant` now does too.
  - manifest-deps found M0.6's new imports. The judge's `hf_quant` and testing's `hf_quant_files` use compressed-tensors,
    now declared at 0.17.0, the version M0.6's owner checks ran (`uv.lock`: compressed-tensors 0.17.0 and loguru; 115
    packages).
  - ModelOpt and amd-quark are imported under `except ImportError` (test-only, "when importable"). manifest-deps now
    skips such guarded imports.
  - The judge loads llama.cpp's `conversion.base` from the checkout it is given via `importlib`, since it is not a
    dependency.
- **Suite** (RTX A5000 ×2, the uv venv): `pytest core std plugins judge testing tools` gave 1145 passed and 62 skipped
  in 18 min 26 s. `run_all.sh` is green. The core is 4123 code lines: `main` has 3844, and steps 1–2 add 279.
- **Not done, as planned:** no lock was written or propagated. Every lock is outdated by the release's core change until
  M0.7's propagation.
- **Coverage note:** this venv has neither ModelOpt nor amd-quark, so `hf_quant`'s ModelOpt and Quark owner checks are
  skipped silently here, as they were for Quark under M0.6.

**Documentation frozen at eb9eef5 (2026-10-14).** Reviewer C's drift check after M0.6's Gate A merge: `docs/adding-a-model.md` and `AGENTS.md` are byte-identical to the reviewed bdabd9e, and the README differs only by the benchmark-placeholder cut. `check_docs` passes, the spot-checked facts hold on the rebased tree, and the four tailoring tests hold. Any later edit to these three files voids the legibility branch point (§4.2). Notes for the branch-point build: the strip list is matched by content (plan.md's §9 row is now at line 509), and the squashed branch point replaces the log, including commit messages that mention the benchmark.

**Owner decisions (2026-10-14):**
- **Publishing: a squashed root (owner question 1).** The private repository's history is kept as is, never rewritten.
  The public repository receives one parentless commit holding exactly the release tree, tagged `v0.1.0`. From
  step 6, local `main` continues from that root, so the release locks and records name public commits. The old line is
  kept as an archive branch and on the private remote. Pushing still waits for the owner's go (step 11). PyPI names:
  not reserved (no wheels in v0).
- **Cold start (escalated lead decision 1, §13.3): accepted.** FlashInfer's CuTe DSL norms compile in memory on each
  start (≈ 1 s). Acceptance 4 reads "compiles nothing but Triton and FlashInfer's CuTe DSL norms".
- **Distribution (owner, 2026-10-14): v0 is a GitHub release only.** It is a tagged source release with release notes
  and the Dockerfile, with no PyPI package and no pushed image. A full pip package with prebuilt wheels is v1 or v2
  work (plan §10).

## 14. User interface (UX step)

Status: **rev 2.1, confirmed by the release panel** (2026-10-14); built, see §14.14. Rev 1 (d7ef8c9) was reviewed by
two release reviewers (serving / ABI; architecture / scope): no blocker, no major; their findings and the lead's
decisions are resolved in §14.13; rev 2 (2932427) was confirmed with three small items, folded in (§14.13). The owner's request: "`holon serve xxx` just works", and keep the interface easy to
extend — this is the simplest case (one model × one GPU); more tuning options (multiple GPUs) and an autotune module
that tunes the serving parameters come later. Built on a branch cut from M0.7's, so its ABI additions join M0.7's 0.2
bump; it merges into M0.7's branch before `main`. M0.7's steps 2–3 run in parallel: this step touches neither
`std.ops`, `std.mixers.gqa`, `std.mixers.gdn` nor `kernels.ggml_cuda`.

### 14.1 Goal and acceptance

1. **Hub references.** `holon serve unsloth/Qwen3.8-27B-GGUF:UD-IQ3_S` and `holon serve Qwen/Qwen3-8B` download into
   the HF cache and serve from the lock, as a downloaded path does today; `holon serve unsloth/Qwen3.8-27B-GGUF` serves
   the file this card has a verified lock for. The chosen file is always logged; a file larger than the GPU's memory is
   refused before any download. With `HF_HUB_OFFLINE=1` a cached reference serves without the network.
2. **Help.** `holon serve --help` gives every flag a help line; the engine flags' lines come from `EngineConfig` field
   metadata, so a field added without help fails the CLI test. It also lists every plugin option an installed plugin
   declares, and the plugins selectable per slot.
3. **`holon models`** lists, for the visible GPU, every locked model under a name `holon serve` accepts, and whether
   `holon serve` would start from this card's lock (verified), another card's (borrowed, unverified) or none
   (outdated).
4. **Claude Code hint.** Once the server answers, `frontends.http` prints the environment lines with the served id.
5. **Short model id** by default; `--plugin-options http.model_name=ID` renames it.
6. **Core stays general.** No hub or format name in `holon.*` (the denylist gains `huggingface`); ≈ 125 core lines.
7. **No extra propagation.** This step merges before M0.7's closing propagation, which covers it (§14.10).

### 14.2 The configuration model (how the interface extends)

`holon serve` takes two kinds of input, and every future option is one of them:

- **Serving settings** shape what is served — outputs, speed, memory: the `EngineConfig` fields (core-wide) and the
  plugin options (`EngineConfig.plugin_options`, each key owned by a plugin). A lock is a saved set of them. Values
  layer **defaults < lock < flags** (D2); the defaults are `EngineConfig`'s, except `holon serve`'s two (at most 8
  sequences without a lock, `--max-model-len auto`), which its help states.
- **Launch arguments** say where and how the process runs: the model reference, `--host`, `--port`, `--frontend`,
  `--plans`, `--no-lock`, `--no-size-check`, `--supervised`. They are per host, like memory budgets, so no lock records them.

| a new … | goes into | and gets, with no other code |
|---|---|---|
| core-wide serving setting (e.g. a device list, a parallel layout) | an `EngineConfig` field with a help string | a flag, a help line, a lock field (unless in `plans.NOT_PLANNED`), lock precedence |
| plugin-owned setting (e.g. `kv.format`) | the plugin's code, declared in its manifest's `[options]` | a help line, a `choices` check before start, lock precedence through `plugin_options` |
| plugin alternative (a drafter, a grammar processor) | a plugin of that slot | a line under "selectable with `--slots`", lock precedence through `slots` |
| launch argument | a `holon serve` flag with help | — (never in a lock) |
| command | a `holon` subcommand (core), or one a distribution adds (§14.9) | — |

There is no third kind: no environment-variable configuration (invariant 3) and no setting held only by a flag. Design
invariant 5's typed deployment file, when it comes, is these two groups in one file, layered between the lock and the
flags; nothing here needs it.

**Which options a plan records is derived, not declared.** `plan_of` leaves out an option exactly when every plugin
declaring it is in the `frontend` slot, which the engine never loads (`http.drop_billing_header`, `http.model_name`):
they cannot shape outputs, so they no longer show as lock differences or move `plan_id`. Undeclared keys stay planned.
No manifest key says "unplanned", so no plugin can hide an output-shaping option. No committed lock carries a frontend
option, so no `plan_id` moves.

### 14.3 Hub references (item 1)

**Decision: weight-source plugins may resolve references; the core asks them and names no hub.** A weight source
knows which files form its checkpoint: the HF source needs a repository's config, index, tokenizer, template and
safetensors files; the GGUF source needs one file among many, chosen by a quantization tag.

| alternative | cost | verdict |
|---|---|---|
| (a) as today: `hf download`, then a path | none | two commands, and the user must know which file suits the card: not "just works" |
| (b) a hub plugin in a new slot, formats declaring fetch patterns in their manifests | a slot, a manifest key, a ≈ 60-line generic plugin | which files form a checkpoint, and how a tag picks one of twenty GGUF files, is format knowledge that patterns express badly |
| (c) **an optional `resolve` hook on weight-source plugins** | one optional Protocol and one record, ≈ 20 core lines, ≈ 75 plugin lines | chosen |
| (d) the core calls the hub library | — | breaks invariant 1 |

A second hub plugs into (c) without touching the sources: a plugin of slot `weight_source` that provides no source and
only resolves (its fetch returns a local path, which the existing sources detect and read). (b) becomes worth it only
if several hubs each need every format; that is the trigger, recorded.

**The hook** (`holon.api`; optional, on the weight-source plugin object; ABI additions):

```python
@dataclass(frozen=True)
class Resolved:
    what: str                       # what will be fetched, for the log, e.g. "<repo>/<file>"
    nbytes: int | None              # its total size from the listing; None: unknown
    fetch: Callable[[], str]        # downloads (into the plugin's cache) and returns the local path

class ResolvesRefs(Protocol):       # holon.api.hooks
    def resolve(self, ref: str) -> Resolved | None:
        """`ref` names a checkpoint that is not a local path (e.g. a hub id): None if this plugin does not fetch it.
        Claiming may look the reference up (a listing, a small index) but never downloads weights; a ValueError
        explains a reference recognised but not servable (plugins cannot import the core's PluginError)."""
```

**The core** (`cli.local_model`; `holon serve` calls it first, in its own process, after checking the option flags
(§14.4), so a bad flag never costs a download), in this order:
1. a `ref` naming an existing path is that path; nothing else is asked, so a local `org/repo` directory stays a path;
2. the repository rule (D-repo, below), which also applies under `--no-lock`: it only names the file;
3. every installed weight-source plugin with the hook (`assembly.claim_ref`; with `--source K`, only the one providing
   `weight_source.K`); a hook's ValueError is kept as a hint. Exactly one claim → it is logged (`holon: <ref>: <what>
   (<size>)`). **The size refusal** applies only when the device is CUDA, `Fingerprint.vram_gib > 0` and `nbytes` is
   known: `nbytes > vram_gib × 2³⁰` (bytes against the card's total memory; v0 has no offload) is refused before
   downloading, naming both sizes, `holon models` and the launch flag `--no-size-check`, which skips it. Else its fetch
   runs, and the local path becomes `EngineConfig.model` and is detected as before. None → `<ref>: no such file or
   directory, and no weight source resolves it` plus the hints; several → name one with `--source`. Claims are
   separate from fetches, so an ambiguous reference never downloads twice.

The download happens before the supervisor starts, so engine processes and restarts get a local path. The lock is
found by the build hash as before (D2): a hub-cache GGUF's blob name is its sha256, so no file is re-read.

**The hub library** stays the plugins' and stays lazy: `huggingface_hub` is imported inside `resolve`, so serving a
local path never imports it. It is declared in std's `pyproject.toml` (for `manifest-deps`; already in the runtime
closure through transformers, so the plans' `uv_lock` hash is unchanged), not in the manifests' `[dependencies]
python`, which list what loading the plugin needs. The library keeps its own configuration (`HF_HOME`; `HF_TOKEN` or
`hf auth login` for gated repositories); core and std read no environment variable.

**Network and cache.** A claim lists the repository on the hub; on a connection error it falls back to the cache's
listing of that repository, and if that fails too, its ValueError names `HF_HUB_OFFLINE=1` and the cache. Offline
(`HF_HUB_OFFLINE=1`), claims and fetches use the cache alone, so a tag chooses among the **cached** files only; the log
line names the file chosen either way. A cached snapshot that lacks what a source needs (e.g. only the tokenizer files
transformers cached) is said so by each source, rather than declined silently.

**References** (the plugins' grammar; the core sees an opaque string):

| reference | claimed by | fetched |
|---|---|---|
| `Qwen/Qwen3-8B` | `std.weights.hf`, if the repository has `config.json` and `*.safetensors` files or a safetensors index | the config, index, tokenizer and template files and the safetensors files — with an index, only those it names (fetched to size them; M0.7 §17) |
| `unsloth/Qwen3.8-27B-GGUF:UD-IQ3_S` | `std.weights.gguf` | the one file the tag names (below) |
| `unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_S` (`.gguf` optional) | `std.weights.gguf` | the file `<stem>.gguf`, searched in every directory of the repository: the GGUF source's name drops the directory (`std/holon_std/weights/gguf/source.py:47-49`). The form `holon models` prints |
| `unsloth/Qwen3.8-27B-GGUF` | the core first (D-repo), else `std.weights.gguf` if the repository has `.gguf` files | this card's verified file; else the repository's only GGUF file, else an error listing its tags |

- **Narrow claims:** the HF source needs safetensors weights, the GGUF source needs `.gguf` files, so a GGUF repository
  that also ships a `config.json` is not ambiguous. A repository with both kinds of weights is (→ `--source`).
- **Tag rule:** the `.gguf` files of the repository (any directory, multimodal projectors excluded) whose stem ends in
  `-TAG` or `.TAG`, case-insensitively; of several, the shortest stem, so `:Q4_K_M` takes `X-Q4_K_M` over
  `X-UD-Q4_K_M`; a tie or no match is an error listing the tags. llama.cpp's `-hf` takes the first file matching
  `TAG[.-]` anywhere in its name (`common/download.cpp`, `find_best_model`), so its pick of `Q4_K_M` depends on
  listing order; ours does not.
- **Split files** (`-00001-of-00003.gguf`) are refused with that reason: the GGUF source reads one file (v1).
- A reference starting with `/`, `.` or `~` is a path only; a typo of a local path that reaches a hub fails with the
  message above, which starts with "no such file or directory".

**D-repo: a repository reference serves this card's verified file** (kept by the lead). After the local-path check
and before the plugins, for a reference of the form `<org>/<repo>` (a trailing `/` ignored), the core looks for locked
model ids whose name continues `<ref>/` (whole path segments). If exactly one of them has a verified lock for this card
(§14.5's lookup), the reference becomes that name, logged
(`holon: unsloth/Qwen3.8-27B-GGUF -> unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_S, this card's lock`). Several
verified (the RTX 5080 locks two Qwen3-8B GGUF files) → an error listing them; none → the plugins see the reference
as given. The rule uses names and path segments only; on every card it makes `holon serve unsloth/Qwen3.8-27B-GGUF`
serve the release's file for that card (5080: UD-IQ3_S; 4090, 5090, 3090: UD-Q4_K_M). ≈ 10 core lines.

### 14.4 Help for every flag (item 2)

- **Engine settings.** Each `EngineConfig` field carries `metadata={"help": …}` through a small helper in
  `engine.py`; today's trailing comments become the help strings. `engine_flags` passes it to argparse and raises on a
  field without one, so every CLI test, and CI, fails until it has help. The rendered ABI listing ignores metadata, so
  `EngineConfig`'s listing does not move.
- **`holon serve --help`** shows two groups, "launch" and "serving settings (override the lock)", each line with
  `EngineConfig`'s default, plus `holon serve`'s two own defaults (§14.2). The epilog lists the plugin options and the
  plugins selectable with `--slots` (every slot not bound per site, except the model, chosen by architecture, and the
  frontend, chosen by `--frontend`), all read from manifests without importing a plugin.
- **Plugin options are declared in the manifest**, by the plugin that reads them (ABI additions: section `[options]`,
  `Option` exported from `holon.api`, `Manifest.options` appended last):

  ```toml
  [options]   # the EngineConfig.plugin_options keys this plugin reads
  "kv.format" = { help = "KV cache format", default = "the model dtype", choices = ["q8_0", "fp8"] }
  "recurrent.checkpoints" = { help = "prefix-cache checkpoints of recurrent state kept", default = "max_seqs" }
  ```

  `Option(help: str, default: str | None = None, choices: tuple[str, ...] = ())`: `help` is required; `default` is
  display text; values stay the plugin's to parse. Several plugins may declare one key (`std.ops` and `std.moe` both
  read `quant.act`); each is listed. A manifest that uses an option key added in a later ABI raises its `core_abi`
  floor (design §4.1, at build).
- **Checks before any engine start**, on each history entry's layered configuration:
  - a key no installed plugin declares → a **warning** naming it (a typo is ignored silently today); the key still
    enters `plugin_options`, so the plan diff shows it;
  - a value must be in the `choices` of **every** declaring plugin that has `choices`;
  - a **flag's** value outside them → an error naming the plugins and their choices, before the model is fetched.
    Today such a value raises `ValueError` inside the engine's build, a start failure of kind "error", retried 4 times
    before the fallback;
  - a **lock's** value outside them (a lock older than a narrowed `choices`) → a warning, and that history entry is
    dropped, so `[defaults + flags]` serves;
  - `key=` still deletes. `choices` are declared only for closed sets (`kv.format`, `quant.act`), and the plugin's own
    check stays authoritative.
- **The declarations** (14 keys, 9 plugins):

  | plugin | keys |
  |---|---|
  | `std.mixers.gqa` | `kv.format` |
  | `std.ops`, `std.moe` | `quant.act` (`a4`, `a8`) |
  | `std.state.recurrent` | `recurrent.policy`, `recurrent.checkpoints` |
  | `std.spec.fixed_k` | `spec.k`, `spec.fallback` |
  | `drafters.ngram`, `drafters.dflash` | `ngram.max`, `ngram.min`; `dflash.path` |
  | `grammar.xgrammar` | `grammar.workers`, `grammar.timeout`, `grammar.budget` |
  | `frontends.http` | `http.drop_billing_header`, `http.model_name` (new, §14.7) |

  `std.mixers.gqa` and `std.ops` get theirs in one small commit on M0.7's branch after its steps 2–3 merge (they are
  edited there); until then `kv.format` and `quant.act` draw the warning on the development branches only.
- **`[tunables]`** (per-site knobs for `Binding.tunables`, declared by no manifest) stays as it is. A future tuner
  reads `choices` (and a `range` key added then) from `[options]` (§14.9). Step 6 (ABI 1.0) decides whether the
  unused `[tunables]` is folded into `[options]` or kept.

### 14.5 `holon models` (item 3)

```text
$ holon models
rtx-5080/cu130-torch2.11 (the first visible GPU); locks in <tree>/plans
  unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_S      this card's lock 7       context 32768, 4 sequences
  unsloth/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M              this card's lock 12      context 32768, 8 sequences
  Qwen/Qwen3-8B                                      rtx-4090's lock 11, unverified
  …                                                  outdated: serves defaults
serve one: holon serve <name>
```

- For each model id under `--plans` (default as `holon serve`), the row is `plans.nearest_class` with the installed
  sources, the function `holon serve` uses, so the two cannot disagree. Verified rows first: they are the recommended
  files for this card (release §6 locks one file per model and card). An unreadable history is a warning row.
- **Every name printed is servable.** `check_locks.py` requires a committed lock's name (its model id without
  `@<build>`) to be `<org>/<repo>` or `<org>/<repo>/<stem>`; all committed ids are. A row under another `--plans`
  whose name is not of that form (a lock written from a local directory) is marked "not servable by name".
- Context and sequences come from the lock's `[engine]` (generic fields). Whether a file fits is not shown: a lock
  records no memory need. A verified lock fits by construction; a borrowed one from a larger card may fail its start
  and fall back to defaults, logged.
- No visible GPU → an error naming `CUDA_VISIBLE_DEVICES`.

### 14.6 Claude Code hint (item 4)

Once uvicorn has bound the port (an `uvicorn.Server` subclass's `startup`; the supervisor has already started the
engine), the plugin prints to stderr:

```text
holon: serving Qwen3.8-27B-UD-IQ3_S at http://127.0.0.1:8000 (OpenAI: /v1; Anthropic: /v1/messages)
holon: Claude Code:
  ANTHROPIC_BASE_URL=http://127.0.0.1:8000 ANTHROPIC_AUTH_TOKEN=holon ANTHROPIC_MODEL=Qwen3.8-27B-UD-IQ3_S ANTHROPIC_DEFAULT_HAIKU_MODEL=Qwen3.8-27B-UD-IQ3_S CLAUDE_CODE_SUBAGENT_MODEL=Qwen3.8-27B-UD-IQ3_S claude
```

`0.0.0.0` and `::` show as `127.0.0.1`; values are shell-quoted. The text is a pure function of (host, port, id) in
the plugin. A rollback keeps the id, so the lines stay valid.

### 14.7 The served model id (item 5)

- **Default: the last path segment of the bundle's `model_name`** (accepted by the lead): `Qwen3.8-27B-UD-IQ3_S`
  instead of `unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_S`, `Qwen3-8B` instead of `Qwen/Qwen3-8B`; a local file's or
  directory's name is unchanged. The frontend serves one model and ignores a request's `model` field, as today, so
  clients configured with the old id keep working.
- **`http.model_name=ID`** (a declared option) renames it, in `/v1/models`, the responses and the hint. Declared only by
  a frontend plugin, it is not planned (§14.2), so it shows as no lock difference.
- **Not a core `--served-model-name` flag:** the id is the frontend's concern, and `FrontendPlugin.serve` carries no
  name; adding one changes a plugin-implemented signature.

### 14.8 Future multi-GPU: where it goes (not built)

| concern | today | with several GPUs |
|---|---|---|
| devices, layout | `EngineConfig.device`, not planned | a device-list field (not planned: per host) and a layout field such as `tensor_parallel` (planned: it changes shapes and numerics); each gets its flag, help and lock field from §14.2; `engine_flags` gains one branch for list-typed fields |
| ranks | the `launcher` slot, one built-in local rank (design §8) | a launcher plugin, selected by `--slots launcher=…` or by the lock |
| lock class | `<gpu>/<toolchain>` from `platform.fingerprint(device)` | the fingerprint of the device group, `<gpu>x<n>/<toolchain>` (e.g. `rtx-4090x2/cu130-torch2.11`): still two segments, so `plans/<model>/<class>/<n>.lock` and `nearest_class`'s glob hold. A mixed group is its own class; the interconnect joins the class only if evidence shows it matters |
| borrowing | own class, same GPU, same SM, same toolchain | GPU count first: a 1-GPU lock may serve on one card of a 2-GPU host (as today); a 2-GPU lock is never borrowed by a 1-GPU host, where it could not start |
| `holon models`, D-repo, the size refusal | the first visible GPU's class and memory | the visible group's class and memory |
| the size refusal with offload | weights against the card's memory (v0 has no offload) | once an offload plugin is selected (v1's expert offload, e.g. a `tier` slot plugin), the check compares against device memory plus that plugin's declared capacity, so "fits" stays out of the core and v1's larger-than-VRAM MoE stays servable |

`holon serve <model>` stays the command: a 2×4090 host finds `rtx-4090x2` locks first, so the layout comes from the
lock, and `--tensor-parallel N` overrides it like any setting.

### 14.9 Future autotune: what exists, what it adds (not built)

**Exists:** `holon_tools tune` (sweeps kernel providers per decode op on this GPU, then propagate's checks: L2 against
goldens and floors, the snapshot, an approval; engine settings come from a human as flags); `propagate` (carries a
lock to new code); the lock writer, the lookup (`plans.find`, `lock_overrides`) and the served-plan diff; the judge's
tiers and L4 rules (design §10.3); M0.7's benchmark runner with the Claude-Code-shaped trace (D3); and, after this
step, the declarations a search space needs: `EngineConfig` field metadata and `[options]`.

**A user-facing `holon tune <model>` would add:**
1. **A search space** from the same two declarations: a `tune` entry in a field's metadata beside its help; `choices`
   and a future `range` in `[options]`. Memory budgets stay per host.
2. **An objective:** a workload (default: the Claude-Code-shaped trace) and D3's style of metric, e.g. median TPOT
   with p95 TTFT and gap guards, measured under the judge's L4 rules.
3. **A driver:** start from the nearest lock and search coordinate-wise. Each candidate is a full engine start
   (minutes); a candidate that does not fit fails fast (`OutOfUnits`).
4. **The judge as the gate** (design D5 and design §10.5's first row): settings that change only scheduling
   (`max_seqs`, `max_step_tokens`, `chunk_with_decode`, speculation depth, `recurrent.checkpoints`) are judged bounded
   with L4 re-scoring; options that change numerics (`kv.format`, `quant.act`) need L2 against goldens and floors,
   which a user install does not have. A first version would search scheduling settings only.
5. **A host-local lock**, written by the tools into a user plans directory: design §4.4's host-local history on top of
   the tree's. `holon serve` would read several directories (`--plans` repeated) and rank across them: a current,
   verified host-local lock first; then the tree's verified lock; then a borrowed one (`nearest_class`'s order). A
   host-local lock that a `git pull` made outdated (it pins other code) falls through to the tree's.
6. **One tuner, two entry points.** The user's `holon tune` and design §10.2's agent-facing `tune.submit(site, space?)`
   drive the same tuner, the same judge gate and the same lock writer; they differ only in who starts them.
7. **Packaging:** tools and judge as an optional extra of the install, and `tune` added as a `holon` subcommand
   through an entry-point group (`holon.commands`), so the core never imports the tools.

This step builds none of it. It makes sure the tuner and `holon serve` will read the same declarations and write and
read the same lock.

### 14.10 ABI, locks, merge order

- **ABI 0.2 additions**, inside M0.7's bump, under the minor rule: `hooks.ResolvesRefs` and the `Resolved` record;
  `Option`, exported from `holon.api`; `Manifest.options`, appended last; the manifest section `[options]`.
  `EngineConfig`'s listing is unchanged. `core/abi.txt` is regenerated at 0.2; `check_abi.py --base main` passes. The
  contracts listings do not change.
- **Locks.** The core, `std.weights.hf` and `.gguf`, and the manifests gaining `[options]` change their `src`, so every
  lock pinning them is outdated. That is already true of every lock until M0.7's closing propagation (its step 6). This
  step merges into M0.7's branch before then, so the one propagation covers it. `frontends.http` is pinned by no lock.
  If the step misses that window, it costs one propagation of its own; release step 7 re-propagates everything anyway.
  `plan_of`'s derived rule (§14.2) moves no committed `plan_id`: no committed lock has a frontend option.
- **Documents at build:** README (quickstart: `holon models`, then `holon serve unsloth/Qwen3.8-27B-GGUF`; the Claude
  Code section points at the printed lines; Options: `holon serve --help`; Docker: mount a cache volume at the image's
  `HF_HOME`), D8's quickstart block, design §4.1 (`[options]`, and the `core_abi` floor rule for later option keys),
  §4.3 (engine-side plugins declare every option they read; a key declared only by a frontend is unplanned), §5.5 (the `resolve` hook) and §7.8 (the hint and id). The README freeze served
  the legibility benchmark, retired in §4.0.

### 14.11 Core lines, tests, checks, estimate

| core change | file | lines |
|---|---|---|
| reference resolution, the log line, the size refusal | `assembly.py` | ≈ 20 |
| D-repo | `cli.py` | ≈ 10 |
| `holon models`, the servable-name mark | `cli.py` | ≈ 27 |
| help: field metadata, `engine_flags`, groups | `engine.py`, `cli.py` | ≈ 15 net |
| `[options]`: `Option`, parse, epilog, the checks (flags error, lock entry dropped) | `api/manifest.py`, `cli.py` | ≈ 35 |
| derived planned options | `plans.py` | ≈ 5 |
| `ResolvesRefs`, `Resolved` | `api/hooks.py`, `api/` | ≈ 14 |
| **total** | | **≈ 125** (cap 6000) |

Plugins: `std.weights.hf` ≈ 30, `std.weights.gguf` ≈ 45 (the tag rule a pure function), `frontends.http` ≈ 20,
manifests ≈ 35 TOML lines; `check_locks.py` gains the name rule (≈ 3 lines).

**Tests** (CPU, lean: three extended, two new, no network):
- `core/tests/test_plans.py`, the parametrized `serve_history` case, extended: per case the `holon models` row
  (verified, borrowed, outdated, none, the unreadable-history warning) and D-repo (`org/repo` → its locked name in
  "lock", unchanged in "borrowed"; a local `org/repo` directory stays a path).
- `core/tests/test_plugins.py`, **new**, parametrized over toy weight-source plugins with the hook: an existing path
  asks none; one claim fetches once and the other never; a claim larger than the GPU is refused before its fetch,
  and fetched with `--no-size-check` or on a CPU device; none raises with both hints; two raise naming `--source`; `source` asks one.
- `core/tests/test_plugins.py`, **new**, `holon serve --help` and the option checks over toy manifests: every
  `EngineConfig` field has a flag and its help, the options and slots are listed; an undeclared key warns and stays in
  `plugin_options`; a flag outside one of two declarers' `choices` fails; a lock value outside them drops that history
  entry with a warning; `key=` passes; the planned options leave out a key declared only by a frontend plugin and keep
  an undeclared one. `core/tests/test_api.py`'s manifest example gains an `[options]` stanza.
- `plugins/tests/test_gguf_source.py`, extended: the tag rule over file lists (plain over `UD-`, case, a directory,
  projectors excluded, splits refused, an unknown tag lists the tags, the `repo/stem` form found in a subdirectory),
  and both sources' claims with the hub listing monkeypatched: HF claims only with `config.json` and safetensors, GGUF
  a bare repository only with `.gguf` files, so a GGUF repository with a `config.json` has one claimant; a claim
  downloads nothing and reports the size; a connection error falls back to the cache listing, and with both failing
  the hint names `HF_HUB_OFFLINE=1`.
- `plugins/tests/test_frontend_http.py`, extended: the short id and `http.model_name` in `/v1/models`, and the hint
  text.

**Checks:** `scripts/ci/run_all.sh` green: the denylist with `huggingface`, core size, `check_abi --base main`,
`manifest-deps` (huggingface-hub in std's `pyproject.toml`), `uv lock --check`, `check_locks` (with the name rule),
`check_docs`, `check_private`. The CPU tests of `core`, `plugins/tests/test_gguf_source.py` and `test_frontend_http.py`.
One recorded end-to-end run on the A5000, each heavy command memory-capped: `holon serve unsloth/Qwen3-0.6B-GGUF:Q4_K_M`
and `holon serve Qwen/Qwen3-0.6B` download (the chosen file logged) and answer a chat completion; the same with
`HF_HUB_OFFLINE=1`; a reference larger than 24 GiB is refused before downloading; the hint's id matches `/v1/models`;
`holon models` against a temporary plans directory holding a tools-written lock for this card shows it verified, and
D-repo picks it.

| # | work | checks | days |
|---|---|---|---|
| 1 | this design through the release panel (rev 1 reviewed; rev 2 re-check) | clean | 0.5 |
| 2 | items 1–5, D-repo, the declarations except `gqa` / `std.ops`, tests, documents | CPU tests, `run_all.sh` | 2.5 |
| 3 | end-to-end run (above) | recorded here | 0.5 |
| 4 | panel code review loop | clean | 0.5–1 |
| 5 | after M0.7's steps 2–3: `kv.format` and `quant.act` declared, on M0.7's branch, before its propagation | `run_all.sh` | 0.1 |

**≈ 4–4.5 days**, beside M0.7, off the release's critical path.

### 14.12 Risks

- **Re-uploaded hub files.** A repository that replaces a file in place (quantizers do) yields a new build: no lock,
  defaults, logged as "no current lock". The remedy is a propagation. A revision in the reference could come later
  (additive; `@` is taken by model ids, so another separator).
- **A file that fits the card's memory but not with its KV cache** passes the size refusal (it compares weights with
  total memory only) and fails its start as today (`OutOfUnits`, at once).
- **`choices` drifting from code.** A manifest's list narrower than the plugin's would refuse a valid flag value.
  Mitigated by declaring `choices` only for the two closed sets and by the error naming the manifest.
- **Disk and gated repositories:** the HF cache grows (README), and gated repositories need `hf auth login` (the
  library's error, with that hint).

### 14.13 Review of rev 1: findings and decisions, resolved

| # | finding or question (reviewer: S = serving / ABI, A = architecture / scope; L = lead) | resolution |
|---|---|---|
| L1 | Keep the bare-repo rule? | Kept; the order is explicit: an existing local path, then D-repo, then the plugins (§14.3) |
| L2 | The shorter `/v1/models` id | Accepted: the frontend ignores a request's `model`, so clients keep working (§14.7) |
| L3 | An undeclared option: warn or error? | Warns; the key still enters `plugin_options`, so the plan diff shows it (§14.4) |
| L4 | Frontend-only options are planned | Derived, not declared: `plan_of` skips an option exactly when every declaring plugin is in the `frontend` slot; undeclared keys stay planned; no new ABI key (§14.2) |
| 1 | S, A: bare-repo claims ambiguous for GGUF repositories that ship a `config.json` | HF claims only with `config.json` and safetensors files or an index; GGUF claims a bare repository only with `.gguf` files (§14.3) |
| 2 | Downloads start before a too-large file is known | The claim reports the listing's total size; the core refuses above the GPU's total memory before fetching (§14.3, ≈ 5 lines) |
| 3 | `choices` checks underspecified | A flag outside is an error; a lock value outside warns and drops that history entry, so `[defaults + flags]` serves; with several declarers the value must be in every declarer's `choices` (§14.4) |
| 4 | Offline and network behaviour | The chosen file is always logged; a connection error falls back to the cache listing, then to a hint naming `HF_HUB_OFFLINE=1`; offline, a tag sees only cached files; `repo/stem` searches every directory, since the GGUF source's name drops it (§14.3) |
| 5 | Names `holon models` prints must be servable | `check_locks.py` requires `<org>/<repo>[/<stem>]` names for committed locks; other rows are marked "not servable by name" (§14.5) |
| 6 | The hub import made serving a local path depend on the library | Imported lazily inside `resolve`; declared in std's `pyproject.toml` for `manifest-deps`, not in the manifests' `[dependencies] python` (§14.3) |
| 7 | ABI items incomplete | `hooks.ResolvesRefs` and `Resolved`, `Option` exported, `Manifest.options` appended last, the `[options]` section; design §4.1 states that a manifest using a later option key raises its `core_abi` floor (§14.4, §14.10) |
| 8 | Host-local and tree locks: which wins | Ranked across directories: a current, verified host-local lock, then the tree's verified lock, then borrowed; an outdated host-local lock falls through (§14.9 item 5) |
| 9 | `holon tune` and the agent-facing `tune.submit` | One tuner, one judge gate, one lock writer; two entry points (§14.9 item 6) |

**Rev 2 re-check (both reviewers confirmed; three small items, folded in):** the size refusal gains `--no-size-check`
and applies only on CUDA with `vram_gib > 0` and a known size, compared in bytes (§14.3); the offload row (§14.8);
the CPU case in the `resolve_ref` test (§14.11); design §4.3's sentence on declared options (§14.10).

Rev 1's `check_docs.py` misses (two commands in `docs/adding-a-model.md`) came from running it outside the project
environment; in the project venv it passes.

### 14.14 As built (2026-10-15)

Commits 821903f (the step), 8f8cfe7 and d620462 (review rounds 1–2), and the commit adding this section (round 3). The design
above holds; what follows records results, deviations and decisions.

**Built.**
- **Core.** `assembly.claim_ref` asks the weight-source plugins' `resolve` hooks; `cli.local_model` orders the
  resolution (an existing path, `repo_ref`, the claim, the log line, the size refusal, the fetch); `cli.locked_models`
  and `models_table` give `holon models`; `engine._setting` puts each `EngineConfig` field's help in its metadata, and
  `engine_flags` uses it; `cli.plugins_help` is `holon serve --help`'s epilog; `cli.check_flags` and `checked_options`
  are the option checks; `plans.frontend_options` keeps frontend-only options out of plans. `holon.api` gains
  `Resolved`, `hooks.ResolvesRefs`, `Option` and `Manifest.options` (manifest section `[options]`).
- **Plugins.** `std/holon_std/weights/{hf,gguf}/hub.py` with each plugin's `resolve`; `[options]` in seven manifests
  (`std.moe`, `std.state.recurrent`, `std.spec.fixed_k`, `drafters.ngram`, `drafters.dflash`, `grammar.xgrammar`,
  `frontends.http`; 13 keys); `frontends.http`: `served_id`, `http.model_name`, `hint` and `_Server.startup`.
- **CI and documents.** `check_locks.py`'s servable-name rule (every committed model id passes); `huggingface` in the
  core denylist; `core/abi.txt`; std's `pyproject.toml` declares `huggingface-hub>=1.0` (`uv.lock` +2 lines, the plans'
  `uv_lock` hash unchanged: `sha256:922c0a6d…`); README (quickstart, references, served id, Claude Code, options,
  Docker with a hub cache volume, `<file>` → `<model>`), D8's quickstart block, design §4.1, §4.3, §5.5 and §7.8.

**Deviations from the design, each decided here.**
- **Hooks raise `ValueError`, not `PluginError`.** Plugins import only `holon.api`, and `PluginError` lives in
  `holon.core` (§14.3 updated). `local_model` turns the core's resulting `PluginError` into a one-line exit, so a
  reference no source resolves prints its hints, not a traceback (found in the end-to-end run).
- **Function names:** `cli.local_model` and `assembly.claim_ref` instead of one `assembly.resolve_ref`, since the order,
  the log and the size refusal are `holon serve`'s, and asking the plugins is assembly's. The log line reads
  `holon: <ref>: <what> (<size>)`.
- **The repository rule applies under `--no-lock` and to `<org>/<repo>` references only.** It names a file, and with
  the latter, `holon serve Qwen` is never expanded to an organisation's one locked model.
- **Option flags are checked before anything is fetched** (`check_flags`; review M1), not on the layered history.
- **`holon models` uses `holon serve`'s own lookup** (`plans.history_id`, then `nearest_class`; review M2), so a
  build under two model ids is "unreadable" in both.
- **The slot listing** covers every slot not bound per site, except the model and the frontend (drafters included).
- **`http.drop_billing_header` declares no `choices`:** the plugin accepts `false`, `0`, `no`, `off` and TOML booleans,
  so a closed list would refuse valid values.
- **The two sources each have their own `listing`** (≈ 20 lines each). They are independent units pinned by `src`; a
  module shared outside both would change neither pin.
- **Cache messages:** when the hub is unreachable or offline, a cached snapshot a source cannot use is reported by that
  source (what it lacks), not declined silently (review M4).
- **`Option` checks its types:** `choices` a list of strings, `default` a string.
- **The ABI listing was written directly at 0.2**, not through `check_abi.py --write`: that refuses additions without
  a bump relative to the committed listing, which is already M0.7's unmerged 0.2. `check_abi.py --base main`
  (0.1 → 0.2) passes. When M0.7 merges, the listing is regenerated anyway.

**Core size:** 4129 → 4321 code lines (+192 against the ≈ 125 estimated). By file: `cli.py` +127 (`holon models`,
the repository rule, the resolution order, the option checks, the epilog, help in `engine_flags`), `assembly.py` +18,
`holon.api` +34 (`__init__` +11, mostly the multi-line import ruff requires; `manifest.py` +16; `model.py` +5;
`hooks.py` +2), `plans.py` +7, `engine.py` +6. The cap is 6000.

**Tests** (CPU, no network; the two new tests are the `test_plugins.py` ones):
- `core/tests/test_plans.py`, the `serve_history` case: the `holon models` rows per case, "not servable by name", the
  repository rule (one verified, several verified, whole segments, an organisation, a local `org/repo` directory).
- `core/tests/test_plugins.py`, new: `test_model_references_resolve_through_one_weight_source` (8 cases: a path, one
  claim, the size refusal, `--no-size-check`, a CPU device, none with both hints, two, `--source`) and
  `test_serve_help_and_option_checks_follow_the_declarations` (help for every field, options and slots listed, a bad
  flag fails before any fetch through `cli.main`, an undeclared key warned and kept, a lock dropped, frontend options
  unplanned).
- `core/tests/test_api.py`: the manifest example's `[options]`, three rejection cases.
- `plugins/tests/test_gguf_source.py`: `test_hub_references_are_claimed_by_one_source` (14 cases, each source called
  separately); the module's skip moved into the `source` fixture, so this test runs without the cached Qwen3-8B files.
- `plugins/tests/test_frontend_http.py`, the `/v1/models` test: the short id, `http.model_name` in `/v1/models` and a
  response, the hint (IPv6 included).
- **Every test bites:** 18 mutations of the new code (the size refusal, the all-declarers rule, the frontend-only rule,
  the lock drop, whole segments, the path check, the tag rule, top-level HF files, the GGUF bare claim, the short id,
  the flag-check order, `history_id`, the HF cache hint, HF's safetensors requirement, HF's cache fallback, `Option`'s
  types, the split-set rule twice) each fail a test.

**Suite and CI** (RTX A5000 ×2, the worktree's uv venv): `pytest core std plugins judge testing tools` gave 1224
passed and 65 skipped in 18 min 47 s at the final code (1219 / 65 at 821903f, before the review fixes);
`scripts/ci/run_all.sh main` is green (import contracts 8 kept).

**End to end** (RTX A5000, GPU 0, memory-capped):
- `holon serve unsloth/Qwen3-0.6B-GGUF:Q4_K_M` downloaded the file (0.4 GiB, logged), served defaults (no lock), printed
  the hint with `Qwen3-0.6B-Q4_K_M`, which `/v1/models` serves; an Anthropic message (sent with `"model": "whatever"`)
  and a chat completion answered.
- `holon serve Qwen/Qwen3-0.6B` (7 files, 1.4 GiB) answered a chat completion; both again with `HF_HUB_OFFLINE=1`.
- `holon serve unsloth/Qwen3-30B-A3B-GGUF:Q8_0` was refused before downloading (30.3 GiB, more than 24 GiB, naming
  `--no-size-check`); with `--plugin-options quant.act=a16` it exits on the flag first (`not one of a4, a8 (std.moe)`).
- `holon serve unsloth/Qwen3.8-27B-GGUF` on the A5000 (no verified lock) exits listing the repository's files.
- `holon models` lists the ten committed models, all "outdated" (every lock pins pre-M0.7 code until M0.7's
  propagation, §13.2). Against a temporary plans directory holding a tools-written lock for Qwen3-0.6B on this card
  (`holon_tools propagate --new-model --approve`, L2 PASS; never committed), it shows "this card's lock 1; context
  4096, 8 sequences", and `holon serve Qwen/Qwen3-0.6B --plans <it>` starts from that lock with no difference from it.

**Review** (one reviewer, the whole change): round 1, no major, 6 minor (M1 flags checked after the download; M2
`holon models` disagreeing with `holon serve` on a two-id build; M3 the repository rule off under `--no-lock`; M4 a
partial cache declined silently; M5 the hub test not checking HF's claim or cache fallback; M6 documents drifting) and 5
nits (slots and several-verified untested, `Option` types, a single file beside a split set, `listing`'s messages,
the hint's IPv6 host and port), all fixed; round 2, one nit (the split-set rule broader than its comment), fixed with
two findings of the end-to-end run (the traceback, an organisation expanded); round 3, two nits (a trailing `/` no longer
stripped, §14.3's text), fixed; round 4 confirmed clean (one wording nit, fixed).

**For the lead.**
1. Core +192 lines against ≈ 125 (breakdown above); 4321 / 6000.
2. The ABI listing at 0.2 was written directly (above); M0.7's merge regenerates it.
3. The follow-up on M0.7's branch after its steps 2–3, before its propagation: `[options]` for `std.mixers.gqa`
   (`kv.format`: `q8_0`, `fp8`; default the model dtype) and `std.ops` (`quant.act`: `a4`, `a8`; default `a8`). Until
   then those two keys draw the undeclared-option warning on the development branches.
4. This branch changes the `src` of the core, `std.weights.*` and the six lock-pinned plugins gaining `[options]`
   (`frontends.http`, the seventh, is pinned by no lock): it must merge into M0.7's branch before
   M0.7's closing propagation, which then covers it. No lock was written or committed here.

## 15. As built (step 6)

Branch `release6` from local `main` at 47429a0 (M0.7 merged); commits b38d6c7 (the step) and the ones after it. The
design above holds; what follows records what was cut, the freeze, the corrections and the demo task. Not done here,
as planned: the panel review of steps 1–6 and the squashed public root (the lead's), and no lock was written or
propagated.

### 15.1 Cut before the freeze

Every name in `core/abi.txt` was checked for a use (read or written by the core, std, plugins, testing, the judge,
the tools, bench or scripts; implemented by a plugin; set by a manifest). A name only declared, or only exercised by
`core/tests`, went; re-adding one later is a minor bump.

- **As listed in step 6's row (M0.7 §17):**
  - the 13 `SLOTS` with no plugin, which the core never loads: `schedule_policy`, `platform`, `launcher`,
    `eviction_policy`, `placement_policy`, `codec`, `plan_transform`, `checkpoint_policy`, `tier`, `mover`,
    `encoder`, `connector`, `exporter`. Nothing in the tree used one (one test manifest example named
    `placement_policy`; it became a `draft_policy` example, as design §4.1's did). `SLOTS` keeps `model`, `sampler`,
    `verifier`, `draft_policy` (single), `unit`, `kernel`, `state_kind`, `weight_source` (per site),
    `logits_processor` (ordered multi), `frontend`, `drafter` (multi);
  - `SchedulePolicy`;
  - manifest `[tunables]`, `Manifest.tunables`, `Tunable` and `Binding.tunables`. No lock carries tunables
    (canonical terms omit defaulted fields), so no committed `plan_id` moves: `check_locks.py` passes.
- **Found by the sweep, also cut:**

  | where | cut | why nothing needed it |
  |---|---|---|
  | `holon.api` | `PlanTransform`, `Platform`, `Ledger`, `Event` (`api/telemetry.py`), `Scale` | the first two belonged to cut slots; the device layer is the core's `holon.core.platform`, which never implemented `Platform`; plugins get pools through `BuildCtx.pool`, never a ledger; no `Event` is built; no `Scale` is |
  | manifest | `variant_of`, `implements`, `graph_policy` (and its vocabulary), `shape_axes` | set by no manifest, read by nothing |
  | `Constraint` | `cpu_isa`, `libs`, and `allows`' `isa` argument | never set; `libs` was not even checked |
  | `KernelSpec` | `in_layouts`, `out_layouts`; the prop `batch_invariant` | two kernel plugins set `in_layouts`, nothing read it (layouts are facts `supports` checks) |
  | `Scheme`, `Layout` | `scales`, `zero_point`; `params`, `readable_by` | every scheme and layout is built from its two names |
  | state | `HitLen.cost`, `Payload.tier`, `SeqView.slot`; `StateKind.reclaimable` and `fork` (and std's two stubs) | the tiers are v1's; the core never called either required member |
  | batches | `ShapeKey` and `BatchPlan.shape`, `TokenSrc.DEVICE_DRAFT`, `Seq.ext`, `BatchPlan.ext`, `DeviceBatch.meta` | never read (`BatchPlan.shape` was only written by `batching.host_plan`, `BatchPlan.ext` only copied by `batching.padded`); the core keys its graphs itself |
  | model, frontend | `ModelSpec.position_fn`, `Tokenizer.eos_token_ids` (and the three bundles' assignments) | the stop set is the weight source's (`WeightSource.eos_token_ids`) |
  | memory | `Pool.device`, `segments`, `base`, `capacity_dev`, `incref`, `decref`, `copy`; `Registry.items` | the core uses them on its own pools; no plugin does |
  | hooks | `Rewindable.rewind`'s `path` | no caller passes it and its implementers do not accept it; tree verification brings it back as a new optional hook, not a parameter (a new parameter changes a plugin-implemented signature: major) |

- **Added, a declaration the code already relied on:** `Pool.num_free` (the recurrent state kind reads it before
  publishing a checkpoint).
- **Kept with no reader, and why:**
  - `[resources] requires`: declared by every kernel manifest (the op contracts each implements). Cutting it is a
    nine-manifest churn, and it is what a resolver check of design invariant 7 would read.
  - `after` / `before` and the `ordered_multi` cardinality: the declared contract of the `logits_processor` slot (v0
    loads at most one). Changing a slot's cardinality after 1.0 would be major; ordering a second processor needs
    them.
  - `numerics` and its four values: set by most manifests; the judge's tiers are defined per class (§10.3).
  - `derived_from`, `license`, `[provenance]`: lineage and third-party metadata (`docs/adding-a-model.md`,
    `check_provenance.py` reads the provenance keys).
  - `ERROR_KINDS`, `FINISH_REASONS`: the frozen vocabularies of §7 (their values are used).
  - `to_term`: it defines `canonical_hash`'s encoding, which §7 freezes. `Ref.param` and `Ref.with_`: the
    recipe-side API of `Ref` (tests and one plugin test use them).
  - `Status.PREEMPTED` (the core sets it; plugins see `Seq.status`); `SamplingParams`' three penalties (the frontend
    carries them, the core refuses non-neutral values); `Slot.draft_slots` and `WeightMap.root` (implemented by the
    core and toy-tested, set by no plugin yet); `LayerDecl.kind` (set by the recipes, part of their hash);
    `Registries.ops` (std registers each op contract's oracle, checked against its manifest); `graph_safe` (recorded
    in lock props).

Core: 4369 → 4235 code lines.

### 15.2 ABI 1.0

- `holon.api.ABI = "1.0"`. All 38 manifests (first-party, `holonlm-testing`'s and the test fixtures') and the
  manifests written inline by tests declare `core_abi = ">=1.0,<2.0"`.
- `core/abi.txt` regenerated (+89 / −120 against main): the removals and reorderings of §15.1, `Pool.num_free`, and,
  after the panel (§15.9), set-like items listed one element per line.
  The contracts listings (`std/contracts.txt`, `plugins/contracts.txt`) do not change. `check_abi.py --base main`
  accepts it as a major bump (0.2 → 1.0); `check_abi.py` no longer renders the `GraphPolicy` vocabulary.
- **Every committed lock is outdated, as expected:** the core's `src`, every manifest's (each changed its
  `core_abi`) and the two kernel plugins', std's two state kinds' and the three bundles' sources moved, and the plans'
  `uv_lock` hash moves with the distributions' versions (sha256:922c0a6d… → sha256:609cd0a7…). No `plan_id` of a
  committed lock changes (`check_locks.py`). Release step 7 propagates on the public root.

### 15.3 Versions

Every distribution (`holonlm`, `-std`, `-plugins`, `-testing`, `-judge`, `-tools`) and the root `holon-workspace` are
0.1.0, and so is `holon.__version__` (the core's unit version in locks). `uv.lock` was updated (seven version lines).
Plugin manifests keep their own unit versions (1.0.0).

### 15.4 Corrections

- **design.md**
  - §3: the Plan has no tunables; `Event` is planned; `Platform` is the core's device layer, not a slot.
  - §4.1: the manifest example is a `draft_policy` plugin with ABI 1.0's sections only (`core/tests/test_api.py` parses
    it from the design); resources are matched as strings and registrations checked against `provides` (the claim of a
    Protocol check after import was never built).
  - §4.2: the slot table is ABI 1.0's; the planned slots are listed as future minor additions.
  - §4.3: per-site bindings carry configs, not tunables.
  - §4.4: rollback in an install (an older lock pins other code: `holon serve` serves the newest lock pinning the
    installed code and falls back to defaults, D2); `holon serve` starts from the lock; `plan_of` lives in
    `holon.core.plans`.
  - §4.5: the frozen ABI, the bump rule, the contracts listings and the cut list; "pre-1.0, may break" is gone.
  - §5–§9: the sketches no longer present cut names as current (`KernelSpec`, `Scheme` / `Layout`, the `Pool` plugins
    see, `StateKind`, `Seq`, `BatchPlan` / `ShapeKey`, graph policies, `Event`, §8's platform and launcher rows, §9's
    offload, checkpoint, encoder and plan-transform rows).
- **plan.md §1**: the runtime dependencies per distribution, transformers among them (the GGUF tokenizer), and
  today's vendored kernels, test-only packages and baselines.
- **docs/adding-a-model.md**: the `core_abi` example. The README has no `core_abi` example (nothing to change).
- M0.7 §17's other notes stay where §17.7 put them: the README's lock table (step 10), GQA's `_fusion` (the release
  review).

### 15.5 The demo task (D7), pre-registered

`bench/demo/` (stdlib-only project; nothing here runs it — step 9 does). It is a small three-module fix chosen to fit
the 32K context, not a claim of general agent capability (the README says so).
- **The project** (`bench/demo/project/`): a shop in three modules — `money` (prices as integer cents), `cart`,
  `discounts` (rules and checkout) — and 40 tests. Seeded: `money.format_cents` drops the zero padding of the cents
  (`5` → `0.5`), which fails tests in all three modules; `discounts.buy_n_get_one`, named in the module's docstring
  and specified by the tests, is missing. As committed: **17 failed, 23 passed**. Its digest is registered in
  `run.py` (`PROJECT_DIGEST`); the harness reads the project once and refuses another.
- **The run** (`bench/demo/run.py`): a fresh copy written from that in-memory project, and a disposable venv with the
  harness's pytest version (installed offline by uv) first on the agent's `PATH`. Claude Code **2.1.300** (M0.5's
  session ran 2.1.296): the binary is resolved once to its real path, its sha256 recorded, and path, hash and version
  re-checked before every run. `claude -p <prompt> --allowedTools Read,Write,Edit,Bash,Glob,Grep --output-format
  stream-json --verbose`, M0.5's environment variables plus the Sonnet and Opus model aliases and
  `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC` (all recorded in `client.env`), auto-update off, a fresh
  `CLAUDE_CONFIG_DIR` (no user settings, memory or hooks), `NO_PROXY` for the local proxy; through
  `bench/capture_proxy.py`; stopped after 30 minutes (its whole process group, also after a normal end) and scored
  as it stands. The prompt: "The Python package `shop` in the current directory fails its tests. Fix the package so
  that `python -m pytest` passes, without editing any file under tests/. When the tests pass, reply with the single
  word done."
- **Success:** after the session, pytest runs in a fresh directory holding only the agent's `shop/` (regular files;
  links are not followed) and the harness's own copy of `tests/`, under the harness's interpreter with its empty
  configuration (`-c`), `--noconftest`, no plugin autoload and no `PYTEST_*` variables; all 40 must pass. Anything
  failing while scoring (a timeout past 10 minutes included) is a recorded miss. The interpreter and every module file
  a scoring run imports (pytest, pluggy, iniconfig, the stdlib, extension modules: found by a probe scored the same
  way), with the site directories' `*.pth` files and any `sitecustomize` / `usercustomize`, are hashed before and
  after each run against one reference digest taken before run 1 (`scorer_reference`, recorded): a run starts only
  in that environment, and a change makes the run **invalid** — not scored, **counted as a miss and published** —
  and stops the series until the environment is restored. A resumed series passes `--reference` (which must match)
  and appends to its own record (`--out`, which must exist and name the same reference, class, commit, model, served
  plan and binary) for the runs left of the plan it recorded (`runs_planned`; `--runs` is then ignored); a new
  series never overwrites a record. Scoring reads bytecode only from a fresh empty
  `PYTHONPYCACHEPREFIX`, so a crafted `.pyc` beside a hashed source is never run. Not guarded: the shared libraries
  the interpreter links (libpython, libc). Before every run the harness's own files (`bench/`, `scripts/ci/`) must
  equal their commit (`git status`), and the run's venv, as made and before its session, must hold the first run's
  pytest (its site-packages' digest); otherwise the series stops before the run. What an agent then installs into
  its own venv is allowed and recorded. Before the runs the harness checks that the project fails as registered and
  that `--gpu-class` is the served lock's class. Runs are sequential on one server, so a later run may
  hit the prefix cache an earlier one filled (the same system prompt and tools), as a user's next session would.
- **Recorded per run:** success (a miss when invalid), validity, the scoring's counts, the venv's digest as made and
  whether the agent changed it, turns (derived from the stream's assistant messages when a session times out), wall
  time, exit and result kind, the tool-call sequence (tool, a file tool's path inside the project or `<outside>`, the
  program a Bash command starts, whether it failed — no file text, no arguments), the final `shop/` diff (synthetic and
  public; withheld if `check_private`'s patterns match it), per request: status, input and output tokens, TTFT, TPOT,
  stop reason, a length refusal; counts per HTTP status, length refusals, the peak context, the distinct sampling
  parameters (`model`, `max_tokens`, `temperature`, `top_p`, `top_k`, `thinking`, `output_config`, `stop_sequences`),
  the tokens served and the prefix tokens saved (the engine's counter over the run). The record also has the served
  plan, its lock (from `plans/`) and the diff (host paths scrubbed), the binary's sha256, the project's digest and the
  commit. Each session's stream and the proxy's lines are kept under the gitignored `bench/.out/demo/`.
- **Step 9's procedure:** `holon serve` the class's 27B lock; one uncounted smoke run of M0.5's Claude Code session
  task at 2.1.300 against it; then the 3 counted runs.
- **`capture_proxy.py`** adds to each request's line `ttft_ms` (the first `content_block_delta`), `total_ms`,
  `usage` (a whole message's, or a stream's `message_start` and last `message_delta`), the stop reason, and for an
  error its type and whether it refused the prompt's length (`max_model_len`) — never its text.
- **Checked here, without the demo:** `tools/tests/test_demo_harness.py` (CPU, ≈ 2 s): the scoring (seeded 23 / 17;
  a module linked to a fixed copy outside is not followed; a dangling link and a failing pytest are recorded misses),
  the project and environment digests, the binary pin, the record's helpers (scrubbing, the lock's class, turns from
  the stream, per-request counts, the `shop/` diff and its private check), the proxy's facts and the per-run venv.
  By hand: `run.py --score` gives 40 / 0 on a reference fix (not committed); with it, deleting a test file, adding a
  skipping test, a clearing `conftest.py` and a `pytest.ini` with `-k nothing` score as the fix alone; on the unfixed
  project those two and a `pytest.py` faking a JUnit pass (in the root or in `shop/`) still score 23 / 17. A whole run
  against a fake server and a fake `claude` recorded the fields above, found the run's venv first on the session's
  `PATH` (`command -v pytest`), refused a `--gpu-class` other than the lock's, and killed a background process the
  session left.

### 15.6 Review

One reviewer on the aggregate change. Round 1: one major and six minors, all fixed.
- **Major:** a `pytest.py` in the project root replaced pytest in the scoring and could fake a pass. The scoring now
  runs in a fresh directory with only the agent's `shop/` and the harness's tests (the prompt names the package).
- **Minors:**
  - a scoring timeout or a session's leftover processes could crash the harness; the scoring timeout is now a miss,
    and the session's process group is killed (its output goes to a file, so a leftover process cannot hold a pipe:
    found by the fake run);
  - `ShapeKey` and `BatchPlan.shape` were never read: cut;
  - `count_tokens` requests counted as messages;
  - a Bash command's `VAR=value` prefix and a `~` path could leak argument text;
  - the proxy's usage parse raised on a non-dict `message`, and the harness went on when the proxy did not start;
  - design §3 and §4.2–§4.3 still named cut features.
- **Nits, fixed:** the design example's `derived_from` follows the `<plugin>@<commit>` form, and the test now
  parses the example from `docs/design.md` itself; a blank line in `hooks.py`; the record lists the client's
  environment variables; the harness's server calls bypass HTTP proxies; the warm prefix cache across runs is stated.

Round 2 confirmed every fix: no findings beyond two wording nits in this section, fixed.

### 15.7 Suite and CI

Full GPU suite (two RTX A5000s, the worktree's uv venv; `pytest core std plugins judge testing tools`): at b604aa9,
**1210 passed, 65 skipped in 20 min 45 s**, the same counts as M0.7's close (1210 / 65); after the panel's fixes, at
6e0f1fa, **1217 passed, 65 skipped in 17 min 51 s** (+7: the per-element listing case and the six harness tests, ≈ 3 s
on the CPU). An earlier run at b38d6c7 had one failure, `tools/tests/test_gpu_lock.py::test_first_lock_for_qwen3_0_6b`,
caused by editing core files while it ran (the core's `src` moved between its two `propagate` calls); it passes since.
`scripts/ci/run_all.sh main` is green at every commit (import contracts 8 kept). At 826287b (the after-run guard, the
last change outside `bench/` and its test): **1217 passed, 65 skipped in 16 min 59 s**; `run_all.sh main` green.

### 15.8 For the lead

1. **The cut went beyond step 6's row** (§15.1's second table: about 40 more names, each with no reader), because
   removing any of them after 1.0 would be a major bump. The kept list records each name kept with no reader and the
   reason. `Pool.num_free` was added: a plugin read it without the ABI declaring it.
2. **Every committed lock is outdated** (the core's and every plugin's `src`, and `uv_lock` 922c0a6d… → 609cd0a7…);
   `holon serve` serves borrowed or default settings until step 7's propagation. No `plan_id` moved.
3. **Claude Code is pinned at 2.1.300** (the version installed now; M0.5 ran 2.1.296). Step 9 needs exactly that
   version (`DISABLE_AUTOUPDATER` is set for the runs, but the installed binary must be at it). The fresh
   `CLAUDE_CONFIG_DIR` keeps the user's settings, memory and hooks out; it has not met a real session yet (step 9).
4. **The README has no `core_abi` example**, so nothing there changed. Its lock table is still M0.7 §17.7's item for
   step 10.
5. Not done here, as instructed: the squashed public root, any change to `main`, any lock.

### 15.9 The release panel's review of step 6 (fixed)

Design goal 2 and D8 are held unchanged while the owner chooses between restating them and restoring the v1 seam
declarations into ABI 1.0. Every code fix below has a test that fails without it (run against the code before it).
- **Per-element ABI listing (major).** `check_abi.py` lists `SLOTS` (`const SLOTS` then `  tier: multi`-style lines),
  `KERNEL_PROPS`, each manifest section's keys, `Numerics`, the IPC frames and the failure kinds one element per line:
  an element added is a minor addition, one removed or changed (a slot's cardinality included) major. Ordered
  vocabularies a consumer switches over (`FINISH_REASONS`, `ERROR_KINDS`) and the `ParseEvent` union stay one line,
  so any new value is major (a comment says why). `core/abi.txt` regenerated at 1.0. Test:
  `test_api.py::test_set_like_abi_items_change_by_element` (a slot, a manifest key and a prop added at 1.1 pass; a
  removal, a cardinality change and a new finish reason need a major).
- **Serving minors:** design §4.5 and §15.1 say `Rewindable.rewind`'s `path` returns as a new optional hook, not a
  parameter (a parameter changes a plugin-implemented signature: major); design §4.3 step 2 is defaults < lock <
  flags, the typed deployment file future.
- **Architecture:** design §6.1 (the Pool's VMM members, a pool's device), §6.3 (a payload's tier, `hit_length`'s
  cost), §6.4 (load futures, per-layer events), §7.7 (`StateKind.fork`; `n > 1` shares by prefix adoption), §8 (the
  `launcher` slot and TP's core side, PD's and multi-device memory's seams) and §9's KV-offload row are marked
  "planned; returns as a minor ABI addition". The README describes the demo as what it is (above).
- **Demo harness (M1, major), isolation:** each run's own venv first on the agent's `PATH`; scoring by the harness's
  interpreter after its packages' digest check (a change: an invalid run); each run's project written from the
  in-memory copy checked against the registered digest.
- **Demo harness minors:** the binary pinned by real path, sha256 and version, re-checked before each run; the
  uncounted smoke in step 9's procedure (D7); per-request records, counts per status, length refusals, the peak
  context, turns from the stream on a timeout; the final `shop/` diff in the record (private-checked) and transcripts
  under `bench/.out/demo/`; `served.diff` scrubbed like the lock; any scoring failure a miss, links not followed,
  `--gpu-class` checked against the lock's class. Optional hidden report-only tests were not added.
  `scripts/ci/check_private.py` now runs its scan in `main()`, so the harness reuses its patterns.
- **Listing note:** enums (`Status`, `TokenSrc`, `Cardinality`) stay one member per line, as before, so a new member
  is a minor addition; `FINISH_REASONS` and `ERROR_KINDS` are vocabularies a client switches over, so a new value is
  major.
- **Review of the fixes** (the step's reviewer): no blocker or major; three minors fixed, each with a failing-first
  test in `tools/tests/test_demo_harness.py` — the recorded `shop/` diff followed a linked `shop/` (now refused, as
  scoring does); the digest guard covered only pytest and pluggy (now every module file a scoring run imports, found
  by a probe, and the interpreter; the shared libraries it links are stated as unguarded); a lock path outside
  `plans/` crashed the class check (now refused with its message). Its confirmation found one more minor, fixed with a
  failing-first test: the after-run digest could raise (a pytest the agent broke) and lose the run; now any failure
  there makes the run invalid, recorded and not scored (`invalid_after`). Nits fixed: hidden files (a local
  `.pytest_cache`) are outside the project digest; the "planned" phrasing in design §6 / §8 / §9; D7's success line.
- **The methodology reviewer's last harness items**, each with a failing-first test in
  `tools/tests/test_demo_harness.py`: one scorer reference digest per series, taken in `main()` before run 1, checked
  before and after every run (`check_environment`, `invalid_after`); an invalid run is a miss and published
  (`outcome`), and the series stops until the environment is restored; the harness refuses to start a run while
  `bench/` or `scripts/ci/` differ from the commit (`harness_dirty`) or when the run's venv holds another pytest than
  the first run's (`venv_digest`, the site-packages by relative path, without the dist-info `RECORD`s, which hash
  the venv's own scripts); `shop_diff()` records a failure instead of raising. CPU tests and `run_all.sh main` green;
  no GPU suite (bench-only change).
- **Their re-check of that round** (each with a failing-first test): the venv's digest is taken as made, before the
  session, and used both for the check against the first run's and in the record (`prepare_venv`); a change the
  agent makes to its own venv is recorded as "agent changed its venv" (`venv_change`), never a stop — before, an
  agent installing into its venv stopped run 2 falsely, and a removed venv could lose the run. Scoring and its probe
  read bytecode only from a fresh empty `PYTHONPYCACHEPREFIX` (a crafted `.pyc` matching its source's size and time
  is run without it, as the test shows). A resumed series (`--reference`) loads its `--out`, checks its scorer
  reference, class, commit and binary sha256, appends and runs only the runs left; without `--reference` an existing
  `--out` is refused (`series_record`). CPU tests and `run_all.sh main` green.
- **Two resume safeguards** (the reviewer's optional items, one commit, a failing-first test): the record stores
  `runs_planned` when the series starts, and a resume runs `runs_planned − len(runs)`, ignoring `--runs`; a resume
  also requires the record's `model` and `served.plan_id`. CPU tests and `run_all.sh main` green.


## 16. As built (step 7)

Branch `release7` from local `main` at 3642a61 (the squashed public root, ABI 1.0, versions 0.1.0). Every lock and
record of this step names a commit on that line. The release environment is `uv sync --frozen --group dev` from the
committed `uv.lock` (CPython 3.12.12, torch 2.11.0+cu130, Triton 3.6.0) inside a CUDA 13.0 container, one per desktop.

### 16.1 Method, fixed before any result was read

Revised once after the step's reviewer read rev 1 (928d993), still before any result. The findings are listed in
§16.8.

**Desktops.**
- Each desktop is clock-screened before its work and again after it, with `bench/clock_check.py` (60 s of 27B decode at
  batch 4).
- It is used only if its SM clock holds steady (p10 / p50 ≥ 0.9).
- The screen's GPU clock and slowdown share are recorded beside the desktop's evidence.

**Locks** (§6), in this order on each class.
1. **The development classes' locks are propagated with nothing added** (pins only). A trace that moves is explained.
2. **The cap-512 bars and D3's MTP rounds run** (below). They run at the class's own lock, or for a class without one
   at the nearest class's settings.
3. **The decided settings are written**, so each lock takes one approval per change:
   - `propagate` gives `chunk_with_decode` 512, or MTP, to a development lock whose bars pass;
   - each **R** cell is written with `tune --class <nearest class>`, using that lock's `[engine]` plus the decided
     settings. The 8B Q4_K_M comes from the rtx-5080's lock, every other cell from the rtx-4090's;
   - the 5090's bf16 lock is refreshed with `propagate` at step 1, gaining the grammar processor and
     `chunk_with_decode` 0.

   Each lock is checked by L2 against the floor of its path and KV format, and by `holon_judge snapshot --runs 3` at
   its settings and pins. It is approved only if both pass.

**`chunk_with_decode` (plan §8a.1's follow-up).**
- **The plain 27B locks keep 512** (M0.7 §15.4). The 3090 was not among M0.7's classes, so its 27B is measured here
  the same way.
- **The a8 8B / 30B GGUF locks are measured at 512 against today's 0.** This covers:
  - every class's 8B Q4_K_M and 30B Q4_K_M;
  - the 5080's 8B UD-IQ3_XXS and 30B UD-IQ3_XXS;
  - the 4090's 30B MXFP4_MOE.

  The command, per lock:
  ```
  python bench/run.py --name release7-cap-<cell> --gpu 0 --model <file> --lock <lock> --backends holon holon@m512 \
      --arm m512='--chunk-with-decode 512' --rounds 4 --traces bench/traces/agent.jsonl bench/traces/long_prompt.jsonl
  python bench/mixed_bars.py <record>
  ```
  The worker sizes `max_model_len` from the traces (30,208 tokens), as in M0.7. The locks' own contexts are 8K and 16K.
- **512 is set where every bar of §15.4's verdict passes:**
  - on the agent trace: max gap p99 and max ≥ 2×, the paired TTFT ≤ 1 + max(15%, 2 sd), and throughput within
    max(5%, 2 sd);
  - on the long prompt: TTFT p99 ≤ +25%;
  - and mixed steps > 0.

  Elsewhere the lock keeps 0, and the failing bar is recorded.
- **The bf16, NVFP4 and FP8 locks keep 0.** They are outside the follow-up, and only the a8 GGUF paths were measured.
- **Prediction:** the gap bars pass everywhere. On the 8B the long prompt's TTFT p99 is the bar at risk. It takes four
  times the steps, each paying the eager mixed step's fixed cost, on a model 3× faster than the 27B.

**MTP (D3), per 27B lock** (the 5080's UD-IQ3_S; the 4090's, 5090's and 3090's UD-Q4_K_M).
- **Trace.** `bench/traces/claude_code.jsonl` is new in `bench/workloads.py`.
  - It is the agent workload (M0.7's runner) with a 16,384-token shared system prompt, standing for Claude Code's
    system prompt and its 22 tool schemas.
  - 16 sessions of 11–17 turns each give 225 requests per round, with contexts up to 25.8K tokens. A session starts
    every 40 s on average, so about two overlap: several sessions, as D3 asks, without the trace saturating the
    slowest card.
  - Generations keep the agent workload's 24–192 tokens (mean 106). Claude Code's own turns run longer: its capture
    averaged ≈ 265 tokens per request, thinking included. This weighs the prefill-driven tail metrics more, a bias
    towards plain.
- **Text.** Prompts are cut from natural text (`--corpus judge/prompts/v1-qwen3_5.jsonl`, 33,396 tokens).
  - `bench/driver.py` now draws each session's turns as one walk through the corpus outside the system prompt's span,
    from a seeded offset (`workloads.appends`). Uniform-id traces are unchanged.
  - Before, each turn drew an independent offset, and 53% of the appended tokens repeated text already in the same
    context. That inflates MTP's acceptance.
  - Now at most 12.8% of a context's 16-grams repeat. The corpus itself repeats 8.2%, from its 24 prompts' chat
    framing.
- **Sampling.** Temperature 1.0. Claude Code's captured requests (`evidence/claude_code_session_rtx-5080_2026-10-14.json`)
  carry no `temperature`, `top_p` or `top_k`, so the Messages API's default of 1.0 applies, as in Holon's Anthropic
  frontend. top-k, top-p and min-p are off.
- **Arms.** Plain is the class's 27B lock (`chunk_with_decode` 512). MTP is that lock plus `--max-draft-tokens 3
  --slots drafter=drafters.mtp draft_policy=std.spec.fixed_k verifier=std.spec.greedy --plugin-options
  spec.k=1:3,3:2`.
  - These are M0.5's served flags and §15.5's. The k schedule is M0.4's.
  - MTP drafts deterministically. So the exact-match verifier accepts a draft d with probability p(d), the same law as
    rejection sampling's, while emitting exactly plain decoding's tokens.
  - `chunk_with_decode` stays the lock's 512, so the MTP arm runs M0.7's alternation (§15.5). This run is the
    alternation's measurement on this trace.
- **Command**, ABBA rounds of fresh processes (D3 says ABAB; run.py alternates the order each round, so neither arm
  always runs first). A run-on uses another `--name` and an even round count.
  ```
  python bench/run.py --name release7-mtp-<class>[-<part>] --gpu 0 --model <27B file> --lock <27B lock> \
      --backends holon holon@mtp --arm mtp='<the flags above>' --rounds 4 --temperature 1.0 \
      --corpus judge/prompts/v1-qwen3_5.jsonl --traces bench/traces/claude_code.jsonl
  python bench/mtp_decision.py <record> [<run-on record> ...]
  ```
- **The rule** (`bench/mtp_decision.py`).
  - **Per-round values:** the median TPOT, the p95 of the per-request maximum gap, and the p95 TTFT, each as the ratio
    MTP / plain.
  - **On** if 1 − mean(TPOT ratio) ≥ 5% and ≥ 2 SE, and for both the gap and the TTFT, mean + t(0.90, n − 1) · SE ≤
    1.10.
  - **One power estimate, from the first 4 rounds.** With their mean and sd held, it finds the fewest rounds (≤ 8) at
    which every criterion would pass. If a mean fails, or no count up to 8 passes, the lock stays plain. Otherwise the
    run goes on to that count, and the rule is applied once there.
  - **Never on** if a round lost an arm or ran fewer than 200 requests.
  - The records must agree (model, class, lock plan, trace, corpus, arms) and sample at 1.0.
  - **Also reported:** the pooled p99s, as point estimates; each arm's KV tokens (the 5080's 27B: 56,208 plain and
    37,392 with MTP in M0.7 step 4's records, `evidence/bench_m07-step4-{plain,mtp}_rtx-5080_*`); prefix tokens saved
    and preemptions; and MTP's accepted drafts per k = 3 verify row. The capture of M0.5's Claude Code session accepted
    1.78 (greedy verifier, Claude Code's own text). A trace accepting more is optimistic for MTP, and the record says
    so.
- **The lock's spelling.** A speculative lock records `chunk_with_decode` 512 explicitly, as measured.
- **Prediction: plain on every class.** TPOT improves well past 5%, but the p95 gap fails. Plain mixes decode rows into
  capped prefill steps, while MTP alternates, so its decoders wait a whole capped step (M0.7 §15.5: agent-trace gap p99
  799 against 240 ms on the 5080).
  - No cross-session prefix hit is possible. Checkpoints are taken at prompt ends (`recurrent.policy` `prompt_end`),
    and no prompt ends at 16,384. So each session's first turn is a cold ≈ 16.5K-token prefill, and the TTFT p95
    compares alternation with mixing on it. From §15.4–§15.5, the long-prompt TTFT p99 ratio MTP / plain at 512 was
    ≈ 1.05–1.12.
  - On the 5080, MTP's smaller KV pool adds evictions when two ≈ 25K-token sessions overlap.
