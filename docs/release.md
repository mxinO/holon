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

1. **Legibility** (§4): over n = 10 runs per side, Holon's median cost ≤ 0.70× a mini-sglang fork's for the same
   agent adding Mixtral, under **both** costings of §4.4, both sides passing the same check; else §4.7.
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
  3 runs per class. Success: the harness restores the tests from its own copy, then `pytest` passes. Recorded: turns,
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
  uv run hf download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-IQ3_S.gguf --local-dir models  # 16 GB; else UD-Q4_K_M
  uv run holon serve models/Qwen3.8-27B-UD-IQ3_S.gguf   # starts from this card's lock; the log names it
  ANTHROPIC_BASE_URL=http://127.0.0.1:8000 ANTHROPIC_AUTH_TOKEN=holon ANTHROPIC_MODEL=<id from /v1/models> \
  ANTHROPIC_DEFAULT_HAIKU_MODEL=<same> CLAUDE_CODE_SUBAGENT_MODEL=<same> claude
  ```
- **D9 — Lean tests.** No new suite: the `test_plans.py` case (D2); one `test_plugins.py` case (a missing or
  out-of-range `core_abi`); static scripts in `run_all.sh`: the `manifest-deps` import scan, `check_abi.py`, the guide
  check, `uv lock --check`, `check_locks.py` (§7). The clean install (step 10) is a scripted run per class, not CI. A
  CPU-only GitHub Actions run of `run_all.sh` and the CPU tests comes with publishing.

## 4. The legibility benchmark (plan §9, design D9)

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
| 5 | Legibility: 10 + 10 runs (the first 2 + 2 the pilot), 3 guide-free, verdict; on fail §4.5 / §4.7 | verdict record; owner informed | 3–4 |
| — | M0.7 closes: one propagation covering D1's environment, D2, the mechanism and grammar on the development classes | its gates; served plan = lock on rtx-5080 / 4090 (acceptance 3's form) | — |
| 6 | ABI 1.0, versions 0.1.0; design §4.1 / §4.4 / §4.5 and plan §1 corrections; demo task committed (D7); panel review of steps 1–6; the squashed public root created locally | review clean | 2 |
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
    - names and signatures, dataclass fields in order with types and defaults;
    - Protocol members, required or optional, and who implements each Protocol;
    - `EngineConfig`, the `BuildCtx.options` keys and the manifest schema;
    - the vocabularies, the IPC frame and failed kinds;
    - the `op.*` / `unit.*` contracts first-party plugins provide.

    It fails on drift and on undeclared `hasattr` / `getattr` literals (a planted one fails). `--write` requires the
    bump: an addition is minor; a removal, a change or a member added to a required plugin-implemented Protocol is
    major (below 1.0, the minor number). The rule is tested. The listing was regenerated twice inside step 2, before
    any lock.
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
