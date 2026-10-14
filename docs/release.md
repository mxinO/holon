# Release (v0) — packaging, locks, legibility benchmark, benchmark suite, ABI freeze: design and plan

Status: **revision 2, for review** (2026-10-14). Rev 1 (97c39ff) was reviewed by a panel of three (A: serving,
packaging, ABI; B: methodology; C: architecture, scope); every finding is resolved in §12. Scope: plan §2's Release row
("every v0 model: packaging, quickstart, rtx-5090 / rtx-3090 locks, an agent demo; the legibility benchmark (§9)
before the ABI freezes; the benchmark suite (decode, prefill, agent trace vs llama.cpp, FreeToken and SGLang where each
runs) on the release classes"), plan §9's legibility row (design D9 in `docs/m03-qwen38.md`, moved before the freeze
by the owner, 2026-10-14), design §4.4 / §4.6 (release classes; "a release commit requires every development and
release class refreshed"). Builds on M0.5 (`holon serve`, the HTTP frontend, the Claude Code session), M0.6
(`docs/m06-lowp.md`, in progress) and M0.7 (`docs/m07-perf.md`, rev 3.1), whose step 0 builds the benchmark runner
reused here. **Publishing — pushing to GitHub, making a repository public — needs the owner's go; this design only
prepares it (§8).**

## 1. Goal and acceptance

A user with an RTX 3090, 4090, 5080 or 5090 installs Holon at the release tag, serves a model at its tuned lock with
one command, and points an OpenAI or Anthropic client, or Claude Code, at it; every published number comes from an
evidence record. Acceptance (each a gate, §9):

1. **Legibility** (§4): Holon's median cost ≤ 0.70× a mini-sglang fork's for the same agent adding Mixtral, both
   sides passing the same check; else the pivot (§4.7) runs before anything freezes.
2. **ABI 1.0** (§7): declared, enforced at load, listed and checked in CI.
3. **Locks** (§6): every cell of the matrix written and approved at the release commit, in the release environment;
   `holon serve` reports a plan equal to the lock's on each class (§3 D2).
4. **Packaging** (D1): a clean install by the README alone, in a fresh environment, on each class serves a chat
   completion and an Anthropic message; a cold start compiles nothing but Triton kernels; the Dockerfile builds.
5. **Benchmarks** (§5): recorded on all four classes; every GGUF cell's in-process pp512 and tg128 (depth 0 and
   8192) ≥ 0.8× llama.cpp at the vendored pin (plan §9); all cells published, losses included.
6. **Agent demo** (D7): the pre-registered task, 3 runs each on the RTX 5080 and 4090; results published either way.
7. **History cleaned and the owner's go**, then publishing (§8).

Non-goals: wheels (PyPI) and a published image; merging the legibility runs' Mixtral packages; DFlash2 in locks or
the suite; new families or formats; v1's offload, VMM and self-improvement; GPUs outside sm_86 / 89 / 120.

## 2. Starting point (blockers found)

| # | Finding | Lands in |
|---|---|---|
| B1 | `holon serve` never reads a lock: tuned pins, KV format, checkpoints and speculation reach a user only as hand-typed flags (M0.5's 27B command has nine). `nearest_class` / `lock_overrides` are called only by the tools | D2 |
| B2 | Locks are found by `model_id` = display name (from the HF cache layout) + build hash: the same file downloaded with `--local-dir` never finds its lock | D2 |
| B3 | The root `pyproject.toml` is virtual (no dependencies, no build system): `uv sync --no-dev` installs nothing. The core imports torch, numpy, msgspec and zmq but declares `dependencies = []`; manifests list third-party imports no dist declares; the dev venv hides this through system site-packages. No `uv.lock` (locks record `uv_lock = "none"`), no console script, a transformers dev build | D1 |
| B4 | `kernels.ggml_cuda` compiles on the user's machine (nvcc of torch's CUDA major, ninja, gcc) into a per-user cache and links libtorch; FlashInfer JIT-compiles at first use (S1: ≈ 100–120 s for all specs per SM) | D1 |
| B5 | `core_abi` / `slot_abi` are parsed (defaults `">=0.0"` / `"0"`), declared by no manifest, checked nowhere; the core calls ≈ 20 optional plugin hooks by `hasattr` / `getattr` that `holon.api` does not declare | §7 |
| B6 | No root README; AGENTS.md's Environment section describes the system-site-packages venv | D8 |
| B7 | Speculation is on in no lock (M0.4 D8 left it open) | D3 |
| B8 | No SGLang backend or pin; FreeToken and llama.cpp found only through environment variables; FreeToken reads GGUF for one other family and three types, not our files | §5 |
| B9 | rtx-5090's Qwen3-8B bf16 lock is outdated; rtx-4090 has no 8B GGUF lock; rtx-3090 has none | §6 |
| B10 | **The history leaks** (reviewer C's scan of every blob reachable from the main and release branches (4952) and every commit message with `check_private.py`'s full pattern set, vendored trees skipped): 182 historical blob versions match the machine patterns — spike results and READMEs, goldens (22), judge results (54), three evidence files, an old `docs/plan.md` — mainly from 4fb5dc4, 8a9eab1, 8f46bd9, 544ab9a, dcf8902 and 4d2cfa2; four merge-commit messages (ace5d5e1d0, 28e9835588, fa915af5e1, 9c3a7fe0a4) match a private name through their branch names. Today's tree passes. The private remote already holds this history | §8 |

## 3. Decisions

- **D1 — Install from source at the release tag with uv; a Dockerfile for hosts without a toolkit; no wheels.**
  - Why not wheels: the extension links libtorch (one binary per torch build and Python), FlashInfer needs nvcc at first
    use, the lock classes pin `cu130-torch2.11`, and PyPI cannot pin torch's CUDA index. Considered and not taken:
    FlashInfer's `flashinfer-jit-cache` wheels (they remove its nvcc need but not ggml's).
  - **Root project:** `dependencies = ["holonlm", "holonlm-std", "holonlm-plugins"]`; `[tool.uv]` with
    `default-groups = []` (no dev group, so no `holonlm-testing`: references never serve), `python-preference =
    "only-managed"`, `environments = ["sys_platform == 'linux' and platform_machine == 'x86_64'"]`; `.python-version`
    3.12; `requires-python = ">=3.12,<3.13"` in every dist; a committed `uv.lock`.
  - **Pins:** torch 2.11.0 from the PyTorch cu130 index (`[tool.uv.sources]`), Triton 3.6, FlashInfer at M0.7's pin
    (0.6.12 today), a **released** transformers on which the bundle tests pass, xgrammar 0.2.8.
  - **Declared dependencies:** the core declares torch, numpy, msgspec, pyzmq; std and plugins declare what they
    import — transformers is a runtime dependency (`std/holon_std/weights/gguf/tokenizer.py` builds the GGUF tokenizer
    with it; plan §1's "judge/test only" is corrected in step 6); FastAPI, uvicorn and xgrammar become plain
    dependencies of `holonlm-plugins` (the `http` and `grammar` extras go: release locks select the grammar processor,
    D3). **`manifest-deps`** becomes an import scan per dist: every third-party top-level import of its modules must map
    (`importlib.metadata.packages_distributions`) into the dist's declared dependency closure.
  - **Host requirements:** Linux x86-64, a driver supporting CUDA 13, the CUDA 13 toolkit (nvcc), ninja, gcc / g++
    (Triton's launcher and torch extensions).
  - **Architectures:** ggml for sm_86 / 89 / 120a; FlashInfer for 8.6 / 8.9 / 12.0f. Other SMs: GGUF refuses at startup
    with the existing message; bf16 untested, said in the README.
  - **`holon prebuild`** (a core CLI command, so it may read manifests; no plugin named): kernel plugins declare their
    build step and JIT cache directories in a new manifest section `[build]` (`module` run as `python -m`, `caches`),
    as design §5.6 rule 7 requires ("JIT caches … are pre-built at install time and their directories are declared").
    The command runs one subprocess per (plugin, SM), default the visible GPUs; FlashInfer's plugin sets
    `FLASHINFER_CUDA_ARCH_LIST` to that one SM in its own process (plugins may read the environment; core and std may
    not). Caches resolve from `XDG_CACHE_HOME`, FlashInfer's and Triton's cache variables, else `~/.cache`.
  - **Dockerfile** (shipped, not published): the public CUDA 13.0 devel Ubuntu 24.04 image + build-essential, the
    source from `git archive` of the tag (no untracked files), `uv sync --frozen`, `holon prebuild --sm 86 89 120`,
    every cache directory pinned by environment variables to paths independent of the user, entry point `holon`.
  - **Measured in step 1:** a cold start compiles nothing but Triton kernels; first-start time (Triton + capture) per
    model against the supervisor's `Policy.start_timeout` (600 s), with the limit raised or the first start
    documented if a model comes close.
- **D2 — `holon serve` starts from the lock** (design §4.3 step 4, §4.4; ≈ 100 core lines, all general).
  - **One lookup by build hash**, in `holon.core.plans`, shared by the core and the tools: the weight source's
    identity → histories whose `model_id` ends in `@<build>` (content, path-independent) → `nearest_class` with the
    installed units' `src`. The tools write new locks under the existing history's `model_id`; `check_locks.py`
    rejects two `model_id`s for one build.
  - **Plan check:** `plan_of` moves into `holon.core.plans` without its git calls; the engine computes its plan with
    `model_id` and `commit` taken from the lock and `uv_lock` hashed from the source tree's `uv.lock`, reports its
    `plan_id` and a diff against the lock (design §4.3) in the log and `/metrics`. Acceptance 3 gates on equality.
  - **Precedence:** user flags > lock > defaults; dict flags merge per key (`--plugin-options a=b` adds to the lock's
    options). Lock kernel pins go to a new `EngineConfig.kernel_preferences` (soft: a provider that does not support
    the site or device falls back to default priority, logged); `--kernels` pins stay hard. `lock_overrides` raises when
    one op has two providers; step 2 checks the MTP locks, and such an op is left unpinned rather than failing.
    `plugin_options` apply wholesale (the core does not classify feature keys).
  - **Memory-shaped settings** (`max_seqs`, `max_model_len`, `max_step_tokens`) are upper bounds, clamped to the local
    KV budget and logged — on verified locks too (a desktop driving a display has 0.5–1 GB less than the card that made
    the lock). With no lock, `max_model_len` is sized from the budget instead of the 8192 default.
  - **Unverified** (borrowed from another class): the same, pins soft. **No lock for the file:** defaults, logged.
  - **Fallback:** the supervisor gets the history `[defaults + flags, lock + flags]`, so a lock that cannot start
    falls back after `rollback_crashes` failed starts. In an install, older locks pin other code; design §4.4's
    "rollback = restart on the previous lock" is corrected for installs in step 6.
  - **Where locks come from:** `--plans DIR`, default `plans/` of the source tree holding the installed core; absent →
    a loud warning, defaults. `--no-lock` serves defaults. A lock naming a second checkpoint (DFlash2's drafter) cannot
    be served from the lock alone: its plan records the drafter by identity, the user passes the path.
  - Tests: one parametrized case in `core/tests/test_plans.py` on toy locks (build-hash match, precedence, merge,
    clamping, soft pins).
- **D3 — Release locks select grammar and decide MTP.**
  - Every release lock selects `grammar.xgrammar` (`slots.logits_processor`): since M0.5's panel round there is one
    graph set with a processor selected (KV −0.12%, TPOT ≤ 0.05%; `docs/m05-grammar.md` §11, panel round item 1), so
    structured output works with no flag. The demo uses the same setting.
  - **MTP** (Qwen3.8-27B, per class) is turned on where, on a Claude-Code-shaped trace (M0.7's runner: the agent
    workload with a ≈ 16K-token shared prefix, several sessions, sampled at the temperature Claude Code sends, read
    from `bench/capture_proxy.py`), the median TPOT improves **and** p99 per-request max inter-token gap and p99 TTFT
    are non-inferior (≤ +10% or 2 sd of the plain rounds, whichever is larger). Cost stated beside it: KV on the RTX
    5080 falls from 58K to 39.6K tokens (`docs/m04-spec.md` §10); the lock keeps M0.7's `chunk_with_decode`, which
    alternates with a drafter (m07 §3.1). The README says how to turn speculation off (`--max-draft-tokens 0`).
  - DFlash2 stays a documented opt-in (second download, D2).
- **D4 — Legibility benchmark: Mixtral against mini-sglang** (§4).
- **D5 — Benchmarks gated at the vendored llama.cpp pin; newest upstream published beside** (§5).
- **D6 — ABI 1.0 at the release** (§7); versions 0.1.0, tag `v0.1.0`.
- **D7 — Agent demo, pre-registered.** `bench/demo/`: a small Python project (three modules, tests failing on a seeded
  bug and a missing function; task: make `pytest` pass without editing tests), committed before the first run.
  `holon serve` the class's 27B lock (5080: UD-IQ3_S; 4090: UD-Q4_K_M); Claude Code at a pinned version, headless, the
  M0.5 environment; 3 runs per class. Success: the harness restores the tests from its own copy, then `pytest` passes.
  Recorded: turns, wall time, served and prefix-saved tokens, TTFT / TPOT, the sampling parameters Claude Code sends,
  the tool-call sequence (no file text). Every miss is published; a smaller task, if any, is pre-registered as a new
  task and reported beside it. Claude Code's auto-compaction assumes a far larger window than the 32K served; the task
  fits, and the README says so.
- **D8 — README and AGENTS.md.** Written in step 4, before the legibility runs (they are part of what is measured).
  README: what Holon is; supported GPUs and the recommended file per card (§6); install at the release tag (D1, both
  paths); quickstart; Claude Code; how to turn speculation off; models served without locks; performance tables
  (rendered, §5.4); the legibility result, pass or fail; known limitations (single GPU, no offload, M0.5's recorded
  truncation shortfall and deferred runaway guards, first-start Triton compile, the context limit, a `git pull` while
  serving makes a supervisor restart import new code — stop the server first). AGENTS.md's Environment section
  describes the uv environment of D1. Quickstart sketch (fixed by step 10's run):

  ```bash
  git clone --branch v0.1.0 <repository> holon && cd holon
  uv sync --frozen                     # Python 3.12, torch 2.11 + CUDA 13 and every pin from uv.lock
  uv run holon prebuild                # GGUF kernels and FlashInfer modules for this GPU, once
  hf download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-IQ3_S.gguf --local-dir models   # 16 GB; 24-32 GB: UD-Q4_K_M
  uv run holon serve models/Qwen3.8-27B-UD-IQ3_S.gguf   # starts from this card's lock; the log names it
  ANTHROPIC_BASE_URL=http://127.0.0.1:8000 ANTHROPIC_AUTH_TOKEN=holon ANTHROPIC_MODEL=<id from /v1/models> \
  ANTHROPIC_DEFAULT_HAIKU_MODEL=<same> CLAUDE_CODE_SUBAGENT_MODEL=<same> claude
  ```
- **D9 — Lean tests.** No new suite: the `test_plans.py` case (D2); one `test_plugins.py` case (a missing or
  out-of-range `core_abi`); static scripts in `run_all.sh`: the `manifest-deps` import scan, `check_abi.py`,
  `uv lock --check`, and `check_locks.py`'s GPU-free mode (§7). The clean install (step 10) is a scripted run per class,
  not CI. A CPU-only GitHub Actions run of `run_all.sh` and the CPU tests comes with publishing.

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
- **Bias against Holon:** Mixtral is probably memorized on the baseline's side — mini-sglang shares SGLang's and
  vLLM's vocabulary (`RopeAttn`, `qkv_proj`, `MoELayer`), whose Mixtral files are widely published.

Rejected: Llama / Mistral / Qwen2 (mini-sglang has them); Phi-3 / Phi-4 (fused `qkv_proj`: a `WeightRule` maps one
tensor to one target, so a core change); Gemma, OLMo 2, Granite, GLM-4 (missing units on both sides); Qwen3.5-MoE,
Qwen3-Next (no GDN in mini-sglang: kernel porting on one side); SmolLM3 (NoPE: a std knob).

### 4.2 Branch points and environment

- **Holon:** main after M0.6 Gate A merges, with steps 1, 2 and 4 (packaging, D2, README / AGENTS.md), as **one
  squashed commit** that contains no `docs/release.md`, no `bench/legibility/` and no mention of Mixtral or the
  benchmark (tree and log grepped). AGENTS.md still says to read design.md and plan.md first (≈ 40K tokens); that cost
  is Holon's and stays. M0.7 changes no model-facing surface; if that surface (recipe terms, `ModelSpec` /
  `WeightMap`, the manifest schema, the model plugin protocol, std knobs) changes before the freeze, the protocol
  re-runs.
- **mini-sglang:** a depth-1 clone at `9a91cfa` (its history holds a ready Mistral diff, 7b670a2).
- Both clones sit under a root with no CLAUDE.md or AGENTS.md in any parent; the memory files Claude Code loads are
  recorded per run.
- **Per run:** a fresh clone, its own venv installed offline from the clone (Holon: the step-1 environment's wheel
  cache; mini-sglang: its pinned venv's), JIT caches warmed in that environment, one RTX A5000 of the development host.
  `$STATIC` per side, verified passing at the branch point: Holon `scripts/ci/run_all.sh` (ruff and import-linter
  installed), mini-sglang `pre-commit run --all-files` (hooks pre-installed).
- **Reference:** one file outside both clones, `$REFERENCE` = transformers 4.57.3's `modeling_mixtral.py`.
- **Agent:** Claude Code at a pinned version, one dated model id, pinned thinking / effort settings and
  auto-compaction setting; headless (`claude -p … --output-format json --allowedTools Read,Write,Edit,Bash,Glob,Grep
  --max-turns 150`), a configuration directory holding credentials only, no web tools, the same Bash timeout on both
  sides. Wall cap 2 h, check time excluded.
- **Prompt** (identical):

  > This repository is an LLM inference engine. Add support for Mixtral (`MixtralForCausalLM`). `$MIXTRAL` is
  > Mixtral-8x7B-Instruct-v0.1 cut to its first four layers (bf16 safetensors, original key names); the engine must load
  > it and generate from token ids. Chat templates and tool-call parsing are out of scope. `$REFERENCE` is the reference
  > implementation. You are done when `$CHECK $MIXTRAL`, `$CHECK $QWEN3` and `$CHECK $QWEN3_MOE` pass (each takes about
  > N minutes and prints a short summary; full logs go to a file) and `$STATIC` passes. Do not modify the checker or
  > the fixtures; do not use the network.

### 4.3 Correctness (one check, both sides)

- **Fixtures**, made by slicing the safetensors directly (original key names; the key list recorded): Mixtral's first
  4 decoder layers with embeddings, final norm, head and tokenizer (≈ 12 GB bf16; refused if the config enables a
  sliding window); regressions Qwen3-0.6B and Qwen3-30B-A3B cut to 4 layers.
- **`$CHECK`:** the engine (bench worker and backend, `holon` or `minisgl`, in the run's venv) greedily generates 64
  tokens for 16 prompts of judge prompt set v1 (≤ 512 tokens, batched at 16); HF transformers fp32 on the CPU scores
  the sequences teacher-forced. Pass: every generated token within δ of the fp32 maximum logit; δ per fixture =
  max(0.1, 2 × the largest shortfall of HF bf16's greedy tokens on 8 disjoint calibration prompts), fixed before any
  run. Output: a fixed summary of ≤ 30 lines, logs to a file.
- **Validated before the runs:** a reference Mixtral port per engine, kept outside the clones, passes all three
  fixtures with margin (each side's worst value against δ recorded); both unmodified engines pass the regression
  fixtures; plants fail on both — `w1` / `w3` swapped, renormalisation off, experts off by one, RoPE θ 1e4, a wrong
  KV-head mapping. A plant that passes strengthens the check before any run.
- A run succeeds only if the harness, after the agent stops, re-runs the three checks and `$STATIC` and they pass.

### 4.4 Cost

- **Primary:** `total_cost_usd` from Claude Code's JSON result. **Second costing** from the raw usage (input, cache
  writes, cache reads, output), with cache rebuilds caused by tool waits longer than the cache lifetime priced as
  reads (they penalize the side with slower checks). The verdict must hold under both.
- Reported: turns, wall time, diff size, and tokens and turns per Holon concept through a file-path → concept map
  fixed before the runs (manifest, entry points, `Ref` / `ModelSpec`, `WeightMap`, bundle, CI rules, documentation
  reading — design.md / plan.md / AGENTS.md counted separately, so a pivot targets the right cause).
- A run failing for infrastructure (API outage, host fault) is re-run; a run failing its checks or a cap counts as
  +∞.

### 4.5 Protocol and rule (fixed before measuring)

- **Pilot:** 2 + 2 runs, not counted, to estimate σ of log cost per side.
- **Counted runs:** n per side fixed from the pilot so that SE(log r) ≈ 1.25 · √((σ_H² + σ_M²) / n) ≤ 0.2
  (1.25: the median's efficiency), at least 3, capped at 10; launched in pairs (one per side, concurrently, on two
  GPUs) so time-varying API latency hits both alike.
- **Rule:** r = median cost (Holon) / median cost (mini-sglang), failures +∞. **r ≤ 0.70 → pass, else fail.** No
  extension band. Both medians infinite → inconclusive: extend once, up to the cap. Reported on their own lines: the
  80% bootstrap interval of r and each side's success rate.
- **Prediction, re-derived:** the tasks are similar in size; Holon saves the loader edits and attention wiring but
  carries ≈ 40K tokens of design and plan if the agent follows AGENTS.md, plus the manifest, entry point and reinstall.
  r ≈ 0.9 (0.6–1.3): a fail is more likely than not.
- **Budget** (owner question 2): pilot 4 runs, counted ≤ 20, extension ≤ the cap: ≤ 24 runs of ≈ 1–2 h; estimated
  $5–25 per run from token counts of comparable sessions, so ≈ $120–600; the pilot's actual cost re-estimates it
  before the counted runs. fp32 scoring needs ≈ 24 GB of host memory per check (two at a time, capped).

### 4.6 After the runs

The agents' Mixtral packages stay on local branches, not merged. Frictions costing a run ≥ 10% of its tokens are fixed
before the freeze if non-breaking (docs, error messages). The result, pass or fail, goes into the README.

### 4.7 If it fails

A pivot trigger: the release stops before step 6; I escalate to the owner with the record, the concept attribution
and a design note on which slots and nouns to collapse, reviewed by the panel before code. After any pivot the full
protocol re-runs. The ABI does not freeze until the owner decides.

## 5. Benchmark suite

### 5.1 Runner and baselines

- `bench/run.py` with M0.7 step 0's additions (max inter-token gap, long prompt, paired TTFT). Added:
  `backends/sglang_http.py` (SGLang's `/generate` with `input_ids`, streamed, ≈ 40 lines), a temperature option for
  D3's sampled trace, `bench/report.py` (README tables from records, no hand-copied numbers).
- **llama.cpp:** the **gate** runs at the vendored pin (M0.6's `cbe28bb3`: the same kernels, apples to apples); the
  newest upstream at suite time is a second, ungated column (speculation cells use it if the pin lacks MTP). FreeToken
  at a pinned commit; SGLang at a pinned release; all pinned in `setup_baselines.sh`.
- **Early (step 3, on current main):** feasibility smoke per (engine, model, class) and tier-1 bars on the RTX 3090
  and 5090, with upstream against the pin, so surprises on the untested classes surface before M0.7 closes.
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
  recorded per baseline: `llama-server -ngl 99 -fa on -kvu -np <max_seqs> -c <KV tokens> -b <step budget> -ub <…>
  -ctk/-ctv <format>`; `llama-bench` with the same `-ctk/-ctv`, `-b`, `-ub`; SGLang `--max-running-requests`,
  `--chunked-prefill-size`, `--max-total-tokens`, `--kv-cache-dtype`, `--mem-fraction-static 0.9`; FreeToken's as in
  `freetoken_http.py`. Greedy, `ignore_eos`, each engine's prefix caching on.
- Prompts are cut from the natural-text corpus (`--corpus`, judge prompt set v1) for every engine (M0.4: uniform ids
  inflate acceptance ≈ 2×).
- **Rounds:** ≥ 3 ABAB, ≥ 4 for the agent trace and the long prompt; per-round values published; one A/A per class on
  the agent trace; a difference under 2 sd of the rounds is printed as "≈".

### 5.3 Metrics and gate

- **Gated** (plan §9): per GGUF cell, in-process pp512 and tg128 at depth 0 and 8192, medians of repetitions 2–5 on
  both sides (`llama_bench.py` vs `llama-bench`), ≥ 0.8× the pin. Below it, the cell is fixed and re-measured at
  ≥ 0.8× before release.
- **Reported:** TTFT, TPOT p50 / p99; per-request max inter-token gap p50 / p99 / max; throughput; KV tokens.
  Speculation tables are labelled with their trace and sampling and add tokens per verify step, the depth-1 share, the
  speedup over Holon plain and the share of repeated output (repetition inflates acceptance).
- **Ratios are the claim;** absolute tokens/s are per board.
- **Predictions** (compared, not gated): GGUF decode 0.97–1.05× and pp512 1.04–1.20× the pin; agent TTFT p50 well below
  `llama-server`'s (57 vs 228 ms, 8B, M0.1); bf16 8B vs FreeToken 0.98–1.0× decode (M0.3, A5000); SGLang no prior
  record; 27B + MTP 1.5–1.9× single stream.

### 5.4 Publication

Records `evidence/bench_release-<cell>_<class>_<date>.json` (`run.py`: public command, Holon commit, the lock's
`plan_id`, the environment digest, baseline commits and flags, KV capacities, per-request rows, per-round summaries;
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

- 9 new locks; every release lock gains the grammar processor (D3) and the 27B locks D3's MTP decision. The 3-bit
  files stay locked only on the 16 GB card, where they are the recommendation. Served with defaults, no lock: Qwen3.5
  (4B / 9B), the bf16 27B and 30B (large-memory correctness paths), M0.6's fixtures and the `amd-quark` file.
- **Environment:** steps 7–9 run in the step-1 uv environment (the Dockerfile's image built on each class's desktop,
  one persistent container), its digest recorded; the development-class locks are re-propagated in it.
- **Process:** `propagate` every existing lock, then `tune` each **R** cell from the nearest class's lock, L2 against
  the existing goldens and floors (per kernel path, not per card), snapshot run-identical ×3. Approvals delegated for
  passing verdicts, recorded here. Approvals are counted per change (design §4.6: one per change, all affected locks
  listed), and the sweep's wall time per class is measured, both against plan §9's triggers (> 30 min, > 1 / day).
- **Release commit:** every class refreshed at it; any later code change propagates before the tag.

## 7. The ABI freeze

**Frozen at ABI 1.0:**
- `holon.api`: every public name and signature; the core's optional plugin hooks, today called by `hasattr` /
  `getattr` (`state_specs`, `planner`, `adopt`, `finalize_weights`, `bind_kernels`, `graph_key`, `substeps`,
  `lookahead`, `admit`, `rewind`, `stats`, `ks`, `build`, `draft_head`, `capacity`, `cut`, `next_publish`,
  `hit_length`, `payload_pool`), declared as optional members of their Protocols; `EngineConfig`'s fields; the
  `BuildCtx.options` keys the core provides; the IPC records (`EngineRequest`, `TokenDelta`, `Abort`).
- The manifest schema (sections, keys, `[build]` of D1), `SLOTS`, the entry-point group `holon.plugins` (value: the
  plugin's package; design §4.1's "path to manifest" is corrected in step 6).
- String vocabularies: `KernelSpec` props, finish reasons, event and message kinds.
- The `op.*@major` contracts (`attention.paged_decode@1`, `qlinear@2`, …) and std unit interfaces (`name@major`,
  knobs whose defaults never change).
- The lock format, the fingerprint class format, `canonical_hash` and `src_hash`.
- Not frozen: `holon.core` internals, the judge, the tools.

**Mechanism:**
- `holon.api.ABI = "1.0"`; `core_abi` is **required** in every manifest after 1.0 (first-party and `holonlm-testing`)
  and checked before the strict key parse, so a manifest from another ABI gets the version message, not an unknown
  key; out of range → startup error (invariant 7). `slot_abi` is removed for v0; per-slot versions return, additively,
  when a slot first needs its own.
- `scripts/ci/check_abi.py` renders the frozen surface — names, signatures, dataclass fields in order with types and
  defaults, Protocol members marked "implemented by plugins" or not, `SLOTS`, manifest keys, `EngineConfig`, the
  options keys and the string vocabularies — and compares it with `core/abi.txt`. A difference fails unless the
  listing is regenerated with the version bumped: added lines → minor, except a member added to a plugin-implemented
  Protocol, which is major like any removed or changed line (a reviewed core ABI change, invariant 9, with a codemod
  for first-party plugins).
- **Golden vectors:** `check_locks.py`'s GPU-free mode re-reads every committed lock (digest and `plan_id` recomputed
  by `canonical_hash`) and checks that every newest lock's unit `src`s match the tree (`src_hash`), so a change to
  either function or an unpropagated code change fails statically.
- **Timing:** the mechanism (loader check, required key, listing, GPU-free `check_locks`) lands before M0.7's close
  with the ABI at 0.1, so M0.7's one propagation covers it and D2; step 6 sets ABI 1.0 and versions 0.1.0 before the
  locks.

**What the legibility benchmark can still change**, until the freeze: anything above — collapsing slots or nouns, the
recipe terms, the model plugin protocol, manifest keys, how plugins register. After it, only by a minor or a reviewed
major.

## 8. Publishing (prepared, not done)

- **History:** the history as it stands is not published (B10). Recommended: a **squashed root commit** of the release
  tree pushed to a fresh public repository. Alternative: `git filter-repo` over every leaking blob and message, then a
  force-push of the private remote — which already holds the early history, and GitHub keeps force-pushed-over commits
  reachable by SHA until it purges them, so that path also needs GitHub support's purge before going public. Owner
  question 1.
- **Step 10's scan:** `check_private.py`'s full pattern set (generic, machine and `.private-patterns`) over every blob
  of the published tree and history, goldens' and results' safetensors metadata included, and over every commit
  message; zero hits. No matched value is ever printed into a tracked file.
- Also before asking: `check_provenance.py` green, `THIRD_PARTY_NOTICES.md` complete, release notes (what v0 serves,
  the §6 matrix, results, the legibility result, known limitations, recorded shortfalls), the tag.
- **On the owner's go only:** push the tree and tag; make the repository public; enable the CPU-only GitHub Actions run.
  The Dockerfile ships; a published image and PyPI wheels do not (v1).

## 9. Steps

| # | Work | Checks | Days |
|---|---|---|---|
| 1 | Packaging (D1): root project, declared deps, `manifest-deps` import scan, pins, `uv.lock`, `.python-version`, `holon prebuild` with `[build]`, Dockerfile | install in a fresh venv without system site-packages: `holonlm-testing` absent, CPU tests pass; cold start compiles only Triton; first start vs `start_timeout` | 2.5 |
| 2 | D2 (serve from the lock) and the ABI mechanism at 0.1 (§7) | `test_plans.py`, `test_plugins.py` cases; GPU-free `check_locks`; `check_abi`; served plan = lock on rtx-5080 / 4090 | 3 |
| 3 | Baselines (§5.1): pins, SGLang backend, temperature option, `report.py`; smoke and tier-1 bars on the 3090 / 5090, upstream vs pin — on current main | smoke matrix and bars recorded | 2 |
| — | M0.6 Gate A merges (Gate B, `triton_fp8`, is off the critical path); M0.7 starts | — | — |
| 4 | README and AGENTS.md (D8); legibility harness: fixtures, `$CHECK`, calibration, reference ports, plants, `$STATIC`, concept map, squashed branch point | δ recorded; validation as stated | 2.5 |
| 5 | Legibility: pilot 2 + 2, n fixed, counted runs, verdict; on fail §4.7 | verdict record; owner informed | 2–4 |
| — | M0.7 closes: one propagation covering D1's environment, D2 and the mechanism | its gates | — |
| 6 | ABI 1.0, versions 0.1.0; design §4.1 / §4.4 / §4.5 and plan §1 corrections; demo task committed (D7); panel review of steps 1–6 | review clean | 1.5 |
| 7 | Locks (§6) in the release environment: propagate all, tune the R cells, grammar, MTP decision (D3) | L2, snapshots ×3; approvals and sweep time recorded | 2 + 3 machine |
| 8 | Benchmark suite (§5) | records; gate | 1.5 + 3 machine |
| 9 | Agent demo (D7) | record | 0.5 |
| 10 | Clean install by the README on each class; history scan (§8); notices; release notes; release status in plan.md closing or deferring each carried CI item (`no-arch-branches`, `reference-provenance`, `isolation-check`, the IPC IDL); panel code review | review clean; zero scan hits | 2 |
| 11 | **Owner's go**, then publishing (§8) | — | 0.5 |

≈ 20–22 developer days and ≈ 6 machine days. **End to end ≈ 4–5 weeks** including M0.6 Gate A's rest (A4–A6) and
M0.7 (≈ 14 days): steps 1–5 run beside them; 6–11 follow M0.7's close. Core diff: D2 (≈ 100 lines), `holon prebuild`
and the ABI check (≈ 50), all general.

## 10. Risks

- **Legibility fails** (predicted r ≈ 0.9): the pivot holds the release (§4.7).
- **Context:** Claude Code assumes a far larger window than 32K; long sessions hit the limit (README, D7).
- **Baselines on sm_120 / sm_86** may not run (mini-sglang needed a patch on sm_120): "—", no patches beyond a
  documented one-liner; step 3 finds out early.
- **Late fixes** after step 7 re-propagate every class; the tag waits.
- **First start:** Triton compiles per model shape; if it nears `start_timeout`, step 1 raises the limit.
- **Desktop availability** (four classes, ≈ 6 machine days): parallel per class; the 3090 is the least exercised.

## 11. Open questions for the owner

1. **Publishing:** a squashed root commit in a fresh public repository (recommended), or `filter-repo` plus a
   force-push of the private remote and GitHub's purge? Reserve the PyPI names `holonlm*` now?
2. **Legibility agent and spend:** proposed default — Claude Code with the current Claude Opus model at its dated id,
   default thinking / effort, on your account; ≤ 24 runs, ≈ $120–600 (§4.5). Agreed, or another model / cap?
3. **FP8 without locks:** if M0.6 Gate B is late, does FP8 ship served with defaults (a scope change), or wait?

Lead decisions recorded (not questions): versions 0.1.0 / ABI 1.0 (D6); Mixtral and `total_cost_usd` with the second
costing (§4); MTP per D3; llama.cpp's vendored pin as the gate, upstream published beside (D5); 3-bit files locked
only on the 16 GB card (§6); Dockerfile shipped, image not published (§8).

## 12. Review findings, resolved

**Rev 1** (A: serving, packaging, ABI; B: methodology; C: architecture, scope).

| Finding | Resolution |
|---|---|
| C-B1: history leaks (182 blob versions, 4 merge messages); §2's scan too narrow | B10 corrected; full-pattern scan of every blob and message in step 10; squash recommended, filter-repo + purge as the alternative; image not published (§8, Q1) |
| A-M1: root project virtual; `uv sync --no-dev` installs nothing | root dependencies, `default-groups = []`; step 1 asserts `holonlm-testing` absent (D1) |
| A-M2: core imports undeclared, hidden by system site-packages | declared; `manifest-deps` as an import scan per dist; step 1 in a fresh venv; AGENTS.md updated in step 4 (D1, D8) |
| A-M3: Python and compiler unpinned | `.python-version`, `<3.13`, `only-managed`, gcc / g++, build-essential (D1) |
| A-M4: prebuild hard-codes plugins; one arch list for all SMs; per-user caches | `holon prebuild` reads the manifests' `[build]`, one subprocess per (plugin, SM); caches pinned in the Dockerfile; cold start checked; `flashinfer-jit-cache` noted (D1) |
| A-M5: "served plan = lock" not computable | `plan_of` in `holon.core.plans` without git; plan with the lock's `model_id` / `commit`; diff reported; gate on equality (D2) |
| A-M6: lock pins hard; feature keys classified by the core | `kernel_preferences` (soft), `--kernels` hard; `plugin_options` wholesale; memory settings as clamped upper bounds (D2) |
| A-M7: no fallback when a lock cannot start | supervisor history `[defaults + flags, lock + flags]`; design §4.4 corrected for installs (D2, step 6) |
| A-M8: two lookups could diverge; tools could fork a history | one build-hash lookup shared; tools write under the existing `model_id`; `check_locks` rejects two ids per build (D2) |
| A-M9: order lets version / ABI edits invalidate fresh locks | ABI 1.0 and 0.1.0 in step 6 before locks; mechanism before M0.7's close; GPU-free `check_locks` (§7, §9) |
| A-M10: listing misses duck-typed hooks, config, vocabularies; Protocol additions | hooks declared; listing covers `EngineConfig`, options keys, field order, vocabularies, plugin-implemented marks; locks as golden vectors; `op.*@major` frozen (§7) |
| A-M11: grammar off in locks | release locks select `grammar.xgrammar`; xgrammar a plain dependency; one graph set since M0.5 (D3) |
| A-m1: `core_abi` optional, parsed after strict keys | required after 1.0, checked first, testing manifests included (§7) |
| A-m2: `slot_abi` unused | dropped for v0 (§7) |
| A-m3: core estimate low | ≈ 100 lines (D2) |
| A-m4: dict flags replace the lock's | merged per key (D2) |
| A-m5: silent missing plans dir; image from a dirty tree | loud warning; `git archive` (D1, D2) |
| A-m6: first start may exceed `start_timeout` | measured in step 1 (D1) |
| A-m7: a lock naming a second checkpoint | not servable from the lock alone; stated (D2) |
| A-m8: README installs `main` | clones the release tag (D8) |
| A-m9: lock resolves for any platform | `environments` restricted (D1) |
| A-m10: design §4.1's entry-point text wrong | corrected in step 6 (§7) |
| A-m11: `lock_overrides` raises on an op with two providers | checked on MTP locks in step 2; such an op left unpinned (D2) |
| A-m12: `git pull` under a running server | documented (D8) |
| B-B1: n = 3 underpowered; extension band ad hoc | pilot for σ; n from SE(log r) ≤ 0.2, cap 10; single 0.70 rule; bootstrap interval and success rate reported; budget (§4.5) |
| B-M1: branch point leaks the benchmark; mini-sglang history has a Mistral diff | squashed commit, tree and log grepped; depth-1 clone; parent-directory memory files excluded and recorded (§4.2) |
| B-M2: reference and fixtures differ by side | one `$REFERENCE`; fixtures sliced with original key names, key list recorded (§4.2, §4.3) |
| B-M3: check output, JIT, timeouts and cache rebuilds bias cost | ≤ 30-line summary; warmed caches; equal Bash timeouts and stated check time; second costing; check time off the wall cap (§4.2–§4.4) |
| B-M4: check not shown to discriminate or to admit a correct port | reference ports per engine pass with recorded margin; regressions pass unmodified; five plants (§4.3) |
| B-M5: shared venvs; static checks unnamed | per-run offline venvs; `$STATIC` per side verified at the branch point (§4.2) |
| B-M6: llama.cpp commit choice moves the gate | gate at the vendored pin; upstream a second column; lock's KV format and `-ub`, `-b`, `-kvu`, `-np` recorded (§5.1, §5.2) |
| B-M7: gated metric unspecified; "fixed" undefined | in-process pp512 / tg128 at depth 0 and 8192, median of repetitions 2–5; fixed = re-measured ≥ 0.8× (§5.3) |
| B-m1: memorization favours the baseline | stated as a bias against Holon (§4.1) |
| B-m2: AGENTS.md's reading list | kept as Holon's cost, attributed separately; prediction re-derived (§4.2, §4.4, §4.5) |
| B-m3: concept attribution post hoc; pivot re-test | path → concept map fixed before runs; full protocol after any pivot (§4.4, §4.7) |
| B-m4: infra failures, unpinned agent | re-run; dated model id, Claude Code version, thinking / effort, auto-compaction pinned (§4.2, §4.4) |
| B-m5: rounds, noise | ≥ 3 rounds, per-round values, A/A per class, "≈" under 2 sd (§5.2) |
| B-m6: speculation tables unlabeled; repetition | labelled by trace and sampling; repeated-output share recorded (§5.3) |
| B-m7: baseline settings unmatched | Holon at its lock; baselines matched to it, flags listed (§5.2) |
| B-m8: demo grading and reporting | pre-registered; tests restored; sampling recorded; misses published (D7) |
| B-m9: absolute tok/s compared across boards | ratios are the claim (§5.3) |
| C-M1: MTP rule ignores tail latency and KV cost | TPOT improves and p99 gap / TTFT non-inferior on a Claude-Code-shaped sampled trace; KV cost and `chunk_with_decode` stated; off switch documented (D3) |
| C-M2: docs written after the benchmark | README and AGENTS.md in step 4, inside the branch point (D8, §4.2) |
| C-M3: locks made in a different environment than users get | steps 7–9 in the step-1 environment, digest recorded; dev classes re-propagated in it (§6) |
| C-M4: verified locks can over-commit memory | memory settings clamped on every lock (D2) |
| C-M5: critical path, upstream risk | M0.7 starts at M0.6 Gate A; early smoke on 3090 / 5090 with upstream vs pin; mechanism before M0.7's close; 4–5 weeks stated (§9) |
| C-M6: lock count and approvals | one recommended file per model per card (9 new); M0.6 files borrow on 5090 / 3090; approvals and sweep time measured (§6) |
| C-m1: rollback to an outdated lock | history of defaults + lock (D2) |
| C-m2: JIT caches undeclared before freeze | `[build]` manifest section before ABI 1.0 (D1, §7) |
| C-m3: transformers a runtime dependency | declared; plan §1 corrected in step 6 (D1) |
| C-m4: DFlash2 tier | dropped (§5.2) |
| C-m5: demo version, compaction | Claude Code pinned; compaction noted (D7) |
| C-m6: README omissions | legibility result, models without locks, speculation off (D8) |
| C-m7: carried CI items | closed or deferred in the release status; CPU-only Actions with publishing (§8, step 10) |
| C-m8: core classifying option keys | `plugin_options` wholesale (D2) |
