# Release (v0) — packaging, locks, legibility benchmark, benchmark suite, ABI freeze: design and plan

Status: **revision 1, draft for the owner and the panel** (2026-10-14). Scope: plan §2's Release row ("every v0
model: packaging, quickstart, rtx-5090 / rtx-3090 locks, an agent demo; the legibility benchmark (§9) before the ABI
freezes; the benchmark suite (decode, prefill, agent trace vs llama.cpp, FreeToken and SGLang where each runs) on the
release classes"), plan §9's legibility row (design D9 in `docs/m03-qwen38.md`: moved before the freeze by the owner,
2026-10-14) and design §4.4 / §4.6 (release classes, "a release commit requires every development and release class
refreshed"). Builds on M0.5 (`holon serve`, the HTTP frontend, the Claude Code session), M0.6 (`docs/m06-lowp.md`,
in progress) and M0.7 (`docs/m07-perf.md`, rev 3.1, reviewed clean), whose step 0 builds the benchmark runner this
release reuses (§5). **Publishing — pushing to GitHub, making the repository public, uploading an image or wheels —
needs the owner's go; this design only prepares it (§8, step 11).**

## 1. Goal and acceptance

A user with an RTX 3090, 4090, 5080 or 5090 installs Holon, serves a GGUF model at its tuned lock with one command,
and points an OpenAI or Anthropic client, or Claude Code, at it; every published number is reproducible from an
evidence record. Acceptance (each item is a gate, §9):

1. **Legibility** (§4, plan §9): Holon is ≥ 30% cheaper than a mini-sglang fork for the same agent adding the same
   model, both sides passing the same correctness check; else the pivot (§4.6) runs before anything freezes.
2. **ABI frozen** (§7): `holon.api` at ABI 1.0, every first-party manifest declaring its range, the loader enforcing
   it, a CI listing that fails on unversioned drift.
3. **Locks** (§6): every cell of the lock matrix written and approved (L2 passing, snapshot run-identical ×3) at the
   release commit, development classes refreshed; `holon serve` starts from them (the served `plan_id` equals the lock's
   on each class).
4. **Packaging** (D1, D2): a clean install, by the README alone, on each release class serves a chat completion and an
   Anthropic message; the container image builds; static checks green.
5. **Benchmarks** (§5): the suite recorded on all four classes; no GGUF cell below 0.8× llama.cpp on decode or prefill
   (plan §9's bar; below it, fixed before release); every cell published, losses included.
6. **Agent demo** (D7): Claude Code against the class's 27B lock completes the demo task in ≥ 2 of 3 runs on the RTX
   5080 and the RTX 4090.
7. **Owner's go**, then publishing (step 11).

Non-goals: PyPI wheels (D1); merging the legibility runs' Mixtral packages; new model families or formats; v1's
offload, VMM and self-improvement layer; GPUs outside sm_86 / sm_89 / sm_120.

## 2. Starting point (blockers found)

What exists: `holon serve` (`python -m holon serve`, supervised, one HTTP frontend with OpenAI chat / completions,
Responses and Anthropic routers), the bench harness (`bench/run.py`: ABAB rounds, client-side timing, evidence records
naming the GPU only, `--corpus` natural text; backends for Holon in-process and over HTTP, `llama-server`, FreeToken,
mini-sglang, nano-vllm), the lock tools (`holon_tools propagate | tune`), 8 (model, class) lock histories on main.

Blockers and gaps (each mapped to a decision):

| # | Finding | Where it lands |
|---|---|---|
| B1 | `holon serve` never reads a lock: the engine resolves fresh with default priorities, so tuned pins, KV format, checkpoints and spec settings reach a user only through hand-typed flags (M0.5's 27B command is nine flags long). `nearest_class` / `lock_overrides` exist but only the tools call them (tools README, "Not yet") | D2 |
| B2 | Lock lookup is by `model_id` = display name + build hash; the display name comes from the HF cache layout, so the same GGUF downloaded with `--local-dir` (or copied) never finds its lock | D2 |
| B3 | No dist declares its third-party runtime dependencies (torch "from the environment"; manifests list `gguf`, `tokenizers`, `transformers`, `flashinfer`, `triton`, `safetensors` but no `pyproject.toml` does); no `uv.lock` (every lock records `uv_lock = "none"`); no console script; the development environment runs a transformers dev build | D1 |
| B4 | `kernels.ggml_cuda` is compiled on the user's machine (`prebuild`, nvcc of torch's CUDA major + ninja, ≈ 10 CPU-minutes per SM) and links libtorch, so a binary wheel would be tied to one torch build; FlashInfer JIT-compiles its modules at first use (≈ 100–120 s for all specs per SM, spike S1), also needing nvcc | D1 |
| B5 | `core_abi` / `slot_abi` are parsed (defaults `">=0.0"` / `"0"`) but declared by no manifest and checked nowhere | §7 |
| B6 | No README at the repository root (`core/pyproject.toml` refers to one) | D8 |
| B7 | Speculation is on in no lock (M0.4 D8 left the turn-on open; every lock has `max_draft_tokens = 0`) | D3 |
| B8 | SGLang has no backend or pin; FreeToken and llama.cpp are found through environment variables only (`setup_baselines.sh` builds mini-sglang and nano-vllm) | D5 |
| B9 | FreeToken reads GGUF for one other model family and three types (Q4_0, Q8_0, Q6_K): it cannot run our GGUF files | D5 |
| B10 | The rtx-5090 Qwen3-8B bf16 lock is outdated; rtx-4090 has no Qwen3-8B GGUF lock; rtx-3090 has none | §6 |

The git history was scanned for host, cluster, user and path patterns in added lines (vendored trees and goldens
excluded): no hit; every commit uses the no-reply identity. Step 10 repeats the scan over every blob, safetensors
metadata included.

## 3. Decisions

- **D1 — Install from source with uv; a container image for hosts without a CUDA toolkit; no PyPI wheels in v0.**
  - Why not wheels: the native extension links libtorch (a binary wheel per torch build and Python), FlashInfer needs
    nvcc at first use anyway, the lock classes pin `cu130-torch2.11`, and PyPI cannot pin torch's CUDA index. A
    source install with a committed `uv.lock` pins everything exactly and makes the locks' `uv_lock` hash real.
  - **Pins** (one place, the workspace `pyproject.toml` + `uv.lock`): Python 3.12; torch 2.11.0 from the PyTorch
    cu130 index through `[tool.uv.sources]`; Triton 3.6 (torch's); FlashInfer at the pin M0.7 closes with (0.6.12
    today); transformers at a **released** version that the bundle tests pass on (chosen in step 1; the dev build goes);
    xgrammar 0.2.8 as the `grammar` extra (unchanged); FastAPI / uvicorn move from the `http` extra to plain
    dependencies (`holon serve` needs a frontend). Every manifest's `[dependencies] python` must be declared by its
    dist: the `manifest-deps` check of plan §3.5, a static script in `run_all.sh`.
  - **Host requirements** (source install): an NVIDIA driver supporting CUDA 13, the CUDA 13 toolkit (nvcc) and
    ninja on `PATH`, Linux x86-64.
  - **CUDA architectures:** `kernels.ggml_cuda` builds sm_86, sm_89 and sm_120a (as today); FlashInfer modules for
    8.6, 8.9 and 12.0f (one sm_120 build serves the 5080 and 5090, S1). Other SMs: GGUF refuses at startup with the
    existing message naming the supported SMs; bf16 paths are untested there and say so in the README.
  - **Console scripts:** `holon` (core: `holon.core.cli:main`; `python -m holon` stays) and `holon-prebuild` in
    `holonlm-plugins` (a module beside the kernel packages that runs `kernels.ggml_cuda`'s prebuild and builds
    `kernels.flashinfer`'s JIT specs for the given SMs, default the visible GPUs). No core change: the core never
    names a kernel.
  - **Container:** a `Dockerfile` from the public CUDA 13.0 devel Ubuntu 24.04 image: uv, `uv sync --frozen` without
    the dev group, `holon-prebuild --sm 86 89 120`, entry point `holon`. The devel base stays (Triton compiles its
    launcher with the system compiler at run time). Built in step 1, published only on the owner's go.
  - `uv sync` without the dev group installs no `holonlm-testing` (references never serve, M0.1).
- **D2 — `holon serve` starts from the lock** (design §4.3 step 4 and §4.4, as designed; the "Not yet" item).
  - In `holon.core.cli`, before the engine starts: the weight source's identity → the lock histories whose
    `model_id` ends in the same `@<build>` (content hash: path- and name-independent; fixes B2) → `nearest_class`
    with the installed units' `src` (an outdated lock is missing) → `lock_overrides`. Precedence: flags the user gave
    > the lock > defaults (deployment pins win, §4.3).
  - **Verified** (own class): the whole lock applies. **Unverified** (borrowed: same GPU on another toolchain, else
    the same SM, else the same toolchain): kernel pins are soft (a provider that does not support this device falls
    back to default priority), numerics settings apply (`dtype`, KV format, `page_size`, graphs, overlap, prefix cache,
    speculation), the memory-shaped `max_seqs`, `max_model_len` and `max_step_tokens` do not (another card's memory),
    and the log says so. **No lock for the file** (another quant, a fine-tune): defaults, logged. The engine never
    refuses to serve for lack of a lock (§4.4).
  - Where locks come from: `--plans DIR`, default the `plans/` directory of the source tree that holds the installed
    core (source installs and the image); absent → no locks. A `holon.plans` entry-point group (dists shipping their
    own locks) is additive and comes with wheels (v1).
  - The startup log and `/metrics` name the lock: `model_id`, class, `plan_id`, verified or not, overridden keys.
    `--no-lock` serves defaults.
  - Core diff ≈ 50 lines, general (no model, kernel or format names); one parametrized case in
    `core/tests/test_plans.py` (build-hash match, precedence, unverified filtering) on toy locks.
- **D3 — Release locks.** The matrix of §6: every v0 model gets a lock on every class where a common file of it fits;
  files without a lock are served with defaults. Speculation: Qwen3.8-27B's locks turn MTP on (M0.4 D8: slot bindings
  and the `spec.k` tunable through `propagate`, one approval) on each class where the agent trace's median
  per-request latency (release → last token, the M0.7 runner) is lower with it than the plain lock at M0.7's mixing
  setting; else that class stays plain and the record says why. DFlash2 needs a second download: documented opt-in
  flags, no lock.
- **D4 — Legibility benchmark: Mixtral, against mini-sglang, before the freeze** (§4).
- **D5 — Benchmark suite: two tiers, one llama.cpp commit, records only** (§5).
- **D6 — ABI 1.0 at the release** (§7).
- **D7 — Agent demo: Claude Code on the 27B, served by Holon.** A small Python project tracked in `bench/demo/`
  (three modules, a failing test suite with a seeded bug and a missing function; the task: make `pytest` pass without
  editing tests). `holon serve` the class's 27B lock (RTX 5080: UD-IQ3_S; RTX 4090: UD-Q4_K_M), Claude Code headless
  (`claude -p … --output-format json`, the M0.5 environment: base URL, model ids, nonessential traffic off), 3 runs per
  class. Success = the harness's own `pytest` passes after the run (the agent's report is not trusted). Record: turns,
  wall time, served tokens, prefix tokens saved, TTFT / TPOT from `/metrics`, the tool-call sequence (no file text).
  The task is fixed before the first run; below 2 of 3 on a class, the owner decides (smaller task or a documented
  limitation), the task is not tuned to the model after seeing results.
- **D8 — README and quickstart.** One root `README.md`: what Holon is (five sentences, link to design), supported GPUs
  and models (the §6 matrix with the recommended file per card), install (D1, both paths), quickstart, Claude Code, the
  performance tables (rendered, §5.4), known limitations (single GPU; no offload; M0.5's recorded truncation shortfall
  and the deferred runaway guards; first-start compile times as measured in step 1), license. Quickstart, sketched
  (commands are fixed by step 10's clean-install run):

  ```bash
  git clone <repository> holon && cd holon
  uv sync --frozen --no-dev            # torch 2.11 + CUDA 13 wheels and every pin from uv.lock
  uv run holon-prebuild                # GGUF kernels and FlashInfer modules for this GPU (one time; minutes)
  hf download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-IQ3_S.gguf --local-dir models   # 16 GB; 24-32 GB: UD-Q4_K_M
  uv run holon serve models/Qwen3.8-27B-UD-IQ3_S.gguf   # starts from this card's lock (log line names it)
  # OpenAI: base_url http://127.0.0.1:8000/v1 · Anthropic: base_url http://127.0.0.1:8000
  ANTHROPIC_BASE_URL=http://127.0.0.1:8000 ANTHROPIC_AUTH_TOKEN=holon ANTHROPIC_MODEL=<id from /v1/models> \
  ANTHROPIC_DEFAULT_HAIKU_MODEL=<same> CLAUDE_CODE_SUBAGENT_MODEL=<same> claude
  ```

  The README states the served context (32K on the 16 GB card) and that requests beyond it are refused with a message.
  No host names, paths or machine details (`check_private.py` covers it).
- **D9 — Lean tests.** No new suite. Added: the `test_plans.py` case (D2), one `test_plugins.py` case (an
  out-of-range `core_abi` is a startup error), and static scripts in `run_all.sh`: `manifest-deps` (D1), the ABI
  listing (§7), `uv lock --check`. The clean install (step 10) is a scripted manual run per class, not CI (it needs the
  GPUs and a full kernel build); the image build is not CI either.
- **D10 — Publishing is prepared, not done.** Step 11 lists the actions; each waits for the owner's go.

## 4. The legibility benchmark (plan §9, design D9)

### 4.1 Model

**Mixtral** (`MixtralForCausalLM`, Mixtral-8x7B-Instruct-v0.1): GQA 32 / 8 heads, head_dim 128, plain RoPE (θ 1e6),
RMSNorm, no attention bias, untied head, 8 experts top-2 with softmax → top-k → renormalise (equal to Qwen3-MoE's
`norm_topk_prob`), per-expert `w1` / `w3` / `w2` and a `gate` router.

- **Holon has every unit and no package:** `std.mixers.gqa` (with `qk_norm=None`), `std.moe@1` (`renormalize=True`),
  `std.ops` rmsnorm / rope / embedding / lm_head, `std.blocks.prenorm`, `WeightRule.stack` for the experts.
- **mini-sglang lacks it too:** at the bench pin `9a91cfa` it registers `LlamaForCausalLM`, `Qwen2ForCausalLM`,
  `Qwen3ForCausalLM`, `Qwen3MoeForCausalLM`, `MistralForCausalLM` and `Mistral3ForConditionalGeneration`; its
  layers (`RopeAttn` without qk-norm, `MoEMLP` with renormalisation) cover Mixtral, its loader hard-codes
  `gate_proj` / `up_proj` merges. Both sides are a model file plus wiring; neither ports a kernel (what D9 rejected).

Rejected candidates:

| Candidate | Why not |
|---|---|
| Llama, Mistral, Qwen2 | mini-sglang has them (a deleted-file re-add, likely memorized) |
| Phi-3 / Phi-4 | fused `qkv_proj` / `gate_up_proj`: a `WeightRule` maps one tensor to one target, so Holon would need a core change |
| Gemma, OLMo 2, Granite, GLM-4 | sandwich or post norms, embedding / residual multipliers, interleaved RoPE: missing units on both sides |
| Qwen3.5-MoE, Qwen3-Next | mini-sglang has no GDN: kernel porting on one side |
| SmolLM3 | NoPE layers: a `gqa` knob on Holon's side (a std change), a flag on mini-sglang's |

### 4.2 Branch points and setup

- **Holon:** the release candidate after M0.6 merges (the post-M0.6 ABI candidate; README and AGENTS.md as they will
  ship; no "mixtral" anywhere in the tree, checked by grep). M0.7 changes no model-facing surface (its holon.api
  changes are `BatchPlan.n_decode` and deleting `mix_phases`; its std.ops minors are kernel-facing), so the runs can
  overlap M0.7. If the model-facing surface (recipe terms, `ModelSpec` / `WeightMap`, the manifest schema, the model
  plugin protocol, std unit knobs) changes before the freeze, one confirmation run per side repeats.
- **mini-sglang:** a fork at `9a91cfa`, its pinned venv (`setup_baselines.sh`), its pre-commit hooks pre-installed.
- Each run: a fresh clone in its own directory, its environment ready (Holon's `.venv` with an offline reinstall of
  its own packages working, since a new plugin needs its entry point registered), one RTX A5000 of the development
  host. The harness keeps its own copy of the checker and fixtures outside both clones.
- **Agent:** Claude Code, one pinned version and model, headless (`claude -p "<prompt>" --output-format json
  --allowedTools Read,Write,Edit,Bash,Glob,Grep --max-turns 150`), in a fresh configuration directory holding
  credentials only: no user CLAUDE.md, memory, MCP servers, skills or plugins. No web tools. Wall cap 2 h.
- **Prompt** (identical for both sides):

  > This repository is an LLM inference engine. Add support for Mixtral (`MixtralForCausalLM`). `$MIXTRAL` is
  > Mixtral-8x7B-Instruct-v0.1 cut to its first four layers (bf16 safetensors); the engine must load it and generate
  > from token ids. Chat templates and tool-call parsing are out of scope. You are done when `$CHECK $MIXTRAL`,
  > `$CHECK $QWEN3` and `$CHECK $QWEN3_MOE` all pass and the repository's own static checks pass. The installed
  > transformers' `modeling_mixtral.py` is the reference. Do not modify the checker or the fixtures; do not use the
  > network.

### 4.3 Correctness (the same check on both sides)

- **Fixtures** (built once by a script in `bench/legibility/`): Mixtral-8x7B-Instruct-v0.1's first 4 decoder layers
  with its embeddings, final norm, head and tokenizer (≈ 12 GB bf16: real weights and real routing; the script
  refuses if the config enables a sliding window); regressions: Qwen3-0.6B, and Qwen3-30B-A3B cut to 4 layers (≈ 6 GB,
  catches a broken shared MoE loader).
- **Check** (`$CHECK`): the engine, through the bench worker and its backend (`holon` or `minisgl`, in its own venv),
  greedily generates 64 tokens for 16 prompts of the judge's prompt set v1 (≤ 512 tokens each, batched at 16); HF
  transformers in fp32 on the CPU scores the engine's sequences teacher-forced. Pass: at every generated position the
  engine's token is within δ of the fp32 maximum logit. δ per fixture = max(0.1, 2 × the largest shortfall of HF bf16's
  greedy tokens) on 8 disjoint calibration prompts, fixed before any run and recorded.
- **Checker validation, once before the runs:** HF bf16 as the "engine" passes; HF with `w1` / `w3` swapped and with
  renormalisation off fails, on the Mixtral fixture.
- **Static checks:** Holon's `scripts/ci/run_all.sh`; mini-sglang's pre-commit hooks. A run succeeds only if the
  harness, after the agent stops, re-runs all three checks and the static checks and they pass.

### 4.4 Cost

From Claude Code's JSON result per run: **cost = `total_cost_usd`** (the primary metric: every token kind weighted as
billed; raw usage — input, cache writes, cache reads, output — recorded so it can be recomputed), plus turns, wall
time, and the diff (files, lines). A run that fails (cap reached or a check failing) has infinite cost.

### 4.5 Protocol and rule (fixed before measuring)

- 3 runs per side, launched in pairs (one Holon, one mini-sglang, concurrently on two GPUs; pairs in sequence), so
  time-varying API latency hits both sides alike.
- **Rule:** r = median cost (Holon) / median cost (mini-sglang). r ≤ 0.70 → pass. r > 0.80 → fail. 0.70 < r ≤ 0.80 →
  3 more runs per side, decided on the medians of 6 at 0.70. Wall time and turns are reported, not gated.
- **Prediction:** r ≈ 0.75 (0.55–0.95). Holon saves the loader edits (expert names are a per-model `WeightRule`) and
  the attention wiring (`Ref` knobs); it pays for the manifest, the entry point and reinstall, and its static checks;
  mini-sglang's whole model layer is ≈ 800 lines that an agent reads in a few turns. A fail is plausible; that is what
  the rule is for.
- **Transcript analysis** (both outcomes): tokens and turns spent per Holon concept (manifest, entry points, `Ref` /
  `ModelSpec`, `WeightMap`, bundle, CI rules), from the transcripts. Frictions that cost a run ≥ 10% of its tokens are
  fixed before the freeze if the fix is non-breaking (docs, error messages); the agents' Mixtral packages stay on local
  branches, not merged.
- Machine and money: ≈ 1–2 h per run; fp32 scoring needs ≈ 24 GB of host memory per check (two at a time, capped).

### 4.6 If it fails

A pivot trigger (plan §9): the release stops before step 6, and I escalate to the owner with the record, the transcript
analysis and a design note proposing which slots and nouns to collapse (the concepts that cost most), to be reviewed
by the panel before any code. The ABI does not freeze until the owner decides.

## 5. Benchmark suite

### 5.1 Runner and baselines

- **Runner:** `bench/run.py` with M0.7 step 0's additions (per-request max inter-token gap, the long-prompt scenario,
  TTFT paired per turn), not duplicated here. Added by the release: `backends/sglang_http.py` (SGLang's native
  `/generate` with `input_ids`, streamed, ≈ 40 lines as `freetoken_http.py`), and `bench/report.py` (renders README
  tables from records; no number is copied by hand).
- **Baselines, pinned in `setup_baselines.sh`:** llama.cpp at one upstream commit, chosen in step 3 as the newest that
  builds for sm_86 / 89 / 120 and runs every GGUF cell including MTP for Qwen3.8 (`llama-server`, `llama-bench`);
  FreeToken at a pinned commit (its own venv); SGLang at a pinned release (its own venv). Step 3's feasibility smoke
  records, per (engine, model, class), whether it runs (fits, loads, answers); a cell that does not run is published
  as "—" with the reason.
- **Where each runs** (predicted; step 3 confirms): llama.cpp on every GGUF cell, and on M0.6's formats by M0.6's
  own bar (same bytes, Q8 activations); FreeToken and SGLang on HF checkpoints only — Qwen3-8B bf16 (24 and 32 GB
  cards), and M0.6's FP8 / NVFP4 checkpoints where they load. No GGUF comparison with SGLang (spike S2: its GGUF
  kernels mis-decode IQ3_S and run at 0.05–0.8× llama.cpp) or FreeToken (B9).

### 5.2 Cells

| Tier | (model file, classes) | Workloads |
|---|---|---|
| 1 | 8B Q4_K_M (all four); 27B: UD-IQ3_S (5080), UD-Q4_K_M (4090, 5090, 3090), each plain and with MTP; 8B bf16 (4090, 5090, 3090) | bars; `decode_b1`, `decode_b8`; agent trace; long prompt |
| 2 | every other locked file of §6 | bars only |
| Spec | 27B + MTP (all four), + DFlash2 (4090, 5090) | `decode_b1` and agent trace on natural text, vs Holon plain and llama.cpp with the same drafter |

- **Bars:** `llama_bench.py` vs `llama-bench` on the same file: pp512, tg128 at depth 0 and 8192, medians of
  repetitions 2–5 (as M0.1–M0.3).
- **Serving:** both engines behind HTTP, same max batch, step budget and KV capacity (`llama-server -c` = Holon's KV
  tokens; SGLang's token pool likewise), memory fraction 0.9, greedy, `ignore_eos`, prefix caching at each engine's
  default. Every serving workload cuts its prompts from the natural-text corpus (`--corpus`, judge prompt set v1) for
  every engine, so speculative and plain cells read the same text (M0.4's decision: uniform ids inflate acceptance ≈ 2×).
- **Rounds:** ≥ 2 ABAB for bars and decode traces, ≥ 4 for the agent trace and the long prompt (M0.7's gap metrics).

### 5.3 Metrics

TTFT p50 / p99; TPOT p50 / p99; per-request max inter-token gap p50 / p99 / max; output throughput; KV capacity in
tokens; bars in tokens/s. Speculation adds tokens per verify step, the share of verify steps at depth 1 and the
speedup over Holon plain at the same settings; never uniform-id numbers.

**Predictions** (from the records so far, to be compared, not gated): GGUF decode 0.97–1.05× llama.cpp, prefill
pp512 1.04–1.20×; agent TTFT p50 well below `llama-server`'s (prefix reuse and admission control; 57 vs 228 ms on the
8B, M0.1); bf16 8B vs FreeToken: decode 0.98–1.0×, agent TTFT within ±5% (M0.3 on the A5000); vs SGLang: no prior
record; 27B + MTP single stream 1.5–1.9× plain, agent TPOT ≈ 1.7×.

**Gate:** plan §9's GGUF bar only (decode and prefill ≥ 0.8× llama.cpp per cell); below it, fixed before release.

### 5.4 Publication

Records `evidence/bench_release-<cell>_<class>_<date>.json` written by `run.py`: command (paths made public by
`run.py`'s `public()`), Holon commit and the lock's `plan_id`, baseline commits and versions, engine flags, KV
capacities, per-request rows, summaries; GPU name only. README tables rendered by `bench/report.py`, each linking its
record; losses stay in. Raw logs stay in `bench/.out/` (ignored).

## 6. Locks

Lock matrix (✓ on main; M6 written by M0.6; **R** written by the release; — no lock: the file does not fit, or a
user of that card picks another file; such a card borrows, D2):

| Model file | rtx-5080 (16 GB, sm_120) | rtx-4090 (24 GB, sm_89) | rtx-5090 (32 GB, sm_120) | rtx-3090 (24 GB, sm_86) |
|---|---|---|---|---|
| Qwen3-8B bf16 | — (does not fit) | ✓ | **R** (refresh: outdated) | **R** |
| Qwen3-8B Q4_K_M | ✓ | **R** | **R** | **R** |
| Qwen3-8B UD-IQ3_XXS | ✓ | — | — | — |
| Qwen3-30B-A3B UD-IQ3_XXS | ✓ | — | — | — |
| Qwen3-30B-A3B Q4_K_M | — (17.3 GiB) | ✓ | **R** | **R** |
| Qwen3.8-27B UD-IQ3_S | ✓ (MTP: **R**) | — | — | — |
| Qwen3.8-27B UD-Q4_K_M | — (15.3 GiB) | ✓ (MTP: **R**) | **R** (MTP per D3) | **R** (MTP per D3) |
| Qwen3-8B NVFP4 | M6 | M6 | **R** | **R** (W4A8) |
| Qwen3-30B-A3B NVFP4 | — | M6 | **R** | **R** (W4A8) |
| Qwen3-30B-A3B MXFP4_MOE (GGUF) | — | M6 | **R** | **R** |
| Qwen3-8B FP8 | M6 (Gate B) | M6 (Gate B) | **R** | — (no FP8 on sm_86, v1) |

16 new locks; MTP decided by D3 on the four 27B locks (a turn-on of an existing lock is one approval). Served without locks, with defaults: Qwen3.5 (4B / 9B), the bf16
Qwen3.8-27B and Qwen3-30B-A3B (correctness paths for large-memory cards), M0.6's fixtures and `amd-quark` file. If
M0.6 Gate B is not closed at the freeze, its FP8 row ships without locks (open question 4).

- **Process:** after the freeze (step 6), on each class's consumer desktop in one persistent container: `propagate` the
  existing locks (the freeze changes every manifest, hence every `src`; snapshots byte-identical expected), then `tune`
  each **R** cell from the nearest class's lock, with L2 against the existing goldens and floors (the floors are per
  kernel path, not per card) and the snapshot trace run-identical over 3 fresh engines. Approvals delegated for
  passing verdicts, recorded in this doc's status.
- **Settings per class:** memory-shaped settings sized on the card (`max_seqs`, `max_model_len`, KV format, checkpoint
  count), following the existing 24 GB and 16 GB locks; the 5090's larger KV recorded. M0.7's mixing cap per class
  comes from M0.7 (5080, 4090) or its runner on the 5090 / 3090.
- **Release commit:** every development and release class refreshed at it (design §4.6); any later code change
  propagates before the tag.

## 7. The ABI freeze

**Frozen at ABI 1.0:** `holon.api` — every public name and signature: the vocabulary's types, Protocols and
dataclasses (`Ref`, `ModelSpec`, `LayerDecl`, `WeightRule` / `WeightMap`, `AuxSpec`, `KernelSpec`, `StateSpec`,
`SamplingParams`, `BatchPlan`, `LogitsProcessor`, drafter / verifier / policy protocols, `FrontendBundle`,
`EngineClient`, the IPC records `EngineRequest` / `TokenDelta` / `Abort`); the manifest schema (`plugin.toml`
sections and keys), the slot names and cardinalities (`SLOTS`) and the entry-point group `holon.plugins`; the lock
format (keys, canonical form, digest rule) and the fingerprint class format; std unit interfaces as versioned today
(`name@major`, knobs whose defaults never change). The documented CLI flags are a user contract kept stable within
v0.x. Not frozen: `holon.core` internals, the judge and tools.

**Mechanism:**
- `holon.api.ABI = "1.0"`; every first-party manifest declares `core_abi = ">=1.0,<2"` and `slot_abi = "1"`; `SLOTS`
  carries each slot's ABI major. The loader refuses a plugin whose range excludes the running ABI or whose slot major
  differs (a startup error, invariant 7). ≈ 15 core lines.
- `scripts/ci/check_abi.py` renders the frozen surface (names, signatures, dataclass fields with types and defaults,
  Protocol members, `SLOTS`, manifest keys) and compares it with the checked-in `core/abi.txt`. A difference fails
  unless the listing is regenerated in the same commit with the version bumped: added lines only → a minor (1.1);
  any removed or changed line → a major, which is a reviewed core ABI change (invariant 9) with a codemod for
  first-party plugins ("breaker pays", design §4.5).
- Package versions become 0.1.0 (tag `v0.1.0`); design §4.5's "v0 is pre-1.0: ABI may break" ends with this freeze.

**What the legibility benchmark can still change**, until the freeze: anything above — collapsing slots or nouns
(the pivot), the recipe terms (`Ref`, `ModelSpec`, `WeightRule`), the model plugin protocol (`bundle`, `draft_head`),
manifest keys (`derived_from`, `implements`, `variant_of`), how plugins register (entry points vs a directory scan).
After it, only additively (minor) or by a reviewed major.

## 8. Publishing (prepared, not done)

Prepared in step 10: the release notes (what v0 serves, the §6 matrix, measured results, known limitations, the
recorded shortfalls); the tag; the image build. Checks before asking: `check_private.py` and `check_provenance.py`
green, `THIRD_PARTY_NOTICES.md` complete for vendored code, the history scan of every blob (§2), the base image's
terms checked for redistribution. **On the owner's go only:** push `main` and the tag to the existing private GitHub
repository; make it public (open question 1: full history or a squashed root); push the image to a registry under
the owner's account. No PyPI upload in v0.

## 9. Steps

| # | Work | Checks | Days |
|---|---|---|---|
| — | M0.6 merges into main (its gates) | — | — |
| 1 | Packaging (D1): runtime deps and pins, released transformers, `uv.lock`, `holon` and `holon-prebuild`, `manifest-deps`, Dockerfile; first-start compile times measured | clean install on the development host; image builds; bundle tests on the pinned transformers | 2 |
| 2 | Serve from locks (D2) | `test_plans.py` case; denylist; served `plan_id` = lock's on rtx-5080 / 4090 | 1.5 |
| 3 | Baselines (§5.1): pins in `setup_baselines.sh`, SGLang backend, `report.py`; feasibility smoke per (engine, model, class) | smoke matrix recorded | 1.5 |
| 4 | Legibility harness (§4): fixtures, checker, calibration, checker validation, clean agent config | δ recorded; validation passes / fails as stated | 1 |
| 5 | Legibility runs: 3 + 3 (+ 3 + 3 in the band), verdict, transcript analysis; on fail, §4.6 | verdict record; owner informed either way | 1–2 |
| — | M0.7 merges into main (its gates; it writes its locks once) | — | — |
| 6 | ABI freeze (§7); README (D8); demo task (D7); panel review of steps 1–6 | `check_abi`; loader case; review clean | 2 |
| 7 | Locks (§6): propagate all, tune the R cells, MTP turn-ons (D3) | L2, snapshots ×3, approvals recorded | 2 + 3 machine |
| 8 | Benchmark suite (§5) on four classes | records; GGUF ≥ 0.8× | 1.5 + 3 machine |
| 9 | Agent demo (D7): 3 runs each on the RTX 5080 and 4090 | record; ≥ 2 of 3 per class | 0.5 |
| 10 | Clean install by the README alone on each class (scripted, fresh container from the CUDA base image); history scan; notices; release notes; plan status (a Release section); panel code review | review clean | 1.5 |
| 11 | **Owner's go**, then publishing (§8) | — | 0.5 |

≈ 15–16 developer days and ≈ 6 machine days (classes in parallel). Steps 1–5 run beside M0.7; 6–11 follow its merge.
Core diff: D2's lock start (≈ 50 lines) and the ABI check (≈ 15), both general.

## 10. Risks

- **The legibility rule fails** (prediction r ≈ 0.75): the pivot holds the release (§4.6). Mixtral is also the first
  real model through `gqa` without qk-norm, a path no test covers; a bug there is fixed in std and charged to Holon's
  side, as it would be for a contributor.
- **Claude Code assumes a larger context than a consumer deployment holds:** long sessions reach 32K; the demo task is
  sized to fit and the README states the limit.
- **Baselines on sm_120 and sm_86:** SGLang or FreeToken may not run on some classes (mini-sglang needed a patch on
  sm_120); such cells are published as "—", not patched beyond a documented one-liner.
- **Lock work after M0.7:** any late fix after step 7 re-propagates every class; the tag waits for it.
- **First start:** Triton and FlashInfer compile on first use (FlashInfer avoided by `holon-prebuild`; Triton per
  model shape, minutes): measured in step 1 and stated in the README.
- **Image size:** the devel base with torch is large (≈ 10+ GB); a runtime base is v1 work.
- **Consumer desktops' availability** (four classes, ≈ 6 machine days): classes run in parallel; the 3090 is the
  least used so far.

## 11. Open questions for the owner

1. **Publishing:** the full history or a squashed root commit when the repository goes public? The image under your
   registry account? Reserve the PyPI names (`holonlm*`) now even though v0 ships no wheels?
2. **Versions:** packages 0.1.0, tag `v0.1.0`, ABI 1.0 — or another numbering?
3. **Legibility:** Mixtral as the model, `total_cost_usd` as the cost, Claude Code (which model) as the agent, and
   the 0.70–0.80 extension band — agreed?
4. **M0.6 Gate B:** does the release wait for the FP8 locks, or ship the FP8 row without locks if Gate B is late?
5. **Speculation by default:** MTP on in the 27B locks where it lowers agent latency (D3), DFlash2 opt-in — agreed?
6. **llama.cpp for the suite:** the newest upstream commit at suite time (what users run; one commit for all cells)
   rather than our vendored ggml pin — agreed?
7. **Lock coverage:** 3-bit files (8B UD-IQ3_XXS, 30B UD-IQ3_XXS, 27B UD-IQ3_S) only on the 16 GB card — enough?
