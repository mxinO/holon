# Holon v0.2.0

The second feature release: Mixture-of-Experts models larger than the card. The experts live in pinned host RAM and
the most used are cached in VRAM, with output byte-identical to all-resident serving. Prefix KV and recurrent-state
checkpoints move to host RAM instead of being dropped. Two new models serve against the engines built for them:
Qwen3.8-Flash-Next against Strata and GLM-5.3-Flash against Project Maya. `holon serve <model>` now takes no hand-set
memory numbers. Everything in [v0.1.1](RELEASE_NOTES_v0.1.1.md) is kept: a plugin, flag or command of v0.1.1 still
works (Upgrade notes).

The core grows from 4,235 lines (v0.1.0) to 5,281 of its 6,000-line cap. The plugin ABI goes from 1.1 to 1.8, each step
additive, so plugins declaring `core_abi = ">=1.0,<2.0"` load unchanged (`core/abi.txt`).

## What's new since v0.1.1

- **Expert offload** (`std.moe.offload@1`, [docs/dev/v1-offload.md](../dev/v1-offload.md)). Every expert lives in
  pinned host RAM, and a VRAM cache holds the most used ones.
  - The cache is seeded from a routing profile; Holon ships the profiles its locks were measured with.
  - Decode fetches its misses over PCIe. A long prompt streams the missing experts through cache slots it lends for
    the step.
  - Placement policies are plugins: `lfu` (the default), `lru`, `static`, `profile_decay` (Strata's), promote-on-fetch
    (Project Maya's), and an `approx` tail skip in no lock.
  - Exact: every placement and policy is byte-identical to all-resident. The output is run-identical over three runs
    and identical across cache sizes and placements on the RTX 5080 and RTX 5090.
  - New lock: the RTX 5080's Qwen3-30B-A3B Q4_K_M, which does not fit 16 GB resident.
- **MoE prefill speed** ([docs/dev/v1-moe-prefill.md](../dev/v1-moe-prefill.md)). The changes:
  - expert calls of up to `pass_rows` rows, and empty tiles skipped;
  - SwiGLU fused into the gate / up store, and the router once per pass;
  - IQ1_M's mmq loader, and Q2_0.

  No core change.
- **KV and checkpoint tiers** (`std.state_tier`, [docs/dev/v1-kv-tiers.md](../dev/v1-kv-tiers.md)).
  - KV pages and recurrent-state checkpoints that VRAM evicts move to a 16 GiB host level, and load back when a
    prompt shares them. An SSD level is an option that no lock sets.
  - Checkpoint policies `branch` and `answer:k` (the last at the end of each answer); `evict.root_pin`.
  - The Qwen3.8-27B locks of the RTX 5080 and RTX 5090 take the tier.
- **Qwen3.8-Flash-Next** (`models.qwen4_exp`, [docs/dev/v1-flash-next.md](../dev/v1-flash-next.md)).
  - The model's parts: QSA sparse attention with its indexer, hyper-connections, GDN's output gate and the MTP head.
  - The PLE n-gram table (26.8 GiB) is held in host RAM, or read from the model file on the SSD.
  - Locks: IQ2_XS on the RTX 5080 (16 GB) and IQ3_S on the RTX 5090 (32 GB). Both offload the experts and draft three
    tokens a step with MTP.
- **MTP pack loaded as it is.** The drafter loads Strata's MTP pack (`mtp-q2_0.gguf`: experts Q2_0, dense projections
  BF16) unchanged. Holon no longer packs or quantizes the head. The README's Qwen3.8-Flash-Next section makes the
  pack with Strata's public tools. Without it, the lock serves without drafting and warns once with the cost.
- **GLM-5.3-Flash** (`models.glm5_next`, [docs/dev/v1-glm.md](../dev/v1-glm.md)).
  - What it serves: Maya's GGUF, text only. The model's parts: MLA with the DSA indexer, KDA (vLLM's copy of
    flash-linear-attention, vendored), mHC hyper-connections and the SwiGLU clamp. Its MTP head is unused.
  - Locks on the RTX 5090 (32K-token prompt steps) and the RTX 5080 (8K steps), each serving one sequence at 33,792
    tokens.
  - On a host short of the lock's RAM, *mode (ii)* applies: exclusive RAM and SSD tiers, with the experts the RAM
    cannot hold read from the model file as they are needed.
- **Auto-config** ([docs/dev/v1-auto-config.md](../dev/v1-auto-config.md)). `holon serve <model>` reads the host's
  available RAM once (with a cgroup v2 limit). From it, it derives:
  - the pinned-memory budget, keeping max(6 GiB, 5%) free;
  - the GPU's slack: a 1 GiB reserve on a shared or display card, 256 MiB on a headless one;
  - the files a lock names (routing profile, MTP pack).

  A KV tier takes what is left. A start refused for host memory retries once with a *spill*: the PLE table from the
  SSD, or the experts the RAM fits.
- **RAM adaptation** (`moe.ram=fit`, [docs/dev/v1-glm-ram.md](../dev/v1-glm-ram.md)).
  - An offload lock short of host RAM sizes its RAM tier from the host budget: one expert fraction, solved once at
    build. Below the working floor the start is refused, naming each term. Below GLM's knee (72 GiB) it warns.
  - This is std's generic spill for every offload lock. GLM on the RTX 5090 refines it with its measured set.
  - **GLM refuses to start below 60 GiB of host RAM** (`MemTotal`, or a lower cgroup limit). A 64 GB machine passes.
    `--plugin-options moe.ram_min=0` overrides the minimum.
- **Fail-stop** ([docs/dev/v1-fail-stop.md](../dev/v1-fail-stop.md)).
  - The CPU-task worker stops the engine at the step's commit when a task fails or a wait times out. The exclusive
    tiers' SSD reads could otherwise leave wrong bytes.
  - A start that is refused releases the VRAM, pinned pools and threads it built.
  - A refused start prints its one-line reason. Ctrl-C exits with status 130.
  - A SIGKILL at startup counts as a memory failure only when the cgroup counted an OOM kill.
- **Locks and process** ([docs/dev/release.md](../dev/release.md) §22).
  - Locks pin only code that can change outputs (`src` scheme 2: no tests, docs or docstrings).
  - Propagation runs at a release or a milestone's close. A code change whose snapshot trace is byte-identical
    advances without L2.
  - Between rounds, an outdated lock is not used: `holon serve` serves another class's current lock or the defaults
    (`plans/defaults.toml`, which selects the grammar processor).
  - Also fixed: torch 2.11's `MemPool.snapshot()` corrupted the host heap, so the activation loan no longer calls it.

## Supported GPUs and models

| GPU | memory | SM | lock class |
|---|---|---|---|
| RTX 5080 | 16 GB | 120 | `rtx-5080` |
| RTX 4090 | 24 GB | 89 | `rtx-4090` |
| RTX 5090 | 32 GB | 120 | `rtx-5090` |
| RTX 3090 | 24 GB | 86 | `rtx-3090` |

Architectures:
- Qwen3, Qwen3-MoE and the Qwen3.5 hybrid (which also serves Qwen3.8), from HF safetensors or GGUF;
- Qwen3.8-Flash-Next (GGUF, with its MTP head);
- GLM-5.3-Flash (GGUF, text only).

The recommended file per card and its lock (`holon models` lists them for the card it runs on):

| model file | RTX 5080 | RTX 4090 | RTX 5090 | RTX 3090 |
|---|---|---|---|---|
| Qwen3-8B bf16 | does not fit | ✓ | ✓ | ✓ |
| Qwen3-8B Q4_K_M | ✓ | ✓ | ✓ | ✓ |
| Qwen3-8B UD-IQ3_XXS | ✓ | borrowed | borrowed | borrowed |
| Qwen3-8B NVFP4 | ✓ | ✓ | borrowed | borrowed |
| Qwen3-8B FP8 | ✓ | ✓ | borrowed | not served (SM 89+) |
| Qwen3-30B-A3B UD-IQ3_XXS | ✓ | borrowed | borrowed | borrowed |
| Qwen3-30B-A3B Q4_K_M | ✓ offload | ✓ | ✓ | ✓ |
| Qwen3-30B-A3B NVFP4 | does not fit | ✓ | borrowed | borrowed |
| Qwen3-30B-A3B-Instruct-2507 MXFP4_MOE | does not fit | ✓ | borrowed | borrowed |
| Qwen3.8-27B UD-IQ3_S | ✓ KV tier | borrowed | borrowed | borrowed |
| Qwen3.8-27B UD-Q4_K_M | does not fit | ✓ | ✓ KV tier | ✓ |
| Qwen3.8-Flash-Next GSQ-RCO IQ2_XS | ✓ offload, MTP | borrowed | borrowed | borrowed |
| Qwen3.8-Flash-Next GSQ-RCO IQ3_S | borrowed | borrowed | ✓ offload, MTP | borrowed |
| GLM-5.3-Flash Maya-S-v2 IQ2_XXS | ✓ offload | borrowed | ✓ offload | borrowed |

- **✓:** this card's lock, written and verified on this card.
- **borrowed:** another class's lock, applied unverified. Its settings apply as they are and its kernel pins become
  preferences.
- Other checkpoints of these architectures serve with defaults.
- The README's table gives each file's hub repository.

The offload and KV-tier locks pin host RAM. Approximate amounts:
- Qwen3.8-27B tier: 16 GiB;
- the RTX 5080's Qwen3-30B-A3B: ≈ 16.4 GiB;
- Qwen3.8-Flash-Next: ≈ 61 GiB on the RTX 5080 and ≈ 74.5 GiB on the RTX 5090;
- GLM-5.3-Flash: ≈ 81 GiB.

With less RAM, a lock serves smaller, as the README's Host RAM table gives per lock.

## Results

Ratios are Holon's advantage: above 1, Holon is faster. Every number comes from the record or table cited beside it.
The full tables, losses included, are in the README's Performance section and the documents cited.

- **Qwen3.8-Flash-Next against Strata `296b59d`**, both exact. Measured at the milestone's close on the locks'
  deployments: batch 1, 8 prompts × 2 rounds (`docs/dev/evidence/fn_close_final_2026-10-24.json`, key `d11`).
  Bars: 1.0 for decode, 0.9 for prefill. Every row met its bar.

  | row, 1K / 4K / 32K | RTX 5080 (IQ2_XS) | RTX 5090 (IQ3_S) |
  |---|---|---|
  | decode, drafting off | 1.086 / 1.090 / 1.071 | 1.167 / 1.143 / 1.111 |
  | decode, MTP | 1.087 / 1.141 / 1.116 | 1.067 / 1.001 / 1.000 |
  | prefill, drafting off | 1.090 / 1.222 / 1.022 | 0.995 / 1.225 / 0.994 |

  - With MTP, Holon decodes at 129–132 tok/s on the RTX 5080 and 175–194 tok/s on the RTX 5090.
  - On 72 hard task items (greedy, thinking on), Holon answered 70 on the RTX 5080 and 72 on the RTX 5090; Strata
    answered 70 on each (`docs/dev/v1-flash-next.md` §22.4).
  - The MTP rows used a Q8_0 head, which Holon no longer makes (Recorded shortfalls).
- **GLM-5.3-Flash against Project Maya v1.1.27**, at its default settings. Measured on one RTX 5090: batch 1, 8 prompts
  × 5 rounds. Each cell is the median of the rounds; the mean is in parentheses where it differs by more than 2
  points (`docs/dev/evidence/glm_close_2026-10-31.json`, `final_table`).
  - **Mode (i)**, RAM holding every expert, on the lock:
    - decode 1.178 / 1.106 / 1.052 at 1K / 4K / 32K;
    - prefill 1.077 / 1.017 / 1.074;
    - 36–41 tok/s of decode.
  - **Mode (ii)**, a 64 GB host (both engines capped at 64 GiB):
    - decode 1.040 (0.985) / 0.982 / 0.923 (0.897);
    - prefill 1.096 / 1.025 / 1.248;
    - 19–24 tok/s of decode.
  - On 72 hard task items, Holon answered 69 and Maya 69 (the close record, `d`).
  - Under the grammar, every constrained output that ran to its end conformed: 283 of the 307 accepted schemas of a
    500-schema JSONSchemaBench sample. 7.8% of constrained outputs were truncated at `max_tokens` 2048, against
    17.0% unconstrained (the close record, `conformance`).
- **GLM by host RAM** (`docs/dev/v1-glm-ram.md` §10). Measured on one RTX 5090 with the job's RAM capped, at what
  `holon serve` gives at each size. Decode in tok/s at 1K / 32K:
  - 64 GiB (63% of the experts in RAM): 24.9 / 19.5;
  - 72 GiB (73%): 33.1 / 24.2;
  - 80 GiB (83%): 35.2 / 30.2;
  - the lock, from about 88 GiB: 41.7 / 36.8.

  Each spill was byte-identical to its lock's snapshot under `exact.py`: GLM on both cards, Qwen3-30B-A3B and
  Qwen3.8-Flash-Next.
- **Qwen3-30B-A3B Q4_K_M offload on the RTX 5080**, against FreeToken `21d9578` (52.7% of the experts cached):
  - decode 1.06 / 1.04 / 1.02 at 1K / 4K / 32K, and 1.19 / 1.23 / 1.10 against llama.cpp's pinned expert cache
    (`docs/dev/evidence/x8_close_rtx-5080_2026-10-18.json`);
  - prefill after the MoE prefill work: 1.850 / 1.136 / 1.005 against FreeToken at the lock's step budget, and
    0.959 against FreeToken's default at 32K (`docs/dev/evidence/mp5_close_2026-10-20.json`).
- **KV tier** on an agent trace whose sessions are parked and resumed (`agent_park`, the RTX 5080's Qwen3.8-27B
  UD-IQ3_S):
  - against the VRAM-only engine: time to first token p50 0.047× and throughput 2.26×
    (`docs/dev/evidence/k4_agent_park_rtx-5080_2026-10-19.json`, `bars.3c`);
  - against `llama-server --cache-ram` sized to serve the trace: TTFT p50 0.122× and throughput 1.79× over 3 rounds
    (`docs/dev/v1-kv-tiers.md` "Step 4", row 3d).
- **Auto-config** (`docs/dev/v1-auto-config.md` §9 record).
  - A bare `holon serve <model>` served every locked model on its lock: 9 of 9 on the RTX 5080 and 7 of 7 on the RTX
    5090.
  - Against the hand-set values, the automatic ones passed 13 of 13 rows on the RTX 5090 and 23 of 25 on the RTX
    5080. The two misses were 0.9995 and 0.9994, from byte-identical profiles.
  - Under an 18 GiB cap, the 27B tier shrank from 16.0 to 11.1 GiB and served.

## Recorded shortfalls

- **GLM-5.3-Flash at 64 GB: decode after a long prompt.** At the close, mode (ii)'s decode was 0.982 / 0.923 of
  Maya's at 4K / 32K (the close record).
  - With RAM adaptation, at 64 GiB only 32K decode stays under Maya: about 0.93 (`docs/dev/v1-glm-ram.md` §11; Maya's
    side from the close, so the ratio is indicative).
  - It was investigated, and no lever predicts 3% or more. The cause is structural: the exclusive tiers' in-graph
    bookkeeping, and a 32K prompt displacing RAM experts that decode then rereads from the SSD.
  - From 72 GiB, Holon is ahead at every context.
- **Qwen3.8-Flash-Next with MTP on the RTX 5090, since the BF16 pack.** The close's MTP rows ran the head's dense
  projections in Q8_0. Holon now loads Strata's pack as it is, with BF16 projections.
  - Re-measured against the Q8_0 head on the same card and build, MTP decode is
    (`docs/dev/evidence/fn_mtp_strata_pack_ab_2026-10-30.json`, `new_over_old`):
    - RTX 5080: 0.991 / 0.976 / 0.988;
    - RTX 5090: 0.990 / 0.982 / 0.989.

    Tokens per verify pass are the same, and prefill is unchanged.
  - Scaled, the RTX 5090's MTP decode at 4K and 32K is about 0.98–0.99 of Strata's, under its 1.0 bar (the README's
    "Qwen3.8-Flash-Next against Strata").
- **GLM-5.3-Flash on the RTX 5080 has no tuned set of its own** for a host short of the lock's RAM. It serves std's
  generic spill (`moe.ram=fit`), checked byte-identical under a 64 GiB cap, at an unmeasured speed. The RTX 5080 was
  not compared with Maya.
- **The generic spill's speed is unmeasured** on Qwen3-30B-A3B and Qwen3.8-Flash-Next (README, Host RAM).
- **Expert offload's prefill against FreeToken** (milestone X, accepted by the owner).
  - At the offload milestone's close, prefill was 0.92–0.93 of FreeToken's at 4K and 0.87–0.88 at 32K
    (`docs/dev/v1-offload.md` "Step 8, part 2").
  - The MoE prefill work then brought the lock to 1.136 / 1.005 at the lock's step budget. Against FreeToken's
    default at 32K it is 0.959 (above).
- **The capped-fit CPU split (F5b) has no lock.** It gained 11.7% by `tune` at a 25% expert fit. No tool writes a lock
  with a CPU split, and `check_locks.py` refuses one committed by hand.
- **KV tiers' resume from SSD beside decoders** missed at 16K and 32K on the default I/O rates. It passes on the host's
  tuned rates, post hoc (the owner's option 2, `docs/dev/v1-kv-tiers.md` "Owner decisions"). The predicted fall of
  re-prefilled answers in the pi test (to ≤ 10% of uncached input) measured 26.1% (row 3g).
- **Test time** (process shortfalls, accepted):
  - GLM's default test run took 301 s against `main`'s 283 s (+6.5%, against ≤ +5%; the close record, `tests`);
  - the MoE prefill work added 6% to the suite.

## Known limitations

- **Model and memory.** One GPU per server. Offload covers a MoE model's experts, plus Qwen3.8-Flash-Next's PLE table
  and token embeddings. The rest of the model and its KV cache must fit the card.
- **Host RAM is pinned** for the server's lifetime. The budget is read from `MemAvailable` and a cgroup v2 limit:
  - under cgroup v1, or with ZFS, pass `--host-memory`;
  - a desktop's systemd-oomd may kill a server whose host runs short regardless.
- **Offload locks serve one sequence** (`max_seqs` 1), so concurrent requests queue. More sequences shrink the expert
  cache, at an unmeasured speed.
- **Context.**
  - The Qwen3.8-27B locks serve 49,152 tokens (40,960 on the RTX 5080).
  - Qwen3.8-Flash-Next, GLM-5.3-Flash and the RTX 5080's Qwen3-30B-A3B offload lock serve 33,792.
  - The other 8B and 30B locks serve 8,192 or 16,384.
  - Without a lock, a model serves min(its trained context, 32,768).
  - Streaming a sparse-attention layer's KV through host RAM, as Strata does from 64K, is not built.
- **Speculation.** It is off in every lock but Qwen3.8-Flash-Next's, and GLM-5.3-Flash's MTP head is unused. On the
  Qwen3.8-27B, MTP's longest pauses failed the rule fixed before measuring. DFlash2 is an opt-in in no lock.
- **Structured output on Qwen3.8-Flash-Next is unverified:** its locks select no grammar processor.
- **A long prompt stalls the streams already decoding** on every 8B and 30B lock (`chunk_with_decode` 0).
  `--chunk-with-decode 512` mixes them.
- **On an RTX 5080 that also drives a display**, the Qwen3.8-27B's context may be clamped to what fits (≈ 260 MiB of
  headroom). Claude Code's ≈ 31K side requests need it above ≈ 32K.
- **Traces are run-identical on one machine**, not across machines. Two RTX 3090 desktops recorded different
  Qwen3.8-27B traces at the same lock.
- **No authentication:** do not expose the server beyond a trusted network.
- **Builds at start.** Every start compiles FlashInfer's CuTe DSL norms (about 1 s); the first start compiles Triton
  kernels.
- **M0.6's low-precision files** run with borrowed, unverified locks on the RTX 5090 and RTX 3090.
- **Grammar runaways.** Under a grammar, an output can run on until `max_tokens`. On the 27B with MTP, 2.66% of
  constrained outputs were truncated; on GLM-5.3-Flash, 7.8%. Set `max_tokens`.
- **GLM-5.3-Flash's mode (ii) is not a lock.** Its RAM share is sized at start, and was measured on one RTX 5090 only.
- **Stop the server before a `git pull`.** The install is editable.

## Locks

There are 27 lock histories, against v0.1.1's 22:

| card | locks |
|---|---|
| RTX 3090 | 4 |
| RTX 4090 | 8 |
| RTX 5080 | 9 |
| RTX 5090 | 6 |

Every lock was propagated at each milestone's close and judged as `docs/dev/release.md` §22 describes. GLM-5.3-Flash's
locks are judged by L2 in its reference form (bf16 references from an RTX PRO 6000).

v0.2.0's own propagation round carried all 27 to the release's code (its version string and `uv.lock`): each snapshot
trace byte-identical over 3 runs, every lock advanced, none stopped for review
(`docs/dev/evidence/release_v0.2.0_propagation_2026-11-02.json`).

## Upgrade notes

- **No v0.1.1 interface is removed.** Every `holon serve` flag, every plugin option and every `holon_judge` command of
  v0.1.1 remains.
- **Start every lock with `holon serve <model>` alone.** Auto-config derives the host budget and the GPU's share.
  `--host-memory BYTES` and `--memory-fraction F` remain as overrides.
- **Qwen3.8-Flash-Next's MTP pack** is Strata's `mtp-q2_0.gguf`, made with Strata's `mtp_fetch.py` and `mtp_pack.py`
  (README, "Qwen3.8-Flash-Next"). Keep its name.
- **For users of `main` since v0.1.1.** Some commands and options were added and removed between releases:
  - `holon_tools mtp-pack` is removed: Holon no longer packs the head;
  - `moe.copy_beside` and `expert_io.beside_ctas` are now constants (on, and 2);
  - `cpu_task.io_threads` is replaced by `cpu_task.io_depth`. The I/O worker is one completion-driven thread, using
    io_uring or native AIO.

  A lock or a flag that sets a removed option is warned about.
- **Repository layout.**
  - Development records moved under `docs/dev/`: design records, `evidence/` to `docs/dev/evidence/`, `spikes/` to
    `docs/dev/spikes/`.
  - Release notes are under `docs/releases/`.
  - Live tools moved to `judge/` and `bench/`.
- **The Docker image's `HOME`** is `/opt/holon/user`. Mount a directory at `/opt/holon/user/.cache/holon` to keep
  Holon's cache across containers (README, Docker).

## Install

Linux x86-64, an NVIDIA driver for CUDA 13, the CUDA 13 toolkit, gcc / g++ and [uv](https://docs.astral.sh/uv/):

```bash
git clone --branch v0.2.0 https://github.com/mxinO/holon.git holon && cd holon
uv sync --frozen
uv run holon prebuild
uv run holon serve unsloth/Qwen3.8-27B-GGUF
```

The README has the rest: the Docker image, Qwen3.8-Flash-Next's MTP pack, the clients, Claude Code and the options.
