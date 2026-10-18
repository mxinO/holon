# v1 — milestone K: KV and checkpoint tiers under the prefix index

Status: **rev 2, after the challenge panel** (2026-10-18). Nothing is built and nothing has been measured for K yet.
Every number below is a CPU-side estimate from earlier evidence (cited), fixed as a prediction before step 0 and step
4 measure it. Rev 2 answers the three-reviewer panel (A correctness, B architecture, C numerics and performance); the
finding-by-finding disposition is §13.

Scope: plan §10's K line ("KV and checkpoint tiers (pinned RAM / SSD under the prefix index, exact, deterministic
resume)", `docs/plan.md:603-604`), the end-of-answer checkpoint item (`docs/plan.md:675-681`), and the requirements
milestone X's panel left for K (`docs/v1-offload.md` §11, lines 585-600). The owner's direction (2026-10-18):
- support the popular strategies: exact by default; a host-RAM tier in the style of Strata, llama.cpp, vLLM and
  SGLang HiCache; an SSD tier; conversation parking as the prefix index with a host tier; swap-on-preemption for free;
- a new strategy is a plugin;
- no large-model runs now: the development vehicle is a model Holon already serves, and the gate runs on an RTX
  5080 or 5090 with an agent trace.

Evidence base:
- the local research note `docs/brainstorm/v1-offload-research.md` ("R §n"; not published), §3.7 (KV offload
  systems), §3.1 C1–C4 (Strata), §3.1b KG1–KG3 (Project Maya), §4.1–4.2 (K1–K4);
- the pi agent test (`docs/agent-test-pi-rtx5090.md`);
- M0.3's checkpoint design (`docs/m03-qwen38.md` D4, §4.8; `docs/m03-in-step-snapshot.md`);
- X's step 0 rates (`docs/v1-offload.md` §13: M1 PCIe, M7 NVMe).

**What changed from rev 1, in one paragraph.** The core now sees only the VRAM boundary: an entry is resident or not,
and one plugin in a new single slot `tier` holds everything below (B2). The host and SSD levels, their eviction, the
copies, the write-back, the virtual clock and the manifest all move to std (`std.state_tier`). The core is ≈ 285
lines instead of 480, with no new device stream, no batched-copy call and no `Payload.tier` (B4). Every tier state
change is logical at decision time, and physical reuse waits on copies and tickets (A5, C5). Resume is gated by a
virtual clock with a per-step cost model (C9), not by a fixed step or by `BLOCKED` (A6). The VRAM checkpoint pool
becomes staging only, so payloads live in the tier from registration and adopt straight from host (A4, C1). A publish
that meets a demoted entry promotes it into the request's own units (A1). Victim choice in VRAM is a plugin slot (B1).
The held-out proof holds out a checkpoint policy and a tier the protocol was not shaped around (B3).

## 0. The references in one table

| | vLLM V1 native offload | SGLang HiCache | LMCache | llama.cpp | Strata (C1–C4) | Project Maya (KG1–KG3) |
|---|---|---|---|---|---|---|
| What moves | KV blocks, cross-layer contiguous chunks | KV pages (layer-first or page-first); a Mamba host pool for hybrids | 256-token KV chunks | whole slot state blobs (`--cache-ram`, default 8 GiB), `--ctx-checkpoints` for hybrids | recurrent checkpoints (C1, ≈ 118 MB each); whole parked conversations (C2); session files (C3); KV streaming for QSA (C4) | KDA checkpoints (≈ 146 MiB) plus indexer pool (KG2); conversation slots on SSD (KG3); DSA latent streaming (KG1) |
| Tiers | GPU → pinned CPU → fs / object / p2p | GPU, pinned host, file / Mooncake / 3FS / NIXL | CPU, disk (O_DIRECT), GDS, remote | RAM; disk via slot save | RAM; files (C3) | RAM; SSD files, buffered writes |
| When it stores | write-through of prompt blocks, optional N-hit threshold | `write_through`, `write_through_selective` (default), `write_back` | on store calls | synchronously, when a slot is reused (an observed 2.65 s stall before a request, `docs/release.md:2690-2692`) | at turn start, every 16K, system-prompt end ≥ 2048 (C1) | system-prompt end, last-turn start, every 16,384 (KG2) |
| Load path | whole prefix, async between steps; request parks in `WAITING_FOR_REMOTE_KVS`; FULL graphs kept | layer-wise load-back in prefill (`memory_pool.py:2786-2791`) | blocking by default; layer-wise opt-in (PIECEWISE) | synchronous | restore whole state | restore whole state |
| Lookup, eviction | chained sha256 block hash; LRU or ARC | radix tree; LRU and others | chained hashes; LRU | linear longest-prefix scan; oldest first | root pinned, others LRU (C1); oldest (C2) | 4 checkpoints; 4 slots |
| Survives restart | fs tier | L3 | disk | slot files on request | C3 on request | `SLOT_KEEP` (opt-in) |
| Exact | yes | yes | yes (CacheBlend / CacheGen approximate) | yes | state yes; a resumed repeat reads its tail through decode (#410) | yes |

Sources: R §3.7 (lines 511-527), R §3.1 C1–C4 (lines 148-151), R §3.1b KG1–KG3 (lines 233-235).

None of them publishes consumer-GPU numbers for its tiers (R §7 line 781). Holon's offer is the same strategy set at
block granularity under one prefix index. A shared system prompt is therefore stored once, not once per parked
conversation as llama.cpp's and Strata's blobs store it. The offer has these properties:
- **exact by construction:** a hit's bytes are the same whichever tier held them;
- **deterministic resume:** batch composition never follows copy or I/O timing;
- **plugins:** the tier, the VRAM victim choice and the checkpoint policies are plugins.

## 1. Goal and acceptance

Keep KV pages and recurrent checkpoints that VRAM cannot hold in host RAM and on SSD instead of dropping them, so that
agent sessions resume from a lower tier rather than by recomputing. This should beat the consumer engines' host
caches on an agent trace, with outputs that do not depend on where state was kept.

1. **Strategies are plugins** (D1, D3).
   - Shipped in K:
     - the tier `std.state_tier` (slot `tier`): a pinned-RAM level and an optional SSD level, with store admission as
       an option (`publish`, the default; `hit:<n>`, HiCache's selective mode; `evict`, write-back) and O_DIRECT as
       an option of the SSD level (Maya KG3's buffered files are `direct = false`);
     - the checkpoint policies of `std.state.recurrent`: `branch` and `answer:<k>` (new), and `prompt_end` and
       `every:<n>` (existing);
     - the VRAM victim policy `evict.root_pin` (Strata C1's "root pinned, others LRU") in the new slot
       `eviction_policy`; with no plugin there, the core keeps today's LRU.
   - **Held-out proof.** The tier protocol and std's policy protocol are tagged at the end of step 3
     (`v1-kv-tiers-api`). After the tag, two strategies this document does not design land as commits whose diff
     touches only their own plugin directory and tests (`check_core_diff.sh`, `git diff --stat`):
     - `policies.last_turn`: Maya KG2's checkpoint at the last turn's start, a `CheckpointPolicy` by `Ref` in
       `recurrent.policy`. It tests whether `SeqView` exposes enough. The prediction is that it needs the untrimmed
       match length, which K does not add. If so, that is recorded as K's proof finding (as X's F10 was,
       `docs/v1-offload.md:1544-1555`) and added as a minor in the same milestone;
     - `state_tiers.zlib`: a host tier the protocol was not shaped around. It is exclusive (stores only at VRAM
       eviction), keeps lossless zlib blobs of variable size in plain memory, decompresses on a CPU thread into its
       own S2 staging pool, and puts the CPU stage in its virtual clock. It must pass tier invariance like
       `std.state_tier`.
   - Every core seam (§4) is proven first with a toy kind and a toy tier in `core/tests`.
2. **Exactness** (D4, §6).
   - **Tier invariance.** The reference is the same configuration with each session's turns run back to back and no
     filler, so nothing is demoted. The tested run interleaves the sessions with a filler that forces every return
     through the tested level. The output streams and each request's `cached_tokens` are byte-identical (panel C2).
     This holds for host only, host + SSD, and SSD restored after a restart.
     - Development: Qwen3-8B on the A5000.
     - 27B: the roomy-vs-tight pair runs on the RTX 5090, where the roomy run fits; the RTX 5080 runs the lock's trace
       for run-identity only.
     - A tiers-off null test shows that the checkpoint pool's size alone does not change output bytes (4 vs 16 units).
   - On a concurrent trace with ≥ 32 loads per level: run-identical step lists and outputs over 3 fresh engines, with
     unseeded I/O and copy delays injected.
   - The new checkpoint policies are judged as today's prefix hits are:
     - a `branch` payload is prefill-produced. It is bitwise equal to the state recomputed with the same chunk
       boundaries, and `bounded` across chunkings (C14);
     - `answer` is decode-produced, so `bounded`;
     - all of them pass the bounded multi-turn check against its one floor (§6).
3. **Development gate.** Qwen3.8-27B UD-IQ3_S on the RTX 5080 with its lock's settings (`max_seqs` 4, q8_0 KV,
   `chunk_with_decode` 512), plus UD-Q4_K_M on the RTX 5090. Same file and host; medians of interleaved rounds. Every
   row is predicted in §8 before it is measured. Rates in the bars are the **link and device** rates: X's M1 blob rate
   for H2D (47.5 GB/s) and K-M4's large-block O_DIRECT rate for SSD, never the path's own (C6).

   | # | Measure | Bar |
   |---|---|---|
   | 3a | Resume microbench: a parked conversation of 4K / 16K / 32K tokens plus a 256-token append, resumed from VRAM, from RAM and from SSD. Units forced fragmented (a scattering prelude; runs per load reported). Two cells: idle, and beside three decoding sessions. ≥ 10 in-process reps per cell over conversations parked once (C18) | RAM: TTFT ≤ VRAM TTFT + 1.5 × KV bytes / 47.5 GB/s + one decode step. SSD: TTFT ≤ VRAM TTFT + 1.5 × (bytes / SSD rate + bytes / 47.5 GB/s) + one decode step |
   | 3b | Three sessions decoding while a fourth resumes 32K tokens from RAM. 30 reps (10 parked conversations × 3 rounds), ≈ 360 inter-token gaps per arm. Control arm: the same conversation resumed from VRAM, with the same append step | over the steps from the resume's admission to its first token: the three decoders' median gap ≤ 1.1× the control's and p90 ≤ 1.2× |
   | 3b-SSD | as 3b, the resume from SSD (report) | reported; engine-loop host time per step during the read reported |
   | 3c | `agent_park` trace (§8): 12 sessions over 4 clients, parked and resumed. Compared with Holon VRAM-only (today). Turns 1–5 and 6+ reported apart | TTFT p50 ≤ 0.25×, output throughput ≥ 1.2× |
   | 3d | `agent_park` vs `llama-server` with `--cache-ram` sized like Holon's host level, plus `--ctx-checkpoints`; same file, card and KV format; the llama.cpp build pinned in the evidence (the default 8 GiB arm reported) | TTFT p50 and p90 ≤ 1.0×, output throughput ≥ 1.0× |
   | 3e | `claude_code` truncated to 2 turns per session (16K shared system prompt, `bench/README.md:58-60`): TTFT of each session's first request, `branch` on vs off, tiers on in both arms | p50 ≤ 0.2× |
   | 3f | Restart (`state_tier.ssd.persist = close`): two engines with the same parking history; one resumes a parked 16K conversation without a restart, the other closes, restarts and resumes it (C3) | TTFT within 3a's SSD bar plus the restore's verification time (reported); outputs and `cached_tokens` byte-identical between the two |
   | 3g | pi agent test re-run on the RTX 5090 (report, not gated: agent outputs are not reproducible) | the share of uncached input that is the previous turn's own output, predicted to fall from 44% to ≤ 10% if K-M1 shows verbatim re-rendering |
   | 3h | Tier overhead on the RTX 5090: `agent_park` with VRAM capped to the RTX 5080's KV capacity, against a roomy run (15.3 GiB weights + ≈ 5 GiB KV + ≈ 5 GiB checkpoints) (C11) | TTFT p50 ≤ 1.1× roomy + one decode step |
   | 3i | Qwen3-30B-A3B offload with the host level on the RTX 5080: decode TPOT while loads run beside X's expert fetches (report, C20) | reported |

4. **Memory** (§7).
   - The host level is allocated exactly from S2 host pools, after the final materialize, and accounted against
     `host_memory` after X's pools.
   - `std.state_tier` refuses at startup a host level smaller than its in-flight bound (`max_seqs` × the units of one
     `max_model_len` load), with a per-pool report. Inclusion is no longer required (D5).
   - The SSD level refuses a directory without its bytes.
   - With tiers off, VRAM is unchanged. With tiers on, no new device pool is added, and the checkpoint pool is sized
     by §7's staging formula.
   - The `dropped_publishes` counter (`stats()`) is 0 on `agent_park` at the tuned pool size.
5. **Core.**
   - Only §4's seams KS1–KS5, one additive ABI minor, each toy-proven. K's ABI is **1.4 or 1.5 by merge order**:
     milestone FN also proposes 1.4 (seams C1, the residual width, and C2, state from blocks and FFN units, ≈ +60
     lines). K's and FN's core changes are serialized. Whichever lands its seams first takes 1.4, and the other
     rebases and takes 1.5.
   - `check_abi --base` and the line check are relative to **K's step-1 base**: the main-branch commit K's seams
     rebase onto, after FN's seams if FN lands first.
   - **At most +300 code lines over that base** (§4 itemizes ≈ 285).
   - A property test shows that `could_fit` and eviction agree under random traffic with demotions, promotions, pins
     and tier drops.
   - With tiers off, every existing lock's snapshot stays byte-identical.
6. **Locks.**
   - The 27B agent locks gain the tier and the `branch` and `answer` policies, on the RTX 5080 and 5090, after the
     bounded check.
   - Every other lock is propagated once, by the track that closes second (D14).

Moved out of K, with reasons in D2 and §10:
- **Project Maya's multi-session ratio** (`docs/v1-offload.md:598`) needs Maya on the same trace, which is GLM's
  milestone. K reports Holon's own N-session / 1-session throughput ratio on `agent_park` instead.
- **`prefill:<n>`** (rev 1's tentative in-step checkpoints) is deferred: no row measured it, and `every:<n>` already
  serves long prompts (B13).

Non-goals of K:
- K4, KV streaming for sparse-attention mixers (FN and GLM; how K4 reads K's host copies is D15);
- layer-wise loading and per-layer ready events (D12);
- CPU attention (`-nkvo`);
- remote tiers (Mooncake, NIXL, object stores);
- compressed or quantized checkpoint codecs as a feature (the zlib held-out is a proof, not a shipped strategy);
- an explicit save / restore HTTP API;
- checkpoints from verify rows (MTP on, see D8);
- VMM rebalancing of KV against experts (plan §10).

## 2. Decisions

**D1 — Strategy set.** Classes:
- **core:** a generic mechanism (§4);
- **T:** the tier plugin (slot `tier`) or an option of `std.state_tier`;
- **E:** a VRAM victim policy (slot `eviction_policy`);
- **K:** a state-kind policy (`std.state.recurrent` options, or a `CheckpointPolicy` plugin by `Ref`).

| # | Strategy | Reference | Class | K default |
|---|---|---|---|---|
| T1 | Host-RAM level for KV pages and payloads, per block, write-through at publish | vLLM native offload, HiCache `write_through`, llama.cpp `--cache-ram`, Strata C1/C2 | core + T | on in the 27B agent locks |
| T1s | Selective store (`state_tier.host.admit = hit:2`) | HiCache `write_through_selective` | T (option) | off |
| T1b | Write-back at VRAM eviction (`admit = evict`) | R §4.2 K2, LMCache | T (option) | off |
| T2 | SSD level below RAM: slot files, O_DIRECT, written back when RAM evicts | HiCache L3, LMCache disk, Strata C3 | T | opt-in (`state_tier.ssd.dir`) |
| T2b | Buffered SSD files (`state_tier.ssd.direct = false`) | Maya KG3 | T (option) | off |
| T3 | Persistence: a manifest written at clean close restores the index at startup | Maya `SLOT_KEEP`, Strata C3, R K3 | core + T | opt-in (`state_tier.ssd.persist = close`) |
| P1 | Conversation parking = the prefix index with T1 / T2 (no separate feature) | Strata C2, llama.cpp `--cache-ram` | — | with T1 |
| P2 | Swap-on-preemption = write-through pages and payloads stored at registration; the resume is a prefix hit | vLLM V0 swap, subsumed (R §4.1) | core | with T1 |
| E1 | VRAM victims: least recently used (today) | all | core | on |
| E2 | `evict.root_pin`: shallow entries with many hits are kept while others can go | Strata C1 | E | off; measured on 3e |
| C1 | `prompt_end` checkpoints (exists, `std/holon_std/state/recurrent/kind.py:46-55`) | Strata C1 turn start | K | on |
| C2 | **`branch`:** a checkpoint where a request leaves a cached chain that continues (the shared system prompt's end, found rather than configured) | Strata C1 "system prompt ≥ 2048", Maya KG2 | K + core fact | on |
| C3 | **`answer:<k>`:** a tentative checkpoint at each aligned position a decode row reaches; the request keeps its last k | plan §10 item; Strata's opt-in tail checkpoints | K + core | on if K-M1 shows the gain, k from K-M1 |
| C5 | `every:<n>` (exists) | Strata, Maya every 16K | K | off |
| H1 | `policies.last_turn` | Maya KG2 last-turn start | K plugin | held-out |
| H2 | `state_tiers.zlib` | none (an unanticipated shape) | T plugin | held-out |
| — | ARC, layer-wise load-back, K4 streaming, `prefill:<n>` | — | — | deferred (§10) |

**D2 — Vehicle, comparisons, and what moves out.**
- **Gate vehicle: Qwen3.8-27B GGUF.** It has both kinds of state K tiers: q8_0 KV for 16 attention layers
  (34,816 B per token) and 48 GDN layers whose checkpoints are 147 MiB each (`docs/m03-qwen38.md:59-60`). It is the
  pi agent test's model, and its RTX 5080 (UD-IQ3_S) and RTX 5090 (UD-Q4_K_M) locks exist
  (`plans/unsloth/Qwen3.8-27B-GGUF/…/rtx-5080/…/10.lock`: `max_seqs` 4, `recurrent.checkpoints` 4, q8_0).
- **Development vehicle for KV-only paths and stream-timing plants: Qwen3-8B** (bf16 and Q4_K_M) on the A5000. It is
  the bounded check's calibration model (`docs/design.md:1307`) and has paged state only (16-token blocks, 144 KiB per
  token in bf16). Tiny random models (`testing/holon_testing`) carry the CPU tests and the logic plants.
- **Qwen3-30B-A3B (offload).** Report-only rows (3i, §7): X's expert host pools and the host level share
  `host_memory` and the H2D link.
- **Gating reference: `llama-server`** with `--cache-ram` and `--ctx-checkpoints`. It is the consumer default (R §4.1)
  and serves the same GGUF files.
- **Report-only references:** SGLang HiCache and vLLM's native offload on Qwen3-8B, on the RTX 5090, if their
  environments in `bench/.envs/` serve it.
- **Maya's multi-session ratio** (×1.41 / ×1.35 at 2 / 4 requests, V100, R §3.1b) is measured on GLM-5.3-Flash, in
  the GLM milestone, with this doc's machinery under it.

**D3 — The core owns the VRAM boundary; one tier plugin owns everything below** (panel B2). The PrefixIndex is
core-owned (design §6.3). Its invariants decide liveness, and only VRAM units bear on liveness: `could_fit`
(`core/holon/core/state.py:366-385`) counts device units, and a lower copy never blocks device progress. So:
- **The core owns:**
  - whether an entry is resident (has VRAM units) and whether the tier holds its paged copy and which payload copies;
  - VRAM eviction (eligibility in the core, victim choice from the slot `eviction_policy`);
  - promotion into VRAM, and gating a request on its loads;
  - waiting for copies before units are reused.
- **The tier plugin (slot `tier`, KS4) owns:**
  - its levels, their storage, capacity and eviction;
  - its copies across the link (streams, batched DMA or an SM gather kernel through a kernel op);
  - its I/O threads, its virtual clock and its manifest;
  - store admission.
- **A state kind owns** where checkpoints are taken (D8).
- **The core never names** a tier, a level, a format or a model (invariant 1, `scripts/ci/check_core_denylist.py:8-11`).

*Why one tier object and not N levels in the core.* Rev 1 put copies per level, per-level child counts, per-level
LRUs, the SSD watermark, write-back tickets and restore in the core: 250 lines of KS1–KS2 against a 137-line
`prefix.py`. The scheduler needs none of it. A VRAM eviction either finds a lower copy or drops the subtree, so lower
eviction never blocks device progress. A10's concern (demoted children pinning a VRAM parent) needs only resident-child
counts. N levels inside std change nothing the core decides.

*Why not X's `Tier`.* `std.tiered`'s `Tier` (`std/holon_std/tiered/protocols.py:46-74`) is a std protocol for
read-only, in-graph expert reads. The core cannot call a std protocol, and K's tier is called by the core between
steps, writes too, and holds index-shaped state. The families are therefore named apart: the slot is `tier`, std's
package is `holon_std/state_tier/` (added to the std independence list, `pyproject.toml:93-99`), and plugins are
`state_tiers.*`, beside X's `tiers.*`. Nothing is imported from `std.tiered`: `files.py`'s `_open` is private and
read-only, so the SSD level has its own few lines (B9).

*Selection.* `EngineConfig.slots["tier"] = "std.state_tier"`, opt-in like the logits processor
(`core/holon/core/engine.py:189-191`). There is no new `EngineConfig` field, so `plan_id`s do not move with tiers off,
and `--slots tier=…` selects the tier as its help says (B7). Its options are plugin options. Host paths and per-host
sizes are declared `planned = false` (KS4, B6). The slot `tier` takes over design §4.2's planned `connector` slot
for prefix-index offload. A v2 PD transfer can implement the same protocol, or its own slot then.

Plugin kinds:

| Kind | Selected by | Contract | Instances in K |
|---|---|---|---|
| tier | `slots["tier"]` (single, opt-in) | `StateTier` (KS4, §4) | `std.state_tier`; held-out `state_tiers.zlib` |
| VRAM victim policy | `slots["eviction_policy"]` (single, opt-in) | `EvictionPolicy.victim(candidates)` (KS5) | `evict.root_pin` |
| checkpoint policy | `recurrent.policy = "prompt_end,branch,answer:2"` (built-in names or `Ref`s) | std's structural `CheckpointPolicy.next(view) → (at, tentative) \| None` | prompt_end, every, branch, answer; held-out `policies.last_turn` |

**D4 — Exact means tier-invariant; decisions never read timing.**
- **Bytes.** Every move is a byte copy of whole units: device ↔ host by the tier's copies, host ↔ SSD by the tier's
  I/O, and a host payload into a slot by the kind's own `adopt` MemOp. A request computes on the same bytes whichever
  level held them, so tier invariance is a property of the copies and of the hit set. No kernel changes.
- **Logical at decision time, physical after the ticket** (A5, C5). Which entries are resident, demoted, promoted,
  dropped or hit, which units are free, and when a request may run are all functions of the calls made so far and the
  steps scheduled. Copy and I/O completion never enter them.
  - A unit is free the moment its entry is demoted or dropped. Its reuse is ordered after the copy that reads it:
    the next launch waits, on the device, for every store whose source units were freed since the last launch. A
    load into reused units is issued after those stores.
  - The tier obeys the same rule for its own storage. A host unit whose bytes go to SSD is free when the write is
    decided, and the read or copy that reuses it waits on the write's ticket.
- **Resume by a virtual clock** (C9, replacing rev 1's fixed `r`).
  - The tier keeps a clock in virtual seconds, advanced at each launch by a fixed per-step cost model
    `a + b·tokens + c·rows` (tune measures `a`, `b` and `c` per lock).
  - A load gets a due time chained over the lanes it passes, each a virtual queue:
    `due_read = max(read_lane, now) + bytes / read_rate`, then
    `due = max(h2d_lane, due_read) + bytes / h2d_rate`.
    A forced write-back (C12) adds its bytes to the write lane ahead of the read that needs the units.
  - A load of at most `stall_bytes` is due now: the step waits for it.
  - A request with loads is runnable at the first step whose clock is ≥ its due time (KS2). A 512-token prefill step
    advances the clock by its own cost (≈ 0.22 s on the 27B), so a resume behind prefill-heavy steps is not
    penalised in decode steps.
  - If nothing else is runnable, the earliest-due load-gated request runs at once, and its step waits for the copy.
    That is deterministic, because whether anything else was runnable is schedule state.
  - The launch of the first step that reads a load waits for it on the device: a late copy stalls that step and never
    reorders anything.
- **What tier invariance does not cover.** Whether a hit happens at all depends on capacities, as today. Comparisons
  that change the hit set are `bounded` (L1.5, `docs/design.md:1394`); acceptance 2's comparisons hold the hit set
  fixed (C2's reference).

**D5 — Write-through by default.** R §4.2 K2 proposed write-back at eviction (line 578). This design keeps
write-through as `std.state_tier`'s default, with write-back as an option (T1b):
- **No D2H burst at eviction.** With write-through the host copy already exists. A VRAM eviction is a pure free, and
  promotion may evict such entries (KS2). Under write-back, evicting a 16K conversation enqueues 570 MB of D2H,
  ≈ 12 ms at 47.5 GB/s, that the reusing step may wait for.
- **Swap-on-preemption is free** (P2).
- **The PCIe cost is small.** The 27B prefills ≈ 2.2–2.5K tokens/s on the RTX 5080
  (`evidence/bars_release-qwen3.8-27b-ud-iq3_s_rtx-5080_2026-10-16.json`). That is ≈ 85 MB/s of KV, ≈ 0.2% of the link,
  on the tier's own D2H stream. A checkpoint store is 154 MB, ≈ 3.2 ms.
- **Inclusion is no longer a startup rule.** Rev 1 needed host ≥ device units so that host eviction always found a
  victim. With the core's two-level view, a tier that drops a copy of a resident entry just clears its flag; that
  entry, evicted later, is stored at eviction or dropped. The host level's only hard floor is its in-flight load
  bound (acceptance 4).
- **Stores are ordered after the in-flight step** (A2). With overlap, the commit of step s-1 publishes blocks that
  step s, already launched, may still be computing (`engine.py:454-463, 507`; `state.py:335`). A paged store is
  issued with `after` = the last launched step's end event. A payload store is issued with `after` = the end event of
  the step that wrote it, the only writer of a staging unit (D7).
- **Store admission is the tier's.** `store` may refuse. A VRAM eviction of a resident entry with no lower copy asks
  again with `evicting = True`. If refused, the entry is dropped with its subtree, which is all non-resident (the
  resident set is prefix-closed), so nothing is orphaned.

**D6 — Eviction invariants** (panel A10, B1).

*Residency.*
- An entry is **resident** if it holds VRAM units. Resident entries are prefix-closed: `resident(child) ⇒
  resident(parent)`.
- Each entry counts its resident children (`kids`) besides all children. A non-resident entry has its paged copy in
  the tier (`lower`). Payload copies are flagged per spec, in VRAM (staging, D7) or in the tier.
- The core keeps an LRU of resident entries only, so the VRAM scan stays today's size whatever the tier holds. Lower
  recency is the tier's (it receives `hit` calls).

*VRAM eviction.*
- **Eligible:** resident, `kids == 0`, units held only by the index, and not load-pinned.
- **Victim choice:** the slot `eviction_policy` gets the eligible entries lazily, least recently used first, and
  returns one. Without a plugin, the first is taken, which is today's order. The plugin bounds its own scan.
- **Outcome:** a victim with a lower copy is demoted (VRAM units freed). Otherwise it is offered to the tier as an
  eviction store, and dropped with its subtree if refused. Demoting a leaf can make its parent eligible, as today's
  upward climb does (`prefix.py:101-104`).

*Lower drops.* The tier evicts its own copies and reports them through `dropped()`, which the core polls at each
schedule and after each `store`. A dropped copy of a resident entry clears its flag. A dropped copy of a non-resident
entry drops that entry and its (non-resident) subtree, and the tier is told about the descendants. Leaf-first order
in the tier is the tier's own obligation; dropping an interior copy loses hits, never correctness.

*Payloads.* With a tier, VRAM payload units are staging only (D7). Without one, today's payload LRU is unchanged.

*Agreement with `could_fit`.*
- `could_fit` keeps today's formula: device units held only by the index count as freeable.
- **Claim:** each such unit belongs to an eligible entry or to an ancestor that becomes one. A request holds exactly
  the chain it follows (`prefix.py:8-10`, kept by `_swap`), so an unheld resident entry has only unheld resident
  descendants, and evicting resident leaves upward frees it. Non-resident children are not counted (`kids`).
- Load pins are references, so pinned units are not counted, and are not eligible either.
- **The property test** (KS1) drives random publish / hit / release / preempt / abort / demote / promote / pin /
  tier-drop / eviction-store-refusal sequences against a toy tier. After every operation it checks:
  - prefix closure of the resident set;
  - that every entry's parent exists;
  - `kids` against a recount;
  - that `could_fit(seq, n, releasing)` is true exactly when releasing those requests and evicting yields the units.

  It extends `core/tests/test_state_liveness.py`'s random-traffic harness.

**D7 — Promotion, publish onto a demoted entry, and payloads through staging.**

*Admission* (A3, C21). Each step proceeds in order, all on index state:
1. The fixed point runs over the matched chain, with payloads in VRAM or in the tier counting alike.
2. The resident prefix is taken (increfed) **first**, so no eviction below can touch the chain being hit.
3. VRAM units for the non-resident tail are allocated from free units plus evictions of eligible entries that have
   a lower copy (pure frees). A promotion never preempts and never issues an eviction store.
4. On a shortfall, the tail is trimmed to what was allocated, and the fixed point runs again on the trimmed chain.
5. The tail entries take those units (the index's reference, plus the request's own as for any hit), are
   **load-pinned**, and the tier's `load` copies their pages in. The `after` events are the last launched step and
   any pending stores on reused units, so a unit freed at the previous commit, still touched by the in-flight step,
   is never overwritten under it (`engine.py:575-576`).
6. A payload adopted from the tier goes **host → the request's slot units directly**. The tier's `Load` names the host
   copy (an S2 host pool of the payload's unit shape). The kind's own `adopt(view, host_payload)` MemOps run as
   today's `pre_forward` adopt in the first step that reads it (`state.py:203-213`), after that step's launch has
   waited for the load (a pending SSD read included). No VRAM payload unit is taken, and nothing is pinned in VRAM.

   The cost is the payload's H2D inside that step: 154 MB ≈ 3.2 ms on the 27B.
7. The request is **load-gated** until the clock reaches its due time (D4).

*Publish onto a demoted or loading entry: promote into place* (A1, the blocker). Today a request that recomputes
blocks the index already holds (a hit trimmed back to the last payload, `state.py:190-201`) swaps onto the cached
units at publish (`state.py:338-345`). This is the common case on agent traces: turn 2 recomputes turn 1's answer
blocks. The rules:
- Entry resident and settled: `_swap`, as today.
- **Entry non-resident:** the request's own units for that block become the entry's VRAM units (the index increfs
  them, so no allocation is needed). The tier `load`s the entry's cached bytes into them, with `after` = the last
  launched step, which may still be writing them. The request then reads turn 1's cached bytes from its next step,
  exactly as the roomy run does after its swap. Prefix closure holds, because publish runs in chain order, so the
  parent is resident by then.
- **Entry loading** (another request's promotion in flight): `_swap` onto its units, and the load joins this
  request's wait set.
- **Rule for every step:** the launch waits on the device for every load in the wait sets of its rows' requests, then
  clears them. This covers a promoted request, a request that hit a loading entry, and a promote-in-place.
- A payload registered on an entry that already has one, in VRAM or in the tier, is freed, as `attach` does today.
  Both runs then keep the same (earlier) bytes.

*Pins and cancellation* (A6). A load is pinned (one extra reference on its units, so `could_fit` and eviction both
skip them) from its issue until a launch has waited on it. That launch is the first step reading it, or the next
step if every request waiting on it is gone. Then the core calls the tier's `loaded`.
- Preempting or aborting a load-gated request releases its references and clears its wait set, and the scheduler
  forgets its gate. Its entries stay index-owned, and the load lands under its pin.
- No load is cancelled, so no copy lands in units somebody else owns, and no event is waited on before it exists.

*Staging payloads* (A4, C1). With a tier, the checkpoint pool (`recurrent.checkpoints`) holds only payloads between
`publish` (the step's launch) and registration (its commit):
- **At registration**, the payload is stored with `after` = its producing step's end event, and its VRAM unit is
  freed logically. The next writer of that unit is a later step's publish, ordered by D4's free rule. The D2H
  (≈ 3.2 ms) runs during the next step, so the wait costs nothing in practice.
- **Tentative payloads** (`answer`) are stored the same way, flagged tentative (KS3); none waits in VRAM for its
  request to release.
- **Demand:** with overlap, a staging unit is held from the launch of step s until the commit of s, which runs after
  the launch of s+1. Peak demand is therefore `max over s of pubs(s) + pubs(s+1)`. A decode row publishes at most
  once per 64 tokens, so once in any two consecutive steps; a prefill row publishes at most once per step. Hence
  `units ≥ max_seqs + p` per payload spec, with `p` the largest number of rows that prefill in two consecutive steps.
  Without overlap, `units ≥ max_seqs`. `2 × max_seqs` is the bound that never drops.
- A publish that finds no unit is skipped, as today (`recurrent/kind.py:73`), and counted in `dropped_publishes`;
  acceptance 4 requires 0 on `agent_park`.
- Without a tier, nothing changes.

*No future object in the ABI.* The load is the tier's object. The core holds it in wait sets and gates. Rev 1's
overloading of `Status.BLOCKED` is gone (A6): load gating is the scheduler's own state, so `idle_for` and the
no-progress check never see it. A preempted request is never left gated. Design §6.4's option (b), layer-wise loading,
stays planned (D12).

**D8 — Checkpoint positions** (`std.state.recurrent`, all exact or bounded, never approximate).

*`branch`.*
- The core reports `SeqView.branch_at`: the matched prefix's length when its last entry has children, else 0. It is
  computed in `admit` before the fixed point trims the hit.
- The policy checkpoints there in-step during prefill. The position is block-aligned, so it is on the 64-token grid
  and costs no cut (`docs/m03-in-step-snapshot.md:94-111`).
- Why: today, a session sharing only a system prompt with earlier sessions gets no recurrent hit at all. Its
  `prompt_end` payloads all lie past the system prompt (`recurrent/kind.py:54`), and the fixed point trims the KV
  match to the last payload (`state.py:190-201`). On `claude_code` (16,384-token system prompt), every session's first
  turn on the 27B then recomputes 16K tokens, ≈ 7 s at 2.2–2.5K tok/s. With `branch`, the second session checkpoints
  at 16,384 and every later one hits it (3e).

*`answer:<k>`.*
- Each decode row that reaches an aligned position publishes a **tentative** payload (`Payload.tentative`, KS3). The
  publish is the post-sample D2D copy that `every:<n>` uses (`state.py:236-258`): 147 MiB every 64 decode tokens,
  ≈ 0.3 ms per ≈ 1 s of decode on the RTX 5080. The D2H store follows at registration.
- The core keeps a request's last `keep_tentative` (= k) tentative payloads. A newer one drops the oldest from its
  entry and from the tier; detaching is safe, because adopt pins hold their own references (A12b). At release
  (finish, abort or preempt) the kept ones simply stay as ordinary payloads.
- A payload registered after its request released (the preempted request's in-flight step, registered through the
  re-hash path, `state.py:281-287`) registers as an ordinary payload (A12a).
- This is plan §10's end-of-answer checkpoint: "the last aligned position the next prompt shares"
  (`docs/plan.md:679-680`).
- **Why k > 1 may be needed.** A template that re-renders the answer differently (a tool call's arguments
  re-serialized from JSON, say) diverges inside the answer, so the useful checkpoint is the last crossing before
  that point. K-M1 measures where re-rendered answers diverge, fixes k (default 2) and predicts 3g.
- Decode-produced checkpoints are `bounded` (`docs/m03-qwen38.md:22-23`), the same class as today's hits across
  generated KV pages (`docs/design.md:900-902`).
- Verify rows publish nothing (`state.py:239`), so with MTP on the policy is idle. The 27B locks run without drafting.

*Payload recency in the tier.* A turn leaves `prompt_end` and k answer payloads on its chain. Later turns match
through those entries without adopting them. The tier therefore orders payload copies by store or adopt (`hit`'s
`adopted` argument), not by entry hits, so stale interior payloads go before live KV.

*Policies are std's; a new one is a plugin.* `recurrent.policy` is a comma list of built-in names or `Ref`s to
plugins with the structural `next(view) → (at, tentative) | None`. Plugins cannot import std
(`pyproject.toml:85-89`, `:93-99`), so the protocol is duck-typed and documented in std. The kind takes the earliest position
over its policies.

**D9 — No `HitLen.cost`: the deepest hit wins; the rate check is tune's.** X's outline lists `HitLen.cost`
(`docs/v1-offload.md:599`). A cost changes a decision only where loading a deeper hit is slower than recomputing the
difference. Per token on these machines, that never happens for RAM and rarely for SSD:

| | Load rate | Load per token | Recompute rate | Recompute per token |
|---|---|---|---|---|
| 27B, RAM | 47.5 GB/s | ≈ 0.73 µs (34,816 B) | 2.2K tok/s (RTX 5080) | ≈ 450 µs |
| 27B, SSD | ≈ 2.6 GB/s (X's M7) | ≈ 13 µs | 2.2K tok/s (RTX 5080) | ≈ 450 µs |
| Qwen3-8B bf16, SSD | ≈ 2.6 GB/s | ≈ 57 µs (144 KiB) | ≈ 8.3K tok/s (RTX 5080 pp512, `docs/plan.md:301`) | ≈ 120 µs |

The worst row is the 8B's KV from SSD, still ≈ 2× faster than recomputing. Only a payload adds a fixed cost: a 154 MB
checkpoint from SSD (≈ 59 ms) loses to recomputing a gap shorter than ≈ 130 tokens, at most ≈ 59 ms per hit.

So K takes the deepest hit (today's fixed point, unchanged). The rate check moves out of the core (B11): tune refuses
to write a lock whose SSD level loads slower per token than the lock's measured recompute rate, with both in tokens
per second. A remote tier slower than recompute would bring back `HitLen.cost`; until then it is a field nothing reads.

**D10 — Copies are the tier's** (rev 1's KS4 leaves the core; B12, C8).
- `std.state_tier` copies on its own H2D and D2H streams. PCIe is full duplex, and a store queued ahead of a load
  would otherwise delay it.
- KV device pools are layer-major (`std/holon_std/mixers/gqa/units.py:136-141`). A 27B q8_0 page is 16 chunks of
  34,816 B, and a fragmented 32K load is 32,768 chunks. Today's per-run loop (`memory.py:141-149`) would spend
  ≈ 0.3 s of host time on that.
- The std copier builds the chunk address lists vectorised and issues them through a kernel op, `state_copy@1`,
  bound like any site. K adds providers in `kernels.expert_io`, which already has both paths
  (`plugins/holon_plugins/kernels/expert_io/plugin.py:84-110`):
  - one `cudaMemcpyBatchAsync` per batch;
  - the SM scatter/gather kernel, 0.97–1.02× DMA in X's M1 (`docs/v1-offload.md:645`).

  K-M2 picks between them. Without a provider (CPU, tests), the copier falls back to the per-run loop.
- C8's point stands: a block-contiguous *host* layout does not reduce the device-side chunk count, so it is dropped
  as a copy fallback.
- The SSD level does use a **unit-major host layout** (a host pool of one segment per unit). A 27B page is then one
  557,056 B iovec (136 × 4,096), aligned for O_DIRECT on 4Kn drives. That is legal because the copies across the
  host pool are the tier's own, not S2 MemOps (which require equal unit shapes, `memory.py:130`).
- Payload adopts (D7) are the kind's MemOps from a host pool of the payload's own shape, so they stay S2-legal.

**D11 — The SSD level** (`std.state_tier`, option `ssd.dir`).

*Layout.* One preallocated slot file per unit shape (`fallocate`), sized by `ssd.bytes`. A slot holds one unit-major
unit.

*I/O.*
- `os.preadv` / `os.pwritev` of whole units on `ssd.io_threads` (default 2: X's M7 reached 2.5–2.7 GB/s from two
  readers, `docs/v1-offload.md:649`). The threads are pinned off the engine's `loop_core`.
- O_DIRECT where the filesystem allows it; otherwise buffered with `posix_fadvise(DONTNEED)`, or by option (T2b).
- K-M4 measures the engine loop's host time per step beside the threads (GIL).

*Write-back and determinism.*
- The SSD level admits at host eviction. The host level keeps a free-unit watermark (`host.reserve`). Its default is
  the units of one `max_model_len` load, so an SSD load seldom forces a write-back (C12).
- Below the watermark, std's step hook demotes host LRU leaves to SSD. The decisions read logical counts only (D4):
  the host unit is free when the write is decided, and a reuse waits on the write's ticket.
- An entry being written back is logically on SSD. A hit on it is a read with an SSD due time, even if its host bytes
  are still physically there. A tier may use them, because the due time is already fixed (A5).
- Copies redundant with a faster level (an SSD copy of an entry whose host copy exists) are dropped first at each
  level (A9).

*Persistence (`persist = close`)* (A7, C4).
- The manifest is a **snapshot written only at a clean `Engine.close`**:
  1. every resident and host-level entry not yet on SSD is written back (sustained write rate, K-M4);
  2. the data is `fdatasync`ed;
  3. the manifest is written to a temporary file with one record per block (key chain via parent, tokens, slot per
     spec and payload, and a crc32 of the slot's data, computed by the I/O thread from the host copy before the
     write), `fsync`ed and renamed into place, and the directory `fsync`ed.
- **At startup**, the manifest is read, then unlinked and the directory `fsync`ed, before any slot is rewritten. A
  record can therefore never name a slot that was reused after it was written. This has the effect of durable
  tombstones for every slot at once, without one per reuse.
- Restore verifies each restored slot's crc on the I/O threads (persisted bytes / read rate, reported in 3f).
  Records failing it are discarded with their descendants. Records are sorted parents first (host eviction is
  leaf-first, so the snapshot order is not), and orphans are discarded.
- A crash leaves no manifest, so nothing is restored. That is strict on purpose: persistence is for planned restarts.
- **Identity** (A13, B6): `plans.state_identity(engine)`, the plan's id computed without `planned = false` options and
  without the clamp-dependent `max_model_len` (`engine.py:262`). If the plan cannot be computed (`plans.py:151-155`),
  a tier asking for persistence fails at startup. Keys are re-hashed from tokens at restore; Python's `hash` of an
  int tuple is not salted, but nothing relies on it across processes.

*Why not X's I/O path.* `kernels.cpu_task`'s workers serve in-graph reads signalled by the GPU, file → host only,
with tasks bound before the worker starts (`plugins/holon_plugins/kernels/cpu_task/plugin.py:53-57`). K's I/O is
host-triggered between steps and writes too.

**D12 — No layer-wise loading in K.** With load gating, a 16K-token RAM load costs ≈ 12 ms of copy and at most one
step of granularity. Layer-wise overlap would save at most that, and it forces eager or PIECEWISE prefill (R §4.2
lines 590-591). It is built with PD in v2, as design §6.4 plans.

**D13 — Comparison method.**
- Same card, file, context, KV format and host; drafting off (the locks').
- `llama-server` runs with `--cache-ram` sized like Holon's host level (gating, C15) and at its 8 GiB default
  (reported). The build is pinned in the evidence. Holon's `bench/backends/llamaserver.py` drives it.
- Interleaved ABBA rounds, ≥ 3. TTFT, TPOT p50 and p99, and throughput are as `bench/README.md:72-95` defines them;
  3b's inter-token gaps are per step, per decoder.
- Hit tokens per level, load bytes, stalls, eviction-store refusals and `dropped_publishes` come from `stats()`.
- Parked conversations are prepared once per process (a filler parks a set), and each rep resumes a different one
  (C18).

**D14 — Serialized core rounds, one propagation.**
- K's and FN's core seams are serialized: whichever merges first takes ABI 1.4, the other rebases and takes 1.5.
- The core's source is pinned in every lock, so a core change moves every lock. Only the track that **closes second**
  propagates the existing locks, once, and that run covers both rounds. If K closes first, K's close records the
  propagation as owed to FN's close.
- Tiers default off, so every existing lock's snapshot stays byte-identical (acceptance 5) and the propagation
  auto-advances.
- The 27B agent locks are re-tuned with the tier and the new policies, with bounded verdicts (acceptance 6).

**D15 — K4 and K's host copies** (B14). K's host level holds whole blocks in S2 host pools (unmapped by default).
K4's sparse-attention streaming (FN, GLM) reads KV per token inside the graph through `std.tiered`'s
`DeviceResolver`. For it to read K's copies instead of keeping a second one, `std.state_tier` would need a `mapped`
option and a std accessor from block key to host units. Both are std-only, so no core change and no `StateTier`
change. Whether K4 shares or copies is decided in FN, on its own measurements.

## 3. What Holon has today

| Piece | State (file:line) |
|---|---|
| Prefix index | Hash chain of blocks of `lcm(page, alignments)` tokens (64 for the GDN hybrids), keys `hash((prev or namespace, *tokens))`, tokens verified on match (`core/holon/core/prefix.py:48-71`). Entries are an LRU `OrderedDict` with per-spec device units, payloads and a child count (`prefix.py:29-46`). The namespace is `canonical_hash(spec)`, with "persistent tiers must add the checkpoint identity" (`core/holon/core/assembly.py:181-191`) |
| Eviction | LRU leaves whose units only the index holds, climbing while short (`prefix.py:89-104`). Payloads are dropped LRU from any entry when publishing needs a unit, by a scan copying the entry list (`prefix.py:114-123`, `state.py:249-250`). Everything drops; nothing demotes |
| `could_fit` | Counts cached units with one reference left once the releasing requests are gone, regardless of tree shape (`state.py:366-385`). Sound only by the closure invariant (`prefix.py:8-10`). No test of its agreement with `evict` |
| Hits | Fixed point of the non-paged kinds' `hit_length` (`state.py:190-201`). Paged hits share units by reference. Payloads are adopted by a `pre_forward` MemOp, pinned until launch (`state.py:203-213, 271-274`) |
| Publishing | Every completed block, generated tokens included, at commit, up to `min(num_computed, len(tokens))`, so with overlap it includes blocks the in-flight step is computing (`state.py:327-349`; `engine.py:454-463, 507, 562-565, 603-606`). A block another request cached is swapped onto (`state.py:343-345`). Payloads are registered at commit if committed tokens cover them (`state.py:276-290`) |
| Recurrent checkpoints | `prompt_end` or `every:<n>`; a device pool of `max_seqs` units; a publish finding no free unit returns None silently (`recurrent/kind.py:27-38, 46-55, 73`); in-step publishes on the 64-token grid |
| Preemption | By recompute: `release` drops all units, `num_computed = 0` (`sched.py:188-195`, `state.py:387-398`). Published blocks stay cached; the live recurrent state is lost |
| BLOCKED | Set only for a logits processor (`engine.py:323-324, 468-486`); skipped by `_prefill` (`sched.py:133-134`); `idle_for` polls at 5 ms only when every waiting request is BLOCKED (`engine.py:335-343`), else `step` raises on no progress (`engine.py:451-453`) |
| Host pools, async MemOps (X's S2) | Exact mmap + `cudaHostRegister` host pools against `host_memory`, requested by `before_materialize` at the latest (`memory.py:218-248`). MemOps with `after` / `lands_by` on one copy stream, between pools of one unit shape (`memory.py:162-206`). Off CUDA a stream runs work at once and events are no-ops (`platform/__init__.py:183-221`) |
| Shared objects, step hooks (S0, S1) | `BuildCtx.shared` with `before_materialize` / `after_materialize` / `shutdown`; `before_launch` / `after_commit` (`hooks.py:150-187`; `engine.py:193-200`) |
| Plans | Every `EngineConfig` field but `NOT_PLANNED`, and every plugin option but checkpoint paths and frontend-only options, is planned (`core/holon/core/plans.py:35-36, 80-104`); the clamp lowers `cfg.max_model_len` from free VRAM (`engine.py:255-262`) |
| std.tiered | `TieredStore`, `ResidencyMap`, `DeviceResolver`, the `Tier` protocol (`std/holon_std/tiered/protocols.py:46-74`): an in-graph expert residency over read-only weight files. Not reused by K (D3) |
| Cut from ABI 1.0 | slot `tier`, `HitLen.cost`, `Payload.tier` (`docs/release.md:1738`) |
| ABI, budget | `ABI = "1.3"` (`core/holon/api/manifest.py:19`); `check_core_loc.py` 4,603 / 6,000 at this branch's base |

## 4. Core seams (one additive ABI minor: 1.4 or 1.5 by merge order)

None names a model, mixer, format, level or tier implementation. Each is proven first with the toy kind of
`core/tests/test_payloads.py` and a toy tier: a dict-backed tier with a host pool, a deferred copy stream and seeded
delays.

**The ABI additions** (all in `holon.api`, all additive):

```python
@dataclasses.dataclass(frozen=True, slots=True)
class Payload:
    units: tuple[int, ...]
    pool: str
    tentative: bool = False        # new (KS3): kept only as one of its request's last `keep_tentative`

@dataclasses.dataclass(frozen=True, slots=True)
class BlockRecord:                 # new: one index entry as a tier sees it
    key: int
    parent: int | None
    tokens: tuple[int, ...]
    units: Mapping[str, tuple[int, ...]]      # spec -> device units of its paged copy (empty: payloads only)
    payloads: Mapping[str, Payload]           # spec -> payload to store (device staging units)

class Copy(Protocol):
    def wait(self) -> None: ...    # the current stream waits for it (the host first waits until it is enqueued)

class Load(Copy, Protocol):
    due: float                                         # on the tier's clock
    payloads: Mapping[tuple[int, str], Payload]        # (key, spec) -> its host copy, for the kind's adopt

class StateTier(Protocol):         # slot `tier`: single, opt-in (`EngineConfig.slots`)
    def attach(self, ctx: TierCtx) -> list[BlockRecord]: ...   # after the final materialize; restored records
    def clock(self) -> float: ...                               # virtual seconds at the step being scheduled
    def store(self, blocks: Sequence[BlockRecord], after: Sequence[Event], evicting: bool
              ) -> tuple[set[tuple[int, str | None]], Copy]: ...  # accepted (key, spec; None: the pages)
    def load(self, keys: Sequence[int], units: Mapping[str, Sequence[int]],
             payloads: Sequence[tuple[int, str]], after: Sequence[Event]) -> Load: ...
    def loaded(self, load: Load) -> None: ...                  # a launch waited on it: unpin
    def hit(self, keys: Sequence[int], adopted: tuple[int, str] | None) -> None: ...
    def drop(self, keys: Sequence[int], spec: str | None = None) -> None: ...
    def dropped(self) -> list[tuple[int, str | None]]: ...     # its own evictions since the last call
    def clear(self) -> None: ...                               # flush_prefix
    def close(self) -> None: ...                               # Engine.close (persist, release)

class EntryView(Protocol):         # new (KS5)
    key: int; depth: int; children: int; hits: int

class EvictionPolicy(Protocol):    # slot `eviction_policy`: single, opt-in
    def victim(self, candidates: Iterator[EntryView]) -> EntryView | None: ...
```

`TierCtx` carries the device pools a tier may copy, `host_pool(…)` (S2 host pools allocated now, against what
`host_memory` has left), `stream()` (a core platform stream, deferred under the CPU test fixture), the namespace, the
identity (or None), `max_seqs`, `max_model_len`, the block size, and the plugin options. The tier registers itself
through `BuildCtx.shared` at build to get S1's step hooks, which advance its clock. `SeqView` gains
`branch_at: int`. Payload kinds may declare `keep_tentative: int`. Manifest `Option` gains `planned: bool = True`.
SLOTS gain `tier` and `eviction_policy`, both single.

**The determinism contract** (the tier's obligation, in `StateTier`'s docstring and enforced by the judge): every
return value and every `dropped()` list is a function of the calls received and the step hooks, never of copy or I/O
completion; reuse of its own storage waits on its own tickets; a copy being loaded is never evicted before `loaded`.

| # | Seam | Today → change | Toy proof | Lines |
|---|---|---|---|---|
| KS1 | **Residency in the index** | Device copies only; eviction drops (`prefix.py:89-133`) → `Entry` gains `lower`, payload-copy flags, `kids` (resident children), `depth`, `hits`. A resident-only LRU. D6's eligibility, demote / eviction store / drop with subtree; `dropped()` handling; stores' `Copy` kept for units freed while read, waited by the next launch and passed as `after` to loads into reused units | the property test (D6); demote then promote bitwise; reuse of a unit with a store in flight waits for it under the deferred stream | 75 |
| KS2 | **Promotion, promote-into-place, load gating** | Hits adopt VRAM only → D7's admission (prefix first, pure-free evictions, trim and re-run), promote-into-place in `publish`, load pins, per-request wait sets waited at launch, the scheduler's gate (`clock() ≥ due`), idle pull-forward, preempt and abort clearing gates | resume step identical with the toy tier's delay at 0, 1 and 2 steps (run-identical step lists); a trimmed hit recomputing demoted blocks reads the cached bytes; preempt and abort of a gated request with a delayed copy (no unit reused before landing); progress with every slot held by gated requests and a request waiting (A6a) | 80 |
| KS3 | **Stores, staging payloads, tentatives** | Payloads permanent from registration; release drops nothing to store → paged `store` at insert (`after` = last launched) and at eviction; payload `store` at registration (`after` = producing step) freeing the staging unit; host-payload adopt through the kind's `adopt` in the reading step; `Payload.tentative` with per-request `keep_tentative`; `dropped_publishes`; `SeqView.branch_at` | tentative decode payloads: a preempted request resumes from the last; finish leaves k; the depth-1 extra row's payload registers only where committed tokens cover it; a staging unit reused while its D2H is delayed is not overwritten early | 60 |
| KS4 | **Slot `tier`, attach, identity, restore** | None (release §15.1) → slot and protocol types; `attach` after the final materialize (host pools allowed then, before graph capture); `plans.state_identity`; `Option.planned` honoured by `plan_of`; restore inserts non-resident entries parents first; `flush_prefix` → `clear`; `close` | restart a toy engine with a persistent toy tier: resume bitwise; another identity is cleared; `planned = false` options leave `plan_id` unchanged | 50 |
| KS5 | **Slot `eviction_policy`** | LRU only → eligible candidates lazily, LRU first, to `victim`; default the first | a toy policy pinning depth ≤ 1 keeps the root under pressure; property test with it on | 20 |

≈ 285 lines; the budget is ≤ +300 over K's step-1 base (rev 1: 480). Step 1 first prototypes KS1 and KS2 on CPU
against `core/tests/test_prefix.py` and prices them. If the prototype exceeds the itemized lines by more than 20%, it
goes to the owner before the rest is built (B2).

Not core: levels, storage, copies (`state_copy@1`), I/O threads, the clock and its cost model, the manifest,
options, checkpoint policies, tune's measurements, and the traces. The deferred CPU stream for the logic plants and
toy proofs (A11) lives in `testing/holon_testing`, as a fixture replacing the core platform's CPU stream and event.
It adds no core lines.

Seam names, from X's outline (`docs/v1-offload.md:599-600`):
- `Payload.tier` is dropped: nothing would read it (B4); the core's `Entry` keeps residency privately;
- "demote" is KS1;
- "cancellable `BLOCKED(future)`" is KS2's load gate with a tier-owned `Load`; loads are never cancelled (D7);
- the `on_finish` publish is KS3's store at registration;
- `HitLen.cost` is dropped (D9);
- the slot name is `tier`.

## 5. std library and plugins

- **`std.state_tier`** (`std/holon_std/state_tier/`, slot `tier`).
  - Levels:
    - host: S2 host pools per device unit shape, payload pools in the payload's own shape, unit-major KV pools when
      an SSD level is configured (D10);
    - SSD: optional.
  - Options:
    - `host.bytes` (default: what `host_memory` has left; `planned = false`);
    - `host.admit` = `publish` | `hit:<n>` | `evict`;
    - `host.reserve`;
    - `ssd.dir`, `ssd.bytes`, `ssd.io_threads` (all `planned = false`);
    - `ssd.direct`;
    - `ssd.persist` = `off` | `close`;
    - the clock: `step_cost` (`a,b,c`), `h2d_rate`, `read_rate`, `write_rate`, `stall_bytes` (tune).
  - Leaf-first LRU per level, with copies redundant with a faster level dropped first (A9), and payloads ordered by
    store or adopt recency (D8).
  - The determinism contract (§4), the copier (D10), the manifest (D11).
  - The in-flight floor check at attach (acceptance 4).
  - ≈ 550 lines.
- **`std.state.recurrent`.**
  - `recurrent.policy` becomes a comma list of built-in names or `Ref`s (structural `CheckpointPolicy`).
  - New built-ins `branch` and `answer:<k>` (which sets `keep_tentative`).
  - The pool becomes staging when a tier is bound (§7's formula sets its size).
  - ≈ 70 lines.
- **`kernels.expert_io`**: `state_copy@1` providers (batched DMA and SM gather over chunk address lists), ≈ 80 lines
  with the kernel's existing code.
- **`evict.root_pin`** (`plugins/holon_plugins/evict/root_pin/`, slot `eviction_policy`): skips candidates with
  `depth ≤ d` and `hits ≥ h` while others exist. ≈ 40 lines.
- **Held-out (after the tag):** `policies.last_turn`, `state_tiers.zlib` (§1).
- **Bench.**
  - `bench/workloads.py` gains `agent_park` (§8).
  - `bench/tier_resume.py` (3a, 3b) parks a set of conversations once with a filler and resumes each.
  - The demo runner records `--raw` transcripts locally for K-M1; they are never committed.
- **Judge.** `holon_judge tiers`: tier invariance with C2's reference, the null test, run-identity with injected
  delays, and the plants of §6, with fired and changed counts.

## 6. Numerics and judge

| Configuration | Class | Check |
|---|---|---|
| Any level, capacity and admission; restore after restart | exact (tier-invariant) | `holon_judge tiers`. Serial `after`-chained trace (8 sessions × 4 turns, history resent): the tested run interleaves with a filler forcing every return through the tested level; the reference runs each session's turns back to back with no filler. Streams and per-request `cached_tokens` byte-identical, ≥ 8 loads per level counted. Concurrent variant: run-identical step lists and outputs × 3 with ≥ 32 loads per level and unseeded delays. CPU tiny model; Qwen3-8B on the A5000; the 27B pair on the RTX 5090 |
| Null test | exact | tiers off, checkpoint pool 4 vs 16 units, back-to-back trace: byte-identical |
| `branch` (in-step, prefill-produced) | as `prompt_end`: bitwise the recomputed state at the same chunk boundaries (a CPU test with equal chunking); bounded across chunkings and vs recompute (L1.5) | the bounded multi-turn check (`docs/design.md:1297-1303`) with a shared-system-prompt session set added |
| `answer:<k>` (decode-produced) | bounded | the same bounded multi-turn check and floor (it resends history, generated tokens included; `bounded.py` has one floor, so no separate one) |

Snapshot traces of existing locks are untouched with tiers off. Tier comparisons compare output streams and
`cached_tokens`, not step lists, because gating adds or moves steps. Run-identity compares step lists too, because
batch composition is what nondeterminism would change.

**Plants** (`judge/holon_judge/planted.py`; C10). Each reports how often it **fired** and how often it **changed an
outcome**; a plant that fired but changed nothing fails its own check. Logic plants run on CPU with the toy tier or
the tiny model and the deferred stream (random deferral up to 2 steps). Stream-timing plants run on the A5000 with
Qwen3-8B, with unseeded delays up to 2 steps on the tier's streams, and in a pressure configuration where eviction
follows publish within 1–2 steps. Tiers-off configurations stay clean.

| Plant | Where | Breaks | Predicted |
|---|---|---|---|
| `promote_unwaited` | CPU + GPU | a launch does not wait on its rows' loads | tier invariance fails (stale bytes) |
| `store_early` (A2) | GPU | a paged store is issued after the committed step, not the last launched | host copies stale; tier invariance fails on a trace where the in-flight step completes published blocks |
| `store_unwaited` | CPU + GPU | reuse of a unit freed while its store reads it does not wait | host copy corrupt; tier invariance fails under pressure |
| `load_no_after` (C10) | GPU | a promotion's copy ignores `after` (the finished request's extra row overwrites it) | tier invariance fails |
| `read_unwaited` (C10) | CPU | an H2D is issued before its SSD read ticket completes | tier invariance fails (SSD trace) |
| `promote_wrong_unit` | CPU | a promotion copies the neighbouring host unit | tier invariance fails; bounded L2 fails |
| `swap_onto_lower` (A1) | CPU | publish onto a non-resident entry keeps the request's own bytes | tier invariance fails on the serial trace (turn 2's recomputed answer) |
| `gate_on_landing` | CPU | a gated request runs at the first step after its copy lands instead of at its due time | run-identical step lists fail with deferral |
| `free_on_ticket` (A5, C5) | CPU | the SSD level frees host units when the write ticket completes | run-identical fails with I/O delays |
| `promote_preempts` (A3) | CPU | promotion may preempt lower requests | the liveness test reports preemptions with no compute shortfall |
| `kids_any_level` | CPU | VRAM eligibility counts non-resident children | the property test fails (`could_fit` yes, eviction short) |
| `interior_drop` | CPU (std) | the host level evicts interior copies first | std's property test fails (subtrees lost); outputs unchanged |
| `branch_off_by_one` (C10) | CPU | the `branch` payload is filed one block early | bounded check fails (KL) |
| `tentative_shift` | CPU | an `answer` payload is filed one block early | bounded check fails |
| `torn_manifest` (C4) | CPU | the manifest is renamed into place before the data `fdatasync`, with the data write killed | restore discards by crc: `cached_tokens` short of the no-restart engine; with the crc check also disabled, outputs differ |
| `stale_slot` (A7) | CPU | the manifest is not unlinked at startup; a slot is rewritten; kill; restart | crc mismatch discards: `cached_tokens` short; with the crc check disabled, outputs differ |

`could_fit` and liveness bugs show as lost progress, not wrong output. The property test and
`test_state_liveness.py` catch them, not the snapshot.

## 7. Memory fit (estimates; step 0 and tune measure)

| Model, file, card | VRAM | Host level | SSD level |
|---|---|---|---|
| Qwen3.8-27B UD-IQ3_S, q8_0 KV, `max_seqs` 4, RTX 5080 16 GB | 10.9 GiB weights; 58K KV tokens ≈ 1.9 GiB; checkpoint staging 5–6 × 147 MiB (formula below; today 4) | KV 34,816 B per token; a payload 154 MB. A parked 16K conversation with its live payloads (≈ 3) ≈ 1.0 GB; `agent_park` parks ≈ 140K tokens (C11's sim, σ = 0.6) ≈ 4.9 GB of KV + ≈ 5.5 GB of payloads | 100 GB ≈ 95 parked 16K conversations |
| Qwen3.8-27B UD-Q4_K_M, RTX 5090 32 GB | 15.3 GiB weights; roomy for 3h | same unit sizes | same |
| Qwen3-8B Q4_K_M, bf16 KV, RTX 5080 | KV 144 KiB per token | 16 GiB ≈ 116K tokens | — |
| Qwen3-30B-A3B Q4_K_M offload + host level, RTX 5080 (report) | X's natural fit | experts ≈ 18 GB of `host_memory` (`docs/v1-offload.md:514`); the tier takes what is left | — |

Notes:
- **Staging formula** (D7): per payload spec, `units ≥ max_seqs + p` with overlap (`p` = most prefill rows in two
  consecutive steps), `≥ max_seqs` at depth 0, and never-dropping at `2 × max_seqs`. For the 27B lock: 4 + 1–2 = 5–6
  units, +147–294 MiB of VRAM over today (≈ 4.4–8.8K KV tokens). Tune measures 5 against 6 with `dropped_publishes`
  required 0, and K-M5 reports the peak on the trace.
- **Host demand of tentatives**: at most `keep_tentative × max_seqs` payloads in flight (8 × 154 MB on the 27B).
- **Host RAM per desktop** is checked before scheduling and is not recorded here.
- **Pinning time.** Page-locking took ≈ 0.33 s per GiB on the A5000 host (`platform/__init__.py:134-136`), so a
  16 GiB level adds ≈ 5 s to startup. Re-measured on the bar desktops.

## 8. Step 0 — measure before freezing, and the predictions

On the RTX 5080 and 5090 desktops, one heavy job at a time under `MemoryMax`, with results in `evidence/`. K-M1's
analysis and K-M5 are CPU-only.

| # | Measure | Decides |
|---|---|---|
| K-M1 | Re-render divergence. The pi agent test re-run once with the capture proxy's `--raw` (local only): per turn, the longest common prefix of (previous prompt + generated tokens) and the next prompt, in tokens and relative to the answer's end; also with a template that strips reasoning | `answer`'s k; 3g's prediction; whether `answer` is on by default |
| K-M2 | Copy paths on S2 pools' own allocation (mmap + `cudaHostRegister`, X's C-r4.5-5) at KV chunk sizes (34,816 B, 65,536 B), GDN conv segments (61,440 B) and checkpoint segments (≈ 3 MiB), H2D and D2H, fragmented runs: the per-run loop, `cudaMemcpyBatchAsync`, the SM gather; host time per call. In the same script (K-M3 folded in, C17): four decode streams of the 27B with 0.5–2 GB H2D loads and 154 MB D2H stores in flight | `state_copy@1`'s path; `h2d_rate`; `stall_bytes`; 3b's prediction; whether stores need a rate cap |
| K-M4 | NVMe `pwritev` / `preadv` of unit-major units (557,056 B) and checkpoints, O_DIRECT and buffered, 1–4 threads, reads and **sustained writes** past the SLC cache, with the engine loop's step time measured beside them (GIL) | `io_threads`; `read_rate`, `write_rate`; the close-flush estimate; D9's SSD rate |
| K-M5 | CPU replay of the **real core index code** with a toy tier on `agent_park`, `claude_code` and `agent`: hit tokens per level, load bytes, evictions, staging peak, per-step host time vs VRAM / RAM / SSD sizes, with step times from the locks | 3c–3e predictions; default `host.bytes`; the staging size; C13's scan cost |

**`agent_park`** (`bench/workloads.py`, seed 0). Four clients run closed-loop. Each owns three sessions and serves them
round-robin, one request at a time.
- The 12 sessions share a 4,096-token system prompt. Each opens with a 512-token task.
- Then 10–14 turns per session, each adding:
  - a tool result: log-normal, median 384, σ = 0.6, capped at 2K;
  - an output of reasoning plus a tool call: log-normal, median 300, σ = 0.6, capped at 2K, released to the next
    prompt verbatim;
  - a gap of 0.5–3 s.
- Contexts end at ≈ 13–19K tokens; ≈ 140K tokens are parked at the end. C11's sim on the RTX 5080's 58K VRAM tokens
  keeps ≥ 90% of a returning session's KV resident through turn 5 and 35–50% after. So KV demotion is exercised mostly
  in the second half, which is why 3c reports turns 1–5 and 6+ apart and why 3h exists.

**Predictions** (RTX 5080, 27B UD-IQ3_S; 47.5 GB/s H2D; prefill 2.2–2.5K tok/s; decode step ≈ 15.7 ms, tg128
63.75 tok/s from the same evidence file).

| Row | Prediction |
|---|---|
| 3a, 16K from RAM | KV 570 MB ≈ 12.0 ms of copy, and a payload adopt of 154 MB ≈ 3.2 ms in both arms. Bar: VRAM + 18.0 ms + one step (33.7 ms). Predicted extra ≈ 12–28 ms if `state_copy@1` reaches ≥ 0.8× of 47.5 GB/s on fragmented units; fails with the per-run loop |
| 3a, 32K from SSD | 1.29 GB at ≈ 2.6 GB/s ≈ 0.50 s + 27 ms H2D. With the default watermark no write-back is forced; a forced one adds 1.29 GB at the sustained write rate (K-M4). Bar ≈ 0.79 s + one step |
| 3b | copy-engine DMA plus ≈ 5% of DRAM bandwidth during the copy (47.5 of 960 GB/s); the 32K load is due ≈ 2 steps out, with the append step in both arms → median gap ≤ 1.05×, p90 ≤ 1.15× (bar 1.1× / 1.2×) |
| 3c | VRAM-only: the 4-unit pool keeps no returning session's payloads, so every hit trims to 0, and each turn recomputes its context (≈ 8K mean, ≈ 3.5 s). Tiered: ≈ 300 MB loaded (≈ 6 ms) plus the append: ≈ 400 tokens with `answer` (≈ 0.18 s), ≈ 700 without (≈ 0.3 s) → TTFT p50 ≈ 0.2–0.35 s, ratio ≈ 0.05–0.1× (bar 0.25×). Throughput gain 1.3–2× from removed prefill (bar 1.2×) |
| 3d | `llama-server`'s save is synchronous (an observed 2.65 s stall), and whole-state blobs repeat the system prompt per conversation. Prefill parity at depth (Holon 1.05× the gate build at 8K, 1.01× upstream, same evidence) → ≤ 1.0× predicted. No margin estimate until step 4; the bar is fixed now |
| 3e | sessions 1–2 recompute 16K (≈ 7 s); sessions 3–16 hit 16,384 → first-turn TTFT ≈ 0.1–0.2 s → p50 ratio ≈ 0.03× (bar 0.2×) |
| 3f | 16K resume as 3a's SSD cell; restore verification ≈ persisted bytes / 2.6 GB/s (≈ 6 s for 16 GB) |
| 3g | 44% (27,646 / 62,682) of uncached input → ≤ 10% if K-M1 finds the answers re-rendered verbatim up to their tool calls |
| 3h | loads ≈ 6 ms against one ≈ 15.7 ms step of granularity → ≤ 1.05× roomy + one step (bar 1.1× + one step) |

**Decision rules** (fixed now):
- A predicted row that step 4 measures below its bar goes to the owner, with the measured cause, before any
  retuning.
- `answer` is on by default only if K-M1 shows ≥ 50% of answers re-rendered verbatim to within one block of a kept
  checkpoint.
- `state_copy@1` takes the faster of DMA and the SM gather. If neither reaches 0.8× of the blob rate on fragmented
  34 KiB chunks, the owner decides before step 2.
- Write-through stays the default unless K-M2 shows stores cost > 2% of decode.

## 9. Steps (milestone K)

K's core round is serialized with FN's (D14): step 1 rebases onto FN's seams if they merge first, and its checks are
relative to that base.

| Step | Work | Checks | Days |
|---|---|---|---|
| 0 | K-M1, K-M2 (with K-M3), K-M4, K-M5; the `agent_park` generator | evidence; the panel reviews the parameters they set | 2 |
| 1 | CPU prototype of KS1–KS2 and its pricing; then KS1–KS5 with toy proofs; the property test; the deferred CPU stream fixture; the design.md edits (§11) | core tests; `check_abi --base <K's base>` additive; ≤ +300 lines over K's base; denylist; tiers-off snapshots unchanged on the A5000 | 6 |
| 2 | `std.state_tier` host level and copier; `state_copy@1` providers; `branch`, `answer`; `holon_judge tiers`; the CPU logic plants | tier invariance (CPU, Qwen3-8B on the A5000); null test; run-identical × 3 with delays; bounded multi-turn for the policies; CPU plants and the A5000 stream-timing plants | 5 |
| 3 | SSD level: write-back, watermark, persistence and restore; `evict.root_pin`; **tag the API** | tier invariance with SSD; 3f's two-engine restart test; `torn_manifest`, `stale_slot`, `read_unwaited`, `free_on_ticket` | 4 |
| 4 | The gate on the RTX 5080 and 5090: 3a–3i, the 27B tier-invariance pair on the 5090; tune sets the clock, rates, `host.bytes` and the staging size; the 27B agent locks | acceptance 3; `dropped_publishes` 0; the locks' bounded and run-identical verdicts | 4 |
| 5 | Held-out plugins after the tag: `policies.last_turn`, `state_tiers.zlib` | plugin-only diffs (or the recorded proof finding and its minor); tier invariance for the zlib tier | 3 |
| 6 | Close: full suite, `run_all.sh`, panel code review; lock propagation if K closes second (D14) | acceptance 4–6 | 2 |

26 working days (rev 1: 22): +1 for the prototype, +1 for the copier and plants, +1 for the held-out tier, +1 for the
gate's added rows. Waiting for FN's core round, if it holds the base, is outside this count.

## 10. Risks and deferred

**Risks.**
- **The copy path is the critical path to 3a and 3b.** If neither DMA batching nor the SM gather reaches 0.8× on
  fragmented chunks, loads cost whole steps, and 3b misses for 32K resumes. Hence K-M2 runs first.
- **The core estimate.** B2's shrink rests on ≈ 285 lines. Step 1 prices it on a prototype before building the rest.
- **Re-rendering decides `answer`'s worth.** If agents' templates re-serialize tool calls or strip reasoning, the
  answer checkpoint helps little. `prompt_end` and `branch` still carry the gain (K-M1, 3g).
- **The held-out policy is predicted to need a core field.** That would be a recorded proof finding and a minor, not
  a silent fix.
- **Host RAM and the link are shared with experts.** On offload deployments the tier gets what X leaves, and loads
  share the H2D link with X's fetches; `h2d_rate` is capped by the lock there (3i).
- **SSD wear.** Write-back only, never through, keeps writes to what falls out of RAM. `persist = close` writes the
  rest once per clean shutdown.
- **Host CPU in the engine loop.** The resident LRU keeps VRAM scans at today's size. Lower-level scans are the tier's.
  Without a tier, `evict_payloads`' entry-list copy (C13) is unchanged; with one, staging units free at registration,
  so it rarely runs. K-M5 replays the real index code to measure both.

**Deferred.**
- `HitLen.cost` (D9);
- layer-wise loading and per-layer events (D12);
- K4 (FN, GLM; D15);
- ARC as an `eviction_policy` plugin;
- `prefill:<n>` tentative in-step checkpoints (B13);
- a per-pool payload LRU for the tiers-off path (C13);
- a save / restore API;
- checkpoint codecs;
- MTP verify-row checkpoints;
- Maya's multi-session ratio (GLM);
- VMM rebalance (plan §10).

## 11. design.md changes (made in step 1 with the seams; B8)

- §4.2: add `tier` and `eviction_policy` to the slot table (single, opt-in). Remove `tier` from "Not re-added
  (v1 offload)" (`docs/design.md:223`), and record that `tier` replaces the planned `connector` for prefix-index
  offload.
- §4.5: the ABI minor (1.4 or 1.5 by merge order) and its contents.
- §6.2: `hit_length`'s "a tier cost with offload" comment (`:847`) becomes "no cost (K D9)".
- §6.3:
  - "A payload's tier is planned" (`:867`) becomes the core's private residency;
  - "what to evict are plugins" (`:883`) and the eviction-policy contract (`:894`) become KS5's `victim`; demotion is
    the core's, by the lower copy;
  - add `Payload.tentative`, `branch_at` and the staging rule.
- §6.4 (`:918-923`): the tier, not a per-request choice, owns loads. Loads land in index-owned, pinned units and are
  never cancelled. BLOCKED stays for logits processors only.
- §9's KV offload row (`:1338`): `std.state_tier`, slot `tier`, KS1–KS5.
- Decision D8's row: K's seams as built.

## 12. Lead's decisions on rev 1's open questions (2026-10-18)

The owner directed "continue until FN and GLM are finished", so these were decided within the milestone and
recorded. Rev 2 keeps them, with the changes noted.
1. **Write-through by default** (D5): kept. Inclusion is no longer a startup rule (B2 makes it unnecessary).
2. **No `HitLen.cost`**: kept. The rate refusal moves from the core to tune (B11).
3. **`branch` and `answer:<k>` on in the 27B agent locks**: only if the gate shows their gain; bounded approval.
4. **Gate against `llama-server`**, with `--cache-ram` sized like Holon's host level as the gating arm (C15);
   SGLang HiCache and vLLM's connector are report-only on Qwen3-8B.
5. **Maya's multi-session ratio moves to GLM**: kept.
6. **SSD persistence identity**: `plans.state_identity`, the plan without unplanned options and without the clamped
   `max_model_len` (A13, B6). It is still strict: any code change invalidates the files.
7. **`agent_park` as the gate trace, plus one pi re-run with local raw capture**: kept, with σ fixed (C11).

## 13. Review log

### Rev 1 (2026-10-18): draft

How the requirements X's panel left for K (`docs/v1-offload.md` §11) are answered (updated for rev 2):

| Requirement | Answered in |
|---|---|
| A10: demoted children must not keep a VRAM parent unevictable while `could_fit` counts it freeable; VRAM entries prefix-closed; leaf-first lower eviction; a property test that `could_fit` and evict agree | D6 (resident-child counts, resident LRU), KS1, the property test, plants `kids_any_level` and `interior_drop` (std) |
| Write-safe cancellation | D7: loads land in index-owned, pinned units and are never cancelled; frees wait on stores; plants `store_unwaited`, `promote_unwaited` |
| C9b: deterministic resume | D4's virtual clock and KS2's gate; plant `gate_on_landing` |
| C14: acceptance that bites | 3a–3i with link and device rates, fragmented units, and 3b's gap metric with a control arm |
| Core seam: `Payload.tier`, `HitLen.cost`, demote, cancellable `BLOCKED(future)`, an `on_finish` publish; name `tier` | §4's name mapping |
| Plan §10's end-of-answer checkpoint | D8 `answer:<k>` with KS3, K-M1 |

### Rev 2 (2026-10-18): the challenge panel

Three reviewers (A correctness, 1 blocker + 6 major + 6 minor; B architecture, 6 major + 8 minor; C numerics and
performance, 1 blocker + 11 major + 9 minor). Each finding was checked against the code at this branch's base:
- A2's in-flight publish: `engine.py:454-463, 507`, `state.py:335`;
- A1's swap: `state.py:338-345`;
- A6's progress rule: `sched.py:133-136`, `engine.py:335-343, 451-453`;
- C1's silent drop: `kind.py:73`;
- B6's planned options: `plans.py:80-104`;
- A13's clamp: `engine.py:262`;
- C8's layer-major pools: `units.py:136-141`;
- C11's trace shape: its sim, re-run at σ 0.6 and 0.9 (138K and 162K tokens parked; ≥ 90% resident through turn 5).

None was rejected outright. Three were taken in a changed form, with reasons below.

| Finding | Disposition | Where / why |
|---|---|---|
| A1 (blocker) publish onto a demoted or loading entry | accepted | D7 promote-into-place; the per-step wait rule over rows' wait sets; plant `swap_onto_lower`; KS2 toy proof |
| A2 stores before the in-flight step writes | accepted, changed for payloads | D5: paged stores `after` = last launched. Payload stores use their producing step, the only writer of a staging unit: using the last launched step would hold staging units one step longer for nothing. GPU plant `store_early` |
| A3 promotion has no failure path; eviction before incref | accepted | D7 admission order (prefix first, pure-free evictions, never preempt, trim and re-run the fixed point); plant `promote_preempts` |
| A4 the 4-unit checkpoint pool cannot carry K's demand | accepted | D7 staging pool, stores at registration, host → slot adopt; §7 formula; `dropped_publishes` = 0 |
| A5 write-back makes free counts timing-dependent; hit during write-back | accepted | D4 logical/physical rule; D11 (an entry being written back is logically on SSD); plant `free_on_ticket` |
| A6 BLOCKED overloading breaks progress; preempt leaves BLOCKED; deadlock | accepted | BLOCKED not used; the scheduler's load gate with idle pull-forward; preempt and abort clear gates; loads pinned and never cancelled (D7); KS2 proof for A6a |
| A7 the manifest can restore wrong bytes | accepted, changed | D11: the manifest is a snapshot written at clean close and unlinked at startup before any slot rewrite (tombstones for every slot at once), plus a per-record data crc verified at restore, instead of per-reuse tombstones or generations; plants `torn_manifest`, `stale_slot` |
| A8 host tier sizing before device capacities are known | accepted | KS4 `attach` after the final materialize; host pools allowed then |
| A9 lower copies of entries back in a faster level never evicted | accepted | D11 / §5: copies redundant with a faster level are dropped first |
| A10 block-contiguous fallback breaks the MemOp shape rule | accepted | D10: dropped; unit-major host layout only where the tier's own copies serve it |
| A11 CPU toys cannot test stream ordering | accepted | deferred CPU stream fixture in `testing/holon_testing` (no core lines); logic plants use it |
| A12 tentative edge cases (a, b, c) | accepted | D8: (a) registers as ordinary; (b) detach always; (c) moot, because `prefill:<n>` is deferred |
| A13 `plan_id` unstable for persistence | accepted | D11 `plans.state_identity` without the clamped length or unplanned options; persistence fails at startup if the plan cannot be computed |
| B1 eviction order hard-coded in the core | accepted | KS5 slot `eviction_policy` (`victim` over eligible candidates, lazily); `evict.root_pin` ships; lower-level victims are the tier's |
| B2 the core takes the whole hierarchy | accepted | D3: the core sees only the VRAM boundary; levels, copies, clock and manifest are std's; ≈ 285 lines, priced by a step-1 prototype |
| B3 held-out proof does not bite | accepted | §1: hold out `policies.last_turn` (predicted to need a field) and `state_tiers.zlib` (exclusive, variable-size, CPU stage); selective and buffered become options |
| B4 `Payload.tier` has no reader | accepted | dropped; residency is the core's private `Entry` state |
| B5 the protocol cannot express D11 | accepted | §4: `BlockRecord`, `TierCtx`, the full `StateTier`; the tier owns its host pools; inclusion is no longer needed |
| B6 SSD options leak host paths into plans | accepted | `Option.planned = false` (KS4); identity on planned settings only |
| B7 slot selection and plan effects | accepted | D3: `slots["tier"]`, opt-in like the logits processor; no `EngineConfig` field, so no `plan_id` move with tiers off; `tier` replaces the planned `connector` for this use |
| B8 design.md edits not listed | accepted | §11 |
| B9 relation to X's tiers; colliding names; disallowed reuse | accepted | D3: `state_tier` / `state_tiers.*`; no import from `std.tiered` |
| B10 KS3 surface | accepted | `Payload.tentative` replaces the per-publish hook; §5's admit sentence removed |
| B11 the core rate floor has no inputs and mismatched units | accepted | D9: tune's check, in tokens per second |
| B12 the fallback layout is unpriced | accepted | D10 (as A10) |
| B13 `prefill:<n>` on without evidence | accepted | deferred |
| B14 K4 compatibility | accepted | D15 |
| C1 (blocker) the staging pool and silent drops | accepted | as A4; the formula `max_seqs + p` (5–6 units on the 27B) and the counter. The lead's "VRAM pool staging only" fits in today's size class because stores free units at registration, not at release |
| C2 tier-invariance reference infeasible on the 5080 | accepted | §1, §6: back-to-back reference, 27B pair on the 5090, tiers-off null test |
| C3 3f's reference | accepted | 3f: two engines with the same parking history |
| C4 SSD restores wrong bytes; `torn_manifest` mispredicted | accepted | as A7; plants re-defined with a crc-disabled variant |
| C5 SSD frees break determinism | accepted | as A5 |
| C6 bars graded against the path's own rate; fragmentation | accepted | acceptance 3: link and device rates; forced fragmentation; a cell with decoders active |
| C7 3b's metric undefined and confounded | accepted | per-step gaps vs a VRAM control arm with the same append; 30 reps, ≈ 360 samples per arm; median and p90 bars |
| C8 the fallback does not address the risk | accepted | D10: SM gather as the alternative path; unit-major host layout for SSD only |
| C9 decode-step resume penalises prefill-heavy steps | accepted | D4: virtual clock with a per-step cost model; lanes chained |
| C10 plants would not catch their bugs | accepted | §6: fired and changed counts, deferral and pressure configs, CPU vs GPU split, the missing plants added |
| C11 3c bar loose; baseline mostly a starved pool | accepted | 3c bar 0.25×, turns reported apart; 3h tier-overhead row; σ fixed and the parked total corrected to ≈ 140K |
| C12 write-back ignored on the SSD path | accepted | D4 write lane in the due time; default watermark = one max load; 3b-SSD row |
| C13 `evict_payloads` is O(N) | accepted in part | K-M5 replays the real index code; with a tier the scan rarely runs. A per-pool payload LRU for the tiers-off path is deferred: tiers-off behaviour must stay byte-identical, and it is not K's regression |
| C14 bitwise claim needs equal chunking | accepted | §1 and §6 qualified; a CPU equal-chunking test; one bounded floor |
| C15 3d's reference and the 0.87× claim | accepted | the sized `--cache-ram` arm gates; the build is pinned; the depth-8K parity is used |
| C16 3e arms and trace | accepted | tiers on in both arms; 2 turns per session |
| C17 step-0 scope | accepted | K-M3 folded into K-M2; S2 allocation and conv segments; sustained writes |
| C18 parking by filler is expensive; few reps | accepted | conversations parked once per process; ≥ 10 reps per cell |
| C19 arithmetic | accepted | 570 MB per 16K; D's granularity from the clock; payload SSD ≈ 59 ms and 130 tokens; 3c with and without `answer` |
| C20 offload coexistence | accepted | 3i report row; `h2d_rate` capped on offload locks |
| C21 promotion short of VRAM | accepted | as A3 (trim, re-run the fixed point) |

**Coordinator's additions.** The core budget is stated relative to K's step-1 base (≤ +300). The ABI is 1.4 or 1.5
by merge order with FN. `check_abi --base` and the line check are taken against that base. Only the track that
closes second propagates the locks (D14).
