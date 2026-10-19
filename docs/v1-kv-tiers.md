# v1 — milestone K: KV and checkpoint tiers under the prefix index

Status: **rev 8.5, step-3 panel review** (2026-10-19): rev 8 (panel-clean), the lead's decisions on step 0's measurements
(§13 "Rev 8.1"), the panel's reviews of step 1 (§13 "Rev 8.2"), step 2 (§13 "Rev 8.3", "Rev 8.4") and step 3 (§13
"Rev 8.5"); steps 1–3 are built (§13 "Step 1 as built", "Step 2 as built", "Step 3 as built"). Every other number
below is a CPU-side estimate from earlier evidence (cited), fixed as a prediction before step 4 measures it. Rev 2 answered the three-reviewer panel (A correctness, B architecture, C numerics and
performance); revs 3 to 8 answer its re-reviews. The finding-by-finding dispositions are in §13.

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
copies, the write-back, the virtual clock and the manifest all move to std (`std.state_tier`). The core is ≈ 300
lines instead of 480, with no new device stream, no batched-copy call and no `Payload.tier` (B4). Every tier state
change is logical at decision time, and physical reuse waits on copies and tickets (A5, C5). Resume is gated by a
virtual clock with a per-step cost model (C9), not by a fixed step or by `BLOCKED` (A6). The VRAM checkpoint pool
becomes staging only, so payloads live in the tier from registration and adopt straight from host (A4, C1). A publish
that meets a demoted entry promotes it into the request's own units (A1). Victim choice in VRAM is a plugin slot (B1).
The held-out proof holds out a checkpoint policy and a tier the protocol was not shaped around (B3).

**What changed from rev 2.**
- **Fencing.** Every host unit a step reads is fenced by that step's end event, passed in `loaded(load, done)`. All
  tier reuse and I/O on such a unit waits for it.
- **Progress.** Rev 3 host-waited load pins that had no waiter. Rev 5 removes load pins altogether (D6, D7).
- **Admission.** Payload stores are mandatory, and admission options govern pages only.
- **std.** It gains a structural `Level` protocol below its host level. SSD is its first instance, and the held-out
  zlib store is a `Level`.
- **Plans.** `host.bytes` stays planned.
- **Clock.** The clock never runs behind a load a launch waited for.
- **Bars and checks.** 3b, 3h, the null test and the bounded check are tightened as the panel showed.

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
     - the tier `std.state_tier` (slot `tier`): a pinned-RAM host level, with page store admission as an option
       (`publish`, the default; `hit:<n>`, HiCache's selective mode; `evict`, write-back). Below it sit `Level`s
       bound by `Ref` in `state_tier.levels` (D3). The first `Level` is `std.state_tier`'s SSD level, with O_DIRECT
       as an option (Maya KG3's buffered files are `direct = false`);
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
     - `state_tiers.zlib`: a `Level` the protocol was not shaped around. It keeps lossless zlib blobs of variable
       size in plain memory instead of fixed slots, and compresses and decompresses on a CPU thread between its
       blobs and the host level's units.
       - It takes pages and payloads alike when the host level evicts them.
       - It reports its CPU rates for the lanes.
       - Its capacity counts uncompressed units, so its refusals are decided before any compression and stay
         deterministic. Compression only saves memory.
       - It must pass tier invariance like the SSD level.
       - Expected diff: ≈ 120 lines plus tests, in its own directory, binding through `state_tier.levels`.
     - Rev 8.5: `state_tiers.zlib` landed after the first tag (af0672d) and passes tier invariance; its `write` took
       the `slots` argument when rev 8.5 moved the tag. `policies.last_turn` is still owed (step 5).
   - Every core seam (§4) is proven first with a toy kind and a toy tier in `core/tests`.
2. **Exactness** (D4, §6).
   - **Tier invariance.** The reference is the same configuration with each session's turns run back to back and no
     filler, so nothing is demoted. The tested run interleaves the sessions with a filler that forces every return
     through the tested level. The output streams and each request's `cached_tokens` are byte-identical (panel C2).
     - It holds for host only, host + SSD, and SSD restored after a restart, under page admission `publish` or
       `evict`.
     - Under `hit:<n>`, the filler's evictions drop unadmitted blocks that the reference keeps. The hit sets then
       differ by construction (A-4), so that configuration is judged by run-identity and `bounded` only.
     - The judge refuses to compare runs whose `cached_tokens` differ.
     - Hybrids with `prompt_end` or `answer` checkpoints through a level are judged by run-identity and `bounded`
       (rev 8.5): the host level drops tentative and superseded checkpoints rather than write them back (D11), so
       the interleaved run's hits differ from the reference's by construction. With `branch` (never superseded)
       they are judged by tier invariance.
     - Development: Qwen3-8B on the A5000.
     - 27B: the roomy-vs-tight pair runs on the RTX 5090, where the roomy run fits; the RTX 5080 runs the lock's trace
       for run-identity only.
     - A tiers-off null test shows that the checkpoint pool's size alone does not change output bytes: 4 vs 16 units,
       `prompt_end` only (so neither run loses a payload), with `cached_tokens` asserted equal (C-5).
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
   | 3b | Three sessions decoding while a fourth resumes 32K tokens from RAM. 30 reps (10 parked conversations × 3 rounds). Control arm: the same conversation resumed from VRAM. Gaps are taken over the **gated steps only**, from the resume's admission to the step before its append: ≈ 180–240 gaps per arm. The append step's gap, the same work in both arms, is reported apart as a ratio (C-3) | the three decoders' median gated-step gap ≤ 1.1× the control's, and p90 ≤ 1.2× |
   | 3b-SSD | as 3b, the resume from SSD (report) | reported; engine-loop host time per step during the read reported |
   | 3c | `agent_park` trace (§8): 12 sessions over 4 clients, parked and resumed. Compared with Holon VRAM-only (today). Turns 1–5 and 6+ reported apart | TTFT p50 ≤ 0.25×, output throughput ≥ 1.2× |
   | 3d | `agent_park` vs `llama-server` with `--cache-ram` sized like Holon's host level, plus `--ctx-checkpoints`; same file, card and KV format; the llama.cpp build pinned in the evidence (the default 8 GiB arm reported) | TTFT p50 and p90 ≤ 1.0×, output throughput ≥ 1.0× |
   | 3e | `claude_code` truncated to 2 turns per session (16K shared system prompt, `bench/README.md:58-60`): TTFT of each session's first request, `branch` on vs off, tiers on in both arms | p50 ≤ 0.2× |
   | 3f | Restart (`state_tier.ssd.persist = close`): two engines with the same parking history; one resumes a parked 16K conversation without a restart, the other closes, restarts and resumes it (C3) | TTFT within 3a's SSD bar. The restore's crc verification runs inside `attach` and is startup time, reported per persisted GB (C-9). Outputs and `cached_tokens` byte-identical between the two |
   | 3g | pi agent test re-run on the RTX 5090 (report, not gated: agent outputs are not reproducible) | the share of uncached input that is the previous turn's own output, predicted to fall from 44% to ≤ 10% if K-M1 shows verbatim re-rendering |
   | 3h | Tier overhead on the RTX 5090: `agent_park` with VRAM capped to the RTX 5080's KV capacity, against a roomy run (C11, C-7). The roomy arm runs tiers off, with ≈ 40 checkpoint units and ≥ 150K KV tokens: ≈ 15.3 + 4.9 + 5.7 ≈ 26 GiB plus graphs and activations, which fits in 32 GB. The ratio includes the per-hit host adopt (≈ 3.2 ms of H2D against ≈ 0.3 ms of D2D in the roomy arm) | TTFT p50 ≤ 1.1× roomy + one RTX 5090 decode step (14.7 ms, rev 8.2) |
   | 3i | Qwen3-30B-A3B offload with the host level on the RTX 5080: decode TPOT while loads run beside X's expert fetches (report, C20) | reported |

4. **Memory** (§7).
   - The host level is allocated exactly from S2 host pools, after the final materialize, and accounted against
     `host_memory` after X's pools.
   - Its size, `state_tier.host.bytes`, is planned and pinned by the lock, as X's pinned pools are (B-N3).
   - `attach` refuses with a per-pool report:
     - if `host_memory` cannot give that size;
     - if the size is below the in-flight bound: `max_seqs` × the units of one `max_model_len` load, plus room for
       the staging pool's payloads, so a payload store is never refused (D5).

     Inclusion is no longer required.
   - The SSD level refuses a directory without its bytes.
   - With tiers off, VRAM is unchanged. With tiers on, no new device pool is added, and the checkpoint pool is sized
     by §7's staging formula.
   - The `dropped_publishes` counter (`stats()`) is 0 on `agent_park` at the tuned pool size.
5. **Core.**
   - Only §4's seams KS1–KS5, one additive ABI minor, each toy-proven. K's ABI is **1.4 or 1.5 by merge order**:
     milestone FN also proposes 1.4. FN rev 3 (7a80674) has seams C1–C3, ≤ +47 lines over its own step-1 base, and
     no FFN branch. K's and FN's core changes are serialized. Whichever lands its seams first takes 1.4, and the
     other rebases and takes 1.5.
   - `check_abi --base` and the line check are relative to **K's step-1 base**: the main-branch commit K's seams
     rebase onto, after FN's seams if FN lands first.
   - **At most +320 code lines over that base** (§4 itemizes ≈ 292; rev 2 had ≤ +300 for ≈ 285). Rev 8.2: the lead
     raised it to +326 for the two frozen slot factories (the owner told); step 1 is at +326.
   - A property test shows that `could_fit` and eviction agree under random traffic with demotions, promotions, loads
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
| T1s | Selective page store (`state_tier.host.admit = hit:2`; payloads are always stored, D5) | HiCache `write_through_selective` | T (option) | off |
| T1b | Page write-back at VRAM eviction (`admit = evict`) | R §4.2 K2, LMCache | T (option) | off |
| T2 | SSD `Level` below RAM: slot files, O_DIRECT, written back when RAM evicts | HiCache L3, LMCache disk, Strata C3 | T (`Level`) | opt-in (`state_tier.levels = "ssd"`, `state_tier.ssd.dir`) |
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
| H2 | `state_tiers.zlib` | none (an unanticipated shape: variable-size blobs, a CPU stage) | `Level` plugin | held-out |
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
  - page store admission (payload stores are mandatory, D5).
- **A state kind owns** where checkpoints are taken (D8).
- **The core never names** a tier, a level, a format or a model (invariant 1, `scripts/ci/check_core_denylist.py:8-11`).

*Why one tier object and not N levels in the core.* Rev 1 put copies per level, per-level child counts, per-level
LRUs, the SSD watermark, write-back tickets and restore in the core: 250 lines of KS1–KS2 against a 137-line
`prefix.py`. The scheduler needs none of it. A VRAM eviction either finds a lower copy or drops the subtree, so lower
eviction never blocks device progress. A10's concern (demoted children pinning a VRAM parent) needs only resident-child
counts. N levels inside std change nothing the core decides.

*Levels below host are std's `Level`s, bound by `Ref`* (B-N2). A new medium or codec must not re-implement a whole
tier: the clock and its lanes, leaf-first LRU, the copier, admission and the determinism contract. So
`std.state_tier` keeps all of those, plus the host level, and binds the levels below host by `Ref` in
`state_tier.levels`, as X's `moe.tiers` binds its tiers (`docs/v1-offload.md:1365-1372`). The list holds built-in
names or `Ref`s, slowest last. A `Level` is structural, so plugins implement it without importing std:

```python
class Level(Protocol):     # std.state_tier's, below its host level (`std/holon_std/state_tier/level.py`; rev 8.5)
    def bind(self, ctx, shapes: Mapping[str, tuple[int, int, float]], identity: str | None) -> Mapping[str, int]:
        ...  # host pool -> (unit bytes, segments, share of the level's storage); returns its units per pool
    def write(self, key: int, spec: str | None, src, slots: Mapping[str, Sequence[int]]) -> None: ...   # blocking
    def read(self, key: int, spec: str | None, dst) -> None: ...                                        # blocking
    def drop(self, key: int, spec: str | None) -> None: ...
    def rates(self) -> Mapping[str, object]: ...    # read / write bytes/s, mixed and read_beside factors, a lane
    def restore(self) -> list[BlockRecord]: ...     # persistent levels, with each copy's slots; [] otherwise
    def close(self, records: Sequence[BlockRecord]) -> None: ...   # persistent levels write their manifest
```

The tier owns the I/O threads (`state_tier.io_threads`, default 3: **a level's concurrency is that thread count**),
so `write`, `read` and `drop` are blocking calls the tier makes on them, not tickets. The tier never runs two calls on
one copy at once, calls `drop` once none is queued or running, and **chooses the slots of every write** (indices
`0 … capacity − 1` per pool) when it decides it, so a write never waits for room (a level without slots, as zlib,
ignores them); a write into a slot waits, on its I/O thread, for the last job on the copy that held it. `shapes` gives
each host pool's share of the level's storage (the tier's split of `host.bytes`, so the zlib and SSD levels size
alike). `src` and `dst` map a pool to the copy's units of it (a block's pages span every paged spec). Restored copies
carry their slots (`units` per paged spec; `Payload(slots, pool)`). K binds one level below host.

- `std.state_tier` decides every placement, refusal and eviction on logical unit counts. A level only moves bytes
  and reports rates. The determinism contract is therefore kept in one place (§4).
- **Argument types** (B-1). Plugins cannot import std's names, so the protocol's docstring defines them structurally,
  using `holon.api` types and builtins only:
  - `ctx` is the `TierCtx` (§4);
  - the host units (`src`, `dst`) map a pool to a sequence with one entry per unit. Each entry is a sequence of that
    unit's writable segment buffers (`memoryview`s): an iovec for `preadv` / `pwritev`, valid until the call returns
    (round 4).
    - S2 payload host pools are segment-major: the 27B's GDN checkpoint unit is 48 segments, one per layer, so its
      entry has 48 views.
    - KV units in the unit-major layout of D10 have a one-element entry.
    - K-M4 checks O_DIRECT alignment for the payload segment sizes; a size that is not aligned goes through a
      bounce buffer or buffered I/O (D11);
  - the rates are a plain mapping: `{"read": bytes/s, "write": bytes/s, "mixed": factor, "read_beside": factor,
    "lane": str | None}`. `mixed` divides the write rate of a write a queued read depends on; `read_beside` multiplies
    the read rate of a read decided while background writes run (D4, rev 8.5); levels naming the same `lane` share one
    virtual lane.

*Why not X's `Tier`.* `std.tiered`'s `Tier` (`std/holon_std/tiered/protocols.py:46-74`) is a std protocol for
read-only, in-graph expert reads. The core cannot call a std protocol, and K's tier is called by the core between
steps, writes too, and holds index-shaped state. The families are therefore named apart: the slot is `tier`, std's
package is `holon_std/state_tier/` (added to the std independence list, `pyproject.toml:93-99`), and plugins are
`state_tiers.*`, beside X's `tiers.*`. Nothing is imported from `std.tiered`: `files.py`'s `_open` is private and
read-only, so the SSD level has its own few lines (B9).

*Selection.* `EngineConfig.slots["tier"] = "std.state_tier"`, opt-in like the logits processor
(`core/holon/core/engine.py:189-191`). There is no new `EngineConfig` field, so `plan_id`s do not move with tiers off,
and `--slots tier=…` selects the tier as its help says (B7). Its options are plugin options.
- Only `ssd.dir` and `io_threads` are declared `planned = false` (KS4, B6; rev 8.5: `ssd.bytes` decides the hit set,
  so it is planned, as `host.bytes` is). The manifest records the slot-file geometry, and restore discards slots out
  of range (A-6b).
- `host.bytes` decides the hit set, so it stays planned and the lock pins it (B-N3). The slot `tier` takes over design §4.2's planned `connector` slot
for prefix-index offload. A v2 PD transfer can implement the same protocol, or its own slot then.

Plugin kinds:

| Kind | Selected by | Contract | Instances in K |
|---|---|---|---|
| tier | `slots["tier"]` (single, opt-in) | `StateTier` (KS4, §4) | `std.state_tier` |
| level below host | `state_tier.levels` (built-in names or `Ref`s, slowest last) | std's structural `Level` (D3) | `ssd`; held-out `state_tiers.zlib` |
| VRAM victim policy | `slots["eviction_policy"]` (single, opt-in) | `EvictionPolicy.victim(candidates)` (KS5) | `evict.root_pin` |
| checkpoint policy | `recurrent.policy = "prompt_end,branch,answer:1"` (built-in names or `Ref`s) | std's structural `CheckpointPolicy.next(view) → (at, tentative) \| None` | prompt_end, every, branch, answer; held-out `policies.last_turn` |

**D4 — Exact means tier-invariant; decisions never read timing.**
- **Bytes.** Every move is a byte copy of whole units: device ↔ host by the tier's copies, host ↔ SSD by the tier's
  I/O, and a host payload into a slot by the kind's own `adopt` MemOp. A request computes on the same bytes whichever
  level held them, so tier invariance is a property of the copies and of the hit set. No kernel changes.
- **Logical at decision time, physical after the ticket** (A5, C5). Which entries are resident, demoted, promoted,
  dropped or hit, which units are free, and when a request may run are all functions of the calls made so far and the
  steps scheduled. Copy and I/O completion never enter them.
  - **The free rule.** A unit is free the moment its entry is demoted or dropped. A copy may still be in flight on
    it: a store reading it, or a load writing it (an evicted loading entry, round 4). Its reuse is ordered after
    that copy:
    - the next launch waits, on the device, for every such copy on units freed since the last launch;
    - a load into reused units is issued with those copies in its `after`.
  - The tier obeys the same rule for its own storage. A host unit whose bytes go to SSD is free when the write is
    decided, and the read or copy that reuses it waits on the write's ticket.
  - **Fences on host units** (A-1, B-N1, C-1). A host unit is touched by device copies on the tier's streams, by a
    step's own `pre_forward` adopt on the compute stream, and by CPU I/O on the levels' threads. The tier keeps each
    unit's **set of outstanding fences**, not only the latest (round 7). **Rev 8.3:** at most one per ordered
    source, since a later fence of one source implies its earlier ones (and a fence of one source never replaces
    another's, so round 7's case, an abandonment told after a reading step, keeps both):
    - the end event of the latest store that wrote the unit (the D2H stream);
    - the `done` of the latest step that read it, from `loaded(load, done)` (steps end in launch order, and `launched`
      tells in launch order);
    - the latest load copy that read it, from a waiterless or abandoned `loaded` (the H2D stream). A payload load's
      own fence is the store already kept.

    Any reuse of the unit waits for all of them first:
    - stream-side (`Fence.wait`) for a copy into it;
    - host-side (`Fence.synchronize`, on the I/O thread) for a level's `read` into it, or a `write` from it.

    The second case also covers a write-back `pwritev` of a unit whose write-through D2H is still in flight. Fences
    order bytes only, never decisions.
  - **Reads after writes** (round 3, A-1). The rule above orders *reuse* after earlier readers. A device read *from*
    a host unit must also follow the device write that filled it. Both kinds of read count: the tier's H2D load, and
    the kind's `pre_forward` adopt. So the tier keeps each host unit's last **write** fence, the end event of the D2H
    store that filled it:
    - it adds the fence to the `after` of every H2D that reads the unit;
    - a `Load`'s own fence covers the write fences of the host payload units it names, so the launch that waits on
      the load also orders the adopt after the store;
    - a level's `read` into a unit and a level's `write` from it are ordered by their tickets and by the host-side
      rule above.
- **Resume by a virtual clock** (C9, replacing rev 1's fixed `r`).
  - The tier keeps a clock in virtual seconds, advanced at each launch by a fixed per-step cost model
    `a + b·tokens + c·rows` (tune measures `a`, `b` and `c` per lock).
  - A load gets a due time chained over the lanes it passes, each a virtual queue:
    `due_read = max(nvme_lane, now) + bytes / read_rate`, then
    `due = max(h2d_lane, due_read) + bytes / h2d_rate`.
  - A forced write-back (C12) adds its bytes ahead of the read that needs the units.
  - A level's reads and writes share one device lane (C-8). Each read is charged `bytes / read_rate` and each write
    `bytes / write_rate`, on that one lane, so serialization alone models the sharing (round 3, C-1). Charging the
    mixed rate on a shared lane would count the sharing twice. K-M4's mixed-I/O rate is kept only as a check: the
    lane's modelled throughput under a read/write mix must be within 15% of it, else the owner hears before step 4.
    **Rev 8.1:** it was not (0.84× at 50/50, 0.76× at 75/25 on the RTX 5080's drive), so while both reads and writes
    are queued on the lane, each is charged at its rate divided by the measured mixed factor (tune measures it per
    host, K-M4's method); with only one kind queued, reads go at `read_rate` and writes at `write_rate`.
    **Rev 8.3 (read priority, C):** reads are charged at the pure `read_rate` always. While any read is queued, no
    write starts except one that frees units a queued read needs (D11), and only such dependent writes are charged
    at `write_rate` divided by the mixed factor; other writes wait and go at `write_rate`. Step 3 re-measures the
    mixed factor under exactly that dispatch. **Rev 8.5 (the lead, on step 3's measurement):** a read decided while
    the background write lane is busy (`wlane > now`) is charged at `read_rate` times the measured read-side factor
    (`read_beside`: reads beside background writes ran at 0.79–0.92 of their pure rate; 0.80 at 3 threads), not the
    throughput factor, and pushes the background lane back by its charged time; dependent writes stay as built.
    The SSD level's defaults are one 3-thread set (the default `io_threads`) on the RTX 5080's drive: read 2.24 GB/s
    (step 4's `tune --ssd-rates`; rev 8.5 had 2.55), write 1.77 GB/s (sustained, K-M4), mixed 0.74, `read_beside`
    0.80; **tune measures them per host** (`tune --ssd-rates`: `ssd.read`, `ssd.write`, `ssd.mixed`,
    `ssd.read_beside`).
  - Every load is charged to the lanes, promote-into-place loads included, though those are never gated (A-7).
  - A load of at most `stall_bytes` is due now: the step waits for it. It still occupies the lanes (rev 8.3).
  - A payload load takes no H2D lane: the kind's adopt reads the host copy inside the step (rev 8.3).
  - A request with loads is runnable at the first step whose clock is ≥ its due time (KS2). A 512-token prefill step
    advances the clock by its own cost (≈ 0.22 s on the 27B), so a resume behind prefill-heavy steps is not
    penalised in decode steps.
  - If nothing else is runnable, the earliest-due load-gated request runs at once, and its step waits for the copy.
    That is deterministic, because whether anything else was runnable is schedule state.
  - **The clock never runs behind a waited load** (C-2). When a launch waits on a load, `loaded` sets
    `clock = max(clock, due)`. For a gated load that is a no-op (its due time has passed). After a pull-forward or a
    promote-into-place, the clock catches up with the copy the step really waited for, so later due times are not
    stacked on time the model never counted. This is still deterministic: which launch waits on which load is
    schedule state.
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
- **Page store admission is the tier's; payload stores are mandatory** (A-3).
  - `store` may refuse a page copy. A VRAM eviction of a resident entry with no lower copy asks again with
    `evicting = True`. If refused, the entry is dropped with its subtree, which is all non-resident (the resident set
    is prefix-closed), so nothing is orphaned.
  - A payload store at registration must be accepted. The staging unit is freed right after it, so a refusal would
    lose the payload: hybrids would get no recurrent hits, and an invariance check would pass vacuously. A tier
    keeps payloads accepted by sizing its host level for them (acceptance 4). A refusal is a contract violation, and
    the core raises.
  - `std.state_tier`'s admission options (`publish`, `hit:<n>`, `evict`) therefore govern pages only. A level below
    host, the zlib held-out included, takes payloads as it takes pages: when the host level evicts them.

**D6 — Eviction invariants** (panel A10, B1).

*Residency.*
- An entry is **resident** if it holds VRAM units. Resident entries are prefix-closed: `resident(child) ⇒
  resident(parent)`.
- Each entry counts its resident children (`kids`) besides all children. A non-resident entry has its paged copy in
  the tier (`lower`). Payload copies are flagged per spec, in VRAM (staging, D7) or in the tier.
- The core keeps an LRU of resident entries only, so the VRAM scan stays today's size whatever the tier holds. Lower
  recency is the tier's (it receives `hit` calls).

*VRAM eviction.*
- **Eligible:** resident, `kids == 0`, units held only by the index. There are no load pins (round 4).
- **A loading entry can be eligible**, once no live request references it. Evicting it is still a pure free:
  - the core demotes it. Once every entry of its load is evicted and no step has read the load, the core calls
    `loaded(load, done = None)`: the load is abandoned (D7). This holds even if a waiterless `loaded(load, fence)`
    came first, in an earlier round (round 6);
  - its units go on the free rule's list (D4), so the next launch waits on the device for the H2D still writing
    them. An abandoned SSD load whose H2D was never enqueued is skipped, so that wait is a no-op, not a stall on its
    read (round 5);
  - its lower copy is still there, because a tier never drops a copy before `loaded` and reports drops only through
    `dropped()`.
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
- **Every live waiter holds its own reference** (round 4; checked against the code paths). So a loading entry that
  a live request waits on is never eligible, and no pin is needed. A request joins a load's wait set in four ways,
  and each takes a reference:
  - promotion at admission (D7 step 5: the request's own reference, as `_take(shared=True)` does for any hit,
    `state.py:173-175`);
  - hitting an entry already loading (it is resident, so it is in the taken prefix);
  - promote-into-place (the request's own units);
  - `_swap` onto a loading entry (`state.py:355-364` increfs the new units).
- **Round 2's wedge** (A-2) is gone. An aborted or preempted gated request releases its references, so its loading
  entries become index-only and eligible. The waiting request's `reserve` evicts them as pure frees, and its first
  step waits on the device for the loads.
- **Round 3's self-preemption** (A-2) is gone too, with today's formula. When a running decoder R is short and a
  lower-ranked gated request G holds the shortfall in loading entries, `could_fit(R, n, [G, …])` counts those units
  (once G is released, only the index holds them), so `_fit` preempts G, not R (`sched.py:166-169, 177-185`).
- **The property test** (KS1) drives random publish / hit / release / preempt / abort / demote / promote / load /
  tier-drop / eviction-store-refusal sequences against a toy tier. After every operation it checks:
  - prefix closure of the resident set;
  - that every entry's parent exists;
  - `kids` against a recount;
  - that `could_fit(seq, n, releasing)` is true exactly when releasing those requests and evicting yields the units,
    with loads in flight in the mix, evicted loading entries included.
  - `test_state_liveness.py` also checks that no running request preempts itself while a lower-ranked gated request
    holds the shortfall in loading entries. It runs with a RAM load and with an SSD load whose read is delayed past
    the eviction (round 5). In the SSD cell the abandoned load is skipped, and `load_enqueue_wait_s` stays 0.
    A third cell aborts the gated request, then evicts its loading entries two rounds later with the SSD read still
    delayed. The waiterless `loaded(load, fence)` is followed by `loaded(load, None)`, and `load_enqueue_wait_s`
    stays 0 (round 6).

  It extends `core/tests/test_state_liveness.py`'s random-traffic harness.

**D7 — Promotion, publish onto a demoted entry, and payloads through staging.**

*Admission* (A3, C21). Each step proceeds in order, all on index state:
1. The fixed point runs over the matched chain, with payloads in VRAM or in the tier counting alike.
2. The resident prefix is taken (increfed) **first**, so no eviction below can touch the chain being hit.
3. VRAM units for the non-resident tail are allocated from free units plus evictions of eligible entries that have
   a lower copy (pure frees). A promotion never preempts and never issues an eviction store.
4. On a shortfall, the tail is trimmed to what was allocated, and the fixed point runs again on the trimmed chain.
5. The tail entries take those units (the index's reference, plus the request's own as for any hit), and the tier's
   `load` copies their pages in. The `after` events are the last launched step and any copies still in flight on
   reused units (D4's free rule), so a unit freed at the previous commit, still touched by the in-flight step, is
   never overwritten under it (`engine.py:575-576`).
6. A payload adopted from the tier goes **host → the request's slot units directly**. Each payload adopt has its own
   `load(…, payloads=[(key, spec)])` and its own `loaded(…, done)`, even when its pages join another request's load
   (round 5): a wait set shared by joining covers pages only. The tier's `Load` names the host copy (an S2 host pool
   of the payload's unit shape). The kind's own `adopt(view, host_payload)` MemOps run as
   today's `pre_forward` adopt in the first step that reads it (`state.py:203-213`), after that step's launch has
   waited for the load (a pending SSD read included). No VRAM payload unit is taken, and nothing is pinned in VRAM.
   - The cost is the payload's H2D inside that step: 154 MB ≈ 3.2 ms on the 27B.
   - That read runs on the compute stream, not on the tier's streams. So the core calls `loaded(load, done)` with
     the reading step's **end event**, and the tier fences every host unit the load or the adopt read on it (D4)
     before any reuse (A-1, B-N1, C-1).
   - This also covers a superseded tentative that the core drops while another request adopts it: the drop is
     logical, and the host unit's reuse waits for the adopting step's end.
   - A request that joins a loading entry later issues its own payload load. Its adopt is therefore fenced by its
     own reading step's end, never by a `loaded` the tier already received for the earlier load.
   - **One SSD read per payload** (round 6). The joiner's payload load must not issue a second read, nor decide
     whether to on timing. So the payload counts as on host, or on its way there, from the moment the first load
     that reads it is *decided*, whether or not the read has completed. The joiner's load:
     - reuses that read's ticket;
     - is charged nothing new on the lane;
     - gets `due = max(first read's due, now)`;
     - has a fence that chains on the first read (host-side on its ticket), then on the host unit's write fence.

     Abandonment never cancels a level read, only a not-yet-enqueued H2D, so a joiner chained on a read whose first
     load was abandoned still gets its bytes.
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
  - The load is not gated, but it is charged to the lanes (D4).
  - From SSD, the next step's launch may host-wait until the read is enqueued: ≈ 7 ms per 512-token chunk on the
    27B. That stall is reported in 3b-SSD (A-7).
- **Entry loading** (another request's promotion in flight): `_swap` onto its units, and the load joins this
  request's wait set, for its pages only (payloads are loaded per adopter, step 6).
- **Rule for every step:** the launch waits on the device for every load in the wait sets of its rows' requests, then
  clears them. This covers a promoted request, a request that hit a loading entry, and a promote-in-place.
- A payload registered on an entry that already has one, in VRAM or in the tier, is freed, as `attach` does today.
  Both runs then keep the same (earlier) bytes.

*Loads without pins, and cancellation* (A6; round 4 drops load pins). A load holds no reference of its own. Its
units are kept by the index and by every live waiter (D6), and their reuse after an eviction is ordered by D4's free
rule. The core calls the tier's `loaded(load, done)` with a fence **at most once**:
- with `done` = the end event of the first step that reads the load (it waited on the load at launch);
- with `done` = the load's own fence, if its last waiter is gone and no step reads it (at the next launch). This
  waiterless call is a notification, and the core waits on nothing for it. The core keeps the entries' `loading`
  mark, so a later hitter still joins the wait set for the pages and waits on the device.

It calls `loaded(load, None)` **at most once** as well: when every entry of the load has been evicted before any step
read it, whether or not a waiterless `loaded(load, fence)` came first, in an earlier round (round 6). The load is then
**abandoned** (round 5). The tier sets an abandoned flag under the lock its I/O thread takes before enqueuing the
H2D. Either the H2D is never issued, or it was issued already and its fence exists for the free rule to wait on.
- If it is never issued, abandonment **completes the pending-enqueue condition as a no-op and wakes its waiters**
  (round 7). A `Fence.wait` already blocked on that load then returns at once, and so does any later one.
- An abandoned SSD load therefore never stalls a launch on its read.

The rest of the rules:
- Preempting or aborting a load-gated request releases its references and clears its wait set, and the scheduler
  forgets its gate. Its entries stay index-owned, the load lands, and the entries are eligible if nothing else
  holds them.
- No load is cancelled, so no copy lands in units somebody else owns without the next launch waiting for it.
- **Who enqueues an SSD load's H2D:** the level's I/O thread, on read completion, through the tier's stream, never
  the engine thread, after checking the abandoned flag under the lock.
- **I/O order** (round 7; rev 8.2). The tier dispatches I/O jobs (level reads, write-backs, and the H2D enqueues
  that follow reads) **in the order they were decided, except that a read may go before a queued write that does not
  free its own units** (D11's read priority). A job's dependencies (the write that frees its unit, the read a
  joiner chains on) were decided earlier and are never overtaken, so no job waits on one queued behind it.
- **The only remaining host wait** is a launch that waits on a live load whose H2D is not yet enqueued (its
  `Fence.wait` first waits on the host until it is): a reader due early, a pull-forward, a promote-into-place, or a
  free of some but not all of a load's entries. Its time is reported in `stats()` as `load_enqueue_wait_s`. The
  progress rule needs no special case: with no pins, an idle schedule either has a gated request to pull forward
  (D4) or units `could_fit` counts.

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
  that point. K-M1 measures where re-rendered answers diverge, fixes k and predicts 3g. **Rev 8.2:** K-M1 found 38 of
  38 answers re-rendered verbatim, so `answer:1` is the default and `answer:2` an opt-in for templates that
  re-serialize (panel C).
- Decode-produced checkpoints are `bounded` (`docs/m03-qwen38.md:22-23`), the same class as today's hits across
  generated KV pages (`docs/design.md:900-902`).
- Verify rows publish nothing (`state.py:239`), so with MTP on the policy is idle. The 27B locks run without drafting.

*Payload recency in the tier.* A turn leaves `prompt_end` and k answer payloads on its chain. Later turns match
through those entries without adopting them. The tier therefore orders payload copies by store or adopt (`hit`'s
`adopted` argument: one `(key, spec)` per adopted payload spec, B-N7), not by entry hits, so stale interior payloads
go before live KV.

*Policies are std's; a new one is a plugin.* `recurrent.policy` is a comma list of built-in names or `Ref`s to
plugins with the structural `next(view) → (at, tentative) | None`. Plugins cannot import std
(`pyproject.toml:85-89`, `:93-99`), so the protocol is duck-typed and documented in std. The kind takes the earliest position
over its policies. **Rev 8.3:** a unit may declare `keep` (default 0), the tentative payloads a request keeps for it;
the kind's `keep_tentative` is the largest `keep` over its policies (`answer:<k>` has k), so the engine's refusal of
tentatives without a tier sees a plugin's too. A position a unit without `keep` calls tentative is not (nothing would
keep it).

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
  bound like any site. K adds providers in a plugin of their own, `kernels.state_copy` (rev 8.3: first planned in
  `kernels.expert_io`, which already has both paths for experts, `docs/v1-offload.md` X step 0):
  - one `cudaMemcpyBatchAsync` per batch;
  - the SM scatter/gather kernel, 0.97–1.02× DMA in X's M1 (`docs/v1-offload.md:645`).

  K-M2 picks between them. Without a provider (CPU, tests), the copier falls back to the per-run loop.
  **Rev 8.1:** loads default to the SM gather (rev 8.2: pinned at 8 CTAs; paced DMA stays tune's alternative). K-M2 measured both at 0.99× the blob rate on fragmented 34,816 B
  chunks, but K-M3 measured one batch-DMA load in flight slowing 4-row decode 1.42× (head-of-line blocking on the
  copy engine) against 1.07× for the gather. D2H stores use batch DMA. The 27B locks pin the provider at step 4.
- C8's point stands: a block-contiguous *host* layout does not reduce the device-side chunk count, so it is dropped
  as a copy fallback.
- The SSD level does use a **unit-major host layout** (a host pool of one segment per unit). A 27B page is then one
  557,056 B iovec (136 × 4,096), aligned for O_DIRECT on 4Kn drives. That is legal because the copies across the
  host pool are the tier's own, not S2 MemOps (which require equal unit shapes, `memory.py:130`).
- Payload adopts (D7) are the kind's MemOps from a host pool of the payload's own shape, so they stay S2-legal.

**D11 — The SSD level** (`std.state_tier`'s first `Level`, option `ssd.dir`).

*Layout.* One preallocated slot file per unit shape (`fallocate`), sized by `ssd.bytes`. A slot holds one unit-major
unit. At `bind`, the level takes an exclusive `flock` on `ssd.dir`, and refuses a directory another engine holds:
two engines on one host would otherwise overwrite each other's slots, and runtime reads are not crc-checked (A-6a).

*I/O.*
- `os.preadv` / `os.pwritev` of whole units on the tier's `io_threads` (rev 8.5: default 3; step 3 measured reads
  at 2.32 GB/s from three against 1.79–1.90 from two on the RTX 5080's drive). The threads are pinned off the engine's
  `loop_core`.
- O_DIRECT where the filesystem allows it; otherwise buffered with `posix_fadvise(DONTNEED)`, or by option (T2b).
- K-M4 measures the engine loop's host time per step beside the threads (GIL).

*Write-back and determinism.*
- The SSD level admits at host eviction. The host level keeps a free-unit watermark (`host.reserve`). Its default is
  the units of one `max_model_len` load, so an SSD load seldom forces a write-back (C12).
- Below the watermark, std's step hook demotes host LRU leaves to SSD. The decisions read logical counts only (D4):
  the host unit is free when the write is decided, and a reuse waits on the write's ticket.
- **Rev 8.1: tentative payloads are never written back.** At host eviction a tentative checkpoint is dropped, not
  written to SSD: K-M5 measured 73–88 GB written per `agent_park` run with an SSD below 4–8 GiB of host, 94% of it
  checkpoints, mostly never read back. A tentative becomes ordinary when its request is released, and the core
  tells the tier (`StateTier.settled`), so this policy sees the current flag.
- **Rev 8.2: superseded payloads are not written back either** (panel C). At host eviction a payload is dropped
  rather than written back if walking down single-child links from its entry reaches an entry holding an ordinary
  payload of the same spec: a later checkpoint of the same chain makes it redundant. An entry with two or more
  children keeps its payload (it is a branch point). The panel's replay of this rule on K-M5's harness cuts payload
  write-back by 42–81% with no TTFT loss; writing back only adopted payloads was rejected (its replay regresses TTFT p50 from 0.40 s to 3.11 s).
- **Rev 8.2: reads go first on the SSD** (panel C). While a read is queued, at most one write job is in flight, and a
  read never overtakes the write that frees its own host units. "Both kinds queued" in D4's mixed charge means the
  virtual lane's state, not the threads'. Step 2 re-measures the mixed factor under exactly this dispatch (one writer,
  two or three readers). **Rev 8.3 (read priority):** step 2 measured 0.77–0.80 under it, no better than without, so
  the dispatch tightens: while any read is queued, no write starts except one that frees units a queued read needs
  (a dependent write). Reads are charged at the pure read rate, the mixed factor only on dependent writes (D4). Step 3
  re-measures exactly this dispatch.
- An entry being written back is logically on SSD. A hit on it is an SSD read with an SSD due time, and its load
  reads the slot. The rev 2 shortcut of reusing host bytes still physically there is dropped (A-5). It raced with a
  reallocation of the logically free unit, and keeping it would need a second fence for a rare case.
- Copies redundant with a faster level (an SSD copy of an entry whose host copy exists) are dropped first at each
  level (A9).

*Persistence (`persist = close`)* (A7, C4).
- The manifest is a **snapshot written only at a clean `Engine.close`**:
  1. every resident and host-level entry not yet on SSD is written back (sustained write rate, K-M4);
  2. the data is `fdatasync`ed;
  3. the manifest is written to a temporary file, `fsync`ed and renamed into place, and the directory `fsync`ed. It
     holds:
     - the slot-file geometry (unit bytes and slot count per shape);
     - one record per block: key chain via parent, tokens, slot per spec and payload, and a crc32 of the slot's
       data, computed by the I/O thread from the host copy before the write.

     Tentative payloads of requests still running at close are written as ordinary payloads, as their release would
     have made them (A-6c).
- **At startup**, the manifest is read, then unlinked and the directory `fsync`ed, before any slot is rewritten. A
  record can therefore never name a slot that was reused after it was written. This has the effect of durable
  tombstones for every slot at once, without one per reuse.
- Restore runs **inside `attach`**, before the index is visible, so a discard can never depend on timing (C-9):
  - records naming slots past the current files' geometry (a smaller `ssd.bytes`) are discarded (A-6b);
  - each remaining slot's crc is verified on the I/O threads; records that fail are discarded with their
    descendants;
  - records are sorted parents first (host eviction is leaf-first, so the snapshot order is not), and orphans are
    discarded.

  Verification is startup time, ≈ 0.38 s per persisted GB at 2.6 GB/s (≈ 38 s for a full 100 GB level), reported
  in 3f. `zlib.crc32` runs at ≈ 4.6 GB/s per thread (panel C's timing), so the drive bounds it.
- A crash leaves no manifest, so nothing is restored. That is strict on purpose: persistence is for planned restarts.
- **Identity** (A13, B6, B-N3): `plans.state_identity(engine)`, the plan's id computed without three things:
  - the tier plugin's own options, which are byte-neutral by tier invariance (`host.bytes` included, which stays
    planned in `plan_id`);
  - `planned = false` options;
  - the clamp-dependent `max_model_len` (`engine.py:262`).

  If the plan cannot be computed (`plans.py:151-155`), a tier asking for persistence fails at startup. Keys are re-hashed from tokens at restore; Python's `hash` of an
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
    units: Mapping[str, tuple[int, ...]]      # store: spec -> device units of its paged copy (empty: payloads only)
                                              # restore: spec -> () for each spec whose paged copy the tier holds
    payloads: Mapping[str, Payload]           # store: spec -> payload (device staging units)
                                              # restore: spec -> Payload((), "") for each payload it holds

class Fence(Protocol):             # new: a point in device order
    def wait(self) -> None: ...    # the current stream waits for it (the host first waits until it is enqueued)
    def synchronize(self) -> None: ...   # the host waits for it

class Stream(Protocol):            # new: a copy stream the core makes (deferred off CUDA under the test fixture)
    def run(self, fn: Callable[[], object], after: Sequence[Fence] = ()) -> Fence: ...

class Load(Fence, Protocol):
    due: float                                         # on the tier's clock
    payloads: Mapping[tuple[int, str], Payload]        # (key, spec) -> its host copy, for the kind's adopt

class TierCtx(Protocol):           # new: what `attach` gets
    pools: Mapping[str, Pool]      # the device pools it may copy: paged specs' and payload pools
    identity: str | None           # plans.state_identity; None if the plan cannot be computed
    max_seqs: int; max_model_len: int; block: int
    options: Mapping[str, object]
    def host_pool(self, name: str, *, unit_bytes: int, segments: int, units: int, mapped: bool = False) -> Pool: ...
    def stream(self) -> Stream: ...

class StateTier(Protocol):         # slot `tier`: single, opt-in; the plugin's `build(ctx: BuildCtx) -> StateTier`
    def attach(self, ctx: TierCtx) -> list[BlockRecord]: ...   # after the final materialize; restored records
    def clock(self) -> float: ...                               # virtual seconds at the step being scheduled
    def store(self, blocks: Sequence[BlockRecord], after: Sequence[Fence], evicting: bool
              ) -> tuple[set[tuple[int, str | None]], Fence]: ...  # accepted (key, spec; None: the pages)
    def load(self, keys: Sequence[int], units: Mapping[str, Sequence[int]],
             payloads: Sequence[tuple[int, str]], after: Sequence[Fence]) -> Load: ...
    def loaded(self, load: Load, done: Fence | None) -> None: ...  # at most once with a fence (the reading step's
                                                                    # end, or the load's own fence if waiterless),
                                                                    # and at most once with None: abandoned (skip
                                                                    # the H2D if not yet enqueued)
    def hit(self, keys: Sequence[int], adopted: Sequence[tuple[int, str]]) -> None: ...
    def drop(self, keys: Sequence[int], spec: str | None = None) -> None: ...
    def dropped(self) -> list[tuple[int, str | None]]: ...     # its own evictions since the last call
    def clear(self) -> None: ...                               # flush_prefix
    def close(self) -> None: ...                               # Engine.close (persist, release)

class EntryView(Protocol):         # new (KS5)
    key: int; depth: int; hits: int

class EvictionPolicy(Protocol):    # slot `eviction_policy`: single, opt-in
    def victim(self, candidates: Iterator[EntryView]) -> EntryView | None: ...
```

Notes on the block (B-N4):
- **Build.** The tier plugin's `build(ctx: BuildCtx) -> StateTier` runs at engine build, as the logits processor's
  does (`engine.py:189-191`), so it can bind its `state_copy@1` site.
- **Step hooks.** The core calls the tier object's own S1 hooks (`before_launch`, `after_commit`) directly, beside
  the shared objects' (`executor.py:183-184`), with no detour through `BuildCtx.shared`. The hooks advance its clock.
- **Fences.** The core's events are `Fence`s (its platform event, which already has `wait` and `synchronize`); the
  tier's copy and load handles are too. `holon.api` had no `Event`; `Fence` and `Stream` are new.
- **Host pools.** `host_pool` allocates S2 host pools now, against what `host_memory` has left.

Other additions:
- `SeqView` gains `branch_at: int`;
- payload kinds may declare `keep_tentative: int`;
- manifest `Option` gains `planned: bool = True`;
- SLOTS gain `tier` and `eviction_policy`, both single.

ARC cannot be written as `victim(candidates)` alone: it needs to observe inserts, hits and evictions and keep ghost
lists (B-N5). It would add optional observer members to `EvictionPolicy` (`inserted`, `touched`, `evicted`), an
additive minor when ARC is built. K does not add them.

**The determinism contract** (the tier's obligation, in `StateTier`'s docstring and enforced by the judge):
- every return value and every `dropped()` list is a function of the calls received and the step hooks, never of
  copy or I/O completion;
- payload stores are accepted (D5);
- reuse of its own storage waits on its own tickets, and reuse of a host unit, or CPU I/O on it, waits on all the
  unit's outstanding fences, the latest of each ordered source: the store that wrote it, the step that read it (a
  `done` given to `loaded`), the load copy no step waited for (D4, rev 8.3);
- I/O jobs are dispatched in the order they were decided; while a read is queued no write starts but one that frees
  units a queued read needs (D11, rev 8.3); abandonment completes a pending enqueue as a no-op
  and wakes its waiters (D7);
- every device read of a host unit follows the unit's last write fence: in the `after` of the tier's H2D, and inside
  the `Load` fence for host payloads a kind's adopt reads (D4);
- a copy being loaded is never evicted before `loaded`, and `loaded` advances the clock to at least the load's due
  time.

| # | Seam | Today → change | Toy proof | Lines |
|---|---|---|---|---|
| KS1 | **Residency in the index** | Device copies only; eviction drops (`prefix.py:89-133`) → `Entry` gains `lower`, payload-copy flags, `kids` (resident children), `depth`, `hits`. A resident-only LRU. D6's eligibility, demote / eviction store / drop with subtree; `dropped()` handling; stores' `Fence` kept for units freed while read, waited by the next launch and passed as `after` to loads into reused units | the property test (D6); demote then promote bitwise; reuse of a unit with a store in flight waits for it under the deferred stream | 75 |
| KS2 | **Promotion, promote-into-place, load gating** | Hits adopt VRAM only → D7's admission (prefix first, pure-free evictions, trim and re-run), promote-into-place in `publish`, per-request wait sets waited at launch, `loaded(load, done)` once per load, D4's free rule extended to loads (no load pins, round 4), the scheduler's gate (`clock() ≥ due`), idle pull-forward, preempt and abort clearing gates | resume step identical with the toy tier's delay at 0, 1 and 2 steps (run-identical step lists); a trimmed hit recomputing demoted blocks reads the cached bytes; preempt and abort of a gated request with a delayed copy (no unit reused before landing); progress with every slot held by gated requests and a request waiting (A6a); progress when an aborted gated request's loading entries hold the units a waiting request needs and nothing runs (round 2, A-2); a running decoder preempts the lower-ranked gated request holding its shortfall, never itself (round 3, A-2), with RAM and SSD loads; an evicted loading entry's units are not written before the next launch waits on the load; an abandoned SSD load is skipped | 79 |
| KS3 | **Stores, staging payloads, tentatives** | Payloads permanent from registration; release drops nothing to store → paged `store` at insert (`after` = last launched) and at eviction; payload `store` at registration (`after` = producing step) freeing the staging unit; host-payload adopt through the kind's `adopt` in the reading step; `Payload.tentative` with per-request `keep_tentative`; `dropped_publishes`; `SeqView.branch_at`; `evict_payloads` checks `num_free` before copying the entry list (behaviour-preserving) and is skipped for staging pools when a tier is bound (C-4) | tentative decode payloads: a preempted request resumes from the last; finish leaves k; the depth-1 extra row's payload registers only where committed tokens cover it; a staging unit reused while its D2H is delayed is not overwritten early | 63 |
| KS4 | **Slot `tier`, attach, identity, restore** | None (release §15.1) → slot and protocol types (`Fence`, `Stream`, `TierCtx`, `BlockRecord`, `Load`, `StateTier`); the tier's S1 hooks called directly; `attach` after the final materialize (host pools allowed then, before graph capture); `plans.state_identity`; `Option.planned` honoured by `plan_of`; restore inserts non-resident entries parents first; `flush_prefix` → `clear`; `close` | restart a toy engine with a persistent toy tier: resume bitwise; another identity is cleared; `planned = false` options leave `plan_id` unchanged | 55 |
| KS5 | **Slot `eviction_policy`** | LRU only → eligible candidates lazily, LRU first, to `victim`; default the first | a toy policy pinning depth ≤ 1 keeps the root under pressure; property test with it on | 20 |

≈ 292 lines; the budget is ≤ +320 over K's step-1 base (rev 5: ≈ 290; rev 4: ≈ 306; rev 3: ≈ 298; rev 2: ≈ 285 within +300; rev 1:
480). Step 1 first
prototypes KS1 and KS2 on CPU
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
  - The host level: S2 host pools per device unit shape, payload pools in the payload's own shape, unit-major KV
    pools when a `Level` is configured below it (D10).
  - The levels below host: `state_tier.levels`, a list of built-in names or `Ref`s to structural `Level`s (D3). The
    built-in `ssd` is the first (D11).
  - Options:
    - `host.bytes`: planned, pinned by the lock; `attach` refuses if `host_memory` cannot give it (B-N3);
      `host.pages`, the share of it for pages; each part is split over its pools by their unit bytes, so every
      paged spec, and every payload spec, gets the same number of units (rev 8.3);
    - `host.admit` = `publish` | `hit:<n>` | `evict` (pages only; payloads are always stored);
    - `host.reserve` (with a level: free units per host pool kept by writing leaves back; default one
      `max_model_len` load's);
    - `levels` (one in K: `ssd` or a level unit's `Ref`);
    - `ssd.dir`, `ssd.bytes`, `io_threads` (only `ssd.dir` and `io_threads` are `planned = false`, rev 8.5);
    - `ssd.direct`, `ssd.persist` = `off` | `close`; `ssd.read`, `ssd.write`, `ssd.mixed` (the clock's rates, K-M4's
      by default, tune's later);
    - the clock: `step_cost` (`a,b,c`), `h2d_rate`, `stall_bytes`; levels report their own rates (tune measures
      them).
  - Leaf-first LRU per level, with copies redundant with a faster level dropped first (A9), and payloads ordered by
    store or adopt recency (D8).
  - The determinism contract and the host-unit fences (§4, D4), the copier (D10).
  - The in-flight floor check at attach (acceptance 4).
  - ≈ 450 lines for the tier and the `Level` protocol, plus ≈ 180 for the SSD level with its manifest.
- **`std.state.recurrent`.**
  - `recurrent.policy` becomes a comma list of built-in names or `Ref`s (structural `CheckpointPolicy`).
  - New built-ins `branch` and `answer:<k>` (which sets `keep_tentative`).
  - The pool becomes staging when a tier is bound (§7's formula sets its size).
  - ≈ 70 lines.
- **`kernels.state_copy`** (rev 8.3: its own plugin, not `kernels.expert_io`): `state_copy@1` providers (batched DMA
  and SM gather over chunk address lists), ≈ 150 lines of CUDA and its build glue.
- **`evict.root_pin`** (`plugins/holon_plugins/evict/root_pin/`, slot `eviction_policy`): skips candidates with
  `depth ≤ d` and `hits ≥ h` while others exist. ≈ 40 lines.
- **Held-out (after the tag):** `policies.last_turn` (≈ 40 lines) and `state_tiers.zlib`, a `Level` (≈ 120 lines
  plus tests; §1).
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
| Null test | exact | tiers off, `prompt_end` only, checkpoint pool 4 vs 16 units, back-to-back trace: `cached_tokens` asserted equal, outputs byte-identical |
| `branch` (in-step, prefill-produced) | as `prompt_end`: bitwise the recomputed state at the same chunk boundaries (a CPU test with equal chunking); bounded across chunkings and vs recompute (L1.5) | the bounded multi-turn check (`docs/design.md:1297-1303`) with a shared-system-prompt session set added. `bounded.Trace.run` submits each turn as one wave (`judge/holon_judge/bounded.py:70-79`), so `branch_at` would never be set inside a wave. The added sessions therefore start in a later wave than the first, and `branch_off_by_one`'s fired count must be > 0 (C-6) |
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
| `adopt_source_reused` (A-1, B-N1, C-1; extended in round 5) | CPU + GPU (rev 8.3: the GPU in run-identity's `fork` loop: the host payload level at its floor, `answer:1`, a fork resending a prompt plus 128 tokens) | `loaded` ignores `done`: a host payload unit a step's adopt reads is reused by a later store or an SSD read without waiting for that step's end (deferred stream, plus an I/O thread delayed into the window). Second variant: a request that joins a loading entry later adopts the payload through the earlier load instead of its own. The second variant also counts SSD reads, which must be one per payload (round 6). It also runs the abandon-while-waited cell with its delayed I/O thread (round 7), and the call order where G is aborted and J is due in the same step, and G's `loaded` is processed second; with a replace-the-latest fence rule, J's fence would be overwritten (round 7) | the adopted recurrent state is corrupt; tier invariance fails, in both variants. A second read of the same payload fails the count |
| `load_reuse_unwaited` (round 4) | CPU + GPU | units of an evicted loading entry are reused without the next launch waiting on the load's fence; the H2D delayed. Rev 8.3: run-identity's `reuse` loop aborts the gated request with the most loading blocks, with no resend, and sends a prefill-heavy request in the same round | the late H2D overwrites the new owner's bytes; run-identity fails. **Rev 8.3:** caught on the CPU with copies deferred up to 8 launches and 32-token answers (deferred up to 2, the copy has landed by the abort); on the A5000 it fires (14 at 0.05 s delays, 40 at 0.5 s) and changes nothing |
| `waiter_unref` (round 4) | CPU | a request joining a loading entry's wait set by `_swap` takes no reference | the loading entry is evicted under a live waiter: the property test fails (`could_fit` counts held units) and tier invariance fails |
| `load_before_store` (round 3, A-1) | CPU + GPU | the tier omits a host unit's write fence from the reading H2D's `after` and from the `Load`'s fence; the D2H stream deferred; eviction then promotion within 1–2 steps, plus a sibling request adopting the same host payload | the promoted pages or the adopted state read the unit before its store lands; tier invariance fails. **Rev 8.3:** end to end the core's own ordering (D4's free rule, a load's `after`) already covers it, so for the host level it is a redundant guard; the tier-only test (a store, then at once a load with an empty `after`, on deferred streams) holds it |
| `store_early` (A2) | GPU | a paged store is issued after the committed step, not the last launched | host copies stale; tier invariance fails on a trace where the in-flight step completes published blocks |
| `store_unwaited` | CPU + GPU | reuse of a unit freed while its store reads it does not wait | host copy corrupt; tier invariance fails under pressure |
| `load_no_after` (C10) | GPU | a promotion's copy ignores `after` (the finished request's extra row overwrites it) | tier invariance fails |
| `read_unwaited` (C10) | CPU | an H2D is issued before its SSD read ticket completes. Also run in the abandon-while-waited cell (round 7): with the I/O thread delayed, a launch is already waiting on the load when it is abandoned | tier invariance fails (SSD trace). In the abandon-while-waited cell, a correct tier wakes the waiter at abandonment, so `load_enqueue_wait_s` stays 0 and the cell does not time out |
| `promote_wrong_unit` | CPU | a promotion copies the neighbouring host unit | tier invariance fails; bounded L2 fails |
| `swap_onto_lower` (A1) | CPU | publish onto a non-resident entry keeps the request's own bytes | tier invariance fails on the serial trace (turn 2's recomputed answer) |
| `gate_on_landing` | CPU | a gated request runs at the first step after its copy lands instead of at its due time | run-identical step lists fail with deferral |
| `free_on_ticket` (A5, C5) | CPU | the SSD level frees host units when the write ticket completes | run-identical fails with I/O delays |
| `promote_preempts` (A3) | CPU | promotion may preempt lower requests | the liveness test reports preemptions with no compute shortfall |
| `kids_any_level` | CPU | VRAM eligibility counts non-resident children | the property test fails (`could_fit` yes, eviction short) |
| `interior_drop` | CPU (std) | the host level evicts interior copies first | std's property test fails (subtrees lost); outputs unchanged |
| `branch_off_by_one` (C10) | CPU | the `branch` payload is filed one block early | bounded check fails (KL) |
| `tentative_shift` | CPU | an `answer` payload is filed one block early | bounded check fails. **Rev 8.3:** the judge's audit checks every payload at its position ((depth + 1) × block, the position the kind published it at); with `answer:1` alone the block before holds no checkpoint, so the plant misfiles and the audit fails run-identity |
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
| K-M4 | NVMe `pwritev` / `preadv` of unit-major units (557,056 B) and checkpoints as per-segment iovecs (48 segments on the 27B; O_DIRECT alignment of each segment size checked), O_DIRECT and buffered, 1–4 threads, reads, **sustained writes** past the SLC cache, and **mixed reads and writes** on one drive, with the engine loop's step time measured beside them (GIL) | `io_threads`; the read and write rates the shared lane charges, and the mixed rate as its check (C-8, round 3 C-1); the close-flush estimate; D9's SSD rate |
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

**Predictions** (RTX 5080, 27B UD-IQ3_S; 47.5 GB/s H2D; prefill 2.2–2.5K tok/s). **Rev 8.2** (panel C): a decode
step is the measured 4-row step at the lock's settings, 22.3 ms on the RTX 5080 and 14.7 ms on the RTX 5090 (K-M3),
not the single-row tg128 step (15.7 / 11.4 ms) rev 8 used; "one decode step" in the bars of 3a and 3h means it.

| Row | Prediction |
|---|---|
| 3a, 16K from RAM | KV 570 MB ≈ 12.0 ms of copy, and a payload adopt of 154 MB ≈ 3.2 ms in both arms. Bar: VRAM + 18.0 ms + one step (40.3 ms). Predicted extra ≈ 12–28 ms if `state_copy@1` reaches ≥ 0.8× of 47.5 GB/s on fragmented units; fails with the per-run loop |
| 3a, 32K from SSD | 1.29 GB at ≈ 2.6 GB/s ≈ 0.50 s + 27 ms H2D. With the default watermark no write-back is forced; a forced one adds 1.29 GB at the sustained write rate (K-M4). Bar ≈ 0.79 s + one step |
| 3b | **Rev 8.2:** the load by the SM gather at 8 CTAs (K-M3: 1.04× on the RTX 5090, 1.07× on the RTX 5080 beside 4-row decode; an unpaced batch DMA 2.07× / 1.42×) → gated-step median gap ≈ 1.04–1.07×, p90 ≤ 1.15× (bar 1.1× / 1.2×); the append step's ratio ≈ 1.0× (reported). Step 2 measures the gather beside compute-bound prefill and its host cost per load, the work-list build included; paced DMA stays tune's alternative |
| 3c | VRAM-only: the 4-unit pool keeps no returning session's payloads, so every hit trims to 0, and each turn recomputes its context (≈ 9–10K at resume by C's σ = 0.6 sim, ≈ 4 s; C-10). Tiered: ≈ 300 MB loaded (≈ 6 ms) plus the append: ≈ 400 tokens with `answer` (≈ 0.18 s), ≈ 700 without (≈ 0.3 s) → TTFT p50 ≈ 0.2–0.35 s, ratio ≈ 0.05–0.1× (bar 0.25×). Throughput gain 1.3–2× from removed prefill (bar 1.2×) |
| 3d | `llama-server`'s save is synchronous (an observed 2.65 s stall), and whole-state blobs repeat the system prompt per conversation. Prefill parity at depth (Holon 1.05× the gate build at 8K, 1.01× upstream, same evidence) → ≤ 1.0× predicted. No margin estimate until step 4; the bar is fixed now |
| 3e | sessions 1–2 recompute 16K (≈ 7 s); sessions 3–16 hit 16,384 → first-turn TTFT ≈ 0.1–0.2 s → p50 ratio ≈ 0.03× (bar 0.2×) |
| 3f | 16K resume as 3a's SSD cell; restore verification inside `attach`, startup time ≈ 0.38 s per persisted GB (≈ 6 s for 16 GB) |
| 3g | 44% (27,646 / 62,682) of uncached input → ≤ 10% if K-M1 finds the answers re-rendered verbatim up to their tool calls |
| 3h | RTX 5090 decode step 14.7 ms (rev 8.2: K-M3's 4-row step; tg128's single-row step is 11.4 ms, `evidence/bars_release-qwen3.8-27b-ud-q4_k_m_rtx-5090_2026-10-15.json`); ≈ 300 MB of KV ≈ 6 ms plus the host adopt ≈ 3.2 ms per hit, against roomy's ≈ 0.3 ms D2D → ≤ 1.05× roomy + one step (bar 1.1× + one step) |

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
| 1 | CPU prototype of KS1–KS2 and its pricing; then KS1–KS5 with toy proofs; the property test; the deferred CPU stream fixture; the design.md edits (§11) | core tests; `check_abi --base <K's base>` additive; ≤ +320 lines over K's base; denylist; tiers-off snapshots unchanged on the A5000 | 6 |
| 2 | `std.state_tier` host level and copier; `state_copy@1` providers; `branch`, `answer`; `holon_judge tiers`; the CPU logic plants | tier invariance (CPU, Qwen3-8B on the A5000); null test; run-identical × 3 with delays; bounded multi-turn for the policies; CPU plants and the A5000 stream-timing plants | 5 |
| 3 | The `Level` protocol and the SSD level: write-back, watermark, shared NVMe lane, `flock`, persistence and restore with geometry; `evict.root_pin`; **tag the API** | tier invariance with SSD; 3f's two-engine restart test; `torn_manifest`, `stale_slot`, `read_unwaited`, `free_on_ticket`, `adopt_source_reused` with a delayed I/O thread | 5 |
| 4 | The gate on the RTX 5080 and 5090: 3a–3i, the 27B tier-invariance pair on the 5090; tune sets the clock, rates, `host.bytes` and the staging size; the 27B agent locks | acceptance 3; `dropped_publishes` 0; the locks' bounded and run-identical verdicts; carried from step 3 (rev 8.5): `adopt_source_reused` with a delayed I/O thread on the GPU, and `free_on_ticket` with I/O sleeps of 0 or ≥ 2 steps | 4 |
| 5 | Held-out plugins after the tag: `policies.last_turn`, the `state_tiers.zlib` `Level` | plugin-only diffs (or the recorded proof finding and its minor); tier invariance with the zlib level | 2 |
| 6 | Close: full suite, `run_all.sh`, panel code review; lock propagation if K closes second (D14) | acceptance 4–6 | 2 |

26 working days (rev 1: 22). Against rev 1: +1 for the prototype, +1 for the copier and plants, +1 for the held-out
work, +1 for the gate's added rows. Against rev 2, the total is unchanged: step 3 gains a day for the `Level`
protocol, and step 5 loses one, because zlib is a `Level` rather than a whole tier. Waiting for FN's core round, if
it holds the base, is outside this count.

## 10. Risks and deferred

**Risks.**
- **The copy path is the critical path to 3a and 3b.** If neither DMA batching nor the SM gather reaches 0.8× on
  fragmented chunks, loads cost whole steps, and 3b misses for 32K resumes. Hence K-M2 runs first.
- **The core estimate.** B2's shrink rests on ≈ 292 lines. Step 1 prices it on a prototype before building the rest.
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
- `flush_prefix` under a running request: its next publish names a parent the flush dropped (a KeyError, as before K);
- `HitLen.cost` (D9);
- layer-wise loading and per-layer events (D12);
- K4 (FN, GLM; D15);
- ARC as an `eviction_policy` plugin, with the optional observer members it needs (B-N5);
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
- §6.4 (`:918-923`): the tier, not a per-request choice, owns loads. Loads land in index-owned units whose reuse waits on the load and are
  never cancelled. BLOCKED stays for logits processors only.
- §9's KV offload row (`:1338`): `std.state_tier`, slot `tier`, KS1–KS5.
- The host-pool timing rule (B-N6). Host pools may also be allocated at a tier's `attach`, after the final
  materialize. Three places say `before_materialize` is the last point, and each gains that exception:
  - design §6.1 (`docs/design.md:817`);
  - `BuildCtx.pool`'s docstring (`core/holon/api/model.py:120`);
  - `RequestsPools.before_materialize` (`core/holon/api/hooks.py:162`).
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
| Write-safe cancellation | D7: loads land in index-owned units and are never cancelled; frees wait on in-flight stores and loads (D4); plants `store_unwaited`, `promote_unwaited`, `load_reuse_unwaited` |
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

### Rev 3 (2026-10-18): the panel's second round

Round 2 found A: 3 major and 4 minor; B: 3 major and 4 minor; C: 3 major and 7 minor. All of round 1 was confirmed
fixed, except that B2 with B3 and B6 were partly open, which B-N2 and B-N3 complete. Each new finding was checked
against the code at this branch's base:
- the adopt is a `pre_forward` MemOp on the compute stream (`executor.py:348`, `memory.py:152-159`);
- "no progress" raises when `idle_for` is 0 (`engine.py:343, 447-453`);
- unplanned options leave plans (`plans.py:80-84`);
- `holon.api` has no `Event` (`core/holon/api/__init__.py`);
- three docstrings say `before_materialize` is the last point (`model.py:120`, `hooks.py:162`, `design.md:817`);
- `evict_payloads` copies the entry list before its early return (`prefix.py:117`);
- `bounded.Trace.run` submits one wave per turn (`bounded.py:70-79`);
- the RTX 5090's 27B tg128 is 87.34 tok/s (≈ 11.4 ms per step).

The lead decided A-1/B-N1/C-1, A-2, A-3, B-N2, B-N3, C-2, C-3, C-8 and C-9; the rest are taken as the reviewers
proposed. None is rejected. Two are taken in a changed form, and one in part.

| Finding | Disposition | Where / why |
|---|---|---|
| A-1 = B-N1 = C-1 (major) host payload unit reused under an in-flight adopt | accepted | `loaded(load, done)` with the reading step's end event. Every reuse of, or CPU I/O on, a host unit waits on its last fence: stream-side for copies, host-side for level I/O. This also covers a write-back `pwritev` racing a D2H (D4, D7, §4 contract). Plant `adopt_source_reused` (CPU deferred stream plus a delayed I/O thread) |
| A-2 (major) waiterless pins wedge the scheduler | accepted | D7: when a schedule would produce no rows while waiterless pins exist, host-wait them, `loaded`, unpin, reschedule. In KS2's progress proof and the liveness test; plant `pin_wedge` |
| A-3 (major) refused payload stores undefined | accepted (the simpler rule) | D5: payload stores at registration are mandatory, and admission options govern pages only. The host level is sized so payloads are never refused (acceptance 4). zlib, as a `Level`, takes payloads with pages at host eviction |
| A-4 tier invariance cannot hold under `hit:<n>` | accepted | acceptance 2: the byte-identity claim covers `publish` and `evict` admission. `hit:<n>` is judged by run-identity and `bounded`, and the judge refuses to compare runs whose `cached_tokens` differ |
| A-5 the write-back host-bytes shortcut races reuse | changed | D11: the shortcut is dropped, so a hit on an entry being written back reads its slot. The reviewer's fence-based fix would keep a second fence for a rare case |
| A-6 persistence edges (a, b, c) | accepted | (a) `flock` on `ssd.dir` at `bind`; (b) the manifest records geometry, and out-of-range slots are discarded; (c) running requests' tentatives are written as ordinary payloads at close |
| A-7 promote-into-place loads not charged to the lanes | accepted | D4 charges every load; D7 reports the SSD host-wait in 3b-SSD |
| B-N2 (major) a new medium means a whole tier | accepted | D3: std's structural `Level` (bind, write, read, drop, rates, restore, close), bound by `Ref` in `state_tier.levels` as `moe.tiers` is. SSD is the first `Level`; zlib is a `Level`, ≈ 120 lines plus tests |
| B-N3 (major) `host.bytes` unplanned contradicts step 4 and X | accepted | `host.bytes` planned and pinned; `attach` refuses if `host_memory` cannot give it. `planned = false` covers only `ssd.dir`, `ssd.io_threads` and `ssd.bytes`. `state_identity` leaves out the tier's options |
| B-N4 ABI types undefined | accepted | §4: `Fence`, `Stream`, `TierCtx` listed; `build(ctx: BuildCtx)` factory; S1 hooks called directly; `BlockRecord` on restore defined |
| B-N5 ARC cannot be a `victim` plugin | accepted in part | §4 and §10 state that ARC needs optional observer members, an additive minor when ARC is built. K adds none, because nothing in K would read them |
| B-N6 §11 misses the host-pool timing rule | accepted | §11: design §6.1, `BuildCtx.pool`, `RequestsPools` |
| B-N7 `hit` allows one adopted payload | accepted | `adopted: Sequence[tuple[int, str]]` |
| C-2 (major) the clock falls behind on pull-forward | accepted | D4: `loaded` sets `clock = max(clock, due)`, a no-op for gated loads |
| C-3 (major) 3b's p90 cannot see interference | accepted | 3b: median and p90 over gated steps only (≈ 180–240 gaps); the append step's ratio reported apart |
| C-4 `evict_payloads` is not rare | changed | KS3: check `num_free` before the list copy (no behaviour change), and skip staging pools when a tier is bound. +3 core lines |
| C-5 the null test can fail for the wrong reason | accepted | `prompt_end` only; `cached_tokens` asserted equal |
| C-6 `branch` never fires in the bounded check | accepted | §6: the added sessions start in a later wave; `branch_off_by_one` must fire |
| C-7 3h underspecified | accepted | 3h: the roomy arm specified (tiers off, ≈ 40 units, ≥ 150K tokens, ≈ 26 GiB); the host-adopt cost stated; the RTX 5090's 11.4 ms step |
| C-8 SSD lanes share one drive | accepted | D4: one shared NVMe lane at K-M4's mixed rate |
| C-9 when verification runs | accepted | D11: inside `attach`; startup time per GB in 3f |
| C-10 3c baseline context low | accepted | 3c: ≈ 9–10K at resume, ≈ 4 s; the ratio is unchanged |

**Budget, ABI, schedule.**
- Core: ≈ 298 lines itemized (KS2 +5 for waiterless pins, KS3 +3 for C-4, KS4 +5 for the API types), so the budget
  rises from ≤ +300 to **≤ +320** over K's step-1 base.
- ABI: unchanged, 1.4 or 1.5 by merge order. `Fence`, `Stream` and `TierCtx` join the same additive minor.
- Schedule: unchanged at 26 days (step 3 +1 for the `Level` protocol, step 5 −1).

### Rev 4 (2026-10-18): the panel's third round

Round 3 found A: 2 major; B: 2 minor; C: 1 minor. Each was checked against the code and rev 3's text:
- **A-1:** rev 3 fenced a host unit's *reuse* after its readers, but no rule ordered a device *read* of the unit
  after the D2H store that filled it.
- **A-2:** `_decode` calls `_fit`, which asks `could_fit` with the lower-ranked admitted requests as `releasing`
  (`sched.py:166-169, 177-185`). Rev 3 left pinned units out of that count, so a running decoder whose shortfall sat
  in a lower-ranked gated request's pinned load would preempt itself.
- **B-1, B-2, C-1:** confirmed in rev 3's §3, §4 and D4 text.

None is rejected.

| Finding | Disposition | Where / why |
|---|---|---|
| A-1 (major) a read of a host unit is not ordered after the store that wrote it | accepted | D4 "Reads after writes": the tier keeps each host unit's last write fence, adds it to the `after` of every H2D reading the unit, and folds it into the `Load` fence for host payloads a kind's adopt reads. §4 contract updated; plant `load_before_store` (deferred D2H stream, eviction then promotion within 1–2 steps, a sibling adopt). Tier-side only: no core lines |
| A-2 (major) load pins make the top-ranked running request preempt itself | accepted | D6: releasable pins (waiters all gone, or all in `releasing`) count as freeable in `could_fit` and as eligible in eviction. Eviction host-waits the load, calls `loaded(load, done = completion)`, unpins and demotes as a pure free. Decisions read schedule state only. Added to the property test and to `test_state_liveness.py` (no self-preemption while a lower-ranked gated request holds the shortfall); plant `pin_self_preempt`. KS2 +8 lines |
| B-1 (minor) the `Level` argument types are std names | accepted | D3: the protocol's docstring defines `ctx` (`TierCtx`), the host units (per-unit writable `memoryview`s, one segment each), a ticket (`done()`, `wait()`) and the rates (a plain mapping with an optional lane name) structurally |
| B-2 (minor) two ABI fields have no reader | accepted | `EntryView.children` and `TierCtx.namespace` are dropped; they come back with the ARC observer minor if a reader appears |
| C-1 (minor) the shared lane counts sharing twice | accepted | D4: reads are charged at `read_rate` and writes at `write_rate` on the one lane, so serialization alone models the sharing. K-M4's mixed rate is a check, with the owner told if the modelled mix is off by more than 15%. This supersedes rev 3's C-8 wording |

**Budget, ABI, schedule.**
- Core: ≈ 306 lines itemized (KS2 85 → 93), within the unchanged **≤ +320** over K's step-1 base.
- ABI: unchanged, 1.4 or 1.5 by merge order; it has two fewer fields than rev 3.
- Schedule: unchanged at 26 days. The two new plants and the liveness case fit in steps 1–3.
- Acceptance 5's cross-reference to FN now matches FN rev 2 (32c99f2): seams C1–C3, ≤ +45 lines over its own base.

### Rev 5 (2026-10-18): the panel's fourth round (all minor)

| Finding | Disposition | Where / why |
|---|---|---|
| A and B: the `Level` host-unit contract is wrong for payloads (S2 payload host pools are segment-major, 48 segments per GDN checkpoint on the 27B) | accepted | D3: a host unit is a sequence of segment `memoryview`s, an iovec for `preadv` / `pwritev`. KV units in the unit-major layout have a one-element entry. K-M4 checks O_DIRECT alignment for every payload segment size |
| B: drop load pins (lead decision: the simpler design) | accepted after verification | D6, D7: no load pins. D4's free rule now covers loads: a unit freed while a copy writes it is reused only after the next launch waits on that copy. An evicted loading entry gets `loaded(load, done = the load's own fence)`, with no host wait. **Verified first** that every live waiter holds its own reference, through promotion (`_take(shared=True)`, `state.py:173-175`), a hit on a loading entry, promote-into-place (own units) and `_swap` (`state.py:355-364`). So a loading entry with a waiter is never eligible. Round 2's wedge: the aborted waiter's entries become index-only and are evicted as pure frees. Round 3's self-preemption: today's `could_fit` counts G's units once G is in `releasing`, so `_fit` preempts G. No scenario needs a pin, so none is kept. Removed: pins, waiterless and releasable pin paths, both host waits, plants `pin_wedge` and `pin_self_preempt`. Added: plants `load_reuse_unwaited` and `waiter_unref`. Kept: the liveness case that a running decoder never preempts itself for a lower-ranked gated request's shortfall |
| C: the host-wait stall at eviction | resolved by the above | No host wait at eviction. The only host wait left is a launch waiting on an SSD load whose H2D the level's I/O thread has not yet enqueued. The I/O thread enqueues the H2D on read completion, never the engine thread, and the wait is reported in `stats()` as `load_enqueue_wait_s` |

**Budget, ABI, schedule.**
- KS2: 93 → **77** lines (pins and both pin paths removed; the free rule's extension to loads +2).
- Core: ≈ 290 lines itemized, within the unchanged **≤ +320** over K's step-1 base.
- ABI: unchanged, 1.4 or 1.5 by merge order.
- Schedule: unchanged at 26 days.

### Rev 6 (2026-10-18): the panel's fifth round

B and C were clean on rev 5. A found two minor issues, both confirmed against rev 5's D7: `loaded` was once per load,
and an evicted loading entry's units waited on the load's fence even when the H2D was never enqueued.

| Finding | Disposition | Where / why |
|---|---|---|
| A: a later joiner could adopt a payload whose host copy the tier may already reuse | accepted (the lead's simpler fix) | D7 step 6: joining a wait set covers pages only. Every payload adopt has its own `load(…, payloads=[(key, spec)])` and its own `loaded(…, done)`. Plant `adopt_source_reused` gains a joining-adopter variant |
| A: evicting an abandoned SSD load stalls the next launch on its read (up to ≈ 0.5 s) | accepted | D6, D7: when every entry of a load is evicted before any step reads it, the core calls `loaded(load, done = None)`. The tier sets an abandoned flag under the lock its I/O thread takes before enqueuing the H2D, so either the H2D is never issued or its fence already exists. The remaining stalls (live loads not yet enqueued) are reported in `load_enqueue_wait_s`. The round-3 liveness case gains an SSD cell |

**Budget, ABI, schedule.** Core ≈ 292 lines (KS2 77 → 79 for the abandon call and per-adopt payload loads), within
the unchanged ≤ +320. ABI unchanged (1.4 or 1.5 by merge order); `loaded`'s `done` becomes optional. Schedule
unchanged at 26 days.

### Rev 7 (2026-10-18): the panel's sixth round

B was clean on rev 6. A and C each found one minor issue, both confirmed against rev 6's D7. Rev 6's "exactly once"
`loaded` blocked abandonment after a waiterless call. Rev 6 also left open whether a late joiner's payload load reads
the SSD again.

| Finding | Disposition | Where / why |
|---|---|---|
| A: abandonment cannot follow a waiterless `loaded` (abort and eviction in different rounds reopen the stall of up to ≈ 0.5 s) | accepted (the lead's smallest-surface fix) | D6, D7, §4: `loaded` takes a fence at most once and `None` at most once, so `None` may follow a waiterless `loaded(load, fence)`. The tier sets the abandoned flag under the same enqueue lock. The waiterless call stays a notification that never waits. The round-3 liveness case gains the cell "abort, evict two rounds later, SSD read delayed", with `load_enqueue_wait_s` = 0 |
| C: a late joiner's payload load must not issue a second SSD read, nor decide that on timing | accepted | D7 step 6: the payload is on host or in transit from the moment the first load is decided. The joiner reuses the read ticket, is charged nothing new on the lane, gets `due = max(first read's due, now)`, and its fence chains on the first read. Abandonment never cancels a level read. The joining variant of `adopt_source_reused` counts SSD reads: one per payload |

**Budget, ABI, schedule.**
- Core: ≈ 292 lines, unchanged. Both rules live in the tier, except one condition in the core's abandon call.
- Budget: within ≤ +320.
- ABI: unchanged (1.4 or 1.5 by merge order); `loaded`'s call rule is restated in its docstring.
- Schedule: unchanged at 26 days.

### Rev 8.1 (2026-10-18): step-0 findings

Step 0 measured K-M1 (RTX 5090), K-M2/K-M3/K-M4 (RTX 5080; the RTX A5000 as a dev reference) and K-M5 (CPU replay of
the step-1 core). K-M4's mixed-I/O check fired one of step 0's stop rules; the lead decided, and the owner was told:

| Finding | Decision |
|---|---|
| K-M4: mixed reads and writes on one drive at 0.84× (50/50) / 0.76× (75/25) of the modelled shared lane | D4: the lane charges the measured mixed factor while both kinds are queued (perf-only, tier-side) |
| K-M5: SSD write-back of `answer` checkpoints is 94% of 73–88 GB written per `agent_park` run, mostly never read back | D11: tentative payloads are never written back; they drop at host eviction |
| K-M5 finding 2: a kept tentative stays flagged after its request is released | the core makes it ordinary at release and tells the tier (`StateTier.settled`, KS3) |
| K-M3: one batch-DMA load in flight slows decode 1.42×; the SM gather 1.07× | D10: `state_copy@1` loads by the SM gather by default; the 27B locks pin it at step 4 |
| K-M5: `answer:<k>` floods today's 4-unit pool without a tier; 16 GiB of host and 5 staging units suffice | `answer:<k>` needs a bound tier: the engine refuses a kind with `keep_tentative` and no tier. The 27B locks' defaults: `host.bytes` 16 GiB, staging 5 units |
| K-M1: 38 of 38 answers re-rendered verbatim | `answer` on by default (k = 1 suffices; 2 kept for templates that re-serialize) |
| K-M5 finding 1 (= the step-1 review's B2): the core could load a copy the tier had dropped in a hook or during another load | the core polls `dropped()` at each commit, after each store and after each load, and re-runs the hit's fixed point after its loads (KS2) |

Results against the predictions are in the evidence records (`evidence/k0_m1_rerender_rtx-5090_2026-10-18.json`,
`k0_m2_copy_*`, `k0_m4_nvme_*`, `k0_m5_replay_cpu_2026-10-18.json`) and the spikes' READMEs (`spikes/k0_tiers/`). The
RTX 5090's K-M2/K-M4 runs, after step 1, match the RTX 5080's (mixed I/O 0.79× / 0.76×, the other rules pass), and its
shorter step makes an unpaced batch-DMA load costlier still: decode 2.07× median beside it, 1.04× beside the SM gather.

### Rev 8.5 (2026-10-19): the panel's review of step 3

| Finding | Disposition | Where |
|---|---|---|
| A1 (major): an urgent write-back could hang: `SsdLevel.write` waited for slot space behind a queued write that never got a thread | fixed | the tier chooses a write's level slots when it decides it (`slots` per pool, from its own free lists) and passes them; a write into a slot waits, on its I/O thread, for the last job on the copy that held it (a dependency the dispatcher moves ahead), so the level never waits for room. `test_a_write_into_a_full_level_returns_at_once` hangs on the old level. The `Level` text changed: the tag moved |
| A2: a second load of a copy whose read-back is in flight waited for it on the engine thread | fixed | a load of a copy a level read wrote (done or not, decided without timing) takes the level path: its H2D is enqueued on an I/O thread after that read, due no earlier than it (`test_a_second_load_of_a_copy_being_read_back_waits_on_an_io_thread` fails on the old tier) |
| A3: restore was quadratic | fixed | the tier builds each pool's free slots once, all slots minus the restored copies' |
| A4: `free_on_ticket` must bite (deferred I/O on the CPU, a seeded lag of 0–3 launches, the host page pool at its floor, run-identity across seeds) | built; it does not bite | the judge's CPU streams defer a level job's completion 0–3 launches (seeded; at once for whatever waits for it). The plant fires 240–267 times and changes nothing, also with a larger watermark and level reads charged like the H2D lane: planted runs move copies between levels (99 level reads against 97) identically across seeds, and where a copy sits reaches neither the schedule nor the outputs here (superseded: "After the tag"). A correct tier stays run-identical under the lag (`test_the_level_passes_run_identity_without_a_watermark_and_with_late_io`). Step 4 tries the GPU with I/O sleeps of 0 or ≥ 2 steps |
| A5: no engine-level checkpoint read-back from the level | fixed | `test_branch_checkpoints_come_back_from_the_level_bitwise`: the tiny hybrid, `branch`, the payload host level at its floor, eight branched prefixes resumed in turn: outputs and hits equal a roomy host level's, `level_payload_reads > 0`. Acceptance 2: hybrids with `prompt_end` / `answer` through a level are judged by run-identity and `bounded` |
| C1 (lead decision on D4): reads beside background writes | done | D4 rev 8.5: `read_beside` (0.80) while `wlane > now`, the read pushing `wlane` back by its time; tested in `test_host_evictions_go_to_the_level_and_come_back_bitwise` |
| C2 (major): the attach page floor refused the 27B's host-only sizing | fixed | the page floor applies only with a level (loads read back into host units); payloads keep theirs |
| C3: I/O threads | done | default 3 |
| C4: crc and restore | done | crc only with persistence; restore verifies through the `O_DIRECT` descriptor into one aligned buffer per verifying thread |
| C5: `self.jobs` | done | it and the slots' last jobs keep unfinished jobs only (pruned at each launch) |
| C6: `adopt_source_reused` with a delayed I/O thread on the GPU | carried | §9 step 4 |
| B1: `ssd.io_threads` | renamed | `state_tier.io_threads` (planned = false); D3: a level's concurrency is the tier's thread count |
| B2: D3's `Level` text | fixed | D3's block and `level.py` match the protocol as built (blocking calls, slots, factors, `ctx` the `TierCtx`) |
| B3: `ssd.bytes` | planned | like `host.bytes`: it decides the hit set |
| B4: the held-out proof | noted | `policies.last_turn` is owed (step 5); zlib took `slots` with the moved tag (§1) |

Core is unchanged (+326 over b5b1e7f). On the A5000, Qwen3-8B through the SSD level on rev 8.5's tier: invariance and
the restart check pass as before (7,402 level reads; 338 blocks restored, none after a crash), and `read_unwaited`,
`stale_slot` and `torn_manifest` with the crc check off are caught (`evidence/k3_judge_tiers_ssd_rtx-a5000_2026-10-19.json`,
`gpu_qwen3-8b_ssd_rev85`). The full CI suite: 1638 passed, 66 skipped, 1 failed (the crc-off plant's patch missed
the verification's new argument; fixed, the level's judge cells re-run: 5 passed).

### Step 4: the panel's review (2026-10-19)

| Finding | Disposition | Where |
|---|---|---|
| A-1 + B-2: a checkpoint pool without a tier still fell to 0 units silently (the lead's decision) | fixed | a set `recurrent.checkpoints` is required (`min_units`: the ledger refuses the start); the unset default stays best effort, takes no checkpoints and cuts nothing at 0 units, and says so once per pool on stderr, at the first request (the kind learns its final units only after the engine's last materialize). Post-tag std follow-up ("After the tag"); `test_without_checkpoint_units_nothing_is_reused` |
| A-1 (doc) | done | the tier's staging refusal at `max_seqs` + 1 is a floor, not a guarantee against drops ("After the tag") |
| B-1: `tune --ssd-rates` used a 27B unit | fixed | its I/O unit is the tuned engine's largest page unit (the level's unit-major page), or `--ssd-unit`; 557,056 B appears only in the 27B's evidence and tests |
| C-1 (major): 3a's SSD cells beside decoders on lock 11's clock | rerun; **the 32K cell misses**; after the lead's bench change (no std change), 16K and 32K miss; stopped | below |
| C-2: the default read rate; tune's write not sustained | fixed | `RATES["read"]` 2.24e9, `write` stays 1.77e9 (post-tag std follow-up); tune writes for 60 s and takes the rate of the last 30 s |
| C-3: 3f on one resume | fixed | the median of 5 resumes per engine (`--resumes`) |
| C-4: the bounded KL p99 by wave | done; noise established | below; lock 11 stands |

**C-1.** 3a beside three decoders on lock 11 (its tuned clock, the SSD level's read rate now 2.24 GB/s), with each resume's
tier counts (`evidence/k4_tier_resume_rtx-5080_2026-10-19.json`, `panel_review`):

| length | VRAM | RAM (bar) | SSD (bar) | SSD gated steps | write-backs during the resume | `load_enqueue_wait_s` |
|---|---|---|---|---|---|---|
| 4K | 161.5 | 178.8 (188.3) | 325.6 (359.2) | 9 | 0 | 0 |
| 16K | 195.7 | 209.0 (236.0) | 641.2 (646.3), pass by 0.8% | 22 | 0 | 0 |
| 32K | 242.3 | 276.1 (300.6) | **1,050.2 (1,030.2): misses by 1.9%** | 39 | 0 | 0 |

The gated steps are the same in every rep, no write-back runs during a resume and the engine never waited for a
level read's enqueue: the reads land before the step that needs them, and the request waits on the clock alone. The
clock's arithmetic accounts for the 39 steps at 32K: the bench's `down()` (the parking's write-backs, drained before
the resume) charged 1.29 GB to the background write lane at 1.77 GB/s while the loop was blocked in the drain, so no
launch advanced the clock and that lane was still ahead of it (`wlane > now`) when the resume was decided. The read is
then charged as beside background writes: 1.29 GB / (2.24 GB/s × `read_beside` 0.80) = 0.72 s, plus 0.03 s of H2D,
over 18.8 ms per 3-row decode step as the tuned clock charges it = 40 steps (39 measured), each ≈ 20.4 ms of wall
time beside the decoders. With rev 8.5's read rate (2.55 GB/s) and the default clock (17.7 ms per such step) the same
arithmetic gives ≈ 37 steps, and that run passed (987.6 ms). Idle, nothing else runs, so the gated request runs at
once and its step waits for the copies themselves (the scheduler's rule), which the clock does not lengthen. So the miss is the clock's: a write lane charged while the loop was blocked stays busy in virtual time
after the real writes finished, and the read rate is 12% lower than before. In serving the loop keeps launching
during write-backs, so the lane drains with the clock; the bench blocks it. **For the lead:** whether the clock's
lanes should catch up on wall time when the loop has not launched (a std change), or the bench should keep stepping
the decoders while it parks, or both. The bar's SSD rate (the drive's 2.68 GB/s) is unchanged.

**C-1, the lead's decision and the rerun.** The lead ruled out a clock change (lanes catching up on wall time would
make decisions timing-dependent, against D4) and had the bench changed instead. **The procedure change was made after
seeing the miss above** (1,050.2 ms against 1,030.2 at 32K): `Bench.down()` now keeps stepping the decoders while its
write-backs run, as serving keeps launching, and drains only when no decoder runs (idle is unchanged). A unit test
pins the clock's charge as it is (`test_the_clock_lanes_move_with_launches_never_with_wall_time`: after the writes
finish in wall time, a read is still charged beside them until launches carry the clock past the write lane). Rerun
on lock 11, same counters (`evidence/k4_tier_resume_rtx-5080_2026-10-19.json`, `lead_decision`; clock screen before
it 2,715 / 2,722 MHz, steady):

| 3a beside decoders | SSD (bar) | gated steps | before the change |
|---|---|---|---|
| 4K | 345.4 (359.1) | 9 | 325.6, 9 |
| 16K | **653.3 (646.7): misses by 1.0%** | 22 | 641.2, 22 |
| 32K | **1,048.0 (1,030.7): misses by 1.7%** | 39 | 1,050.2, 39 |

The gated steps did not move: stepping the decoders only until the real writes finish does not carry the clock past
the write lane, because the clock charges the write-backs at 1.77 GB/s (the default sustained rate) and the drive
wrote them at ≈ 2.3–2.6 GB/s; ≈ 0.5 s of steps advances the clock ≈ 0.5 s against ≈ 0.73 s charged, so the read is
still charged beside writes. The first attribution (the blocked loop) was therefore only part of it: the rest is the
clock's conservative write rate, which serving shares. Per the lead's rule, stopped here: nothing else was changed.
3a idle on lock 11, to confirm: VRAM / RAM / SSD 124.6 / 124.9 / 239.4 ms at 4K, 157.3 / 167.2 / 449.8 at 16K and
206.0 / 228.5 / 797.8 at 32K, against the first idle run's 124.5 / 124.6 / 238.4, 157.4 / 167.5 / 451.2 and 206.1 /
228.9 / 740.1 (lock 10's default clock): unchanged but the SSD arm at 32K (+7.8%, bar 993.9). Idle the request is
gated one step and its step waits for the copies (32K: 0.48 s of enqueue wait), so that difference is the drive's,
not the clock's.

**C-3.** 3f on lock 11, five parked 16K conversations resumed by each engine: restarted median 450.4 ms (570.4, then
447.2–450.6), no restart 448.3 ms; `cached_tokens` and outputs identical for all five; bar 608.0 ms: pass.

**C-2.** `tune --ssd-rates` with the sustained write, on the RTX 5080's drive: read 2.20 GB/s; writes 3 deep held
2.35 GB/s over the first 30 s and 2.26 GB/s over the last 30 s (138 GB written, no cache cliff at this depth);
mixed 0.857, `read_beside` 0.872. The default write stays 1.77 GB/s (K-M4's sustained rate), as decided.

**C-4.** `bounded --policies`, KL p99 by wave (the late sessions, second and third waves, are the ones that meet the
system prompt's end as a branch point and adopt the `branch` checkpoints):

| sessions | arm | all | first wave (positions) | late (positions) |
|---|---|---|---|---|
| 8 | baseline | 0.0195 | 0.0166 (4,096) | 0.0259 (2,048) |
| 8 | `branch`, tier | 0.0432 (failed) | 0.0504 | 0.0337 |
| 8 | `answer:1`, tier | 0.0350 | 0.0371 | 0.0213 |
| 16 | baseline | 0.0236 | 0.0281 (8,192) | 0.0139 (4,096) |
| 16 | `branch`, tier | 0.0226 | 0.0273 | 0.0136 |
| 16 | `answer:1`, tier | 0.0310 | 0.0455 | 0.0145 |

The late sessions are in line in both runs (8: 1.30× the baseline's late p99, inside the rule's 2×; 16: 0.98×). The
8-session excess sits in the first-wave sessions, which never adopt a `branch` checkpoint, and is gone at 16. Noise,
as established; lock 11 stands and the fired-row analysis was not needed.

Core is unchanged. The full CI suite with these fixes: 1643 passed, 66 skipped; `run_all.sh` clean. With the bench
change and its test: 1644 passed, 66 skipped.

### Step 4 (2026-10-19): the gate

Same-session interleaved runs (the MoE prefill track's §5.2 method): 3a and 3b interleave their arms per rep in one
engine; the served rows (3c–3e, 3h, 3i) are `bench/run.py` ABBA rounds of fresh worker processes, each round's
paired request ratios reduced to a geomean and the rounds to their median, with pooled and per-round percentiles
beside. Clocks: the RTX 5080 screen before the runs (2,715 / 2,722 MHz p10 / p50, steady) and per 3a/3b row
(SM p50 2,610–2,820 MHz on every row but those an SSD read idled, below); the RTX 5090 screen after its runs (2,302 /
2,340 MHz, power-capped and steady). The bars' rates are §8's: 47.5 GB/s link, 2.68 GB/s SSD, one decode step 22.3 ms
(RTX 5080) / 14.7 ms (RTX 5090).

Harness added in step 4 (bench, tools and judge; std and core unchanged, so nothing here is a post-tag std
follow-up): `bench/tier_resume.py` (3a, 3b, 3b-SSD, 3f); the `claude_code_2turns` trace (3e); the Holon bench
backend records `dropped_publishes` and the tier's counts; `tune --clock` and `tune --ssd-rates` (below); `tier_resume.py --blocking-sync` (3b's engine-loop time) and 3f's
restart timed in parts; the HTTP bench backend fails a request the server errors; `holon_judge tiers --io-sleep`
(level I/O jobs sleep 0 or the given seconds, even odds) and the `store_no_after` plant; run-identity compares the
tier's counts (below). Evidence: `evidence/k4_tier_resume_rtx-5080_2026-10-19.json` (3a, 3b, 3f, the clock),
`evidence/k4_agent_park_rtx-5080_2026-10-19.json` (3c, 3d), `evidence/bench_k4-3e-claude-code_rtx-5080_2026-10-19.json`,
`evidence/bench_k4-3h-agent-park_rtx-5090_2026-10-19.json`, `evidence/bench_k4-3i-offload-agent-park_rtx-5080_2026-10-19.json`,
`evidence/bench_k4-3d-agent-park_rtx-5080_2026-10-19.json` (3d sized), `evidence/k4_judge_tiers_rtx-a5000_2026-10-19.json` (plants), `evidence/k4_judge_tiers_rtx-5090_2026-10-19.json` (the
27B pair), `evidence/k4_locks_2026-10-19.json` (the held locks and their checks).

**The bar table** (Qwen3.8-27B UD-IQ3_S on the RTX 5080 at lock 10's settings plus the tier: `host.bytes` 16 GiB,
staging 5, `prompt_end,branch,answer:1`, the tuned clock; UD-Q4_K_M on the RTX 5090; medians):

| # | Measured | Bar | Verdict |
|---|---|---|---|
| 3a idle, 4K | VRAM 124.5, RAM 124.6, SSD 238.4 ms | RAM ≤ 151.3, SSD ≤ 322.1 | pass |
| 3a idle, 16K | VRAM 157.4, RAM 167.5, SSD 451.2 ms | RAM ≤ 197.7, SSD ≤ 608.0 | pass |
| 3a idle, 32K | VRAM 206.1, RAM 228.9, SSD 740.1 ms | RAM ≤ 264.4, SSD ≤ 994.0 | pass |
| 3a beside 3 decoders, 4K | VRAM 161.4, RAM 178.4, SSD 325.1 ms | RAM ≤ 188.2, SSD ≤ 359.1 | pass |
| 3a beside, 16K | VRAM 195.5, RAM 208.4, SSD 620.4 ms | RAM ≤ 235.9, SSD ≤ 646.2 | pass (SSD by 4%) |
| 3a beside, 32K | VRAM 242.7, RAM 275.8, SSD 987.6 ms | RAM ≤ 301.1, SSD ≤ 1,030.6 | pass (SSD by 4%) |
| 3a beside, rerun on lock 11 (panel C-1) | 4K SSD 325.6, 16K 641.2, 32K **1,050.2** ms | SSD ≤ 359.2 / 646.3 / 1,030.2 | **32K misses** ("the panel's review") |
| 3a beside, lock 11, the loop stepping while parking (the lead's bench change, made after the miss) | 4K SSD 345.4, 16K **653.3**, 32K **1,048.0** ms | SSD ≤ 359.1 / 646.7 / 1,030.7 | **16K and 32K miss** ("the panel's review") |
| 3b (32K from RAM, 30 resumes, 285 gated gaps) | gated median 20.27 vs 20.27 ms (1.000×), p90 23.58 vs 23.07 (1.022×); append step 1.001× | ≤ 1.1×, p90 ≤ 1.2× | pass |
| 3b-SSD (report) | 1,020 gated gaps (34 per resume): 1.000× median, 1.000× p90; append 1.001× | reported | — |
| 3c (2 rounds) | tier 265 ms p50 against VRAM-only 5,597 ms: **0.047×** (geomean of pairs 0.059×; turns 1–5 0.078×, 6+ 0.045×); throughput **2.26×** (124.0 vs 54.9 tok/s, both rounds) | TTFT p50 ≤ 0.25×, throughput ≥ 1.2× | pass |
| 3d (3 rounds, the lead's sizing) | against `llama-server --cache-ram 16384 --ctx-checkpoints 32 -c 78848` (upstream `27c7a769d`; 4 slots × the trace's longest request, 19,712 tokens, of unified KV: 1.5× Holon's 52,304, conservative for Holon): p50 261 vs 2,134 ms (**0.122×**; geomean of pairs 0.134×), p90 0.088×, throughput **1.79×** (124.0 vs 69.2 tok/s); no request failed in any round. At the matched KV (52,304) the reference fails (below) | p50, p90 ≤ 1.0×, throughput ≥ 1.0× | pass |
| 3e (3 rounds, temperature 1.0) | each session's first request: `branch` 141 ms p50 against 8,110 ms without (**0.017×**; geomean of pairs 0.041×; p90 0.71×: sessions 1–2 compute the 16K prompt in both arms) | p50 ≤ 0.2× | pass |
| 3f (16K, `persist = close`) | restarted 575.5 ms (no restart 447.8); outputs and `cached_tokens` identical; 2,560 blocks and 20 payloads restored | ≤ 608.0 ms | pass |
| 3h (RTX 5090) | tier, VRAM capped to the RTX 5080's state bytes: p50 159.0 ms against roomy 274.9 ms; with roomy's own policies (`prompt_end,branch`, the overhead measure) 265.2 ms (0.963×; geomean of pairs 0.742×) | ≤ 1.1 × 274.9 + 14.7 = 317.1 ms | pass |
| 3i (report, 2 rounds) | Qwen3-30B-A3B Q4_K_M offload (the natural-fit lock 2, `max_seqs` 2 so a resume's loads run beside the other sequence's decode and expert fetches) with an 8 GiB host level against without: decode TPOT p50 11.0 vs 11.8 ms, p90 15.4 vs 17.7 ms; TTFT p50 2,609 vs 4,056 ms; throughput 1.17×; 100 loads (52.7 GB) per run. Host memory 24 GiB for the experts' pinned pools, 32 GiB with the level | reported | — |

Notes on the rows:
- 3a ran on the default clock (`step_cost` 0.015, 0.0004, 0.0005), before the tune; 3b, 3f and the served rows on
  the tuned one.
- `dropped_publishes` is 0 on `agent_park` at staging 5 (3c's tiered arm: 2,138 payload stores, 496 payloads evicted
  from host at the default `host.pages` 0.25) and on 3e's arms.
- 3c ran 2 rounds, both with the tiered arm first: the first session's later rounds were lost to the llama-server
  hang (below); the two rounds agree to 1% and the margin is 5×. The tiered arm repeats in 3d's three rounds (p50
  261–264 ms, 124.0 tok/s).
- 3a: every load fully fragmented (a run per unit: 256 / 1,024 / 2,048 runs). The bars are on medians; the SSD arm's
  p90 is over its bar beside decoders (874 ms at 16K, 1,350 ms at 32K) and idle at 32K (1,068 ms): single slow reads
  of the drive. Rows whose SSD read idled the GPU sampled 180–1,537 MHz (an idle card, not a throttle).
- 3b-SSD's engine-loop host time (rerun with CUDA waits sleeping, the engine thread's `time.thread_time`): 0.56 ms
  per gated step (p90 0.65) against 0.54 ms per plain decode step: the resume adds ≈ 0.02 ms to the loop. The
  process's CPU time per step (every thread, the I/O threads included) was 6.9 ms; the first record's 26–34 ms per
  step counted CUDA spin-waits and is superseded. The rerun's gaps repeat the first (1,020 gaps, 0.999× median,
  1.000× p90).
- 3f's restart timed in parts (rerun: restarted 565.6 ms, `cached_tokens` and outputs identical): the restore (the
  manifest and every persisted unit's crc, read back with `O_DIRECT` on the level's threads) took 3.09 s for 7.24 GB
  persisted, **0.43 s per GB**, inside the tier's attach (4.68 s, with the 16 GiB host pools' pinning), inside an
  engine build of 11.7 s.
- 3f's first run missed (7,239 ms, `cached_tokens` 0) because of the bench, not the tier: engines A, B and B2 ran in
  one process and A's device memory was not freed, so B measured ≈ 160 MB less, and its staging pools fell to 0 units
  (below). B stored no payloads, B2 restored pages only, and the hybrid's hit trimmed to 0. The bench now frees each
  engine and records every engine's pools (`device_pools`); the rerun above has 5 staging units in all three.
- 3h: the roomy arm runs tiers off with 40 checkpoint units and 215,056 KV tokens (q8_0, as §8 sizes it); the
  capped arms run the tier with exactly the RTX 5080's state bytes (3,341,516,800: 51,760 KV tokens, 5 staging units).
  The gate arm (`answer:1`) is faster than roomy because roomy cannot keep `answer` checkpoints without a tier.
  The first RTX 5090 desktop lost its GPU mid-run (`unspecified launch failure`, then no device); the cell was rerun
  on another one.

**The tier's clock** (`tune --clock`, new): `state_tier.step_cost` fitted to timed steps of the lock's engine
(prefill of one row at 256–2,048 tokens, decode at 1–4 rows) in relative error. RTX 5080 (IQ3_S, q8_0):
a, b, c = 13.7 ms, 0.366 ms/token, 1.33 ms/row (every sample within 5%); RTX 5090 (Q4_K_M, q8_0, 3h):
10.7 ms, 0.217 ms/token, 0.435 ms/row; RTX 5090 (bf16, the lock): 11.1 ms, 0.198, 0.382. A plain least-squares fit
let the prefill steps decide and missed 1-row decode by 37% (a = 0); `tools/tests/test_tune.py` holds the RTX 5080's
samples.

**The SSD level's rates** (`tune --ssd-rates DIR`, new: through the tier's `IoThreads` over its `SsdLevel`, 3 threads,
`O_DIRECT` units of the tuned engine's largest page unit, or `--ssd-unit`: 557,056 B for the 27B at q8_0; taken into
the lock only when its flags bind the level): on the RTX
5080 desktop's drive, twice: read 2.24 / 2.24 GB/s, write 2.57 / 2.60 GB/s (3 deep for 12 s; one at a time 2.73),
mixed 0.776 / 0.773, `read_beside` 0.841 / 0.845. Against the defaults (2.55, 1.77, 0.74, 0.80): reads 12% slower,
writes 45% faster here (the default is K-M4's one-deep sustained rate), the factors within 5%. A first run measured
reads right after a 31.7 GB write burst (1.84 GB/s, `read_beside` 1.02); tune now reads first and that run is void.
The 27B locks bind no SSD level, so no lock takes these.

**The 27B agent locks** (acceptance 6):
- RTX 5090, UD-Q4_K_M: `5.lock` → `6.lock` by `tune --clock --approve` over lock 5's settings plus the tier
  (`slots.tier = std.state_tier`, `host.bytes` 16 GiB, staging 5, `prompt_end,branch,answer:1`, the step cost; the
  copies' sites bound to `kernels.state_copy`, whose loads are the SM gather). Verdict approve: the snapshot trace
  byte-identical to lock 5's, L2 passing (+0.38% ± 0.69% ΔPPL vs floor); attention stays `kernels.triton_attn`
  (287 vs 194 tok/s). Snapshot ×3 run-identical; `bounded --policies` passes (baseline, `branch` + tier, `answer:1`
  + tier).
- RTX 5080, UD-IQ3_S: `10.lock` → `11.lock` the same way (step cost 13.7 ms, 0.366 ms/token, 1.34 ms/row). Verdict
  approve: snapshot trace byte-identical to lock 10's, L2 passing (+0.67% ± 0.46% ΔPPL overall); snapshot ×3
  run-identical. `bounded --policies` at 8 sessions failed for `branch` (KL p99 0.0432 ± 0.016 against the baseline's
  0.0195 ± 0.004); rerun at 16 sessions (the lead's ≥ 2×) every arm passes: KL p99 baseline 0.0236 (± 0.0049),
  `branch` + tier 0.0226 (± 0.0047), `answer:1` + tier 0.0310 (± 0.0092). The 8-session failure was the p99's
  noise at 6,144 positions; the fired-row analysis the lead asked for if it still failed was not needed.
- Both locks are committed (`plans/…/rtx-5080/cu130-torch2.11/11.lock`, `plans/…/rtx-5090/cu130-torch2.11/6.lock`).
  They pin the std sources they were tuned on (before the post-tag follow-ups below, which change neither
  configuration's behaviour: staging 5 meets the demand, and checkpoints are asked for); the close's propagation
  carries them with every other lock (D14).
- Both need `--host-memory` (≥ 17,179,869,184 bytes) at launch: host memory is a deployment's, as for X's offload
  lock (the lead: accepted). A lock has no notes field; the tier's refusal names the flag and the bytes, and the
  README says so.

**The 27B tier-invariance pair** (RTX 5090, UD-Q4_K_M, host + SSD level, payload host pools at their floor, I/O
sleeps 0 or 0.1 s; `evidence/k4_judge_tiers_rtx-5090_2026-10-19.json`): with a tight device pool (2.008 GB)
invariance, run-identity (56 loads, 92 level reads per run) and the restart check (106 blocks restored) pass; the
tiers-off null test (4 vs 16 checkpoint units) passes on a roomy pool (4.6 GB), where invariance is refused by its
own rule (the larger pool keeps other checkpoints in the reference, so `cached_tokens` differ).

**The GPU timing plants** (RTX A5000, `evidence/k4_judge_tiers_rtx-a5000_2026-10-19.json`):
- `store_no_after` (new: a D2H store without its `after`): caught (fired 498, changed 194; Qwen3-8B through the SSD
  level).
- `free_on_ticket` with I/O jobs sleeping 0 or 0.1 s (≥ 4 steps): on Qwen3-8B the first run fired 1,184 times with
  steps and outputs identical, but its two runs' tier counts differed (614 against 570 write-backs, 687 against 664
  level reads): the window opened, and run-identity did not compare where copies sit. **Run-identity now compares the
  tier's counts** (all but wall times; clean runs were count-identical in every run here). With the rule, 2 and then 4
  more planted runs did not reproduce the divergence (fired 1,228 and 2,456 times, counts identical). On the Qwen3.5-4B
  hybrid with the payload host pools at their floor it crashed on the tier's internal check (`no host unit for a
  payload … every copy loading`), as on the CPU. So: caught once by timing on the GPU (not reproducibly), and at the
  floors by the tier's check.
- `adopt_source_reused` (Qwen3.5-4B, payload floor, `answer:1`, identity and fork loops, delayed I/O): fired 288
  times, changed nothing. Its join variant: caught (changed 4). The main variant's guard (a reading step's `done`
  kept for reuse) stays covered by the tier-only tests (rev 8.4); the GPU did not open its window here.

**Found in step 4:**
- **Staging pools fell silently to 0 units under memory pressure.** The recurrent kind's checkpoint pool is a fixed
  pool (`min_units` 0, `pref_units` = `recurrent.checkpoints`); the ledger grants the preference only if every other
  pool keeps its minimum, else the minimum, 0. Lock 10's settings plus the tier sit ≈ 90 MB above that cliff on the
  RTX 5080 (3.34 GB state budget against 3.25 GB for staging 5 beside the paged floor); 160 MB less free memory gave
  0 staging units, no payload stores, `dropped_publishes` 0, and every hybrid hit trimmed to 0, with no error
  (3f's first run; the 27B pair's 16-unit null run the same way). Fixed as post-tag std follow-ups ("After the
  tag"): with a tier the attach refuses a staging pool below `max_seqs` + 1 (a floor, not a guarantee against
  drops); without one, a set `recurrent.checkpoints` is required at start and the unset default warns at 0 units.
- `bench/driver.py` refused traces where every session has a client (`agent_park`: `clients.pop(None)`); fixed.
- `holon_judge tiers`' JSON report failed on the null test's pools keyed by units; fixed (the clean-checks test
  asserts it).
- **3d's reference cannot serve `agent_park` at the matched KV.** `llama-server` at `-c` = Holon's KV tokens
  (52,304, unified, 4 slots; contexts reach ≈ 19.6K, so four of them exceed the pool) fails requests with
  `Context size has been exceeded` (6 per run). It happened in the default 8 GiB arm of the first session and in the
  `--cache-ram 16384` arm of the second; the first session's 16 GiB arm finished clean (p50 0.134×, p90 0.093×,
  throughput 1.78×). Holon at the same KV preempts instead. Kept as a reported failure of the reference; 3d is gated
  on the sized arm (the lead, item 4). The bench's HTTP client now fails a request the server errors or leaves
  unfinished instead of waiting forever (checked against a stub server: both cases raise).

### After the tag (b596ba7, `v1-kv-tiers-api`)

The tag stays at b596ba7 from now on (the lead, on reviewer B's rule). A std fix after it is a **post-tag std
follow-up**, recorded here as X recorded its own, and never moves the tag. `policies.last_turn` lands strictly after
b596ba7 as a plugin-only diff, verified with `scripts/ci/check_core_diff.sh` and `git diff --stat` (step 5).

| Follow-up | Change |
|---|---|
| C (minor, rev 8.5's review): the SSD level's default rates were 2-thread values with 3 threads the default | one 3-thread set: read 2.55 GB/s (K-M4, O_DIRECT, 3 threads), write 1.77 GB/s (sustained; K-M4's 3-thread window was 2.20, as at 1, 2 and 4 threads), mixed 0.74 (step 3 at 3 threads), `read_beside` 0.80; D4: tune measures them per host |
| A (rev 8.5's review): the slot-ordering guard (a write into a level slot waits for the last job of the copy that held it) had no test that bites | `test_a_write_into_a_level_slot_waits_for_its_last_copy`: a level storing by (shape, slot) at its floor (one page slot), two I/O threads; the watermark writes X back, its write held, then writes Y back into X's slot (X evicted at once); released, Y reads back bitwise and the level saw X's write first. Y is a background write-back here; the urgent path (a read's dependent write ordered after the queued write it needs) is covered together with `test_a_read_reusing_units_a_queued_write_frees_takes_that_write_ahead`. The plant `slot-reuse-unwaited` (the guard dropped) fails it; deleting the line fails the clean cell |
| Found by that test: `IoThreads.drain` waited for queued jobs only, not running ones | it waits for every job taken and not done (std, post-tag; `close` was unaffected: it joins the threads) |
| A (rev 8.5's review): `free_on_ticket` had no configuration that bites; the rationale given ("never needed before the next commit") was false: the watermark frees units for the same commit's stores | the judge's lagged I/O also completes deferred jobs at each commit's start and each `Engine.step` on a seeded coin per job. On the tiny hybrid with the page and payload host pools **at their floors**, `host.reserve` one load, run-identity across 3 seeds: the plant fires 40 times and **bites as a refusal**: the units it holds back are the floor's, so a read-back finds no host unit (`no host units to read … back`), a crash, which the judge's rule does not count as caught (a crash is not an outcome). **One or more units over the floors** (1 page + 1 payload unit, 4 + 2): it fires 120 times and the three seeds' runs are step- and output-identical, to each other and to the clean run: no decision in that trace needed the units it held back by the time it decided. **Recorded (the lead, on A):** caught at the floors by the tier's internal check (`RuntimeError` from `_read_back`); above the floors the window did not open on this trace; the guard (host units free when the write is decided) stays required, it is not redundant. Step 4 tries the GPU with sleeps of 0 or ≥ 2 steps |
| The lead (step 4, item 2): a payload staging pool that the ledger shrank to 0 units dropped every hybrid hit with `dropped_publishes` 0 | `std.state_tier` refuses at attach when a payload staging pool has fewer units than `max_seqs` + 1 (§7's formula with `p` = 1: the tier sees neither the overlap nor the step budget; the 27B locks' staging 5 meets it), naming each pool's shortfall in bytes (`test_attach_refuses_a_payload_level_below_the_payloads_in_flight`). **That refusal is a floor, not a guarantee against drops:** a step that publishes for more rows than the pool has free units still drops (counted in `dropped_publishes`), and `max_seqs` + 1 does not cover two prefill rows publishing in consecutive steps. Two judge tests that ran a tier at staging = `max_seqs` now give it `max_seqs` + 1. (Its first version also made `std.state.recurrent` cut for and publish into a 0-unit pool, to count the drops; replaced by the panel's A-1 + B-2 row below.) |
| The panel (step 4, A-1 + B-2; the lead's decision): a checkpoint pool without a tier still fell to 0 units silently | `std.state.recurrent` requests a set `recurrent.checkpoints` as `min_units` too, so the ledger refuses a start that cannot hold them (as pinned pools are refused); left unset, the default (`max_seqs` units) stays best effort as before step 4: 0 units means no cuts and no publishes, and the kind says so once per checkpoint pool on stderr (`holon: plugin.….checkpoints got 0 units …`). The warning comes at the first request: the kind learns its final units only after the engine's last materialize, which no hook a kind sees marks (the start-up profile's materialize gives every best-effort pool 0). `test_without_checkpoint_units_nothing_is_reused`: set and short is refused; `0` and unset run without reuse, with the warning only for unset; it fails on the pre-step-4 kind (no refusal, no warning) |
| The panel (step 4, C-2): the default read rate | `ssd.RATES["read"]` 2.24e9 (step 4's `tune --ssd-rates` on the RTX 5080's drive; was 2.55e9); `write` stays 1.77e9 (sustained). Tune's write window is now sustained: 60 s of writes, the rate over the last 30 s (past the drive's write cache) |
| The lead (step 4, item 3): `--host-memory` at launch, as X's offload lock | a tier refused host memory names the flag and the bytes its pools take (`launch with --host-memory N`; tested in the attach test above); README "Options". A lock has no notes field: the refusal and the README carry it |

Step 3 closed panel-clean (A, B and C, 2026-10-19). The full CI suite at its close: 1641 passed, 66 skipped.

### Step 3 as built (2026-10-19)

- **The `Level` protocol** (`std/holon_std/state_tier/level.py`; D3 "as built"). Deviations from rev 8's sketch, all
  std's: the tier owns the I/O threads, so `write`, `read` and `drop` are blocking calls on them, not tickets; `bind`
  gets each host pool's share of the level's bytes; `src` / `dst` map a pool to the copy's units of it; restored
  payloads carry their unit count; **K binds one level below host** (`state_tier.levels` takes one name or `Ref`).
- **The SSD level** (`ssd.py`, built-in `ssd`). One preallocated slot file per host pool shape, slots 4 KiB-aligned;
  `flock` on the directory (a second engine is refused), a directory without the bytes refused; `O_DIRECT` for units
  whose segments and offset are aligned, else buffered with `DONTNEED`; a crc32 per written unit. `persist = close`:
  `fdatasync`, then the manifest written, `fsync`ed, renamed, the directory `fsync`ed; at bind the manifest is read and
  unlinked (directory `fsync`ed) before any slot can be rewritten; restore keeps a record only with the same identity,
  every slot inside the current geometry and holding its crc (verified on threads), and its parent kept. Persistence
  needs the state identity (refused without it).
- **The tier over a level** (`tier.py`). Host evictions write back (the copy is on the level from that call; its host
  units free at once, their reuse waiting for the write), except tentative payloads and superseded ones (rev 8.1,
  8.2), which are dropped; host victims are leaves of the host level; a copy on both levels leaves the host level
  without a write; the level evicts redundant copies first, then least recently used leaves (a child copy on either
  level keeps a page copy). The watermark (`host.reserve`, default one `max_model_len` load's units) writes host
  leaves back in `before_launch`. A load of a level copy reads it into host units (one read per copy; later loads of
  it wait for that read); its H2D is enqueued by an I/O thread once its reads are done, in decision order, unless it
  was abandoned (abandonment completes the enqueue wait; `load_enqueue_wait_s`). Page host pools are unit-major with a
  level (D10). Host-unit fences gain two ordered sources: the latest write-back from the unit (`io`) and the latest
  level load that read it no step waited for (`level`, its own fence: enqueues keep decision order). I/O tickets
  among a copy's `after` are waited for on the host (a stream cannot, and off CUDA waits for nothing).
- **I/O and read priority** (`iothreads.py`; D4, D11 rev 8.3). Jobs run in decision order on `ssd.io_threads`; while
  a read is queued no write starts but one a queued job depends on: a job submitted to the read queue takes the
  queued writes it depends on (transitively) ahead of it. Level drops run once the copy's last job has. The clock
  charges level reads at the read rate on the level's lane, behind the writes they depend on (at the write rate over
  the mixed factor); other write-backs go on their own lane at the write rate.
- **Found while building (GPU):** a read reusing host units freed by an earlier *background* write-back depended on
  that write, which could not start while reads were queued; with both I/O threads holding such jobs the engine
  waited forever for an enqueue (Qwen3-8B, the first invariance run). Only the writes freeing units inside the
  read's own eviction had been moved ahead. Fixed in the dispatcher (every queued write a read-queue job depends on,
  transitively), with a failing-without test (`test_a_read_reusing_units_a_queued_write_frees_takes_that_write_ahead`).
- **`evict.root_pin`** (`plugins/holon_plugins/evict/root_pin/`, slot `eviction_policy`), options `root_pin.depth`
  and `root_pin.hits`; its test binds it through an engine.
- **The mixed factor under exactly this dispatch** (`spikes/k3_tiers/`, `evidence/k3_mixed_nvme_rtx-5080_2026-10-19.json`;
  the RTX 5080 desktop's NVMe, outside the container: its `/tmp` is tmpfs, so a first run there was void and is not
  recorded). Predicted 0.85–0.95, measured **0.79–0.82 with 2 threads and 0.73–0.74 with 3**: no better than step 2's
  0.77–0.80. Reads beside background writes ran at 0.79–0.92 of the pure read rate for their share (predicted within
  10%). A comparison arm that starts no write while a read is queued *or running* did worse (0.62–0.77, reads
  0.67–0.80): the drive's own read/write interference, not the dispatch. The literal rule only bounds writes while a
  read waits; with reads always in flight it rarely does (the 75/25 target reached 0.51). D4's "reads at the pure
  read rate" therefore under-charges a read beside background writes by ≈ 10–20% on this drive: **for the owner
  before step 4** (D4's 15% check). The engine loop's 3–5 ms host body beside 2 or 3 I/O threads: no slowdown (−3% to
  +1%), as predicted.
- **The judge.** `holon_judge tiers` with `state_tier.levels = ssd` adds the level's reads to its minimums (8 per
  invariance run, 32 per run-identity run), unseeded sleeps before every level I/O job (the copy delays' bound on the
  GPU, 5 ms on the CPU), and the **restart check** (3f's two engines, acceptance 2's "restored after a restart"): A
  resumes parked sessions with no restart; B closes (`persist = close`) and B2, restored from B's directory, resumes
  them with A's outputs and `cached_tokens`; then B2 stops without a snapshot (a crash) and B3 must restore nothing.
  CPU: the tiny dense model through the level (invariance, run-identity, restart) and the tiny hybrid (invariance with
  pages from the level and checkpoints written back; `branch` only: a dropped superseded `prompt_end` or tentative
  checkpoint changes which requests hit in one order and not the other, so with them the comparison is refused).
  A checkpoint read back from the level is the bare tier's test.
- **Plants on the level** (CPU, caught): `read_unwaited` (the H2D enqueued at once), `torn_manifest` (the close's
  data writes lost after the manifest; restore discards by crc, `cached_tokens` short) and its variant with the crc
  check off (outputs differ), `stale_slot` (the manifest kept at startup: after a crash the next start restores
  records naming rewritten slots). `free_on_ticket` (host units freed when the write completes): see "After the tag"
  for where it bites and where it does not (the rationale first given here was wrong: the watermark frees units for
  the same commit's stores).
- **Not built in step 3:** `adopt_source_reused` with a delayed I/O thread on the GPU, and core liveness cells with an
  SSD load: the core does not tell a level load from a host one; the tier-side part (an abandoned level load is never
  enqueued and never waited for) is `test_an_abandoned_load_is_never_enqueued_and_never_waited_for`.
- **GPU (RTX A5000, Qwen3-8B; the level on the dev host's network filesystem, buffered: functional, not a rate;
  `evidence/k3_judge_tiers_ssd_rtx-a5000_2026-10-19.json`):** invariance through the level passes (7,402 level reads,
  44 blocks promoted into place); run-identity passes with 640 level reads per run once 16 sessions overflow device
  and host (with 8, the closed loop fits them and reads nothing from the level); the restart check passes (338 and 434
  blocks restored, nothing after a crash); `read_unwaited` (fired 133, changed 47), `torn_manifest` and its crc-off
  variant (changed 8 each) and `stale_slot` (changed 1) are caught.
- **Found while testing:** without the watermark (`host.reserve = 0`) a load's later read-back could make room by
  evicting a copy the same load had just read back (a crash: the load then found it gone). The load's copies are now
  marked loading before any read-back; `test_the_level_without_a_watermark_passes_run_identity` fails without it.

### Step 2 as built (2026-10-19)

- **`state_copy@1` providers** (`kernels.state_copy`): loads by the SM gather (8 CTAs × 256 threads, unroll 2; rows over
  64 KiB tiled across the CTAs), stores by one `cudaMemcpyBatchAsync` merging consecutive rows; the fact `copy`
  overrides either. L0 bitwise against the oracle on sm_86 and sm_120 (16 cells), memcheck clean. First built in
  `kernels.expert_io` (the step-2 measurements below ran there); rev 8.3 moved them, unchanged, to a plugin of their
  own, so `expert_io` is byte-identical to step 1's and no expert lock needs a propagation.
- **Measured on the RTX 5080** (`evidence/k2_state_copy_rtx-5080_2026-10-18.json`,
  `evidence/k2_mixed_nvme_rtx-5080_2026-10-18.json`, `spikes/k2_tiers/`):
  - the gather beside compute-bound prefill (27B, 512-token chunks): a 0.57 or 1.29 GB load per step costs nothing
    measurable; loads back to back cost 1.175× (predicted 1.03–1.08×: the gather holds 8 of 84 SMs, likely an extra
    wave of the step's kernels; not profiled). The cost tracks copy time, ≈ 5 ms per 1.29 GB load;
  - the host cost of a load, the tier's row lists included: 67 µs at 16K and 115 µs at 32K (≈ 4 ns per chunk). A DMA
    store's submit is 4.6 ms at 16K and 19.9 ms at 32K (the driver's), so large write-backs may take `copy = sm`;
  - the mixed NVMe factor under rev 8.2's reads-first dispatch (one writer, 2–3 readers): 0.77–0.80, as without it:
    the drive's own interference. D4's lane charges ≈ 0.79 while both kinds are queued (tune measures it per host).
- **std.state_tier's host level and the recurrent kind's policies** (`2b3ce54`): as §5 and D8 describe; the host
  level's split of `host.bytes` between pages and payloads is the option `host.pages` (a share, default 0.25: K-M5's
  page peak was ≤ 4.8 of 16 GiB).

- **`holon_judge tiers`** (`judge/holon_judge/tiers.py`; `evidence/k2_judge_tiers_rtx-a5000_2026-10-19.json`):
  invariance with a back-to-back reference, the null test (hybrids), run-identity ×3, and an audit after every step
  (resident prefix closure and `kids`, unit references, refused admissions only without a free slot, the tier's
  leaf-first drops). Clean on Qwen3-8B bf16 on the RTX A5000 through the `state_copy@1` provider (58 requests, 140 page
  loads, 44 blocks promoted into place; run-identity with 133 loads and 8 aborted load-gated requests, also with
  0.3 s delays) and on the tiny dense and hybrid models on the CPU (the hybrid: 122 page and 115 payload loads, null
  test passing). `bounded --policies` on the 27B UD-Q4_K_M (8 + 4 late sessions): `branch` and `answer:1` with the
  tier pass.
- **Plants** (fired / changed): caught on the GPU (Qwen3-8B) — promote_unwaited 388/147, store_early 274/20,
  store_unwaited 504/96, load_no_after 300/5, swap_onto_lower 44/17 (the real model shows the recomputed bytes); on the
  CPU — promote_unwaited, waiter_unref, store_unwaited, promote_wrong_unit, gate_on_landing, promote_preempts,
  kids_any_level, interior_drop, and branch_off_by_one (27B, fired 70). **Deviations from §6's predictions:**
  - `load_before_store` and `load_reuse_unwaited` never change an outcome: D4's free-rule wait at the next launch and
    a load's `after` (the last launched step) already order the read after the store, and a late copy into an evicted
    loading block drains, in stream order, before the next writer. The tier's write fences are the second guard; no
    configuration reached a window where only they matter (rev 8.3: `load_reuse_unwaited` is caught on the CPU with
    copies up to 8 launches late; `load_before_store` is a redundant guard with a tier-only test);
  - `tentative_shift` fails the bounded check on the tiny hybrid but not on the 27B at 8 + 4 sessions (fired 288, KL
    inside 2× the floor): a decode-produced checkpoint one block early is within the bounded class there. Step 4's
    lock trace is larger; the doc's prediction stands open (rev 8.3: the shifted payload mostly finds its block taken
    and is lost; where it misfiles, the judge's audit catches it);
  - `adopt_source_reused` (both variants) cannot change an outcome on the CPU (a step's adopt has finished by its
    launch): it moves to step 3's delayed-I/O-thread cells and the GPU (rev 8.3: on the A5000 the join variant is
    caught, the other still changes nothing);
  - the hybrid invariance trace opens with two warm-up requests that make the system prompt's end a branch point:
    otherwise back-to-back and interleaved orders place `branch` checkpoints differently and the judge, rightly,
    refuses to compare (`cached_tokens` differ);
  - the plants' pressure configuration uses write-back admission (`host.admit = evict`) and later due times, without
    which `store_unwaited` has no window.
- **Found by the judge:** std's host level counted a parent's child copies only for children stored after it, so
  under write-back or `hit:<n>` admission (leaves stored first) a parent could be evicted as a leaf. Fixed (a set of
  children with page copies) with a regression test; the judge record above predates the fix, which only changes
  evictions under those admissions.

### Rev 8.4 (2026-10-19): the panel's second round on step 2

| Finding | Disposition | Where |
|---|---|---|
| A (major): a second `loaded(L, None)` replaced a unit's newer `load` fence with L's older copy (abort, re-promotion by L2, then L1's abandonment: a reuse waited on L1's copy while L2 still read) | fixed | `HostTier.loaded` records the readers' fence on a load's first call only: that call's `done`, or its copy, already covers a later abandonment. `plugins/tests/test_state_tier.py::test_a_reused_host_unit_waits_for_its_readers` fails on bd6142f |
| A (minor): the host level's reuse fences had no tier-only tests | accepted | on recording streams that drive `HostTier` directly: after `loaded(L, step)`, a forced reuse's store has the step in its `after`; after `loaded(L1, L1)`, `loaded(L2, L2)`, `loaded(L1, None)`, it has L2's copy; a payload load leaves no `load` entry and keeps the store as the unit's fence |
| A (minor): the plants' records | accepted | `load_reuse_unwaited`: the GPU cannot open this window; covered on the CPU (copies up to 8 launches late). `adopt_source_reused`'s main guard (a reading step's `done` kept for reuse) is covered by the tier-only tests above |
| B (minor, decision): core back to +326 | done | `Fence.done()`, its platform implementation and `late_loads` are dropped (3a and 3b measure lateness directly); the gated counters move into std's tier, from its loads' due times and `before_launch`: `gated_loads` (due after the clock when issued) and `gated_steps` (launches while one is not due). `core/tests/test_tiers.py` asserts them no more; `test_page_loads_take_the_lane_and_payload_loads_do_not` does |
| C (minor) | resolved by B's | |

Core: +326 code lines over b5b1e7f; ABI 1.4's listing as at step 2's first commit. The full CI suite on b79fc15:
1614 passed, 66 skipped.

### Rev 8.3 (2026-10-19): the panel's review of step 2

The outputs were found correct; the findings are structure, performance and test gaps. Dispositions (the lead's
decisions):

| Finding | Disposition | Where |
|---|---|---|
| B (major): `state_copy@1` in `kernels.expert_io` | accepted | its own plugin, `kernels.state_copy` (`k_rows`, the DMA path, `StateCopy`, a copy of the build glue); `expert_io` byte-identical to step 1's (3a5719d), so no expert lock is owed a propagation; the L0 cells moved to `plugins/tests/test_state_copy.py` (16 / 16 on sm_86); D10 and the as-built note |
| C (major 2): payload-only loads charged to the H2D lane | accepted | only page bytes go on the lane; a payload-only load is due at once (its adopt reads the host copy in-step; D4) |
| C (major 3): the payload share split evenly over payload pools | accepted | split by unit bytes, as pages already were: every payload spec gets as many units (§5) |
| A (major), C (minor 5): host fences and per-key records grow without bound | accepted | a host unit keeps at most one fence per ordered source, the latest store, reading step and unwaited load copy (D4); `loading` and `reading` entries go at 0; `hits` and `kids` are pruned when a key is dropped or its page copy released; copies are indexed by key, so `drop` reads only the key's |
| C (minor 4): small loads off the lane | accepted | every page load occupies the lane; one of at most `stall_bytes` is still due at once (D4) |
| C (minor 6): counters | accepted (rev 8.4: reworked, see there) | std: `late_loads` (a non-blocking `Fence.done()` when a step first reads the load; ABI 1.4's `Fence` gains `done`, for statistics only) and `payload_load_bytes`; core: `gated_requests` (admitted ahead of their loads' due time) and `gated_steps` (steps run while one waited) in `Engine.stats()` |
| A, B (minor): a policy unit cannot keep tentatives | accepted | a unit may declare `keep`; `keep_tentative` is the largest over the policies, so the engine's refusal without a tier sees it; a keep-less unit's tentative positions are ordinary (D8, the kind's docstring) |
| B (minors) | accepted | `TierCtx.pools`' key rule (a key equal to its pool's name is a payload pool); `StateTierPlugin.build`: kernel sites at `attach` only for kernels without a workspace; the hybrid engine test of `plugins/tests/test_state_tier.py` dropped (the judge's hybrid invariance is the same check) |
| A (minor 3): plants | accepted | see below |
| C (major 1): read priority | carried into step 3 | D4, D11: while any read is queued no write starts but one that frees units a queued read needs; reads at the pure read rate, the mixed factor only on dependent writes; step 3 re-measures exactly that dispatch |

**The plants (A, minor 3)** (`evidence/k2_review_plants_rtx-a5000_2026-10-19.json`, the CPU cells in
`judge/tests/test_judge_tiers.py` and `plugins/tests/test_state_tier.py`):
- `load_before_store`: a redundant guard for the host level. End to end the core's ordering (D4's free rule, a load's
  `after`) already puts the read after the store: on the A5000 it never fired, on the CPU it fired without a change.
  The tier-only test (a store, then at once a load with an empty `after`, on deferred streams) fails without the
  tier's write fence;
- `load_reuse_unwaited`: run-identity's `reuse` loop is A's configuration (the gated request with the most loading
  blocks aborted with no resend, a prefill-heavy request in the same round). On the A5000 (Qwen3-8B) it fires 14
  times at 0.05 s delays and 40 at 0.5 s, and changes nothing. On the CPU the configuration alone does not do it:
  with copies at most 2 launches late, the copy has landed by the abort. With copies up to 8 launches late and
  32-token answers it is caught, in the plain loop too. An eviction policy taking the most recent blocks first (the
  loading ones) was tried and is not needed;
- `adopt_source_reused`: run-identity's `fork` loop on Qwen3.5-4B bf16 (A5000) in A's window (the host payload level
  at 6 units per pool, its floor 5; `answer:1`; forks of a prompt plus 128 tokens). The join variant is caught
  (fired 98, 72 outputs changed); the variant where `loaded` ignores the reading step fires 108 times and changes
  nothing. **Open:** its window needs a store to reuse an adopted host unit while the adopting step still runs, but
  the adopted copy is the payload level's most recent (the load and `hit` move it to the end), so floor-many other
  stores must land within that one step;
- `tentative_shift`: the audit now checks every payload at (depth + 1) × block. One block back of an `answer`
  position there is normally `prompt_end`'s or the previous answer's checkpoint, so the shifted payload finds the
  block taken and is only lost; that is why the 27B's bounded check passed. With `answer:1` alone the block is bare:
  the plant misfiles (30 in a CPU run) and the audit fails run-identity; without the audit nothing changes.

**The clean checks after these changes:**
- Qwen3-8B (A5000):
  - invariance passes (140 page loads, 44 blocks promoted into place);
  - run-identity passes (133 loads per run, 123 of them late when their step first read them);
  - the `reuse` loop passes (111 loads, 7 aborts);
  - promote_unwaited 389 / 140, store_early 275 / 16, store_unwaited 503 / 107, load_no_after 301 / 25 and
    swap_onto_lower 44 / 17 (fired / changed) are still caught.
- Qwen3.5-4B (A5000, 555 MB state memory):
  - run-identity and the `fork` loop pass (112 and 140 loads per run; 86 and 98 payload evictions);
  - invariance is refused (6 regenerates' `cached_tokens` differ).
- **Found:** the null test failed there. With 16 checkpoint units the ledger gave the KV pool 567 units against 174:
  a state memory too small for 16 units shrinks the checkpoint pool and grows paged state, so page evictions move
  too. The CPU's pools differ as well (51 / 43 units) with equal hits. The null test now refuses when `cached_tokens`
  differ, as invariance does, and reports the paged pools; rerun alone on the 4B, it is refused.

Core: +332 code lines over b5b1e7f, 6 over the lead's +326: `Fence.done` (1), its platform implementation (2), the
scheduler's two counters (2) and their `stats()` entry (1), all item 6's. Rev 8.4 took them out again (+326).

### Rev 8.2 (2026-10-18): the panel's review of step 1

The core was found correct, with no blocker. Dispositions:

| Finding | Disposition | Where |
|---|---|---|
| A1: the engine tier test never reached promote-into-place, abandonment or the waiterless notice | accepted | `test_steps_and_outputs_never_depend_on_when_copies_land`: 8 conversations over 4 sequences in 20 pages, longer contexts, 3 aborts of load-gated requests; it asserts promote-into-place, abandoned and waiterless counts > 0 (8 / 7 / 3 at delay 2). The `swap_onto_lower` plant now fails it |
| A2: the unguarded-policy plant hung its test | accepted | the test's policy raises on a second `victim` call: the unguarded core fails at once |
| A3: the 27 plants could not be re-run | accepted | `spikes/k1_core/plants.py` pairs each of 28 core plants with the test it must fail; `evidence/k1_core_plants_cpu_2026-10-18.json` records 28 / 28. `store_no_after` stays a GPU plant for step 2 |
| A (re-review): `swap_onto_lower` failed only by crashing; its semantic form (promote with no load) passed every engine test | accepted | The plant is now that semantic form, paired with the state-level promote-into-place test and the engine tier-invariance test. That test now reaches promote-into-place serially: each turn first re-sends the previous prompt cut to whole blocks (a regenerate), whose last block is recomputed while only the tier holds it. A tight pool alone cannot do it serially: with one request at a time, every index-only block is a pure free. The runner checks the unplanted baseline passes first, and counts a plant as caught only when its tests fail by assertions (a crash or a collection error does not count) |
| B1: freeze the two slot factories | accepted, at +326 | `hooks.StateTierPlugin.build(ctx) -> StateTier` and `hooks.EvictionPolicyPlugin.build(ctx) -> EvictionPolicy` in the ABI listing (in `holon.api.hooks`, beside `BuildsPolicy`: exporting them from `holon.api` would have cost 11 lines of import block). Core is +326 over b5b1e7f, the lead's raised cap for them |
| B2: tidy-ups | accepted | load tracks unpacked where read; `plans.frontend_options` → `unplanned_options`; the engine's `evicting` → `victims`; restore's `if p:` |
| C1 (major): SSD write-back | accepted | D11, rev 8.2: a payload superseded down single-child links is dropped, not written back; branch points keep theirs (std, step 3) |
| C2 (major): read priority on SSD dispatch | accepted | D11 and §4's contract, rev 8.2; the mixed factor re-measured under it in step 2 |
| C3: the gather and 3b | accepted | D10: the SM gather pinned at 8 CTAs, paced DMA tune's alternative; 3b's prediction ≈ 1.04–1.07×; step 2 measures the gather beside prefill and its per-load host cost |
| C4: `answer:1` the default | accepted | D8 |
| C5: host-side wins | one taken, one skipped | admission takes a hit's resident units with one incref per spec (taken, it saves a line). Promote-into-place batched into one `load` per commit is **skipped**: it does not fit under the cap (≈ +3 lines) |
| C6: predictions on the measured 4-row step | accepted | §8: 22.3 / 14.7 ms; the bars' "one decode step" is that step (3a's RAM bar 40.3 ms, 3h's 14.7 ms) |

### Step 1 as built (2026-10-18)

KS1–KS5 as §4 describes, ABI 1.4, +326 code lines over b5b1e7f (+320 before rev 8.2's frozen slot factories). KS1+KS2 were priced on the prototype
at ≈ +180 lines against the itemized 154 (+17%, under the +20% stop line). Additions and deviations:
- `Pool.segments` in the API (a tier addresses units by segment), `Seq.branch_at` behind `SeqView.branch_at`,
  `StateTier.settled` (rev 8.1);
- `RestoresPayloads.adopt` copies from `payload.pool` (the kind's pool or a tier's host copy): std's recurrent kind
  changes in step 2;
- with an eviction policy the upward climb is off (each victim through the policy); a policy returning nothing falls
  back to the least recently used entry, and one returning a non-candidate raises;
- the plugins' `build(ctx)` for the two new slots: `hooks.StateTierPlugin`, `hooks.EvictionPolicyPlugin` (rev 8.2);
- the core's restore skips records that do not re-hash to their key, whose parent is unknown or that lack pages;
- an entry's payloads in the tier are adopted through one load (a tier may evict only within `store`, `load` and its
  step hooks, which the core polls after);
- tiers-off snapshots are byte-identical to b5b1e7f on the RTX A5000 (Qwen3-8B, the 27B hybrid, the 30B with
  offload; `evidence/k1_tiers_off_snapshots_rtx-a5000_2026-10-18.json`); every core plant fails its test
  (`evidence/k1_core_plants_cpu_2026-10-18.json`);
- found on the way, not K's: publishing after `flush_prefix` while a request runs raises a KeyError in
  `PrefixIndex.insert` (the request's chain names a parent that is gone), as at b5b1e7f. It is a follow-up (§10).

### Rev 8 (2026-10-18): the panel's seventh round

B was clean on rev 7. A and C each found one tier-side minor issue, both confirmed against rev 7's D4 and D7. Rev 7
kept only a unit's *last* fence, so a later waiterless `loaded` could replace a reader's `done`. Rev 7 also left the
I/O threads' job order, and what an abandonment does to a launch already waiting, unstated.

| Finding | Disposition | Where / why |
|---|---|---|
| A: a host unit's fence must combine fences, not keep the latest | accepted (the lead's simpler rule) | D4, §4: each host unit keeps its set of outstanding fences. Stores' ends, reading steps' `done`, and waiterless or abandoned `loaded` fences only add; they are pruned as they complete, and reuse waits on all of them. The joining variant of `adopt_source_reused` adds the call order "G aborted, J due in the same step, G's `loaded` processed second" |
| C: I/O ordering | accepted | D7, §4 contract: I/O jobs are dispatched FIFO in the order they were decided, so a dependency is always an earlier job. Abandonment completes the pending-enqueue condition as a no-op and wakes its waiters. The abandon-while-waited case runs in the existing delayed-I/O-thread plants (`read_unwaited`, `adopt_source_reused`) |
| Coordinator: FN cross-reference | done | acceptance 5: FN rev 3 (7a80674), C1–C3, ≤ +47 lines, no FFN branch |

**Budget, ABI, schedule.**
- Core: unchanged at ≈ 292 lines, because both rules are the tier's. The budget stays ≤ +320.
- ABI: unchanged (1.4 or 1.5 by merge order). The §4 contract gains the two rules.
- Schedule: unchanged at 26 days.
