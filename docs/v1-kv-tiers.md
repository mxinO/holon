# v1 — milestone K: KV and checkpoint tiers under the prefix index

Status: **rev 1, draft for the challenge panel** (2026-10-18). Nothing is built and nothing has been measured for K
yet; every number below is a CPU-side estimate from earlier evidence (cited) and is fixed as a prediction before step
0 and step 4 measure it. The review log is §12.

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
block granularity under one prefix index — so a shared system prompt is stored once, not once per parked
conversation as llama.cpp's and Strata's blobs store it — with these properties:
- **exact by construction:** a hit's bytes are the same whichever tier held them;
- **deterministic resume:** batch composition never follows copy timing;
- **plugins:** tiers and checkpoint policies are plugins.

## 1. Goal and acceptance

Keep KV pages and recurrent checkpoints that VRAM cannot hold in host RAM and on SSD instead of dropping them, so that
agent sessions resume from a lower tier rather than by recomputing. This should beat the consumer engines' host
caches on an agent trace, with outputs that do not depend on where state was kept.

1. **Strategies are plugins** (D1, D3).
   - Shipped in K:
     - the tiers `std.tier.host` (pinned RAM, write-through, inclusive of VRAM) and `std.tier.ssd` (O_DIRECT slot
       files, write-back from RAM, optional persistence);
     - the checkpoint policies of `std.state.recurrent`: `branch` and `answer` (new), `prefill:<n>` (new, tentative
       in-step checkpoints), and `prompt_end` and `every:<n>` (existing).
   - **Held-out proof.** The std tier API is tagged at the end of step 3 (`v1-kv-tiers-std-api`). After the tag, two
     strategies this document does not design land as commits whose diff touches only their own plugin directory and
     tests (checked with `check_core_diff.sh` and `git diff --stat`):
     - `tiers.selective`: HiCache's default, a host tier that stores a block only at its second hit and drops cold
       ones;
     - `tiers.buffered`: Maya KG3's buffered-write file tier, with no O_DIRECT and no persistence.
   - Every core seam (§4) is proven first with a toy kind and toy tiers in `core/tests`.
2. **Exactness** (D4, §6).
   - **Tier invariance, any tier configuration.** On a serial multi-turn trace that forces every resumed session
     through a lower tier, the output streams and each request's `cached_tokens` are byte-identical to the same trace
     run with VRAM large enough that nothing is demoted. This holds for host only, host + SSD, and SSD restored after
     a restart.
   - On a concurrent trace with ≥ 32 loads per tier: run-identical over 3 fresh engines.
   - The new checkpoint policies are judged as today's prefix hits are:
     - `branch` and `prefill:<n>` are prefill-produced, so their payloads are bitwise equal to the recomputed state;
     - `answer` is decode-produced, so `bounded`;
     - all of them pass the bounded multi-turn check against its floor.
3. **Development gate.** Qwen3.8-27B UD-IQ3_S on the RTX 5080, with its lock's settings (`max_seqs` 4, q8_0 KV,
   four checkpoints, `chunk_with_decode` 512), plus UD-Q4_K_M on the RTX 5090. Same file and host, medians of
   interleaved rounds. Every row is predicted in §8 before it is measured.

   | # | Measure | Bar |
   |---|---|---|
   | 3a | Resume microbench: one parked conversation of 4K / 16K / 32K tokens plus a 256-token append, resumed from VRAM, from RAM and from SSD | RAM-hit TTFT ≤ VRAM-hit TTFT + 1.5 × bytes / measured H2D rate + one decode step (the granularity of a deterministic resume, D4, for loads above the stall threshold); SSD-hit TTFT ≤ VRAM-hit TTFT + 1.5 × (bytes / measured read rate + bytes / H2D rate) + one decode step |
   | 3b | Three sessions decoding while a fourth resumes 32K tokens from RAM, repeated 20× | the three sessions' TPOT p99 inside the load windows ≤ 1.2× their TPOT p99 outside them |
   | 3c | `agent_park` trace (§8, new): 12 sessions over 4 clients, parked and resumed. Compared with Holon VRAM-only (today) | TTFT p50 ≤ 0.4×, output throughput ≥ 1.2× |
   | 3d | `agent_park` vs `llama-server` with `--cache-ram` (default 8 GiB) and `--ctx-checkpoints`, same file, card and KV format | TTFT p50 and p90 ≤ 1.0×, output throughput ≥ 1.0× |
   | 3e | `claude_code` trace (16K shared system prompt, `bench/README.md:58-60`): TTFT of each session's first request, `branch` on vs off | p50 ≤ 0.2× |
   | 3f | Restart (`tier.ssd.persist = close`): a parked 16K conversation resumed after an engine restart | TTFT within 3a's SSD bar; output byte-identical to the same resume before the restart |
   | 3g | pi agent test re-run on the RTX 5090 (report, not gated: agent outputs are not reproducible) | the share of uncached input that is the previous turn's own output, predicted to fall from 44% to ≤ 10% if K-M1 shows verbatim re-rendering |

4. **Memory** (§7).
   - The host tier is allocated exactly from S2 host pools and accounted against `host_memory`.
   - A host tier that stores at publish must hold at least the device units of every shape it copies (inclusion,
     D5). Otherwise startup refuses it with a per-pool report.
   - The SSD tier refuses a directory without its bytes.
   - With tiers off, VRAM is unchanged; with tiers on, no new device pool is added.
5. **Core.**
   - Only §4's seams KS1–KS6, an additive ABI minor (1.4), each toy-proven.
   - A property test shows that `could_fit` and eviction agree under random traffic with demotions and loads.
   - At most +480 code lines (4,603 → ≤ 5,100 of 6,000).
   - With tiers off, every existing lock's snapshot stays byte-identical, so propagation auto-advances.
6. **Locks.**
   - The 27B agent locks gain the host tier and the `branch` and `answer` policies (on the RTX 5080 and 5090 after
     the bounded check).
   - Every other lock is propagated once, at K's close (D14).

Moved out of K, with reasons in D2 and §11:
- **Project Maya's multi-session ratio** (`docs/v1-offload.md:598`). It needs Maya on the same trace, which is GLM's
  milestone. K reports Holon's own N-session / 1-session throughput ratio on `agent_park` instead.

Non-goals of K:
- K4, KV streaming for sparse-attention mixers (in FN and GLM, through `std.tiered`'s `DeviceResolver`);
- layer-wise loading and per-layer ready events (D12);
- CPU attention (`-nkvo`);
- remote tiers (Mooncake, NIXL, object stores);
- compressed or quantized checkpoint codecs;
- an explicit save / restore HTTP API;
- checkpoints from verify rows (MTP on, see D8);
- VMM rebalancing of KV against experts (plan §10).

## 2. Decisions

**D1 — Strategy set.** Classes:
- **core:** a generic mechanism (§4);
- **T:** a tier plugin (storage and I/O; slot `tier`);
- **K:** a state-kind policy (`std.state.recurrent` options).

| # | Strategy | Reference | Class | K default |
|---|---|---|---|---|
| T1 | Host-RAM tier for KV pages and payloads, per block, write-through at publish, inclusive of VRAM | vLLM native offload, HiCache `write_through`, llama.cpp `--cache-ram`, Strata C1/C2 | core + T | on in the 27B agent locks |
| T2 | SSD tier below RAM: slot files, O_DIRECT, written back when RAM evicts | HiCache L3, LMCache disk, Maya KG3, Strata C3 | core + T | opt-in (`tier.ssd.dir`) |
| T3 | Persistence: the SSD tier's manifest restores the index at startup | Maya `SLOT_KEEP`, Strata C3, R K3 | core + T | opt-in (`tier.ssd.persist = close`) |
| P1 | Conversation parking = the prefix index with T1 / T2 (no separate feature) | Strata C2, llama.cpp `--cache-ram` | — | with T1 |
| P2 | Swap-on-preemption = write-through pages plus the release store of tentative checkpoints (KS3); the resume is a prefix hit | vLLM V0 swap, subsumed (R §4.1) | core | with T1 |
| C1 | `prompt_end` checkpoints (exists, `std/holon_std/state/recurrent/kind.py:46-55`) | Strata C1 turn start | K | on |
| C2 | **`branch`:** a checkpoint where a request leaves a cached chain that continues (the shared system prompt's end, found rather than configured) | Strata C1 "system prompt ≥ 2048", Maya KG2 | K + core fact | on |
| C3 | **`answer:<k>`:** a tentative checkpoint at each aligned position a decode row reaches; the request keeps its last k, which become ordinary payloads when it releases (the end-of-answer checkpoint) | plan §10 item; Strata's opt-in tail checkpoints | K + core | on, k from K-M1 |
| C4 | **`prefill:<n>`:** tentative in-step checkpoints every n tokens of a prefill, the last kept (swap-on-preemption for long prompts) | Strata/Maya every 16K (non-tentative) | K + core | n = 4096 |
| C5 | `every:<n>` (exists) | Strata, Maya every 16K | K | off |
| H1 | `tiers.selective`: store at the second hit | HiCache default | T | held-out |
| H2 | `tiers.buffered`: buffered file tier | Maya KG3 | T | held-out |
| — | Root pinning (Strata), ARC (vLLM), layer-wise load-back (HiCache), K4 streaming | — | — | deferred (§10) |

**D2 — Vehicle, comparisons, and what moves out.**
- **Gate vehicle: Qwen3.8-27B GGUF.** It has both kinds of state K tiers: q8_0 KV for 16 attention layers (34 KiB per
  token) and 48 GDN layers whose checkpoints are 147 MiB each (`docs/m03-qwen38.md:59-60`). It is the pi agent test's
  model, and its RTX 5080 (UD-IQ3_S) and RTX 5090 (UD-Q4_K_M) locks exist
  (`plans/unsloth/Qwen3.8-27B-GGUF/…/rtx-5080/…/10.lock`: `max_seqs` 4, `recurrent.checkpoints` 4, q8_0).
- **Development vehicle for KV-only paths: Qwen3-8B** (bf16 and Q4_K_M), on the A5000. It is the bounded check's
  calibration model (`docs/design.md:1307`), has paged state only (16-token blocks, 144 KiB per token in bf16),
  and is quick to iterate on. Tiny random models (`testing/holon_testing`) carry the CPU tests.
- **Qwen3-30B-A3B (offload).** A report-only coexistence row: X's expert host pools and the host tier share
  `host_memory` (§7).
- **Gating reference: `llama-server`** with `--cache-ram` and `--ctx-checkpoints`. It is the consumer default (R §4.1)
  and serves the same GGUF files.
- **Report-only references:** SGLang HiCache and vLLM's native offload on Qwen3-8B, on the RTX 5090, if their
  environments in `bench/.envs/` serve it.
- **Maya's multi-session ratio** (×1.41 / ×1.35 at 2 / 4 requests, V100, R §3.1b) is measured on GLM-5.3-Flash.
  It moves to the GLM milestone, with this doc's machinery under it (open question 5).

**D3 — Mechanism in the core, storage in tiers, positions in kinds.** The PrefixIndex is core-owned (design §6.3).
Its invariants decide liveness, so the core owns which tier holds which copy, eviction order, copies and loads.
- **A tier plugin (KS5) owns** its storage and I/O, and its store admission: whether to store a copy at publish, at a
  hit, or at eviction from the tier above.
- **A state kind owns** where checkpoints are taken (D8).
- **The core never names** a tier, a format or a model (invariant 1, `scripts/ci/check_core_denylist.py:8-11`).

Plugin kinds:

| Kind | Selected by | Contract | Instances in K |
|---|---|---|---|
| tier | `EngineConfig.tiers = ["std.tier.host", "std.tier.ssd"]` (slot `tier`, fastest first) | `StateTier` protocol (KS5): `bind`, host `pool` or storage `write` / `read` / `drop` with tickets, `admit(event, hits)`, `bytes_per_step`, `stall_bytes`, `restore` | host, ssd; held-out selective, buffered |
| checkpoint policy | `recurrent.policy = "prompt_end,branch,answer:2,prefill:4096"` (a list; an entry may be a `Ref` to a plugin implementing `CheckpointPolicy`: `next(view) → (at, tentative) | None`) | std's `CheckpointPolicy` | prompt_end, every, branch, answer, prefill |

Tiers are a slot, not open options holding a `Ref` as X's expert store was (`docs/v1-offload.md` D3), because here
the core calls them. Release §15.1 cut the `tier` slot because nothing called it (`docs/release.md:1738`); re-adding a
slot is a minor (`core/holon/api/manifest.py:41-42`).

**D4 — Exact means tier-invariant; resume is deterministic.**
- **Bytes.** Every move between tiers is a byte copy of whole pool units:
  - device ↔ host by core MemOps between pools of one unit shape (`core/holon/core/memory.py:127-149`);
  - host ↔ storage by the tier's own I/O.

  A hit adopts only from VRAM units (D7). A resumed request therefore computes on the same bytes whatever tier
  held them, and tier invariance (acceptance 2) is a property of the copies alone. No kernel changes.
- **Decisions use index state only, never timing.** Which entries are demoted, promoted, dropped or hit depends only
  on the host-side index and pool counts.
  - A copy still in flight never changes a decision: freeing units waits for their copy instead of skipping them
    (KS1). A late copy costs time, never a different choice.
  - Write-through stores and SSD write-backs complete whenever they complete; nothing reads their completion except
    a free, which waits.
- **Resume at a fixed step** (panel C9b).
  - A request whose hit needs a load is BLOCKED until a step `r` fixed when it is admitted, from a per-tier virtual
    schedule kept on the host:
    - `due[t] = max(due[t], s) + ⌈bytes / bytes_per_step[t]⌉` for each tier the load passes through, where `s` is
      the step being scheduled;
    - `bytes_per_step` is the tier's pinned option (tune sets it to ≈ 0.8 × the measured rate × the lock's decode
      step time);
    - a load of at most `stall_bytes` gets `r = s`: the step's launch waits for it, D = 0.
  - Step `r`'s launch waits on the copy's event: a late copy stalls `r` and never skips it (S2's `lands_by`,
    `memory.py:188-191`).
  - When the only runnable work is load-blocked requests, the engine waits for the earliest-due load and runs it at
    the next step. This is deterministic: whether anything else was runnable is itself index state.
- **What tier invariance does not cover.** Whether a hit happens at all depends on capacities, as today. With tiers,
  more requests hit, and those hits are prefix hits, already `bounded` against recompute (L1.5,
  `docs/design.md:1394`). Comparisons that change the hit set are bounded; acceptance 2's comparisons hold the hit
  set fixed.

**D5 — Write-through by default; the host tier includes VRAM.** R §4.2 K2 proposed write-back at eviction (line 578).
This design reverses that default:
- **No stall at eviction.**
  - Write-back must wait for a D2H copy before freeing VRAM units. Evicting a 16K conversation means 557 MB, about
    11.7 ms at X's measured 47.5 GB/s (`docs/v1-offload.md:645`), inside the scheduler.
  - With write-through the copy already exists, and eviction just frees.
- **Swap-on-preemption is free.** A preempted request's published pages already have host copies (P2).
- **The PCIe cost is small.** The 27B prefills about 2.2–2.5K tokens per second on the RTX 5080
  (`evidence/bars_release-qwen3.8-27b-ud-iq3_s_rtx-5080_2026-10-16.json`: pp512 2,544, at depth 8K 2,154). That is
  ≈ 85 MB/s of KV, ≈ 0.2% of the D2H link, on a stream of its own (KS4). Each checkpoint stored is 147 MiB, ≈ 3.2 ms.
- **Inclusion makes host eviction total.** If host units ≥ device units per shape (checked at startup), a host-tier
  allocation can always free a unit by evicting a demoted leaf:
  - every host unit is either a resident entry's copy, of which there are fewer than device units, or a demoted
    entry's;
  - a demoted subtree always has a leaf.
- **Tentative checkpoints are not written through** (D8). They are stored when their request releases.
- **Other choices are a tier's `admit`.** Write-back is `admit("evict")`; HiCache's selective mode is
  `admit("hit", hits ≥ 2)` (H1).
  - A tier that does not store at publish is not inclusive.
  - With such a tier, a VRAM eviction of an entry that has no lower copy drops the entry together with its subtree.
    Every descendant of an entry evictable from VRAM lives only in lower tiers (D6), so dropping it orphans nothing.

**D6 — Eviction invariants** (panel A10).

*Levels.*
- An entry's **level** is the fastest tier holding all of its paged copies: 0 is VRAM, then the configured tiers in
  order.
- **Levels are monotone along a chain:** `level(parent) ≤ level(child)`. In particular, VRAM-resident entries are
  prefix-closed.
- Each entry counts its children per level (`kids[l]`).
- Each tier keeps its own LRU of the entries at its level (`collections.OrderedDict`, as `prefix.py:45` does for VRAM
  today). The VRAM scan stays as large as today's, whatever the lower tiers hold.

*Eviction from tier t.*
- Eviction takes the least recently used entry at level t that has no child at a level ≤ t. Children in lower tiers
  never pin a parent, which is panel A10's case.
- The entry moves to the next tier that holds, or admits, a copy. If there is none, it is dropped with its subtree.
- For the last tier, "no child at a level ≤ t" means a true leaf.

*Payloads.*
- A payload copy is independent of the tree, as today (`prefix.py:12-15`). It can be dropped from any entry at any
  tier.
- A payload's VRAM copy is dropped first wherever a lower copy exists.

*Agreement with `could_fit`.*
- `could_fit` keeps today's formula (`core/holon/core/state.py:366-385`). It counts device units held only by the
  index as freeable.
- **Claim:** each such unit belongs to a resident entry no request holds. A request holds exactly the chain it follows
  (`prefix.py:8-10`, kept by `_swap`, `state.py:355-364`), so all of that entry's resident descendants are unheld too.
  Evicting resident leaves upward therefore frees the unit: demoted children are not counted.
- Loading entries (D7) are index-held too. Eviction waits for their copy, then frees them.
- **The property test** (KS1) drives random publish / hit / release / preempt / demote / promote / tier-evict
  sequences. After every operation it checks:
  - level monotonicity;
  - that every entry's parent exists;
  - that host units are within capacity;
  - that `could_fit(seq, n, releasing)` is true exactly when releasing those requests and evicting actually yields
    the units.

  It extends `core/tests/test_state_liveness.py`'s random-traffic harness.

**D7 — Loads land in index-owned VRAM units; cancellation is write-safe by construction.**

*Admission.*
- A hit whose chain has demoted entries, or whose payload lives only below VRAM, **promotes** them at admission.
- The core allocates VRAM units for each demoted entry's paged copies, and a VRAM payload unit (evicting as `reserve`
  does, `state.py:292-312`). The index takes its reference on them, and the request takes its own as for any hit
  (`state.py:173-176`).
- The copies are async MemOps:
  - `after` = the last launched step, so a unit freed at the previous commit and still touched by the in-flight step
    is never overwritten under it (the extra row of a finished request, `core/holon/core/engine.py:575-576`);
  - `lands_by` = `r`.
- From a storage tier, the read into host units comes first (a ticket, then such a MemOp at the read's due step).
  The core waits on the ticket there if it is late.
- The entry is marked **loading until `r`**. A later request that hits a loading entry resumes no earlier than `r`.
- The request then adopts from VRAM as today (`recurrent/kind.py:88-90`; paged hits need no copy).

*Cancellation.*
- Aborting or preempting a loading request only drops its references. The entries stay index-owned, and their copies
  land into units nobody else can reach.
- Freeing such a unit (by eviction) waits for its copy (KS1).
- This meets "a load lands in staging and reaches device units only through a MemOp of the resuming step, or
  released units wait on the load's event" (`docs/v1-offload.md:593-594`) by its second clause, applied to every free.

*No future object in the ABI.*
- `Status.BLOCKED` exists (`core/holon/api/batch.py:21-26`), and the scheduler already skips BLOCKED requests without
  stopping the queue (`core/holon/core/sched.py:133-134`).
- The load itself is core-internal.
- Design §6.4's option (b), layer-wise loading, stays planned (D12).

**D8 — Checkpoint positions** (`std.state.recurrent`, all exact or bounded, never approximate).

*`branch`.*
- The core reports `SeqView.branch_at`: where the request left a cached chain that continues past it. This is the
  matched prefix when its last entry has children, else 0. It is computed in `admit` before the fixed point trims the
  hit.
- The policy checkpoints there in-step during prefill. The position is block-aligned, so it is on the 64-token grid
  and costs no cut (`docs/m03-in-step-snapshot.md:94-111`).
- Why: today, a session sharing only a system prompt with earlier sessions gets no recurrent hit at all. Its
  `prompt_end` payloads all lie past the system prompt (`recurrent/kind.py:54`), and the fixed point trims the KV
  match to the last payload (`state.py:190-201`).
  - On `claude_code` (16,384-token system prompt), every session's first turn on the 27B recomputes 16K tokens: about
    7 s at the RTX 5080's 2.2–2.5K tok/s.
  - With `branch`, the second session checkpoints at 16,384 and every later one hits it (acceptance 3e).

*`answer:<k>`.*
- Each decode row that reaches an aligned position publishes a **tentative** payload (KS3). Today's decode publishes
  of `every:<n>` use the same path: a post-sample D2D copy, `state.py:236-258`. A 147 MiB copy every 64 decode
  tokens is ≈ 0.3 ms per ≈ 1 s of decode on the RTX 5080, ≈ 0.03%.
- A request keeps its last k tentative payloads. At release (finish, abort or preempt) they become ordinary payloads,
  and the tiers' store admission runs on them.
- This is plan §10's end-of-answer checkpoint: "the last aligned position the next prompt shares"
  (`docs/plan.md:679-680`).
- **Why k > 1 may be needed.** A template that re-renders the answer differently (a tool call's arguments
  re-serialized from JSON, say) diverges inside the answer, and the useful checkpoint is then the last crossing
  before that point.
- K-M1 measures, on the pi trace, where re-rendered answers diverge from the generated tokens. It fixes k (default 2)
  and predicts 3g.
- Decode-produced checkpoints are `bounded` (`docs/m03-qwen38.md:22-23`), the same class as today's hits across
  generated KV pages (`docs/design.md:900-902`).
- Verify rows publish nothing (`state.py:239`), so with MTP on the policy is idle. The 27B locks run without drafting.

*`prefill:<n>`.*
- Tentative in-step checkpoints every n tokens of a long prefill, with the last kept.
- A request preempted mid-prompt then resumes from at most n tokens back, instead of from its previous `prompt_end`,
  which on a 30K prompt is 30K tokens of recompute.
- Each one costs a 147 MiB D2D copy per n = 4,096 tokens (≈ 1.7 s of prefill).

*Tentative bookkeeping is core (KS3); positions are the policy's.*
- A policy is a std `CheckpointPolicy`, `next(view) → (at, tentative) | None`, listed in `recurrent.policy`.
- A new policy is a plugin whose `Ref` appears in that list, with no std diff.

**D9 — No `HitLen.cost`: the deepest hit wins, and slow tiers are refused at startup.** X's outline lists
`HitLen.cost` (`docs/v1-offload.md:599`). A cost changes a decision only where loading a deeper hit is slower than
recomputing the difference. Per token on these machines, that never happens for RAM and rarely for SSD:

| | Load rate | Load per token | Recompute rate | Recompute per token |
|---|---|---|---|---|
| 27B, RAM | 47.5 GB/s | ≈ 0.73 µs (34 KiB) | 2.2K tok/s (RTX 5080) | ≈ 450 µs |
| 27B, SSD | ≈ 2.6 GB/s (X's M7) | ≈ 13 µs | 2.2K tok/s (RTX 5080) | ≈ 450 µs |
| Qwen3-8B bf16, SSD | ≈ 2.6 GB/s | ≈ 57 µs (144 KiB) | ≈ 8.3K tok/s (RTX 5080 pp512, `docs/plan.md:301`) | ≈ 120 µs |

The worst row is the 8B's KV from SSD, still about 2× faster than recomputing. Only a payload adds a fixed cost: a
147 MiB checkpoint from SSD (≈ 57 ms) loses to recomputing a gap shorter than ≈ 125 tokens, at most ≈ 57 ms per hit.

So K takes the deepest hit (today's fixed point, unchanged). It refuses at startup any tier whose
`bytes_per_step / bytes per token` is below the deployment's recompute rate (the tier reports its rate; tune records
both). A remote tier slower than recompute would bring back `HitLen.cost`; until then it is a field nothing reads
(the reason release §15.1 cut it).

**D10 — Copies: two streams, per-op events, batched cross-device copies.**

*Two streams.* Today one copy stream serves both directions (`memory.py:162-206`). K gives H2D and D2H a stream each:
- PCIe is full duplex;
- a write-through store queued ahead of a load would otherwise delay it.

*Per-op events.* `Copies.add` returns each op's done event, which is what an eviction's wait reads.

*Batched cross-device copies.*
- Today `copy_units` issues one `copy_` per segment per run of consecutive units (`memory.py:141-149`). KV pools are
  layer-major with one segment per attention layer (`std/holon_std/mixers/gqa/units.py:136-141`). A 27B q8_0 page is
  16 chunks of 34,816 B.
- A 32K-token load with fragmented units would be 32,768 `copy_` calls: ≈ 0.3 s of host time in the engine loop,
  which fails 3a and 3b on its own.
- The cross-device path instead builds the chunk address lists (vectorised) and issues one `cudaMemcpyBatchAsync` per
  op batch, through `platform` as `cudaHostRegister` is called today (`core/holon/core/platform/__init__.py:130-143`).
  X's M1 measured that call at 46.7–48.4 GB/s on 2.8 MB blobs (`docs/v1-offload.md:645`).
- If K-M1 finds it below 0.8× the blob rate at 34–64 KiB chunks, step 1 adds a block-contiguous host layout instead:
  - the host tier's pool keeps one block per unit (`unit_bytes` × pages per block, one segment);
  - MemOps across it scatter with the same batch call over larger host runs.

  No gather kernel goes into the core (invariant: no native code in core).

**D11 — The SSD tier.**

*Layout.*
- One preallocated slot file per unit shape (`fallocate`), sized by `tier.ssd.bytes`.
- A tier unit is a slot of `unit_bytes`, so tier units map 1:1 to host units and the core's allocation is the same
  as for pools.

*I/O.*
- Reads and writes go straight from and to the host tier's pinned units with `os.preadv` / `os.pwritev` over the
  unit's per-segment chunks. The GIL is released in the syscall.
- They run on `tier.ssd.io_threads` (default 2: X's M7 reached 2.5–2.7 GB/s from two readers or depth 2,
  `docs/v1-offload.md:649`). The threads are pinned off the engine's `loop_core` (`core/holon/api/hooks.py:157-158`).
- I/O is O_DIRECT where the filesystem allows it and every chunk is a multiple of the device's logical block. For
  example, 34,816 B is a multiple of 512 but not of 4,096. Otherwise each thread copies through an aligned bounce
  buffer, or falls back to buffered I/O with `posix_fadvise(DONTNEED)`.
- It reuses `std.tiered.files`' open-with-fallback (`std/holon_std/tiered/files.py:48-53`).

*Write-back.*
- The SSD tier admits at host eviction.
- The host tier keeps a free-unit low watermark (`tier.host.reserve`, default 2% of its units). Below it, the core
  demotes host-level LRU leaves to SSD between steps, so an allocation rarely waits on a write.
- A host unit being written is freed when its ticket completes. An allocation that finds none free waits for one,
  never skips (D4).

*Persistence (`persist = close`).*
- An append-only manifest records, per block: parent record, tokens, slot ids per spec, payload slots, and a crc32.
- Each record is appended after its data's `fdatasync`.
- A clean `Engine.close` first writes back every entry not yet on SSD (≈ 16 GB of host tier at 2.6 GB/s ≈ 6 s). A
  crash keeps whatever the manifest committed; chains whose ancestors are missing are discarded.
- The manifest header holds the deployment identity, `plan_of(engine).plan_id` (`core/holon/core/plans.py:65-86`).
  Any change to the code, recipe, checkpoint or class invalidates the files. This is strict on purpose (open
  question 6).
- Keys are recomputed from tokens at restore, so Python's `hash` is never persisted.

*Why not X's I/O path.* `kernels.cpu_task`'s I/O workers serve in-graph reads signalled by the GPU, file → host only,
with tasks bound before the worker starts (`plugins/holon_plugins/kernels/cpu_task/plugin.py:53-57`;
`cpu_task.cu:521-545`). K's I/O is host-triggered between steps and writes too. K4 (FN, GLM) is where it is reused.

**D12 — No layer-wise loading in K.** With BLOCKED plus a fixed resume step, a 16K-token RAM load costs about one
decode step, ≈ 12 ms of TTFT. Layer-wise overlap would save at most that, and it forces eager or PIECEWISE prefill
(R §4.2 lines 590-591). It is built with PD in v2, as design §6.4 plans.

**D13 — Comparison method.**
- Same card, file, context, KV format and host; drafting off (the locks').
- `llama-server` runs at its defaults (`--cache-ram` 8 GiB) and with `--cache-ram` sized like Holon's host tier.
  Holon's `bench/backends/llamaserver.py` drives it.
- Interleaved ABBA rounds, ≥ 3. TTFT, TPOT p50 and p99, and throughput as `bench/README.md:72-95` defines them.
- Hit tokens per tier, load bytes, stalls and eviction waits come from Holon's `stats()`.
- Reference commits are pinned in the evidence.

**D14 — One core round, one propagation.** K's seams land on this branch. Tiers default off, so every existing lock
stays byte-identical (acceptance 5), and the locks are propagated once at the close. The 27B agent locks are
re-tuned with tiers and the new policies, with bounded verdicts (acceptance 6).

## 3. What Holon has today

| Piece | State (file:line) |
|---|---|
| Prefix index | Hash chain of blocks of `lcm(page, alignments)` tokens (64 for the GDN hybrids), keys `hash((prev or namespace, *tokens))`, tokens verified on match (`core/holon/core/prefix.py:48-71`). Entries are an LRU `OrderedDict` with per-spec device units, payloads and a child count (`prefix.py:29-46`). The namespace is `canonical_hash(spec)`, with "persistent tiers must add the checkpoint identity" (`core/holon/core/assembly.py:181-191`) |
| Eviction | LRU leaves whose units only the index holds, climbing while short (`prefix.py:89-104`). Payloads are dropped LRU from any entry when publishing needs a unit (`prefix.py:114-123`, `state.py:249-250`). Everything drops; nothing demotes |
| `could_fit` | Counts cached units with one reference left once the releasing requests are gone, regardless of tree shape (`state.py:366-385`). Sound only by the closure invariant (`prefix.py:8-10`). No test of its agreement with `evict` (`core/tests/test_prefix.py`) |
| Hits | Fixed point of the non-paged kinds' `hit_length` (`state.py:190-201`). Paged hits share units by reference. Payloads are adopted by a `pre_forward` MemOp, pinned until launch (`state.py:203-213, 271-274`) |
| Publishing | Every completed block, generated tokens included, at commit (`state.py:327-349`; `engine.py:562-565, 603-606`). Payloads are registered at commit if committed tokens cover them (`state.py:276-290`) |
| Recurrent checkpoints | `prompt_end` or `every:<n>`; a device pool of `max_seqs` units (`recurrent/kind.py:27-38, 46-55`); in-step publishes on the 64-token grid (`docs/m03-in-step-snapshot.md`) |
| Preemption | By recompute: `release` drops all units, `num_computed = 0` (`sched.py:188-195`, `state.py:387-398`). Published blocks stay cached; the live recurrent state is lost |
| BLOCKED | Exists; set only for a logits processor (`engine.py:323-324, 468-486`); skipped by `_prefill` (`sched.py:133-134`); polled at 5 ms (`engine.py:335-343`) |
| Host pools, async MemOps (X's S2) | Exact mmap + `cudaHostRegister` host pools against `host_memory` (`memory.py:218-248`; `platform/__init__.py:130-143`). MemOps with `after` / `lands_by` on one copy stream; a late copy stalls the `lands_by` step (`memory.py:162-206`; `executor.py:342-348, 387-389`) |
| Shared objects, step hooks (S0, S1) | `BuildCtx.shared` with `before_materialize` / `after_materialize` / `shutdown`; `before_launch` / `after_commit` (`hooks.py:150-187`; `engine.py:193-200`) |
| std.tiered | `TieredStore`, `ResidencyMap`, `DeviceResolver`, the `Tier` protocol (`std/holon_std/tiered/protocols.py:46-74`): an in-graph expert residency over read-only weight files. Reusable for K: S2 pools (core), `files.py`'s O_DIRECT open. Not reusable: the store, map and resolver (in-step, expert-shaped), and `cpu_task`'s I/O (D11) |
| Cut from ABI 1.0 | slot `tier`, `HitLen.cost`, `Payload.tier` (`docs/release.md:1738`) |
| ABI, budget | `ABI = "1.3"` (`core/holon/api/manifest.py:19`); `check_core_loc.py` 4,603 / 6,000; denylist without "tier", "page", "checkpoint", "host" |

## 4. Core seams (additive ABI minor → ABI 1.4)

None names a model, mixer, format or tier implementation. Each is proven first with the toy kind of
`core/tests/test_payloads.py` and toy tiers (a host pool tier, and a storage tier backed by a dict with a delaying
thread).

| # | Seam | Today → change | Toy proof | Lines |
|---|---|---|---|---|
| KS1 | **Tiers in the prefix index** (`Payload.tier`, demote) | Device copies only; eviction drops (`prefix.py:89-133`) → `Payload.tier: int = 0` (a field with a default). An entry holds its copies per tier, a level, children per level (`kids`), and a `busy` event while a copy in flight touches its units. Each tier has an LRU at its level. D6's eviction rule: demote to the next tier, or drop with the subtree; payload copies are dropped anywhere, VRAM first. Freeing units waits for `busy`. `could_fit` keeps its formula, now backed by the property test | the property test (D6); demote / promote bitwise; eviction waits for a delayed copy and never skips it | 140 |
| KS2 | **Loads and deterministic resume** (cancellable BLOCKED) | Hits adopt VRAM only; BLOCKED for logits processors only → admission promotes the hit's lower copies into index-owned VRAM units with MemOps (`after` = last launched, `lands_by = r`), or a storage read then a MemOp at its due step. `r` comes from the per-tier virtual schedule (D4); D = 0 under `stall_bytes`. The request is BLOCKED until `r`; hits on loading entries resume no earlier; when only load-blocked requests remain, the engine waits for the earliest. No new ABI (`Status.BLOCKED` exists) | resume at the same step with the toy tier's delay at 0 and 5 ms (run-identical); a late copy stalls `r`; abort and preemption of a loading request with a delayed copy (no unit reused before landing); liveness under random traffic | 110 |
| KS3 | **Release store and tentative payloads** (the `on_finish` publish) | Payloads are permanent from registration; release drops everything (`state.py:276-290, 387-398`) → optional hook `TentativePayloads.tentative(seq, at) -> bool` and attribute `keep_tentative: int` on payload kinds. The core keeps a request's last k tentative payloads per spec (a newer one drops the oldest if only the index holds it); they are not stored down until the request releases (finish, abort, preempt), when they become ordinary payloads and tiers' `admit("publish")` runs. `SeqView.branch_at: int` (core-implemented protocol: minor) | a toy kind publishing tentative payloads in decode: a preempted request resumes from the last one; finish leaves the last k; a payload of the depth-1 extra row registers only where committed tokens cover it | 60 |
| KS4 | **Copies: two streams, per-op events, batched cross-device copies** | One copy stream; per-segment, per-run copies (`memory.py:127-206`) → an H2D and a D2H stream; `Copies.add` returns each op's event; cross-device `copy_units` issues one batched DMA per op batch (`platform.copy_batch`, `cudaMemcpyBatchAsync`), keeping the loop as the fallback | fragmented units demoted and restored bitwise through the batch path; an H2D and a D2H in flight together; host calls per 1,000-unit load counted | 70 |
| KS5 | **`StateTier` protocol, slot `tier`, `EngineConfig.tiers`** | None (release §15.1) → slot `tier` (multi); `EngineConfig.tiers: list[str] = []` (planned; fastest first). `StateTier`: `bind(ctx, shapes) → units per shape` (host tiers request S2 pools here, after the shared objects' `before_materialize`); host tiers `pool(shape) → str`; storage tiers `write(shape, src_pool, src_units, units)`, `read(shape, units, dst_pool, dst_units)` → ticket (`done()`, `wait()`), `drop`; `admit(event: "publish" \| "hit" \| "evict", hits: int) -> bool`; `bytes_per_step`, `stall_bytes`. Startup checks: the first tier is a host tier; inclusion if it stores at publish; the rate floor (D9). `flush_prefix` clears every tier | a toy host tier and a toy storage tier through the protocol; inclusion and rate refusals reported | 60 |
| KS6 | **Persistent identity and restore** | Namespace from the recipe hash only (`assembly.py:188-191`) → the engine hands storage tiers the namespace and the deployment identity (`plan_of(engine).plan_id`) once built. `restore(identity, namespace) → records` (block tokens, parent, slots per spec, payload slots). The core re-hashes the chains, inserts entries at that tier's level, and discards records whose ancestors are missing | restart a toy engine with a persistent toy tier: resume bitwise; a torn last record is ignored; another identity is cleared | 40 |

≈ 480 lines. Not core: the tiers' storage, I/O threads, manifests and options; checkpoint policies; tune's rate
measurement; the traces.

Seam names follow X's outline (`docs/v1-offload.md:599-600`):
- `Payload.tier` and demote are KS1;
- cancellable `BLOCKED(future)` is KS2's BLOCKED plus a core-internal load, with no future type in the ABI;
- the `on_finish` publish is KS3's release store;
- `HitLen.cost` is dropped (D9);
- the name is `tier` throughout.

## 5. std library and plugins

- **`std.tier.host`** (`std/holon_std/tier/host/`).
  - Bind: one S2 host pool per shape (`ctx.pool(f"tier.host.{shape}", unit_bytes=…, segments=…, device="host")`,
    unmapped: only DMA reads it), sized by `tier.host.bytes`. The default is what `host_memory` has left after the
    shared objects' pools, split across shapes in proportion to their device bytes.
  - `admit`: `"publish"` (write-through), except tentative payloads (KS3).
  - Options `bytes_per_step` and `stall_bytes` (tune), `reserve` (the SSD watermark, D11).
  - ≈ 120 lines.
- **`std.tier.ssd`** (`std/holon_std/tier/ssd/`).
  - Options: `dir`, `bytes`, `io_threads`, `persist` (`off` \| `close`).
  - Slot files, `preadv` / `pwritev` tickets on its threads, O_DIRECT where aligned (D11), the manifest and
    `restore`.
  - `admit("evict")`.
  - ≈ 300 lines.
- **`std.state.recurrent`.**
  - `recurrent.policy` becomes a comma list of policies, each a built-in name or a `Ref`. The `CheckpointPolicy`
    protocol is `next(view) → (at, tentative) | None`. The kind takes the earliest position over its policies.
  - New built-ins `branch`, `answer:<k>` (sets `keep_tentative`) and `prefill:<n>`. `prompt_end` and `every:<n>` are
    unchanged.
  - `recurrent.checkpoints` stays the VRAM pool size. With a host tier the VRAM pool is a staging cache: copies with
    a host copy drop first.
  - ≈ 90 lines.
- **Bench.**
  - `bench/workloads.py` gains `agent_park` (§8).
  - `bench/tier_resume.py` (3a, 3b) parks a conversation by filling VRAM with a filler session, then resumes it.
  - The demo runner records `--raw` transcripts locally for K-M1; they are never committed.
- **Judge.** `holon_judge tiers`: the tier-invariance check (acceptance 2) and the plants of §6.

## 6. Numerics and judge

| Configuration | Class | Check |
|---|---|---|
| Host / SSD tiers at any capacity and store policy; restore after restart | exact (tier-invariant) | `holon_judge tiers`: a serial `after`-chained trace (8 sessions × 4 turns, history resent, a filler forcing every return through the tested tier) against the same trace with VRAM roomy: streams and per-request `cached_tokens` byte-identical, with ≥ 8 loads per tier counted. A concurrent variant is run-identical × 3 with ≥ 32 loads per tier. On CPU, a tiny model with tiers on vs off is bitwise. Qwen3-8B on the A5000, then the 27B; on the RTX 5080 the lock's trace |
| `branch`, `prefill:<n>` (in-step, prefill-produced) | as `prompt_end`: the payload is bitwise the recomputed state; resume is bounded vs recompute (L1.5) | the bounded multi-turn check (`docs/design.md:1297-1303`) with a shared-system-prompt session set added |
| `answer:<k>` (decode-produced) | bounded | the bounded multi-turn check (it resends history, generated tokens included); its own floor |

Snapshot traces of existing locks are untouched with tiers off. The tier comparisons compare output streams and
`cached_tokens`, not step lists: a BLOCKED wait can add empty steps in the serial trace (`judge/holon_judge/snapshot.py`
records a step list per `engine.step()`).

Plants (`judge/holon_judge/planted.py`), with verdicts predicted now. Each changes only configurations that use its
path; tiers-off configurations stay clean, and each plant reports how often it fired.

| Plant | Breaks | Predicted |
|---|---|---|
| `load_early` | a promotion's MemOp loses `lands_by`, and an unseeded 0–2 ms delay is put on the H2D stream | fails tier invariance (stale bytes read) or run-identical (a race); which one is recorded |
| `store_unwaited` | eviction frees VRAM units without waiting for their write-through copy; a delay on the D2H stream | the reused units overwrite the source before the D2H reads it, so a later RAM hit reads corrupt KV: tier invariance fails |
| `promote_wrong_unit` | a promotion copies the host units of the neighbouring page | tier invariance fails; bounded L2 fails |
| `resume_on_landing` | a BLOCKED request resumes at the first step after its copy lands instead of at `r` | run-identical × 3 fails on the concurrent trace; tier invariance on the serial trace passes (it cannot bite there — recorded) |
| `interior_drop` | host-tier eviction ignores leaf-first | the property test fails (orphans); outputs unchanged (lost hits only) |
| `kids_any_level` | VRAM leaf-ness counts children at every level (today's `children`) | the property test fails (`could_fit` yes, eviction short); the liveness test reports a decode request preempting itself on a pressure trace |
| `tentative_shift` | an `answer` payload is filed one block early (its state is 64 tokens later than its position) | bounded check fails (KL); tier invariance unaffected |
| `torn_manifest` | the manifest record is appended before its data's `fdatasync`, with a kill injected between them | the restart test fails: a restored block reads garbage and the resume differs from the pre-restart resume |

`could_fit` and liveness bugs show as lost progress, not wrong output, so they are caught by the property test and
`test_state_liveness.py`, not by the snapshot.

## 7. Memory fit (estimates; step 0 and tune measure)

| Model, file, card | VRAM (unchanged by K) | Host tier | SSD tier |
|---|---|---|---|
| Qwen3.8-27B UD-IQ3_S, q8_0 KV, `max_seqs` 4, RTX 5080 16 GB | 10.9 GiB weights; 58K KV tokens ≈ 1.9 GiB; 4 checkpoints ≈ 588 MiB (`docs/m03-qwen38.md:285`) | KV 34 KiB per token; checkpoint 147 MiB. A parked 16K conversation with 2 checkpoints ≈ 0.85 GiB, so 16 GiB holds ≈ 19 of them, or 490K KV tokens. Inclusion needs ≥ 2.5 GiB | 100 GB ≈ 110 parked 16K conversations |
| Qwen3.8-27B UD-Q4_K_M, RTX 5090 32 GB | from its lock | same unit sizes | same |
| Qwen3-8B Q4_K_M, bf16 KV, RTX 5080 | KV 144 KiB per token | 16 GiB ≈ 116K tokens | — |
| Qwen3-30B-A3B Q4_K_M offload + KV tier, RTX 5080 (report) | X's natural fit | experts ≈ 18 GB of `host_memory` (`docs/v1-offload.md:514`); the tier takes `tier.host.bytes` of the rest | — |

Notes on the table:
- **VRAM checkpoint pool.** `branch`, `answer` and `prefill` add VRAM payload demand. With a host tier the pool is a
  staging cache, so `max_seqs` units (4) is the starting point.
- **Tune** measures 4 against 6 units. Each extra unit costs ≈ 4.3K KV tokens, and VRAM KV now holds only live
  sessions.
- **Host RAM per desktop** is checked before scheduling and is not recorded here.
- **Pinning time.** Page-locking took ≈ 0.33 s per GiB on the A5000 host (`platform/__init__.py:134-136`), so a
  16 GiB tier adds ≈ 5 s to startup. This is re-measured on the bar desktops.

## 8. Step 0 — measure before freezing, and the predictions

On the RTX 5080 and 5090 desktops, one heavy job at a time under `MemoryMax`, with results in `evidence/`. K-M1's
analysis and K-M5 are CPU-only.

| # | Measure | Decides |
|---|---|---|
| K-M1 | Re-render divergence. The pi agent test re-run once with the capture proxy's `--raw` (local only): per turn, the longest common prefix of (previous prompt + generated tokens) and the next prompt, in tokens and relative to the answer's end; also with a template that strips reasoning | `answer`'s k; 3g's prediction; whether `answer` is on by default |
| K-M2 | Copy paths at KV chunk sizes (34,816 B, 65,536 B) and checkpoint units (≈ 3 MiB per segment), H2D and D2H, fragmented runs: today's loop, `cudaMemcpyBatchAsync`, and a block-contiguous host layout; host time per call | D10's path; `bytes_per_step`; `stall_bytes` |
| K-M3 | Four decode streams of the 27B with 0.5–2 GB H2D loads and 154 MB D2H stores in flight on the two streams | 3b's prediction; whether stores need a rate cap |
| K-M4 | NVMe `pwritev` / `preadv` of block-sized and checkpoint-sized units, O_DIRECT and buffered, 1–4 threads, with the engine loop's step time measured beside it (GIL) | D11's thread count and path; the SSD rate for D9 |
| K-M5 | CPU replay of the index with tiers on `agent_park`, `claude_code` and `agent`: hit tokens per tier, load bytes, evictions and stalls vs VRAM / RAM / SSD sizes, with step times from the locks | 3c–3e predictions; default `tier.host.bytes` |

**`agent_park`** (`bench/workloads.py`, seed 0). Four clients run closed-loop. Each owns three sessions and serves them
round-robin, one request at a time, so a session returns after its client's two other sessions have each taken a
turn.
- The 12 sessions share a 4,096-token system prompt. Each opens with a 512-token task.
- Then 10–14 turns per session, each turn adding:
  - a tool result: log-normal, median 384, capped at 2K;
  - an output of reasoning plus a tool call: log-normal, median 300, capped at 2K, released to the next prompt
    verbatim;
  - a gap of 0.5–3 s.
- Contexts end at ≈ 10–14K tokens; the parked total is ≈ 110K tokens.

**Predictions** (RTX 5080, 27B UD-IQ3_S; 47.5 GB/s H2D; prefill 2.2–2.5K tok/s; decode step ≈ 15.7 ms, tg128
63.75 tok/s from the same evidence file).

| Row | Prediction |
|---|---|
| 3a, 16K resume from RAM | 557 MB of KV + 154 MB checkpoint ≈ 15 ms of copy, D = 1 step; extra TTFT ≈ 15–31 ms against a bar of 22 ms + one step (pass if K-M2's batch path reaches ≥ 0.8× of 47.5 GB/s; fails with today's per-segment loop) |
| 3a, 32K from SSD | ≈ 1.27 GB at 2.6 GB/s ≈ 0.49 s + 27 ms H2D (bar ≈ 0.78 s + one step) |
| 3b | DMA on copy engines plus ≈ 5% of DRAM bandwidth during a copy (47.5 of 960 GB/s); loads under `stall_bytes` stall one step by ≤ 10% → TPOT p99 ≤ 1.1× (bar 1.2×) |
| 3c | VRAM-only: a returning session's chain has an LRU stack distance of ≈ 11 sessions × 9K ≈ 100K tokens > 58K, so it is evicted, and its own checkpoints are gone from the 4-unit pool → ≈ 9K tokens recomputed per turn, TTFT ≈ 4 s. Tiered: ≈ 300 MB loaded (≈ 6 ms, D ≤ 1) plus the append (≈ 400 tokens, ≈ 0.18 s) → TTFT p50 ≈ 0.25–0.4 s, ratio ≈ 0.07–0.1× (bar 0.4×). Throughput gain 1.3–2× from removed prefill (bar 1.2×) |
| 3d | `llama-server`'s save is synchronous (an observed 2.65 s stall, `docs/release.md:2690-2692`); whole-state blobs repeat the system prompt per conversation; its prefill runs at 0.87× Holon's (pp512 2,219 vs 2,544 in the same evidence) → ≤ 1.0× predicted, without a margin estimate until step 4 measures it (the bar is fixed now) |
| 3e | Sessions 1–2 recompute 16K (≈ 7 s); sessions 3–16 hit 16,384 → first-turn TTFT ≈ 0.1–0.2 s → p50 ratio ≈ 0.03× (bar 0.2×) |
| 3g | 44% (27,646 / 62,682) of uncached input → ≤ 10% if K-M1 finds the answers re-rendered verbatim up to their tool calls |

**Decision rules** (fixed now):
- A predicted row that step 4 measures below its bar goes to the owner, with the measured cause, before any
  retuning.
- `answer` is on by default only if K-M1 shows ≥ 50% of answers re-rendered verbatim to within one block of a kept
  checkpoint.
- D10 takes the batch path if it is ≥ 0.8× the blob rate at 34 KiB chunks, else the block-contiguous host layout.
- Write-through stays the default unless K-M3 shows stores cost > 2% decode.

## 9. Steps (milestone K)

| Step | Work | Checks | Days |
|---|---|---|---|
| 0 | K-M1–K-M5; the `agent_park` generator | evidence; the panel reviews the parameters they set | 2 |
| 1 | KS1–KS6 with toy proofs (ABI 1.4); the property test; `platform.copy_batch` | core tests; `check_abi --base` additive; LOC ≤ 5,100; denylist; tiers-off snapshots unchanged on the A5000 | 5 |
| 2 | `std.tier.host`; `branch`, `answer`, `prefill` policies; `holon_judge tiers` | tier invariance (Qwen3-8B, 27B, A5000); concurrent run-identical × 3; bounded multi-turn for the policies; plants 1–7 | 4 |
| 3 | `std.tier.ssd` with write-back, watermark demotion and persistence; restart test; **tag the std tier API** | tier invariance with SSD; restart byte-identical; `torn_manifest` | 4 |
| 4 | The gate on the RTX 5080 and 5090: 3a–3g; tune sets `bytes_per_step`, `stall_bytes`, `tier.host.bytes` and the checkpoint pool size; the 27B agent locks | acceptance 3; the locks' bounded and run-identical verdicts | 3 |
| 5 | Held-out plugins after the tag: `tiers.selective`, `tiers.buffered` | plugin-only diffs; tier invariance for each | 2 |
| 6 | Close: full suite, `run_all.sh`, panel code review, one propagation of the existing locks | acceptance 4–6 | 2 |

## 10. Risks and deferred

**Risks.**
- **The batch copy path is the critical path to 3a and 3b.** If neither `cudaMemcpyBatchAsync` nor the
  block-contiguous layout reaches 0.8× at KV chunk sizes, loads fall back to whole-step stalls. 3b would then miss
  for 32K resumes. Hence K-M2 runs first.
- **Re-rendering decides `answer`'s worth.** If agents' templates re-serialize tool calls or strip reasoning, the
  answer checkpoint helps little. `prompt_end` and `branch` still carry the gain (K-M1, 3g).
- **Host RAM is shared with experts.** On offload deployments the host tier gets what X leaves. The coexistence row is
  report-only.
- **SSD wear.** Write-back only, never through, keeps writes to what falls out of RAM. `persist = close` writes the
  rest once per clean shutdown.
- **Host CPU in the engine loop.** Index operations grow with the tiers' entry count (≈ 10–50K blocks). The per-tier
  LRUs keep each eviction's scan bounded by its own tier. K-M5 measures the per-step host time.

**Deferred.**
- `HitLen.cost` (D9);
- layer-wise loading and per-layer events (D12);
- K4 (FN, GLM);
- root pinning and ARC (as tier or policy plugins after the tag);
- a save / restore API;
- checkpoint codecs;
- MTP verify-row checkpoints;
- Maya's multi-session ratio (GLM);
- VMM rebalance (plan §10).

## 11. Open questions for the owner

1. **Write-through by default, host inclusive of VRAM** (D5). This reverses the research note's write-back proposal
   (R line 578) to keep stalls out of eviction and make preemption swaps free. Agree?
2. **Drop `HitLen.cost`** (D9). It was on X's outline list, but no K tier is slower than recompute, and a startup rate
   check refuses one that would be. Agree?
3. **`branch` and `answer` on by default in the 27B agent locks** after the bounded check (`answer` subject to K-M1).
   This moves those locks' snapshots, with bounded L2 approval at step 4.
4. **References.** Is `llama-server` with `--cache-ram` and `--ctx-checkpoints` the gating reference (3d), with SGLang
   HiCache and vLLM's offload connector report-only on Qwen3-8B?
5. **Move Maya's multi-session ratio** (`docs/v1-offload.md:598`) to the GLM milestone? K reports Holon's own
   N-session ratio.
6. **SSD persistence keyed on `plan_id`.** Any code change invalidates the files, which is conservative. Would the
   owner rather key on the recipe and checkpoint only, accepting `bounded` resumes across Holon upgrades?
7. **Gate setup.** The new `agent_park` trace as the gate trace, and one pi re-run with local `--raw` capture for
   K-M1?

## 12. Review log

### Rev 1 (2026-10-18): draft, not yet reviewed

How the requirements X's panel left for K (`docs/v1-offload.md` §11) are answered:

| Requirement | Answered in |
|---|---|
| A10: demoted children must not keep a VRAM parent unevictable while `could_fit` counts it freeable; VRAM entries prefix-closed; host evicts leaf-first; a property test that `could_fit` and evict agree | D6 (levels, per-level children, per-tier leaf-first LRU), KS1, the property test, plants `interior_drop` and `kids_any_level` |
| Write-safe cancellation | D7: loads land in index-owned VRAM units; any free of a unit with a copy in flight waits for it; plant `store_unwaited` |
| C9b: deterministic resume at a fixed step offset | D4's virtual schedule, KS2, plant `resume_on_landing` |
| C14: acceptance that bites | 3a (RAM-hit TTFT bound, with the step-granularity term made explicit), 3b (TPOT p99 ≤ 1.2× during loads), 3c–3d (agent trace vs VRAM-only and `llama-server`); Maya's ratio → GLM (open question 5) |
| Core seam: `Payload.tier`, `HitLen.cost`, demote, cancellable `BLOCKED(future)`, an `on_finish` publish; name `tier` | KS1 (`Payload.tier`, demote), KS2 (BLOCKED with a core-internal load), KS3 (release store), D9 (`HitLen.cost` dropped, open question 2); slot and field named `tier` |
| Plan §10's end-of-answer checkpoint | D8 `answer:<k>` with KS3, K-M1 |
