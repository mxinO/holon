# v1 — expert offload: MoE models larger than VRAM on one consumer GPU

Status: **draft rev 2** (2026-10-17), for the panel's re-review. Rev 1 drew 52 findings (A 20, B 15, C 17); §12
logs each and where rev 2 answers it. No code yet. Evidence base: the local research note
`docs/brainstorm/v1-offload-research.md` (not published; "R §n"), which cites every reference engine by `file:line`
in its local clone; Project Maya is R §3.1b.

Scope: plan §10 (v1, "Expert caching/offload"), with the owner's direction of 2026-10-16/17:
- support **the most popular strategies of today**, with **Strata** as the primary reference, FreeToken second, and
  llama.cpp's static split as the baseline and fallback only;
- **a new strategy must be easy to add as a plugin**: host-side policies between steps in Python, in-step policies
  as kernel plugins;
- **target models: Qwen3.8-Flash-Next** (compared with Strata) and **GLM-5.3-Flash** (compared with Project Maya, the
  Strata-derived GLM engine). Qwen3.5-122B-A10B leaves the plan.

**The v1 offload program is four milestones** (D2), so that each closes on its own evidence and one branch changes
the core at a time:

| Milestone | Doc | Closes on | Core |
|---|---|---|---|
| **X — expert offload** (this doc) | `docs/v1-offload.md` | the development gate: Qwen3-30B-A3B on the RTX 5080, at its natural fit and at a capped ≈ 25% fit | S0–S5 (§4) |
| K — KV and checkpoint tiers | own doc, after X | resume from RAM / SSD, exact; agent-trace TTFT and concurrency | prefix-index tiers (§11) |
| FN — Qwen3.8-Flash-Next | own doc, in parallel with X | the model on Holon, then offload vs Strata (bars §1.4) | hyper-connection residual; listed now (§4.7) |
| GLM — GLM-5.3-Flash | own doc, after FN's shared units | the model on Holon, then offload vs Project Maya (bars §1.4) | none expected beyond FN's |

The VMM ledger and live rebalance (plan §10's third offload line) is X's second wave or its own milestone (D13).

## 0. The references in one table

| | Strata (primary) | Project Maya (GLM) | FreeToken | llama.cpp |
|---|---|---|---|---|
| Models | Qwen3.8-Flash-Next only | GLM-5.3-Flash only | many | many |
| Expert tiers | VRAM cache over all experts pinned in RAM (or mmap of the files when RAM is short) | exclusive VRAM / pinned RAM / NVMe: each expert in exactly one; RAM sized from free RAM less 6 GB | VRAM cache over pinned RAM | static per-layer CPU or GPU; new `--moe-cache-mib` LRU |
| Residency | seeded from a routing profile; every 4 windows up to 96 swaps by decayed routing counts (×0.7, gain ≥ 1.5); the victim is unmapped before its slot is overwritten | profile warm start; the device promotes a RAM or disk miss into one of 3 spare slots per layer; every step refills the spares by evicting the layer's LRU expert, demoted to RAM if hotter than RAM's victim, else disk-only | in-graph timestamp LRU, fetch on miss | fixed (LRU in the new cache) |
| Decode miss | a share over PCIe to the GPU, the rest computed by the CPU in RAM, through in-graph spin waits; share `L/(L+1.85R)` refit from live timings | RAM: the CPU takes the coldest by count (a per-miss-count table, re-timed live), the GPU gathers the rest zero-copy; disk: the GPU waits while a host thread reads the expert from the GGUF with `O_DIRECT` | offload / CPU / hybrid by a bandwidth benchmark | CPU computes |
| Prefill | non-resident experts stream through a ring, next layer during this layer's attention; a CPU share for small chunks | chunks up to 32K, buffers borrowed from the expert pool's tail; resident experts in place, routed misses staged, likely RAM experts prestaged | every expert of the layer through a double buffer, evicting cached slots | selected experts uploaded for batches ≥ 32 |
| Output-changing defaults | route tail skip (missed low-rank experts dropped, +10–24% decode); timing-driven splits; CPU/GPU rounding moves with placement | none approximate; runs differ by timing (17 of 70 stream pairs identical by default, 70 of 70 with its fixed-placement recipe); MTP off on one GPU | — | — |

Both leaders are single-model engines with large switch sets (Strata: 529 `STRATA_*` variables). Holon's offer is
the same strategy set as plugins over any MoE model plugin, **exact by default, with reproducible placement**.

## 1. Goal and acceptance

Run MoE models whose experts do not fit in VRAM on one consumer GPU, as fast as the strongest single-model engine
for that model, with exact defaults and reproducible runs.

1. **Strategies are plugins** (D3, D16).
   - Shipped in X: placement policies `static` (F1), `profile_decay` (F2–F3) and `lru` (F4); miss paths PCIe fetch
     (F5a) and CPU compute (F5b) with a per-miss-count split; the SSD tier (F9); prefill streaming (F7).
   - **Held-out proof:** the std offload API is tagged at the end of step 3. After the tag, three strategies land as
     commits whose diff touches only their own plugin directory and tests (no core, no std diff; checked with the
     existing `check_core_diff.sh` and `git diff --stat`): **F10** promote-on-fetch (host and in-graph parts, the
     hardest case), **A1** route tail skip (`approx`), and **one strategy this document does not name**, chosen from
     M6's replay (e.g. LFU or ARC).
   - Every core seam (§4) is proven first with a toy plugin in `core/tests`.
2. **Exactness** (D4, §6):
   - **every placement is exact on the GPU path:** with the CPU share off (the default unless D4's rule picks the
     split), outputs are bitwise independent of which experts are resident, of the arena size and slot order, and of
     the policy — snapshot traces byte-identical to `std.moe@1` all-resident, with adaptive placement **on** in the
     snapshot;
   - with the CPU split on: run-identical over 3 fresh engines on the snapshot trace, and bounded (L2 within budget)
     against a path-aware floor (§6).
3. **Development gate** (milestone X's close), Qwen3-30B-A3B Q4_K_M on the RTX 5080, same file and host, no drafting,
   medians of interleaved rounds, at the natural fit (≈ 47% of experts in VRAM at 32K) **and** capped to ≈ 25%:

   | Against | Decode | Prefill |
   |---|---|---|
   | FreeToken (`auto` and `offload`) | ≥ 1.0× | ≥ 1.0× |
   | llama.cpp static split (`--fit`, tuned `--n-cpu-moe`) | ≥ 1.5× | ≥ 1.0× |
   | llama.cpp `--moe-cache-mib` | ≥ 1.1× | ≥ 1.0× |
   | Holon F1 (static) vs llama.cpp's static split | ≥ 0.95× | — |

4. **Program bars** (acceptance of FN and GLM, recorded here so the offload design answers to them; proposed, owner
   to confirm). Like-for-like only: Holon's exact defaults against each reference's no-approximation settings, and
   Holon's opt-in `approx` against the reference's defaults:

   | Against | Model, card | Decode | Prefill |
   |---|---|---|---|
   | Strata, no approximation (`STRATA_ROUTE_TAIL_SKIP=0 STRATA_IQ_MT_MIN=1`) | Flash-Next IQ2_XS on RTX 5080; IQ3_S on RTX 5090 | ≥ 1.0× | ≥ 0.9× |
   | Strata defaults vs Holon with A1 | same | ≥ 1.0× | ≥ 0.9× |
   | Project Maya defaults (no approximation by default) | GLM-5.3-Flash Maya-S on RTX 5090 | ≥ 1.0× | ≥ 0.9× |

   Holon-exact against Strata's defaults is reported, not gated. Plan §9's kill line (< 0.8× → rework the offload
   path before adding features) applies to the like-for-like rows.
5. **Memory:** pinned host memory is allocated exactly (no power-of-two rounding) and accounted against
   `host_memory`; VRAM and host splits are pinned by the lock; startup refuses with a per-pool report when a pinned
   split does not fit.
6. **Core:** only the seams of §4, each an additive ABI minor (ABI 1.2), toy-proven; ≈ 400 new code lines
   (estimates in §4), core ≤ 6000 (4248 today). Suite time grows only by the new plugins' own tests.
7. **Locks:** offload deployments get locks on the classes they run on; the existing locks are propagated once, at
   X's close (D15).

Non-goals of X: KV and checkpoint tiers (milestone K); the two model plugins (FN, GLM); multi-GPU layer split and
peer-GPU tiers; dense (HF bf16) expert offload — v1 offloads quantized experts adopted as `QTensor`s; EXL3;
predictive prefetch (D11); a CPU kernel matching the GPU's rounding (D7).

## 2. Decisions

**D1 — Strategy set: Strata's and Maya's, exact by default.** Classes: **H** host policy (Python, between steps),
**D** in-step device kernel (kernel plugin), **C** CPU kernel, **std** library.

| # | Strategy | Reference | Class | X default |
|---|---|---|---|---|
| F1 | Static split: a frozen residency map | llama.cpp `--n-cpu-moe` | H | fallback without a profile |
| F2 | Profile-seeded expert cache: VRAM slots filled from a ranked (layer, expert) profile | Strata X1, Maya G6 | H | on |
| F3 | Adaptive swaps by decayed routing counts on a fixed cadence of committed steps | Strata X2 | H | on |
| F4 | LRU fetch-on-miss, logical-step timestamps, victims chosen in the graph | FreeToken, llama.cpp cache | D + H | comparison arm |
| F5a | Misses fetched over PCIe by an in-graph zero-copy gather into staging slots | Strata X5 mode 2, FreeToken | D | on |
| F5b | Misses computed by the CPU in RAM through the in-graph handshake; which misses go to the CPU is the provider's rule (Strata: last in routing order; Maya: coldest by count) | Strata X4–X6, Maya G2, G9 | D + C | by D4's rule |
| F7 | Prefill: non-resident experts stream through a ring, next layer during this layer, when D8's time rule says so; else routed-only staging | Strata X8, Maya | D + std | on |
| F8 | Prefill CPU share for small chunks | Strata X9 | C + H | only if step 0 shows ≥ 15% |
| F9 | SSD tier: experts outside the RAM tier read from the GGUF's byte ranges with `O_DIRECT` by dedicated I/O workers into pinned staging | Maya G4–G5 | std + C | on when RAM is short |
| F10 | Promote-on-fetch into device-owned spare slots per layer; each step boundary refills the spares | Maya G1–G3 | D + H | held-out plugin (acceptance 1) |
| A1 | Route tail skip; Maya's miss / disk skips | Strata A1 (default there), Maya AG1–AG3 | D + H | off; `approx` |

**D2 — Targets, the development vehicle, and four milestones.**
- **FN: Qwen3.8-Flash-Next** (48 layers, 512 experts top-10 + shared, QSA + indexer alternating with GDN,
  hyper-connections, the PLE n-gram table on SSD, MTP), compared with Strata on its own files.
- **GLM: GLM-5.3-Flash** (45 layers: 11 MLA + DSA indexer (top-2048), 34 KDA; 288 experts top-8 + 1 shared, the
  first 3 dense; mHC; sigmoid `noaux_tc` routing with a SwiGLU clamp of 10), compared with Project Maya on Maya-S.
  Every Maya quant and Unsloth's UD-IQ4_XS use upstream ggml types only, all in `kernels.ggml_cuda` (IQ1_M aside).
  GLM brings DSA forward from plan v2; its router and clamp are new kernel facts (`moe_topk` variant, a
  `linear_experts` fact), not core.
- **Development vehicle: Qwen3-30B-A3B Q4_K_M** (18.6 GB; 48 layers × 128 experts top-8) on the RTX 5080, where it
  does not fit: its plugin (`models.qwen3_moe`) and its RTX 4090 locks exist, so offload starts now and the 4090 runs
  the all-resident reference of the same file. FreeToken and llama.cpp run it too. (Rev 2 named Qwen3.5-35B-A3B and
  said its plugin exists; it does not — `models.qwen3_5` has no MoE. Qwen3.6-35B-A3B, GDN + MoE with a shared expert,
  is FN's natural first step and lands there.) The lead chose the vehicle (the owner was asked; reversible).
- **Why four milestones** (panel B12): X must not wait on two model plugins; KV tiers share only S2 with experts and
  carry their own index invariants (§11); the model milestones need core of their own (§4.7), and one branch changes
  the core at a time. X's seams are designed with the model milestones' needs listed, so no seam is redesigned later.

**D3 — Mechanism vs policy; what a strategy plugin is.** In-step work (resolve, fetch, expert GEMM over slots, CPU
compute, row patch, combine) is native, selected through op contracts. Its parameters — the residency map, the split
table, quotas — live in device or mapped buffers owned by the offload store. Between steps, pure-Python policies read
the step's routing counts and return **edits** (D4). Plugin kinds, all chosen by **open plugin options holding a
`Ref`** (no new core slots; the core never calls these, release §15.1's reason for cutting the slots stands):

| Kind | Option (example) | Contract | Instances in X |
|---|---|---|---|
| expert store | `moe.store = "std.moe.offload@1"` (absent: resident) | a std unit `ctx.unit(Ref)` binds | resident, offload |
| placement policy (host) | `moe.placement = "plugins.placement.profile_decay@1"` | `PlacementPolicy` protocol in `std.tiered`: `seed(profile, budget) → edits`, `after_commit(counts, view) → edits`, `device_state() → {name: spec}`, `resolve_variant: str` | static, profile_decay, lru |
| miss path (device + CPU) | kernel selection through facts | `moe_resolve@1`, `expert_fetch@1`, `cpu_experts@1`, `moe_patch_rows@1` (§5) | fetch, CPU |
| expert tier (host) | `moe.tiers = ["std.tiered.pinned@1", "std.tiered.gguf_direct@1"]` | `Tier` protocol in `std.tiered` | pinned RAM, SSD |
| route transform (`approx`) | `moe.route_transform = Ref` — explicit option only, never by kernel priority | `moe_route_transform@1` between `moe_topk` and `moe_resolve` | none in X before the tag; A1 after |

A new policy, tier or transform is a plugin naming its own Ref; a policy that needs device state declares it
(`device_state`) and names the `moe_resolve` provider it pairs with (`resolve_variant` is a kernel fact), so F4 and
F10 need no std change.

**D4 — Exact by default; reproducible placement.**
- **Placement-invariant GPU path.** An expert's output must not depend on how its GEMM is launched. The vendored
  expert-id mmq switches to stream-k when tile efficiency is < 90%, splitting K at points that depend on the slot
  count (`mmq.cuh:1471-1478, 1093-1094`; on the RTX 5080 it engages at 16–85 slots). X adds a vendor patch forcing
  tiled mmq on the expert-id path, and an L0 test that an expert's rows are bitwise equal across slot counts
  {E, E+1, 16, 74}, slot permutations and launch groupings, for mmvq (both kernels) and mmq.
- **The default lock computes every expert on the GPU** (F5a only). Then every placement policy is exact, and
  adaptive placement runs in the snapshot (refines design §7.10's "adaptive policies frozen", which assumed a
  placement-dependent consumer). `holon_tools tune` measures the best F5b split against F5a per (model, class); the
  lock takes the split only if it gains ≥ 5% decode, and is then `bounded` (Maya measured its CPU lane at 4–10% on a
  5090, R §3.1b).
- **Decisions use committed counts, take effect at fixed steps.** A policy's edits computed after commit c are
  applied at launch c+2 (both overlap paths: at depth 1 commit c follows launch c+1; on the synchronous paths it
  precedes it — the fixed +2 makes them agree). An edit is **two-phase** (Strata unmaps first, `generate.cpp:8895`):
  at c+2 the device apply kernel **unmaps** the victim (its routes become misses), the copy into the freed slot is an
  async MemOp ordered after that step's last read of the slot, and at c+2+d (d fixed, default 2) the apply kernel
  **maps** the new expert after the stream waits on the copy's event — a stall if it has not landed, never a
  timing-dependent result. The ring's lent slots and RAM-tier victims follow the same rule.
- **One writer per map region.** The residency map is one device buffer written only by the apply kernel (host
  edits, stream-ordered) and by in-graph resolvers for regions the policy declares device-owned (F10's spares, F4's
  victims). Host edits are **conditional** (applied only if the slot still holds the expected expert, else skipped
  and reported); the device's own changes return in a per-step delta log with the counts, so the host's view is the
  device's, lagged by a fixed number of steps.
- **Split decisions are deterministic.** The F5b split is a per-miss-count table `plan[n]` = the number of a layer's
  n distinct misses the CPU takes, frozen by tune (C4); ties by best routing rank, then expert id. It is
  deployment-level (a device buffer), never per request: misses are grouped across rows (D12).
- **Pinned sizes.** The lock pins the VRAM expert-slot counts and the RAM-tier slot counts (startup refuses if they
  do not fit), so placement never follows free memory at start (Maya's top non-determinism, R §3.1b).
- **Padding rows** (graph buckets) are masked in `moe_resolve`: no fetch, no CPU job, no counts.

**D5 — The offload store, one per engine.** `std.moe@1` keeps its interface and binds its store through
`ctx.unit(Ref)` from the `moe.store` option. Every MoE layer (the MTP layer's included) gets the **same** store
object through `BuildCtx.shared` (S0). The store owns:
- **arenas per slot class** — one per (role, ggml type, shape), global across layers (a GGUF mixes types across
  layers and roles: Q4_K_M has Q6_K `ffn_down`; Maya-S has IQ2_XXS gate/up and IQ2_S / IQ3_XXS down). A class's
  arena is `[S, …]` with the contiguous expert stride `linear_experts@2` already assumes; its top k slots are **null
  slots** (zero weights), one per routing rank, so ids stay unique per token (ggml's id helper keeps one entry per
  (token, expert): a shared null slot corrupts rows on the mmq path, panel A2); then staging slots for fetched misses
  (k × the largest row count of a graphed step, reused layer after layer), then cache slots;
- the **residency map** `[L, E, roles] → slot | −1−tier` (tier codes say where a miss lives: RAM, SSD);
- the host tiers, the policy object, the split table, the per-step count and delta buffers (double-buffered per
  launch, so a prefill-only step's telemetry is never overwritten before its commit reads it).

Per MoE layer the graph runs:

```
router → moe_topk → [moe_route_transform] → moe_resolve → [publish CPU jobs; signal] → expert_fetch
       → linear_experts(gate_up, class arena) → linear_experts(down, class arena) → [wait] → moe_patch_rows
       → moe_combine
```

Brackets are absent in the default lock. Shared experts stay resident. The plan is computed on the device
(FreeToken style); M5 measures a host-planned variant (Strata's flag A), which would be another `moe_resolve` provider.

**D6 — The CPU handshake is a kernel plugin, not core** (panel A12, B2; drift from design §7.4). S6's protocol — one
word per layer in mapped pinned memory, spin-kernel wait (memop wait on displays with a watchdog), a persistent
native worker, zero-copy payloads — ships as `kernels.cpu_task` (`[build]`, prebuilt like the other native plugins)
providing `cpu_task@1` signal / wait ops and a worker service, shared through S0. The core keeps only the lifecycle:
shared objects are shut down after `drain` (or abort, drain, stop), and the engine can pin its loop to a reserved
core (`loop_core`). Worker counts are tune's (S6 saturated DRAM with 2–4 threads); F9's I/O workers are separate
from compute workers, so a blocking read never stalls a layer's CPU experts.

**D7 — CPU kernels: ggml-cpu, bitwise per token and per thread count.** `kernels.ggml_cpu` vendors ggml-cpu (MIT) at
`kernels.ggml_cuda`'s pin and runs `vec_dot` over GGUF blocks in place. `cpu_experts@1` requires a token's result to
be bitwise the same alone or grouped (2–8 tokens) and at 1 or N threads (rows split across threads, never K); L0
checks both. Faster rows (Strata's multi-token i-quant kernels, MIT, design §5.6) are admitted only if bitwise equal
to ggml-cpu's per-token dot (Maya checks this at first use). ISA tiers are kernel constraints (`Constraint.cpu_isa`);
ggml-cpu's reductions differ by ISA, so a lock that uses F5b has the CPU's ISA tier in its class (S5). Dropped from
rev 1: a CPU kernel matching the GPU's rounding — it would need the GPU's SwiGLU and q8_1 quantization rebuilt
without fast math and per-path reduction orders (panel C2); F5a-by-default makes it unnecessary for exactness.

**D8 — Prefill streams what is missing, when it pays in time.** Per chunk and layer, F7 streams layer l+1's
non-resident experts through a ring while layer l runs if their bytes over the measured PCIe rate fit under layer l's
compute at that chunk size, or beat routed-only staging's exposed transfer time; else routed-only staging (C6:
coverage passes 50% at ≈ 64 tokens for top-8-of-256, so a coverage rule would stream almost always). Resident
experts are never re-sent (FreeToken's flaw). Ring slots are lent from the store's own arena tail and returned under
D4's two-phase rule — plugin-internal. Maya's prestage of likely RAM experts is an F7 option measured in step 4.
Prefill stays eager.

**D9 — VRAM: the expert arena fills what is left above a paged floor.** A plugin pool may declare `fill=True` with a
priority: after weights, workspaces, graphs, slot state and fixed plugin pools, a deployment `paged_min_tokens`
floor goes to paged state, then fill pools by priority up to their `pref_units`, then paged state takes any
remainder. The floor defaults to `max_model_len` for one sequence; benchmark locks name their context. Experts take
everything above the floor (C15: returning VRAM to idle KV loses to Strata and Maya on single-user traces); the floor
is the knob, set by L4. The lock pins the resulting slot counts (D4).

**D10 — Host tiers: exact allocation, exclusive when RAM is short.** Host pools are allocated once, before weights
load and outside the device profile cycle, with `cudaHostAlloc` or `mmap` + `cudaHostRegister` (torch's pinned
allocator rounds to a power of two: 86 GB would pin 128 GiB, panel A6), and accounted against `host_memory`. The
RAM tier holds all experts when they fit; otherwise it is **exclusive of VRAM** (an expert lives in VRAM, RAM or SSD,
moved by D2H demotion as Maya does): with an inclusive tier, GLM on a 64 GB host sends ≈ 7.5% of routes to SSD
instead of ≈ 2.0% (C8). F9 reads the GGUF's byte ranges with `O_DIRECT`; weights load the same way, so the page
cache never double-counts the file. It needs each expert slice's file extent from the weight source (S4).

**D11 — No predictive prefetch.** Strata's Foresight measured 0.91–1.00× on every box (R §3.1 E1) and Maya's
read-ahead lost too (R §3.1b EG1). Re-open only if M6 shows next-layer recall ≥ 85% and M1 ≥ 30% idle PCIe time in
decode.

**D12 — Batching is first-class.** Misses of all rows in a step (decode, verify windows, mixed steps) are grouped per
expert before the fetch and CPU paths, so a miss is paid once per step. Maya sends each request's routes through the
tiers one after another; Strata turns batching off by default.

**D13 — VMM elastic pools are the second wave** (Strata X3: KV grows into the expert arena and back without
recapture). After F1–F9, in X if time allows, else its own milestone; `quiesce` arrives with it (not before: X's map
changes are stream-ordered).

**D14 — Comparison method.** Same card, file, context and KV dtype, same host; drafting matched. Drafting is
placement-aware: on one offloaded GPU MTP loses on GLM (consecutive tokens share ≈ 1 of 8 experts per layer; Maya
turns it off), while Strata gains 1.6–1.8× from it on Flash-Next; the lock decides per (model, class). Every reference
runs its defaults and its no-approximation settings; each comparison pins the reference's commit. Metrics follow
Strata's tables: decode tok/s at 1K / 4K / 32K / 128K (256 tokens, greedy), prefill at 1K / 4K / 32K, the agent trace
(TTFT / TPOT p50 and p99, wall time), expert hit rate and fetch / CPU / SSD shares, peak VRAM and RAM, L2 quality,
run-to-run identity. Reference engines build in `bench/.envs/`.

**D15 — One core round, one propagation.** X's seams land on this branch; the 22 existing locks are propagated
once, at X's close. FN's core seam (§4.7) follows on its own branch after X merges.

**D16 — The held-out proof replaces "designed-together" plugins** (panel B6): see acceptance 1.

## 3. What Holon has today

| Piece | State |
|---|---|
| MoE | `std.moe@1` (`std/holon_std/moe/units.py`): f32 router, `moe_topk@1` (softmax, unique ids), gate/up merged with SwiGLU and down as `linear_experts@2` over `[E, …]` stacks in VRAM, `moe_combine@1` in f32; named in M0.2 (`docs/m02-moe.md` D2) as the offload seam |
| Expert kernels | `kernels.ggml_cuda`: mmvq up to llama.cpp's `use_mmvq_id` row limit, else mmq; GGUF types incl. IQ1_S / IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S / IQ4_NL / Q2_K–Q6_K / Q8_0 |
| CPU handshake | spike S6 only (`spikes/s6_cpu_handshake/README.md`): 1.9–2.8 µs per spin handshake on the desktops, captured in FULL graphs, drain-first stop |
| Memory | `Ledger` with fixed device pools, sized after weights load and re-materialized around the profile; `BuildCtx.pool(name, unit_bytes, segments, min_units, pref_units)`; a second request of one pool name raises; `pinned_empty` uses torch's pinned allocator |
| MemOps | `MemOp(src_pool, src_units, dst_pool, dst_units, phase)` between equal-shaped device pools, eager, on the step stream |
| Sharing between units | none: `BuildCtx.unit` builds a new instance per call; `static(name)` shares tensors by name |
| Lifecycle | `ShutsDown` on logits processors, called by `Engine.close` |
| CLI | refuses a checkpoint larger than VRAM before fetching (`core/holon/core/cli.py`) |
| Fingerprint | GPU SKU, SM, VRAM, toolchain; no CPU facts |
| Cut from ABI 1.0 | slots `eviction_policy`, `placement_policy`, `tier`, `mover`, `connector`, `codec`, `checkpoint_policy`; `HitLen.cost`, `Payload.tier`; `Pool.base`, `capacity_dev`, `device`, `segments`; `Constraint.cpu_isa`; prop `batch_invariant` (`docs/release.md` §15.1) |

## 4. Core seams (additive ABI minors → ABI 1.2)

None names experts, KV or formats. Each is proven first with a toy plugin.

| # | Seam | Today → change | Toy proof | Lines |
|---|---|---|---|---|
| S0 | **Shared objects and their lifecycle** | none → `BuildCtx.shared(key, factory)`: one object per engine per key, built on first request; hooks it may implement: `after_materialize(ctx)` (after the ledger's final sizes, before graph capture: seeding), `shutdown()` (after `drain`; generalizes `ShutsDown`); EngineConfig `loop_core` | two toy units share one object; it seeds a pool before capture and stops after drain | 60 |
| S1 | **Step hooks** | planner `prepare_device` only → a shared object may register `before_launch(step) → (writes, mem_ops)` (device writes and MemOps enqueued before the replay, stream-ordered) and `after_commit(step, telemetry)` (declared telemetry buffers, copied D2H with the step's tokens into a per-launch double buffer) | a toy counter and a conditional map edit applied at launch c+2 under depth 1 and on the synchronous paths alike | 80 |
| S2 | **Host pools and async cross-device MemOps** | device pools only; eager equal-shape MemOps → `BuildCtx.pool(…, device="host", mapped=True)`: allocated before weights load with `cudaHostAlloc` or `mmap` + `cudaHostRegister`, exact bytes against EngineConfig `host_memory`, a device alias; a MemOp may cross host ↔ device and carry `after` (an event of a step) and `lands_by` (a later step whose launch waits on it); the core runs it on a copy stream | a toy unit demotes a unit to host and back, bitwise; a late copy stalls the `lands_by` step, never skips it | 150 |
| S3 | **Fill pools** | fixed plugin pools → `BuildCtx.pool(…, fill=True, priority=p)`, filled after EngineConfig `paged_min_tokens`, capped at `pref_units`; the lock pins the resulting units | a toy fill pool and paged state split a budget; a pinned count that does not fit refuses at start | 40 |
| S4 | **Weight extents** | `QTensor` parts carry tensors only → an optional per-part `extent(path, offset, nbytes)` for unmodified file slices (absent when a kernel's `prepare` repacked them) | the toy weight source reports extents that read back bitwise | 20 |
| S5 | **CPU facts** | `Fingerprint` has none → `Fingerprint.cpu_isa` (a tier name such as `x86-avx2`, `x86-avx512`; never a CPU model) and `Constraint.cpu_isa`; a lock class gains the tier only for deployments that run CPU kernels | a toy CPU kernel selected by tier | 30 |
| — | **CLI size check** | refuses checkpoints larger than VRAM → skips the refusal when the resolved lock or options select an offload store, and checks host memory instead | — | 10 |

≈ 390 lines. Not core: the store, `TieredStore`, tiers, residency maps, policies, calibration, route traces and
profiles, the ring (design §6.5). No slot is re-added; `quiesce`, layer-wise loading and per-layer ready events stay
planned (D13, milestone K).

### 4.7 Core needs of the model milestones (listed now, built there)

- **FN, GLM: hyper-connection residuals.** `Model.trunk` and `tap` pass `[T, H]` hidden states; hyper-connections
  carry `[T, n, H]` streams (n = 4) through every layer. FN's doc designs the shape change (a residual-width fact on
  the trunk); it is FN's core seam.
- **FN: the PLE n-gram table** (28.8 GB, a few rows per token) reads through S2's host pools and the SSD tier's I/O
  workers; no further seam expected.
- **FN, GLM: sparse attention (QSA, DSA)** uses `page_group` and the planner path that exist; KV streaming (Strata
  C4) belongs to milestone K.

## 5. std library and kernel contracts

**`std.tiered`** (std): `TieredStore` over tiers, `ResidencyMap` (device buffer + conditional edits + delta log), the
`PlacementPolicy` and `Tier` protocols, `DeviceResolver`; profile and trace tooling (`route_dump` from the
`after_commit` counts; `holon_tools profile` ranks (layer, expert) pairs; profiles ship in model bundles).

Kernel contracts (owned by `std.moe`; each with `reference.py` and a CPU provider in `testing`):

| Contract | Signature (sketch) | Notes |
|---|---|---|
| `moe_resolve@1` | `(ids [T,k], weights, map, layer, plan table, state…, row mask) → slot_ids [T,k], fetch list, CPU jobs, counts, delta` | deterministic; misses → the rank's null slot; padding rows masked; providers by `resolve_variant` (plain, LRU, promote-on-fetch) with the policy's declared device state |
| `moe_route_transform@1` | `(ids, weights, map) → ids, weights` | `approx` only, bound by explicit option |
| `expert_fetch@1` | `(fetch list, host aliases, staging slots)` | in-graph zero-copy gather (dynamic-address DMA is not graph-safe) |
| `linear_experts@2` | unchanged, over a class arena `[S, …]` | the forced-tiling patch (D4) inside `kernels.ggml_cuda` |
| `cpu_experts@1` (CPU) | `(jobs grouped by expert, activations, host blocks, mapped f32 rows)` | bitwise per token across group sizes and thread counts; ISA constraint |
| `moe_patch_rows@1` | `(y [T,k,H], CPU rows, job map)` | before `moe_combine@1` |
| `cpu_task@1` | `signal(slot)`, `wait(slot)` graph ops + worker service | `kernels.cpu_task` (D6) |

Streams and events never appear in std (invariant 6): the store's copies are S2 MemOps run by the core; in-graph
gathers are kernels.

## 6. Numerics and judge

| Configuration | Class | Check |
|---|---|---|
| offload store, any placement, any policy, F5a only (the default) | exact | snapshot byte-identical to `std.moe@1` all-resident, with adaptive placement on; L0 placement-invariance (D4) |
| F5b split on | bounded | L2 against a **path-aware floor**: the Tier B floor emulates q8_1 for GPU experts and Q8_K for exactly the experts the frozen split sends to the CPU (a `Blocking(256, f32)` mode in the judge's q8_1 emulation plus the split rule); KL ratio predicted on the CPU before the GPU run (Q8_K's error is 1.3–2.7× q8_1's, panel C10); run-identical ×3 |
| A1 tail skip | approx | human gate: L2 KL, top-1, task check; never in a default lock |

Floors: the development vehicle's all-resident reference is its RTX 4090 lock on the same file (the offload store
at F5a on the RTX 5080 must match the 4090's all-resident output only where the two classes' kernels agree, so the
bitwise check runs on one card: the 5080 offload store against the 5080 all-resident store with the experts forced
resident through a smaller context); the A5000 runs the bf16 floor as today. FN and GLM take their engine floors the same way (no
RTX PRO 6000 engine runs needed).

Plants (`judge/holon_judge/planted.py`), predicted verdicts fixed now; each reports how often it fired, and the
snapshot trace contains ≥ 8 swap rounds:

| Plant | Breaks | Predicted |
|---|---|---|
| `resolve_off_by_one` | a miss maps to the neighbouring slot | snapshot differs; L2 fails |
| `map_before_landing` | the map step skips the wait on the copy | run-identical fails (timing-dependent); L2 may pass |
| `map_stale_slot` | a deterministic map to a slot already reassigned | snapshot differs; L2 fails |
| `cpu_wait_skipped` | on 1% of layers the graph reads the CPU rows without waiting (F5b) | run-identical fails; L2 may pass |
| `cpu_gate_up_order` | the CPU reads the merged gate / up halves swapped (F5b) | L2 fails; F5a configurations clean |

Each plant changes only the configurations that use its path. Batch-invariance of `cpu_experts` is L0's own
mutation test, not a plant.

## 7. Memory fit (estimates; step 0 measures)

Budget = (VRAM free after weights and the CUDA context) × `memory_fraction` 0.9 − the worst step's peak.

| Model, file, context | Card | Non-expert weights | Expert bytes | Expert cache | Host tiers |
|---|---|---|---|---|---|
| Qwen3-30B-A3B Q4_K_M, 32K | RTX 5080 16 GB | ≈ 1.2 GB | ≈ 17.4 GB (6,144 × ≈ 2.8 MB) | ≈ 8.2 GB ≈ 47% (capped run ≈ 25%) | RAM ≈ 18 GB |
| Qwen3.8-Flash-Next IQ2_XS, 32K, MTP on | RTX 5080 16 GB | ≈ 3.7 GB + the MTP layer | ≈ 36 GB (24,576 × ≈ 1.46 MB) | ≈ 3.8–7 GB ≈ 11–20% (MTP-layer experts through the store or resident) | RAM ≈ 36 GB; PLE 28.8 GB on SSD |
| Qwen3.8-Flash-Next IQ3_S, 32K | RTX 5090 32 GB | same | ≈ 50 GB | ≈ 18.7 GB ≈ 37% | RAM ≈ 50 GB |
| GLM-5.3-Flash Maya-S, 32K, MTP off | RTX 5090 32 GB | ≈ 6.9 GB | ≈ 86 GB (12,096 × ≈ 7.1 MB) | ≈ 18 GB ≈ 21% | RAM exclusive + SSD on a 64 GB host |

The KV floor grows with context (30B-A3B: 96 KiB per token in bf16, ≈ 3 GB at 32K), so each benchmark lock names its context. GLM
routes ≈ 2.4 GB of experts per token on Maya-S; above ≈ 85% hits its decode is GPU-bound (≈ 4.7 ms of GPU reads
against ≈ 3.4 ms of misses at 90%), below it miss-bound. Host RAM per desktop class is checked before scheduling
(not recorded here).

## 8. Step 0 — measure before freezing

On the RTX 5080 and 5090 desktops (the bar cards), one heavy job at a time under `MemoryMax`, results in `evidence/`.

| # | Measure | Decides |
|---|---|---|
| M1 | H2D: pinned copies, `cudaMemcpyBatchAsync` of expert-sized blobs, the zero-copy SM gather (8–128 CTAs) | the fetch path's rate; ring depth |
| M2 | Contention: CPU expert GEMV concurrent with the gather; per-expert cost per path under contention | `plan[n]` (D4); whether F5b can gain ≥ 5% at all |
| M3 | CPU expert throughput: ggml-cpu `vec_dot` on Q4_K / IQ2_XXS / IQ2_S / IQ2_XS / IQ3_XXS / IQ3_S experts, 1–8 tokens, AVX2 vs AVX-512, 1–N threads; Strata's kernels beside them | F5b's worth; borrowing multi-token kernels; worker counts |
| M5 | Handshake with a real payload: k ≤ 10 jobs per layer, rows 1–8, device- vs host-planned | D5's plan location; `cpu_task@1` |
| M6 | Routing traces of the development vehicle (A5000) on the agent and mixed traces; offline replay of static, profile_decay (0.7 / 4 / 96 / 1.5), lru, promote-on-fetch, LFU, Belady at 10–60% cached, reporting **bytes per token per path** (hit, fetch, admission, CPU, SSD) converted to time with M1–M3's rates; profile transfer between traces; distinct experts per step and window; next-layer recall; per-layer prefill compute vs chunk size | the default policy and its knobs; D8's rule; D11 |
| M7 | NVMe `O_DIRECT` reads of expert-sized ranges at queue depths 1–32 with 1–8 I/O workers | F9's reader |

Decision rules (fixed now): the default policy is the one with the least modelled time per token on M6's held-out
trace, confirmed by L4 on the development gate; F5b enters a lock only at ≥ 5% decode gain (D4); F7 by D8's time
rule; F8 only at ≥ 15% on < 1K-token chunks with a fixed share; A1 never in a default lock.

## 9. Steps (milestone X)

| Step | Work | Checks | Days |
|---|---|---|---|
| 0 | M1–M3, M5–M7; host RAM per class | evidence; the panel reviews the parameters they set | 3 |
| 1 | S0–S5 and the CLI change with toy proofs (ABI 1.2); the forced-tiling vendor patch | core tests; `check_abi --base` additive; LOC; L0 placement-invariance | 5 |
| 2 | `std.tiered`, the offload store and options, class arenas with null and staging slots, `moe_resolve` / `expert_fetch` (CUDA + oracle + CPU provider), the pinned tier, `static` (F1), F5a | all-resident bitwise to `std.moe@1`; L0 | 5 |
| 3 | `profile_decay` (F2–F3) and `lru` (F4), two-phase edits, route tap and profile tool, prefill streaming (F7); **tag the std offload API** | snapshots with adaptive placement on, ×3; plants 1–3 | 5 |
| 4 | Development gate on the RTX 5080 (natural and capped fits) vs FreeToken and llama.cpp; tune | acceptance 3; L4 | 3 |
| 5 | F9 (SSD tier with I/O workers, exclusive RAM tier) | exactness; a RAM-capped run | 3 |
| 6 | `kernels.cpu_task`, `kernels.ggml_cpu`, F5b with `plan[n]`, the path-aware floor | L0 incl. bitwise per token and per thread count; bounded L2; plants 4–5; tune's ≥ 5% rule decides the dev lock | 5 |
| 7 | Held-out plugins after the tag: F10, A1, one unnamed strategy (acceptance 1) | plugin-only diffs; A1 judged as `approx` | 3 |
| 8 | Close: full suite, panel code review, one propagation of the existing locks; D13 if time allows | acceptance 5–7 | 2 |

F5b comes after the development gate on purpose: F5a-only must already be competitive, and tune then shows what the
CPU adds.

## 10. Risks and deferred

- **F5a-only may lose to CPU-heavy splits on small cards with Gen4 links** (the 3090 / 4090 class): the split
  exists for that, as a bounded lock where it gains ≥ 5%.
- **The model milestones are the critical path to the program bars.** X closes without them.
- **Strata and Maya move fast;** each comparison pins the reference commit, and bars are re-checked against the
  newest release only at FN's and GLM's close.
- **Exactness costs speed against Strata's defaults** (tail skip +10–24%); the like-for-like bars and A1 opt-in make
  the trade visible instead of hidden.
- **Host memory:** S2 accounts every pinned byte; development runs are capped with `MemoryMax`.
- **Deferred:** D13 (VMM); predictive prefetch (D11); a GPU-matching CPU kernel (D7); peer GPUs and layer split;
  dense expert offload; EXL3.

## 11. Milestone K outline (KV and checkpoint tiers), with the requirements rev 1's review found

K1 checkpoints and K2 KV pages demoted to pinned RAM instead of dropped (Strata C1–C2, llama.cpp `--cache-ram`,
vLLM's native offload, SGLang HiCache); K3 an SSD tier surviving restarts; K4 KV streaming for sparse-attention
mixers (with FN / GLM); the end-of-answer checkpoint (plan §10). Its own doc must answer:
- **Eviction invariants** (panel A10): demoted children must not keep a VRAM parent unevictable while `could_fit`
  counts it as freeable (preemption would then fail liveness); VRAM entries stay prefix-closed and the host tier
  evicts leaf-first; a property test that `could_fit` and `evict` agree.
- **Cancellation must be write-safe:** a load lands in host staging and reaches device units only through a MemOp of
  the resuming step, or released units wait on the load's event.
- **Deterministic resume** (panel C9b): a `BLOCKED` request rejoins at a fixed step offset, waiting if its copy has
  not landed, so batch composition never follows timing.
- **Acceptance that bites** (panel C14): RAM-hit TTFT ≤ VRAM-hit TTFT + 1.5 × bytes / measured rate; concurrent
  sessions' TPOT p99 during a load ≤ 1.2×; multi-session throughput against Maya's measured ratio on the same trace.
- Its core seam (prefix-index tiers: `Payload.tier`, `HitLen.cost`, demote, cancellable `BLOCKED(future)`, an
  `on_finish` publish) is designed there; the name stays `tier`, not `kv_tier` (it also stores checkpoints).

## 12. Review log

### Rev 1 → rev 2 (panel: A correctness, B architecture, C numerics / performance)

| Finding | Resolution |
|---|---|
| A1 [blocker] victim still mapped while its slot is overwritten | D4 two-phase edits (unmap, copy after the last read, map after the copy's event), fixed lag c+2 |
| A2 shared null slot breaks ggml's id helper on mmq | D5: one null slot per routing rank at the top of each class arena; L0 duplicate-id fuzz |
| A3 per-layer views vs the contiguous expert stride; staging per layer | D5: class arenas global across layers; staging reused layer after layer |
| A4, B1 [blocker] no per-engine shared object | S0 `BuildCtx.shared` |
| A5, B8 host pools vs weight loading and materialize; no seeding point | S2 allocates host pools before weights load; S0 `after_materialize` |
| A6 torch pinned allocator rounds to a power of two | S2 / D10 exact `cudaHostAlloc` or `mmap` + register |
| A7, C9a sizes re-measured at start change placement | D4 / D9 / S3: the lock pins VRAM and RAM-tier slot counts |
| A8, B7 CPU ISA not in the lock class | S5 `cpu_isa` tier for CPU-kernel deployments; D7 |
| A9 padding rows routed | D4 / `moe_resolve` row mask |
| A10, C9b prefix-tier invariants, cancellation, resume timing | moved to milestone K with these requirements (§11) |
| A11 device-written maps vs host flips | D4 one writer per region, conditional edits, delta log |
| A12, B2 native code in core | D6: `kernels.cpu_task`; core keeps lifecycle (S0) |
| A13, C1 mmq stream-k breaks placement invariance | D4 forced-tiling patch + L0 across slot counts, permutations, groupings |
| A14 telemetry overwritten by the next launch | S1 per-launch double buffer |
| A15 lag differs between paths | D4 fixed c+2 on every path |
| A16 per-request split share | D4 deployment-level table only |
| A17, C16 worker counts; I/O stalls compute | D6 worker counts from tune; separate I/O workers |
| A18 snapshot freezes adaptive policies; plant definitions | D4 refines §7.10 (placement-invariant path); plants redefined (§6) |
| A19 `torch.cuda` banned in std | S2 MemOps run the copies; §5 note |
| A20 nits (mmvq limit, `prepare_device` is a planner hook, CLI size refusal, memory_fraction, null units) | §3, §4 (CLI row), §7 budget line; null slots are the store's own |
| B3 re-added slots unused by core | D3: open options holding Refs; no slot re-added |
| B4 hidden core work around S5 (eviction, end-of-answer publish) | moved to milestone K (§11) |
| B5 contracts not open to F4 / F9 / F10 / A1 / new tiers | D3 protocols, `resolve_variant`, device state, tier codes, `moe_route_transform@1`, `Tier` protocol |
| B6 acceptance 1 not a real proof | D16 / acceptance 1: held-out plugins after the API tag |
| B9 file extents are core ABI | S4 |
| B10 fill shape underspecified; KV naming | S3 priority + cap, `paged_min_tokens`, config additions listed |
| B11 `kv_tier` naming; S2 / S5 overlap | milestone K, named `tier` |
| B12 scope too large | four milestones (D2), model core needs listed (§4.7) |
| B13 closed `moe.store` choice | D3 open Ref |
| B14 drift list, line budget | §4 line estimates; drifts below |
| B15 `quiesce` without a user | dropped until D13 |
| C2 D7 stretch mis-scoped | dropped (D7); F5a default gives exactness |
| C3 exact default via F5a | D4 |
| C4 split formula unsound at small miss counts | D4 `plan[n]` table |
| C5 default policy by hit rate | §8 by modelled time from bytes per path, F10 included, L4-confirmed |
| C6 coverage rule streams almost always | D8 time rule |
| C7 memory fit off (MTP layer, GLM trunk, KV floor) | §7 rewritten; contexts named |
| C8 inclusive RAM tier | D10 exclusive when RAM is short |
| C10 path-blind floor | §6 path-aware floor |
| C11 plant predictions; missing S1 bug class | §6 plants redefined, firing counts, ≥ 8 swap rounds |
| C12 bar consistency; llama.cpp arms | acceptance 3–4: like-for-like rows, per-arm llama.cpp bars, F1 ≥ 0.95× |
| C13 high-hit dev gate; "exact switches" label | capped ≈ 25% run; "no approximation" |
| C14 acceptance 4 trivially met | moved to milestone K with sharper bars (§11) |
| C15 split rule units; GLM GPU-bound claim | D9 experts take all above the floor; §7 corrected |
| C17 resource waste | 4090 dropped from step 0; M7 trimmed; no RTX PRO 6000 engine floors; L2 rows merged; batch invariance in L0 |

**Drifts from `docs/design.md`, recorded:** §7.4 (the CPU handshake lives in a kernel plugin, not
`holon.core.platform`); §6.5 and §4.2 (placement policies and tiers are std protocols bound by open options, not
re-added slots; Movers are S2 MemOps); §6.1 (the fill step and pinned split in the budget order); §6.4 and D8
(per-layer ready events and layer-wise loading stay planned; prefix tiers move to milestone K); §7.10 (adaptive
placement may run in snapshots when the consumer is placement-invariant); §9's expert-cache row (no VMM in the first
wave). Design is updated with X's code, as M0.7 did.
