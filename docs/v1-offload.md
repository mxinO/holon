# v1 — expert and KV offload: MoE models larger than VRAM on one consumer GPU

Status: **draft rev 1** (2026-10-17), for the panel. No code yet. Evidence base: the local research note
`docs/brainstorm/v1-offload-research.md` (not published; "R §n" below), which cites every reference engine by
`file:line` in its local clone.

Scope: plan §10 (v1, "Expert caching/offload" and "KV offload to pinned RAM / SSD"), with the owner's direction of
2026-10-16/17:
- support **the most popular strategies of today**, with **Strata** as the primary reference, FreeToken second, and
  llama.cpp's static split as the baseline and fallback only;
- **a new strategy must be easy to add as a plugin**: host-side policies between steps in Python, in-step policies
  as kernel plugins;
- **target models: Qwen3.8-Flash-Next** (compared with Strata) and **GLM-5.3-Flash** (compared with Project Maya, the
  Strata-derived GLM engine). Qwen3.5-122B-A10B leaves the plan.

The VMM ledger and live rebalance (plan §10's third offload line) is this milestone's second wave (D13). The two new
models are model milestones of their own (D2); this document lists only what offload needs from them.

## 0. The references in one table

| | Strata (primary) | Project Maya (GLM) | FreeToken | llama.cpp |
|---|---|---|---|---|
| Models | Qwen3.8-Flash-Next only | GLM-5.3-Flash only | many | many |
| Expert tiers | VRAM cache over all experts pinned in RAM (or mmap of the files when RAM is short) | VRAM / RAM / NVMe | VRAM cache over pinned RAM | static: per-layer CPU or GPU; new `--moe-cache-mib` LRU |
| Residency | seeded from a routing profile; every 4 windows up to 96 swaps by decayed routing counts (×0.7, gain ≥ 1.5) | three-tier, adapts while chatting (R §3.1b) | in-graph timestamp LRU, fetch on miss | fixed (LRU in the new cache) |
| Decode miss | split: a share over PCIe to the GPU, the rest computed by the CPU in RAM, through in-graph spin waits; share `L/(L+1.85R)` refit from live timings | Strata's, plus NVMe reads | offload / CPU / hybrid by a bandwidth benchmark | CPU computes |
| Prefill | non-resident experts stream through a ring, next layer during this layer's attention; a CPU share for small chunks | Strata's | every expert of the layer through a double buffer, evicting cached slots | selected experts uploaded for batches ≥ 32 |
| KV / conversation state | RAM checkpoints, parked conversations, session files, KV streaming for sparse attention | (R §3.1b) | none | `--cache-ram` prompt blobs |
| Output-changing defaults | route tail skip (missed low-rank experts dropped); timing-driven splits; CPU/GPU rounding changes with placement | expert skipping (R §3.1b) | — | — |

Both leaders are single-model engines with large switch sets (Strata: 529 `STRATA_*` variables; R §3.1). Holon's
offering is the same strategy set as plugins over any MoE model plugin, exact by default, with reproducible
placement.

## 1. Goal and acceptance

Run MoE models whose experts do not fit in VRAM on one consumer GPU, at least as fast as the strongest
single-model engine for that model, with every default strategy exact or bounded and runs reproducible.

1. **Strategies are plugins** (D3). Shipped: ≥ 3 placement policies (static, profile + decayed counts, LRU),
   2 miss paths (CPU compute, PCIe fetch) composed by a split, ≥ 2 KV tier plugins (pinned RAM, SSD), and one
   `approx` strategy (route tail skip, opt-in). Proof that a new strategy is one plugin: the second placement
   policy, the second miss path and the `approx` strategy each land in a commit whose diff touches only its own
   plugin directory (and its tests); `check_core_diff.sh` empty and no std diff for those commits. Every core seam
   (§4) is first proven with a toy plugin in `core/tests`.
2. **Exactness** (D4, §6):
   - a deployment whose experts all fit, run through the offload unit, is **bitwise** equal to `std.moe@1`
     (snapshot traces byte-identical);
   - experts fetched over PCIe compute bitwise as resident ones (the same kernel over a slot arena);
   - with the CPU share on: **run-identical** over 3 fresh engines on the snapshot trace (placement and the split
     depend on routing counts and committed steps only, never on timing), and **bounded** (L2 within budget) against
     the all-GPU floor of the same file;
   - KV and checkpoint tiers: a prefix hit served from RAM or SSD is bitwise equal to the same hit served from VRAM.
3. **Performance bars** (§9; proposed, owner to confirm), same card, same file, same context and KV dtype, drafting
   matched, medians of interleaved rounds:

   | Against | Model, card | Decode | Prefill |
   |---|---|---|---|
   | Strata, exact switches (`STRATA_ROUTE_TAIL_SKIP=0 STRATA_IQ_MT_MIN=1`) | Flash-Next IQ2_XS / IQ3_XXS on RTX 5080; IQ3_S on RTX 5090 | ≥ 1.0× | ≥ 0.9× |
   | Strata, defaults | same | ≥ 0.85× exact; ≥ 1.0× with Holon's `approx` tail skip | ≥ 0.9× |
   | Project Maya, exact settings (R §3.1b) | GLM-5.3-Flash Maya-S on RTX 5090 | ≥ 1.0× | ≥ 0.9× |
   | Project Maya, defaults | same | ≥ 0.85× exact; ≥ 1.0× with `approx` | ≥ 0.9× |
   | FreeToken, llama.cpp (`--fit`, tuned `--n-cpu-moe`, `--moe-cache-mib`) | Qwen3.5-35B-A3B Q4_K_M on RTX 5080 (development gate) and both targets | ≥ 1.0× FreeToken; ≥ 1.5× llama.cpp | ≥ 1.0× |

   Plan §9's kill line applies: below 0.8× of the reference on decode after a performance pass, the offload path is
   reworked before features are added.
4. **Agent traffic:** on the pi-style agent trace, a conversation evicted from VRAM resumes from the RAM tier with
   TTFT ≥ 5× better than recomputing it (32K-token prefix), and Holon serves ≥ 4 agent sessions at once with total
   decode throughput ≥ 1.3× one session's (Maya reports ×1.35–2.0 for four).
5. **Memory:** pinned and mapped host memory is accounted against a deployment `host_memory` budget (default: what
   the host reports available, minus a headroom) and never exceeded; startup refuses with a per-pool report.
6. **Core:** only the seams of §4, each an additive ABI minor (ABI 1.2), toy-proven; core ≤ 6000 code lines (4248
   today). Suite time does not grow beyond the new plugins' own tests (lean tests).
7. **Locks:** offload deployments get locks on the classes they run on; one propagation of the existing locks at the
   close (the core seams change core bytes; D15).

Non-goals: multi-GPU layer split and peer-GPU expert tiers (Maya and Strata have them; v1 stays single GPU, TP=2 is
its own milestone); EXL3 weights; the expert-predictive prefetch that Strata measured as a loss (D11); approximate
KV retrieval; CPU attention.

## 2. Decisions

**D1 — Strategy set: Strata's, exact by default.** The first set (R §4.2), each with its Holon class — **H** host
policy (Python, between steps), **D** in-step device kernel (kernel plugin), **C** CPU kernel (native, in the
worker), **std** library:

| # | Strategy | Reference | Class | Default |
|---|---|---|---|---|
| F1 | Static split: a frozen residency map, no swaps, no fetch | llama.cpp `--n-cpu-moe` | H | fallback without a profile |
| F2 | Profile-seeded expert cache: VRAM slots filled from a ranked (layer, expert) profile | Strata X1 | H + D | on |
| F3 | Adaptive swaps by decayed routing counts, on a fixed cadence of committed steps | Strata X2 | H | on |
| F4 | LRU fetch-on-miss, victims chosen in the graph | FreeToken, llama.cpp cache | D | comparison arm |
| F5 | CPU/PCIe split of decode misses through the in-graph handshake | Strata X4–X5 | D + C + H | on |
| F6 | CPU expert kernels, misses of a verify window or batch grouped per expert | Strata X6–X7 | C | on |
| F7 | Prefill: non-resident experts stream through a ring, next layer during this layer | Strata X8 | D + std | on |
| F8 | Prefill CPU share for small chunks | Strata X9 | C + H | only if step 0 shows ≥ 15% |
| F9 | SSD expert tier under the RAM tier | Maya, Strata's low-RAM mode | std + H | on when RAM is short |
| K1 | Recurrent checkpoints demoted to a pinned-RAM tier | Strata C1 | std + H | on |
| K2 | KV pages demoted to a pinned-RAM tier (parking; swap on preemption comes free) | Strata C2, llama.cpp `--cache-ram`, vLLM, HiCache | std + H | on |
| K3 | SSD tier under K1/K2, surviving restarts | Strata C3, LMCache | std + H | opt-in |
| K4 | KV streaming for sparse-attention mixers: full KV in RAM, a CLOCK cache of pages in VRAM | Strata C4 | D + std | with the model milestones |
| A1 | Route tail skip (approx) | Strata A1 (default there) | H + D | off; `approx` |

Reason: Strata is where users are (26k stars in three weeks) and its set is complete; FreeToken's and llama.cpp's
strategies are degenerate cases (F4, F1). Every strategy that changes outputs is an opt-in `approx` plugin (D4).

**D2 — Target models and a development vehicle.**
- **Qwen3.8-Flash-Next** (primary; compared with Strata on its own files) and **GLM-5.3-Flash** (compared with
  Project Maya on its Maya-S file, which uses only upstream ggml types: IQ2_XXS / IQ2_S / IQ3_XXS experts, Q6_K and
  Q8_0 elsewhere; Holon's `kernels.ggml_cuda` already has all of them).
- Both are new architectures, built as model milestones in parallel with this one: Flash-Next needs QSA sparse
  attention with its indexer, the PLE n-gram table (28.8 GB, read a few rows per token from SSD), hyper-connections
  and MTP; GLM-5.3-Flash needs MLA with the DSA indexer (top-2048), KDA linear attention, mHC (Sinkhorn
  hyper-connections) and sigmoid `noaux_tc` routing. They share the indexer, hyper-connection and linear-attention
  patterns, so the second reuses the first's std units. GLM pulls DSA forward from plan v2.
- **Development vehicle: Qwen3.5-35B-A3B Q4_K_M** (≈ 21 GB) on the RTX 5080 — its plugin exists, so offload work
  starts now; it is the development gate (acceptance 3, last row), not a headline target.

Reason: the comparisons that matter are same-model; the offload code is model-agnostic and must not wait for two
model plugins. Recorded: the lead chose the development vehicle (the owner was asked; reversible).

**D3 — Mechanism vs policy; what a "strategy plugin" is.** In-step work (resolve, fetch, expert GEMM over slots,
CPU compute, row patch, combine) is native: kernels selected through op contracts. Its *parameters* — the residency
map, the split share, quotas — live in device or mapped buffers. Between steps, pure-Python host policies read the
step's routing counts and rewrite those parameters, ordered on the stream. Plugin kinds:

| Kind | Slot / contract | Instances in v1 |
|---|---|---|
| expert placement policy (host) | `placement_policy` slot (single; re-added, ABI minor) | `static`, `profile_decay`, `lru` (with the in-graph variant of F4) |
| miss path (device + CPU) | op contracts `moe_resolve@1`, `expert_fetch@1`, `cpu_experts@1`, `moe_patch_rows@1` (std) | GPU fetch, CPU compute, the split |
| expert tier (host) | std `TieredStore` tiers | pinned RAM, SSD |
| KV / payload tier | `kv_tier` slot (multi, ordered; re-added) | pinned RAM, SSD |
| checkpoint and eviction policies | `checkpoint_policy`, `eviction_policy` (re-added) | `prompt_end` (exists), end of answer (plan §10), every-N; LRU with a pinned root |
| route transform (approx) | a policy plugin hooked into `moe_resolve`'s input | tail skip |

Nothing in core names experts, layers or formats (invariant 1). The MoE unit picks its expert store by a
deployment plugin option (`moe.store = resident | offload`, D5); model recipes do not change.

**D4 — Exact by default; reproducible placement.** Strata's speed without its output-changing defaults (R §4.3):
- placement changes come from **routing counts on a fixed cadence of committed steps** (every N steps), never from
  wall-clock timings; copies are issued at a commit and the residency map flips at a fixed later step boundary,
  waiting for the copies if they have not landed (a stall, never a timing-dependent result);
- the CPU/PCIe split and the prefill CPU share are **calibrated once per (card class, CPU, quant)** by
  `holon_tools tune` and frozen in the lock's `[engine]` (a request may override within the deployment's bounds);
- CPU expert kernels are **batch-invariant** by contract: a token's result is bitwise the same alone or grouped
  with other tokens (Strata's `STRATA_IQ_MT_MIN=1`, made mandatory; L0 enforces it);
- which misses go to the CPU is a deterministic function of the routing (by rank, then expert id).
Consequences: snapshot traces stay run-identical with offload on; a residency change moves an expert between the GPU
and the CPU, whose roundings differ, so served outputs are **bounded** with respect to placement (the same class as
batch shape today, design §7.10). **Stretch (D7):** a placement-invariant CPU kernel makes them exact.

**D5 — The offload MoE unit is a std variant, selected by deployment option.** `std.moe@1` keeps its interface and
binds an expert store (`resident` today, `offload` new) through `ctx.unit`. The offload store owns the slot arena
(a plugin pool, D9), the residency map (device, double-buffered, `[L, E] → slot | −1`), the host tiers, and the
per-layer graph shape:

```
router → moe_topk → moe_resolve → [publish x / CPU jobs; signal] → expert_fetch → linear_experts(gate_up, slots)
       → linear_experts(down) → wait → moe_patch_rows → moe_combine
```

Misses map to a **null slot** whose weights are zero, so `linear_experts@2` and `moe_combine@1` are unchanged and
accumulation keeps slot order independent of placement. The plan is computed on the device (FreeToken style), which
avoids Strata's host round trip per layer (its "flag A"); step 0 (M5) measures whether a host-planned variant is
faster, and either is a `moe_resolve` provider. Shared experts stay resident.

**D6 — The CPU handshake is S6's.** One 32-bit word per layer in mapped pinned memory, a spin-kernel wait (memop
wait on displays with a watchdog), a persistent native worker, zero-copy payloads, stop only after drain (design
§7.4). The core owns the worker pool (physical cores minus one; the Python loop pinned to the free core), fail-open
`abort()`, and the lifecycle; a kernel plugin supplies the compute function (a C ABI pointer from its native
library). GPU work between signal and wait (hit GEMMs, the PCIe share) hides the CPU time.

**D7 — CPU kernels: ggml-cpu first, then Strata's multi-token i-quant kernels.** `kernels.ggml_cpu` vendors
ggml-cpu (MIT) at the same pin as `kernels.ggml_cuda` and runs `vec_dot` over GGUF blocks in place; it covers every
type the targets use. Multi-token kernels for i-quants (one weight decode, up to 8 tokens) are borrowed from Strata
(MIT, design §5.6) once M3 shows the gain. ISA tiers are kernel constraints (`Constraint.cpu_isa`, re-added), not
ad hoc gates. Stretch: a CPU kernel that consumes the GPU's own q8_1 activations and reduces in mmvq's order —
integer dot products are exact, so only the f32 reduction order must match; if M4 shows it bitwise at < 1.3× the
ggml kernel's cost, residency policies become `exact`.

**D8 — Prefill streams only what is missing.** From a chunk-coverage threshold up (M6; Strata uses 1024 tokens),
layer l+1's non-resident experts stream through a ring on a copy stream while layer l runs; below it, routed-only
staging. Resident experts are never re-sent and the ring does not evict cache slots (FreeToken's two prefill flaws,
R §3.2). Prefill stays eager (design §7.4).

**D9 — VRAM: experts fill what is left above a KV floor.** A plugin pool may declare `fill=True`: after weights,
workspaces, graphs, slot state and fixed plugin pools, a deployment `kv_min_tokens` floor goes to paged KV, then the
expert arena takes the rest up to all experts, then KV takes any remainder (FreeToken's order, R §3.2). The floor
defaults to `max_model_len` for one sequence, so long contexts still fit; the lock records the split.

**D10 — Host tiers are std, accounted by the core.** Pinned host pools with a device alias are a core allocation
(D9's ledger, S2); what lives in them — all experts, a hot subset over an SSD tier (F9), demoted KV pages and
checkpoints (K1–K3) — is std and plugins. The RAM tier for experts holds all experts when they fit the host budget;
otherwise the hottest by the profile, with F9 serving the rest from the GGUF files (pread / O_DIRECT, staged into
pinned memory, R §3.1b for Maya's method).

**D11 — No predictive prefetch in the first set.** Strata's Foresight measured 0.91–1.00× on every box (R §3.1 E1).
Re-open only if M6 shows next-layer recall ≥ 85% and M1 shows ≥ 30% idle PCIe time during decode.

**D12 — Batching is first-class.** Misses of all rows in a step (decode rows, verify windows, mixed steps) are
grouped per expert before the CPU and PCIe paths, so a miss is paid once per step. Strata turns batching off by
default and its tail skip is off with batches; Holon's `max_seqs` stays as the lock sets it.

**D13 — VMM elastic pools are the second wave.** `Pool` resize `vmm` (stable base, a device capacity scalar,
segment map/unmap through `quiesce`), so KV grows into the expert arena and shrinks back without recapturing graphs
(Strata X3). It lands after F1–F9 and K1–K2, in this milestone if time allows, else as its own.

**D14 — Comparison method.** Same card, same GGUF, same context and KV dtype, same host, drafting matched (MTP on
for both or off for both). Every reference runs both its defaults and its exact settings. Metrics follow Strata's
tables: decode tok/s at 1K / 4K / 32K / 128K context (256 tokens, greedy), prefill at 1K / 4K / 32K, the agent
trace (TTFT / TPOT p50 and p99, prefix reuse, wall time), expert hit rate and CPU / PCIe shares, peak VRAM and RAM,
quality (L2 KL, top-1, ΔPPL against the Q8_0 or bf16 reference), run-to-run identity. Reference engines are built in
`bench/.envs/`.

**D15 — One core round, one propagation.** All core seams land on this branch and the existing 22 locks are
propagated once, at the close (as M0.7 did). If plan §10's "locks pin only output-affecting code" lands first, the
seams outdate fewer locks; it is not a prerequisite.

## 3. What Holon has today

| Piece | State |
|---|---|
| MoE | `std.moe@1` (`std/holon_std/moe/units.py`): f32 router, `moe_topk@1`, gate/up merged with SwiGLU and down as `linear_experts@2` over `[E, …]` stacks fully in VRAM, `moe_combine@1` in f32; named in M0.2 (`docs/m02-moe.md` D2) as the offload seam |
| Expert kernels | `kernels.ggml_cuda` (mmvq up to a per-shape crossover, else mmq) for the GGUF types incl. IQ1_S / IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S / IQ4_NL / Q2_K–Q6_K / Q8_0; `kernels.triton_experts` for dense stacks |
| CPU handshake | spike S6 only (`spikes/s6_cpu_handshake/README.md`): 1.9–2.8 µs per spin handshake on the desktops, captured in FULL graphs, 10k-replay soak, drain-first stop |
| Memory | `Ledger` with fixed device pools; `BuildCtx.pool(name, unit_bytes, segments, min_units, pref_units)`; `resize != "fixed"` raises (`core/holon/core/memory.py`) |
| MemOps | `MemOp(src_pool, src_units, dst_pool, dst_units, phase)` between device pools of one unit shape, run eagerly around the step (`core/holon/api/state.py`, `memory.run_mem_ops`) |
| Prefix index | hash chain over full pages, refcounted, LRU leaf eviction; M0.3 recurrent payloads in a plugin pool with `prompt_end` checkpoints (design §6.3) |
| Scheduling | `Status.BLOCKED` (used by logits processors), `Engine.drain()`, device SeqTable, depth-1 overlap |
| Cut from ABI 1.0, to return as minors | slots `eviction_policy`, `placement_policy`, `tier`, `mover`, `connector`, `codec`, `checkpoint_policy`; `HitLen.cost`, `Payload.tier`; `Pool.base`, `capacity_dev`, `device`, `segments`; `Constraint.cpu_isa`; the prop `batch_invariant` (`docs/release.md` §15.1) |

## 4. Core seams (additive ABI minors → ABI 1.2)

Each is the smallest general shape; none names experts, KV or formats. Each is proven first with a toy plugin in
`core/tests/test_payloads.py` (or a sibling test) before std uses it.

| # | Seam | Today → change | Toy proof |
|---|---|---|---|
| S1 | **CPU task handshake** | none (S6 spike) → `BuildCtx.cpu_task(name, slots, fn)` returning graph-capturable `signal(slot)` / `wait(slot)` ops on mapped one-word flags; the core owns a native worker pool, core pinning, fail-open abort, stop after drain; `fn` is a C ABI function pointer from a kernel plugin's native library. Implementation per fingerprint: spin kernel (default) or stream memop | a toy CPU task adds 1 to a mapped tensor inside a captured graph; abort and drain paths |
| S2 | **Host pools and cross-device MemOps** | device pools only; MemOps need equal unit shapes on one device → `BuildCtx.pool(…, device="host", mapped=True)`: pinned host memory with a device alias, accounted against `host_memory`; MemOps between host and device pools on the copy engine, on the step stream or a side stream with an event the step waits on | a toy kind demotes a unit to a host pool and adopts it back bitwise |
| S3 | **Unit step hooks** | `prepare_device` (in-graph) only → `BuildCtx.on_step(before_launch=…, after_commit=…)`: `before_launch` enqueues stream-ordered writes (map flips, parameter scalars, MemOps) before the replay; `after_commit` receives the step's telemetry tensors (copied D2H with its tokens, no extra sync); `quiesce(cb)` for changes not ordered on the stream | a toy unit counts calls by a device tensor and flips a double buffer between replays; a replayed graph sees each flip at the step it was ordered for |
| S4 | **A pool that fills the rest** | fixed plugin pools at `pref_units` → `BuildCtx.pool(…, fill=True)` joins the split after a deployment `kv_min_tokens` floor (D9) | a toy fill pool and paged KV split a budget as specified |
| S5 | **Prefix-index tiers** | drop-only LRU → `Payload.tier` and `HitLen.cost` restored; eviction may **demote** a payload or page run to a `kv_tier` plugin (`store(units) → handle`, `load(handle, dst) → Future`, `drop(handle)`); a hit below VRAM puts the request in `BLOCKED(future)` with destination units reserved and the future cancellable (abort, preemption) | a toy tier with a delayed future: the request waits, cancel releases the units, the resumed output equals the VRAM hit |

Not core: the offload store, `TieredStore`, `DeviceResolver`, residency maps, placement policies, calibration, route
traces and profiles, the ring, checkpoint policies, the tiers themselves (design §6.5). Restored slot names
(`placement_policy`, `kv_tier`, `checkpoint_policy`, `eviction_policy`) and `Constraint.cpu_isa` are ABI listing
additions with their manifests' `core_abi` floor raised to 1.2.

## 5. std library and kernel contracts

**`holonlm-std-tiered`** (std): `TieredStore[K]` over tiers (VRAM slot arena, pinned RAM, SSD), a `ResidencyMap`
(device, double-buffered), and the two resolvers of design §6.5 — `DeviceResolver` (experts, sparse KV pages) and
`HostPlannedResolver` (KV pages and checkpoints, before the step).

Kernel contracts (owned by `std.moe`; each with `reference.py` and a CPU provider in `testing`):

| Contract | Signature (sketch) | Notes |
|---|---|---|
| `moe_resolve@1` | `(ids [T,k], res [L,E] i32, layer, share, budget) → slot_ids [T,k], fetch list, CPU jobs (mapped), counts += ` | deterministic in its inputs; misses → null slot; split by routing rank then expert id; the LRU provider also writes victims; per-(layer, expert) route counts accumulate in a device tensor read by `after_commit` |
| `expert_fetch@1` | `(fetch list, host arena alias, staging slots)` | in-graph zero-copy gather (DMA with dynamic addresses is not graph-safe) |
| `linear_experts@2` | unchanged, over the slot arena viewed `[S, …]` per layer | per-layer views: GGUF files mix types across layers |
| `cpu_experts@1` (CPU) | `(jobs grouped by expert, activations, host blocks, mapped f32 output rows)` | batch-invariant per token (L0 checks a token alone vs in groups of 2–8); gate+up fused, SwiGLU in f32, down |
| `moe_patch_rows@1` | `(y [T,k,H], CPU rows, job map)` | writes CPU rows into their (t, j) places before `moe_combine@1` |
| `kv_resolve@1` (K4) | `(selected blocks, page map) → pages | fetch`, CLOCK victims | sparse-attention mixers only; with the model milestones |

**Route traces and profiles** (std tooling): a `route_dump` tap writes per-step (layer, expert, rank, token) records
from the `after_commit` counts; `holon_tools profile` ranks (layer, expert) pairs into a profile file; profiles ship
per model in the model plugin's bundle, seeded from shared traces (Strata ships one).

## 6. Numerics and judge

Classes (design §10.3):

| Configuration | Class | Check |
|---|---|---|
| offload unit, all experts resident | exact | snapshot byte-identical to `std.moe@1` |
| PCIe fetch only (share 1) | exact | snapshot byte-identical to all-resident |
| CPU share > 0, fixed placement | bounded | L2 vs the all-GPU floor of the same file; snapshot run-identical ×3 |
| adaptive placement (F3, F4) | bounded; run-identical on the snapshot trace | as above, swaps on in the snapshot (the trace is fixed, so counts are) |
| placement-invariant CPU kernel (stretch) | exact | snapshot byte-identical across placements |
| KV / checkpoint tiers (K1–K3) | exact | resumed output byte-identical to the VRAM hit |
| tail skip (A1) | approx | human gate: L2 KL, top-1, task check; never in a default lock |

Floors: the development vehicle's all-GPU floor runs on the RTX A5000 (21 GB fits in 24 GB); Flash-Next's and
GLM's on the RTX PRO 6000 (96 GB). L2 prompt sets: `v1-qwen3_5.jsonl` for the development vehicle; the model
milestones add theirs.

Plants (`judge/holon_judge/planted.py`), with predicted verdicts fixed before measuring:

| Plant | Breaks | Predicted |
|---|---|---|
| `resolve_off_by_one` | a miss maps to the neighbouring slot | exact rows fail (snapshot differs); L2 fails |
| `stale_map` | the map flip lands one step late on 1% of flips | run-identical fails (flip timing), L2 passes |
| `cpu_row_swap` | two CPU rows patched into each other's (t, j) | L2 fails; all-GPU configurations clean |
| `cpu_batch_variant` | the CPU kernel's result depends on the group size | L0 batch-invariance fails; L2 passes |
| `tier_wrong_block` | a demoted page restored to the wrong unit | KV-tier exactness fails; L2 fails |

Each plant must change only the configuration that uses its path (the all-resident baseline stays clean).

## 7. Memory fit (estimates; step 0 measures)

| Model, file | Card | Non-expert weights | Expert bytes | Expert cache | Host RAM for experts |
|---|---|---|---|---|---|
| Qwen3.5-35B-A3B Q4_K_M | RTX 5080 16 GB | ≈ 2 GB | ≈ 19.5 GB (10,240 experts, ≈ 1.9 MB) | ≈ 11 GB ≈ 55% | ≈ 20 GB |
| Qwen3.8-Flash-Next IQ2_XS | RTX 5080 16 GB | ≈ 6 GB | ≈ 36 GB (24,576 × ≈ 1.46 MB) | ≈ 7 GB ≈ 20% | ≈ 36 GB, plus the 28.8 GB PLE table on SSD |
| Qwen3.8-Flash-Next IQ3_S | RTX 5090 32 GB | ≈ 6 GB | ≈ 50 GB | ≈ 22 GB ≈ 44% | ≈ 50 GB |
| GLM-5.3-Flash Maya-S | RTX 5090 32 GB | ≈ 6 GB | ≈ 92 GB (12,096 × ≈ 7.6 MB) | ≈ 22 GB ≈ 24% | ≈ 92 GB, or the hottest part with F9 |

Expert cache = card − non-expert weights − KV floor (≈ 1 GB at 32K, 8-bit KV) − graphs and workspaces (≈ 1.5 GB).
GLM's routed bytes per token are ≈ 2.6 GB (8 × 42 layers × 7.6 MB), so its decode is bound by the miss path even at
an 80% hit rate; it is the case F9 and the split are built for. Host RAM per desktop class is checked before
scheduling (not recorded here).

## 8. Step 0 — measure before freezing

On the RTX 5080, 5090 and 4090 desktops, one heavy job at a time under `MemoryMax`, results in `evidence/`. Only
what decides a design parameter; the reference-engine runs move to step 7–8, where they are needed.

| # | Measure | Decides |
|---|---|---|
| M1 | H2D: pinned `cudaMemcpyAsync`, `cudaMemcpyBatchAsync` of expert-sized blobs, zero-copy SM gather (8–128 CTAs); Gen4 vs Gen5 | `L` in the split; whether the in-graph gather reaches DMA rates; ring depth |
| M2 | DRAM contention: CPU GEMV (2–7 threads) concurrent with DMA and with the gather | the constant `c` in `L/(L + c·R)` (Strata's 1.85) |
| M3 | CPU expert throughput: ggml-cpu `vec_dot` on Q4_K / IQ2_XXS / IQ2_S / IQ2_XS / IQ3_XXS / IQ3_S experts, 1–8 tokens per job, AVX2 vs AVX-512; Strata's kernels beside them | `R`; borrowing Strata's multi-token kernels (D7); F8's gain |
| M4 | Placement-invariant CPU kernel: mmvq's q8_1 activations and reduction order emulated on the CPU, bitwise vs the GPU, cost vs ggml (CPU-only, local) | D7's stretch |
| M5 | Handshake with a real payload: k ≤ 10 tasks per layer, rows 1–8, device-planned vs host-planned | D5's graph shape; S1's contract |
| M6 | Routing traces of the development vehicle on the agent and mixed traces (A5000 / RTX PRO 6000): per-layer skew, hit-rate curves vs cache fraction (5–60%) for static, LRU, decayed counts (0.7 / 4 / 96 / 1.5), LFU and Belady; profile transfer between traces; distinct experts per verify window and per batch; prefill chunk coverage; next-layer router recall. Repeated on Flash-Next and GLM when they run | the default policy and its knobs; F7's threshold; D11 |
| M7 | Tier costs: demote / promote of a 32K prefix and a 61 MB GDN checkpoint; NVMe pread / O_DIRECT throughput at queue depths 1–32 | `BLOCKED` vs a synchronous MemOp; F9's reader |

Decision rules (fixed now):
- split `f = L/(L + c·R)`, `c` from M2, rounded to 0.05 and clamped to [0.1, 0.6], stored per lock class;
- experts get VRAM above the KV floor while M6's curve still gains ≥ 1 point of hit rate per extra 5% of experts,
  else the rest returns to KV;
- the default policy is whichever of `profile_decay` and `lru` wins on M6's held-out trace (expected: Strata's);
- F7 streams when M6's chunk coverage is ≥ 50%, else routed-only staging; F8 ships only at ≥ 15% on < 1K-token
  chunks with a fixed share;
- tail skip is never in a default lock.

## 9. Steps

| Step | Work | Checks | Days |
|---|---|---|---|
| 0 | M1–M7; host RAM per class | evidence JSONs; panel reviews the decisions they set | 3 |
| 1 | Core seams S1–S5 with toy proofs (ABI 1.2); `check_abi --base` additive only | core tests; ABI listing; core LOC | 5 |
| 2 | `std.tiered`, the offload store and `moe.store` option, `moe_resolve` / `expert_fetch` / `moe_patch_rows` (CUDA + oracle + CPU provider), pinned RAM tier, `static` policy (F1), PCIe-only path; route tap and profile tool | all-resident bitwise to `std.moe@1`; share-1 bitwise; L0 | 5 |
| 3 | `kernels.ggml_cpu` + `cpu_experts@1`, S1 with a real payload, the split (F5, F6), calibration in `holon_tools tune` | L0 incl. batch invariance; bounded L2 on the A5000; run-identical ×3 | 5 |
| 4 | `profile_decay` (F2–F3) and `lru` (F4) policies — each its own plugin commit (acceptance 1); prefill ring (F7); development gate on the RTX 5080 vs FreeToken and llama.cpp | snapshots; L4 vs references; core diff empty for the policy commits | 5 |
| 5 | K1 and K2 (pinned-RAM tiers, demote / `BLOCKED` load), the end-of-answer checkpoint policy (plan §10) | KV-tier exactness; agent trace TTFT (acceptance 4) | 4 |
| 6 | F9 (SSD expert tier) and K3 (SSD KV tier, opt-in) | exactness; RAM-short deployment runs | 4 |
| 7 | Flash-Next (after its model milestone): offload deployment, profile, locks on the 5080 / 5090, comparison with Strata (defaults and exact) | acceptance 3 rows 1–2 | 4 |
| 8 | GLM-5.3-Flash (after its model milestone): the same vs Project Maya on the 5090 | acceptance 3 rows 3–4 | 3 |
| 9 | A1 tail skip as an `approx` plugin (its own commit); F8 if step 0 qualifies it; D13 if time allows; close: full suite, panel code review, one propagation | acceptance 1, 6, 7 | 3 |

Steps 7 and 8 wait on the model milestones; steps 0–6 do not. K4 (KV streaming) lands with the first model
milestone that has sparse attention.

## 10. Risks and deferred

- **The model milestones are the critical path** to the headline comparisons. Flash-Next's PLE table and QSA, and
  GLM's mHC and DSA, are each new std units; if either slips, steps 7–8 move and the development gate stands alone.
- **Strata and Maya move fast** (releases every few days). Each comparison pins the reference's commit in its
  evidence; bars are re-checked against the newest release at the close, not chased in between.
- **CPU kernels on desktop CPUs** decide decode speed at low hit rates. If ggml-cpu is < 0.8× Strata's kernels at
  M3, D7's borrowing moves into step 3.
- **Exactness costs speed.** Tail skip is worth +10–24% on 12–16 GB cards in Strata's numbers; Holon must recover
  it through hit rate (profile, swaps, batching) and overlap. If the exact bar (≥ 0.85× defaults) fails, the gap is
  reported, the `approx` plugin is offered opt-in, and defaults stay exact.
- **Host memory:** pinned allocations beyond the budget can take the host down. S2 accounts every pinned byte;
  development runs are capped with `MemoryMax`.
- **Deferred:** D13 if the milestone runs long; layer-wise KV load (per-layer ready events, design §6.4) — it saves
  tens of milliseconds against seconds of recompute and forces eager or piecewise prefill; peer-GPU tiers and the
  layer split; predictive prefetch (D11); EXL3.

## 11. Review log

(empty — rev 1 goes to the panel)
