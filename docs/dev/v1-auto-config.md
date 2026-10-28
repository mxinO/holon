# v1 — Auto config: `holon serve <model>` with no hand-set numbers

Status: **design, rev 1 (2026-10-28), not built.** For the owner's review; open questions in §10.

The owner's request (2026-10-28): "I see in the readme, we require users to manually set the --host-memory? that is
not good, we should have default config and auto tune, easy to use is important, we might need a auto tune library
for more complicated configs later".

**Goal.** On a supported card, `holon serve <model>` starts every README lock at the lock's speed or better, with no
flag and no file prepared by hand. When the host lacks RAM, it degrades or refuses with the shortfall in bytes. Pinned
host memory never pushes the host into swap. Every value chosen automatically is printed with its reason and can be
overridden by a flag. Automatic values are derived when the server starts, from the machine and the model. They are
never stored in a lock (README:61, "Host memory is a deployment's, never a lock's").

Estimates are marked *est.* and claims not checked against code or a measurement are marked *unverified*. Everything
else cites `file:line` at `de1f0d28`.

## 1. Inventory: what a user supplies by hand today

| Lock (README:39-53) | Hand-supplied today | Why it is manual |
|---|---|---|
| Resident locks (8B, 30B on 24–32 GB, 27B on RTX 4090 / 3090) | nothing | — |
| Qwen3.8-27B, RTX 5080 and 5090 (KV tier) | `--host-memory 17179869184` (README:66) | `host_memory` defaults to 0, meaning no host pools (`engine.py:87`). The tier's host level of `state_tier.host.bytes=16GiB` (in the lock) is refused, and the start is retried 4× before defaults serve (README:392-393, `supervisor.py:350-354`) |
| Qwen3-30B-A3B Q4_K_M, RTX 5080 (offload) | `--host-memory 21474836480`; the routing profile `profile_x4_gate.json`, built with `holon_tools profile … --max-seqs 8 --max-model-len 4096 --host-memory …` and placed in the HF cache's `blobs/` (README:326-334) | host memory as above. The lock names the profile file, and the store reads it beside the checkpoint (`store.py:873-898`); a missing file raises, so the start fails 4 × 600 s, then exits (README:427-431) |
| Qwen3.8-Flash-Next IQ2_XS, RTX 5080 | `--host-memory 70000000000 --memory-fraction 0.97`; the MTP pack (a Strata clone at `296b59d`, `mtp_fetch.py`, `mtp_pack.py`, `holon_tools mtp-pack`); the profile (with 4 more options copied from the lock); `--plugin-options mtp.path=…` (README:346-361) | host memory as above. `memory_fraction` is not planned (`plans.py:36`), so it cannot sit in a lock. The lock records the pack only as an identity, `[sources] "mtp.path" = "mtp-q2_0-q8@a55280a0417d4b19"`, never as a path (`plans.py:121-125`). Without the pack the drafter finds no head (`drafters/mtp/plugin.py:33-37`). The profile tool builds its engine from flags, not from the lock (`tools/holon_tools/__main__.py:188-194`), so the lock's options must be retyped |
| Qwen3.8-Flash-Next IQ3_S, RTX 5090 | the same with `86000000000` and `0.985` | the same |
| A borrowed lock | the lent lock's flags and files (README:80-82) | the same causes |
| No lock | nothing, but a file larger than VRAM is refused before download unless `--host-memory` or `--no-size-check` is given (`cli.py:214`); offloading an unlocked MoE needs `moe.store=…` by hand | the size check uses the presence of `--host-memory` as its sign of offload |

**Why the Flash-Next locks need 0.97 / 0.985.** `state_memory` is `int(free × fraction) − peak` (`executor.py:230`).
The share `1 − fraction` of the memory left after weights is the slack, and on these locks every byte past the slack
goes to the expert cache, a fill pool. At 0.9 the slack is ≈ 10% of free VRAM, gigabytes on these cards. What the
slack has to cover was measured: the allocator's segments over the worst step beyond its live peak, 0.05–0.07 GB after
168562e (§23.2 and §23.4 of `v1-flash-next.md`), and the memory outside the allocator, which grows a little at start
(§23.1). Each extra point of share bought ≈ +0.35 points of experts cached (§23.2), and the 5090 also fits at 0.99
(§23.4). The share was hand-tuned only because it is a share. The rule in D2 derives the slack instead.

## 2. Principles

- **P1. The deployment's values, not the lock's.** Only `EngineConfig` fields that plans do not record
  (`plans.py:36-37`: `host_memory`, `memory_fraction`, …), and files a lock names by identity, become automatic.
  Nothing written into a lock changes.
- **P2. One rule each, shown at start, overridable.** Each automatic value has a single deterministic rule over
  machine facts and a measurement. The engine prints the value and its inputs on one line. An explicit flag replaces
  the rule.
- **P3. Never a silent plan change.** Every automatic value leaves the plan id unchanged, except one: the RAM-short
  spill (D1). Like any flag, the spill shows as `differs from the lock`.

## 3. Decisions

### D1. Host memory: automatic by default

**Today.** Host pools are allocated when requested, within a byte budget `host_memory` (`memory.py:231-250`). Every
host pool in the tree requests `min_units == pref_units`, so its size is fixed before the request:
- `moe/cpu.py:79-80`;
- `tiered/pinned.py:34-35`;
- `tiered/rows.py:86-87, 122`;
- `tiered/store.py:402-403`;
- `cpu_task/plugin.py:62`;
- the tier's `host_pool(units=…)`, `engine.py:292-294`.

The budget is therefore only a cap. The amount pinned is always what the build asks for. A refusal already reports
every pool held (`memory.py:241-244`).

**Rule.** `EngineConfig.host_memory: int | None = None`, where None means automatic.
- **When.** Each host-pool request is checked against the host's live state as it is made. The cap is never computed
  ahead of time.
- **The test:** `need ≤ available − headroom`.
  - `available` is the lesser of two figures:
    - `MemAvailable` from `/proc/meminfo`;
    - for the process's cgroup and each of its ancestors, `min(memory.max, memory.high)` minus
      `(memory.current − inactive_file)`. On cgroup v1 the files are `memory.limit_in_bytes`, `memory.usage_in_bytes`
      and `total_inactive_file`.
  - Pools already pinned are already gone from `available`, so they are not subtracted a second time.
  - `headroom = max(6 GiB, 10% of the RAM limit)`. The RAM limit is the lesser of `MemTotal` and the lowest cgroup
    limit.
- **Why live, per request.** Pools are pinned in sequence: rows at bind, the store at `before_materialize`, the tier
  late. The model file's page cache, being reclaimable, is already counted in `MemAvailable`.
- **What the headroom covers** (*est.*):
  - the grammar workers' 1.6–2.2 GB (README:294-295), which start asynchronously;
  - the frontend process and the engine's own growth after build;
  - a few GB for the desktop.

  Below that much free memory, the kernel can only reclaim file pages and swap other processes' anonymous memory,
  because pinned pages cannot move. That is the swapping and lag the owner wants to avoid.
- **Containers.** Checked on this host (2026-10-28): under `systemd-run --user --scope -p MemoryMax=16G`,
  `/proc/meminfo` still shows the host's total, while the scope's own `memory.max` reads 16 GiB. The cgroup path is
  nested several levels deep, so the walk over ancestors is needed. That pinned pages are charged to
  the cgroup is *unverified*: it is the kernel's standard accounting for faulted anonymous pages, and the RAM-short
  test (§9) checks it. `RLIMIT_MEMLOCK` is not consulted. `cudaHostRegister` pins through the driver, and the
  Flash-Next locks pinned 70–86 GB without raising the limit (*unverified* that no host raised it).
- **Explicit `--host-memory BYTES`** (also `70GB` or `64GiB`, parsed like `state_tier`'s `nbytes`,
  `tier.py:75-80`) is a fixed cap, as it is today. A request that would cut into the headroom is warned about once
  but not refused, because the user chose the value. `--host-memory 0` still means no host pools.
- **The size check before download** (`cli.py:214`) becomes: refuse when the file is larger than VRAM plus
  `available − headroom`. It no longer looks at whether `--host-memory` was given.

**When RAM is short.** Three steps, in this order:
1. **Smaller pools, where they are elastic.** The state tier's host level is requested last (`late=True`,
   `engine.py:292-294`), and its size is a planned option (`state_tier.host.bytes`). `TierCtx` gains
   `host_left() -> int` (the live `available − headroom`, or what remains of an explicit cap). The tier then takes
   `min(host.bytes, host_left())`, never below its floor (`tier.py:180-196`), and logs the size it took. The tier
   moves state without changing it (README:399-402), so a smaller level only changes how many prefixes survive.
2. **Spill to a slower, byte-identical place.** A host-pool request may name a spill, `BuildCtx.pool(…,
   spill={option: value})`: the plugin options under which the plugin serves without that pool. The core never reads
   them. Under the automatic budget, a refusal (`OutOfUnits`) carries the spills of the refused pool and of every held
   pool. Largest pool first, the supervisor picks just enough of them to cover the shortfall. It then restarts the
   same history entry once with those options layered on, logged as a retry (ipc payload +1 field; supervisor
   `_launch`, `supervisor.py:343-357`). Today only `std.tiered.rows@1` declares a spill: `<table>:ssd` plus
   `cpu_task.io_threads=4`. That moves Flash-Next's PLE table (26.8 GiB) to the SSD, byte for byte the RAM tier's,
   at a cost of 16% / 26% of prefill at 1K / 4K and 6–7% of MTP decode on the RTX 5090 (`v1-flash-next.md` §22.1).
3. **Refusal**, when nothing is left to spill. The message names:
   - the pool and the bytes it needs;
   - `available`, the headroom and where the limit comes from (meminfo or cgroup);
   - the shortfall;
   - each held pool.

   Its exit is the existing "memory" kind (`ipc.py:48`), which falls back to defaults at once rather than after 4
   retries.

The offloaded experts declare no automatic spill. Their RAM-short mode (`moe.ram` with a disk tier) is 1.9–8.3×
slower (`v1-offload.md:367`), and its size depends on the shortfall. The refusal names it instead (Q5).

### D2. GPU memory: the slack is measured, not a share

`EngineConfig.memory_fraction: float | None = None`, where None means automatic. The startup profile already measures
the worst step's live peak and the graphs' reserve (`executor.py:207-230`). One more measurement comes from the same
run: the allocator's overshoot, `max_memory_reserved` growth minus `max_memory_allocated` growth over the profiled
step (`platform.peak_memory`, `platform/__init__.py:91-99`; +3 lines).

```
slack  = min(free / 10, overshoot + RESERVE)          RESERVE = 256 MiB
budget = free − peak − slack                          (explicit --memory-fraction F: int(free × F) − peak, as today)
```

- **The min.** It caps the slack at today's 10%, so no lock gets a smaller budget, and so a smaller context, than at
  0.9. The tight case is the RTX 5080's 27B with its 40,960-token context (README:443-445).
- **RESERVE** covers what the profile cannot see: lazy allocations at a first request, and other processes, such as
  a display, growing later. On the Flash-Next cards the rule's slack is ≈ 0.3 GB (overshoot 0.05–0.07 GB, §23.4),
  against ≈ 0.44 GB at 0.985 on the 5090 (§23.1), where 0.99 fit with 0.2 GB to spare (§23.4). Automatic therefore
  caches at least as many experts as the hand-set shares (*est.*; measured in §9).

### D3. Where derived artifacts live

- **The location.** One function, `holon.api.cache_dir(*parts)`, returns `$XDG_CACHE_HOME/holon/…`, else
  `~/.cache/holon/…`. It is the only read of an environment variable for a location (Q4). The existing weight-digest
  caches (`std/…/gguf/source.py:26`, `hf/source.py:21`) move onto it unchanged, since by default it is the same
  `~/.cache/holon/weights`.
- **Contents:** `profiles/` holds learned routing counts (D4), and `sources/<identity>/<file>` holds the files a lock
  names by identity (D5).
- **What goes elsewhere.** Hub downloads go through `huggingface_hub`, which honours `HF_HOME`. Holon writes nothing
  into the HF cache itself; the README's profile in `blobs/` was the only case. Nothing is written into the repository
  or `plans/`.
- **Docker.** The README's existing `-e HOME=/cache -v <cache dir>:/cache` (README:127-133) also persists these
  artifacts.

### D4. The routing profile: optional, learned, and never fatal

A profile only seeds what the expert cache holds first, and outputs never depend on it (README:317-324).
- **Never fatal (std, `store.py:886-910`).** A named profile that is absent, or was counted on other files, logs a
  warning and the store seeds from the learned profile, else starts cold. This removes the 4 × 600 s failure
  (README:427-431).
- **Learned.** The store already counts routes at each commit (the taps `holon_tools profile` uses,
  `tools/holon_tools/profile.py:24-27`). It also adds them to a running total, written atomically to
  `cache_dir("profiles", <sha256 of store.checkpoint>.json)` on close and every 10 minutes of serving. The key comes
  from `store.py:878-884`: the file names and sizes, and a hub blob's name is its content hash. The format is the
  tool's.
- **Lookup order:** the lock's named file beside the checkpoint (compatible with every existing setup and the bench),
  then the learned profile, then cold.
- **Not shipped yet.** The three lock profiles stay out of the repository unless D4 misses its speed bar (§9, Q6).
  The cold first start is unmeasured, and §9 measures it.
- **The tool.** `holon_tools profile` starts from the lock as `propagate` does (`__main__.py:59-83`), so its
  documented command needs no copied options. It stays as the way to make a lock's evidence profile.

### D5. The MTP pack: made on first use, found by identity

- **General (core `cli`).** For each `[sources]` key of the lock that the flags do not give, `holon serve` looks in
  `cache_dir("sources", <identity>)`. If the file is missing, it asks the model plugin through a new optional hook
  (the pattern of `TrainedContext`, `api/hooks.py:236-238`):
  `make_source(cfg, source, key, identity, directory) -> path | None`. It then sets the option to the path.
- **Checked.** The engine already compares the identity of the file it opened with the lock's (`plans.py:121-125`,
  `served`), so a wrong file shows as `differs from the lock: source:mtp.path`.
- **When it fails.** If no plugin makes the file, or making it fails (for example, offline), the start is refused
  before loading, with the reason and the alternative `--max-draft-tokens 0` (README:374-376).
- **Specific to the model (`models.qwen4_exp`).** For `mtp-q2_0-q8@a55280a0417d4b19`, the hook:
  1. reads the head's 31 tensors from `Qwen/Qwen3.8-Flash-Next` at revision `de4b8e4d`, by HTTP range reads through
     `huggingface_hub`'s session (endpoint and token honoured), each checked against its sha256;
  2. packs them with the experts in Q2_0. The pack must hash to `a22206a4…` (`v1-flash-next.md` §13), which makes our
     quantizer bit-exact to Strata's;
  3. rewrites the dense projections to Q8_0 (`tools/holon_tools/mtp_pack.py:73-107`, moved into the plugin; the tool
     keeps its command line);
  4. checks the sha256 prefix and renames atomically into the cache.

  Strata's `mtp_fetch.py` reads 31 tensors spread over 28 of the checkpoint's 131 shards
  (Strata `296b59d`, `tools/mtp_fetch.py:3-5`). The download size and first-start time are *unverified*.
- **Licence.** Strata is MIT (`THIRD_PARTY_NOTICES.md:50`). Reproducing its pack bit for bit means following its Q2_0
  per-block MSE scale search exactly. That is an adaptation to be declared under AGENTS.md's third-party rules, or the
  pack is published instead (Q3).

### D6. The launch lines' plugin options

| Option(s) in the README's launch lines | Class | After |
|---|---|---|
| `rows.place=ple:host,token_embd:host ggml_cuda.pass_rows=4096 moe.ring_lend=tail` (README:357-358) | already in the lock | retyped only because `profile` ignored the lock (D4) |
| `mtp.path=…` (README:360) | a lock source, not planned as a path | D5 |
| `moe.placement.profile=` to serve without one (README:322-324) | workaround | obsolete (D4) |
| `rows.place=ple:ssd,token_embd:host cpu_task.io_threads=4` (README:377-380) | RAM-short mode | the spill (D1) |
| `--host-memory`, `--memory-fraction` | deployment | automatic (D1, D2) |
| speculation on the 27B (README:255-257), `logits_processor` on Flash-Next (README:278-280) | opt-in features with measured trade-offs | stay flags |
| `http.model_name`, `http.drop_billing_header` | preferences | stay flags |
| `state_tier.levels=ssd`, `state_tier.ssd.dir` (README:401-402) | needs a directory the user chooses | stays a flag; its rates are measured per drive (`tune --ssd-rates`, `tune.py:156`), for the later library |
| `loop_core`, `CUDA_VISIBLE_DEVICES` | per host | stay flags |

### D7. The automatic-tuning seam (minimal now)

Two kinds of derivation exist already, and both fit one shape: `value = rule(machine facts, measurement)`.
- **At start, every time, never stored.** These are the host budget (D1), the GPU slack (D2), the tier's size and the
  spill. The facts are `platform.fingerprint` (`platform/__init__.py:22-30`) and the new `platform.host_memory_facts()`
  (total, available, limit, and its source). The measurement is the build's own: pool requests and the startup
  profile.
  - The convention: a field or option whose value is None or `auto` derives it.
  - The engine collects each result as `(name, value, why)` in `engine.derived`.
  - It prints them (§7) and reports them in `served()` and `/metrics`, beside the plan.

  No protocol and no registry.
- **By search, once per GPU class, stored in locks.** `tune.tune(pins, space, rate)` (`tune.py:55-75`) is already a
  generic coordinate descent over `dict[str, str]` with a measured rate. Kernel pins and F5b's `split`
  (`tune.py:82-98`) use it. Its results change plans, so they go through `propagate` into a lock.

**What the later library adds, and only then:**
1. Search spaces over byte-neutral deployment options per machine: `moe.cache` sizes, `ggml_cuda.pass_rows`,
   `moe.pipeline`, `max_seqs` against cache share, offload or RAM-short modes, `cpu_task.threads`.
2. A per-machine record of the winners, `cache_dir("deploy", <model id>, <class>, <RAM class>.toml)`. `holon serve`
   layers it between the lock and the flags, which is one more layer in `serve_history` (`cli.py:109-125`, about +8
   core lines). It holds only values that P1 allows (not planned, or byte-neutral), so it can never stand in for a
   lock.
3. A `holon tune` command for users that runs the search on their own machine, built on `tune.tune`.

None of this is built now.

## 4. Core impact (AGENTS.md invariant 5; `check_core_loc.py` reports 5019 / 6000 now)

| Change | Where | Code lines (*est.*) |
|---|---|---|
| host facts: meminfo, cgroup v2 ancestors and v1 | `platform` | 25 |
| the allocator's overshoot in `peak_memory` | `platform` | 3 |
| the automatic host budget, headroom, refusal report, spills on `OutOfUnits` | `memory.py` | 15 |
| `host_memory` / `memory_fraction` as `\| None`; the start log; `derived` | `engine.py` | 12 |
| the slack rule | `executor.py` | 5 |
| `TierCtx.host_left` | `api/tier.py`, `engine.py` | 3 |
| `BuildCtx.pool(spill=)` | `api/model.py`, `build.py` | 3 |
| the spill retry | `ipc.py`, `supervisor.py` | 12 |
| lock sources from the cache or `make_source` | `cli.py`, `api/hooks.py` | 22 |
| size check against VRAM + RAM; `float \| None` flags; byte-size parsing | `cli.py` | 7 |
| `cache_dir()` | `holon.api` | 5 |
| **total** | | **≈ 110 → ≈ 5130 / 6000** |

- **Nothing in core names a model.** Spill options are opaque mappings, source identities opaque strings, and
  `make_source` is the model plugin's business.
- **The ABI gains** `cache_dir`, `MakesSources`, `TierCtx.host_left` and `pool(spill=)`. The last changes a listed
  signature, so `check_abi` will ask for a minor bump.
- **Toy-plugin proofs in `core/tests`:**
  - a host pool with a spill;
  - a model plugin with `make_source`;
  - an elastic tier.
- **Outside core** (std about +60, plugins about +200 *est.*):
  - std: the store (D4), the tier's shrink, the rows plugin's spill declaration, and the weight caches moved onto
    `cache_dir`;
  - `models.qwen4_exp`: the fetch, Q2_0 and Q8_0 (D5). `plugins/pyproject.toml` declares `gguf` and
    `huggingface-hub` for it (`manifest-deps`);
  - tools: `profile` starts from the lock.

## 5. Exactness and locks

- **No plan id changes.** `host_memory` and `memory_fraction` are not planned (`plans.py:36-37`). The tier keeps its
  planned `host.bytes`, and its own options are already outside `state_identity` (`plans.py:150-160`). Source files
  are planned by identity, which D5 checks.
- **Outputs.**
  - *Fixed host pools:* the same bytes as with a hand-set cap.
  - *Expert cache size* (D2): exact by offload's placement invariance (README:305-310).
  - *A smaller tier:* moves state without changing it. The 27B locks' `branch,answer` checkpoints change where a
    resumed prompt is recomputed from, and are KL-bounded rather than bitwise (README:399-402). The snapshot trace
    does not reach the tier (`judge/holon_judge/snapshot.py:129-155`), and §9 checks this.
  - *The PLE spill:* byte-identical (`v1-flash-next.md` §22.1), but the plan differs (`rows.place`), so it is logged.
  - *The learned profile:* exact by placement invariance.
- **Propagation: no lock is affected.**
  - `propagate` records each snapshot on the stored `pool_bytes` of its own class (`propagate.py:150-176`). The
    newest lock of all 25 lock classes stores `pool_bytes` in its snapshot (checked 2026-10-28), so automatic sizing
    cannot move their traces.
  - Only a first lock (`--new-model`) or another class's measures pools, now under D2.
  - The Flash-Next, tier and offload propagations no longer need `--host-memory` or `--memory-fraction` flags.
- **Context clamp.** `max_model_len` is planned, and D2's `min` keeps every lock's budget at or above 0.9's, so no
  lock is clamped further than today. §9 checks it on each card.

## 6. Build steps

1. Core D1 and D2 with their CPU tests (fake `/proc` and `/sys` trees, the ledger's budget, the slack rule). Toy
   proofs.
2. std: the tier's shrink and the rows spill; the supervisor's retry.
3. D3 and D4: the cache, the never-fatal profile, learning.
4. D5: the hook in `cli`, then `make_source` in `models.qwen4_exp`, bit-exact against `a22206a4…` and `a55280a0…`
   (CPU only).
5. Tools and the README (§8), then §9's acceptance runs on the bar cards.

One reviewer for the aggregate (CLAUDE.md), with the review loop at steps 1, 4 and 5.

## 7. What a user sees

At start, after the lock line, one line per automatic value:

```text
holon: host memory: pinned <P> GiB, <L> GiB left (kept free: <H> GiB = max(6 GiB, 10% of <T> GiB), cgroup: no limit; --host-memory caps)
holon: GPU memory: state pools <B> GiB = free <F> − step peak <P> − graphs <G> − slack <S> (allocator <O> + reserve 0.25); --memory-fraction F sets a share
holon: mtp.path: ~/.cache/holon/sources/mtp-q2_0-q8@a55280a0417d4b19/mtp-q2_0-q8.gguf (built on first use)
holon: routing profile: learned (~/.cache/holon/profiles/3f…json, 41 min of serving)
```

When RAM is short:

```text
holon: host memory short by <S> GiB: plugin.tiered.pinned.… needs <N> GiB, <A> GiB available above the <H> GiB kept free (meminfo); held: plugin.tiered.rows.ple 26.8 GiB
holon: retrying once with rows.place=ple:ssd,token_embd:host cpu_task.io_threads=4 (byte-identical; slower prefill)
```

**Overrides:**
- `--host-memory BYTES|70GB`;
- `--memory-fraction F`;
- `--plugin-options mtp.path=FILE` and `moe.placement.profile=NAME` (or `=`: none);
- `--plugin-options state_tier.host.bytes=…`;
- `--plugin-options rows.place=…` (a spill is never applied over an explicit value of its option).

## 8. The README afterwards

- **"Launch flags" is deleted** (README:61-74).
- **The model table** (README:39-53) keeps its rows, and its caption becomes: "Start any of them with
  `holon serve <name>`".
- **A short "Host RAM" table** replaces the launch flags. Each row's figure is the start line's measured pinned
  bytes, filled in at §9:

  | Lock | Pinned host RAM | With less RAM |
  |---|---|---|
  | Qwen3.8-27B, RTX 5080 and 5090 | the 16 GiB KV tier | the tier shrinks to fit (down to its floor) |
  | Qwen3-30B-A3B Q4_K_M, RTX 5080 | the experts, N GiB | refused, with the shortfall |
  | Qwen3.8-Flash-Next IQ2_XS, RTX 5080 / IQ3_S, RTX 5090 | experts, PLE 26.8 GiB, embeddings: N / N GiB | the PLE table moves to the SSD, then a refusal |

- **Quickstart** becomes `uv run holon serve unsloth/Qwen3.8-27B-GGUF` (README:139).
- **The Flash-Next section** becomes `uv run holon serve ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, noting that
  the first start builds the MTP pack once. The Strata clone and the four commands move to `tools/README.md` as the
  way to reproduce the pack.
- **Known limitations:** the "file a lock names … fails slowly" item and the host-RAM flag item go.

## 9. Test plan and acceptance

**CPU** (`pytest`; lean, parametrized):
- host facts over fake trees: cgroup v2 with a limited ancestor, `memory.high`, v1, no cgroup;
- the ledger's automatic budget: fit, refusal text, spills chosen largest-first, an explicit cap warned;
- the slack rule;
- the supervisor's single spill retry, with a toy plugin;
- lock sources from the cache and from a toy `make_source`;
- the store with a missing profile (cold), a profile for other files, and learning round-tripped on the tiny MoE
  model;
- D5's pack built from a fixture and compared with Strata's on the tiny model's tensors (`v1-flash-next.md` §5.1
  layout).

**GPU acceptance** (the bar cards, one GPU each, under a fresh `XDG_CACHE_HOME`, with the HF cache holding only the
model files):
1. **Bare start.** Every README lock starts on its card with only `holon serve <model>`, with no
   `differs from the lock` line and a plan id equal to the lock's. On Flash-Next this includes building the MTP pack,
   timed.
2. **Speed.** `bench/offload_ctx.py` on both Flash-Next cards and the RTX 5080's 30B, automatic against today's
   hand-set flags (with the lock's profile), 2 ABBA rounds. Decode and prefill must be ≥ 0.99× at every context, and
   the experts cached (`bench/ledger.py`) at least the hand-set share. The run is repeated with the learned profile
   after one session of the agent trace, and the cold first start (first 8 prompts) is reported. The learned profile
   must also reach ≥ 0.99×, else ship the profiles (Q6). For resident locks, the state-pool bytes are ≥ 0.9's, with
   the same `max_model_len`.
3. **RAM-short**, under `systemd-run --user --scope -p MemoryMax=…`, which also tests the cgroup path:
   - 27B tier lock, a limit 8 GiB under its need: the tier shrinks and serves;
   - Flash-Next, a limit 20 GiB under its need: one retry with the PLE spill, then it serves;
   - Flash-Next, a limit 40 GiB under: a refusal naming the shortfall, within one start, not 4 × 600 s.

   `memory.events` must show `oom_kill 0` throughout.
4. **No swap.** In each start of (1) and through 10 minutes of the agent trace afterwards:
   - `pswpin` / `pswpout` in `/proc/vmstat` do not move;
   - `MemAvailable` stays ≥ headroom − 1 GiB;
   - PSI `some avg10` stays below 1% (`/proc/pressure/memory`).
5. **VRAM soak.** Every lock at automatic, through the agent trace plus the 30K-token prompt, ends with no CUDA OOM,
   including the RTX 5080 driving a display.
6. **Locks.** `propagate` of each changed unit's locks on its class gives a byte-identical trace (§5).

## 10. Questions for the owner

1. **Headroom.** `max(6 GiB, 10% of the RAM limit)`, kept free of pinned memory: right, or a different floor or
   share?
2. **GPU reserve.** `RESERVE = 256 MiB` beyond the measured overshoot (D2): enough for a card that drives a display?
3. **MTP pack.** Build it on first use, adapting Strata's MIT Q2_0 quantizer and range fetch with provenance, outside
   the "kernels only" borrowing rule? Or publish `mtp-q2_0-q8.gguf` on the Hub under the project's account and
   download it (simpler; it needs the Qwen3.8-Flash-Next licence's permission, *unverified*)?
4. **`XDG_CACHE_HOME`.** Allow it as the one environment variable read for a location (`holon.api.cache_dir`), an
   explicit exception to invariant 3? Or keep `~/.cache/holon` only?
5. **Spill automatically.** Apply the PLE spill automatically (D1 step 2), or only name it in the refusal? Keep the
   experts' RAM-short mode (1.9–8.3× slower) manual, as decided?
6. **The routing profile.** Learned plus cold (D4), or also ship the three locks' profiles (small JSON files) for
   lock speed on the very first start?
