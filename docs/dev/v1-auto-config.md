# v1 — Auto config: `holon serve <model>` with no hand-set numbers

Status: **rev 3 (2026-10-28), built; merged with GLM's close on `v1-glm-close`, every outdated lock propagated there,
byte-identical (§5).** Rev 2 answered the
three-reviewer panel on rev 1 (`ba2d2fcb`): A for correctness, B for architecture, C for performance and UX. Rev 3
answers their re-reviews of rev 2 (`97fe7271`) and takes the coordinator's shape decisions. Every finding is answered
in §11; the owner's questions are in §10.

The owner's request (2026-10-28): "I see in the readme, we require users to manually set the --host-memory? that is
not good, we should have default config and auto tune, easy to use is important, we might need a auto tune library
for more complicated configs later".

**Goal.**
- On a supported card, `holon serve <model>` starts every README lock at the lock's speed, with no flag and no file
  prepared by hand.
- When the host lacks RAM, it degrades, or refuses once, naming the shortfall in bytes.
- Pinned host memory never pushes the host into swap.
- Every value chosen automatically is printed with its reason and can be overridden by a flag.
- Automatic values are derived only by `holon serve`, when it starts, from the machine and the model. They are never
  stored in a lock (README:61), and never derived by in-process engines (propagate, the judge, pytest) unless a tool
  asks for it (the bench's automatic arms).

Estimates are marked *est.*, and claims not checked against code or a measurement are marked *unverified*. Everything
else cites `file:line` at `de1f0d28`.

## 1. Inventory: what a user supplies by hand today

| Lock (README:39-53) | Hand-supplied today | Why it is manual |
|---|---|---|
| Resident locks (8B, 30B on 24–32 GB, 27B on RTX 4090 / 3090) | nothing | — |
| Qwen3.8-27B, RTX 5080 and 5090 (KV tier) | `--host-memory 17179869184` (README:66) | `host_memory` defaults to 0, meaning no host pools (`engine.py:87`). The tier's 16 GiB host level (`state_tier.host.bytes`, in the lock) is refused with a `RuntimeError` (`tier.py:204-209`), kind "error", so the start is retried 4× before defaults serve (`supervisor.py:350-354`) |
| Qwen3-30B-A3B Q4_K_M, RTX 5080 (offload) | `--host-memory 21474836480`; the routing profile `profile_x4_gate.json`, made by `holon_tools profile … --host-memory …` and placed in the HF cache's `blobs/` (README:326-334) | host memory as above. The lock names the profile file, which the store reads beside the checkpoint (`store.py:873-898`); a missing file raises, so the start fails 4 × 600 s, then exits (README:427-431) |
| Qwen3.8-Flash-Next IQ2_XS, RTX 5080 | `--host-memory 70000000000 --memory-fraction 0.97`; the MTP pack (a Strata clone at `296b59d` and three commands); the profile, with 4 options retyped from the lock; `--plugin-options mtp.path=…` (README:346-361) | host memory as above. `memory_fraction` is not planned (`plans.py:36`), so it cannot sit in a lock. The lock records the pack only as an identity, `[sources] "mtp.path" = "mtp-q2_0-q8@a55280a0417d4b19"`, never as a path (`plans.py:121-125`). The profile tool builds its engine from flags, not from the lock (`tools/holon_tools/__main__.py:188-194`) |
| Qwen3.8-Flash-Next IQ3_S, RTX 5090 | the same with `86000000000` and `0.985` | the same |
| A borrowed lock | the lent lock's flags and files (README:80-82) | the same causes |
| No lock | nothing, but a file larger than VRAM is refused before download unless `--host-memory` or `--no-size-check` is given (`cli.py:214`) | the size check takes `--host-memory` as the sign of offload |

**Why the Flash-Next locks need 0.97 / 0.985.**
- `state_memory = int(free × fraction) − peak` (`executor.py:230`), so the slack is a share, `1 − fraction` of the
  VRAM left after weights.
- On these locks, every byte past the slack goes to the expert cache, a fill pool.
- At 0.9 the slack is ≈ 1.1 GiB on the 5080 and ≈ 2.7 GB on the 5090, where what it must cover is small:
  - the allocator's segments over the reserved step beyond its live peak: 0.08 GB on the 5090 and 0.19 GB on the 5080
    after 168562e (`v1-flash-next.md` §23.2), then 0.09 GB and 0.07 GB after 585c1862 (§23.3);
  - what grows outside the allocator.
- The share was tuned by hand only because it is a share. D2 derives the slack instead.

## 2. Principles

- **P1. Deployment values, not the lock's.** Only `EngineConfig` fields that plans do not record (`plans.py:36-37`),
  and files a lock names by identity, become automatic. Nothing written into a lock changes.
- **P2. Only `holon serve` derives, gated on `clamp_context` (B-R5).** `clamp_context` is already the "fit to this
  machine" bit:
  - it is unplanned and hidden from the flags (`cli.py:52`);
  - it is set only by `holon serve` (`cli.py:115`) and off in-process (`engine.py:73`).

  It now also lets the state tier take less than its option asks. `memory_fraction=0` gates itself: only `holon
  serve` passes 0 (a `float | None` field would change a listed ABI type, a major bump). The host budget is taken in
  the CLI and reaches the engine as a plain `host_memory` int. In-process engines (propagate, tune, the judge, pytest)
  keep 0 and 0.9 and never read the live RAM.
  - The bench's automatic arms opt in with `--auto` (A-R2-M2, C-N2). It sets `clamp_context` and `memory_fraction`
    0, and takes the host budget the way `holon serve` does.
- **P3. Host values once per session.** The host budget, the spill and the tier's size are computed once, when the
  server first starts, and kept across crash restarts (B-R7). VRAM is re-measured at every start, because it must fit
  the card as it is then. The `(name, value, why)` start line is a log format, nothing more.
- **P4. No silent plan change.** The automatic values leave the plan id unchanged. Two degradations change the plan,
  each by design and each shown as `differs from the lock` with its cost: the spill (D1) and speculation turned off
  (D5).

## 3. Decisions

### D1. Host memory: one budget, taken by the CLI

**The rule.** Unless `--host-memory` is given, `holon serve` takes one reading just before the first engine start:
`budget = max(0, available − headroom)`.
- **Where it goes.** The budget is written into every history entry as `host_memory`, a plain int (B-R4). Every
  restart reuses it, with no IPC or engine change.
- **Why it matches a hand-set cap.** Every host pool requests a fixed size (`min_units == pref_units`, verified by A),
  so the budget is only a cap. The bytes pinned are what the build asks for, the same as under a hand-set cap.
- **`memory.py` is unchanged.** Its refusal (`memory.py:236-244`) already names every pool held.
- **The reading precedes the grammar workers.** The CLI holds only torch and the model plugin, a few hundred MB that do
  not grow.

**Available RAM** (`platform.host_memory()`):

```
available = min(MemAvailable,                                        # /proc/meminfo
                min over the cgroup and its ancestors of
                    limit − memory.current + active_file + inactive_file − file_dirty − file_writeback)
limit     = min(memory.max, memory.high)                             # cgroup v2, under /sys/fs/cgroup
headroom  = max(6 GiB, 5% of min(MemTotal, the lowest limit))
```

- **What is counted reclaimable.** Both file lists count, as `MemAvailable` counts them (A-M1, C-M1). Dirty and
  writeback pages are subtracted, since they free only after I/O (A-R2).
- **cgroup v2 only (B-R2).** The path from `/proc/self/cgroup` is joined to `/sys/fs/cgroup`. When it does not exist
  (a container under `cgroupns=host`), the mount point's own files are read, which are the container's.
- **cgroup v1** logs one line, "cgroup v1: container limits not read; pass --host-memory", and uses `MemAvailable`.
  Every current desktop distribution is v2.
- **Checked on this host** (2026-10-28): under `systemd-run --user --scope -p MemoryMax=16G`, `/proc/meminfo` shows
  the host's total, while the scope's `memory.max` reads 16 GiB.
- **Unverified:** that pinned pages are charged to the cgroup; §9 checks it.
- **The 6 GiB floor** is provisional. §9.3 measures the growth after the snapshot (the grammar workers, the engine),
  and the floor becomes that growth plus ≈ 3 GiB for a desktop (C).

**One `holon serve` at a time pins (A-M3; after the code review, B A1).** The CLI's `pin_lock()` takes an exclusive
`flock` on the per-user `~/.cache/holon/pin.lock`.
- **Who takes it:** every `holon serve` start, explicit `--host-memory` ones too, from its reading until the first
  engine reports ready; and every restart, for its admission and start. In-process engines take none: their caps are
  the caller's (P2), and a host-wide lock would serialize test shards, propagations and the bench.
- **Opening it:** `O_RDWR | O_CREAT` without `O_EXCL`, so two first starts that create it at once open the same file
  (A C-M1). A cache that cannot be written, or a lock held past 900 s, is a warning, never a failed start.
- **Not serialized:** two users' starts at the same moment, and two containers without memory limits. The README
  says so.

**Crash restarts (A-R2-M3, A C-M3).**
- The supervisor keeps each history entry's last ready report, which carries the pinned bytes (`host_bytes`, from
  `plans.served`). Before a restart it runs the CLI's `admit_restart(config, that entry's report)`, a context held
  across the start.
- **An entry that has run:** refused with kind "memory" if its `host_bytes > available − headroom/2`, read live under
  the pin lock. The refusal rolls back to the spill entry, or gives up with the shortfall.
- **An entry that never ran** (the spill after a refused lock, the defaults): its derived cap is lowered to what is
  free above half the headroom, so the ledger refuses an oversized pool rather than pinning into the headroom. It is
  never checked against another entry's bytes.
- **Under `--host-memory`:** only the lock; the cap is the user's.

**When RAM is short.**
1. **The KV tier shrinks.** Under `clamp_context`, `TierCtx.host_left()` returns the budget less the host pools held;
   otherwise it returns None and the tier is strict, as today.
   - The tier takes `min(state_tier.host.bytes, host_left())`, split by its shares.
   - Below its floor it raises `OutOfUnits`, now exported by `holon.api`.
   - The tier's `except Exception` around `host_pool` (`tier.py:204-209`) is narrowed to `OutOfUnits`, which it
     re-raises with its message, so the kind stays "memory".
   - The start line says "state tier host level 16 → 9.5 GiB".
2. **The spill is a history entry (B-M2).** An optional plugin hook,
   `spill(options, checkpoint) -> (options, note) | None`, returns the options under which the plugin serves with less
   host memory, and a note of the cost.
   - **Who is asked:** the model plugin, and the plugins declaring the lock's options.
   - **The history** is `[defaults?, lock + spill, lock]`. A spill is dropped if a flag sets any of its keys.
   - **An empty options mapping** means a manual alternative: no entry is made, and the note is printed only in a
     refusal. This is how std's offload names its experts' RAM-short mode (`moe.ram`, 1.9–8.3× slower).
   - **Storage is the hook's business (B-R3).** `models.qwen4_exp` offers `rows.place` with `ple:ssd` (at the I/O
     worker's default depth: GLM removed `cpu_task.io_threads`) unless the checkpoint's disk is positively `hdd` or
     `network`.
     - It reads the filesystem type from `/proc/self/mountinfo` and `queue/rotational` from sysfs, following
       `slaves/` through LUKS, LVM and md to the real drives (C-N6).
     - `unknown` offers the spill and says so in the note.
   - **The spill entry is a memory-only target (A-R2, C-N4).** The supervisor rolls back into it only after a "memory"
     failure. Crashes and transient failures skip it, and the log says which.
   - **The doomed first build.** It is timed in §9. A memo of refused budgets is added only if it exceeds 60 s.
3. **Refuse once (A-B1, A-R2-M1).**
   - **When the defaults entry is dropped.** Only when a lock entry exists, the checkpoint's bytes exceed the card's
     VRAM, and the flags give neither `--host-memory` nor any plugin option (the only way to choose an offload store
     by hand).
   - **An empty history** (for example when `checked_options` drops a large model's lock) exits with a message, not a
     traceback.
   - **Every failed start's reason is printed** (`supervisor.py:343-357`).
   - **On a give-up after a host-memory refusal,** the CLI prints, in order (C, on the 32 GB case):
     1. the GiB to free;
     2. the locks of the same repository for this card whose files fit its VRAM. These need no pinned RAM, and are
        found from `holon models`' rows and the weight source's listed sizes, with no model names in core;
     3. `--host-memory N` to start anyway, at the risk of swapping;
     4. the plugins' manual notes, such as the RAM-short mode and its cost.

**The 30B offload lock on a 32 GB host refuses** when the desktop and Holon's processes use more than ≈ 8.8 GiB
(*est.*). The lock pins its experts, 16.35 GiB read from the GGUF header, plus small pools.
- **The refusal's alternative.** It names `unsloth/Qwen3-30B-A3B-GGUF:UD-IQ3_XXS`, this card's lock, which fits
  without pinned RAM. It finds it through the general search in step 3.
- **The floor** comes from §9.3's measurement, not from lowering it now (C).

**The size guard before download (A-M7).** A reference with a lock for this card, verified or borrowed, is not
size-checked. Anything else keeps today's VRAM check (`cli.py:214`).

### D2. GPU memory: the slack is measured where the cache is short

`memory_fraction` 0 derives. `holon serve` passes 0 unless the flag is given.

```
budget(0.9) = int(free × 0.9) − peak                                  # today's
if a fill pool would stay below its preferred size at budget(0.9):    # C-N3
    budget = free − peak − min(free / 10, overshoot + reserve)
reserve = 1 GiB if the card drives a display or a process outside Holon holds memory on it, else 256 MiB
```

- **Only where it buys something.** The ledger's own sizing (`Ledger._sizes`) decides whether a fill pool would get
  less than its preference at 0.9.
  - Today only the tiered store's arena is a fill pool (`store.py:385`).
  - Flash-Next's arena wants every expert, so the rule applies there.
  - The 30B's arena is pinned below its preference, so it keeps 0.9, the slack it was validated with.
  - Resident and tier locks keep 0.9.
- **Overshoot.** The allocator's reserved growth less the live peak, measured after `empty_cache()` on the step whose
  peak is reserved (the small step under `lend_activations`, `executor.py:217-220`), and clamped at 0.
- **"Shared" (C-N1, A-R2).** It comes from NVML (`nvidia-ml-py`, a new core dependency, already in `uv.lock`):
  - display active; or
  - a compute or graphics process with memory on the device that is visible in this PID namespace (`/proc/<pid>`
    exists), not the engine itself and not its parent.

  The start line prints `headless` or `shared (<why>)`.
  - Processes in another PID namespace are not visible, so a container's card reads headless unless a display is
    active. That is logged.
  - Without NVML (arm64: `nvidia-ml-py` is locked for x86-64 only), the card counts as headless, logged with
    `--memory-fraction` as the override.
- **What it buys (C §1.1, *est.*).** The headless 5090 lands between 0.985 and 0.99. The headless 5080 ties 0.97. A
  display-driving 5080 costs ≈ 2 points of experts against 0.97.
- **Never less than today.** The `min` keeps every budget at or above 0.9's.

### D3. Cache: `~/.cache/holon` only

- **The location.** A private constant in core, `Path.home() / ".cache" / "holon"`, which the std weight caches
  already use. There is no XDG and no environment read. Build products follow XDG; user data follows `$HOME`.
- **Contents:** `sources/<identity>/` and `sources/<identity>.failed` (D5). Hub downloads go through
  `huggingface_hub` and honour `HF_HOME`. Nothing is written into the repository or `plans/`.
- **Docker.**
  - The image gains `ENV HOME=/opt/holon/user`, with `/opt/holon/user/.cache/holon` created writable by all. Without
    that, a bind mount there would leave its root-owned parents unwritable for `--user` (B-R6).
  - This moves `HOME` for every tool in the image, root runs included: Python's user site and tool configs under `~`.
    That is harmless, because the image keeps nothing under `/root`.
  - The README's Docker lines mount `-v <cache dir>:/opt/holon/user/.cache/holon`. Its `-e HOME=/cache` recipe keeps
    working.
  - A cache that cannot be written is never fatal.

### D4. Routing profiles: shipped as data, never fatal

- **Never fatal (std, `store.py:886-910`).** A named profile that is absent, unreadable, malformed or counted on other
  files logs a warning, and the store falls through to the next source.
- **Shipped as std data** in `std/holon_std/tiered/profiles/`, written by `holon_tools profile --ship FILE`, which
  validates an existing profile and imports it.
- **Matching.** A shipped profile matches the store's checkpoint (`store.py:878-884`) when the file sizes agree, and
  the names too where both are 64-hex hub blob names. For a hub download that is a match by content hash. A profile is
  only a seed, so a size-only match risks speed, never output.
- **Lookup order:** the lock's named file beside the checkpoint, then the shipped profile, then cold. The store logs
  which one it used.
- **What ships now.** The 30B's `profile_x4_gate.json`, the file its lock's evidence used. The two Flash-Next profiles
  need a GPU run on their cards (§9).
- **The tool.** `holon_tools profile` starts from the lock, as `propagate` does.

### D5. The MTP pack: found by identity, else no speculation (the owner's decision, 2026-10-30)

The owner decided: no published pack and no local build. Main has meanwhile moved to loading Strata's pack as it is
(`mtp-q2_0@a22206a4d74eafbf`, BF16 projections; `holon_tools mtp-pack` removed).
- **Lookup (core `cli`).** For each `[sources]` key of the lock that no flag gives (`mtp.path=` empty counts as given),
  the CLI checks that the identity is shaped `^[A-Za-z0-9._+-]+@[0-9a-f]{16}$` and takes the file in
  `~/.cache/holon/sources/<identity>/`. The engine compares the opened file's identity with the lock's
  (`plans.py:121-125`).
- **When it is missing, no speculation.** Both lock entries get `max_draft_tokens 0`, with one warning: the key, the
  identity, where it was looked for, and the plugin's hint.
  - The hint is an optional hook, `source_hint(identity) -> str | None`, asked of the same owners as `spill`. It
    replaces rev 3's `MakesSources.make_source` in the ABI (`HintsSources`; 1.6 is not released, so the swap is free).
  - `models.qwen4_exp`'s hint gives the cost (24–37% of decode, measured) and Strata's commands from the README,
    writing the pack into that directory.
  - An unreadable cache is the same as a missing file.
- **Flags that turn speculation off** (`--max-draft-tokens 0`) skip the lookup.
- **Removed:** the published-pack download, its memo and link, and `models.qwen4_exp`'s `huggingface-hub`
  dependency.

### D5b. The launch lines' plugin options

Unchanged from rev 2:
- the profile command's options come from the lock;
- `mtp.path` is D5;
- the PLE-to-SSD flags are the spill;
- `--host-memory` and `--memory-fraction` are automatic;
- features and preferences stay flags.

### D6. The auto-tune seam

**The seam is two things the code has after this design:**
- (a) an unset or `auto` value means "derive": `--max-model-len auto`, `memory_fraction=0`, and the CLI's host
  budget;
- (b) `serve_history`'s layering of defaults, lock and flags, where a per-machine record would slot in between the
  lock and the flags.

**What `tune` is.** `tune.tune()` is one pass over the decode ops, keeping the pinned provider unless beaten by 3%.
`split` is F5b's own two-stage search. Both run offline per GPU class and land in locks through `propagate`.

**The later library adds:**
- searches over byte-neutral options per machine;
- a per-machine record layered by `serve_history`;
- `holon tune` for users;
- learned profiles.

## 4. Core impact (invariant 5; `check_core_loc.py`: 5019 / 6000 before)

As built after the code review, `check_core_loc.py` counts **+201 → 5220 / 6000**, against rev 3's estimate of
≈ 115 and the coordinator's target of ≈ +170 after B's part A:

| Change | Where | Code lines |
|---|---|---|
| `host_ram()` (meminfo + cgroup v2, an unreadable cgroup falling back to meminfo), `host_headroom()`, `gpu_shared()` (None without NVML), the overshoot in `peak_memory` | `platform` | 43 |
| `memory_fraction` 0, `TierCtx.host_left`, the GPU start line | `engine.py` | 6 |
| the slack rule with the fill-pool test, via the ledger's public `sizes` | `executor.py` | 17 |
| the per-user pin lock; the budget; per-entry restart admission; the history `[defaults?, lock + spill, lock]`; `fit_lock` (sources, spills, notes; owners skip kernels and survive load failures); `source_file`; `checkpoint_bytes`; the size guard on `repo_ref`'s name; the refusal's notes | `cli.py` | 111 |
| memory-only entries with their notes, `admit` per entry, each failure printed, the one `_say` | `supervisor.py` | 13 |
| the refusal's shortfall and override; `host_bytes` | `memory.py`, `plans.py` | 5 |
| `OutOfUnits`; `Spills`, `HintsSources`; `TierCtx.host_left` | `holon.api` | 6 |

B's part A took out the engine-side lock, the `/tmp` open sequence, the regex over our own message, the hub lookups in
the error path and the regex over the weight source's display string. What brought the total back above +170 is what
the code review asked to add: the lock on explicit serves and restarts, per-entry admission with the cap lowered for
an entry that never ran, the unreadable-cgroup fallback, and the "unknown" reserve without NVML (≈ +20).

- The ABI gains only additions, so the bump is minor (1.5 → 1.6).
- `nvidia-ml-py` becomes core's fifth dependency (B-R1).

## 5. Exactness and locks

- **Unchanged from rev 2:** the automatic values change no plan id, and the two plan-changing degradations are
  logged. D2 moves no trace, because `propagate` pins `pool_bytes`.
- **The tier's `clamp_context` gate is off in-process,** so no lock-writing path can shrink the tier.
- **Propagation.** The core change, std's tier and store, and `models.qwen4_exp` outdate **all 25 lock classes**.
  They were propagated in one combined round with GLM's close (`v1-glm-close`, e0e1e518: 27 locks with GLM's two,
  every snapshot trace byte-identical over three runs, none needing L2), then again after the close's fix round.

## 6. Build steps

1. **Core:** the API additions, `platform`, the engine and executor, the supervisor, then the CLI, with CPU tests.
2. **std:** the tier's gate and `OutOfUnits`; the store's never-fatal and shipped lookup; std offload's manual spill
   note.
3. **plugins:** `models.qwen4_exp`'s `spill` and `source_hint`.
4. **Tools and bench:** `profile` from the lock and `--ship`; the bench's `--auto`; the Dockerfile; the README.
5. **GPU on an idle RTX A5000 only,** while the bar cards are busy. §9's bar-card runs come after the combined
   propagation.

## 7. What a user sees

```text
holon: host memory: budget <B> GiB = available <A> − kept free 6.0 GiB (max 6 GiB, 5% of <T> GiB; cgroup: none)
holon: GPU memory: state pools <S> GiB = free <F> − peak <K> − slack <L> (allocator <O> + reserve 0.25 GiB, headless)
holon: routing profile: shipped (profile_x4_gate.json)
holon: start failed (memory): host memory <B> GiB too small: plugin.tiered.pinned.… needs …, <L> GiB left
holon: retrying the lock with rows.place=ple:ssd,token_embd:host (spill, after a memory refusal): PLE table read from the SSD, byte-identical; prefill −16/−26% at 1K/4K, MTP decode −6/−7% (RTX 5090)
holon: WARNING: mtp.path mtp-q2_0@a22206a4d74eafbf unavailable (not in ~/.cache/holon/sources/mtp-q2_0@a22206a4d74eafbf/; serving without the MTP head costs 24–37% of decode (measured on an RTX 5080 and 5090). Strata's public tools make it …): speculation off
```

A give-up after a host-memory refusal ends with the four ordered hints of D1 step 3. The overrides are as in rev 2.

## 8. The README afterwards

- **As rev 2:** the Launch flags section is deleted; the model table's caption, the one-line Quickstart, the Docker
  cache volume, and the Host RAM table with "RAM for full speed".
- **"Other use" in the table includes Holon's own ≈ 3 GiB** (C-N7, *est.*, measured in §9.3).
- **The Flash-Next section** makes the pack with Strata's commands straight into the cache directory; without it
  the server runs without MTP and says so.
- **Known limitations** add:
  - serialized starts;
  - two containers without memory limits;
  - cgroup v1;
  - ZFS's ARC;
  - oomd.

## 9. Test plan and acceptance

**CPU** (`pytest`, lean):
- `host_memory()` over fake trees: v2 with a limited ancestor, `memory.high`, file and dirty pages, the
  `cgroupns=host` root, v1, none;
- `pin_lock`: a second holder waits, then warns; a cache that cannot be made only warns;
- `gpu_shared` with a stubbed NVML: own PIDs excluded, invisible PIDs, no NVML;
- the slack rule: no fill pool, a fill pool at its preference, a fill pool short of it;
- history: defaults kept without a lock, or with `--host-memory` or plugin-option flags; an empty history exits with
  a message; a spill dropped by a flag; the spill entry reached only by a memory failure;
- `admit` refusing a restart;
- D5: the identity check, the fallback to no speculation on both entries, the failure memo, `--max-draft-tokens 0`;
- the tier: shrinks under `clamp_context` and is strict without it; the floor raises `OutOfUnits`;
- the store: missing, malformed and mismatched profiles fall through; a shipped profile matches by hash and by size;
- `qwen4_exp`'s storage classifier on fake mountinfo and sysfs, LUKS-on-NVMe included.

**GPU acceptance** is as rev 2's §9, with these changes:
- **§9.1** asserts `headless` on the bar cards.
- **§9.2** runs the automatic arms with the bench's `--auto`.
- **§9.3 measures the headroom's floor:** Holon's RSS at the reading, and its growth after it.
- **§9.3 and §9.4 need a host with swap** (the owner's desktop, or a node with a swapfile) before the merge. A run
  reported "not run" is a failure of the acceptance, not a pass (C-N7).
- **The bar-card runs wait for the combined propagation.** On this branch, GPU checks run only on an idle RTX A5000.

## 10. Questions for the owner

1. **Reserve.** 1 GiB on a shared card: accept?
2. **Publishing the MTP pack** on the project's Hub repository: go? Until then, Flash-Next serves without MTP unless
   the user builds the pack by hand.
3. **The doomed-build memo:** only if §9 times the doomed build over 60 s?

## 11. Findings and answers

| Finding | Answer |
|---|---|
| A-B1 refusal rolls back into defaults that cannot hold the model; tier not "memory"; spill not persisted, flag precedence, merging, `--no-supervised` | **Fixed.** No defaults entry when the checkpoint exceeds VRAM; refuse once with the last reason; every failure printed (D1). The tier raises `OutOfUnits` from the API. The spill is a history entry: it persists, flags win in `cli`, and each plugin's hook merges its own options. `--no-supervised` serves the newest entry only, as today, and gets no spill or fallback (stated) |
| A-M1 active page cache counted as used | **Fixed:** both file lists are reclaimable (D1) |
| A-M2 cgroup path in containers; v1 | **Fixed:** the mount root's own files as fallback; rev 3 trims to v2 at `/sys/fs/cgroup` (B-R2), v1 logged |
| A-M3 concurrent starts race | **Fixed:** host-wide `flock` from snapshot to pinned |
| A-M4 tier sized, then re-checked live | **Fixed:** one snapshot; the tier sizes against `host_left()`; below the floor `OutOfUnits` |
| A-M5 thin slack on shared GPUs | **Changed:** D2 only with fill pools; reserve 1 GiB when shared (NVML); browser in the soak |
| A-M6 `--max-draft-tokens 0` offline | **Fixed:** skipped lookup; unavailable source → no speculation, never a refusal |
| A-M7 size guard lost for unlocked files | **Fixed:** RAM ignored unless the reference has a lock for this card |
| A-M8 degradation in lock-writing and in-process paths; learned profile writes | **Fixed:** gated on `clamp_context` (rev 3), set only by `holon serve`; learning dropped; profile used is reported, bench refuses a substituted one |
| A-M9 30B offload on 32 GB; headroom cap | **Answered:** D1 states when it starts (≤ ≈ 8.8 GiB other use) and refuses once otherwise; headroom now `max(6 GiB, 5%)`; a 32 GiB scope in §9.3; Q1 |
| A-m1 overshoot measurement | **Fixed:** after `empty_cache`, on the reserved step, clamped at 0 |
| A-m2 pinned pages on the anon LRU | **Kept in §9:** scope `pswpout` / `swap.current` checks on the oldest bar kernel |
| A-m3 systemd-oomd | **Fixed:** README note; scope PSI in §9.3 |
| A-m4 ZFS ARC | **Fixed:** README note; the refusal names `--host-memory` |
| A-m5 learned-profile robustness | **Moot:** learning dropped; the malformed-profile fall-through kept |
| A-m6 D5 fetch details | **Fixed:** 206 + `Content-Range`, temp dir with cleanup, per-identity `flock`, identity sanitized, numpy port tested on three ISAs, a hash mismatch discarded |
| A-m7 `__post_init__` on None; `float \| None` flags | **Fixed:** counted in §4 |
| A-m8 the core change outdates every lock | **Fixed:** propagation plan in §5 |
| A-m9 XDG moves the digest cache | **Moot:** no XDG |
| A-m10 kernel stack growth from the slack | **Kept:** the soak (§9.6) covers it; D2's 256 MiB only applies headless with fill pools |
| B-M1 XDG breaks Docker persistence | **Fixed:** `~/.cache/holon` only; Dockerfile `HOME` and a volume (D3) |
| B-M2 spill as a history entry | **Adopted** (D1) |
| B-M3 budget once at start | **Adopted**, and carried across restarts |
| B-M4 D7 misdescribes `tune`; the seam | **Fixed:** D6 rewritten; no `/metrics` |
| B-m1 ABI wording | **Fixed:** additions only, minor bump |
| B-m2 `cache_dir` not ABI | **Adopted:** a private constant |
| B-m3 `make_source(identity, directory)`, any plugin | **Adopted** |
| B-m4 byte parser and v1 as possible YAGNI | **Byte parser dropped; v1 kept** at the coordinator's call (`hierarchical_memory_limit` is one read) |
| B-m5 a shrunk tier shown as a derived value | **Fixed:** start line |
| B-m6 D5 degrade, not refuse | **Adopted:** no speculation, with a warning |
| C-M1 cgroup formula | **Fixed** (= A-M1) |
| C-M2 restarts re-derive | **Fixed:** the budget is carried in the history entry; the spill is an entry |
| C-M3 spill onto a slow or network FS | **Fixed:** rev 3: refused only on `hdd` or `network` (B-R3), stacked devices resolved |
| C-M4 256 MiB on a display card; resident locks thinner | **Changed:** reserve 1 GiB when shared; resident and tier locks keep 0.9 |
| C-M5 bar not noise-aware | **Fixed:** the §23.7 ship rule, 0.1-point tolerance, 4 rounds on 1K rows |
| C-M6 D2 and D4 confounded | **Fixed:** arms (a)–(d) |
| C-M7 vacuous no-swap tests | **Fixed:** desktop-sized scopes with swap, scope counters, lag probe, numeric need and time bounds |
| C-M8 degraded speed not benched | **Fixed:** §9.5 |
| C-m1 size guard | **Fixed** (= A-M7) |
| C-m2 the pack's build size, time, RAM, disk | **Fixed:** in the CLI before the supervisor, progress, per-slice quantizing, cleanup, resumable; Q3 weighs (a) |
| C-m3 grammar workers counted twice | **Fixed:** the snapshot precedes them |
| C-m4 learning's cost | **Moot:** learning dropped |
| C-m5 start log: experts cached, spill cost, cold note, tier shrink | **Fixed** (§7) |
| C-m6 README "RAM for full speed" and the pinning limitation | **Fixed** (§8) |
| C §1.1 overshoot misquote | **Fixed** (§1): 0.08 / 0.19 GB after 168562e, 0.09 / 0.07 GB after 585c1862; the 5080's lending step unmeasured |
| **Re-reviews of rev 2** | |
| A-R2-M1 dropping defaults empties lock-less histories | **Fixed:** dropped only with a lock entry and no `--host-memory` or plugin-option flags; an empty history exits with a message (D1) |
| A-R2-M2 / C-N2 the bench cannot run automatic arms | **Fixed:** the bench's `--auto` (P2) |
| A-R2-M3 a carried cap pins into the headroom on restart | **Fixed:** `admit` reads live once and refuses (kind "memory") when `host_bytes > available − headroom/2` |
| A-R2 page cache: dirty and writeback | **Fixed:** subtracted |
| A-R2 `flock` open sequence, in-process pinners, serialized starts, containers | **Fixed** (D1): no `O_CREAT` on an existing file, `fchmod`, failures warn; in-process pinners lock; queueing and containers documented (superseded after the code review: a per-user CLI lock, none in-process) |
| A-R2 / C-N4 the spill entry catches every rollback | **Fixed:** memory-only target, logged |
| A-R2 the tier's `except Exception` | **Fixed:** narrowed to `OutOfUnits`, re-raised as `OutOfUnits` |
| A-R2 the speculation-off fallback on both entries | **Fixed** (D5) |
| A-R2 / C-N1 "shared" counts Holon's own processes; PID namespaces; arm64 | **Fixed** (D2): own PID and parent excluded, only visible PIDs, no NVML → headless with a log line; §9.1 asserts `headless` |
| B-R1 core lines | **Taken in shape;** as built +218 with the later asks (§4); `nvidia-ml-py` named |
| B-R2 cgroup v2 only | **Taken:** v1 logs and falls back to `MemAvailable` |
| B-R3 storage in the plugin; offer unless `hdd` / `network` | **Taken** (D1 step 2) |
| B-R4 reading and `flock` in the CLI | **Taken:** no IPC carry; the budget is a plain int in every entry |
| B-R5 gate on `clamp_context` | **Taken:** no `adaptive` field |
| B-R6 Dockerfile path and `HOME` scope | **Taken** (D3) |
| B-R7 P3 wording | **Fixed:** "host values" |
| C-N3 D2 on the 30B for no gain | **Fixed:** D2 applies only when a fill pool would stay short of its preference at 0.9 |
| C-N5 the fallback's cost; memo failures | **Fixed:** the plugin's reason states the cost; `sources/<identity>.failed` for a day |
| C-N6 stacked devices | **Fixed:** `slaves/` followed; LUKS-on-NVMe in the CPU tests |
| C-N7 "not run" is not a pass; "other use" | **Fixed** (§9, §8) |
| C, the 32 GB case | **Taken:** refuse, with the four ordered hints; the floor from §9.3's measurement |
| Coordinator: MTP pack | **Changed:** published-pack download is the primary path; no local build (not cheap); speculation off until published |
| **Code review of the build (2a55ab3b)** | |
| A C-B1 no lock re-propagated | **Merge gate, as planned:** the combined round with GLM's close. Found meanwhile: on `main` itself 22 of 25 newest locks are already outdated (they pin an older `holon`, `kernels.ggml_cuda` and `std.weights.gguf` than `main` holds), so `main` already serves defaults for them; the round fixes both |
| A C-M1 the pin lock's create race | **Fixed:** per-user file opened `O_RDWR \| O_CREAT` (B A1) |
| A C-M2 an unwritable cache crashes Flash-Next | **Fixed:** `fit_lock` catches `OSError`; the plugin's memo write is suppressed; test |
| A C-M3 admission against the first entry's bytes | **Fixed:** reports per entry; an entry that never ran gets its cap lowered to what is free; test |
| A minor 1 the "not published" memo | **Fixed:** never memoized; a published fetch's failure is remembered an hour, keyed by what was published |
| A minor 2 concurrent first fetches | **Fixed:** the link is made under a temporary name and renamed |
| A minor 3, B the defaults kept on any plugin option | **Fixed:** dropped; only `--host-memory` keeps the defaults entry of a lock larger than the card |
| A minor 4 shipped profiles override "no profile" | **Changed:** `moe.placement.profile=none` asks for a cold cache (the CLI's `key=` deletes, so an empty value cannot be told from unset); README and manifest say so |
| A minor 5 an unreadable cgroup; `why` | **Fixed:** falls back to meminfo, saying so; `why` names the binding level |
| A minor 6 NVML failure shown as headless | **Fixed:** "unknown without NVML", reserve 256 MiB, `--memory-fraction` named |
| A minor 7 unlocked pinning starts | **Fixed:** explicit serves and restarts take the lock; the bench's `--auto` arms take none (said in its docstring) |
| A minor 8 Flash-Next off its lock until the pack is published | **For the owner:** unchanged (§10 Q2) |
| B A1–A5 | **Taken:** per-user CLI lock, the ledger's own refusal text, `repo_ref`'s name, one `_say`, owners over options and sources, kernels skipped, load failures survived |
| B: `executor.slack` on `_sizes` | **Fixed:** `Ledger.sizes` is public |
| B O1, O2 (drop the spill tag, admission) | **Kept** at the coordinator's call |
| B part B (a `deploy` slot) | **Deferred** to the auto-tune library |
| **The owner's MTP decision (2026-10-30), and main's load-only pack** | |
| No published pack, no local build | **Done:** the download path, memo and link are gone; a missing pack serves without speculation with one warning (the cost and Strata's commands); `MakesSources` became `HintsSources.source_hint` (D5) |
| The Flash-Next routing profiles | **Shipped:** both, counted with Strata's pack (`mtp-q2_0.gguf`, 889,017,216 bytes, sha256 `a22206a4…`), under the names the new locks give; the store now matches a profile on its largest file, the target's, so a deployment without the MTP head still seeds from it |
| Rebased on main (75d73fc2) | **Done:** README conflicts resolved for the load-only pack |
