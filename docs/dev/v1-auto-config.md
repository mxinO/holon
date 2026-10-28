# v1 — Auto config: `holon serve <model>` with no hand-set numbers

Status: **design, rev 2 (2026-10-28), not built.** Rev 2 answers the three-reviewer panel on rev 1 (`ba2d2fcb`):
A for correctness, B for architecture, C for performance and UX. It takes B's simplifications and the coordinator's
shape decisions. Every finding is answered in §11; the owner's questions are in §10.

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
  stored in a lock (README:61), and never derived by in-process engines: propagate, the judge, the bench, pytest.

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
- **P2. Only `holon serve` derives.** A new unplanned field, `EngineConfig.adaptive: bool = False`, marks a server's
  start; `holon serve` sets it.
  - The defaults stay as today: `host_memory` 0 and `memory_fraction` 0.9.
  - In-process engines (propagate, tune, the judge, the bench, pytest) therefore behave exactly as now: strict, never
    shrinking, never reading the live RAM.
- **P3. One rule per value, printed and overridable.** Each value is computed once, when the server first starts, and
  kept across crash restarts (D1). The `(name, value, why)` start line is a log format, nothing more.
- **P4. No silent plan change.** The automatic values leave the plan id unchanged. Two degradations do change the plan,
  each by design and each shown as `differs from the lock` with its cost: the spill (D1) and speculation turned off
  (D5).

## 3. Decisions

### D1. Host memory: one automatic budget, taken at start

**The rule.** Under `holon serve`, `host_memory` is None unless a flag gives it.
- **When the budget is taken.** At `engine.py:143`, where the ledger is created, the engine takes once
  `host_memory = max(0, available − headroom)`.
- **What it covers.** Every pool in the tree requests a fixed size (`min_units == pref_units`; A verified each site),
  so the budget is only a cap. The bytes pinned are what the build asks for, the same as under a hand-set cap.
- **`memory.py` is unchanged.** Its per-pool refusal report (`memory.py:236-244`) already names every pool held.
- **The snapshot precedes the grammar workers,** which are built later (`engine.py:189-192`). The headroom therefore
  covers their 1.6–2.2 GB once, not twice (C-m3).

**Available RAM** (`platform.host_memory_facts()`, a machine fact):

```
available = min(MemAvailable,                                  # /proc/meminfo
                min over the cgroup and its ancestors of
                    limit − memory.current + active_file + inactive_file)        # v2: memory.stat
limit     = min(memory.max, memory.high);                     v1: hierarchical_memory_limit (≥ 2^62: none),
                                                                  usage_in_bytes, total_active_file + total_inactive_file
headroom  = max(6 GiB, 5% of min(MemTotal, the lowest cgroup limit))
```

- **Page cache is reclaimable.** Both file lists count as reclaimable, as `MemAvailable` counts them. The model file's
  page cache is charged to the cgroup, and a file read twice sits on the active list (A-M1, C-M1). Shmem is on the anon
  lists, so it is not counted.
- **Resolving the cgroup.** The cgroup is found through `/proc/self/mountinfo`: the cgroup2 mount, or v1's `memory`
  mount. The mount's root is stripped from the path in `/proc/self/cgroup`, then joined to the mount point.
  - When the joined path does not exist (a container under `cgroupns=host`), the mount point's own files are used,
    which in a container are the container's. It is never "no limit" (A-M2).
  - v2 walks the ancestors up to the mount point. v1 reads `hierarchical_memory_limit`.
- **Checked on this host** (2026-10-28): under `systemd-run --user --scope -p MemoryMax=16G`, `/proc/meminfo` still
  shows the host's total while the scope's `memory.max` reads 16 GiB, several levels deep. The cgroup term is
  therefore needed.
- **Unverified:** that pinned pages are charged to the cgroup. That is the kernel's standard charging of faulted
  anonymous pages; §9 checks it.
- **Why this headroom.** What it covers does not grow with RAM: the grammar workers, the frontend, the engine's growth
  after the build, and a desktop's working set. With 5%, the 6 GiB floor governs up to 120 GiB.

**One Holon pinning at a time (A-M3).**
- The engine holds a host-wide `flock` on `/tmp/holon-host-memory.lock` (mode 0666) from the snapshot until its host
  pools are pinned, after the tier attaches (`engine.py:250-252`). A second Holon start's snapshot therefore sees the
  first one's pinned pages.
- A wait over 600 s logs a warning and goes on.
- In a container, `/tmp` is the container's own, and so is its cgroup limit.

**An explicit `--host-memory BYTES`** is a fixed cap, as today. If it exceeds `available − headroom`, a warning says
so once, and the cap is kept.

**Crash restarts keep the first start's values (C-M2).**
- The engine's ready message carries its derived `host_memory`.
- The supervisor writes it into the running history entry (`self.history[self.version]`). Every later restart of
  that entry uses that number as a fixed cap and does not read the live RAM.
- A restart never spills or shrinks differently mid-session. If the host has since run short, the pinning warns or
  fails as a fixed cap would.
- The spill is a history entry of its own (below), so it persists too.
- VRAM is re-measured at every start, because it must fit the card as it is then (D2).

**When RAM is short, three steps.**
1. **The KV tier shrinks.** Under `adaptive`, `TierCtx.host_left()` returns the budget less the pools held (B-M3).
   - Without `adaptive` it returns None, and then the tier takes its option or fails, as today (A-M8).
   - The tier takes `min(state_tier.host.bytes, host_left())`, split by its shares, against that one number. There is
     no second live read (A-M4).
   - Below its floor (`tier.py:180-196`) it raises `OutOfUnits`, now exported by `holon.api`, not a `RuntimeError`.
   - The start line says "state_tier host level 16 → 9.5 GiB".
2. **The spill is a history entry (B-M2).**
   - **Building it.** An optional plugin hook, `spill(options, storage) -> (options, note) | None`, is shaped like
     `TrainedContext` (`api/hooks.py:236`). `holon serve` asks the model plugin and the plugins that declare the lock's
     options. The supervisor's history becomes `[defaults (if they can hold the model), lock + spill, lock]`.
   - **Explicit flags win.** A spill is dropped if a flag sets any of its keys; the flags are local to `cli.py`.
   - **When it runs.** A RAM-short lock refuses with `OutOfUnits`, kind "memory". The existing rollback
     (`supervisor.py:352`) then starts `lock + spill`.
   - **Merging.** One hook per plugin rewrites its own compound options, so two spills on one option never collide.
   - **Today's only spill is `models.qwen4_exp`'s.** The model plugin knows its PLE n-gram table and the measured cost:
     `rows.place` with `ple:ssd` (its other tables unchanged), and `cpu_task.io_threads=4`. Its note gives the cost:
     prefill −16% / −26% at 1K / 4K and MTP decode −6% / −7% on the RTX 5090 (`v1-flash-next.md` §22.1); on the
     5080, unmeasured until §9.
   - **Only onto a local SSD (C-M3).** `storage` is a machine fact, `platform.storage(path)`, for the checkpoint's
     filesystem: `ssd`, `hdd`, `network` or `unknown`. It is read from mountinfo's fstype and the block device's
     `queue/rotational`. The hook offers its spill only on `ssd`. Elsewhere the refusal names the flags.
   - **Not in the history:** the experts' RAM-short mode (`moe.ram` with a disk tier), 1.9–8.3× slower
     (`v1-offload.md:367`). It is a different product, named in the refusal with its cost.
   - **The doomed first build.** A RAM-short host runs the lock's build until its first refused pool: the GPU weights,
     plus for Flash-Next the 26.8 GiB PLE table read and pinned at bind (`rows.py:79-90`), before the store's request
     fails (`store.py:365`). §9 times it.
     - If it exceeds 60 s, `holon serve` adds a memo: a refusal records `<budget, lock digest>` under
       `~/.cache/holon/refused/`, and a later start with a budget no larger skips the lock entry.
     - Predicting the need before the build would need each plugin's pool sizes, which are known only once it is
       built. That is not cheap, so no prediction.
3. **Refuse once (A-B1).**
   - **No defaults entry when they cannot hold the model.** That is when the checkpoint's bytes exceed the card's
     VRAM, the same fact the size guard uses. For the offload locks the history is then `[lock + spill, lock]`. When
     everything refuses, the supervisor gives up at once with the last reason: the shortfall, the pool, `available`,
     the headroom and its source, the pools held, and `--host-memory N` as the override.
   - **Every failed start's reason is printed** (`_launch`, `supervisor.py:343-357`), not only the last.
   - The 27B tier locks fit VRAM, so their defaults stay: below the tier's floor they serve without a tier, with a
     warning.

**The 30B offload lock on a 32 GB host (A-M9).**
- **The need.** It pins its experts, 16.35 GiB (read from the GGUF header, `Qwen3-30B-A3B-Q4_K_M.gguf`), plus small
  pools.
- **The arithmetic.** On a 32 GB host (MemTotal ≈ 31.3 GiB *est.*), the headroom is 6 GiB, so the budget is enough
  when `MemAvailable` at the engine's start is ≥ ≈ 22.5 GiB. That is when the desktop and Holon's own processes then
  use ≤ ≈ 8.8 GiB (*est.*).
- **Otherwise it refuses once.** It has no spill, and its defaults cannot hold the 17.3 GiB file, so there is no
  defaults entry. The refusal names `--host-memory N` (the user takes the risk) and the experts' RAM-short mode.
- **Today it starts with `--host-memory` and may swap.** That trade is the owner's (Q1), and §9 measures it in a
  32 GiB scope.

**The size guard before download (A-M7, C-m1).**
- A reference with a lock for this card in `holon models`' rows, verified or borrowed, is not size-checked. Its lock
  decides whether it fits.
- Anything else keeps today's VRAM check (`cli.py:214`), bypassed only by `--host-memory` or `--no-size-check`.

### D2. GPU memory: the slack is measured where the cache depends on it

`memory_fraction: float | None`; `holon serve` passes None unless the flag is given.

```
slack  = min(free / 10, overshoot + reserve)     if the ledger has a fill pool (a cache sized by the remainder)
slack  = free / 10                               otherwise: today's 0.9
budget = free − peak − slack                     (--memory-fraction F: int(free × F) − peak, as today)
reserve = 1 GiB if the card drives a display or another process holds memory on it, else 256 MiB
```

- **Fill pools only.** Without a fill pool, the extra bytes only lengthen the prefix cache; `max_model_len` and
  `max_seqs` are planned. That gain is unmeasured, while the 10% is the slack every resident lock was validated with,
  and the room a desktop's GPU clients grow into (A-M5, C-M4).
  - "Has a fill pool" is a ledger fact (`Pool.fill`), not a list of locks.
  - Today it holds for the offload locks only. The tier locks' KV is paged, not filled, so they stay at 0.9.
- **Overshoot.** It is the allocator's reserved growth less the live peak, measured after `empty_cache()` on the step
  whose peak is reserved, clamped at 0.
  - Under `lend_activations` that is the small step (`executor.py:217-220`). On the 5080 that is the 0.34 GB
    decode/verify step, whose overshoot is unmeasured (C).
  - +4 lines in `platform.peak_memory` (`platform/__init__.py:91-99`).
- **Reserve.** "Shared" comes from NVML: display active, or another PID with memory on the device. Without NVML,
  shared is assumed.
  - `nvidia-ml-py` is already in `uv.lock` through FlashInfer; core declares it.
  - The 256 MiB covers lazy allocations after the start (§22.9 of `v1-flash-next.md`).
  - The 1 GiB covers a compositor or browser growing later.
- **What it buys (C §1.1, *est.*):**
  - The headless 5090 lands between 0.985 and 0.99: experts cached ≈ +0.2 points.
  - The headless 5080 ties 0.97 within ±0.05 points.
  - A display-driving 5080 costs ≈ 2 points against 0.97, about −1 to −1.5% MTP decode. That is the price of not
    starving the desktop.
- **Never less than today.** The `min` keeps every budget ≥ today's 0.9. Since it applies only with a fill pool, no
  resident or tier lock's pools change.

### D3. Cache: `~/.cache/holon` only

- **The location.** A private constant in core, `pathlib.Path.home() / ".cache" / "holon"`, the same root the std
  weight-digest caches already use (`std/…/gguf/source.py:26`, `hf/source.py:21`). No XDG, so no environment read and
  no exception to invariant 3 (B-M1).
  - Build products follow XDG (the kernel extensions, `cpu_task/extension.py:22`), while user data follows `$HOME`.
  - `cache_dir` is not ABI. The core passes `directory` to the hook.
- **Contents:** `sources/<identity>/` (D5), and `refused/` only if D1's memo is built. Hub downloads go through
  `huggingface_hub`, which honours `HF_HOME`. Nothing is written into the HF cache, the repository or `plans/`.
- **Docker.**
  - The image sets `XDG_CACHE_HOME=/opt/holon/cache` for its prebuilt kernels (Dockerfile:16-20), and no `HOME`. Run
    as root, `HOME` is `/root` in the container's layer. Under `--user` with no passwd entry, `HOME` is `/`, which is
    not writable.
  - Rev 2 adds `ENV HOME=/opt/holon/user` (created `a+rwX` like the other caches).
  - The README's Docker lines mount `-v <cache dir>:/opt/holon/user/.cache/holon`, which keeps the weight digests and
    the MTP pack across containers. README's `-e HOME=/cache` recipe keeps working.
  - *Unverified:* that an `ENV HOME` overrides Docker's own `HOME` under `--user`; §9 checks it.
  - A cache that cannot be written is never fatal: the pack is then built to a temporary directory for this start
    only, with a warning.

### D4. Routing profiles: shipped as data, never fatal

- **Never fatal (std, `store.py:886-910`).** A named profile that is absent, unreadable, mismatched or malformed
  (bad JSON, wrong groups or lengths, negative counts) logs a warning, and the store falls through to the next source.
  This removes the 4 × 600 s failure.
- **Shipped (B Q6, C Q6).** The three locks' profiles ship as std data in `std/holon_std/tiered/profiles/`, tens to
  ≈ 150 KB of JSON each (the 30B's is 27.5 KB).
  - They are written by `holon_tools profile --ship FILE`, which validates an existing profile and imports it (AGENTS:
    data the tools write).
  - The 30B ships the very `profile_x4_gate.json` its lock's evidence used. The Flash-Next profiles are re-made with
    their recorded recipe (60 task prompts, 128 tokens; §20.1) where the measured files are not at hand.
- **Keyed by checkpoint hash.**
  - A shipped profile matches when its recorded files equal the store's (`store.py:878-884`) by size, and by name
    where both names are the 64-hex content hashes the hub cache uses. For a hub download that is a match by content
    hash.
  - A local copy under another name matches by size alone. That is safe because a profile is only a seed: a wrong
    seed costs speed, never output (README:305-310). No lock today binds F5b's CPU experts (A), so placement invariance
    holds for every lock.
- **Lookup order:** the lock's named file beside the checkpoint (today's setups and the bench), then the shipped
  profile, then cold. The start line names which one was used. The bench prints it, and refuses when the lock's named
  profile was not the one used (A-M8).
- **Deferred to the auto-tune library:** profiles learned while serving.
- **The tool.** `holon_tools profile` starts from the lock as `propagate` does (`__main__.py:59-83`), so its command
  needs no retyped options.

### D5. The MTP pack: found by identity, made on first use, never a refusal

**General (core `cli`).**
- **Lookup.** For each `[sources]` key of the lock that no flag gives (`mtp.path=` empty counts as given; A-M6), the
  CLI looks in `~/.cache/holon/sources/<identity>/`.
  - The identity must match `^[A-Za-z0-9._+-]+@[0-9a-f]{16}$` before it is joined into a path.
- **The hook.** If the file is missing, the CLI asks installed plugins in name order, as `claim_ref` does, through an
  optional hook (B-m3): `make_source(identity, directory) -> path | None`.
  - The hook runs in the CLI, before the supervisor, so it is outside the 600 s start timeout.
  - A per-identity `flock` stops two first starts building the same file twice.
  - The engine already compares the opened file's identity with the lock's (`plans.py:121-125`).
- **When the file is unavailable** (offline, no maker, a failed build), the server serves without speculation. The
  lock entry gets `max_draft_tokens 0` with a warning: "mtp.path unavailable (reason): speculation off, decode
  slower" (B-m6). This is never a refusal.
  - Flags that turn speculation off (`--max-draft-tokens 0`) skip the lookup entirely, so this works offline (A-M6).
  - Files a lock names by identity are drafter heads today (`ctx.weight_source`, `build.py:148-153`). If one ever
    belongs to something else, its start fails as it would today.

**The body (owner Q3): build locally, or download a published pack.**
- **Licence.** The Qwen3.8-Flash-Next model card names the **Qwen Community License 1.0** (its `LICENSE`, read
  2026-10-28). It grants the right to "use, copy, modify, merge, publish, distribute, sublicense … and create
  derivative works", on these conditions:
  - the copyright and permission notice is included in all copies or substantial portions;
  - commercial products over 100M monthly active users or US$20M monthly revenue display the model name;
  - a Model-as-a-Service or AI-work-assistant business needs a separate licence from Qwen for commercial use.

  Publishing the derived pack, with the licence beside it, is therefore permitted by its text. This is not a legal
  opinion; the owner decides.
- **(a) Publish.** The hook downloads `mtp-q2_0-q8.gguf` (≈ 0.82 GB, C-m2) from the project's Hub repository at a
  pinned revision, and checks the identity's hash. It is the simplest body, and it is a quick first start.
- **(b) Build, in `models.qwen4_exp`** (needs `gguf` and `huggingface-hub` declared in `plugins/pyproject.toml`, B):
  1. Range-read the head's 31 tensors (5.214 GB) from `Qwen/Qwen3.8-Flash-Next` at `de4b8e4d`. Each read requires
     status 206 with the matching `Content-Range` (Strata's #327 check), and each tensor its sha256. Partial downloads
     resume per tensor.
  2. Q2_0 per-block MSE search in numpy, one expert slice at a time, so the RSS stays bounded (≤ 2 GiB *est.*).
     Strata's run took 310 s of server CPU (C-m2). The result must hash to `a22206a4…`; the fixture test runs on AVX2,
     AVX-512 and arm64 (A).
  3. The Q8_0 rewrite (`tools/holon_tools/mtp_pack.py:73-107`, moved into the plugin), checked against
     `a55280a0417d4b19`.
  4. Atomic rename from a temporary directory on the cache's filesystem, which is deleted on failure along with the
     BF16 intermediates. Progress is printed: bytes fetched, then "quantizing, ≈ N min".

  A hash mismatch discards the build: "the build differs from the lock's pack", then no speculation.
- **The hook is the same for (a) and (b).**

### D5b. The launch lines' plugin options

| Option(s) in the README's launch lines | Class | After |
|---|---|---|
| `rows.place=ple:host,token_embd:host ggml_cuda.pass_rows=4096 moe.ring_lend=tail` (README:357-358) | already in the lock | retyped only because `profile` ignored the lock (D4) |
| `mtp.path=…` (README:360) | a lock source | D5 |
| `moe.placement.profile=` to serve without one (README:322-324) | workaround | obsolete (D4) |
| `rows.place=ple:ssd,token_embd:host cpu_task.io_threads=4` (README:377-380) | RAM-short mode | the spill (D1) |
| `--host-memory`, `--memory-fraction` | deployment | automatic (D1, D2) |
| speculation on the 27B, `logits_processor` on Flash-Next | opt-in features with measured trade-offs | stay flags |
| `http.*`, `state_tier.levels=ssd` and `ssd.dir`, `loop_core`, `CUDA_VISIBLE_DEVICES` | preferences, or need a user's choice | stay flags |

### D6. The auto-tune seam (B-M4)

**The seam is two things the code has after this design.**
- (a) An unset or `auto` value means "derive it": `--max-model-len auto` (`cli.py:39-42`), and now `host_memory` and
  `memory_fraction` under `adaptive`.
- (b) `serve_history`'s layering of defaults, lock and flags (`cli.py:94-125`), where a per-machine record would slot
  in between the lock and the flags.

`(name, value, why)` is the start line's format only, with no `/metrics`.

**What `tune` is.**
- `tune.tune()` (`tune.py:55-75`) is one pass over the decode ops. For each op it picks the fastest provider, keeping
  the pinned one unless beaten by a 3% margin.
- `split` (`tune.py:82-98`) is F5b's own two-stage search: threads at half the misses, then the share.
- Both run offline, once per GPU class, and their results change plans, so they reach a lock through `propagate`.
- Start-time derivation never touches a plan. The two do not share a shape and are not forced into one.

**The later library adds:**
1. Search spaces over byte-neutral deployment options per machine: `moe.cache` sizes, `ggml_cuda.pass_rows`,
   `moe.pipeline`, `max_seqs` against cache share, offload and RAM-short modes, `cpu_task.threads`, the SSD tier's
   rates.
2. A per-machine record of the winners and of learned profiles, `~/.cache/holon/deploy/<model id>/<class>/<RAM
   class>`. `holon serve` layers it between the lock and the flags (≈ +8 core lines). It may hold only P1's values, so
   it never stands in for a lock.
3. A `holon tune` command that runs the search on the user's machine.

None of this is built now.

## 4. Core impact (invariant 5; `check_core_loc.py`: 5019 / 6000 now)

| Change | Where | Code lines (*est.*) |
|---|---|---|
| host facts: meminfo, cgroup v2 and v1 via mountinfo | `platform` | 35 |
| `storage(path)`: fstype and rotational, from mountinfo | `platform` | 12 |
| `gpu_shared(device)` through NVML | `platform` | 10 |
| overshoot in `peak_memory`, after `empty_cache` | `platform` | 4 |
| the host-wide pin `flock` | `platform`, `engine.py` | 8 |
| `host_memory` / `memory_fraction` as `\| None`, `adaptive`, `__post_init__` (A-m7), the snapshot, the start lines, `TierCtx.host_left` | `engine.py` | 20 |
| the slack rule, fill pools only | `executor.py` | 6 |
| history `[defaults?, lock + spill, lock]`; the spill hooks; the lock sources and `make_source`; the `max_draft_tokens 0` fallback; the size guard; `float \| None` flags | `cli.py` | 40 |
| print each failed start; keep the derived budget across restarts | `supervisor.py`, `ipc.py` | 8 |
| `OutOfUnits` exported; `Spills`, `MakesSources` protocols; `TierCtx.host_left` | `holon.api` | 10 |
| **total** | | **≈ 150 → ≈ 5170 / 6000** |

- `memory.py` is unchanged apart from importing `OutOfUnits` from the API. There is no XDG and no `cache_dir` in the
  ABI.
- **The ABI gains** `OutOfUnits`, `Spills`, `MakesSources` and `TierCtx.host_left`. All are additions, so the bump is
  minor (B-m1).
- **Nothing in core names a model.** Spill options are opaque, identities opaque, storage and sharing machine facts.
- **Toy proofs in `core/tests`:**
  - a model plugin with `spill` and `make_source`;
  - a tier sized by `host_left`;
  - the restart keeping the budget.
- **Outside core** (*est.*):
  - std, about +50: the tier, the store's never-fatal lookup and shipped data;
  - `models.qwen4_exp`, about +30 for (a) or about +200 for (b);
  - tools: `profile` from the lock, and `--ship`;
  - the Dockerfile, +2.

## 5. Exactness and locks

- **No plan id changes from automatic values.** `host_memory`, `memory_fraction` and `adaptive` are not planned. The
  tier keeps its planned `host.bytes`, and its options are outside `state_identity` (`plans.py:150-160`). Source files
  are planned by identity.
- **The two plan-changing degradations are logged as differences from the lock:**
  - the spill: byte-identical (`v1-flash-next.md` §22.1);
  - speculation off: drafts never change a greedy output, since the verify decides.
- **Outputs.**
  - Fixed host pools: the same bytes.
  - The expert cache under D2: exact by placement invariance (README:305-310).
  - The shipped profile: a seed, so exact.
  - A shrunk tier moves state without changing it. Where the 27B locks' `branch,answer` checkpoints decide what is
    recomputed, outputs stay within the KL bounds those checkpoints were accepted under (README:399-402).
- **In-process engines are strict (P2).** The shrink, the spill and the fallback cannot occur in propagate, tune or the
  judge. Their Flash-Next, tier and offload runs still pass `--host-memory` as today.
- **D2 moves no trace.** `propagate` records snapshots on the stored `pool_bytes` of their class
  (`propagate.py:150-176`), and the newest lock of all 25 classes stores it (checked, rev 1).
- **The propagation plan (A-m8).** The change edits the core, whose source every lock pins (`plans.py:114`), and
  std's tier and store, and `models.qwen4_exp`. It therefore outdates **all 25 lock classes**.
  - In the same round as the build, before merge, `propagate` runs on each class's card: about 25 runs over the four
    cards at 2–3 snapshot runs each, sharded as in `release.md` §22.
  - The expected verdict everywhere is "advance, byte-identical trace", since no planned value changes. Any other
    verdict stops the merge.
  - The new locks go in with the code, so `main` never serves defaults for an outdated lock.

## 6. Build steps

1. Core: P2, D1's snapshot and restart carry, D2, the `flock`, the facts, with CPU tests over fake `/proc` and `/sys`
   trees and NVML stubs.
2. std: the tier's `host_left` and `OutOfUnits`, and the never-fatal profile lookup with shipped data. The history and
   hooks in `cli`.
3. D5's hook, its body per Q3, and the Dockerfile `HOME`.
4. Propagate all 25 classes (§5); README and `tools/README.md` (§8).
5. §9 on the bar cards.

One reviewer for the aggregate, with the panel re-reviewing this rev.

## 7. What a user sees

After the lock line, one line per derived value:

```text
holon: host memory: budget <B> GiB = available <A> − kept free 6.0 (max 6 GiB, 5% of <T> GiB; cgroup: none); pinned <P> GiB; --host-memory caps
holon: GPU memory: state pools <S> GiB = free <F> − step peak <K> − slack <L> (allocator <O> + reserve 0.25, headless); experts cached 48.6%
holon: mtp.path: ~/.cache/holon/sources/mtp-q2_0-q8@a55280a0417d4b19/ (made on first use)
holon: routing profile: shipped (Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S…profile.json)
```

When RAM is short, with Flash-Next on a local SSD:

```text
holon: start failed (memory): host memory <B> GiB too small: plugin.tiered.pinned.… needs <N> GiB, <L> GiB left; plugin.tiered.rows.ple holds 26.8 GiB
holon: RAM short by <S> GiB: retrying with rows.place=ple:ssd,token_embd:host cpu_task.io_threads=4 — PLE table read from the SSD, byte-identical; prefill −16/−26% at 1K/4K, MTP decode −6/−7% (RTX 5090); free <S> GiB or pass --host-memory to keep it in RAM
```

**Overrides:**
- `--host-memory BYTES`;
- `--memory-fraction F`;
- `--plugin-options mtp.path=FILE` (or `=`: none), `moe.placement.profile=NAME`, `state_tier.host.bytes=…`,
  `rows.place=…` (a flag on a spill's key drops that spill);
- `--max-draft-tokens 0`.

## 8. The README afterwards

- **"Launch flags" is deleted** (README:61-74).
- **The model table's caption** becomes "start any of them with `holon serve <name>`".
- **The Quickstart** is `uv run holon serve unsloth/Qwen3.8-27B-GGUF`.
- **The Flash-Next section** is one line, noting the first start's pack. Strata's commands move to `tools/README.md`.
- **Docker** gains the cache volume (D3).
- **Known limitations:**
  - the "file a lock names … fails slowly" item goes;
  - "pins ≈ 61 / 74.5 GiB for the server's lifetime" stays (C-m6), with "ZFS's ARC is not counted as available:
    pass `--host-memory`" (A-m4) and the oomd note (A-m3).
- **A Host RAM table.** Its figures are *est.* from C §1.2 and the 16.35 GiB measured above, filled in by §9:

| Lock | Pinned | RAM for full speed | With less RAM |
|---|---|---|---|
| Qwen3.8-27B, RTX 5080 and 5090 | 16 GiB KV tier | 32 GB with ≤ 6 GiB other use | the tier shrinks (agent_park cost: §9) |
| Qwen3-30B-A3B Q4_K_M, RTX 5080 | experts, ≈ 16.4 GiB | 32 GB with ≤ ≈ 8 GiB other use; 48 GB comfortable | refused once, naming `--host-memory` |
| Qwen3.8-Flash-Next IQ2_XS, RTX 5080 | experts, PLE, embeddings, ≈ 61 GiB | ≈ 80 GB | on a local SSD: the PLE table from the SSD (cost: §9); else refused |
| Qwen3.8-Flash-Next IQ3_S, RTX 5090 | ≈ 74.5 GiB | 96 GB with ≤ ≈ 10 GiB other use; 128 GB comfortable | the same |

## 9. Test plan and acceptance

**CPU** (`pytest`, lean):
- host facts over fake trees: v2 with a limited ancestor, `memory.high`, active file pages, a container root under
  `cgroupns=host`, v1 `hierarchical_memory_limit`, no cgroup;
- `storage` over fake mountinfo and sysfs;
- `gpu_shared` stubbed;
- the slack rule, with and without fill pools;
- history building: no defaults when the checkpoint exceeds VRAM; a spill dropped by a flag;
- the supervisor: refuse once, every failure printed, the budget kept across a restart (toy plugins);
- D5: the identity check, offline fallback to no speculation, `--max-draft-tokens 0` without network;
- the store: missing, malformed and mismatched profiles fall through, shipped matching by hash and by size;
- for (b): the pack's fixture test on AVX2, AVX-512 and arm64.

**GPU acceptance.** Bar cards, one GPU each, a fresh `~/.cache/holon`, the HF cache holding only the model files:
1. **Bare start.** Every README lock starts with only `holon serve <model>`: no `differs from the lock` line, the plan
   id equal to the lock's, and Flash-Next's pack made and timed.
2. **Speed (C-M5, C-M6).** `bench/offload_ctx.py` on both Flash-Next cards and the 5080's 30B, in 4 ABBA rounds on
   the 1K rows and 2 elsewhere, with the arms:
   - (a) hand-set flags with the lock's profile beside the checkpoint;
   - (b) automatic with the same profile beside it, which judges D1 and D2 alone;
   - (c) automatic with the shipped profile and none beside it, against (b);
   - (d) automatic and cold, the first 8 prompts, reported only.

   **The ship rule of `v1-flash-next.md` §23.7:** no row of (b)/(a) or (c)/(b) below 1.0 by more than 2 SE, and
   experts cached (`bench/ledger.py`) ≥ the hand-set share − 0.1 point. One more arm runs (b) on a display-driving
   5080 with the 1 GiB reserve, reported.
3. **RAM-short, in desktop-sized scopes with swap.** The scopes are `systemd-run --user --scope -p MemoryMax=32G | 64G
   | 96G -p MemorySwapMax=…`. `SwapTotal > 0` is checked first; without swap, the test is reported as not run. Each
   run is judged on the scope's own counters:
   - `memory.swap.current` 0;
   - `memory.stat` `pgmajfault` flat;
   - `memory.events` `high`, `max` and `oom_kill` 0;
   - the scope's `memory.pressure`.

   "Need" is the measured peak `memory.current` of an unlimited run. The cases:
   - 27B tier lock, need − 8 GiB: the tier shrinks to ≈ 8 GiB and serves;
   - Flash-Next 5080 in 64G: a spill, its doomed build timed (D1's memo if > 60 s);
   - Flash-Next 5090 in 64G: a spill or a refusal, as C §1.2 predicts;
   - the 30B offload lock in 32G, with a 6 GiB working set in a sibling scope: starts or refuses as D1 states;
   - Flash-Next in need − 40 GiB: refused once within ≤ 3 min, naming the shortfall;
   - a checkpoint on a network filesystem: the spill is not offered.
4. **Lag (C-M7).** A probe in a sibling scope times a 1 ms sleep and touches its 512 MiB working set, through each
   start of (3) and 10 minutes of the agent trace. Its p99 must be ≤ 20 ms. A real 32 GB desktop repeats the 30B case
   if one is available.
5. **Degraded speed (C-M8).**
   - One `offload_ctx` round of the spilled Flash-Next on both cards: the 5080's cost is unmeasured.
   - `agent_park` on the 27B tier at the shrunk size.
   - The numbers go into the README's "With less RAM" column.
6. **VRAM soak.** The three offload locks at automatic, through the agent trace and the 30K-token prompt, end with no
   CUDA OOM. On the display-driving 5080, a hardware-accelerated browser is opened during the soak (A-M5).
7. **Locks.** The §5 propagation gives a byte-identical trace on all 25 classes.
8. **Docker.** `--user` with the cache volume keeps the pack across two containers (D3).

## 10. Questions for the owner

1. **Headroom.** `max(6 GiB, 5%)` refuses the 5080's 30B offload lock on a busy 32 GB desktop, where today it starts
   and may swap. Accept that, or lower the floor?
2. **Reserve.** 1 GiB on a shared card, at ≈ 2 points of experts on a display-driving 5080: accept?
3. **MTP pack: (a) publish on the Hub, or (b) build locally?**
   - The licence permits redistribution with its notice (D5).
   - (a) costs a ≈ 0.82 GB download and needs the project's Hub account.
   - (b) costs ≈ 5 GB of range reads, ≈ 5+ min of CPU, and adapting Strata's MIT quantizer under provenance, outside
     the "kernels only" rule.
4. **The doomed build memo** (D1): build it only if §9 times the doomed build over 60 s, or always?

## 11. Findings and answers

| Finding | Answer |
|---|---|
| A-B1 refusal rolls back into defaults that cannot hold the model; tier not "memory"; spill not persisted, flag precedence, merging, `--no-supervised` | **Fixed.** No defaults entry when the checkpoint exceeds VRAM; refuse once with the last reason; every failure printed (D1). The tier raises `OutOfUnits` from the API. The spill is a history entry: it persists, flags win in `cli`, and each plugin's hook merges its own options. `--no-supervised` serves the newest entry only, as today, and gets no spill or fallback (stated) |
| A-M1 active page cache counted as used | **Fixed:** both file lists are reclaimable (D1) |
| A-M2 cgroup path in containers; v1 | **Fixed:** mountinfo resolution, the mount root's own files as fallback, v1 `hierarchical_memory_limit` |
| A-M3 concurrent starts race | **Fixed:** host-wide `flock` from snapshot to pinned |
| A-M4 tier sized, then re-checked live | **Fixed:** one snapshot; the tier sizes against `host_left()`; below the floor `OutOfUnits` |
| A-M5 thin slack on shared GPUs | **Changed:** D2 only with fill pools; reserve 1 GiB when shared (NVML); browser in the soak |
| A-M6 `--max-draft-tokens 0` offline | **Fixed:** skipped lookup; unavailable source → no speculation, never a refusal |
| A-M7 size guard lost for unlocked files | **Fixed:** RAM ignored unless the reference has a lock for this card |
| A-M8 degradation in lock-writing and in-process paths; learned profile writes | **Fixed:** `adaptive` set only by `holon serve`; learning dropped; profile used is reported, bench refuses a substituted one |
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
| C-M3 spill onto a slow or network FS | **Fixed:** `storage == ssd` only; a network-FS case in §9.3 |
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
