# v1 — Fail-stop: three follow-ups of plan §10 as one small milestone

Status: **built and cleared by A (2026-10-31); released with RAM-adapt (`docs/dev/v1-glm-ram.md`), one lock round for
both (2026-11-01).**

Three items from `docs/plan.md` §10:
1. `cpu_task@1` fail-stop by construction (A's m1).
2. A start-refused exclusive-tier engine leaks its store.
3. The four ac-oom minors, plus the executor's always-zero memory-note term (A's C-m5).

No contract is added. One existing contract's caller duty becomes the provider's.

## F1. `cpu_task@1` stops the engine itself

**Today.** A wait gives up after its time bound (`cpu_task.timeout`), or a task fails. Either one ORs a bit into the
instance's `errors` word, and the worker fails open: every later wait returns at once, so no graph hangs. Stopping the
engine is left to each consumer, which must zero the word every step and check it at the commit:
- the exclusive store watches it (`store.watch(self.io.errors)`, added in `dea23094` after silent wrong bytes);
- `std.moe.cpu` watches it;
- `std.tiered.rows`' SSD reads have their own `_Errors` shared object.

A consumer that forgets serves wrong bytes. The word is also a static per child namespace, so an auxiliary model's
instance has a word of its own. The store watches its own instance's word, not that one.

**Change.** The word belongs to the engine's worker:
- **`kernels.cpu_task`:** `Service`, the `cpu_task.<pool>` shared object, owns one word per pool and engine, and
  every instance's `errors` is that word.
- **Step hooks:** `Service` zeroes the word at each launch (`before_launch`, as `_Errors` does), reports it as
  telemetry, and raises at the commit when it is set (`after_commit`). The raise names the pool and the bits (1 a
  task failed, 2 a wait timed out).
- **The device side is unchanged:** the waits stay fail-open, so no graph hangs, and only the commit stops the engine.
- **Watches removed:** `std.tiered.store` drops `watch(self.io.errors)`, `std.moe.cpu` drops its watch, and
  `std.tiered.rows`' `_Errors` keeps only its descriptor's close. `store.watch` stays for `linear_experts_arena@1`'s
  word.
- **Contract text** (`std.moe.kernels`): `errors` is "zeroed by the provider each step and checked at the step's
  commit, which stops the engine; consumers may read it".

**Test.** Exclusive tiers with `cpu_task.timeout` 1e-6, a bound no read meets. The store no longer watches the word,
yet the commit refuses with the worker's message. This replaces the store-specific match in
`test_an_ssd_read_past_its_time_bound_stops_the_engine`. The CPU-share test's dead-worker check now expects the
worker's message.

## F2. A refused start releases what it built

**Measured on an RTX A5000.** An exclusive-tier engine refused at its start (`moe.lend_peak = require` without a tail
loan) kept 166 MiB of VRAM, its 108 MiB of pinned pools and the `moe.move` thread.
- The move thread (made in `after_materialize`) holds the store, which holds the build context and the rest.
- Starting that thread at the first move frees the VRAM but not the pinned pools: something else keeps them past
  `gc.collect()`.
- Shutting the shared objects down and freeing the host pools, as `close` does, returns both.

**Change, in core.** `Engine.__init__` runs the build (`_build`, today's body unchanged). If the build raises,
`_abandon` releases what it built before re-raising:
- the device synchronized;
- the processor's and the shared objects' `shutdown` (each error suppressed, so the refusal's own reason is the one
  raised);
- the tier closed;
- `ledger.free(host=True)`.

This covers every plugin with a helper (the cpu_task workers, the grammar processor's), not only the store. The
lazy move thread is not needed.

**Test.** Refuse an exclusive-tier start in-process. Then the VRAM the caching allocator holds and the live pinned
regions (`platform._Pinned`, counted) return to their level before the start, and no `moe.move` thread remains.
The `cpu_task.io` worker's 832-byte word pool stays after a normal `close` too; it is checked separately, and
recorded if it is by design (a coordinator left behind).

## F3. The ac-oom minors and C-m5

| # | Change | Where |
|---|---|---|
| 1 | A refused start prints its one-line reason only. The engine process sends `failed`, logs the traceback at debug (`logging.getLogger("holon")`) and exits with status 1 (`SystemExit`, so `multiprocessing` prints no traceback). A failing step keeps its traceback | `ipc.serve` |
| 2 | Ctrl-C ends `holon serve` with no traceback, exit status 130. The engine is still stopped by the existing `finally` blocks | `cli.main` |
| 3 | `files.read`'s bounce windows are `MAP_PRIVATE`, so their pages read as anon, not shmem | `std.tiered.files` |
| 4 | A SIGKILL at startup is a memory failure only if the cgroup's `memory.events` `oom_kill` count rose during the start. A kill with no OOM kill counted, or where the count cannot be read (cgroup v1, no access), is a plain failure: "killed (SIGKILL) while starting, not by the OOM killer" (or "no OOM count readable") | `platform.oom_kills()`, `ipc.EngineProcess.start` |
| 5 | The GPU-memory start line drops "allocator 0.00 +" (always zero). The overshoot stays in the slack, where it is 0 too, so nothing moves | `executor.slack` |

## Core lines (`check_core_loc.py`: 5,230 / 6,000 before)

| Change | Module | Lines |
|---|---|---|
| F2 `_build` / `_abandon` | `engine` | +13 |
| F3.1 reason only | `ipc` | +2 |
| F3.2 Ctrl-C | `cli` | +3 |
| F3.4 `oom_kills()` and its use | `platform`, `ipc` | +9 |
| F3.5 the note | `executor` | 0 |

About +27, to ≈ 5,257. F1 is plugin and std only. No ABI change: `cpu_task@1`'s `errors` keeps its type and name.

## Locks

The core change outdates all 27 locks by their `holon` pin. Of the other units:
- `std.tiered` is pinned by 5 histories: Flash-Next ×2, GLM ×2 and the RTX 5080 30B offload.
- `std.moe` is pinned by 11.
- No lock pins `kernels.cpu_task`. No lock runs exclusive tiers or a CPU share: those are the spills and opt-ins.

No kernel, plan or output changes, so every snapshot trace should come back byte-identical. The combined round after
GLM's RAM-adapt task propagates them; this branch propagates nothing.

## Tests and evidence

- New: the F1 test, the F2 test (GPU), and an F3.4 test with a stubbed `memory.events` in the existing SIGKILL test.
  Existing tests updated: the two match patterns above, and F3.3's window test checks the map's flags.
- The default suite runs on an RTX A5000, then again with `CUDA_VISIBLE_DEVICES=` (CPU CI's terms). Scratch goes to
  `local/tmp/fail-stop/`.

## As built (the panel's notes, A and B on 2df368a3)

**F1:**
- **Contract:** `cpu_task@1` stays `@1`, its text rewritten in place (B's option b; no lock pins it). The stop at
  the commit is the provider's duty. One word per worker kind and engine is shared by auxiliary models' instances.
  Consumers may read the word, never need to watch it, and cannot recover from a bit. Both providers meet the duty:
  `kernels.cpu_task` raises at the commit, and the oracle `std.moe.reference.cpu_task` runs each task synchronously.
- **Sticky word (A):** the word is zeroed once, at the first step served (the first `before_launch`, after the
  start's profile and graph warm-ups), and never again. A bit a side stream sets after a step's snapshot is caught at
  a later commit, never zeroed away.
- **Attribution (A, optional) is not built.** The message names the worker and the bits; naming the failing slot
  needs the native worker to record it.

**F2:**
- **One release path (A, B):** `Engine._release()` serves both `close` (after its drain) and a start refused
  part-built (`__init__` runs `_build`, then `_release` on any exception). It:
  - synchronizes the device;
  - shuts down the processor and the shared objects, then closes the tier, each error kept and the rest still run;
  - drops the executor, the processor, the tier and the shared objects;
  - frees the pools, collects, and releases the allocator's cache.
  `ctx`, `processor`, `tier` and `executor` have class-level `None` defaults.
- **The leak's holders:**
  - The move thread holds the store, and through it the build.
  - A second holder kept the pinned pools past `gc.collect()`. `platform.device_alias` made a class per call, and
    that class kept the last alias's host memory alive until the next one was made. That holder was also behind
    `cpu_task.io`'s 832-byte word pool, which outlived a normal `close`. The alias's interface is now a
    `SimpleNamespace`, and both are fixed: after a refused start and after a `close`, no pinned region stays alive.
- **Bounds on the release path (A):** every shutdown on it ends without waiting on another party:
  - the `cpu_task` workers at `cpu_task.timeout`; a coordinator past it is left behind, as at `close`;
  - the move thread once its submitted copies are done;
  - the grammar watchdog's join at 5 s;
  - the state tier's I/O threads once their queued reads and writes end.
  No new bound is added.
- **Test:** it measures after the refusal's exception is discarded. No pinned region stays alive, VRAM is no higher
  than before the start, and no `moe.move` thread remains.

**F3:**
- **F3.1:** no logger. A refusal ("memory", "plugin") prints its reason only (the engine exits with status 1); an
  "error" keeps its traceback on stderr.
- **F3.4:**
  - The count is read from the engine's own cgroup and below: v2's `memory.events`, or v1's
    `memory.oom_control`, under the v1 memory mount.
  - It shares `_cgroup()` with `host_ram`.
  - When the count cannot be read, the kind stays "memory".
- **F3.5:** `peak_memory`'s overshoot is removed. `grown - max(allocated peak, grown)` is 0 by arithmetic, so the
  measurement went too: the `out` dict, the slack's term and the note's "allocator" term. The slack is the reserve,
  at most the 10%, and nothing moves.

**Core lines:** the build is +36 (5,230 → 5,266), with A's code-review minors:

| Module | Lines | What |
|---|---|---|
| `engine` | +12 | the shared `_release` |
| `platform` | +14 | `oom_kills` and `_cgroup`, less the overshoot |
| `ipc` | +6 | |
| `cli` | +5 | |
| `executor` | −1 | |
