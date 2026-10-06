# Spike S6 — CPU handshake inside CUDA graphs

## Question

Can a FULL CUDA graph hand work to a persistent native CPU worker and wait for the result without
`cudaLaunchHostFunc` (design §7.4, §6.5, plan §3.1 S6, D8)? Specifically:

1. Does stream capture (torch 2.11 / CUDA 13.0 / driver 580) accept the stream-memop flag pair
   (`cuStreamWriteValue32` + `cuStreamWaitValue32`) and the kernel-flag alternative?
2. What does one handshake cost, and what does a step with N = 24/48/75 handshakes cost, compared with `cudaLaunchHostFunc`?
3. Can independent GPU work overlap the CPU work, on the same stream or on a branch?
4. Does it hold up over at least 10k replays, with a slow worker, on shutdown and abort? Can a double-buffered residency map be swapped safely (`quiesce`)?
5. What needs to be re-measured on the consumer desktop (RTX 5080 node)?

## Setup

| | |
|---|---|
| host | the A5000 server: 2× AMD <cpu> (<cpu>, AVX2, no AVX-512), 251 GiB, shared (load avg ≈ 22 during the run) |
| GPU | RTX A5000 (sm_86), `CUDA_VISIBLE_DEVICES=7`, driver <version>, PCIe, NUMA node 1 |
| software | torch 2.11.0+cu130, nvcc 13.0.48, Python 3.12.12, g++ 11.4, Linux <version> |
| CPU worker | 8 threads pinned to cores 24–31 (GPU-local NUMA node, one per physical core); Python main thread kept off them |
| memop caps | 64-bit memops 1, wait NOR 1, flush-remote-writes 0, host-native-atomics 0 |

```bash
cd spikes/s6_cpu_handshake
CUDA_VISIBLE_DEVICES=7 ./build.sh                       # optional; bench.py builds on first use (~20 s)
CUDA_VISIBLE_DEVICES=7 python bench.py --out results/a5000_$(hostname -s).json   # ~9 min
#   --threads 8 --cores 24-31 --quick --only capture,latency,...   (see --help)
```

Files: `bench.py` (the whole benchmark, one entry point), `csrc/worker.cpp` (worker, pool, handshake ops,
pybind), `csrc/kernels.cu` (flag/spin/busy kernels), `build.sh`, `results/a5000.{json,log}` (raw data).

## Design of the handshake

**Protocol.** Each slot (one per MoE layer) has one 32-bit word on its own 64-byte line of mapped pinned
memory (`cudaHostAllocMapped`):

```
GPU (graph):  write word := REQ(1)        ...independent GPU work...        wait word >= DONE(2)
CPU (coord):  sees REQ (acquire) -> runs slot task on pool -> store DONE (release)
```

The GPU writes REQ only after it has seen DONE for that slot on the previous replay, and the CPU writes DONE only
after it has seen REQ. So a single word needs no separate reset write, and both immediates are constants that can be
baked into a graph. FreeToken does it differently: it uses two words per slot, `done := 0`, `ready := 1`, then waits on `done`, so each
handshake costs one more GPU op.

**Data path.** The expert input `x` and output `y` live in per-slot pinned buffers. Two variants:
- `copy`: memcpy nodes D2H `x` before the signal and H2D `y` after the wait.
- `zc` (zero-copy): the producing kernel writes `x` straight into mapped memory through a CUDA alias
  (`device_view` = `cudaHostGetDevicePointer` + `from_blob`), and the consumer kernel reads `y` over PCIe.

Ordering: a default `cuStreamWriteValue` carries a memory barrier, and the kernel flavour uses a
`thread_scope_system` release store. The coordinator's acquire load of REQ makes `x` visible to the pool. The pool
workers' `fetch_sub(release)` → coordinator `acquire` → `store(DONE, release)` makes `y` visible before DONE.

**GPU-side flavours** (all of them are captured by `torch.cuda.graph` unchanged):

| mode | signal | wait | graph nodes |
|---|---|---|---|
| `memop` | `cuStreamWriteValue32` | `cuStreamWaitValue32(GEQ)`: the GPU front-end polls, no SM is occupied | 2 `batch_mem_op` |
| `kernel` | 1-thread kernel, system-scope release store | 1-thread kernel, acquire spin + `__nanosleep(64)` | 2 kernels |
| `hybrid` | memop write | spin kernel | 1 memop + 1 kernel |
| `hostfunc` | `cudaLaunchHostFunc` runs the task inline on the driver callback thread (1 per layer, which is the best case; FreeToken used 2) | — | 1 `host` |

**Worker.** A native coordinator thread busy-polls all slot words with `_mm_pause`. It stays hot for 100 ms after
the last request, then dozes in 50 µs sleeps. It runs the task on a persistent spinning pool and computes chunk 0 itself.
No Python and no GIL are involved. The task is a bf16 GEMV `y[768] = W[768,2048] x[2048]` (3 MiB of weights per layer,
75 layers = 225 MiB, so it sweeps DRAM), written with AVX2 intrinsics. The driver memop entry points are resolved
with `cudaGetDriverEntryPointByVersion`, so the extension does not link libcuda.

**Control.** `abort()` switches the worker to fail-open: it answers every REQ with DONE and skips the compute, so
queued replays drain. `stop()` joins the threads and returns the number of slots still in REQ ("stranded"). That
number must be 0, because a stranded wait never completes. `set_delay` / `set_spin` inject slow or fixed-length work.

## Results (A5000 host; raw data in `results/a5000.json`)

### Q1 — capture

All four flavours capture and replay correctly. A 2-layer graph with on-device verification
(`max|y − W·x|` ≤ 4.8e-7 over 50 replays) contains:
memop `batch_mem_op: 4, kernel: 29`; kernel `kernel: 33`; hybrid `batch_mem_op: 2, kernel: 31`;
hostfunc `host: 2, kernel: 29`. Each memop becomes its own graph node.
Gotcha: the runtime `cudaGraphNodeGetType` returns `cudaErrorUnknown` on batch-memop nodes, because the CUDA 13.0
runtime enum has no value 0x0c. Introspection must use the driver's `cuGraphNodeGetType`.

### Q2 — handshake cost (empty CPU task)

Overhead is the step p50 minus the same graph without handshakes, divided by N. 1000 replays each; p99 is in the JSON.

| data path | memop | kernel | hybrid | hostfunc |
|---|---|---|---|---|
| none (flag only), slope over N | 4.0 µs | 3.2 µs | 4.7 µs | 41 µs |
| copy (2 memcpy nodes), slope | 2.4 µs | 13.0 µs | 9.9 µs | 31 µs |
| zc (zero-copy), slope | 5.4 µs | **3.0 µs** | 5.2 µs | 40 µs |

Per-step overhead at p50 in µs (zc path):

| N | memop | kernel | hybrid | hostfunc |
|---|---|---|---|---|
| 24 | 129 | 71 | 125 | 924 |
| 48 | 257 | 140 | 250 | 1895 |
| 75 | 402 | 222 | 390 | 3011 |

The data path matters as much as the flag. Adding the two tiny memcpy nodes costs ≈ 14 µs per layer even without any
handshake (GPU-only graph at N=75: 1380 µs with copies vs 348 µs with no data). Zero-copy costs ≈ 2 µs per layer
(492 µs). The memop flavour hides some of the copy latency (2.4 µs), while kernel waits followed by memcpy nodes are slow (13 µs).

**The memop wait gets slower to notice DONE the longer the CPU takes** (`wakeup` experiment: the CPU task is a
pure spin of S µs, N=24, zc):

| S (µs) | 0 | 2 | 5 | 10 | 20 | 50 | 100 | 200 |
|---|---|---|---|---|---|---|---|---|
| memop µs/handshake | 4.8 | 5.2 | 10.4 | 5.4 | 5.1 | 11.5 | 9.0 | 11.0 |
| kernel | 3.1 | 2.2 | 2.2 | 2.3 | 2.5 | 2.3 | 2.5 | 2.9 |
| hybrid | 4.6 | 3.9 | 3.3 | 3.3 | 3.3 | 3.4 | 3.7 | 3.8 |
| hostfunc | 35 | 36 | 34 | 39 | 39 | 23 | 30 | 38 |

A spin kernel's wake-up is flat at 2–3 µs. A failed front-end semaphore acquire seems to be retried on a coarser
schedule (≈ 5–10 µs, non-monotonic). That is a hypothesis about driver/front-end behaviour; we did not verify it.

**With the real CPU task** (N=75, bf16 GEMV per layer, 8 threads). The overhead is computed from the mean step minus the GPU-only
step minus the CPU compute time measured inside the worker:

| data | mode | step mean | CPU compute | overhead/handshake | overhead/step |
|---|---|---|---|---|---|
| zc | memop | 4986 µs | 3763 µs | 9.7 µs | 725 µs |
| zc | kernel | 4336 µs | 3628 µs | **2.8 µs** | 210 µs |
| zc | hybrid | 4420 µs | 3634 µs | 3.9 µs | 289 µs |
| zc | hostfunc | 6245 µs (p99 13.6 ms) | 3368 µs | 31.7 µs | 2379 µs |
| copy | memop | 5675 µs | 3660 µs | 8.3 µs | 622 µs |

CPU task standalone (`cpu_task`, 225 MiB sweep): 135 / 75 / 57 / 44 / 37 µs for 1 / 2 / 4 / 8 / 16 threads, which is
23 → 86 GB/s. In-graph per-layer compute is noisier, 41–64 µs, on this shared host.

### Q3 — overlap

N=24, zc, with a fixed GPU busy kernel of 63.8 µs per layer, which is longer than the CPU task (35–49 µs in-graph).
"Ideal overlapped" is the GPU-only time (1714 µs) plus the handshakes.

| mode | placement | step mean | |
|---|---|---|---|
| none | GPU work only | 1714 µs | reference |
| memop | serial (GPU work after the wait) | 2923 µs | no overlap |
| memop | same stream, between signal and wait | 1903 µs | CPU fully hidden |
| memop | forked branch | 1966 µs | hidden |
| kernel | same stream | 1909 µs | hidden |
| kernel | branch | **1826 µs** | hidden |
| hostfunc | serial | 3771 µs | no overlap |
| hostfunc | branch forked *before* the host node | 2209 µs | overlaps, but ≈ 20 µs per layer of host-node cost remains |

Yes: GPU work enqueued between signal and wait, or on a branch, runs while the CPU computes.
A separate stream outside the graph is not blocked by a graph parked on a wait: while a replay waited 300 ms on a stalled worker, a 1 ms
kernel on another stream finished in 1.05 ms.
Note: for a branch to overlap a host node, the fork must come before the host node in stream order.

### Q4 — robustness

All long runs were StepLoop-like: at most 2 replays in flight, and the host waits on the event of step i−2.
Each replay verified every layer on the device against a GPU GEMV.

| run | replays | handshakes (expected) | max abs err | outcome |
|---|---|---|---|---|
| memop / zc | 20 000 | 1 500 000 (1 500 000) | 7.2e-7 | no deadlock, no lost or duplicated handshake |
| memop / copy | 10 000 | 750 000 (750 000) | 7.2e-7 | ok |
| kernel / zc | 10 000 | 750 000 (750 000) | 6.0e-7 | ok |
| hostfunc / zc | 5 000 | 375 000 (375 000) | 6.0e-7 | ok |

- **Slow worker** (2 ms sleep on every 97th task, 300 replays): steps stretch (p50 9.6 ms, p99 21 ms) and results stay
  exact. Nothing times out.
- **2 s stall** of a single task: the replay simply takes 2.01 s. No CUDA error and no watchdog (this is a compute GPU on Linux;
  a display GPU or Windows/WDDM may behave differently).
- **Abort with 20 replays queued** (each would take 75 × 20 ms ≈ 1.5 s): after `abort()` the stream drained in 13 ms,
  with 334 tasks answered fail-open and 0 stranded. Outputs of those replays are garbage and must be discarded.
- **Restart** of the same worker object after the abort: the already-captured graph keeps working, exact over 200 replays.
  `stop()` after a clean drain: 0 stranded.
- **Not survivable:** stopping the worker (not aborting it) while a replay is queued. The next REQ is never answered,
  and the stream, and with it the process, hangs forever. Teardown must be quiesce → stop, or abort → drain → stop.
  As far as we can read it, FreeToken's teardown writes `done = INT64_MAX` once. Later layers of the same replay reset
  `done := 0` and would hang. Our fail-open answers every request until the threads are joined.

### Q4 — quiesce and the double-buffered residency map

Test setup: an 8-layer graph (memop + 50 µs of GPU work per layer). It reads the active map `maps[sel]`
(65 536 int64 entries) at the first and the last layer. A violation is a torn map or a version change inside one step.
3000 steps, swapped every 10 steps:

| protocol | what the host does at a swap | drains | violations | step |
|---|---|---|---|---|
| `quiesce` | drain in-flight replays (sync), write the inactive buffer, flip `sel` | 300 | **0** | 0.68 ms |
| `ordered` | no drain: H2D into the inactive buffer and `sel.fill_` **on the replay stream** between replays; the host staging buffer is guarded by an event | 0 | **0** | 0.56 ms |
| `unsafe` (negative control) | write the *active* buffer from another stream while replays are queued | 0 | 321 | 0.59 ms |

Sketch, matching design §7.3:

```python
def quiesce(self, cb):                     # StepLoop
    while self.inflight: self.commit(self.inflight.popleft())   # waits on each step's event
    cb()                                   # nothing is in flight: mutate freely
# Device-resident resolver maps do not need quiesce. Both the update and the flip are stream-ordered work
# enqueued between replays:
def swap_map(self, new_host):              # DeviceResolver, called between launch_forward calls
    b = 1 - self.active
    self.staged[b].synchronize()           # staging buffer b free again
    self.staging[b].copy_(new_host); self.maps[b].copy_(self.staging[b], non_blocking=True)
    self.staged[b].record(); self.sel.fill_(b); self.active = b
```

`ordered` is free. Each `quiesce` drain cost ≈ 1.2 ms here, about two steps of lost pipelining. Use
quiesce only for mutations that are not stream-ordered: host- or worker-side state (expert weight sets, a
pinned map the CPU reads), VRAM expert slots refilled by a side-stream copy, worker restart, and binding swaps.
The CPU worker itself does not need the map: the graph passes the miss list (`ids → -1`) in the slot's payload.

## Decision

**The seam is sound for the core ABI.** Stream capture accepts both flag flavours on torch 2.11 / CUDA 13 /
driver 580. A persistent native worker answers 1.5 M in-graph handshakes with no loss, overlaps GPU work, and can
be aborted, drained, and restarted under a live graph. The cost is 2.8–9.7 µs per handshake, against 31–41 µs
for `cudaLaunchHostFunc`. At 75 layers that is 0.2–0.7 ms per step instead of 2.4–3.0 ms.

Changes this spike asks for in the design:
1. `Platform` should expose a **flag-pair primitive** (`signal(slot)`, `wait(slot)`) with two backends: memop
   (front-end, no SM) and spin-kernel (1 thread, flat ~2–3 µs wake-up). §7.4 currently names `cuStreamWaitValue`
   only, but on this host the kernel wait is 3.5× cheaper under real CPU work. Choose the backend per fingerprint class as a
   judged tunable. Keep memop available for laptops, where a resident spinner can cause power-capping (FreeToken's report).
2. The expert payload must be **zero-copy** (kernels write `x` and read `y` in mapped pinned memory). Two memcpy nodes
   per layer cost ≈ 14 µs, more than the handshake itself.
3. Make the single-word REQ/DONE protocol, fail-open `abort()`, and the rule "stop only after drain" part of the contract.
   Swap device-resident maps stream-ordered. `quiesce` is for state that is not stream-ordered.

## Open issues and what to re-measure on the RTX 5080 desktop

Run the same command inside the persistent container:
`CUDA_VISIBLE_DEVICES=0 python bench.py --threads 7 --out results/rtx5080_<host>.json`, and repeat with `--threads 8`.
The script builds for the visible SM (sm_120) and picks GPU-local physical cores.

- **Wake-up of memop vs kernel waits** (`wakeup`, `with_work`): front-end polling on Blackwell/GeForce may differ.
  This decides the default backend.
- **CPU side:** with 8 cores, the coordinator and 7 pool threads spinning hot leave nothing for the Python step loop
  and driver threads. Measure `--threads 7` vs 8, and the GEMV bandwidth on dual-channel <memory> (expect ≈ 60–80 GB/s
  peak; here 86 GB/s with 16 threads, 72 GB/s with 8). A single CCD may also change the pool barrier cost.
- **PCIe 5.0 latency** to mapped memory: zc reads of `y` and the spin-kernel polls.
- **hostfunc baseline** on the desktop, to confirm the FreeToken-reported 30–50 µs on consumer parts.
- Whether a parked graph trips anything when the GPU also drives a display (watchdog/TDR). It did not here, but this is a headless A5000.
- Power: GPU utilization and clocks with a resident spin kernel vs memop waits (`nvidia-smi dmon` during `with_work`).

Not covered by this spike: multiple tasks per slot (top-k CPU experts per layer with a real router payload), more than one
graph sharing a worker concurrently (the per-slot words assume one in-flight request per slot, which holds for
serialized replays on one stream), `cuStreamBatchMemOp` to fuse the write and wait into one node, and the
coordinator's idle doze latency (≤ 50 µs for the first layer after 100 ms of idle; not measured).
The numbers above come from a busy shared host, and p99 tails (for example hostfunc 13.6 ms) partly reflect that noise.

## Consumer cards (RTX 4090 / 5080 / 5090 desktops), 2026-10-06

Answers Q5. One desktop per class, everything inside one persistent container per desktop (CUDA 13.0.0 devel image,
nvcc 13.0.48, g++ 13; uv venv with torch 2.11.0+cu130, Python 3.12.12). GPUs are headless.

| | rtx-4090 | rtx-5080 | rtx-5090 |
|---|---|---|---|
| GPU | RTX 4090 (sm_89), PCIe 4.0 x16 | RTX 5080 (sm_120), PCIe 5.0 x16 | RTX 5090 (sm_120), PCIe 5.0 x16 |
| CPU | <cpu> (<cpu>, 8C/16T, 2 CCX, AVX2), 128 GB <memory> | <cpu> (<cpu>, 8C/16T, 1 CCD, 96 MB L3), 128 GB <memory> | same as 5080 |
| driver (all: Linux <version>) | <version> | <version> | <version> |
| memop caps | 64-bit 1, wait NOR 1, flush-remote-writes 0, host-native-atomics 0 | same | same |
| `KERNEL_EXEC_TIMEOUT` | 0 | 0 | 0 |

Runs per class (`results/<class>_*.json` + `.log`): `--threads 7` (`_t7`), `--threads 8` (`_t8`), a repeat of the
latency/wake-up/work/pyloop experiments at 7 threads (`_t7_rep2`), and 7 threads with the Python thread pinned to
the remaining physical core (`_t7_pinmain`, new flag `--pin-main`). `S6_BUILD_DIR=build/<class>` keeps one
`-march=native` build per CPU. Worker cores are 0–6 (or 0–7); CPUs 8–15 are their SMT siblings.

New experiments in `bench.py` (all in the JSON): `pyloop` (Python step loop under a hot worker), `pcie` (GPU access
latency to mapped pinned memory), `power` (NVML power/clock/util while graphs wait on the CPU, and during a 3 s
park), plus `kernel_exec_timeout` / display state in `env`.

### Handshake cost (empty CPU task, zero-copy data path), µs per handshake (slope over N = 1/24/48/75)

Three runs per class: t7 / t8 / t7 repeat.

| mode | rtx-4090 | rtx-5080 | rtx-5090 | A5000 (above) |
|---|---|---|---|---|
| memop | 4.49 / 4.50 / 4.50 | 4.36 / 4.28 / 3.53 | 3.01 / 3.02 / 3.05 | 5.35 |
| **kernel** | **2.83 / 2.83 / 2.81** | **2.05 / 2.05 / 1.96** | **1.97 / 1.97 / 2.00** | 2.97 |
| hybrid | 4.40 / 4.40 / 4.38 | 4.05 / 4.05 / 3.39 | 3.05 / 3.06 / 3.09 | 5.22 |
| hostfunc | 23.8 / 24.0 / 25.5 | 24.2 / 32.6 / 23.8 | 20.7 / 28.4 / 28.9 | 40.3 |

Per-step overhead at N = 75 (zc, p50, t7): memop 337 / 332 / 227 µs, kernel 212 / 154 / 148 µs, hostfunc
1792 / 1796 / 1613 µs (4090 / 5080 / 5090). With the two memcpy nodes (`copy`) instead of zero-copy, a GPU-only
75-layer graph costs 1325 / 1114 / 1084 µs vs 398 / 381 / 391 µs: zero-copy stays mandatory on desktops.

**Wake-up vs CPU task length** (N = 24, CPU task = pure spin of S µs; µs per handshake, t7; the repeat run agrees
within 0.3 µs for memop/kernel/hybrid):

| S (µs) | 0 | 2 | 5 | 10 | 20 | 50 | 100 | 200 |
|---|---|---|---|---|---|---|---|---|
| 4090 memop | 4.4 | 4.7 | 9.7 | 4.9 | 5.0 | 8.1 | 7.6 | 8.8 |
| 4090 kernel | 2.8 | 2.6 | 2.7 | 2.8 | 2.2 | 2.7 | 2.8 | 2.3 |
| 5080 memop | 3.3 | 3.7 | 4.8 | 7.3 | 9.5 | 4.4 | 5.4 | 4.0 |
| 5080 kernel | 1.9 | 1.9 | 2.0 | 2.1 | 1.9 | 1.9 | 1.9 | 2.1 |
| 5090 memop | 3.5 | 3.6 | 4.8 | 7.2 | 9.4 | 4.2 | 5.3 | 4.0 |
| 5090 kernel | 2.0 | 2.0 | 2.1 | 2.0 | 1.9 | 2.2 | 2.2 | 2.1 |
| hybrid (all) | 3.1–4.1 | | | | | | | 3.2–3.6 |
| hostfunc (all) | 21–35 | | | | | | | |

The A5000 pattern holds on GeForce Ada and Blackwell: the front-end memop wait is non-monotonic in the CPU time
(up to 9.5–9.7 µs), the 1-thread spin kernel is flat at 1.9–2.2 µs (Blackwell) / 2.2–2.8 µs (Ada).

**With the real CPU task** (N = 75, bf16 GEMV 768 × 2048 per layer, 225 MiB sweep; zc; mean step / CPU compute
measured in the worker / overhead per handshake):

| class | run | memop | kernel | hybrid | hostfunc |
|---|---|---|---|---|---|
| rtx-4090 | t7 | 8094 / 7027 / 8.8 | **7617 / 6996 / 2.8** | 7698 / 7037 / 3.4 | 9839 / 6984 / 32.6 |
| | t8 | 8235 / 7154 / 8.9 | 7647 / 7026 / 2.8 | 7814 / 7154 / 3.3 | 9739 / 7020 / 30.8 |
| | t7 pinned main | 8061 / 6930 / 9.7 | 7540 / 6921 / 2.9 | 7619 / 6959 / 3.4 | 9588 / 6962 / 29.7 |
| rtx-5080 | t7 | 5421 / 4367 / 9.0 | **3985 / 3443 / 2.1** | 4882 / 4190 / 4.1 | 6805 / 3531 / 38.6 |
| | t8 | 5146 / 4147 / 8.2 | 4007 / 3464 / 2.1 | 4628 / 3943 / 4.0 | 6447 / 3544 / 33.6 |
| | t7 pinned main | 4417 / 3353 / 9.1 | 3889 / 3356 / 2.0 | 3961 / 3353 / 3.0 | 5965 / 3451 / 28.4 |
| rtx-5090 | t7 | 4703 / 3836 / 6.3 | **3966 / 3419 / 2.1** | 4371 / 3751 / 3.0 | 6208 / 3453 / 31.5 |
| | t8 | 4658 / 3776 / 6.5 | 3999 / 3453 / 2.1 | 4299 / 3679 / 3.1 | 5776 / 3536 / 24.6 |
| | t7 pinned main | 4408 / 3342 / 9.0 | 3879 / 3332 / 2.1 | 3978 / 3355 / 3.1 | 5587 / 3433 / 23.5 |

GPU-only step: 403–411 (4090), 381–383 (5080), 391 (5090) µs. The spin-kernel wait saves 0.5–1.4 ms per 75-layer
step over memop on every class.

### CPU side on an 8-core desktop

GEMV bandwidth (`cpu_task`, 225 MiB sweep, per-layer µs / GB/s; 1 / 2 / 4 / 8 threads):

| host | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| <cpu> + <memory> (4090 desktop) | 160 / 19.7 | 98 / 32.2 | 89 / 35.5 | 93 / 34.0 |
| <cpu> + <memory> (5080 desktop) | 64 / 49.5 | 46 / 68.0 | 47 / 67.7 | 48 / 66.2 |
| <cpu> + <memory> (5090 desktop) | 61 / 51.7 | 45 / 69.8 | 43 / 73.4 | 46 / 67.8 |

Dual-channel <memory> saturates at **66–73 GB/s with 2–4 threads**; <memory> at **32–36 GB/s**. More than 4 threads buys
nothing for a bandwidth-bound expert GEMV, so a desktop CPU-expert pool needs ≤ 4 compute threads. The 96 MB L3
of the <cpu> matters: in `overlap` (N = 24, 72 MiB of weights) the per-layer CPU task drops to 12–15 µs
(L3-resident) vs 43–47 µs from DRAM; the <cpu> (2 × 16 MB L3) stays at ~94 µs.

**Does a hot worker starve the Python loop?** `pyloop`: each step runs a fixed pure-Python workload (`py_work`,
p50 µs), then replays the 75-layer graph (≤ 2 in flight). Alone = same workload, no GPU, worker idle:

| class | run | main thread CPUs | alone | GPU-only graph | kernel wait | memop wait |
|---|---|---|---|---|---|---|
| rtx-4090 | t7 | 7–15 | 417 | 431 | 584 | 570 |
| | t8 | 8–15 | 419 | 439 | 599 | 585 |
| | t7 pinned | 7, 15 | 332 | 311 | 330 | 333 |
| rtx-5080 | t7 | 7–15 | 189 | 196 | 377 | 355 |
| | t8 | 8–15 | 186 | 193 | 423 | 414 |
| | t7 pinned | 7, 15 | 188 | 189 | 191 | 195 |
| rtx-5090 | t7 | 7–15 | 184 | 183 | 401 | 330 |
| | t8 | 8–15 | 180 | 186 | 420 | 405 |
| | t7 pinned | 7, 15 | 176 | 178 | 181 | 184 |

Yes: whenever the Python thread can land on the SMT sibling of a spinning worker it runs **1.4× (<cpu>) to 2.3×
(<cpu>) slower** — with 8 workers it has no other choice, and with 7 workers the scheduler still puts it there
(repeat run: 373 / 341 µs on the 5080, 355 / 340 on the 5090). Pinning it to the free physical core (both SMT
threads of core 7) removes the effect entirely (≤ 5 % over alone), and also speeds up the worker itself (CPU
compute 3350 vs 3440–4370 µs per step on <cpu>, because the main thread no longer competes with a worker's SMT
sibling). Step time p50 with kernel waits: 5080 4309 (t7) / 4266 (t8) / 3858 µs (pinned), 5090 4390 / 4435 /
3866, 4090 7791 / 7733 / 7537.

### PCIe and mapped memory

`pcie`: a kernel that reads 768 fp32 from mapped pinned memory costs **+0.50–0.56 µs** over the same kernel on
device memory, on all three (PCIe 4.0 4090: +0.55; PCIe 5.0 5080/5090: +0.50/+0.53); posted writes to mapped
memory cost +0.00–0.02 µs. That is the whole zero-copy read penalty per layer, and it bounds how fast a spin kernel
can observe DONE: the 1.9–2.8 µs spin-kernel handshake (GPU write → CPU sees → CPU write → GPU poll sees) is a
few such round trips plus the coordinator's poll.

### Power, clocks and the display watchdog

NVML, 10 s windows; "waiting" = N = 24 graphs whose CPU task is a 200 µs spin (GPU parked in a wait ~95 % of the
time); "parked 3 s" = one handshake stalled for 3 s. W (mean) / SM MHz / reported util %:

| class | idle (context open) | waiting, memop | waiting, kernel | parked 3 s, memop | parked 3 s, kernel |
|---|---|---|---|---|---|
| rtx-4090 | 58.9 / 2520 / 0 | 60.5 / 2520 / 4 | 81.1 / 2700 / 100 | 59.0 / 2520 / 0 | 81.6 / 2700 / 100 |
| rtx-5080 | 49.0 / 2610 / 0 | 50.5 / 2790 / 3 | 62.8 / 2790 / 100 | 49.1 / 2617 / 0 | 62.6 / 2790 / 100 |
| rtx-5090 | 78.8 / 2393 / 0 | 96.5 / 2685 / 3 | 113.7 / 2685 / 100 | 82.1 / 2408 / 0 | 114.2 / 2685 / 98 |

(hybrid = kernel within 1 W; t8 run within 0.5 W.) A resident 1-thread spin kernel costs **+12–14 W (5080), +17–32 W
(5090), +21–23 W (4090)** over a memop wait, keeps the SM clock at boost and shows the GPU as 100 % utilized; GPU
temperatures stayed ≤ 56 °C, no clock throttling. On a desktop that is a cost, not a hazard; for laptops (power-
capped) memop stays the right choice.

Watchdog: `CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT` = 0 and no display is attached on all three desktops, so the
Linux display watchdog cannot fire; a 2 s stall of one handshake simply takes 2.007–2.016 s with no error. A
**display-attached** GeForce (timeout = 1) was not available — a graph parked in a spin kernel there is untested and
must keep stalls under the X/Wayland watchdog (seconds); the memop wait occupies no SM and is the safe choice when
`KERNEL_EXEC_TIMEOUT` = 1.

### Robustness (all classes, t7 and t8)

Identical outcome to the A5000: all four flavours capture (memop 4 `batch_mem_op` nodes, max err ≤ 4.8e-7 over 50
replays); long runs memop/zc 1.5 M, memop/copy 750 k, kernel/zc 750 k, hostfunc/zc 375 k handshakes = expected,
max err 7.2e-7; slow worker exact; abort with 20 replays queued drains in 8–10 ms (334 fail-open, 0 stranded),
restart exact; `quiesce` / `ordered` map swaps 0 violations, `unsafe` 300. A 1 ms kernel on another stream finishes
in 1.03 ms while a graph is parked 300 ms. Overlap (N = 24, GPU work matched to the CPU task) is complete with
kernel waits (5080: 509 µs on a branch vs 498 GPU-only; 4090: 2460 vs 2413).

### Decision for the consumer classes

1. **Default wait backend on all three GeForce classes: spin kernel** (`kernel` mode, or `hybrid` if the signal
   must not occupy an SM): 1.9–2.8 µs per handshake flat, 2.0–2.9 µs under the real CPU task, vs memop 3.0–4.5 µs
   empty and 6.3–9.7 µs under real work. Signal side: memop write or 1-thread release store are within 1 µs; keep
   the flag-pair primitive with both backends. Fall back to memop when `KERNEL_EXEC_TIMEOUT` = 1 (display GPU) or
   on power-capped parts. Never `cudaLaunchHostFunc` (21–39 µs, ×10).
2. **Worker threads on an 8-core desktop: 7, plus the Python step loop pinned to the 8th physical core (both SMT
   threads).** 8 workers (or 7 without pinning) slow the Python loop 1.4–2.3×; pinned, there is no measurable
   interference. Of the 7, ≤ 4 compute threads saturate DRAM for GEMV; the rest are wasted spinning.
3. Zero-copy payloads (+0.5 µs per mapped read) and the single-word REQ/DONE protocol carry over unchanged.
4. Budget: on the 5080 desktop a 75-layer CPU-expert step at 3 MiB/layer is DRAM-bound at ≈ 3.35 ms of CPU time
   (68 GB/s); on <cpu> / <memory> it is 2× that. Handshakes add only 0.15 ms (kernel waits).
