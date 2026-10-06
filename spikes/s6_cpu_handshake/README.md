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
