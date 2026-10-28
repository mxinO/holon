"""S6 -- CPU handshake inside CUDA graphs. Single self-contained benchmark.

    CUDA_VISIBLE_DEVICES=0 python bench.py --out results/<host>.json [--threads 8] [--cores 24-31] [--quick]
    python bench.py --build-only

Builds the extension (csrc/) on first use with torch.utils.cpp_extension (needs nvcc + ninja + a C++ compiler).
Every graph is a torch.cuda.CUDAGraph; the extension only adds the handshake ops and the native CPU worker.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROWS, COLS = 768, 2048  # one simulated CPU expert GEMV per handshake: y[768] = W[768, 2048] (bf16) @ x[2048]
MODES = ("none", "memop", "kernel", "hybrid", "hostfunc")  # hybrid = memop signal + kernel wait
DATA = ("none", "copy", "zc")
TOL = 1e-3


def load_ext():
    major, minor = torch.cuda.get_device_capability()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
    from torch.utils.cpp_extension import load

    # S6_BUILD_DIR: one build per host class (-march=native differs between hosts)
    build = HERE / os.environ.get("S6_BUILD_DIR", "build")
    build.mkdir(parents=True, exist_ok=True)
    return load(
        "s6_handshake",
        [str(HERE / "csrc" / "worker.cpp"), str(HERE / "csrc" / "kernels.cu")],
        extra_cflags=["-O3", "-march=native"],
        extra_cuda_cflags=["-O3"],
        build_directory=str(build),
    )


# ---------------------------------------------------------------------------------------------- host facts


def parse_cpus(text: str) -> list[int]:
    out = []
    for part in text.strip().split(","):
        a, _, b = part.partition("-")
        out += range(int(a), int(b or a) + 1)
    return out


def pick_cores(n: int) -> list[int]:
    """GPU-local, allowed CPUs, one logical CPU per physical core."""
    allowed = sorted(os.sched_getaffinity(0))
    p = torch.cuda.get_device_properties(0)
    local = Path(f"/sys/bus/pci/devices/{p.pci_domain_id:04x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.0/local_cpulist")
    if local.exists():
        near = set(parse_cpus(local.read_text()))
        allowed = [c for c in allowed if c in near] or allowed
    seen, cores = set(), []
    for c in allowed:
        sib = Path(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list")
        key = sib.read_text().strip() if sib.exists() else str(c)
        if key not in seen:
            seen.add(key)
            cores.append(c)
    return cores[:n]


def sibling_key(c: int) -> str:
    sib = Path(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list")
    return sib.read_text().strip() if sib.exists() else str(c)


def sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as e:  # noqa: BLE001 -- best-effort host facts
        return f"<{e}>"


def host_facts(ext, cores) -> dict:
    p = torch.cuda.get_device_properties(0)
    return {
        "gpu": p.name,                  # host details (name, CPU, RAM, driver, OS) stay out of records
        "sm": f"{p.major}.{p.minor}",
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "nvcc": sh("nvcc --version | tail -1"),
        "python": sys.version.split()[0],
        "cores_used": cores,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "memop_caps": ext.memop_caps(),
        "kernel_exec_timeout": kernel_exec_timeout(),
    }


# ---------------------------------------------------------------------------------------------- the rig


class Rig:
    """N simulated MoE layers; each hands one GEMV to the CPU worker through one handshake slot."""

    def __init__(self, ext, slots: int, cores: list[int], verify: bool = False):
        self.ext = ext
        g = torch.Generator().manual_seed(0)
        self.w = (torch.randn(slots, ROWS, COLS, generator=g) * 0.02).bfloat16()
        self.x_host = torch.zeros(slots, COLS).pin_memory()
        self.y_host = torch.zeros(slots, ROWS).pin_memory()
        self.base = torch.randn(slots, COLS, generator=g).cuda()
        self.x = torch.zeros(slots, COLS, device="cuda")
        self.y = torch.zeros(slots, ROWS, device="cuda")
        self.step = torch.zeros((), device="cuda")
        self.err = torch.zeros((), device="cuda")
        self.xv = ext.device_view(self.x_host)  # zero-copy aliases of the pinned buffers
        self.yv = ext.device_view(self.y_host)
        self.w_dev = self.w.float().cuda() if verify else None
        self.side = torch.cuda.Stream()
        self.worker = ext.Worker(self.w, self.x_host, self.y_host, cores)
        self.worker.start()

    def close(self) -> int:
        torch.cuda.synchronize()
        return self.worker.stop()

    def layer(self, l: int, mode: str, data: str, gpu_ns: int, place: str, verify: bool):
        """data: "none" (flag only), "copy" (memcpy nodes D2H x / H2D y), "zc" (kernels write x / read y in
        mapped pinned memory directly)."""
        ext, w = self.ext, self.worker
        x = self.xv[l] if data == "zc" else self.x[l]
        torch.sin(self.base[l] + self.step * 0.37 + l, out=x)  # the layer's GPU work producing the expert input
        if data == "copy":
            self.x_host[l].copy_(x, non_blocking=True)
        if gpu_ns and place == "branch":  # independent GPU work on a branch forked before the hand-off
            self.side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.side):
                ext.busy(gpu_ns)
        if mode in ("memop", "hybrid"):
            w.signal_memop(l)
        elif mode == "kernel":
            w.signal_kernel(l)
        elif mode == "hostfunc":
            w.hostfunc(l)  # runs the task inline; this stream is blocked until it returns
        if gpu_ns and place == "same":  # independent GPU work on the same stream, between signal and wait
            ext.busy(gpu_ns)
        if mode == "memop":
            w.wait_memop(l)
        elif mode in ("kernel", "hybrid"):
            w.wait_kernel(l)
        if gpu_ns and place == "branch":
            torch.cuda.current_stream().wait_stream(self.side)
        if gpu_ns and place == "serial":
            ext.busy(gpu_ns)
        if data == "copy":
            self.y[l].copy_(self.y_host[l], non_blocking=True)
        elif data == "zc":
            torch.mul(self.yv[l], 1.0, out=self.y[l])  # the consumer kernel reads y over PCIe
        if verify:  # reference on the GPU from a device-resident copy of x (no PCIe reads in the GEMV)
            if data == "zc":
                torch.sin(self.base[l] + self.step * 0.37 + l, out=self.x[l])
            ref = self.w_dev[l] @ self.x[l]
            torch.maximum(self.err, (self.y[l] - ref).abs().max(), out=self.err)

    def body(self, n, mode, data="zc", gpu_ns=0, place="serial", verify=False):
        self.step.add_(1)
        for l in range(n):
            self.layer(l, mode, data, gpu_ns, place, verify)

    def capture(self, n, mode, keep=False, **kw) -> torch.cuda.CUDAGraph:
        self.body(n, mode, **kw)  # eager warm-up (also exercises eager memops)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph(keep_graph=keep)
        with torch.cuda.graph(g):
            self.body(n, mode, **kw)
        if keep:
            g.instantiate()
        torch.cuda.synchronize()
        return g


def time_replays(ext, g, reps: int, worker=None, chunk: int = 200) -> dict:
    """Per-replay GPU time (us) from events between replays; a 5 ms gate lets the host run ahead.
    With a worker, also the mean CPU compute time per replay (cpu_us), measured inside the worker."""
    times = []
    g.replay()
    torch.cuda.synchronize()
    c0 = worker.compute_ns() if worker else 0
    for c in range(0, reps, chunk):
        k = min(chunk, reps - c)
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(k + 1)]
        ext.busy(5_000_000)
        ev[0].record()
        for i in range(k):
            g.replay()
            ev[i + 1].record()
        torch.cuda.synchronize()
        times += [ev[i].elapsed_time(ev[i + 1]) * 1e3 for i in range(k)]
    out = summarize(times)
    if worker:
        out["cpu_us"] = (worker.compute_ns() - c0) / 1e3 / reps
    return out


def summarize(xs: list[float]) -> dict:
    s = sorted(xs)
    q = lambda f: s[min(len(s) - 1, int(f * len(s)))]  # noqa: E731
    return {"n": len(s), "mean": statistics.fmean(s), "p50": q(0.5), "p90": q(0.9), "p99": q(0.99), "max": s[-1]}


class Watchdog:
    """Kills the process if a phase hangs (a lost handshake deadlocks the stream forever)."""

    def __init__(self, seconds: float, what: str):
        self.t = threading.Timer(seconds, self._fire, args=(what, seconds))
        self.t.daemon = True

    @staticmethod
    def _fire(what, seconds):
        print(f"WATCHDOG: '{what}' did not finish in {seconds}s -- deadlock", flush=True)
        os._exit(3)

    def __enter__(self):
        self.t.start()

    def __exit__(self, *exc):
        self.t.cancel()


# ---------------------------------------------------------------------------------------------- experiments


def exp_capture(ext, cores) -> dict:
    """Q1: is each handshake flavour accepted by stream capture, and is the data path exact?"""
    out = {}
    rig = Rig(ext, 2, cores, verify=True)
    for mode in MODES[1:]:
        try:
            with Watchdog(60, f"capture {mode}"):
                g = rig.capture(2, mode, keep=True, verify=True)
                types = ext.graph_node_types(g.raw_cuda_graph())
                rig.err.zero_()
                for _ in range(50):
                    g.replay()
                torch.cuda.synchronize()
            out[mode] = {
                "captured": True,
                "node_types": {t: types.count(t) for t in sorted(set(types))},
                "max_abs_err_50_replays": rig.err.item(),
            }
        except Exception as e:  # noqa: BLE001 -- record the failure, keep going
            out[mode] = {"captured": False, "error": repr(e)}
        print("capture", mode, out[mode], flush=True)
    out["stranded_after_stop"] = rig.close()
    return out


def exp_latency(ext, cores, reps) -> dict:
    """Q2: per-handshake round trip with an empty CPU task; N handshakes per step; three data paths."""
    rig = Rig(ext, 75, cores)
    rig.worker.set_compute(False)
    rows = []
    for data in DATA:
        for n in (1, 24, 48, 75):
            base = None
            for mode in MODES:
                with Watchdog(300, f"latency {mode} n={n}"):
                    g = rig.capture(n, mode, data=data)
                    t = time_replays(ext, g, reps)
                    del g
                if mode == "none":
                    base = t
                row = {"data": data, "n": n, "mode": mode, "step_us": t}
                if mode != "none":
                    row["overhead_per_step_us_p50"] = t["p50"] - base["p50"]
                    row["overhead_per_handshake_us_p50"] = row["overhead_per_step_us_p50"] / n
                rows.append(row)
                print(f"latency data={data:4s} n={n:3d} {mode:8s} p50={t['p50']:8.1f}us p99={t['p99']:8.1f}us "
                      f"per-hs={row.get('overhead_per_handshake_us_p50', 0):6.2f}us", flush=True)
    rig.close()
    return {"rows": rows, "slope_us_per_handshake": slopes(rows)}


def slopes(rows) -> dict:
    """Least-squares slope of (step p50 - baseline p50) over N: the marginal cost of one handshake."""
    out = {}
    for data in DATA:
        for mode in MODES[1:]:
            pts = [(r["n"], r["overhead_per_step_us_p50"]) for r in rows if r["data"] == data and r["mode"] == mode]
            mx = statistics.fmean(p[0] for p in pts)
            my = statistics.fmean(p[1] for p in pts)
            out[f"{mode}/{data}"] = sum((x - mx) * (y - my) for x, y in pts) / sum((x - mx) ** 2 for x, _ in pts)
    print("slopes us/handshake", {k: round(v, 2) for k, v in out.items()}, flush=True)
    return out


def exp_cpu_task(ext, cores) -> dict:
    """CPU cost of the simulated expert (bf16 GEMV 768x2048, 3 MiB of weights per layer), vs threads."""
    out = {}
    w = (torch.randn(75, ROWS, COLS) * 0.02).bfloat16()
    x, y = torch.randn(75, COLS).pin_memory(), torch.zeros(75, ROWS).pin_memory()
    for t in (1, 2, 4, 8, 16):
        if t > len(cores):
            break
        wk = ext.Worker(w, x, y, cores[:t])
        wk.start()
        hot = wk.task_us(0, 2000)  # same 3 MiB every time: cache-resident
        cold = statistics.fmean(wk.task_us(s, 1) for _ in range(20) for s in range(75))  # 225 MiB sweep
        wk.stop()
        ref = (w[0].float() @ x[0]).sub(y[0]).abs().max().item()
        out[t] = {"hot_us": hot, "sweep75_us": cold, "gbps_sweep": ROWS * COLS * 2 / cold / 1e3, "max_abs_err": ref}
        print(f"cpu task threads={t:2d} hot={hot:7.1f}us sweep={cold:7.1f}us ({out[t]['gbps_sweep']:.1f} GB/s)",
              flush=True)
    return out


def exp_wakeup(ext, cores, reps) -> dict:
    """Q2: does the GPU-side wait get slower to notice DONE the longer the CPU takes? The CPU task is a pure
    spin of S us (no memory traffic); overhead/handshake = (step mean - GPU-only mean - CPU time) / N."""
    n = 24
    rig = Rig(ext, n, cores)
    rig.worker.set_compute(False)
    base = time_replays(ext, rig.capture(n, "none"), reps)
    rows = []
    for mode in MODES[1:]:
        g = rig.capture(n, mode)
        for spin in (0, 2, 5, 10, 20, 50, 100, 200):
            rig.worker.set_spin(spin)
            with Watchdog(300, f"wakeup {mode} {spin}"):
                t = time_replays(ext, g, reps // 2, rig.worker)
            ov = (t["mean"] - base["mean"] - t["cpu_us"]) / n
            rows.append({"mode": mode, "spin_us": spin, "step_us": t, "overhead_per_handshake_us": ov})
            print(f"wakeup {mode:8s} spin={spin:4d}us step={t['mean']:8.1f}us per-hs={ov:6.2f}us", flush=True)
        del g
    rig.worker.set_spin(0)
    rig.close()
    return {"n": n, "data": "zc", "none_step_us": base, "rows": rows}


def exp_with_work(ext, cores, reps) -> dict:
    """Q2 with the real CPU task: N=75 layers per step, each computes its GEMV on the CPU worker.
    overhead/handshake = (step mean - GPU-only step mean - CPU compute per step measured in the worker) / 75."""
    n = 75
    rig = Rig(ext, n, cores)
    rows = []
    for data in ("copy", "zc"):
        base = None
        for mode in MODES:
            with Watchdog(300, f"work {mode}"):
                g = rig.capture(n, mode, data=data)
                t = time_replays(ext, g, reps, rig.worker)
                del g
            base = base or t
            row = {"data": data, "mode": mode, "step_us": t}
            if mode != "none":
                row["overhead_per_handshake_us"] = (t["mean"] - base["mean"] - t["cpu_us"]) / n
            rows.append(row)
            print(f"work n={n} data={data:4s} {mode:8s} mean={t['mean']:8.1f}us p99={t['p99']:8.1f}us "
                  f"cpu={t['cpu_us']:8.1f}us per-hs={row.get('overhead_per_handshake_us', 0):6.2f}us", flush=True)
    rig.close()
    return {"n": n, "rows": rows}


def exp_overlap(ext, cores, reps) -> dict:
    """Q3: independent GPU work while the CPU computes -- same stream between signal and wait, or on a branch."""
    n = 24
    rig = Rig(ext, n, cores)
    g = rig.capture(n, "memop")
    cpu_layer = time_replays(ext, g, 200, rig.worker)["cpu_us"] / n
    del g
    gpu_ns = int(cpu_layer * 1000)  # GPU work matched to the in-graph CPU work per layer
    cases = [("none", "serial"), ("memop", "serial"), ("memop", "same"), ("memop", "branch"), ("kernel", "same"),
             ("kernel", "branch"), ("hostfunc", "serial"), ("hostfunc", "branch")]
    rows = []
    for mode, place in cases:
        with Watchdog(300, f"overlap {mode} {place}"):
            g = rig.capture(n, mode, gpu_ns=gpu_ns, place=place)
            t = time_replays(ext, g, reps, rig.worker)
            del g
        rows.append({"mode": mode, "place": place, "step_us": t})
        print(f"overlap {mode:8s} {place:6s} mean={t['mean']:8.1f}us cpu={t['cpu_us']:8.1f}us", flush=True)

    # Another stream outside the graph must keep running while a graph is parked on a wait.
    rig.worker.set_delay(300_000, 10**12)  # next task stalls 300 ms
    g = rig.capture(1, "memop")  # the eager warm-up eats this stall
    rig.worker.set_delay(300_000, 10**12)
    other = torch.cuda.Stream()
    with Watchdog(30, "other-stream"):
        g.replay()
        t0 = time.perf_counter()
        with torch.cuda.stream(other):
            ext.busy(1_000_000)
        other.synchronize()
        other_ms = (time.perf_counter() - t0) * 1e3
        torch.cuda.synchronize()
        parked_ms = (time.perf_counter() - t0) * 1e3
    rig.worker.set_delay(0, 0)
    rig.close()
    print(f"overlap other stream: 1 ms kernel done after {other_ms:.1f} ms, parked graph after {parked_ms:.0f} ms")
    return {
        "n": n,
        "cpu_us_per_layer": cpu_layer,
        "gpu_busy_us_per_layer": gpu_ns / 1e3,
        "rows": rows,
        "other_stream_1ms_kernel_while_graph_parked_ms": other_ms,
        "parked_graph_done_ms": parked_ms,
    }


def exp_robust(ext, cores, long_reps) -> dict:
    """Q4: long replay runs with on-device verification; slow worker; stall; abort while in flight; restart."""
    out = {}
    rig = Rig(ext, 75, cores, verify=True)
    w = rig.worker
    long_runs = (("memop", "zc", long_reps), ("memop", "copy", long_reps // 2), ("kernel", "zc", long_reps // 2),
                 ("hostfunc", "zc", long_reps // 4))
    for mode, data, reps in long_runs:
        g = rig.capture(75, mode, data=data, verify=True)
        rig.err.zero_()
        torch.cuda.synchronize()
        t0, k0 = time.perf_counter(), w.tasks()
        done = []
        with Watchdog(1800, f"long {mode}/{data}"):
            for i in range(reps):  # StepLoop-like: at most 2 replays in flight, host waits on step i-2's event
                if len(done) >= 2:
                    done.pop(0).synchronize()
                g.replay()
                done.append(torch.cuda.Event())
                done[-1].record()
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        key = f"long_{mode}_{data}"
        out[key] = {"replays": reps, "handshakes": w.tasks() - k0, "expected": 75 * reps,
                    "max_abs_err": rig.err.item(), "wall_s": dt, "step_ms": dt / reps * 1e3}
        print("robust", key, out[key], flush=True)
        del g

    g = rig.capture(75, "memop", verify=True)
    rig.err.zero_()
    w.set_delay(2000, 97)  # every 97th task sleeps 2 ms
    with Watchdog(300, "slow worker"):
        t = time_replays(ext, g, 300, w, chunk=100)
    w.set_delay(0, 0)
    out["slow_worker_2ms_every_97"] = {"step_us": t, "max_abs_err": rig.err.item()}
    print("robust slow", out["slow_worker_2ms_every_97"], flush=True)

    w.set_delay(2_000_000, 10**12)  # one 2 s stall: no driver timeout, the wait just waits
    t0 = time.perf_counter()
    with Watchdog(60, "2s stall"):
        g.replay()
        torch.cuda.synchronize()
    out["single_2s_stall"] = {"replay_s": time.perf_counter() - t0, "max_abs_err": rig.err.item()}
    w.set_delay(0, 0)
    print("robust stall", out["single_2s_stall"], flush=True)

    # Abort with 20 replays queued (each would take 75 x 20 ms): fail-open drains the stream.
    w.set_delay(20_000, 1)
    for _ in range(20):
        g.replay()
    time.sleep(0.1)
    a0, t0 = w.aborted(), time.perf_counter()
    w.abort()
    with Watchdog(60, "abort drain"):
        torch.cuda.synchronize()
    drain_s = time.perf_counter() - t0
    stranded = w.stop()
    w.set_delay(0, 0)
    out["abort_in_flight"] = {"drain_s": drain_s, "aborted_tasks": w.aborted() - a0, "stranded": stranded}
    print("robust abort", out["abort_in_flight"], flush=True)

    w.start()  # restart the same worker; the existing graph keeps working
    rig.err.zero_()
    with Watchdog(120, "restart"):
        for _ in range(200):
            g.replay()
        torch.cuda.synchronize()
    out["restart_after_abort"] = {"replays": 200, "max_abs_err": rig.err.item()}
    del g
    out["stranded_after_clean_stop"] = rig.close()
    print("robust restart", out["restart_after_abort"], "stranded", out["stranded_after_clean_stop"], flush=True)
    return out


def exp_quiesce(ext, cores, steps) -> dict:
    """Q4: swapping a double-buffered residency map that the graph reads, under queued replays.

    The graph reads the active map (maps[sel]) at the first and the last layer; a step must see exactly one
    version (no torn map, no mid-step swap). Protocols:
      quiesce  drain in-flight replays (StepLoop.quiesce), then write the inactive buffer and flip sel
      ordered  no drain: write the inactive buffer and flip sel *on the replay stream* between replays
      unsafe   negative control: write the active buffer from another stream while replays are queued
    """
    E, n, every = 1 << 16, 8, 10
    rig = Rig(ext, n, cores)
    rig.worker.set_compute(True)
    maps = torch.zeros(2, E, dtype=torch.int64, device="cuda")
    sel = torch.zeros(1, dtype=torch.int64, device="cuda")
    ar = torch.arange(E, device="cuda")
    bad = torch.zeros((), dtype=torch.int64, device="cuda")
    seen = torch.zeros((), dtype=torch.int64, device="cuda")

    def read_version():
        m = maps.index_select(0, sel)[0]
        v = m[0] // E
        bad.add_((m - ar != v * E).any().long())  # torn: entries from two versions
        return v

    def body():
        v0 = read_version()
        rig.body(n, "memop", gpu_ns=50_000, place="same")
        v1 = read_version()
        bad.add_((v0 != v1).long())  # swapped under an in-flight step
        seen.copy_(v1)

    body()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        body()
    side = torch.cuda.Stream()
    staging = [torch.empty(E, dtype=torch.int64).pin_memory() for _ in range(2)]
    staged = [torch.cuda.Event() for _ in range(2)]
    out = {}
    for proto in ("quiesce", "ordered", "unsafe"):
        maps.copy_((ar + 0).expand(2, E))
        sel.zero_()
        bad.zero_()
        torch.cuda.synchronize()
        active, version, drains = 0, 0, 0
        t0 = time.perf_counter()
        with Watchdog(600, f"quiesce {proto}"):
            for s in range(steps):
                g.replay()
                if s % every != every - 1:
                    continue
                version += 1
                if proto == "quiesce":
                    torch.cuda.current_stream().synchronize()  # drain: commit every in-flight step
                    drains += 1
                    maps[1 - active].copy_(ar + version * E)
                    sel.fill_(1 - active)
                    active ^= 1
                elif proto == "ordered":
                    b = 1 - active
                    staged[b].synchronize()  # the host staging buffer may still feed an earlier copy
                    staging[b].copy_(torch.arange(E) + version * E)
                    maps[b].copy_(staging[b], non_blocking=True)
                    staged[b].record()
                    sel.fill_(b)
                    active = b
                else:
                    with torch.cuda.stream(side):  # not ordered after the queued replays
                        staging[0].copy_(torch.arange(E) + version * E)
                        maps[active].copy_(staging[0], non_blocking=True)
                    side.synchronize()
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        out[proto] = {"steps": steps, "swaps": version, "drains": drains, "violations": bad.item(),
                      "last_version_seen": seen.item(), "step_ms": dt / steps * 1e3}
        print("quiesce", proto, out[proto], flush=True)
    del g
    rig.close()
    return out


# ---------------------------------------------------------------------------------------------- desktop extras
# Added for the consumer desktops (README "Consumer cards"): Python-loop starvation, PCIe latency to
# mapped pinned memory, power/clocks with resident spin waits, kernel-exec-timeout (display watchdog) attribute.


def py_work(n: int = 3000) -> float:
    """Fixed pure-Python stand-in for the StepLoop's per-step scheduling work; returns its wall time in us."""
    t0 = time.perf_counter()
    d = {}
    for i in range(n):
        d[i & 255] = d.get(i & 255, 0) + i * 3 % 7
    sorted(d.values())
    return (time.perf_counter() - t0) * 1e6


def exp_pyloop(ext, cores, steps) -> dict:
    """Does a hot worker (coordinator + pool spinning) starve the Python step loop? StepLoop-like: per step the
    main thread does py_work(), replays the N=75 graph, keeps <= 2 replays in flight. Compared with the same loop
    on a GPU-only graph (worker idle/dozing) and with py_work alone (no GPU)."""
    out = {"main_thread_cpus": sorted(os.sched_getaffinity(0)), "worker_cores": cores}
    idle = [py_work() for _ in range(steps)]
    out["py_work_alone_us"] = summarize(idle)
    rig = Rig(ext, 75, cores)
    for mode in ("none", "kernel", "memop"):
        g = rig.capture(75, mode, data="zc")
        torch.cuda.synchronize()
        work, step, done = [], [], []
        t_prev = time.perf_counter()
        with Watchdog(600, f"pyloop {mode}"):
            for i in range(steps):
                work.append(py_work())
                if len(done) >= 2:
                    done.pop(0).synchronize()
                g.replay()
                done.append(torch.cuda.Event())
                done[-1].record()
                t = time.perf_counter()
                step.append((t - t_prev) * 1e6)
                t_prev = t
            torch.cuda.synchronize()
        out[mode] = {"py_work_us": summarize(work), "step_us": summarize(step[10:])}
        print(f"pyloop {mode:7s} py_work p50={out[mode]['py_work_us']['p50']:7.1f}us "
              f"p99={out[mode]['py_work_us']['p99']:7.1f}us step p50={out[mode]['step_us']['p50']:8.1f}us "
              f"(alone p50={out['py_work_alone_us']['p50']:.1f}us)", flush=True)
        del g
    rig.close()
    return out


def exp_pcie(ext, cores, reps) -> dict:
    """GPU access latency to mapped pinned memory: a graph of R tiny kernels that each read (or write) 768 fp32
    over PCIe vs the same kernels on device memory; the per-kernel difference ~ one PCIe round trip (all loads of
    a kernel are in flight together). Also the CPU->GPU->CPU flag round trip is the handshake itself (latency)."""
    R = 200
    host = torch.zeros(R, ROWS).pin_memory()
    hv = ext.device_view(host)
    dev = torch.zeros(R, ROWS, device="cuda")
    dst = torch.zeros(R, ROWS, device="cuda")
    out = {}
    for name, src, to in (("read_device", dev, dst), ("read_mapped", hv, dst),
                          ("write_device", dst, dev), ("write_mapped", dst, hv)):
        g = torch.cuda.CUDAGraph()
        for i in range(R):  # warm-up
            torch.mul(src[i], 1.0, out=to[i])
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            for i in range(R):
                torch.mul(src[i], 1.0, out=to[i])
        t = time_replays(ext, g, reps)
        out[name] = {"us_per_kernel_p50": t["p50"] / R, "step_us": t}
        del g
    out["read_extra_us"] = out["read_mapped"]["us_per_kernel_p50"] - out["read_device"]["us_per_kernel_p50"]
    out["write_extra_us"] = out["write_mapped"]["us_per_kernel_p50"] - out["write_device"]["us_per_kernel_p50"]
    print("pcie", {k: (round(v["us_per_kernel_p50"], 2) if isinstance(v, dict) else round(v, 2))
                   for k, v in out.items()}, flush=True)
    return out


class Nvml:
    """Background NVML sampler (power W, SM clock MHz, GPU util %, temperature C) at ~100 ms."""

    def __init__(self):
        import pynvml
        pynvml.nvmlInit()
        self.nv = pynvml
        idx = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
        self.h = pynvml.nvmlDeviceGetHandleByIndex(idx)
        self.rows, self.on = [], False

    def sample(self):
        nv, h = self.nv, self.h
        return (nv.nvmlDeviceGetPowerUsage(h) / 1e3, nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM),
                nv.nvmlDeviceGetUtilizationRates(h).gpu, nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU))

    def _loop(self):
        while self.on:
            self.rows.append(self.sample())
            time.sleep(0.1)

    def __enter__(self):
        self.rows, self.on = [], True
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()
        return self

    def __exit__(self, *exc):
        self.on = False
        self.t.join()

    def stats(self, skip: int = 10) -> dict:
        rows = self.rows[skip:] or self.rows
        cols = list(zip(*rows))
        return {k: {"mean": statistics.fmean(c), "max": max(c)} for k, c in
                zip(("power_w", "sm_mhz", "util_pct", "temp_c"), cols)} | {"samples": len(rows)}


def exp_power(ext, cores, seconds) -> dict:
    """GPU power/clocks while graphs mostly *wait* for the CPU: N=24, the CPU task is a 200 us spin per layer, so
    the GPU sits in a wait ~95 % of the time (memop: front-end, no SM; kernel/hybrid: a resident 1-thread spin
    kernel). Plus one 3 s park on a single stalled handshake per mode, and the idle GPU as a reference."""
    nv = Nvml()
    out = {}
    time.sleep(2)
    with nv:
        time.sleep(seconds)
    out["idle"] = nv.stats()
    rig = Rig(ext, 24, cores)
    rig.worker.set_compute(False)
    for mode in ("memop", "kernel", "hybrid"):
        g = rig.capture(24, mode)
        rig.worker.set_spin(200)
        n = 0
        with Watchdog(seconds * 4 + 60, f"power {mode}"), nv:
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < seconds:
                for _ in range(20):
                    g.replay()
                n += 20
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
        rig.worker.set_spin(0)
        out[f"waiting_{mode}"] = nv.stats() | {"step_us": dt / n * 1e6}
        rig.worker.set_delay(3_000_000, 10**12)  # next task stalls 3 s: the graph is parked on one wait
        with Watchdog(60, f"park {mode}"), nv:
            g.replay()
            torch.cuda.synchronize()
        rig.worker.set_delay(0, 0)
        out[f"parked3s_{mode}"] = nv.stats(skip=5)
        print("power", mode, {k: out[k] for k in (f"waiting_{mode}", f"parked3s_{mode}")}, flush=True)
        del g
    print("power idle", out["idle"], flush=True)
    rig.close()
    return out


def kernel_exec_timeout() -> int | None:
    """CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT (1 = display watchdog active on this GPU)."""
    import ctypes
    try:
        cu = ctypes.CDLL("libcuda.so.1")
        dev, v = ctypes.c_int(), ctypes.c_int()
        cu.cuInit(0)
        cu.cuDeviceGet(ctypes.byref(dev), 0)
        cu.cuDeviceGetAttribute(ctypes.byref(v), 17, dev)
        return v.value
    except OSError:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--threads", type=int, default=8, help="CPU worker threads (coordinator + pool)")
    ap.add_argument("--cores", type=str, default=None, help="explicit CPU list, e.g. 24-31 (default: GPU-local)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", type=str, default=None, help="comma list of experiments")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--pin-main", action="store_true", help="pin the Python thread to cores free of worker threads")
    args = ap.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "pin the GPU explicitly with CUDA_VISIBLE_DEVICES"
    torch.cuda.init()
    t0 = time.time()
    ext = load_ext()
    print(f"extension ready in {time.time() - t0:.0f}s", flush=True)
    if args.build_only:
        return
    sweep = parse_cpus(args.cores) if args.cores else pick_cores(16)
    cores = sweep[: args.threads]
    torch.set_num_threads(1)
    others = [c for c in os.sched_getaffinity(0) if c not in cores]
    if args.pin_main:  # main thread only on physical cores without a worker thread (not on a worker's SMT sibling)
        busy = {sibling_key(c) for c in cores}
        others = [c for c in others if sibling_key(c) not in busy] or others
    os.sched_setaffinity(0, others or os.sched_getaffinity(0))
    reps = 200 if args.quick else 1000
    res = {"env": host_facts(ext, cores), "args": vars(args) | {"out": str(args.out)}}
    print(json.dumps(res["env"], indent=1), flush=True)
    exps = {
        "capture": lambda: exp_capture(ext, cores),
        "cpu_task": lambda: exp_cpu_task(ext, sweep),
        "latency": lambda: exp_latency(ext, cores, reps),
        "wakeup": lambda: exp_wakeup(ext, cores, reps),
        "with_work": lambda: exp_with_work(ext, cores, reps),
        "overlap": lambda: exp_overlap(ext, cores, reps),
        "robust": lambda: exp_robust(ext, cores, 2000 if args.quick else 20000),
        "quiesce": lambda: exp_quiesce(ext, cores, 300 if args.quick else 3000),
        "pyloop": lambda: exp_pyloop(ext, cores, 500 if args.quick else 3000),
        "pcie": lambda: exp_pcie(ext, cores, reps),
        "power": lambda: exp_power(ext, cores, 3 if args.quick else 10),
    }
    only = args.only.split(",") if args.only else list(exps)
    for name in only:
        t = time.time()
        res[name] = exps[name]()
        res[name + "_wall_s"] = time.time() - t
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(res, indent=1, default=str))
    print("done", flush=True)


if __name__ == "__main__":
    main()
