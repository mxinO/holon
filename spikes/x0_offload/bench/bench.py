"""Milestone X step 0 (docs/v1-offload.md §8): M1, M2, M3, M5, M7 and per-layer prefill expert GEMMs.

    CUDA_VISIBLE_DEVICES=0 python bench.py --tag rtx-5080 --out ../../../evidence/x0_bench_rtx-5080_<date>.json \
        --nvme-dir <node-local dir> [--only m1,m3,m2,m5,m7,prefill]

Builds ggml-cpu (native ISA, and AVX2 for M3's comparison, run in a child process) from the vendored llama.cpp
(`--llama`), then this directory's extension (csrc/) against it, into `--build`. Expert blocks come from the
Qwen3-30B-A3B GGUFs in the HF cache (real Q4_K_M and UD-IQ3_XXS experts; IQ2_* experts quantized from real Q4_K
weights with ggml's quantizer).
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
GGUF_REPO = "models--unsloth--Qwen3-30B-A3B-GGUF"
Q4KM, UDIQ3 = "Qwen3-30B-A3B-Q4_K_M.gguf", "Qwen3-30B-A3B-UD-IQ3_XXS.gguf"
HIDDEN, INTER, EXPERTS, TOP_K, LAYERS = 2048, 768, 128, 8, 48
T = {"q4_k": 12, "q6_k": 14, "iq2_xxs": 16, "iq2_xs": 17, "iq3_xxs": 18, "iq3_s": 21, "iq2_s": 22, "iq4_xs": 23}
NAMES = {v: k for k, v in T.items()}
# M1 blob classes (bytes, rounded to 256): Flash-Next-like IQ2_XS expert, the Q4_K_M expert of this model (mean of
# its Q6_K-down and Q4_K-down layers), GLM-like Maya-S expert.
BLOBS = {"1.46MB": 1_460_000 // 256 * 256, "2.8MB": 2_856_960, "7.1MB": 7_100_000 // 256 * 256}
MB = 1e6
# Host arenas tile the real experts over this many slots (distinct addresses): a CPU sweep over > 1 GB of experts
# reads DRAM even on a desktop L3 of up to 96 MB, as decode does (each layer's experts are revisited only after the
# other 47 layers' ~1 GB).
SLOTS = 384
_CORES: list[int] = []


# ---------------------------------------------------------------------------------------------- build / host


def build_ext(isa: str, llama: Path, build: Path):
    ggml = build / f"ggml-{isa}"
    libs = [ggml / "ggml/src/libggml-cpu.a", ggml / "ggml/src/libggml-base.a"]
    if not all(p.exists() for p in libs):
        subprocess.run(["bash", str(HERE / "build_ggml.sh"), str(llama), str(ggml), isa], check=True)
    major, minor = torch.cuda.get_device_capability()
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    from torch.utils.cpp_extension import load

    out = build / f"ext-{isa}"
    out.mkdir(parents=True, exist_ok=True)
    return load(f"x0_{isa}", [str(HERE / "csrc/x0.cpp"), str(HERE / "csrc/kernels.cu")],
                extra_cflags=["-O3", "-march=native", "-std=c++17"], extra_cuda_cflags=["-O3"],
                extra_include_paths=[str(llama / "ggml/include")], extra_ldflags=[str(p) for p in libs],
                build_directory=str(out))


def physical_cores() -> list[int]:
    """One logical CPU per physical core of the process's CPUs, read once (before main() pins its own thread)."""
    global _CORES
    if _CORES:
        return _CORES
    seen, cores = set(), []
    for c in sorted(os.sched_getaffinity(0)):
        sib = Path(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list")
        key = sib.read_text().strip() if sib.exists() else str(c)
        if key not in seen:
            seen.add(key)
            cores.append(c)
    _CORES = cores
    return cores


def worker_cores() -> list[int]:
    """Compute cores: all physical cores but the first, which keeps the OS, the Python thread and the GPU feeder
    (a pool thread on the first core ran 1.7x slower than on any other on the desktops)."""
    return physical_cores()[1:]


def host_core() -> int:
    return physical_cores()[0]


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def env_facts(ext) -> dict:
    p = torch.cuda.get_device_properties(0)
    feats = ext.cpu_features()
    return {"gpu": p.name, "sm": f"{p.major}.{p.minor}", "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "nvcc": sh("nvcc --version | tail -1"), "python": sys.version.split()[0],
            "ggml_cpu_build": feats, "isa_tier": "x86-avx512" if feats["avx512"] else "x86-avx2",
            "llama_cpp_commit": sh(f"git -C {ARGS.llama} rev-parse HEAD"),
            "commit": sh(f"git -C {HERE} rev-parse HEAD"), "dirty": bool(sh(f"git -C {HERE} status --porcelain -- .")),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}


def summarize(xs) -> dict:
    xs = sorted(xs)
    return {"p50": statistics.median(xs), "mean": statistics.fmean(xs), "p10": xs[len(xs) // 10],
            "p90": xs[len(xs) * 9 // 10], "n": len(xs)}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------------------------------------- GGUF experts


def gguf_path(name: str) -> Path:
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub" / GGUF_REPO / "snapshots"
    return next(hub.glob(f"*/{name}"))


_READERS = {}


def reader(name):
    from gguf import GGUFReader

    if name not in _READERS:
        _READERS[name] = GGUFReader(gguf_path(name))
    return _READERS[name]


def layer_parts(name: str, layer: int) -> dict:
    """{role: (ggml type, uint8 [E, bytes per expert])} for gate, up, down of one layer (a memmap view)."""
    out = {}
    for t in reader(name).tensors:
        if t.name.startswith(f"blk.{layer}.ffn_") and t.name.endswith("_exps.weight"):
            role = t.name.split(".")[2].removeprefix("ffn_").removesuffix("_exps")
            data = np.fromfile(reader(name).data.filename, dtype=np.uint8, count=int(t.n_bytes),
                               offset=int(t.data_offset))         # one large read: page faults over NFS are slow
            out[role] = (int(t.tensor_type), data.reshape(EXPERTS, -1))
    return out


def find_layer(name: str, gu: str, down: str) -> int | None:
    types = {t.name: NAMES.get(int(t.tensor_type)) for t in reader(name).tensors}
    return next((layer for layer in range(LAYERS) if types[f"blk.{layer}.ffn_gate_exps.weight"] == gu
                 and types[f"blk.{layer}.ffn_down_exps.weight"] == down), None)


class Experts:
    """One expert class in an exact mapped pinned arena (gate | up | down per expert, GGUF bytes)."""

    def __init__(self, ext, name: str, gu: int, down: int, gate, up, dn):
        self.name, self.gu, self.down, self.cls = name, gu, down, ext.ExpertClass(HIDDEN, INTER, gu, down)
        n = len(gate)
        self.slots = SLOTS
        self.arena = ext.HostArena(SLOTS * self.cls.bytes)
        view = self.arena.tensor().view(SLOTS, self.cls.bytes)
        g = self.cls.down_off // 2
        for e in range(SLOTS):          # tile the n source experts over all slots (distinct addresses)
            view[e, :g] = torch.from_numpy(np.ascontiguousarray(gate[e % n]))
            view[e, g:2 * g] = torch.from_numpy(np.ascontiguousarray(up[e % n]))
            view[e, 2 * g:] = torch.from_numpy(np.ascontiguousarray(dn[e % n]))
        self.real = n

    @property
    def ptr(self):
        return self.arena.host_ptr()


def real_class(ext, file: str, layer: int, label: str) -> Experts:
    p = layer_parts(file, layer)
    assert p["gate"][0] == p["up"][0]
    return Experts(ext, label, p["gate"][0], p["down"][0], p["gate"][1], p["up"][1], p["down"][1])


def quantized_classes(ext, n: int = 16) -> list[Experts]:
    """IQ2_XXS / IQ2_XS / IQ2_S experts quantized from n real Q4_K experts (layer 6: Q4_K gate, up, down), uniform
    importance (no imatrix: the bytes and the compute are the type's; the accuracy is not the point); and the Maya-S
    mix (IQ2_XXS gate/up, IQ2_S down) from the same."""
    src = layer_parts(Q4KM, 6)
    f32 = {}
    for role, rows, cols in (("gate", INTER, HIDDEN), ("up", INTER, HIDDEN), ("down", HIDDEN, INTER)):
        t, data = src[role]
        f32[role] = ext.dequantize(t, torch.from_numpy(np.ascontiguousarray(data[:n])), n * rows, cols)
    q = {}
    for name in ("iq2_xxs", "iq2_xs", "iq2_s"):
        t0 = time.time()
        for role, cols in (("gate", HIDDEN), ("up", HIDDEN), ("down", INTER)):
            q[name, role] = ext.quantize(T[name], f32[role], torch.ones(cols), 8).view(n, -1).numpy()
        log(f"quantized {n} experts to {name} in {time.time() - t0:.1f}s")
    out = [Experts(ext, name, T[name], T[name], q[name, "gate"], q[name, "up"], q[name, "down"])
           for name in ("iq2_xxs", "iq2_xs", "iq2_s")]
    out.append(Experts(ext, "iq2_xxs+iq2_s (Maya-S mix)", T["iq2_xxs"], T["iq2_s"], q["iq2_xxs", "gate"],
                       q["iq2_xxs", "up"], q["iq2_s", "down"]))
    return out


# ---------------------------------------------------------------------------------------------- GPU timing


def gpu_time_us(ext, fn, reps: int, gate_ns: int = 3_000_000) -> float:
    """Mean GPU time of fn() back to back, behind a busy kernel so the host runs ahead of the GPU."""
    fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ext.busy(gate_ns)
    a.record()
    for _ in range(reps):
        fn()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) * 1000 / reps


# ---------------------------------------------------------------------------------------------- M1


def blob_table(src_ptrs, dst_ptrs, size):
    t = torch.tensor([[s, d, size // 16] for s, d in zip(src_ptrs, dst_ptrs, strict=True)], dtype=torch.int64)
    return t, t.cuda()


def m1(ext) -> dict:
    out = {"copy": [], "blobs": {}}
    arena = ext.HostArena(1 << 30)
    arena.tensor().view(torch.int64).random_()
    host, dev = arena.host_ptr(), torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
    src = arena.tensor()
    for mb in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
        n = mb << 20
        us = gpu_time_us(ext, partial(dev[:n].copy_, src[:n], non_blocking=True), max(3, 2048 // mb))
        out["copy"].append({"mib": mb, "us": us, "gbps": n / us / 1e3})
        log(f"M1 copy {mb} MiB: {n / us / 1e3:.1f} GB/s")
    g = torch.Generator().manual_seed(0)
    for label, size in BLOBS.items():
        slots = (1 << 30) // size
        rows = []
        for n in (1, 2, 4, 8, 16, 32, 64):
            pick = torch.randperm(slots, generator=g)[:n].tolist()
            th, td = blob_table([host + s * size for s in pick], [dev.data_ptr() + i * size for i in range(n)], size)
            reps = max(5, 400 // n)
            r = {"n": n, "bytes": n * size}
            r["memcpy_each_gbps"] = n * size / gpu_time_us(ext, partial(ext.memcpy_each, th), reps) / 1e3
            r["memcpy_batch_gbps"] = n * size / gpu_time_us(ext, partial(ext.memcpy_batch, th), reps) / 1e3
            gather = {}
            for unroll in (2, 4, 8):
                for ctas in (2, 4, 8, 16, 24, 32, 48, 64, 96, 128):
                    us = gpu_time_us(ext, partial(ext.gather, td, n, ctas, 256, unroll), reps)
                    gather[f"{ctas}x256u{unroll}"] = n * size / us / 1e3
            best = max(gather, key=gather.get)
            r["gather_gbps"], r["gather_best"], r["gather_best_gbps"] = gather, best, gather[best]
            ok = [int(k.split("x")[0]) for k, v in gather.items() if v >= 0.95 * gather[best]]
            r["gather_saturating_ctas"] = min(ok)
            dma = max(r["memcpy_each_gbps"], r["memcpy_batch_gbps"])
            r["gather_over_dma"] = gather[best] / dma
            # correctness of the gather: the staging slots equal the source blobs
            ext.gather(td, n, 32, 256, 4)
            torch.cuda.synchronize()
            for i, s in enumerate(pick[:2]):
                assert torch.equal(dev[i * size:(i + 1) * size].cpu(), src[s * size:(s + 1) * size]), "gather mismatch"
            rows.append(r)
            log(f"M1 {label} n={n}: each {r['memcpy_each_gbps']:.1f} batch {r['memcpy_batch_gbps']:.1f} "
                f"gather {r['gather_best_gbps']:.1f} ({best}, saturates at {r['gather_saturating_ctas']} CTAs)")
        # the best gather at n = 8 inside a CUDA graph (decode's in-graph fetch)
        r8 = next(r for r in rows if r["n"] == 8)
        ctas, rest = r8["gather_best"].split("x")
        unroll = int(rest.split("u")[1])
        pick = torch.randperm(slots, generator=g)[:8].tolist()
        th, td = blob_table([host + s * size for s in pick], [dev.data_ptr() + i * size for i in range(8)], size)
        ext.gather(td, 8, int(ctas), 256, unroll)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(4):
                ext.gather(td, 8, int(ctas), 256, unroll)
        us = gpu_time_us(ext, graph.replay, 50) / 4
        out["blobs"][label] = {"size": size, "rows": rows, "graph_gather_n8_gbps": 8 * size / us / 1e3,
                               "graph_gather_config": r8["gather_best"]}
        log(f"M1 {label} in-graph gather n=8: {8 * size / us / 1e3:.1f} GB/s")
    del arena
    return out


# ---------------------------------------------------------------------------------------------- M3


def m3(ext, classes, threads_list, tokens_list=(1, 2, 4, 8)) -> dict:
    cores = worker_cores()
    out = {}
    for ex in classes:
        res = {"gate_up": NAMES[ex.gu], "down": NAMES[ex.down], "expert_bytes": ex.cls.bytes,
               "real_experts": ex.real, "slots": ex.slots, "rows": []}
        for nt in threads_list:
            cpu = ext.Cpu(cores[:nt])
            assert cpu.threads() == nt
            for tokens in tokens_list:
                for njobs in ((1, 8) if tokens == 1 else (1,)):
                    iters = max(64, ex.slots // njobs)
                    ts = cpu.time(ex.cls, ex.ptr, ex.slots, njobs, tokens, iters + 16, 0)[16:]
                    us = statistics.median(ts) / njobs
                    res["rows"].append({"threads": nt, "tokens": tokens, "jobs": njobs, "us_per_expert": us,
                                        "gbps": ex.cls.bytes / us / 1e3, "batch_us": summarize(ts)})
            del cpu
        res["bitwise"] = bitwise(ext, ex, cores, max(threads_list))
        best = min((r for r in res["rows"] if r["tokens"] == 1 and r["jobs"] == 1), key=lambda r: r["us_per_expert"])
        log(f"M3 {ex.name}: best 1-token {best['us_per_expert']:.1f} us ({best['gbps']:.1f} GB/s, "
            f"{best['threads']} thr); bitwise mismatches {res['bitwise']['mismatches']}")
        out[ex.name] = res
    return out


def bitwise(ext, ex, cores, nmax: int) -> dict:
    """D7: a token's output alone == in a group of 2..8 tokens; == at 1 vs N threads; one job == in a batch of 4."""
    g = torch.Generator().manual_seed(3)
    x = torch.randn(1, 8, HIDDEN, generator=g)
    one, many = ext.Cpu(cores[:1]), ext.Cpu(cores[:nmax])
    checks = mism = 0
    for e in (0, 37, 101):
        alone = torch.zeros(8, HIDDEN)
        for t in range(8):
            y = torch.zeros(1, 1, HIDDEN)
            one.run(ex.cls, ex.ptr, [e], x[:, t:t + 1].contiguous(), y)
            alone[t] = y[0, 0]
        for size in range(2, 9):
            for cpu in (one, many):
                y = torch.zeros(1, size, HIDDEN)
                cpu.run(ex.cls, ex.ptr, [e], x[:, :size].contiguous(), y)
                checks += 1
                mism += not torch.equal(y[0].view(torch.int32), alone[:size].view(torch.int32))
        xs = torch.cat([x[:, :4]] * 4)                       # 4 jobs (experts e, e+1, e+2, e+3) of the same 4 tokens
        y = torch.zeros(4, 4, HIDDEN)
        many.run(ex.cls, ex.ptr, [e, e + 1, e + 2, e + 3], xs.contiguous(), y)
        checks += 1
        mism += not torch.equal(y[0].view(torch.int32), alone[:4].view(torch.int32))
    return {"checks": checks, "mismatches": int(mism), "threads": [1, nmax], "group_sizes": [2, 8]}


# ---------------------------------------------------------------------------------------------- M2


def m2(ext, ex, m1_res, threads_list, duration_ms=300) -> dict:
    cores, feeder = worker_cores(), host_core()
    half = ex.slots // 2
    size = ex.cls.bytes
    dev = torch.empty(16 * size, dtype=torch.uint8, device="cuda")
    # GPU blobs: the upper half of the slots (whole expert blobs), staged into 16 device slots round robin; the CPU
    # computes the lower half
    ids = list(range(half, ex.slots))
    th, td = blob_table([ex.ptr + e * size for e in ids], [dev.data_ptr() + (i % 16) * size for i in range(len(ids))],
                        size)
    th8, td8 = th[:8].contiguous(), td[:8].contiguous()
    best = m1_res["blobs"]["2.8MB"]["rows"][3]["gather_best"] if m1_res else "32x256u4"
    ctas, unroll = int(best.split("x")[0]), int(best.split("u")[1])
    out = {"gather_config": best, "expert_bytes": size, "steady": [], "split": {}}

    def run(nt, mode):
        cpu = ext.Cpu(cores[:nt]) if nt else None
        r = ext.contend(cpu, ex.cls, ex.ptr, half, 1, mode, th8, td8, ctas, 256, unroll, feeder, duration_ms)
        return {k: None if v != v else v for k, v in r.items()}  # NaN: that path did not run

    for mode in ("dma", "gather"):
        r = run(0, mode)
        out["steady"].append({"threads": 0, "mode": mode, **r})
        log(f"M2 {mode} alone: {r['gpu_gbps']:.1f} GB/s")
    for nt in threads_list:
        for mode in ("none", "dma", "gather"):
            r = run(nt, mode)
            out["steady"].append({"threads": nt, "mode": mode, **r})
            log(f"M2 {nt} thr + {mode}: cpu {r['cpu_us_per_expert']:.1f} us/expert, gpu {r['gpu_gbps']}")
    # the per-miss-count split, measured: n misses, k on the CPU, n - k gathered, started together
    for nt in [t for t in threads_list if t >= 2]:
        cpu = ext.Cpu(cores[:nt])
        table = {}
        e = 0
        for n in range(1, 11):
            for k in range(0, n + 1):
                xs = []
                for rep in range(40):
                    cpu_ids = [(e + j) % half for j in range(k)]
                    e += k
                    first = (rep * 3) % (len(ids) - 10)
                    xs.append(ext.split_once(cpu, ex.cls, ex.ptr, cpu_ids, "gather", th, td, first, n - k, ctas, 256,
                                             unroll))
                xs = xs[5:]
                table[f"{n},{k}"] = {"cpu_us": statistics.median(x[0] for x in xs),
                                     "gpu_us": statistics.median(x[1] for x in xs),
                                     "path_us": statistics.median(max(x[0], x[1]) for x in xs),
                                     "wall_us": statistics.median(x[2] for x in xs)}
        plan = {}
        for n in range(1, 11):
            ks = {k: table[f"{n},{k}"]["path_us"] for k in range(n + 1)}
            k = min(ks, key=ks.get)
            plan[n] = {"k": k, "path_us": ks[k], "fetch_only_us": ks[0], "gain": ks[0] / ks[k] - 1}
        out["split"][nt] = {"table": table, "plan": plan}
        log(f"M2 split {nt} thr: plan " + " ".join(f"{n}:{p['k']}" for n, p in plan.items()))
        del cpu
    return out


# ---------------------------------------------------------------------------------------------- M5


def m5(ext, ex, nthreads: int, rows_list=(1, 2, 4, 8), k_list=(1, 2, 4, 8, 10), reps=60) -> dict:
    cores = worker_cores()
    hs = ext.Handshake(ex.cls, ex.ptr, ex.slots, LAYERS, max(rows_list), cores[:nthreads])
    hs.start()
    step = torch.zeros((), dtype=torch.int64, device="cuda")
    outs = torch.zeros(LAYERS, max(rows_list), HIDDEN, device="cuda")
    side = torch.cuda.Stream()
    planned, done = ext.kPlanned(), ext.kDone()

    def body(plan, k, rows, gpu_ns, branch):
        step.add_(1)
        for layer in range(LAYERS):
            if plan == "host":
                hs.route(layer, step, rows)
            else:
                hs.plan(layer, step, k, rows)
            if plan != "gpu_only":
                hs.signal(layer)
            if plan == "host":
                hs.wait(layer, planned)
            if gpu_ns:  # concurrent GPU work (e.g. the fetch of the GPU share), on a branch or the same stream
                if branch:
                    side.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(side):
                        ext.busy(gpu_ns)
                else:
                    ext.busy(gpu_ns)
            if plan != "gpu_only":
                hs.wait(layer, done)
            if gpu_ns and branch:
                torch.cuda.current_stream().wait_stream(side)
            hs.consume(layer, outs[layer], k, rows)

    def timed(plan, k, rows, gpu_ns=0, branch=False):
        hs.set_host_plan(plan == "host", k, rows)
        body(plan, k, rows, gpu_ns, branch)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            body(plan, k, rows, gpu_ns, branch)
        for _ in range(5):
            g.replay()
        torch.cuda.synchronize()
        c0, n0 = hs.compute_ns(), hs.tasks()
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(reps + 1)]
        ext.busy(3_000_000)
        ev[0].record()
        for i in range(reps):
            g.replay()
            ev[i + 1].record()
        torch.cuda.synchronize()
        steps = [ev[i].elapsed_time(ev[i + 1]) * 1000 for i in range(reps)]
        tasks = hs.tasks() - n0
        cpu_layer = (hs.compute_ns() - c0) / 1000 / max(tasks, 1)
        ok = None
        if plan != "gpu_only":  # spot check: slot 0's CPU rows equal a fresh ggml-cpu run of its jobs (bitwise)
            ids = hs.ids(0)
            x = hs.x(0)[:rows].clone().unsqueeze(0).repeat(len(ids), 1, 1).contiguous()
            y = torch.zeros_like(x)
            ext.Cpu(cores[:1]).run(ex.cls, ex.ptr, ids, x, y)
            ok = torch.equal(y.view(torch.int32), hs.y(0)[:len(ids), :rows].view(torch.int32))
        del g
        return {"step_us": summarize(steps), "cpu_us_per_layer": cpu_layer, "tasks": tasks, "exact": ok}

    out = {"threads": nthreads, "rows": []}
    for rows in rows_list:
        for k in k_list:
            r = {"rows": rows, "k": k}
            r["gpu_only"] = timed("gpu_only", k, rows)
            for plan in ("device", "host"):
                r[plan] = timed(plan, k, rows)
            base = r["gpu_only"]["step_us"]["mean"]
            for plan in ("device", "host"):
                p = r[plan]
                p["overhead_us_per_layer"] = (p["step_us"]["mean"] - base - p["cpu_us_per_layer"] * LAYERS) / LAYERS
            # overlap: concurrent GPU work as long as the CPU's per-layer time, on a branch
            w = int(r["device"]["cpu_us_per_layer"] * 1000)
            r["work_ns"] = w
            r["gpu_only_work"] = timed("gpu_only", k, rows, w, True)
            for plan in ("device", "host"):
                q = timed(plan, k, rows, w, True)
                exposed = q["step_us"]["mean"] - r["gpu_only_work"]["step_us"]["mean"]
                q["exposed_us_per_layer"] = exposed / LAYERS
                q["hidden_fraction"] = 1 - exposed / max(q["cpu_us_per_layer"] * LAYERS, 1e-9)
                r[plan + "_work"] = q
            out["rows"].append(r)
            log(f"M5 rows={rows} k={k}: cpu {r['device']['cpu_us_per_layer']:.1f} us/layer, overhead dev "
                f"{r['device']['overhead_us_per_layer']:.1f} host {r['host']['overhead_us_per_layer']:.1f} us/layer; "
                f"with work exposed dev {r['device_work']['exposed_us_per_layer']:.1f} host "
                f"{r['host_work']['exposed_us_per_layer']:.1f}; exact {r['device']['exact']}/{r['host']['exact']}")
    out["stranded"] = hs.stop()
    return out


# ---------------------------------------------------------------------------------------------- M7


def m7(ext, directory: str, file_gib: int, seconds: float) -> dict:
    d = Path(directory)
    path = d / f"x0_nvme_test_{os.getpid()}.bin"
    stat = os.statvfs(d)
    free = stat.f_bavail * stat.f_frsize
    size = min(file_gib << 30, free // 4) // (64 << 20) * (64 << 20)
    if size < 2 << 30:
        return {"skipped": f"less than 8 GiB free at the local mount ({free >> 30} GiB)"}
    out = {"file_bytes": size, "rows": []}
    cores = worker_cores()
    try:
        secs = ext.nvme_fill(str(path), size)
        out["fill_gbps"] = size / secs / 1e9
        log(f"M7 wrote {size >> 30} GiB in {secs:.1f}s")
        for label, nbytes in BLOBS.items():
            for workers in (1, 2, 4, 8):
                for qd in (1, 2, 4, 8, 16, 32):
                    if qd < workers:
                        continue
                    r = ext.nvme_read(str(path), size, nbytes, qd, workers, "aio", seconds, cores)
                    out["rows"].append({"blob": label, "mode": "aio", "workers": workers, "qd": qd, **r})
                    log(f"M7 {label} aio w={workers} qd={qd}: {r['gbps']:.2f} GB/s p50 {r['lat_p50_us']:.0f} us")
                r = ext.nvme_read(str(path), size, nbytes, workers, workers, "pread", seconds, cores)
                out["rows"].append({"blob": label, "mode": "pread", "workers": workers, "qd": workers, **r})
                log(f"M7 {label} pread w={workers}: {r['gbps']:.2f} GB/s p50 {r['lat_p50_us']:.0f} us")
    finally:
        if path.exists():
            path.unlink()
        out["test_file_removed"] = not path.exists()
        log(f"M7 test file removed: {out['test_file_removed']}")
    return out


# ---------------------------------------------------------------------------------------------- prefill


def expert_layer(lib, chunk, prec, gate_up, down, x, ids):
    """LinearExperts' mmq path: per chunk, gate/up with ids into f32, f32 SwiGLU, down with ids."""
    (tgu, wgu, wsgu), (tdn, wdn, wsdn) = gate_up, down
    tokens = len(x)
    gu = torch.empty(min(tokens, chunk), TOP_K, 2 * INTER, device="cuda")
    h = torch.empty(min(tokens, chunk), TOP_K, INTER, device="cuda")
    y = torch.empty(tokens, TOP_K, HIDDEN, device="cuda")
    for s in range(0, tokens, chunk):
        xs, ic = x[s:s + chunk], ids[s:s + chunk].contiguous()
        lib.mmq_id(tgu, prec, wgu, xs, ic, gu[:len(ic)], wsgu)
        lib.swiglu(gu[:len(ic)], h[:len(ic)])
        lib.mmq_id(tdn, prec, wdn, h[:len(ic)], ic, y[s:s + chunk], wsdn)
    return y


def prefill(chunks=(64, 256, 1024, 4096), reps=10) -> dict:
    """One MoE layer's routed-expert GEMMs (gate/up mmq with ids, SwiGLU, down mmq with ids) with kernels.ggml_cuda's
    llama.cpp kernels, as LinearExperts runs them (512-token chunks), uniform random top-8 routing."""
    from holon_plugins.kernels.ggml_cuda import extension as gx
    from holon_plugins.kernels.ggml_cuda.plugin import CHUNK, PREC

    major, minor = torch.cuda.get_device_capability()
    lib = gx.load(major * 10 + minor)
    dev = torch.cuda.current_device()
    out = {}
    for layer in (0, 6):
        p = layer_parts(Q4KM, layer)
        tgu, tdn = p["gate"][0], p["down"][0]
        gu = torch.cat([torch.from_numpy(np.ascontiguousarray(p[r][1])).view(EXPERTS, INTER, -1)
                        for r in ("gate", "up")], dim=1).reshape(EXPERTS * 2 * INTER, -1).cuda()
        dn = torch.from_numpy(np.ascontiguousarray(p["down"][1])).reshape(EXPERTS * HIDDEN, -1).cuda()
        ws_gu, ws_dn = (torch.empty(lib.workspace_bytes_id(dev, k, CHUNK, TOP_K, EXPERTS), dtype=torch.uint8,
                                    device="cuda") for k in (HIDDEN, INTER))
        label = f"q4_k gate/up, {NAMES[tdn]} down"
        rows = []
        for tokens in chunks:
            g = torch.Generator(device="cuda").manual_seed(tokens)
            x = torch.randn(tokens, 1, HIDDEN, device="cuda", generator=g).bfloat16()
            ids = torch.rand(tokens, EXPERTS, device="cuda", generator=g).topk(TOP_K, dim=1).indices.int()
            run = partial(expert_layer, lib, CHUNK, PREC["q8_1"], (tgu, gu, ws_gu), (tdn, dn, ws_dn), x, ids)
            us = [gpu_time_us(_EXT, run, 1, 1_000_000) for _ in range(reps)]
            distinct = int(ids.unique().numel())
            rows.append({"tokens": tokens, "ms": statistics.median(us) / 1000, "distinct_experts": distinct})
            log(f"prefill {label} T={tokens}: {statistics.median(us) / 1000:.3f} ms/layer ({distinct} experts)")
        out[label] = {"expert_bytes": gu.numel() // EXPERTS + dn.numel() // EXPERTS, "rows": rows}
    return out


# ---------------------------------------------------------------------------------------------- main


def main():
    global ARGS, _EXT
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="GPU class, e.g. rtx-5080")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="m1,m3,m2,m5,m7,prefill")
    ap.add_argument("--isa", default="native", choices=("native", "avx2"))
    ap.add_argument("--llama", default=str(REPO / "spikes/s2_gguf/third_party/llama.cpp"))
    ap.add_argument("--build", default=str(REPO / "local/tmp/x0-bench/build"))
    ap.add_argument("--nvme-dir", default=None)
    ap.add_argument("--nvme-gib", type=int, default=16)
    ap.add_argument("--nvme-seconds", type=float, default=1.5)
    ap.add_argument("--threads", default="1,2,4,6,7", help="M3 thread counts")
    ap.add_argument("--m2-threads", default="1,2,4,6")
    ap.add_argument("--m5-threads", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-avx2-child", action="store_true")
    ap.add_argument("--merge", action="store_true", help="update --out instead of replacing it")
    ARGS = ap.parse_args()
    ARGS.llama = Path(ARGS.llama).resolve()
    torch.cuda.init()
    torch.cuda.set_stream(torch.cuda.Stream())  # cudaMemcpyBatchAsync refuses the legacy default stream
    build = Path(ARGS.build) / ARGS.tag
    ext = _EXT = build_ext(ARGS.isa, ARGS.llama, build)
    allowed = os.sched_getaffinity(0)
    physical_cores()
    torch.set_num_threads(1)  # torch's CPU pool would share the one core below with 0.1 s OpenMP spins
    for tid in os.listdir("/proc/self/task"):  # this thread and every runtime thread stay off the worker cores (S6)
        os.sched_setaffinity(int(tid), {host_core()})
    only = set(ARGS.only.split(","))
    out = Path(ARGS.out)
    res = json.loads(out.read_text()) if ARGS.merge and out.exists() else {}
    res.setdefault("runs", []).append({"command": " ".join([Path(sys.argv[0]).name] + sys.argv[1:]),
                                       "date": time.strftime("%Y-%m-%d"), "env": env_facts(ext)})
    res.update({"tag": ARGS.tag, "env": res["runs"][-1]["env"], "host_core": host_core(),
                "worker_cores": worker_cores()})

    def save():
        out.write_text(json.dumps(res, indent=1, default=float) + "\n")

    threads = [int(t) for t in ARGS.threads.split(",") if int(t) <= len(worker_cores())]
    if "m1" in only:
        res["m1"] = m1(ext)
        save()
    q4k_q6k = q4k_q4k = None
    if only & {"m3", "m2", "m5"}:
        q4k_q6k = real_class(ext, Q4KM, 0, "q4_k+q6_k (Q4_K_M, layer 0)")
    if "m3" in only:
        q4k_q4k = real_class(ext, Q4KM, 6, "q4_k+q4_k (Q4_K_M, layer 6)")
        classes = [q4k_q6k, q4k_q4k]
        for gu, dn in (("iq3_xxs", "iq3_xxs"), ("iq3_s", "iq3_s")):
            layer = find_layer(UDIQ3, gu, dn)
            if layer is not None:
                classes.append(real_class(ext, UDIQ3, layer, f"{gu}+{dn} (UD-IQ3_XXS, layer {layer})"))
        classes += quantized_classes(ext, 4 if ARGS.quick else 16)
        if find_layer(UDIQ3, "iq3_s", "iq3_s") is None:
            log("no IQ3_S/IQ3_S layer in the UD file; quantizing IQ3_S")
            src = layer_parts(Q4KM, 6)
            q = {}
            for role, rows, cols in (("gate", INTER, HIDDEN), ("up", INTER, HIDDEN), ("down", HIDDEN, INTER)):
                f = ext.dequantize(src[role][0], torch.from_numpy(np.ascontiguousarray(src[role][1][:16])), 16 * rows,
                                   cols)
                q[role] = ext.quantize(T["iq3_s"], f, None, 8).view(16, -1).numpy()
            classes.append(Experts(ext, "iq3_s+iq3_s (quantized)", T["iq3_s"], T["iq3_s"], q["gate"], q["up"],
                                   q["down"]))
        res[f"m3_{ARGS.isa}"] = m3(ext, classes, threads[:2] if ARGS.quick else threads)
        save()
        del classes, q4k_q4k
        if ARGS.isa == "native" and ext.cpu_features()["avx512"] and not ARGS.no_avx2_child:
            child = out.with_suffix(".avx2.json")
            log("M3 again with the AVX2 build (child process)")
            subprocess.run([sys.executable, __file__, "--tag", ARGS.tag, "--out", str(child), "--only", "m3",
                            "--isa", "avx2", "--threads", ARGS.threads, "--build", ARGS.build, "--llama",
                            str(ARGS.llama)] + (["--quick"] if ARGS.quick else []), check=True,
                           preexec_fn=lambda: os.sched_setaffinity(0, allowed))  # not this thread's one core
            sub = json.loads(child.read_text())
            res["m3_avx2"], res["env"]["avx2_build"] = sub["m3_avx2"], sub["env"]["ggml_cpu_build"]
            child.unlink()
            save()
    if "m2" in only:
        m2_threads = [int(t) for t in ARGS.m2_threads.split(",") if int(t) <= len(worker_cores())]
        res["m2"] = m2(ext, q4k_q6k, res.get("m1"), m2_threads, 150 if ARGS.quick else 300)
        save()
    if "m5" in only:
        res["m5"] = m5(ext, q4k_q6k, ARGS.m5_threads, *(((1, 8), (1, 10), 20) if ARGS.quick else ()))
        save()
    if "prefill" in only:
        res["prefill"] = prefill(reps=3 if ARGS.quick else 10)
        save()
    if "m7" in only:
        res["m7"] = m7(ext, ARGS.nvme_dir, ARGS.nvme_gib, ARGS.nvme_seconds) if ARGS.nvme_dir else {
            "skipped": "no node-local NVMe directory given"}
        save()
    log("done", out)


if __name__ == "__main__":
    main()
