"""FN's speed round A (the 5090's ≈ 1 GiB outside the allocator): device memory outside torch's allocator per startup
stage — the CUDA context; the host pools' device mappings (`platform.host_memory`'s, measured at each call); the
rest of the model's build (kernel modules loaded as their first launches run, library handles); the startup
profile; the graphs' execs — then after served requests; and the context's per-thread stack, which the driver
reserves for every thread the device can hold (SMs × threads per SM × bytes) and grows to the largest frame a launch
needs, never back. With `--stack N` the stack is set to N bytes before the engine starts: its value after the
requests is the largest frame the engine's kernels needed.
Usage: outside.py <model> '<engine flags>' [--seqs 1] [--stack N] [--requests requests.json]"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import pathlib
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # (CUDA starts before the engine)
import torch  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
CU = ctypes.CDLL("libcuda.so.1")
STACK = 0                                     # CU_LIMIT_STACK_SIZE


def stack() -> int:
    v = ctypes.c_size_t()
    assert CU.cuCtxGetLimit(ctypes.byref(v), STACK) == 0
    return v.value


def outside() -> int:                         # MiB outside torch's allocator
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return (total - free - torch.cuda.memory_reserved()) >> 20


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("flags")
    ap.add_argument("--seqs", type=int, default=1)
    ap.add_argument("--stack", type=int, default=None)
    ap.add_argument("--requests", default=None)
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT / "spikes/fn6"))
    torch.zeros(1, device="cuda")
    props = torch.cuda.get_device_properties(0)
    out = {"gpu": props.name, "sms": props.multi_processor_count,
           "threads_per_sm": props.max_threads_per_multi_processor, "stack_default": stack(), "context": outside()}
    if a.stack is not None:
        assert CU.cuCtxSetLimit(STACK, ctypes.c_size_t(a.stack)) == 0
        out["stack_set"], out["context_with_stack_set"] = stack(), outside()
    from holon.core import executor, platform
    host_memory, mappings = platform.host_memory, []

    def host_memory_(nbytes, device, mapped):
        before = outside()
        t = host_memory(nbytes, device, mapped)
        mappings.append({"gib": round(nbytes / 2**30, 2), "mapped": mapped, "device_mib": outside() - before})
        return t
    platform.host_memory = host_memory_
    stages = {}
    sm, cg = executor.Executor.state_memory, executor.Executor.capture_graphs

    def sm_(self, ledger, fraction):
        stages["built"] = outside()
        r = sm(self, ledger, fraction)
        stages["profiled"] = outside()
        return r

    def cg_(self):
        stages["before_capture"] = outside()
        cg(self)
        stages["captured"] = outside()
    executor.Executor.state_memory, executor.Executor.capture_graphs = sm_, cg_
    from _engine import engine
    from holon.api import SamplingParams
    e = engine(a.model, a.flags, max_seqs=a.seqs, max_model_len=33792)
    stages["startup"], out["stack_after_startup"] = outside(), stack()
    if a.requests:
        p = json.loads(pathlib.Path(a.requests).read_text())
        longest = max((r["ids"] for r in p["requests"]), key=len)
        for n in (1024, 4096, 32768):
            e.generate([longest[:n]], SamplingParams(temperature=0.0, max_tokens=64, ignore_eos=True))
        e.generate([longest[:1024]], SamplingParams(temperature=0.8, top_p=0.9, max_tokens=64, ignore_eos=True))
        stages["after_requests"], out["stack_after_requests"] = outside(), stack()
    maps = sum(m["device_mib"] for m in mappings)
    out |= {"stages_mib": stages, "host_mappings": mappings, "split_mib": {
        "context": out.get("context_with_stack_set", out["context"]),
        "host_pool_mappings": maps,
        "rest_of_build": stages["built"] - out.get("context_with_stack_set", out["context"]) - maps,
        "profile": stages["profiled"] - stages["built"],
        "graph_execs": stages["captured"] - stages["before_capture"],
        "requests": stages.get("after_requests", stages["startup"]) - stages["startup"]},
        "stack_mib_per_kib": props.multi_processor_count * props.max_threads_per_multi_processor >> 10}
    print(json.dumps(out), flush=True)
    e.close()


if __name__ == "__main__":
    main()
