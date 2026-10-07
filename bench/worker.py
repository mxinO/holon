"""Replay traces on one backend, inside that backend's own environment; writes raw results as JSON.

Started by `run.py` with the backend's venv python (each baseline has its own torch stack), so this file and the
driver import nothing outside the stdlib and the backend's engine (and torch, under --blocking-sync).
"""
from __future__ import annotations

import argparse
import ctypes
import importlib
import importlib.metadata
import json
import pathlib
import time

import workloads
from driver import ROW, replay, warmup_sessions

CU_CTX_SCHED_BLOCKING_SYNC = 0x4


def blocking_sync() -> None:
    """Make every CUDA wait in this process sleep instead of spin, so process CPU time counts host work only.

    Two switches: the primary context's scheduling flag (set through the driver API before any framework creates
    the context) covers device/stream syncs and synchronous copies; event waits ignore it and spin unless the
    event was created blocking, so torch's Event defaults to blocking from here on (all engines measured use torch).
    """
    cuda = ctypes.CDLL("libcuda.so.1")
    dev = ctypes.c_int()
    for call, *call_args in (("cuInit", 0), ("cuDeviceGet", ctypes.byref(dev), 0),
                             ("cuDevicePrimaryCtxSetFlags_v2", dev, CU_CTX_SCHED_BLOCKING_SYNC)):
        if rc := getattr(cuda, call)(*call_args):
            raise RuntimeError(f"{call} failed with CUresult {rc}")

    import torch

    class BlockingEvent(torch.cuda.Event):
        def __new__(cls, *args: object, **kwargs: object) -> BlockingEvent:
            if len(args) < 2:
                kwargs.setdefault("blocking", True)
            return super().__new__(cls, *args, **kwargs)

    torch.cuda.Event = torch.cuda.streams.Event = BlockingEvent


def _version(dist: str) -> str | None:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--traces", nargs="+", required=True)
    ap.add_argument("--max-batch", type=int, required=True)
    ap.add_argument("--max-step-tokens", type=int, required=True)
    ap.add_argument("--blocking-sync", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.blocking_sync:
        blocking_sync()

    traces = {pathlib.Path(p).stem: workloads.load(p) for p in args.traces}
    longest = max(workloads.context_len(s) for sessions in traces.values() for s in sessions)
    mod = importlib.import_module(f"backends.{args.backend}")
    t = time.perf_counter()
    backend = mod.Backend(args.model, args.max_batch, -(-(longest + 1) // 256) * 256, args.max_step_tokens)
    load_s = time.perf_counter() - t

    cases = []
    for name, sessions in traces.items():
        # Salts name the trace: prompts of one trace never prefix-match another trace's or the warmup's.
        replay(backend, warmup_sessions(sessions), salt=f"{name}/warmup")
        done, steps = replay(backend, sessions, salt=f"{name}/run")
        done.sort(key=lambda r: (r.sid, r.turn))
        cases.append({"trace": name, "requests": [r.row() for r in done], "steps": steps})
        print(f"{args.backend} {name}: {len(done)} requests, {len(steps)} steps", flush=True)

    info = backend.info() | {"load_s": round(load_s, 1), "blocking_sync": args.blocking_sync,
                             "packages": {p: _version(p) for p in mod.PACKAGES}}
    pathlib.Path(args.out).write_text(json.dumps({"backend": args.backend, "info": info, "columns": ROW,
                                                  "cases": cases}, default=str))


if __name__ == "__main__":
    main()
