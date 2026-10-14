"""First start of `holon serve` (release design D1): what a cold start compiles and how long it takes, against the
supervisor's `Policy.start_timeout`.

Native code comes from `holon prebuild`, run beforehand with the same cache variables (FLASHINFER_WORKSPACE_BASE =
<caches>/fi, XDG_CACHE_HOME = <caches>/xdg). Per model, `holon serve` starts in a fresh process with empty Triton and
CuTe DSL caches; recorded: the time until it serves (/health), the first chat completion's latency, and what each JIT
cache gained — FlashInfer's modules and the GGUF extension must gain nothing.

Usage (from the repository root, inside Holon's .venv):
  python bench/cold_start.py --gpu 2 --caches /tmp/caches <model>... [--out evidence/cold_start_<class>_<date>.json]
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
import urllib.request

from run import ROOT, git, gpu_class, model_id


def entries(path: pathlib.Path, pattern: str) -> set[str]:
    return {str(p.relative_to(path)) for p in path.glob(pattern)} if path.is_dir() else set()


def watched(caches: pathlib.Path) -> dict[str, set[str]]:
    return {"flashinfer": entries(caches / "fi", ".cache/flashinfer/*/*/cached_ops/*"),
            "ggml_cuda": entries(caches / "xdg", "holon/ggml_cuda/*/ext/*.so"),
            "triton": entries(caches / "run" / "triton", "*"), "cute_dsl": entries(caches / "run" / "cute", "**/*")}


def request(port: int, path: str, body: dict | None = None, timeout: float = 5) -> dict:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def start(model: str, gpu: int, caches: pathlib.Path, port: int, timeout: float) -> dict:
    shutil.rmtree(caches / "run", ignore_errors=True)
    before = watched(caches)
    env = os.environ | {"CUDA_VISIBLE_DEVICES": str(gpu), "FLASHINFER_WORKSPACE_BASE": str(caches / "fi"),
                        "XDG_CACHE_HOME": str(caches / "xdg"), "TRITON_CACHE_DIR": str(caches / "run" / "triton"),
                        "CUTE_DSL_CACHE_DIR": str(caches / "run" / "cute")}
    command = [sys.executable, "-m", "holon", "serve", model, "--port", str(port)]
    t0 = time.monotonic()
    proc = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"holon serve exited: {proc.stdout.read()[-3000:]}")
            if time.monotonic() - t0 > timeout:
                raise TimeoutError(f"not serving after {timeout} s")
            try:
                request(port, "/health", timeout=1)
                break
            except OSError:
                time.sleep(1)
        ready = time.monotonic() - t0
        name = request(port, "/v1/models")["data"][0]["id"]
        t1 = time.monotonic()
        request(port, "/v1/chat/completions", {"model": name, "max_tokens": 32, "temperature": 0,
                                               "messages": [{"role": "user", "content": "Count from 1 to 20."}]}, 600)
        first = time.monotonic() - t1
        metrics = request(port, "/metrics")
    finally:
        proc.terminate()
        log = proc.communicate(timeout=60)[0]
    after = watched(caches)
    return {"model": model_id(model), "serving_s": round(ready, 1), "first_completion_s": round(first, 2),
            "start_s": metrics.get("start_s"), "plan_id": metrics.get("plan_id"),
            "compiled": {k: len(after[k] - before[k]) for k in after},
            "log": [line for line in log.splitlines() if line.startswith("holon:")]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+")
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--caches", type=pathlib.Path, required=True)
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args()
    from holon.core.supervisor import Policy

    rows = []
    for model in a.models:
        rows.append(start(model, a.gpu, a.caches, a.port, a.timeout))
        print(json.dumps(rows[-1]), flush=True)
    record = {"question": "What does a cold `holon serve` compile, and how long until it serves (D1)?",
              "date": datetime.date.today().isoformat(), "commit": git("rev-parse", "HEAD"),
              "gpu_class": gpu_class(a.gpu), "start_timeout_s": Policy().start_timeout,
              "command": f"python bench/cold_start.py --gpu {a.gpu} --caches <dir> "
                         + " ".join(model_id(m) for m in a.models),
              "method": __doc__.split("\n\n")[1].replace("\n", " "), "results": rows}
    if a.out:
        a.out.write_text(json.dumps(record, indent=1) + "\n")
        print(f"wrote {a.out.relative_to(ROOT) if a.out.is_absolute() else a.out}")


if __name__ == "__main__":
    main()
