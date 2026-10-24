"""Session-wide test environment, set before any test module initializes CUDA (docs/release.md §22).

- The suite sharded over idle GPUs with pytest-xdist: with several GPUs in CUDA_VISIBLE_DEVICES, worker `gw<n>`
  keeps the n-th of them (cycling), so each worker has one GPU to itself, and its share of the CPUs for torch's
  threads (the CPU-engine tests ran up to 30× slower with every worker using all of them):

      CUDA_VISIBLE_DEVICES=0,1,2,3 pytest -m "" -n 4 --dist loadfile

  With fewer GPUs listed than workers, workers share GPUs (warned). A worker keeps its session fixtures, e.g.
  plugins/tests' shared Qwen3-0.6B engine (≈ 2 GB of VRAM), until it ends, in every worker that ran a test using it.

- FlashInfer's JIT modules cached per environment: their build files name the environment's paths, so environments
  sharing the home directory's cache (one per worktree) rebuild every module the other one built last."""
import os
import sys

import pytest

os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", sys.prefix)
if worker := os.environ.get("PYTEST_XDIST_WORKER"):
    workers = int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1"))
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, len(os.sched_getaffinity(0)) // workers)))
    if "," in (gpus := os.environ.get("CUDA_VISIBLE_DEVICES", "")):
        ids = gpus.split(",")
        os.environ["CUDA_VISIBLE_DEVICES"] = ids[int(worker.removeprefix("gw")) % len(ids)]


def pytest_configure(config: pytest.Config) -> None:
    workers = getattr(config.option, "numprocesses", None)
    gpus = [g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g]
    if "PYTEST_XDIST_WORKER" not in os.environ and isinstance(workers, int) and workers > len(gpus):
        config.issue_config_time_warning(pytest.PytestConfigWarning(
            f"{workers} workers and {len(gpus) or 'no'} GPUs listed in CUDA_VISIBLE_DEVICES: workers share GPUs "
            "(list one idle GPU per worker)"), stacklevel=2)
