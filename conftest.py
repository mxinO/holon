"""Session-wide test environment, set before any test module initializes CUDA (docs/dev/release.md §22).

- The suite sharded over idle GPUs with pytest-xdist: with several GPUs in CUDA_VISIBLE_DEVICES, worker `gw<n>`
  keeps the n-th of them (cycling), so each worker has one GPU to itself, and its share of the CPUs for torch's
  threads (the CPU-engine tests ran up to 30× slower with every worker using all of them):

      CUDA_VISIBLE_DEVICES=0,1,2,3 pytest -m "" -n 4 --dist loadfile

  With fewer GPUs listed than workers, workers share GPUs (warned). A worker keeps its session fixtures, e.g.
  plugins/tests' shared Qwen3-0.6B engine (≈ 2 GB of VRAM), until it ends, in every worker that ran a test using it.

- FlashInfer's JIT modules cached per environment: their build files name the environment's paths, so environments
  sharing the home directory's cache (one per worktree) rebuild every module the other one built last."""
import importlib
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


@pytest.fixture(scope="module", autouse=True)
def _prebuilt(request):
    """The engine never compiles (a missing prebuilt module raises `HostError` naming the add-on and `holon
    prebuild`), so a test module lists the native extensions it needs in `PREBUILT` (e.g. `("expert_io",)`) and any
    missing one is built here for the visible GPU's SM, into the user cache, before its tests run: a fresh cache does
    not fail them."""
    names = getattr(request.module, "PREBUILT", ())
    if not names:
        return
    import torch
    if not torch.cuda.is_available():
        return
    major, minor = torch.cuda.get_device_capability()
    build_missing(names, 10 * major + minor)


def build_missing(names, sm: int) -> None:
    """Build each of the extensions `names` whose `load` finds no module for `sm` (a `HostError`); ggml_cpu, keyed by
    CPU tier rather than SM, has its missing tiers built by its prebuild."""
    from holon.api import HostError

    for name in names:
        if name == "ggml_cpu":
            importlib.import_module("holon_plugins.kernels.ggml_cpu.prebuild").main([])
            continue
        ext = importlib.import_module(f"holon_plugins.kernels.{name}.extension")
        try:
            ext.load(sm)
        except HostError:
            ext.build(sm, os.cpu_count()) if name == "ggml_cuda" else ext.build(sm)
