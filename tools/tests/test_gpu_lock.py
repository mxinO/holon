"""The first lock of a real model on this GPU class (`propagate --new-model`), then a no-op re-run."""
import pathlib

import pytest
import torch
from holon.core import platform
from holon_tools import locks
from holon_tools.__main__ import main

SNAP = max(pathlib.Path.home().glob(".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*"), default=None)
GOLDENS = pathlib.Path(__file__).resolve().parents[2] / "judge" / "goldens" / "qwen3-0.6b"


@pytest.mark.skipif(SNAP is None or not torch.cuda.is_available(), reason="needs Qwen3-0.6B + GPU")
def test_first_lock_for_qwen3_0_6b(tmp_path, capsys):
    flags = [str(SNAP), "--plans", str(tmp_path), "--golden", str(GOLDENS / "v1-fp32.safetensors"),
             "--floor", str(GOLDENS / "v1-bfloat16-floor.safetensors")]
    assert main(["propagate", *flags, "--new-model", "--approve"]) == 0
    assert "PASS" in capsys.readouterr().out
    (lock,) = locks.heads(tmp_path)
    assert lock.plan.model_id.startswith("Qwen/Qwen3-0.6B@") and lock.snapshot and lock.sm > 0
    assert lock.plan.fingerprint_class == platform.fingerprint(torch.device("cuda")).cls
    assert main(["propagate", *flags]) == 0
    assert "verdict: unchanged" in capsys.readouterr().out
