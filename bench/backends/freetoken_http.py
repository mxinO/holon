"""FreeToken behind its OpenAI server (`ft serve`, in its venv from `setup_baselines.sh`; `FREETOKEN_PYTHON` names
another), text prompts
(`backends.text`: it takes no token ids); `--max-running-requests` = the max batch, `--max-prefill-length` = the step
budget, `--memory-ratio` 0.9 (Holon's fraction); `FREETOKEN_ARGS` adds flags. Its distributed rendezvous binds a fixed
port, so one FreeToken server at a time on a host."""
from __future__ import annotations

import http.client as client
import json
import os
import pathlib
import shlex
import time

from backends import http, text

PACKAGES = ()
PYTHON = pathlib.Path(__file__).resolve().parents[1] / ".envs" / "freetoken" / "bin" / "python"


class Backend(http.Backend):
    def __init__(self, model: str, max_batch: int, max_model_len: int, max_step_tokens: int, *,
                 temperature: float = 0.0, kv_tokens: int | None = None, kv_format: str | None = None) -> None:
        if kv_format is not None:
            raise ValueError(f"FreeToken is not matched to kv.format={kv_format} (its KV is the model dtype)")
        port = http.free_port()
        self.flags = ["--max-running-requests", str(max_batch), "--max-prefill-length", str(max_step_tokens),
                      *(["--num-tokens", str(kv_tokens // 64 * 64)] if kv_tokens else []),   # whole pages
                      "--max-seq-len-override", str(text.headroom(max_model_len)),
                      "--memory-ratio", "0.9", "--served-model-name", "m",
                      *shlex.split(os.environ.get("FREETOKEN_ARGS", ""))]
        server = http.Server([os.environ.get("FREETOKEN_PYTHON", str(PYTHON)), "-m", "freetoken", "--model-path", model,
                              "--host", "127.0.0.1", "--port", str(port), *self.flags], port)
        _wait_ok(port)                               # its /health answers 200 while still loading
        prompts = text.Prompts(model, temperature)
        super().__init__(server, "/v1/completions", prompts.body, prompts.parse)

    def info(self) -> dict:
        return {"config": {"flags": " ".join(self.flags)}}


def _wait_ok(port: int, timeout_s: float = 900) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        conn = client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/health")
        if json.loads(conn.getresponse().read()).get("status") == "ok":
            return
        time.sleep(1)
    raise TimeoutError("FreeToken not ready")
