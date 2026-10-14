"""SGLang behind its native server (`sglang.launch_server`, in its own venv from `setup_baselines.sh`; `SGLANG_PYTHON`
names another), matched to a Holon lock (release design §5.2): `--max-running-requests` = the max batch,
`--chunked-prefill-size` = the step budget, `--max-total-tokens` = Holon's measured KV tokens, the lock's KV format
(`--kv-cache-dtype`), `--mem-fraction-static 0.9`; radix (prefix) cache at its default. Token-id prompts to
`/generate` (`input_ids`, no tokenizer process: `--skip-tokenizer-init`), streamed, `ignore_eos`, greedy unless a
temperature is given. `SGLANG_ARGS` adds flags. HF checkpoints only (no GGUF comparison: spike S2)."""
from __future__ import annotations

import os
import pathlib
import shlex
import subprocess

from backends import http

PACKAGES = ()
PYTHON = pathlib.Path(__file__).resolve().parents[1] / ".envs" / "sglang" / "bin" / "python"
KV_DTYPES = {None: "auto", "fp8": "fp8_e4m3"}     # Holon's kv.format -> SGLang's --kv-cache-dtype


class Backend(http.Backend):
    def __init__(self, model: str, max_batch: int, max_model_len: int, max_step_tokens: int, *,
                 temperature: float = 0.0, kv_tokens: int | None = None, kv_format: str | None = None) -> None:
        if kv_format not in KV_DTYPES:
            raise ValueError(f"SGLang has no KV cache dtype for kv.format={kv_format}")
        python = os.environ.get("SGLANG_PYTHON", str(PYTHON))
        port = http.free_port()
        self.flags = ["--max-running-requests", str(max_batch), "--chunked-prefill-size", str(max_step_tokens),
                      *(["--max-total-tokens", str(kv_tokens)] if kv_tokens else []),
                      "--kv-cache-dtype", KV_DTYPES[kv_format], "--mem-fraction-static", "0.9",
                      "--context-length", str(max_model_len), "--skip-tokenizer-init",
                      *shlex.split(os.environ.get("SGLANG_ARGS", ""))]
        self.version = subprocess.run([python, "-c", "import sglang; print(sglang.__version__)"],
                                      capture_output=True, text=True).stdout.strip()
        server = http.Server([python, "-m", "sglang.launch_server", "--model-path", model, "--host", "127.0.0.1",
                              "--port", str(port), *self.flags], port)   # /health: 503 until warmed up
        self.temperature = temperature
        super().__init__(server, "/generate", self._body, _parse)

    def _body(self, prompt: list[int], max_tokens: int) -> dict:
        return {"input_ids": prompt, "stream": True, "sampling_params": {
            "max_new_tokens": max_tokens, "temperature": self.temperature, "ignore_eos": True}}

    def info(self) -> dict:
        return {"config": {"flags": " ".join(self.flags)}, "version": self.version}


def _parse(event: dict, state: dict) -> tuple[list[int], bool]:
    """A chunk's new ids: `output_ids` is cumulative (or the delta, with incremental streaming), and
    `completion_tokens` counts them all, so the last `completion_tokens - seen` ids are new either way."""
    meta, ids = event["meta_info"], event.get("output_ids") or []
    new = meta["completion_tokens"] - state.get("seen", 0)
    state["seen"] = meta["completion_tokens"]
    return (ids[-new:] if new > 0 else []), meta.get("finish_reason") is not None
