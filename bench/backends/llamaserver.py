"""llama.cpp's `llama-server` (M0.1 design §4.8): `LLAMA_SERVER` names the binary; one slot per running sequence
over a unified KV cache (`-kvu`) of `LLAMA_SERVER_CTX` tokens (Holon's measured KV tokens; default: max running
sequences × max context); `-b` = the step budget; `LLAMA_SERVER_ARGS` adds flags (e.g. `-cram 0`, `-ub 512`).
Token-id prompts, greedy, `ignore_eos`, `n_predict`; prompt cache at its default. Every request's server-side
timings are kept."""
from __future__ import annotations

import os
import shlex
import subprocess

from backends import http

PACKAGES = ()


class Backend(http.Backend):
    def __init__(self, model: str, max_batch: int, max_model_len: int, max_step_tokens: int) -> None:
        binary = os.environ["LLAMA_SERVER"]
        ctx = int(os.environ.get("LLAMA_SERVER_CTX", max_batch * max_model_len))
        port = http.free_port()
        self.flags = ["-ngl", "99", "-fa", "on", "-kvu", "--parallel", str(max_batch), "-c", str(ctx), "-b",
                      str(max_step_tokens), *shlex.split(os.environ.get("LLAMA_SERVER_ARGS", ""))]
        self.version = subprocess.run([binary, "--version"], capture_output=True, text=True).stderr.strip()
        server = http.Server([binary, "-m", model, "--host", "127.0.0.1", "--port", str(port), "--no-webui",
                              *self.flags], port)
        super().__init__(server, "/completion", _body, _parse)

    def info(self) -> dict:
        timings = [e.get("timings", {}) for e in self.finals]
        return {"config": {"flags": " ".join(self.flags)}, "version": self.version,
                "server_timings": {k: sum(t.get(k, 0) for t in timings)
                                   for k in ("prompt_n", "prompt_ms", "predicted_n", "predicted_ms", "cache_n")}}


def _body(prompt: list[int], max_tokens: int) -> dict:
    return {"prompt": prompt, "n_predict": max_tokens, "temperature": 0.0, "ignore_eos": True, "stream": True,
            "return_tokens": True, "cache_prompt": True}


def _parse(event: dict) -> tuple[list[int], bool]:
    return event.get("tokens", []), bool(event.get("stop"))
