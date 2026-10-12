"""Holon behind its OpenAI frontend driven like `freetoken_http` (`backends.text`: text prompts, the server's own
token count), for comparisons with servers that take no token ids. `HOLON_SERVE_ARGS` adds engine flags."""
from __future__ import annotations

import os
import shlex
import sys

from backends import http, text

PACKAGES = ("torch", "triton", "flashinfer-python")


class Backend(http.Backend):
    def __init__(self, model: str, max_batch: int, max_model_len: int, max_step_tokens: int) -> None:
        port = http.free_port()
        self.flags = ["--max-seqs", str(max_batch), "--max-model-len", str(max_model_len), "--max-step-tokens",
                      str(max_step_tokens), *shlex.split(os.environ.get("HOLON_SERVE_ARGS", ""))]
        server = http.Server([sys.executable, "-m", "holon", "serve", model, "--port", str(port), *self.flags], port)
        prompts = text.Prompts(model)
        super().__init__(server, "/v1/completions", prompts.body, prompts.parse)

    def info(self) -> dict:
        return {"config": {"flags": " ".join(self.flags)}}
