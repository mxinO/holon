"""Holon behind its own OpenAI frontend (`python -m holon serve`, this checkout), for comparisons with servers
(M0.1 design §4.8): token-id prompts to /v1/completions, greedy, `ignore_eos`, token ids streamed back
(`return_token_ids`). `HOLON_SERVE_ARGS` adds engine flags (e.g. `--memory-fraction 0.98`)."""
from __future__ import annotations

import os
import shlex
import sys

from backends import http

PACKAGES = ("torch", "triton", "flashinfer-python")


class Backend(http.Backend):
    def __init__(self, model: str, max_batch: int, max_model_len: int, max_step_tokens: int) -> None:
        port = http.free_port()
        self.flags = ["--max-seqs", str(max_batch), "--max-model-len", str(max_model_len), "--max-step-tokens",
                      str(max_step_tokens), *shlex.split(os.environ.get("HOLON_SERVE_ARGS", ""))]
        server = http.Server([sys.executable, "-m", "holon", "serve", model, "--port", str(port), *self.flags], port)
        super().__init__(server, "/v1/completions", _body, _parse)

    def info(self) -> dict:
        return {"config": {"flags": " ".join(self.flags)}, "served": served(self.server)}

    def stats(self) -> dict[str, int]:
        """The engine's counters (/metrics): prefix hits, and real drafts and verify rows by drafts accepted; the
        frontend process's CPU time."""
        metrics = self.server.get("/metrics")
        return {k: v for k, v in metrics.items() if k == "prefix_tokens_saved" or k.startswith("spec_")} | {
            "frontend_cpu_ms": round(self.server.cpu_ms())}


def served(server: http.Server) -> dict:
    """What `holon serve` runs: the plan, the lock it started from (flags over it) and how the plan differs."""
    metrics = server.get("/metrics")
    return {key: metrics.get(key) for key in ("plan_id", "lock", "diff")}


def _body(prompt: list[int], max_tokens: int) -> dict:
    return {"prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0, "ignore_eos": True, "stream": True,
            "return_token_ids": True}


def _parse(event: dict, state: dict) -> tuple[list[int], bool]:
    (choice,) = event["choices"]
    return choice.get("token_ids", []), choice["finish_reason"] is not None
