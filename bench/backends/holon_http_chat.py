"""Holon behind its chat endpoint (`frontends.http`), for the frontend's own cost (docs/dev/m05-grammar.md §11, step 5):
each trace prompt decoded to text (`backends.text`) as one user message offered tools, so the reasoning and tool
parsers read every stream (`tool_choice` auto: unconstrained, D8); greedy, `ignore_eos`. Its counters add the
frontend process's CPU time (`frontend_cpu_ms`; the engine runs in a child process) and its info each stream's
largest gap between events. `HOLON_SERVE_ARGS` adds engine flags; `HOLON_TOKENIZER`: an HF checkpoint for the
trace text (beside a .gguf there is none)."""
from __future__ import annotations

import os
import shlex
import statistics
import sys

from backends import http, text
from backends.holon_http import served

PACKAGES = ("torch", "triton", "flashinfer-python")
TOOLS = [{"type": "function", "function": {"name": name, "description": f"{name} a file or command",
                                           "parameters": {"type": "object", "properties": {
                                               "path": {"type": "string"}, "content": {"type": "string"}},
                                               "required": ["path"]}}}
         for name in ("read", "write", "edit", "bash", "glob", "grep")]


class Backend(http.Backend):
    def __init__(self, model: str, max_batch: int, max_model_len: int, max_step_tokens: int) -> None:
        port = http.free_port()
        self.flags = ["--max-seqs", str(max_batch), "--max-model-len", str(text.headroom(max_model_len) + 512),
                      "--max-step-tokens", str(max_step_tokens), *shlex.split(os.environ.get("HOLON_SERVE_ARGS", ""))]
        server = http.Server([sys.executable, "-m", "holon", "serve", model, "--port", str(port), *self.flags], port)
        self.prompts = text.Prompts(os.environ.get("HOLON_TOKENIZER", model))   # a .gguf has no tokenizer.json
        super().__init__(server, "/v1/chat/completions", self._body, self._parse)

    def _body(self, prompt: list[int], max_tokens: int) -> dict:
        return {"model": "m", "messages": [{"role": "user", "content": self.prompts.tok.decode(prompt)}],
                "tools": TOOLS, "max_tokens": max_tokens, "temperature": 0.0, "ignore_eos": True, "stream": True,
                "stream_options": {"include_usage": True}}

    def _parse(self, event: dict, state: dict) -> tuple[list[int], bool]:
        choices, usage, n = event.get("choices") or [], event.get("usage"), 0
        if choices and any(choices[0].get("delta", {}).get(k) for k in ("content", "reasoning_content",
                                                                         "tool_calls")):
            n = 1
        if usage:                                   # the last chunk: the server's own count settles the total
            n = max(n, usage["completion_tokens"] - state.get("seen", 0))
        state["seen"] = state.get("seen", 0) + n
        return [self.prompts.stand_in] * n, bool(usage)

    def info(self) -> dict:
        gaps = sorted(g * 1e3 for g in self.gaps)
        return {"config": {"flags": " ".join(self.flags)}, "served": served(self.server), "max_event_gap_ms": {
            "streams": len(gaps), "p50": round(statistics.median(gaps), 1),
            "p99": round(gaps[int(0.99 * (len(gaps) - 1))], 1), "max": round(gaps[-1], 1)} if gaps else None}

    def stats(self) -> dict[str, int]:
        return {"frontend_cpu_ms": round(self.server.cpu_ms())}
