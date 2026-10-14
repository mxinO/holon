"""llama.cpp's `llama-server` (M0.1 design §4.8), matched to a Holon lock (release design §5.2): one slot per running
sequence (`-np`) over a unified KV cache (`-kvu`) of `kv_tokens` (Holon's measured KV tokens; else `LLAMA_SERVER_CTX`,
else max running sequences × max context) in the lock's KV format (`-ctk/-ctv`); `-b` = the step budget, `-ub 512`.
The binary is the gate pin's build from `setup_baselines.sh` unless `LLAMA_SERVER` names another (e.g. the upstream
build); `LLAMA_SERVER_ARGS` adds flags (e.g. `-cram 0`). Token-id prompts, `ignore_eos`, `n_predict`, greedy unless
a temperature is given; prompt cache at its default. Every request's server-side timings are kept (drafts and
accepted drafts are the backend's stats).

Tokens are counted by the server's `tokens_predicted`: a token whose text ends in an incomplete UTF-8 character is
not streamed (llama.cpp holds its bytes back, `server-context.cpp` `process_token`), and the next chunk carries only
its own id. A held-back token therefore counts when its bytes reach the client, and its id — never sent in stream
mode — is stood in for by the id that arrives; only a multi-turn trace's history sees the stand-in (about 2 of 126
agent requests per round), costing llama.cpp at most the re-prefill of that turn's tail."""
from __future__ import annotations

import functools
import os
import pathlib
import shlex
import subprocess

from backends import http

PACKAGES = ()
GATE = pathlib.Path(__file__).resolve().parents[1] / ".envs" / "llama.cpp-gate" / "bin" / "llama-server"
KV_TYPES = {None: "f16", "q8_0": "q8_0"}       # Holon's kv.format -> llama.cpp's cache type (bf16 KV ~ f16)


class Backend(http.Backend):
    def __init__(self, model: str, max_batch: int, max_model_len: int, max_step_tokens: int, *,
                 temperature: float = 0.0, kv_tokens: int | None = None, kv_format: str | None = None) -> None:
        binary = os.environ.get("LLAMA_SERVER", str(GATE))
        ctx = kv_tokens or int(os.environ.get("LLAMA_SERVER_CTX", max_batch * max_model_len))
        if kv_format not in KV_TYPES:
            raise ValueError(f"llama.cpp has no KV cache type for kv.format={kv_format}")
        kv = KV_TYPES[kv_format]
        port = http.free_port()
        self.flags = ["-ngl", "99", "-fa", "on", "-kvu", "-np", str(max_batch), "-c", str(ctx), "-b",
                      str(max_step_tokens), "-ub", "512", "-ctk", kv, "-ctv", kv,
                      *shlex.split(os.environ.get("LLAMA_SERVER_ARGS", ""))]
        self.version = version(binary)
        server = http.Server([binary, "-m", model, "--host", "127.0.0.1", "--port", str(port), "--no-webui",
                              *self.flags], port)
        super().__init__(server, "/completion", functools.partial(_body, temperature=temperature), _parse)

    def stats(self) -> dict[str, int]:
        """Drafts and accepted drafts so far (the finished requests' timings; 0 without speculation)."""
        timings = [e.get("timings", {}) for e in self.finals]
        return {"spec_drafted": sum(t.get("draft_n", 0) for t in timings),
                "spec_accepted": sum(t.get("draft_n_accepted", 0) for t in timings)}

    def info(self) -> dict:
        timings = [e.get("timings", {}) for e in self.finals]
        return {"config": {"flags": " ".join(self.flags)}, "version": self.version,
                "server_timings": {k: sum(t.get(k, 0) for t in timings)
                                   for k in ("prompt_n", "prompt_ms", "predicted_n", "predicted_ms", "cache_n")}}


def version(binary: str) -> str:
    """The build's "version: N (commit)" line (the rest names the compiler and devices)."""
    out = subprocess.run([binary, "--version"], capture_output=True, text=True).stderr
    return next((line for line in out.splitlines() if line.startswith("version:")), out.strip())


def _body(prompt: list[int], max_tokens: int, temperature: float) -> dict:
    return {"prompt": prompt, "n_predict": max_tokens, "temperature": temperature, "top_k": 0, "top_p": 1.0,
            "min_p": 0.0, "ignore_eos": True, "stream": True, "return_tokens": True, "cache_prompt": True}


def _parse(event: dict, state: dict) -> tuple[list[int], bool]:
    seen, now = state.get("seen", 0), event.get("tokens_predicted", 0)
    state["seen"] = max(seen, now)
    toks = event.get("tokens", [])
    missing = now - seen - len(toks)
    return ([toks[-1] if toks else 0] * missing + toks) if missing > 0 else toks, bool(event.get("stop"))
