"""L2 candidate side: teacher-force the prompt set through a Holon engine via prompt logprobs.

The engine contract (S5): the fp32 log-softmax of the engine's own logits, top-k ids and logprobs plus the logprob
of the actual token, at every prompt position 1..n-1. `Logprobs` carries no tail mass, so it is derived from the
top-k (`tail_from_topk`).
"""
from __future__ import annotations

import time
from typing import Protocol

import torch
from holon.api import Logprobs, SamplingParams, TokenDelta

from . import gguf
from .golden import hardware, model_meta, resolve_model
from .recording import Prompt, PromptSet, Recording, tail_from_topk


class EngineLike(Protocol):
    def add(self, token_ids: list[int], params: SamplingParams) -> str: ...

    def step(self) -> list[TokenDelta]: ...

    def has_work(self) -> bool: ...


def score(engine: EngineLike, prompts: PromptSet, k: int = 20, meta: dict | None = None) -> Recording:
    """Run every prompt with `prompt_logprobs=k` and pack the answers as a recording comparable to the golden; with
    expert kernels that count their calls per path, the calls made while scoring (`expert_calls`: Tier B's path
    check; the engine's start-up — memory profiling, graphs — excluded)."""
    counters = _expert_counters(engine)
    before = [dict(c) for c in counters]
    params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=k)
    index = {engine.add(p.ids, params): i for i, p in enumerate(prompts.prompts)}
    parts: list[list[Logprobs]] = [[] for _ in prompts.prompts]
    t0 = time.perf_counter()
    while engine.has_work():
        for delta in engine.step():
            # Chunked prefill may deliver a prompt's logprobs over several deltas; they arrive in position order.
            if delta.prompt_logprobs is not None:
                parts[index[delta.rid]].append(delta.prompt_logprobs)
    rows = [_rows(p, got, k) for p, got in zip(prompts.prompts, parts, strict=True)]
    wall = round(time.perf_counter() - t0, 2)
    calls = {"expert_calls": {path: sum(c[path] - b[path] for c, b in zip(counters, before, strict=True))
                              for path in counters[0]}} if counters else {}
    return Recording.from_rows(rows, prompts.meta() | {"source": "holon", "wall_s": wall} | (meta or {}) | calls)


def _expert_counters(engine: EngineLike) -> list[dict[str, int]]:
    """The per-path call counters of the engine's expert kernels (none for other engines or kernels)."""
    model = getattr(getattr(engine, "executor", None), "model", None)
    return [] if model is None else [m._kernel.calls for m in model.modules()
                                     if hasattr(getattr(m, "_kernel", None), "calls")]


def _rows(prompt: Prompt, parts: list[Logprobs], k: int) -> tuple[torch.Tensor, ...]:
    token = [x for lp in parts for x in lp.token]
    top_ids = [x[:k] for lp in parts for x in lp.top_ids]
    top = [x[:k] for lp in parts for x in lp.top]
    n = len(prompt.ids) - 1
    if not len(token) == len(top_ids) == len(top) == n or any(len(x) != k for x in top_ids + top):
        raise ValueError(f"{prompt.id}: expected top-{k} prompt logprobs at {n} positions, got {len(token)}")
    top_lp = torch.tensor(top, dtype=torch.float32)
    return torch.tensor(top_ids, dtype=torch.int32), top_lp, torch.tensor(token), tail_from_topk(top_lp)


def score_model(model: str, prompts: PromptSet, dtype: torch.dtype = torch.bfloat16, device: str = "cuda",
                k: int = 20, source: str | None = None, max_step_tokens: int | None = None,
                state_memory: int | None = None, max_seqs: int = 64, kv: str | None = None) -> Recording:
    """Build a Holon engine for `model` (hub id, checkpoint directory or .gguf file; `source` names the weight source,
    else it is detected) and score `prompts` through it, whole prompts per step unless `max_step_tokens` says less
    (GGUF Tier B's decode run: steps small enough that every quantized matmul takes mmvq). On the CPU (Tier A)
    `state_memory` sizes the state pools, which only a GPU can measure. `kv` is the KV cache format (the
    deployment's `kv.format` option; None: the model dtype)."""
    from holon.core.engine import Engine, EngineConfig

    path = resolve_model(model)
    max_len = max(len(p.ids) for p in prompts.prompts) + 1     # room for the one sampled token
    engine = Engine(EngineConfig(model=str(path), device=device, dtype=dtype, max_model_len=max_len,
                                 max_step_tokens=max_step_tokens or max(max_len, 4096), source=source,
                                 state_memory=state_memory, max_seqs=max_seqs,
                                 plugin_options={"kv.format": kv} if kv else {}))
    meta = (gguf.file_meta(path) if path.suffix == ".gguf" else model_meta(path)) | {
        "weight_source": source or engine.source.kind, "max_step_tokens": engine.cfg.max_step_tokens,
        "dtype": str(dtype).removeprefix("torch."), "device": torch.device(device).type,
        "hardware": hardware(torch.device(device)), "torch": torch.__version__, **({"kv": kv} if kv else {})}
    return score(engine, prompts, k, meta)
