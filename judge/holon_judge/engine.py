"""L2 candidate side: teacher-force the prompt set through a Holon engine via prompt logprobs.

The engine contract (S5): the fp32 log-softmax of the engine's own logits, top-k ids and logprobs plus the logprob
of the actual token, at every prompt position 1..n-1. `Logprobs` carries no tail mass, so it is derived from the
top-k (`tail_from_topk`).
"""
from __future__ import annotations

import collections
import time
from typing import Protocol

import torch
from holon.api import Logprobs, SamplingParams, TokenDelta

from . import gguf
from .golden import hardware, model_meta, resolve_model
from .placement import cpu_split, split_digest
from .recording import Prompt, PromptSet, Recording, tail_from_topk


class EngineLike(Protocol):
    def add(self, token_ids: list[int], params: SamplingParams) -> str: ...

    def step(self) -> list[TokenDelta]: ...

    def has_work(self) -> bool: ...


def score(engine: EngineLike, prompts: PromptSet, k: int = 20, meta: dict | None = None) -> Recording:
    """Run every prompt with `prompt_logprobs=k` and pack the answers as a recording comparable to the golden; with
    quantized kernels that count their calls per path and activation mode (keys `path/mode`: `kernels.ggml_cuda`'s
    dense and expert instances, found by `_counters`), the calls made while scoring (`calls`: Tier B's path and mode
    check; the engine's start-up — memory profiling, graphs — excluded) and its KV cache format (`kv`, unless the
    model dtype: a floor must emulate the same, `compare`). A quantized LM head's calls are counted apart
    (`head_calls`: their mode is checked, not their path): its row counts are not the step's but the prompt-logprob
    slices' and the sampled rows' (one per prompt finishing in the step, never scored), so a prefill run's head takes
    mmvq on a few rows whatever the trunk takes."""
    kv = getattr(getattr(engine, "cfg", None), "plugin_options", {}).get("kv.format")
    counters, head = _counters(engine), _counters(engine, head=True)
    # each ggml qlinear's mmvq limit, by shape (D2's per-shape floors, `compare`)
    limits = {f"{k.name}/{k.k}x{k.out}": k.mmvq_rows for k in [*_kernels(engine), *_kernels(engine, head=True)]
              if isinstance(getattr(k, "mmvq_rows", None), int)}
    before, head_before = _total(counters), _total(head)
    split = cpu_split(engine) if hasattr(engine, "ctx") else None     # F5b: its floor must know the CPU's experts
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
    calls = {name: {key: n - start.get(key, 0) for key, n in sorted(_total(c).items())}
             for name, c, start in (("calls", counters, before), ("head_calls", head, head_before)) if c}
    return Recording.from_rows(rows, prompts.meta() | {"source": "holon", "wall_s": wall} | (meta or {}) | calls
                               | ({"kv": kv} if kv else {}) | ({"mmvq_rows": limits} if limits else {})
                               | ({"cpu_split": split_digest(split)} if split else {}))


def _total(counters: list[dict[str, int]]) -> dict[str, int]:
    """The calls counted so far per `path/mode`, over every counter."""
    total: dict[str, int] = {}
    for c in counters:
        for key, n in c.items():
            total[key] = total.get(key, 0) + n
    return total


def _counters(engine: EngineLike, head: bool = False) -> list[dict[str, int]]:
    return [k.calls for k in _kernels(engine, head)]


def _kernels(engine: EngineLike, head: bool = False) -> list:
    """The engine's quantized kernels that count their calls (none for other engines or kernels): every object with a
    `calls` Counter held by an attribute of a module, or of a plain object a module holds (a linear's quantized weight
    holds its kernel), each once; those of the model's `lm_head` if `head`, else the others'."""
    model = getattr(getattr(engine, "executor", None), "model", None)
    lm_head = getattr(model, "lm_head", None)
    in_head = {id(m) for m in lm_head.modules()} if isinstance(lm_head, torch.nn.Module) else set()
    found: dict[int, object] = {}

    def visit(obj, depth: int) -> None:
        if isinstance(getattr(obj, "calls", None), collections.Counter):
            found[id(obj.calls)] = obj
        elif depth and hasattr(obj, "__dict__") and not isinstance(obj, (torch.nn.Module, torch.Tensor)):
            for value in vars(obj).values():
                visit(value, depth - 1)

    for m in [] if model is None else model.modules():
        if (id(m) in in_head) != head:
            continue
        for value in vars(m).values():
            visit(value, 1)
    return list(found.values())


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
                state_memory: int | None = None, max_seqs: int = 64, kv: str | None = None,
                act: str | None = None) -> Recording:
    """Build a Holon engine for `model` (hub id, checkpoint directory or .gguf file; `source` names the weight source,
    else it is detected) and score `prompts` through it, whole prompts per step unless `max_step_tokens` says less
    (GGUF Tier B's decode run: steps small enough that every quantized matmul takes mmvq). On the CPU (Tier A)
    `state_memory` sizes the state pools, which only a GPU can measure. `kv` is the KV cache format (the
    deployment's `kv.format` option; None: the model dtype), `act` the quantized weights' `quant.act` (None: its
    default, docs/dev/m06-lowp.md D3)."""
    from holon.core.engine import Engine, EngineConfig

    path = resolve_model(model)
    max_len = max(len(p.ids) for p in prompts.prompts) + 1     # room for the one sampled token
    options = {key: value for key, value in (("kv.format", kv), ("quant.act", act)) if value}
    engine = Engine(EngineConfig(model=str(path), device=device, dtype=dtype, max_model_len=max_len,
                                 max_step_tokens=max_step_tokens or max(max_len, 4096), source=source,
                                 state_memory=state_memory, max_seqs=max_seqs,
                                 plugin_options=options))
    meta = (gguf.file_meta(path) if path.suffix == ".gguf" else model_meta(path)) | {
        "weight_source": source or engine.source.kind, "max_step_tokens": engine.cfg.max_step_tokens,
        "gpu_class": engine.ctx.fingerprint.gpu,
        "dtype": str(dtype).removeprefix("torch."), "device": torch.device(device).type,
        "hardware": hardware(torch.device(device)), "torch": torch.__version__}
    return score(engine, prompts, k, meta)
