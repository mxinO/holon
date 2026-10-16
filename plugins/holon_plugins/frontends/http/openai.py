"""OpenAI chat and text completions (`/v1/chat/completions`, `/v1/completions`; bodies in `schemas.py`): choices and
stream chunks — chat with reasoning and tool calls, streamed as `tool_calls` deltas — logprobs, usage (with the
prompt tokens the prefix cache served, `prompt_tokens_details.cached_tokens`)."""
from __future__ import annotations

import itertools
import json
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import aclosing
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
from holon.api import (
    EngineClient,
    FrontendBundle,
    Logprobs,
    ParseEvent,
    Reasoning,
    Text,
    Tokenizer,
    ToolArgsDelta,
    ToolCallStart,
)

from .pipeline import Blocks, Chat, Efforts, pieces
from .schemas import ChatRequest, CompletionRequest, GenerationRequest
from .stream import Piece, cached, generate
from .web import BadRequest, EventStream, collect, failure, rejected, unless_disconnected

# A token's logprob row: its text, its logprob (None for the unscored first prompt token) and the ranked top-k.
Row = tuple[str, float | None, list[tuple[str, float]] | None]
MASKED = -9999.0       # the logprob reported for a token a constrained request may not take (−inf: not JSON)


def router(client: EngineClient, bundle: FrontendBundle, model_id: str, efforts: Efforts) -> APIRouter:
    r = APIRouter()
    tok = bundle.tokenizer

    @r.post("/v1/completions")
    async def completions(req: CompletionRequest, request: Request) -> Response:
        if not req.prompts:
            raise BadRequest("empty prompt list")
        if req.stream and len(req.prompts) > 1:
            raise BadRequest("streaming takes a single prompt")
        prompts = [(p, tok.encode(p)) if isinstance(p, str) else (tok.decode(p), p) for p in req.prompts]
        rid, params = f"cmpl-{uuid.uuid4().hex}", req.params()
        rids = [rid] if len(prompts) == 1 else [f"{rid}-{i}" for i in range(len(prompts))]
        gens = [generate(client, tok, ids, params, r, req.stops) for (_, ids), r in zip(prompts, rids, strict=True)]
        fmt = Completion(tok, prompts, req.n, req.echo)
        return await _respond(request, req, fmt, gens, rid, sum(len(ids) for _, ids in prompts), params.n)

    @r.post("/v1/chat/completions")
    async def chat(req: ChatRequest, request: Request) -> Response:
        prepared = Chat(bundle, req.conversation(), efforts)
        rid, params = f"chatcmpl-{uuid.uuid4().hex}", req.params(prepared.ext)
        gen = pieces(client, bundle, prepared, params, rid, req.stops)
        return await _respond(request, req, ChatCompletion(tok), [gen], rid, len(prepared.ids), params.n)

    async def _respond(request: Request, req: GenerationRequest, fmt: ChatCompletion | Completion,
                       gens: list[AsyncIterator[Piece]], rid: str, n_prompt: int, n: int) -> Response:
        envelope = {"id": rid, "object": fmt.object, "created": int(time.time()), "model": model_id}
        if req.stream:
            # Wait for the first piece before answering, so that a rejected request gets its error status.
            first = await unless_disconnected(request, anext(gens[0]))
            if first is None:
                await gens[0].aclose()
                return Response(status_code=499)
            if first.finish_reason == "error":
                raise rejected(first)
            return EventStream(_events(fmt, {**envelope, "object": fmt.chunk_object}, first, gens[0], n,
                                       n_prompt if req.include_usage else None, req.return_token_ids))
        results = await unless_disconnected(request, collect(gens))
        if results is None:
            return Response(status_code=499)
        failed = failure(results) is not None
        merged = [merge([p for p in pieces if p.index == k], k, failed) for pieces in results for k in range(n)]
        choices = [fmt.choice(i, p) | ({"token_ids": p.token_ids} if req.return_token_ids else {})
                   for i, p in enumerate(merged)]
        n_completion = sum(len(p.token_ids) for pieces in results for p in pieces)
        n_cached = sum(cached(pieces) for pieces in results)
        return JSONResponse({**envelope, "choices": choices, "usage": usage(n_prompt, n_completion, n_cached)})

    return r


async def _events(fmt: ChatCompletion | Completion, envelope: dict[str, Any], first: Piece,
                  gen: AsyncIterator[Piece], n: int, n_prompt: int | None, token_ids: bool) -> AsyncIterator[str]:
    """Server-sent events of one request; the final usage chunk only if asked for (`n_prompt` given); with
    `token_ids`, a chunk for every piece, carrying its token ids. A failure after the first piece finishes every
    choice still running with "error"."""
    seen: set[int] = set()
    finished: set[int] = set()
    n_completion = n_cached = 0
    async with aclosing(gen):
        p: Piece | None = first
        while p is not None:
            if p.finish_reason == "error":
                for k in sorted(set(range(n)) - finished):
                    yield _event({**envelope, "choices": [fmt.failed(k)]})
                break
            n_completion += len(p.token_ids)
            n_cached = cached([p]) or n_cached
            if p.finish_reason:
                finished.add(p.index)
            chunks = fmt.chunks(p, p.index not in seen, token_ids)
            seen.add(p.index)
            for i, chunk in enumerate(chunks):
                yield _event({**envelope, "choices": [chunk | ({"token_ids": p.token_ids} if token_ids and not i
                                                                else {})]})
            p = await anext(gen, None)
    if n_prompt is not None:
        yield _event({**envelope, "choices": [], "usage": usage(n_prompt, n_completion, n_cached)})
    yield "data: [DONE]\n\n"


def _event(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def merge(pieces: list[Piece], index: int, failed: bool) -> Piece:
    """One choice's pieces as a single piece (non-streaming responses); a choice still running when its request
    failed finishes with "error"."""
    lps = [p.logprobs for p in pieces if p.logprobs is not None]
    reason = pieces[-1].finish_reason if pieces and pieces[-1].finish_reason else "error" if failed else None
    return Piece(
        index, "".join(p.text for p in pieces), [t for p in pieces for t in p.token_ids],
        Logprobs(*(sum((getattr(lp, f) for lp in lps), []) for f in ("token", "top_ids", "top"))) if lps else None,
        next((p.prompt_logprobs for p in pieces if p.prompt_logprobs is not None), None), reason,
        events=[e for p in pieces for e in p.events])


def usage(prompt_tokens: int, completion_tokens: int, cached_tokens: int) -> dict[str, Any]:
    """OpenAI's usage; `cached_tokens`: prompt tokens the prefix cache served (the engine's count)."""
    return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens_details": {"cached_tokens": cached_tokens}}


def _rows(tok: Tokenizer, ids: list[int], lp: Logprobs) -> list[Row]:
    def text(i: int) -> str:
        return tok.decode([i])
    return [(text(t), max(v, MASKED), [(text(i), max(x, MASKED)) for i, x in zip(ti, tv, strict=True)])
            for t, v, ti, tv in zip(ids, lp.token, lp.top_ids, lp.top, strict=True)]


def _call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


def _finish(reason: str | None, called: bool) -> str | None:
    return "tool_calls" if called and reason == "stop" else reason


class ChatCompletion:
    """Chat: the text as `content`, reasoning as `reasoning_content`, tool calls as `tool_calls` (finish reason
    "tool_calls"); streamed as deltas of each, a call as its name then its arguments' pieces."""

    object, chunk_object = "chat.completion", "chat.completion.chunk"

    def __init__(self, tok: Tokenizer) -> None:
        self.tok = tok
        self.called: set[int] = set()       # streamed choices that made a tool call

    def choice(self, index: int, p: Piece) -> dict[str, Any]:
        blocks = Blocks(_call_id)
        blocks.push(p.events)
        text, reasoning = ("".join(b.text for b in blocks.blocks if b.kind == k) for k in ("text", "reasoning"))
        calls = [{"id": b.id, "type": "function", "function": {"name": b.name, "arguments": b.text}}
                 for b in blocks.calls]
        message = {"role": "assistant", "content": text if text or not calls else None}
        message |= ({"reasoning_content": reasoning} if reasoning else {}) | ({"tool_calls": calls} if calls else {})
        return {"index": index, "message": message, "logprobs": self._logprobs(p),
                "finish_reason": _finish(p.finish_reason, bool(calls))}

    def chunks(self, p: Piece, first: bool, always: bool) -> list[dict[str, Any]]:
        deltas = [d for e in p.events if (d := self._delta(p.index, e))]
        if first:
            deltas[:1] = [{"role": "assistant", **(deltas[0] if deltas else {"content": ""})}]
        if not deltas and (always or p.logprobs is not None or p.finish_reason):
            deltas = [{}]
        chunks = [{"index": p.index, "delta": d, "logprobs": None, "finish_reason": None} for d in deltas]
        if chunks:
            chunks[0]["logprobs"] = self._logprobs(p)
            chunks[-1]["finish_reason"] = _finish(p.finish_reason, p.index in self.called)
        return chunks

    def failed(self, index: int) -> dict[str, Any]:
        return {"index": index, "delta": {}, "logprobs": None, "finish_reason": "error"}

    def _delta(self, index: int, e: ParseEvent) -> dict[str, Any] | None:
        if isinstance(e, ToolCallStart):
            self.called.add(index)
            call = {"index": e.index, "id": _call_id(), "type": "function",
                    "function": {"name": e.name, "arguments": ""}}
            return {"tool_calls": [call]}
        if isinstance(e, ToolArgsDelta):
            return {"tool_calls": [{"index": e.index, "function": {"arguments": e.text}}]}
        if isinstance(e, Text):
            return {"content": e.text}
        return {"reasoning_content": e.text} if isinstance(e, Reasoning) else None

    def _logprobs(self, p: Piece) -> dict[str, Any] | None:
        if p.logprobs is None:
            return None

        def entry(s: str, v: float | None) -> dict[str, Any]:
            return {"token": s, "logprob": v, "bytes": list(s.encode())}
        return {"content": [{**entry(s, v), "top_logprobs": [entry(a, b) for a, b in top]}
                            for s, v, top in _rows(self.tok, p.token_ids, p.logprobs)]}


class Completion:
    """Text completions. With `echo`, a choice's first text (and logprobs) starts with its prompt's."""

    object = chunk_object = "text_completion"

    def __init__(self, tok: Tokenizer, prompts: list[tuple[str, list[int]]], n: int, echo: bool) -> None:
        self.tok, self.prompts, self.n, self.echo = tok, prompts, n, echo
        self.prompt_lp: dict[int, Logprobs] = {}   # the engine scores a prompt once, on its first sequence
        self.offset: dict[int, int] = {}           # per choice: text offset of its next token

    def choice(self, index: int, p: Piece, first: bool = True) -> dict[str, Any]:
        prompt = index // self.n
        if p.prompt_logprobs is not None:
            self.prompt_lp[prompt] = p.prompt_logprobs
        echo = self.echo and first
        text, ids = self.prompts[prompt]
        logprobs = None
        if p.logprobs is not None:
            rows = _rows(self.tok, p.token_ids, p.logprobs)
            if echo:
                # The engine scores the prompt on sequence 0; a sibling streamed ahead of it shows it unscored.
                lp = self.prompt_lp.get(prompt)
                scored = _rows(self.tok, ids[1:], lp) if lp else [(self.tok.decode([t]), None, None) for t in ids[1:]]
                rows = [(self.tok.decode(ids[:1]), None, None), *scored, *rows]
            logprobs = self._logprobs(index, rows)
        return {"index": index, "text": text + p.text if echo else p.text, "logprobs": logprobs,
                "finish_reason": p.finish_reason}

    def chunks(self, p: Piece, first: bool, always: bool) -> list[dict[str, Any]]:
        keep = first or always or p.text or p.logprobs is not None or p.finish_reason
        return [self.choice(p.index, p, first)] if keep else []

    def failed(self, index: int) -> dict[str, Any]:
        return {"index": index, "text": "", "logprobs": None, "finish_reason": "error"}

    def _logprobs(self, index: int, rows: list[Row]) -> dict[str, Any]:
        start = self.offset.get(index, 0)
        offsets = list(itertools.accumulate((len(s) for s, _, _ in rows), initial=start))
        self.offset[index] = offsets.pop()
        return {"tokens": [s for s, _, _ in rows], "token_logprobs": [v for _, v, _ in rows],
                "top_logprobs": [None if top is None else dict(top) for _, _, top in rows], "text_offset": offsets}
