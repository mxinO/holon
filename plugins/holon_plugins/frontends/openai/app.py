"""The HTTP app: OpenAI chat and text completions over an EngineClient, plus /v1/models, /health and /metrics."""
from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator, Awaitable
from contextlib import aclosing
from typing import Any

import anyio
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from holon.api import EngineClient, FrontendBundle, SamplingParams

from . import responses
from .schemas import ChatRequest, CompletionRequest, GenerationRequest
from .stream import Piece, generate


class BadRequest(Exception):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def _rejected(p: Piece) -> BadRequest:
    """A failed request's status by the error's kind (TokenDelta.kind): the request is wrong (400), the server is
    busy (503), or the engine failed (500)."""
    return BadRequest(p.error or "request failed", {"client": 400, "busy": 503}.get(p.kind, 500))


def _error_body(status: int, message: str) -> dict[str, Any]:
    kind = "invalid_request_error" if status == 400 else "server_error"
    return {"error": {"message": message, "type": kind, "code": status}}


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse(_error_body(status, message), status)


def build_app(client: EngineClient, bundle: FrontendBundle) -> FastAPI:
    app = FastAPI(title="Holon")
    tok = bundle.tokenizer
    started = int(time.time())

    @app.exception_handler(RequestValidationError)
    async def _invalid(_: Request, e: RequestValidationError) -> JSONResponse:
        return _error(400, "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()))

    @app.exception_handler(BadRequest)
    async def _bad(_: Request, e: BadRequest) -> JSONResponse:
        return _error(e.status, str(e))

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {}

    @app.get("/metrics")
    async def metrics() -> dict[str, Any]:
        return await client.stats()

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": bundle.model_name, "object": "model", "created": started,
                                            "owned_by": "holon"}]}

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest, request: Request) -> Response:
        if not req.prompts:
            raise BadRequest("empty prompt list")
        if req.stream and len(req.prompts) > 1:
            raise BadRequest("streaming takes a single prompt")
        prompts = [(p, tok.encode(p)) if isinstance(p, str) else (tok.decode(p), p) for p in req.prompts]
        fmt = responses.Completion(tok, prompts, req.n, req.echo)
        return await _respond(request, req, fmt, [ids for _, ids in prompts], req.params(), "cmpl")

    @app.post("/v1/chat/completions")
    async def chat(req: ChatRequest, request: Request) -> Response:
        try:
            ids = bundle.render_chat(req.messages, tools=req.tools, **req.chat_template_kwargs)
        except Exception as e:  # noqa: BLE001 — a template that rejects the conversation is a client error
            raise BadRequest(f"chat template: {e}") from e
        return await _respond(request, req, responses.Chat(tok), [ids], req.params(), "chatcmpl")

    async def _respond(request: Request, req: GenerationRequest, fmt: responses.Chat | responses.Completion,
                       prompts: list[list[int]], params: SamplingParams, kind: str) -> Response:
        rid = f"{kind}-{uuid.uuid4().hex}"
        rids = [rid] if len(prompts) == 1 else [f"{rid}-{i}" for i in range(len(prompts))]
        gens = [generate(client, tok, ids, params, r, req.stops) for ids, r in zip(prompts, rids, strict=True)]
        envelope = {"id": rid, "object": fmt.object, "created": int(time.time()), "model": bundle.model_name}
        n_prompt = sum(map(len, prompts))

        if req.stream:
            # Wait for the first piece before answering, so that a rejected request gets its error status.
            first = await _unless_disconnected(request, anext(gens[0]))
            if first is None:
                await gens[0].aclose()
                return Response(status_code=499)
            if first.finish_reason == "error":
                raise _rejected(first)
            events = _events(fmt, {**envelope, "object": fmt.chunk_object}, first, gens[0],
                             n_prompt if req.include_usage else None, req.return_token_ids)
            return _EventStream(events, media_type="text/event-stream")

        results = await _unless_disconnected(request, _collect(gens))
        if results is None:
            return Response(status_code=499)
        if errors := [p for pieces in results for p in pieces if p.finish_reason == "error"]:
            raise _rejected(errors[0])
        merged = [responses.merge([p for p in pieces if p.index == k]) for pieces in results for k in range(params.n)]
        choices = [fmt.choice(i, p) | ({"token_ids": p.token_ids} if req.return_token_ids else {})
                   for i, p in enumerate(merged)]
        n_completion = sum(len(p.token_ids) for pieces in results for p in pieces)
        return JSONResponse({**envelope, "choices": choices, "usage": responses.usage(n_prompt, n_completion)})

    return app


async def _collect(gens: list[AsyncIterator[Piece]]) -> list[list[Piece]]:
    """Every piece of each prompt of a batch, generated concurrently. A rejected prompt fails the whole batch, so it
    cancels the others (which aborts them). An anyio task group, unlike `asyncio.gather`, lets their shielded aborts
    complete when the client disconnects."""
    results: list[list[Piece]] = [[] for _ in gens]

    async def fill(pieces: list[Piece], gen: AsyncIterator[Piece]) -> None:
        async for p in gen:
            pieces.append(p)
            if p.finish_reason == "error":
                tg.cancel_scope.cancel()

    async with anyio.create_task_group() as tg:
        for pieces, gen in zip(results, gens, strict=True):
            tg.start_soon(fill, pieces, gen)
    return results


async def _events(fmt: responses.Chat | responses.Completion, envelope: dict[str, Any], first: Piece,
                  gen: AsyncIterator[Piece], n_prompt: int | None, token_ids: bool) -> AsyncIterator[str]:
    """Server-sent events of one request; the final usage chunk only if asked for (`n_prompt` given); with
    `token_ids`, a chunk for every piece, carrying its token ids."""
    seen: set[int] = set()
    n_completion = 0
    async with aclosing(gen):
        p: Piece | None = first
        while p is not None:
            if p.finish_reason == "error":
                e = _rejected(p)
                yield _event(_error_body(e.status, str(e)))
                break
            n_completion += len(p.token_ids)
            new = p.index not in seen
            seen.add(p.index)
            if new or p.text or p.logprobs is not None or p.finish_reason or token_ids:
                chunk = fmt.chunk(p.index, p, new) | ({"token_ids": p.token_ids} if token_ids else {})
                yield _event({**envelope, "choices": [chunk]})
            p = await anext(gen, None)
    if n_prompt is not None:
        yield _event({**envelope, "choices": [], "usage": responses.usage(n_prompt, n_completion)})
    yield "data: [DONE]\n\n"


def _event(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


class _EventStream(StreamingResponse):
    """Closes its body however the response ends. Starlette cancels a stream whose client went away, but leaves the
    body generator suspended if that happens while a chunk is being sent; closing it aborts the engine request."""

    async def stream_response(self, send: Any) -> None:
        try:
            await super().stream_response(send)
        finally:
            await self.body_iterator.aclose()  # type: ignore[union-attr]


async def _unless_disconnected[T](request: Request, aw: Awaitable[T]) -> T | None:
    """`aw`'s result, or None if the client disconnects first. Starlette notices disconnects only while streaming a
    body, so a handler waiting on the engine watches for them itself."""
    result = None
    async with anyio.create_task_group() as tg:
        async def watch() -> None:
            while (await request.receive())["type"] != "http.disconnect":
                pass
            tg.cancel_scope.cancel()

        tg.start_soon(watch)
        result = await aw
        tg.cancel_scope.cancel()
    return result
