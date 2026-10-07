"""The HTTP app: OpenAI chat and text completions over an EngineClient, plus /v1/models, /health and /metrics."""
from __future__ import annotations

import asyncio
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
    pass


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": "invalid_request_error", "code": status}}, status)


def build_app(client: EngineClient, bundle: FrontendBundle) -> FastAPI:
    app = FastAPI(title="Holon")
    tok = bundle.tokenizer
    started = int(time.time())

    @app.exception_handler(RequestValidationError)
    async def _invalid(_: Request, e: RequestValidationError) -> JSONResponse:
        return _error(400, "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()))

    @app.exception_handler(BadRequest)
    async def _bad(_: Request, e: BadRequest) -> JSONResponse:
        return _error(400, str(e))

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
            # Wait for the first piece before answering, so that a rejected request gets a 400 status.
            first = await _unless_disconnected(request, anext(gens[0]))
            if first is None:
                await gens[0].aclose()
                return Response(status_code=499)
            if first.finish_reason == "error":
                raise BadRequest(first.error)
            events = _events(fmt, {**envelope, "object": fmt.chunk_object}, first, gens[0],
                             n_prompt if req.include_usage else None)
            return _EventStream(events, media_type="text/event-stream")

        async def collect(gen: AsyncIterator[Piece]) -> list[Piece]:
            return [p async for p in gen]
        results = await _unless_disconnected(request, asyncio.gather(*map(collect, gens)))
        if results is None:
            return Response(status_code=499)
        if errors := [p.error for pieces in results for p in pieces if p.finish_reason == "error"]:
            raise BadRequest(errors[0])
        choices = [fmt.choice(i * params.n + k, responses.merge([p for p in pieces if p.index == k]))
                   for i, pieces in enumerate(results) for k in range(params.n)]
        n_completion = sum(len(p.token_ids) for pieces in results for p in pieces)
        return JSONResponse({**envelope, "choices": choices, "usage": responses.usage(n_prompt, n_completion)})

    return app


async def _events(fmt: responses.Chat | responses.Completion, envelope: dict[str, Any], first: Piece,
                  gen: AsyncIterator[Piece], n_prompt: int | None) -> AsyncIterator[str]:
    """Server-sent events of one request; the final usage chunk only if asked for (`n_prompt` given)."""
    seen: set[int] = set()
    n_completion = 0
    async with aclosing(gen):
        p: Piece | None = first
        while p is not None:
            if p.finish_reason == "error":
                yield _event({"error": {"message": p.error, "type": "invalid_request_error", "code": 400}})
                break
            n_completion += len(p.token_ids)
            new = p.index not in seen
            seen.add(p.index)
            if new or p.text or p.logprobs is not None or p.finish_reason:
                yield _event({**envelope, "choices": [fmt.chunk(p.index, p, new)]})
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
