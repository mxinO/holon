"""HTTP plumbing the routers share: errors by kind, waiting on the engine while watching the client, event streams."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable
from typing import Any

import anyio
from fastapi import Request
from fastapi.responses import StreamingResponse

from .stream import Piece


class BadRequest(Exception):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def rejected(p: Piece) -> BadRequest:
    """A request that failed before its first token, by the error's kind (TokenDelta.kind): the request is wrong
    (400), the server is busy (503), or the engine failed (500). After the first token a failure is a finish reason
    ("error"), never a status (D7)."""
    return BadRequest(p.error or "request failed", {"client": 400, "busy": 503}.get(p.kind, 500))


async def collect(gens: list[AsyncIterator[Piece]]) -> list[list[Piece]]:
    """Every piece of each prompt of a batch, generated concurrently. A failed prompt fails the whole batch, so it
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


def failure(results: list[list[Piece]]) -> Piece | None:
    """The error piece of a collected batch, if any; raised as a status if its prompt had no token yet."""
    for pieces in results:
        if pieces and pieces[-1].finish_reason == "error":
            if not any(p.token_ids for p in pieces):
                raise rejected(pieces[-1])
            return pieces[-1]
    return None


async def unless_disconnected[T](request: Request, aw: Awaitable[T]) -> T | None:
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


KEEPALIVE = 15.0      # seconds without an event after which a stream sends a keepalive (clients and proxies drop
                      # quiet connections: a call's arguments are held back until it closes)


class EventStream(StreamingResponse):
    """Server-sent events, with `keepalive` (a comment by default) after every KEEPALIVE seconds of silence, that close
    their body however the response ends. Starlette cancels a stream whose client went away, but leaves the body
    generator suspended if that happens while a chunk is being sent; closing it aborts the engine request."""

    def __init__(self, events: AsyncIterator[str], keepalive: str = ": keepalive\n\n") -> None:
        super().__init__(_paced(events, keepalive), media_type="text/event-stream")

    async def stream_response(self, send: Any) -> None:
        try:
            await super().stream_response(send)
        finally:
            await self.body_iterator.aclose()  # type: ignore[union-attr]


async def _paced(events: AsyncIterator[str], keepalive: str) -> AsyncIterator[str]:
    """`events`, and `keepalive` after every KEEPALIVE seconds without one. The next event is awaited in a task of its
    own, so that a timeout never cancels `events`."""
    pending: asyncio.Future[str] | None = None
    try:
        while True:
            pending = pending or asyncio.ensure_future(anext(events))
            done, _ = await asyncio.wait({pending}, timeout=KEEPALIVE)
            if not done:
                yield keepalive
                continue
            finished, pending = pending, None
            try:
                event = finished.result()
            except StopAsyncIteration:
                return
            yield event
    finally:
        with anyio.CancelScope(shield=True):                 # the request must still be aborted when cancelled
            if pending is not None:
                pending.cancel()
                await asyncio.wait({pending})
            await events.aclose()  # type: ignore[attr-defined]  # an async generator in practice
