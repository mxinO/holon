"""The frontend ↔ engine process boundary (design §7.8).

The engine runs in its own process and speaks token ids. Frontends reach it through `Client` (an `EngineClient`):
requests, aborts and control messages go in over one ZeroMQ socket, and each engine step's deltas come back as one
message over another. Messages are two frames — a kind and a msgspec/msgpack payload of the API's plain records — so
a native frontend could speak the same protocol.
"""
from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import multiprocessing
import os
import pathlib
import tempfile
import uuid
from collections.abc import AsyncIterator
from typing import Any

import msgspec
import zmq
import zmq.asyncio

from holon.api import Abort, EngineRequest, SamplingParams, TokenDelta

_encode = msgspec.msgpack.encode
_DECODE = {b"submit": msgspec.msgpack.Decoder(EngineRequest), b"abort": msgspec.msgpack.Decoder(Abort),
           b"control": msgspec.msgpack.Decoder(str), b"deltas": msgspec.msgpack.Decoder(list[TokenDelta]),
           b"stats": msgspec.msgpack.Decoder(dict[str, Any]), b"ready": msgspec.msgpack.Decoder(str),
           b"failed": msgspec.msgpack.Decoder(str)}


@dataclasses.dataclass(frozen=True)
class Endpoints:
    requests: str            # frontends -> engine
    results: str             # engine -> frontends

    @classmethod
    def local(cls) -> Endpoints:
        base = pathlib.Path(tempfile.mkdtemp(prefix="holon-"))
        return cls(f"ipc://{base / 'requests'}", f"ipc://{base / 'results'}")


def serve(engine_factory: Any, endpoints: Endpoints) -> None:
    """The engine process: build the engine, then step while there is work and serve messages in between."""
    ctx = zmq.Context()
    inbox, outbox = ctx.socket(zmq.PULL), ctx.socket(zmq.PUSH)
    inbox.bind(endpoints.requests)
    outbox.bind(endpoints.results)
    try:
        engine = engine_factory()
    except Exception as e:  # noqa: BLE001 — reported to the launcher, which raises it
        outbox.send_multipart([b"failed", _encode(f"{type(e).__name__}: {e}")])
        raise
    outbox.send_multipart([b"ready", _encode("ready")])
    parent = os.getppid()
    while True:
        deltas: list[TokenDelta] = []
        if not engine.has_work() and not inbox.poll(1000):
            if os.getppid() != parent:           # the launcher died without shutting us down: free the GPU
                return
            continue
        timeout = 0
        while inbox.poll(timeout):
            kind, payload = inbox.recv_multipart()
            msg = _DECODE[kind].decode(payload)
            if kind == b"submit":
                try:
                    engine.add(msg.token_ids, msg.params, rid=msg.rid)
                except (ValueError, NotImplementedError) as e:      # a rejected request, not an engine failure
                    deltas.append(TokenDelta(msg.rid, [], finish_reason="error", error=str(e)))
            elif kind == b"abort":
                deltas += engine.abort(msg.rid)
            elif msg == "stats":
                outbox.send_multipart([b"stats", _encode(engine.stats())])
            elif msg == "flush_prefix":
                deltas += engine.flush_prefix()
            elif msg == "shutdown":
                deltas += engine.drain()
                if deltas:
                    outbox.send_multipart([b"deltas", _encode(deltas)])
                return
            timeout = 0
        if engine.has_work():
            deltas += engine.step()
        if deltas:
            outbox.send_multipart([b"deltas", _encode(deltas)])


class Client:
    """An `EngineClient` over the engine process's endpoints (asyncio)."""

    def __init__(self, endpoints: Endpoints) -> None:
        self._ctx = zmq.asyncio.Context()
        self._requests = self._ctx.socket(zmq.PUSH)
        self._requests.connect(endpoints.requests)
        self._results = self._ctx.socket(zmq.PULL)
        self._results.connect(endpoints.results)
        self._queues: dict[str, asyncio.Queue[TokenDelta]] = {}
        self._stats: asyncio.Future[dict[str, Any]] | None = None
        self._reader: asyncio.Task[None] | None = None

    async def wait_ready(self) -> None:
        kind, payload = await self._results.recv_multipart()
        if kind == b"failed":
            raise RuntimeError(f"engine failed to start: {_DECODE[kind].decode(payload)}")
        self._reader = asyncio.get_running_loop().create_task(self._read())

    async def generate(self, token_ids: list[int], params: SamplingParams, rid: str | None = None
                       ) -> AsyncIterator[TokenDelta]:
        rid = rid or uuid.uuid4().hex
        queue: asyncio.Queue[TokenDelta] = asyncio.Queue()
        self._queues[rid] = queue
        finished = 0
        try:
            await self._requests.send_multipart([b"submit", _encode(EngineRequest(rid, list(token_ids), params))])
            while finished < params.n:
                delta = await queue.get()
                if delta.finish_reason is not None:
                    finished += params.n if delta.finish_reason == "error" else 1
                yield delta
        finally:
            del self._queues[rid]
            if finished < params.n:                                  # the caller left early
                await self.abort(rid)

    async def abort(self, rid: str) -> None:
        await self._requests.send_multipart([b"abort", _encode(Abort(rid))])

    async def stats(self) -> dict[str, Any]:
        self._stats = asyncio.get_running_loop().create_future()
        await self._requests.send_multipart([b"control", _encode("stats")])
        return await self._stats

    async def control(self, op: str) -> None:
        """`flush_prefix` or `shutdown`."""
        await self._requests.send_multipart([b"control", _encode(op)])

    async def _read(self) -> None:
        while True:
            kind, payload = await self._results.recv_multipart()
            if kind == b"stats":
                if self._stats is not None and not self._stats.done():
                    self._stats.set_result(_DECODE[kind].decode(payload))
                continue
            for delta in _DECODE[kind].decode(payload):
                if (queue := self._queues.get(delta.rid)) is not None:
                    queue.put_nowait(delta)

    def close(self) -> None:
        if self._reader is not None:
            self._reader.cancel()
        self._ctx.destroy(linger=0)


def _engine_main(config: Any, endpoints: Endpoints) -> None:
    from .engine import Engine  # imported in the child: CUDA initializes only there

    serve(lambda: Engine(config), endpoints)


def launch(config: Any) -> tuple[multiprocessing.process.BaseProcess, Endpoints]:
    """Start an engine process for `config` (an EngineConfig); connect a `Client` to the returned endpoints."""
    endpoints = Endpoints.local()
    proc = multiprocessing.get_context("spawn").Process(target=_engine_main, args=(config, endpoints), daemon=True)
    proc.start()
    return proc, endpoints


@contextlib.asynccontextmanager
async def engine_process(config: Any) -> AsyncIterator[Client]:
    """An engine process and a ready client for it; shuts the engine down on exit."""
    proc, endpoints = launch(config)
    client = Client(endpoints)
    try:
        await client.wait_ready()
        yield client
    finally:
        with contextlib.suppress(Exception):
            await client.control("shutdown")
        client.close()
        proc.join(timeout=30)
        if proc.is_alive():
            proc.kill()
