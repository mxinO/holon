"""The frontend ↔ engine process boundary (design §7.8).

The engine runs in its own process and speaks token ids. Frontends reach it through `Client` (an `EngineClient`):
requests, aborts and control messages go in over one ZeroMQ socket, and each engine step's deltas come back as one
message over another. Messages are two frames — a kind and a msgspec/msgpack payload of the API's plain records — so
a native frontend could speak the same protocol.

Failure model: if the engine fails inside a step it reports an error for every live request before exiting; if it
dies outright the client notices (the process is gone) and ends every open request with an error. Either way no
frontend waits forever; restarting the engine is the supervisor's job (design §7.7).
"""
from __future__ import annotations

import asyncio
import collections
import contextlib
import dataclasses
import multiprocessing
import os
import pathlib
import shutil
import tempfile
import uuid
from collections.abc import AsyncIterator, Callable
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
_POLL_MS = 1000


@dataclasses.dataclass(frozen=True)
class Endpoints:
    requests: str            # frontends -> engine
    results: str             # engine -> frontends
    directory: str           # holds the sockets; removed by the launcher

    @classmethod
    def local(cls) -> Endpoints:
        base = pathlib.Path(tempfile.mkdtemp(prefix="holon-"))
        return cls(f"ipc://{base / 'requests'}", f"ipc://{base / 'results'}", str(base))


def serve(engine_factory: Callable[[], Any], endpoints: Endpoints) -> None:
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
    while os.getppid() == parent:                 # the launcher died without shutting us down: free the GPU
        deltas: list[TokenDelta] = []
        timeout = 0 if engine.has_work() else _POLL_MS
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
        try:
            if engine.has_work():
                deltas += engine.step()
        except Exception as e:  # noqa: BLE001 — every live request learns why before the process exits
            error = f"engine failed: {type(e).__name__}: {e}"
            outbox.send_multipart([b"deltas", _encode(deltas + engine.fail_all(error))])
            raise
        if deltas:
            outbox.send_multipart([b"deltas", _encode(deltas)])


class Client:
    """An `EngineClient` over the engine process's endpoints (asyncio)."""

    def __init__(self, endpoints: Endpoints, alive: Callable[[], bool] = lambda: True) -> None:
        self._alive = alive                       # whether the engine process still runs
        self._ctx = zmq.asyncio.Context()
        self._requests = self._ctx.socket(zmq.PUSH)
        self._requests.connect(endpoints.requests)
        self._results = self._ctx.socket(zmq.PULL)
        self._results.connect(endpoints.results)
        self._queues: dict[str, asyncio.Queue[TokenDelta]] = {}
        self._stats: collections.deque[asyncio.Future[dict[str, Any]]] = collections.deque()
        self._reader: asyncio.Task[None] | None = None
        self._dead: str | None = None             # why the engine is gone, once it is

    async def wait_ready(self) -> None:
        while not await self._results.poll(_POLL_MS):
            if not self._alive():
                raise RuntimeError("engine process exited during startup")
        kind, payload = await self._results.recv_multipart()
        if kind == b"failed":
            raise RuntimeError(f"engine failed to start: {_DECODE[kind].decode(payload)}")
        self._reader = asyncio.get_running_loop().create_task(self._read())

    async def generate(self, token_ids: list[int], params: SamplingParams, rid: str | None = None
                       ) -> AsyncIterator[TokenDelta]:
        if self._dead is not None:
            raise RuntimeError(self._dead)
        rid = rid or uuid.uuid4().hex
        if rid in self._queues:
            raise ValueError(f"request id {rid!r} is already in flight")
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
            if finished < params.n and self._dead is None:           # the caller left early
                await self.abort(rid)

    async def abort(self, rid: str) -> None:
        await self._requests.send_multipart([b"abort", _encode(Abort(rid))])

    async def stats(self) -> dict[str, Any]:
        if self._dead is not None:
            raise RuntimeError(self._dead)
        future = asyncio.get_running_loop().create_future()
        self._stats.append(future)                # replies arrive in request order (one socket each way)
        await self._requests.send_multipart([b"control", _encode("stats")])
        return await future

    async def control(self, op: str) -> None:
        """`flush_prefix` or `shutdown`."""
        await self._requests.send_multipart([b"control", _encode(op)])

    async def _read(self) -> None:
        while True:
            if not await self._results.poll(_POLL_MS):
                if not self._alive():
                    self._fail("engine process exited")
                    return
                continue
            kind, payload = await self._results.recv_multipart()
            if kind == b"stats":
                self._stats.popleft().set_result(_DECODE[kind].decode(payload))
                continue
            for delta in _DECODE[kind].decode(payload):
                if (queue := self._queues.get(delta.rid)) is not None:
                    queue.put_nowait(delta)

    def _fail(self, reason: str) -> None:
        """End every open request and pending stats call; refuse new ones."""
        self._dead = reason
        for rid, queue in self._queues.items():
            queue.put_nowait(TokenDelta(rid, [], finish_reason="error", error=reason))
        while self._stats:
            self._stats.popleft().set_exception(RuntimeError(reason))

    def close(self) -> None:
        if self._reader is not None:
            self._reader.cancel()
        self._ctx.destroy(linger=0)


def _engine_main(config: Any, endpoints: Endpoints) -> None:
    from .engine import Engine  # imported in the child: CUDA initializes only there

    serve(lambda: Engine(config), endpoints)


def launch(config: Any, target: Callable[..., None] = _engine_main
           ) -> tuple[multiprocessing.process.BaseProcess, Endpoints]:
    """Start an engine process for `config` (an EngineConfig); connect a `Client` to the returned endpoints."""
    endpoints = Endpoints.local()
    proc = multiprocessing.get_context("spawn").Process(target=target, args=(config, endpoints), daemon=True)
    proc.start()
    return proc, endpoints


@contextlib.asynccontextmanager
async def engine_process(config: Any, target: Callable[..., None] = _engine_main) -> AsyncIterator[Client]:
    """An engine process and a ready client for it; shuts the engine down on exit. `target` runs in the child
    (tests substitute engines that fail on purpose)."""
    proc, endpoints = launch(config, target)
    client = Client(endpoints, alive=proc.is_alive)
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
        shutil.rmtree(endpoints.directory, ignore_errors=True)
