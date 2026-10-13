"""Engines behind an HTTP server, driven like the in-process ones (stdlib only): the backend starts the server, `add`
streams one request on its own thread, and `step` hands back the tokens that arrived — waiting a little for the
first, so the driver's clock keeps releasing arrivals. A "step" here is a client poll, not an engine step, so only
the request timings (TTFT, TPOT, throughput) compare across backends.
"""
from __future__ import annotations

import atexit
import http.client
import json
import os
import queue
import socket
import subprocess
import threading
import time
from collections.abc import Callable, Iterator

from driver import Delta

POLL_S = 0.02                 # longest a step waits for a token


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    """A server process on a local port, up once `health` answers 200; stopped with this process."""

    def __init__(self, cmd: list[str], port: int, health: str = "/health", timeout_s: float = 900) -> None:
        self.cmd, self.port = cmd, port
        self.proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, start_new_session=True)
        atexit.register(self.stop)
        deadline = time.monotonic() + timeout_s
        while not self._healthy(health):
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited with {self.proc.returncode}: {cmd}")
            if time.monotonic() > deadline:
                raise TimeoutError(f"server not healthy after {timeout_s} s: {cmd}")
            time.sleep(0.5)

    def _healthy(self, path: str) -> bool:
        try:
            conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
            conn.request("GET", path)
            return conn.getresponse().status == 200
        except OSError:
            return False

    def stop(self) -> None:
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, 15)
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, 9)

    def get(self, path: str) -> dict:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=60)
        conn.request("GET", path)
        return json.loads(conn.getresponse().read())

    def stream(self, path: str, body: dict) -> Iterator[dict]:
        """The JSON objects of a server-sent event stream."""
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3600)
        conn.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
        resp = conn.getresponse()
        if resp.status != 200:
            raise RuntimeError(f"{path}: HTTP {resp.status}: {resp.read()[:500]!r}")
        for line in resp:
            line = line.strip()
            if line.startswith(b"data: ") and line != b"data: [DONE]":
                yield json.loads(line[6:])


class Backend:
    """Streams requests to `server` at `path`; `body(prompt, max_tokens)` is a request, `parse(event, state)` the
    token ids an event brings and whether the request finished (`state`: a dict per request, for parsers that count).
    Each request's last event is kept (`finals`: e.g. server-side timings)."""

    def __init__(self, server: Server, path: str, body: Callable[[list[int], int], dict],
                 parse: Callable[[dict, dict], tuple[list[int], bool]]) -> None:
        self.server, self.path, self.body, self.parse = server, path, body, parse
        self.arrived: queue.Queue[Delta | BaseException] = queue.Queue()
        self.in_flight = 0
        self.finals: list[dict] = []                 # every request's last event

    def add(self, rid: int, prompt: list[int], max_tokens: int) -> None:
        self.in_flight += 1
        threading.Thread(target=self._run, args=(rid, prompt, max_tokens), daemon=True).start()

    def _run(self, rid: int, prompt: list[int], max_tokens: int) -> None:
        try:
            state: dict = {}
            for event in self.server.stream(self.path, self.body(prompt, max_tokens)):
                toks, done = self.parse(event, state)
                if done:
                    self.finals.append(event)
                if toks or done:
                    self.arrived.put((rid, toks, done))
        except BaseException as e:  # noqa: BLE001 — surfaced by step() on the driver's thread
            self.arrived.put(e)

    def step(self) -> list[Delta]:
        out = []
        try:
            item = self.arrived.get(timeout=POLL_S)
            while True:
                if isinstance(item, BaseException):
                    raise item
                out.append(item)
                self.in_flight -= item[2]
                item = self.arrived.get_nowait()
        except queue.Empty:
            return out

    def has_work(self) -> bool:
        return self.in_flight > 0
