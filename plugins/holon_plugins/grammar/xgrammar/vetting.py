"""Vetting constraints before the engine process compiles them (docs/m05-grammar.md D5). An xgrammar compile cannot be
cancelled, and a compiled grammar can take gigabytes, so each new constraint is first compiled in a worker process
(`worker.py`) that can be killed: past the timeout (`busy`, retry later), or when its resident set grows `budget`
bytes past its size at the job's start (`client`: the schema is too complex). A compiled size over `max_size` is
`client` too; a worker that dies without reporting (a kernel OOM kill, a crash) is `busy`.

One long-lived watchdog thread starts, watches (every `poll` seconds) and respawns the workers, so PR_SET_PDEATHSIG —
which follows the thread that created a child — kills them with it. Workers are fresh interpreters (spawned, never
forked from the engine's CUDA process) with a lean environment and no GPU, and get the engine's `TokenizerInfo` as a
file. At most `queue` constraints wait for a worker; more are rejected `busy`.
"""
from __future__ import annotations

import collections
import dataclasses
import json
import os
import selectors
import subprocess
import sys
import tempfile
import threading
import time

from holon.api import RequestError

_LEAN = {"OMP_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false", "MALLOC_ARENA_MAX": "2", "CUDA_VISIBLE_DEVICES": ""}
_PAGE = os.sysconf("SC_PAGE_SIZE")


@dataclasses.dataclass(frozen=True, slots=True)
class Outcome:
    ok: bool
    kind: str = ""              # on failure: "client" (the constraint's own fault) or "busy" (ours, for now)
    message: str = ""
    seconds: float = 0.0        # the compile's wall time
    size: int = 0               # the compiled grammar's bytes
    growth: int = 0             # how far the compile grew the worker's resident set, at its peak (bytes)


@dataclasses.dataclass(slots=True, eq=False)
class _Worker:
    proc: subprocess.Popen
    spawned: float
    ready: bool = False
    key: str | None = None      # the job it runs
    started: float = 0.0
    base: int = 0               # resident bytes at the job's start


def _resident(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/statm") as f:
            return int(f.read().split()[1]) * _PAGE
    except (OSError, IndexError):
        return 0


class Vetting:
    def __init__(self, tokenizer_json: str, vocab_size: int, stops: list[int], workers: int = 2,
                 timeout: float = 10.0, budget: int = 1 << 30, max_size: int = 64 << 20, queue: int = 8,
                 poll: float = 0.05) -> None:
        self.timeout, self.budget, self.max_size, self.queue, self.poll = timeout, budget, max_size, queue, poll
        with tempfile.NamedTemporaryFile("w", prefix="holon-grammar-", suffix=".json", delete=False) as f:
            f.write(tokenizer_json)
        self._args = [f.name, str(vocab_size), json.dumps(sorted(stops))]
        self._lock = threading.Lock()
        self._waiting: collections.deque[tuple[str, str]] = collections.deque()   # (key, constraint JSON)
        self._running: set[str] = set()
        self._done: list[tuple[str, Outcome]] = []
        self.starts: list[tuple[float, int]] = []    # per worker start: seconds until ready, resident bytes then
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._watch, args=(workers,), name="grammar-vetting", daemon=True)
        self._thread.start()

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    def submit(self, key: str, constraint: str) -> None:
        """Vet `constraint` (JSON text) under `key`, unless it is already waiting or running."""
        if not self.alive:
            raise RequestError("busy", "the constraint checker is down")
        with self._lock:
            if key in self._running or any(k == key for k, _ in self._waiting):
                return
            if len(self._waiting) >= self.queue:
                raise RequestError("busy", f"{len(self._waiting)} constraints are waiting to be checked")
            self._waiting.append((key, constraint))

    def cancel(self, key: str) -> None:
        """Drop `key` if it still waits for a worker (one already running finishes)."""
        with self._lock:
            self._waiting = collections.deque((k, c) for k, c in self._waiting if k != key)

    def drain(self) -> list[tuple[str, Outcome]]:
        """The outcomes reached since the last call."""
        with self._lock:
            done, self._done = self._done, []
        return done

    def shutdown(self) -> None:
        self._stop.set()
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=5)
        try:
            os.unlink(self._args[0])
        except OSError:
            pass

    # -- the watchdog thread ----------------------------------------------------------------------------------------
    def _watch(self, n: int) -> None:
        workers: list[_Worker | None] = [None] * n
        sel = selectors.DefaultSelector()
        try:
            while not self._stop.is_set():
                now = time.monotonic()
                for i, w in enumerate(workers):
                    if w is None:
                        workers[i] = w = self._spawn(sel, now)
                    failed = self._check(w, now)
                    if failed is not None:
                        self._kill(sel, w)
                        workers[i] = None
                        if w.key is not None:
                            self._finish(w, failed)
                    elif w.ready and w.key is None:
                        self._assign(w, now)
                for event, _ in sel.select(timeout=self.poll):
                    self._read(sel, event.data)
        finally:
            for w in workers:
                if w is not None:
                    self._kill(sel, w)
            sel.close()

    def _spawn(self, sel: selectors.BaseSelector, now: float) -> _Worker:
        module = f"{__package__}.worker"
        # -P: the working directory is not on the path (a directory named like a module there would shadow it).
        proc = subprocess.Popen([sys.executable, "-P", "-m", module, str(os.getpid()), *self._args],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=os.environ | _LEAN)
        w = _Worker(proc, now)
        sel.register(proc.stdout, selectors.EVENT_READ, w)
        return w

    def _check(self, w: _Worker, now: float) -> Outcome | None:
        """Why `w` must go: it died, its job ran past the timeout, or the job grew it past the budget."""
        if w.proc.poll() is not None:
            return Outcome(False, "busy", f"the constraint checker exited ({w.proc.returncode})")
        if w.key is None:
            return None
        if now - w.started > self.timeout:
            return Outcome(False, "busy", f"compiling the constraint took over {self.timeout:g} s")
        if _resident(w.proc.pid) - w.base > self.budget:
            return Outcome(False, "client", f"schema too complex: compiling it needs over {self.budget >> 20} MB")
        return None

    def _assign(self, w: _Worker, now: float) -> None:
        with self._lock:
            if not self._waiting:
                return
            w.key, constraint = self._waiting.popleft()
            self._running.add(w.key)
        w.started, w.base = now, _resident(w.proc.pid)
        try:
            w.proc.stdin.write(constraint + "\n")
            w.proc.stdin.flush()
        except OSError:                          # it died: `_check` reports it, with this job
            pass

    def _read(self, sel: selectors.BaseSelector, w: _Worker) -> None:
        line = w.proc.stdout.readline()
        if not line:                             # it exited: `_check` reports it
            sel.unregister(w.proc.stdout)
            return
        out = json.loads(line)
        if out.get("ready"):
            w.ready = True
            self.starts.append((time.monotonic() - w.spawned, _resident(w.proc.pid)))
        elif not out["ok"]:
            self._finish(w, Outcome(False, "client", f"invalid constraint: {out['error']}"))
        elif out["size"] > self.max_size:
            self._finish(w, Outcome(False, "client", f"schema too complex: it compiles to {out['size'] >> 20} MB "
                                    f"(at most {self.max_size >> 20} MB)", out["seconds"], out["size"], out["growth"]))
        else:
            self._finish(w, Outcome(True, seconds=out["seconds"], size=out["size"], growth=out["growth"]))

    def _finish(self, w: _Worker, outcome: Outcome) -> None:
        with self._lock:
            self._running.discard(w.key)
            self._done.append((w.key, outcome))
        w.key = None

    @staticmethod
    def _kill(sel: selectors.BaseSelector, w: _Worker) -> None:
        w.proc.kill()
        w.proc.wait()
        if w.proc.stdout in sel.get_map():
            sel.unregister(w.proc.stdout)
        w.proc.stdout.close()
        w.proc.stdin.close()
