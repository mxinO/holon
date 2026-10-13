"""`grammar.xgrammar`: the logits processor that constrains requests carrying `ext["grammar.constraint@1"]`
(`constraint.py`) with xgrammar (docs/m05-grammar.md D3–D6).

A request's constraint is checked at `open` (malformed, a strict schema's unsupported feature: `client`), vetted in a
worker process (`vetting.py`), then compiled for use by one of two helper threads (xgrammar releases the GIL), never
on the engine thread; `ready` flips when the compiled grammar is there. Compiled grammars are cached by constraint
(LRU, `cache_bytes`); refused constraints are remembered (`client` for the process's life, `busy` for 10 minutes); a
helper backlog past `backlog` seconds of vetted compile time is rejected `busy`.

Each sequence has a `GrammarMatcher` that advances only at `commit` and stops on the engine's stop tokens plus the
request's `stop_token_ids`. A verify row's columns are masked by a probe (D3, `probe`): column j with the state after
the committed tokens and drafts d0 … d(j−1), up to the first draft the grammar refuses, then rolled back. Options
(`EngineConfig.plugin_options`): `grammar.workers`, `grammar.timeout` (s), `grammar.budget` (MB).
"""
from __future__ import annotations

import collections
import dataclasses
import json
import time
import weakref
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import numpy as np
import torch
import xgrammar as xgr
from holon.api import RequestError

from . import kernel
from .constraint import KEY, check, grammar
from .vetting import Outcome, Vetting

BUSY_TTL = 600.0                                 # seconds a `busy` refusal is remembered


class Plugin:
    def register(self, regs: Any) -> None:
        pass

    def build(self, ctx: Any, vocab: Any) -> XGrammar:
        opts = ctx.options.get("plugins", {})
        stops = sorted(vocab.stop_tokens)
        info = xgr.TokenizerInfo.from_huggingface(vocab.tokenizer(), vocab_size=vocab.size, stop_token_ids=stops)
        return XGrammar(info, stops, workers=int(opts.get("grammar.workers", 2)),
                        timeout=float(opts.get("grammar.timeout", 10.0)),
                        budget=int(float(opts.get("grammar.budget", 1024)) * (1 << 20)))


@dataclasses.dataclass(slots=True, eq=False)
class _Request:
    key: str                                     # its constraint, as JSON text
    stops: list[int]
    stop_row: torch.Tensor                       # the mask allowing only its stop tokens
    grammar: xgr.CompiledGrammar | None = None
    matchers: dict[int, xgr.GrammarMatcher] = dataclasses.field(default_factory=dict)    # per sequence index
    failed: dict[int, str] = dataclasses.field(default_factory=dict)     # a mask `fill` could not build


def probe(matcher: xgr.GrammarMatcher, drafts: list[int], out: torch.Tensor, stop_row: torch.Tensor) -> int:
    """Mask a row's columns into `out` ([n, width]): column j with the state after `drafts[:j]` — through the column
    after the last draft; after a draft the grammar refuses, every column allows everything (the verifier never
    reaches it), and after a stop-token draft the next column allows the stop tokens. Leaves `matcher` as it found
    it; returns the number of refused drafts (0 or 1)."""
    accepted, refused = 0, 0
    for j in range(len(out)):
        if j:
            if j > len(drafts) or matcher.is_terminated():
                out[j:] = -1
                break
            if not matcher.accept_token(drafts[j - 1]):
                out[j:], refused = -1, 1
                break
            accepted += 1
        if matcher.is_terminated():
            out[j] = stop_row
        else:
            matcher.fill_next_token_bitmask(out, j)
    matcher.rollback(accepted)
    return refused


class XGrammar:
    def __init__(self, info: xgr.TokenizerInfo, stops: list[int], *, workers: int = 2, timeout: float = 10.0,
                 budget: int = 1 << 30, cache_bytes: int = 512 << 20, backlog: float = 10.0) -> None:
        self.info, self.stops, self.cache_bytes, self.backlog = info, frozenset(stops), cache_bytes, backlog
        self.mask_width = xgr.get_bitmask_shape(1, info.vocab_size)[1]
        self.vetting = Vetting(info.serialize_json(), info.vocab_size, sorted(info.stop_token_ids), workers,
                               timeout, budget)
        self._compiler = xgr.GrammarCompiler(info, max_threads=2, cache_enabled=False)
        self._helpers = ThreadPoolExecutor(2, thread_name_prefix="grammar-compile")
        weakref.finalize(self, _stop, self.vetting, self._helpers)       # engines are often dropped, not closed
        self._requests: dict[str, _Request] = {}
        self._cache: collections.OrderedDict[str, xgr.CompiledGrammar] = collections.OrderedDict()
        self._compiling: dict[str, Future] = {}
        self.vetted: dict[str, Outcome] = {}          # constraints the workers compiled
        self._refused: dict[str, tuple[str, str, float]] = {}     # constraint -> kind, message, until
        self.drafts_refused = 0

    # -- the request lifecycle (engine thread) ---------------------------------------------------------------------
    def open(self, rid: str, params: Any, prompt: list[int]) -> bool:
        c = params.ext.get(KEY)
        if c is None:
            return False
        if params.ignore_eos:
            raise RequestError("client", "ignore_eos cannot be combined with a constraint (it ends at a stop token)")
        try:
            check(c)
        except ValueError as e:
            raise RequestError("client", str(e)) from None
        stops = sorted(self.stops | set(params.stop_token_ids))
        if not stops:
            raise RequestError("client", "a constraint needs a stop token; the model declares none")
        key = json.dumps(c, separators=(",", ":"))
        self._known_refusal(key)
        if key not in self._cache:
            self._prepare(key)
        row = np.zeros(self.mask_width, dtype=np.uint32)
        np.bitwise_or.at(row, np.array(stops) >> 5, np.uint32(1) << (np.array(stops, dtype=np.uint32) & 31))
        self._requests[rid] = _Request(key, stops, torch.from_numpy(row.view(np.int32)))
        return True

    def ready(self, rid: str) -> bool:
        req = self._requests[rid]
        if req.grammar is not None:
            return True
        for key, outcome in self.vetting.drain():
            if outcome.ok:
                self.vetted[key] = outcome
            else:
                self._refused[key] = outcome.kind, outcome.message, (
                    float("inf") if outcome.kind == "client" else time.monotonic() + BUSY_TTL)
        key = req.key
        if (compiled := self._cache.get(key)) is None:
            self._known_refusal(key)
            future = self._compiling.get(key)
            if future is None:
                if key in self.vetted:
                    self._compile(key)
                elif not self.vetting.alive:
                    raise RequestError("busy", "the constraint checker is down")
                return False
            if not future.done():
                return False
            del self._compiling[key]
            try:
                compiled = future.result()
            except Exception as e:                  # it compiled in the worker: ours to retry later
                self._refused[key] = "busy", f"compiling the constraint failed: {e}", time.monotonic() + BUSY_TTL
                raise RequestError("busy", f"compiling the constraint failed: {e}") from None
            self._cache[key] = compiled
            while sum(g.memory_size_bytes for g in self._cache.values()) > self.cache_bytes and len(self._cache) > 1:
                self._cache.popitem(last=False)
        self._cache.move_to_end(key)
        req.grammar = compiled
        return True

    def commit(self, seq: Any, out: list[int]) -> None:
        req = self._requests[seq.rid]
        if why := req.failed.pop(seq.index, None):
            raise RequestError("engine", why)
        matcher = self._matcher(req, seq.index)
        for t in out:
            if not matcher.accept_token(t):
                raise RequestError("engine", f"token {t} breaks the request's constraint")

    def close(self, rid: str) -> None:
        req = self._requests.pop(rid, None)
        if req is None or req.grammar is not None or any(r.key == req.key for r in self._requests.values()):
            return
        future = self._compiling.get(req.key)
        if future is not None and future.cancel():
            del self._compiling[req.key]
        self.vetting.cancel(req.key)

    def shutdown(self) -> None:
        _stop(self.vetting, self._helpers)

    # -- masks ---------------------------------------------------------------------------------------------------
    def fill(self, rows: list[Any], buf: np.ndarray) -> None:
        masks = torch.from_numpy(buf)
        for row in rows:
            req, out = self._requests[row.seq.rid], masks[row.start: row.start + row.n]
            try:
                self.drafts_refused += probe(self._matcher(req, row.seq.index), row.drafts, out, req.stop_row)
            except Exception as e:                  # never raise here: the row's commit reports it
                out[:] = -1
                req.failed[row.seq.index] = f"the constraint's mask failed: {e}"

    def apply(self, logits: torch.Tensor, mask: torch.Tensor, rows: torch.Tensor) -> None:
        kernel.apply(logits, mask, rows)

    # -- compiling -----------------------------------------------------------------------------------------------
    def _prepare(self, key: str) -> None:
        if key in self._compiling:
            return
        if key in self.vetted:
            self._compile(key)
        else:
            self.vetting.submit(key, key)

    def _compile(self, key: str) -> None:
        queued = sum(self.vetted[k].seconds for k, f in self._compiling.items() if not (f.running() or f.done()))
        if queued + self.vetted[key].seconds > self.backlog:
            raise RequestError("busy", f"{queued:.1f} s of constraint compiles are queued")
        self._compiling[key] = self._helpers.submit(self._build, key)

    def _build(self, key: str) -> xgr.CompiledGrammar:
        return self._compiler.compile_grammar(grammar(json.loads(key)))

    def _known_refusal(self, key: str) -> None:
        if (refused := self._refused.get(key)) is not None:
            kind, message, until = refused
            if time.monotonic() < until:
                raise RequestError(kind, message)
            del self._refused[key]

    def _matcher(self, req: _Request, index: int) -> xgr.GrammarMatcher:
        if (matcher := req.matchers.get(index)) is None:
            matcher = req.matchers[index] = xgr.GrammarMatcher(req.grammar, override_stop_tokens=req.stops)
        return matcher


def _stop(vetting: Vetting, helpers: ThreadPoolExecutor) -> None:
    vetting.shutdown()
    helpers.shutdown(wait=False, cancel_futures=True)
