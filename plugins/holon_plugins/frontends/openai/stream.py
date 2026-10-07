"""Engine deltas → text pieces per choice: detokenization, stop strings, and aborting what is no longer read."""
from __future__ import annotations

import dataclasses
from collections.abc import AsyncIterator

import anyio
from holon.api import EngineClient, Logprobs, SamplingParams, TokenDelta, Tokenizer

from .detok import Detokenizer


@dataclasses.dataclass(slots=True)
class Piece:
    """What one engine delta adds to one choice. `token_ids` and `logprobs` cover every generated token (the
    terminating EOS/stop token included); `text` excludes that token and anything from a stop string on."""

    index: int
    text: str
    token_ids: list[int]
    logprobs: Logprobs | None = None
    prompt_logprobs: Logprobs | None = None
    finish_reason: str | None = None   # "stop" | "length" | "abort" | "error"
    error: str | None = None


class Choice:
    """One of a request's `n` sequences as text. Stop strings are matched here, across token boundaries; text that
    could still turn out to start a stop string is held back until it cannot."""

    def __init__(self, tokenizer: Tokenizer, prompt_ids: list[int], stop: list[str]) -> None:
        self.detok = Detokenizer(tokenizer, prompt_ids)
        self.stop = stop
        self.hold = max(map(len, stop), default=1) - 1
        self.text = ""
        self.sent = 0                      # text[:sent] has been handed out
        self.finish_reason: str | None = None

    def feed(self, d: TokenDelta) -> Piece:
        ids = d.token_ids[:-1] if d.finish_reason == "stop" else d.token_ids   # the engine stopped on EOS/stop id
        searched = max(0, len(self.text) - self.hold)                          # a new match cannot start earlier
        self.text += self.detok.push(ids, final=d.finish_reason is not None)
        self.finish_reason = d.finish_reason
        hits = [i for i in (self.text.find(s, searched) for s in self.stop) if i >= 0]
        if hits:
            self.text, self.finish_reason = self.text[:min(hits)], "stop"
        end = len(self.text) if self.finish_reason else max(self.sent, len(self.text) - self.hold)
        text, self.sent = self.text[self.sent:end], end
        return Piece(d.index, text, d.token_ids, d.logprobs, d.prompt_logprobs, self.finish_reason)


async def generate(client: EngineClient, tokenizer: Tokenizer, prompt_ids: list[int], params: SamplingParams,
                   rid: str, stop: list[str]) -> AsyncIterator[Piece]:
    """Stream a request's pieces until every choice has finished. A rejection is a single piece with finish_reason
    "error". Whenever iteration ends before the engine finished every sequence — all choices hit stop strings, or
    the consumer went away (cancellation, `aclose`) — the request is aborted."""
    choices = [Choice(tokenizer, prompt_ids, stop) for _ in range(params.n)]
    running = set(range(params.n))     # sequences the engine has not finished
    deltas = client.generate(prompt_ids, params, rid)
    try:
        async for d in deltas:
            if d.finish_reason == "error":
                running.clear()
                yield Piece(d.index, "", [], finish_reason="error", error=d.error or "request rejected")
                return
            if d.finish_reason:
                running.discard(d.index)
            choice = choices[d.index]
            if choice.finish_reason is None:                # else it ended on a stop string; the engine runs on
                yield choice.feed(d)
            if all(c.finish_reason for c in choices):
                return
    finally:
        # Shielded: a client disconnect cancels this task, and the engine must still hear of it. The client's own
        # abort-on-close is not enough under that cancellation, hence the explicit one.
        with anyio.CancelScope(shield=True):
            await deltas.aclose()   # type: ignore[attr-defined]  # an async generator in practice
            if running:
                await client.abort(rid)
