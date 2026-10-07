"""OpenAI response shapes: choices and stream chunks for chat and text completions, logprobs, usage."""
from __future__ import annotations

import itertools
from typing import Any

from holon.api import Logprobs, Tokenizer

from .stream import Piece

# A token's logprob row: its text, its logprob (None for the unscored first prompt token) and the ranked top-k.
Row = tuple[str, float | None, list[tuple[str, float]] | None]


def merge(pieces: list[Piece]) -> Piece:
    """One choice's pieces as a single piece (non-streaming responses)."""
    lps = [p.logprobs for p in pieces if p.logprobs is not None]
    return Piece(
        pieces[0].index, "".join(p.text for p in pieces), [t for p in pieces for t in p.token_ids],
        Logprobs(*(sum((getattr(lp, f) for lp in lps), []) for f in ("token", "top_ids", "top"))) if lps else None,
        next((p.prompt_logprobs for p in pieces if p.prompt_logprobs is not None), None), pieces[-1].finish_reason)


def usage(prompt_tokens: int, completion_tokens: int) -> dict[str, int]:
    return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens}


def _rows(tok: Tokenizer, ids: list[int], lp: Logprobs) -> list[Row]:
    def text(i: int) -> str:
        return tok.decode([i])
    return [(text(t), v, [(text(i), x) for i, x in zip(ti, tv, strict=True)])
            for t, v, ti, tv in zip(ids, lp.token, lp.top_ids, lp.top, strict=True)]


class Chat:
    object, chunk_object = "chat.completion", "chat.completion.chunk"

    def __init__(self, tok: Tokenizer) -> None:
        self.tok = tok

    def choice(self, index: int, p: Piece) -> dict[str, Any]:
        return {"index": index, "message": {"role": "assistant", "content": p.text}, "logprobs": self._logprobs(p),
                "finish_reason": p.finish_reason}

    def chunk(self, index: int, p: Piece, first: bool) -> dict[str, Any]:
        delta = {"role": "assistant", "content": p.text} if first else {"content": p.text} if p.text else {}
        return {"index": index, "delta": delta, "logprobs": self._logprobs(p), "finish_reason": p.finish_reason}

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

    def chunk(self, index: int, p: Piece, first: bool) -> dict[str, Any]:
        return self.choice(index, p, first)

    def _logprobs(self, index: int, rows: list[Row]) -> dict[str, Any]:
        start = self.offset.get(index, 0)
        offsets = list(itertools.accumulate((len(s) for s, _, _ in rows), initial=start))
        self.offset[index] = offsets.pop()
        return {"tokens": [s for s, _, _ in rows], "token_logprobs": [v for _, v, _ in rows],
                "top_logprobs": [None if top is None else dict(top) for _, _, top in rows], "text_offset": offsets}
