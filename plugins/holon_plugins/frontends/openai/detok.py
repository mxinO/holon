"""Incremental detokenization."""
from __future__ import annotations

from collections.abc import Sequence

from holon.api import Tokenizer

_CONTEXT = 4   # prompt tokens decoded ahead of the first output token, for tokenizers whose text depends on context


class Detokenizer:
    """Turns a growing token sequence into text deltas.

    Decoding token by token is wrong twice over: a multi-byte character can span tokens (byte-level BPE renders the
    incomplete bytes as U+FFFD), and some tokenizers render a token differently after context (SentencePiece's
    leading space). So each call decodes a short window — the tokens released last time as context, then the new
    ones — and releases the text past the context only once it no longer ends in an incomplete character.
    """

    def __init__(self, tokenizer: Tokenizer, context: Sequence[int] = ()) -> None:
        self.tokenizer = tokenizer
        self.window = list(context[-_CONTEXT:])
        self.read = len(self.window)        # window[:read] is context whose text was already released (or is prompt)

    def push(self, ids: Sequence[int], final: bool = False) -> str:
        """Add tokens and return the newly complete text; `final` releases whatever is left, complete or not."""
        self.window += ids
        if self.read == len(self.window):
            return ""
        before = self.tokenizer.decode(self.window[:self.read])
        text = self.tokenizer.decode(self.window)
        if not final and (len(text) <= len(before) or text.endswith("�")):
            return ""
        self.window, self.read = self.window[self.read:], len(self.window) - self.read
        return text[len(before):]
