"""Frontend ↔ engine boundary (design §7.8).

The frontend speaks HTTP and text; the engine speaks token ids. These records are the language-neutral IPC schema:
plain fields only, so a native frontend can implement the same protocol later.
"""
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Protocol

from .batch import SamplingParams

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclasses.dataclass(slots=True)
class EngineRequest:
    rid: str
    token_ids: list[int]
    params: SamplingParams


@dataclasses.dataclass(slots=True)
class Logprobs:
    """Per position: the chosen token's logprob and the ranked top-k (ids and logprobs as parallel lists)."""

    token: list[float]
    top_ids: list[list[int]]
    top: list[list[float]]


@dataclasses.dataclass(slots=True)
class TokenDelta:
    rid: str
    token_ids: list[int]
    logprobs: Logprobs | None = None
    prompt_logprobs: Logprobs | None = None
    finish_reason: str | None = None


@dataclasses.dataclass(slots=True)
class Abort:
    rid: str


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...


class Parser(Protocol):
    """Token-level reasoning/tool-call parser with a `channel()` state machine (design §7.8)."""

    def feed(self, text: str) -> Iterator[dict[str, Any]]: ...

    def channel(self) -> str: ...


class FrontendBundle(Protocol):
    """Shipped by the model plugin: tokenizer, chat template, parsers."""

    tokenizer: Tokenizer
    chat_template: str

    def reasoning_parser(self) -> Parser | None: ...

    def tool_parser(self) -> Parser | None: ...
