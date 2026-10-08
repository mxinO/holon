"""Frontend ↔ engine boundary (design §7.8).

The frontend speaks HTTP and text; the engine speaks token ids. These records are the language-neutral IPC schema:
plain fields only, so a native frontend can implement the same protocol later.
"""
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Protocol

from .batch import SamplingParams

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator


@dataclasses.dataclass(slots=True)
class EngineRequest:
    rid: str
    token_ids: list[int]
    params: SamplingParams


@dataclasses.dataclass(slots=True)
class Logprobs:
    """Per position: the chosen token's logprob and the ranked top-k (ids and logprobs as parallel lists), from the
    raw logits (before temperature and filters). Prompt logprobs cover prompt positions 1 … n-1 (the first token
    has no context)."""

    token: list[float]
    top_ids: list[list[int]]
    top: list[list[float]]


@dataclasses.dataclass(slots=True)
class TokenDelta:
    rid: str
    token_ids: list[int]
    index: int = 0                     # which of the request's `n` sequences
    logprobs: Logprobs | None = None
    prompt_logprobs: Logprobs | None = None
    finish_reason: str | None = None   # "stop" | "length" | "abort" | "error"
    error: str | None = None           # why the request was rejected (finish_reason "error")


@dataclasses.dataclass(slots=True)
class Abort:
    rid: str


class Tokenizer(Protocol):
    eos_token_ids: frozenset[int]

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...


class Parser(Protocol):
    """Token-level reasoning/tool-call parser with a `channel()` state machine (design §7.8)."""

    def feed(self, text: str) -> Iterator[dict[str, Any]]: ...

    def channel(self) -> str: ...


class FrontendBundle(Protocol):
    """Shipped by the model plugin (`ModelPlugin.bundle(source)`): tokenizer, chat template, parsers."""

    model_name: str
    tokenizer: Tokenizer

    def render_chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
                    add_generation_prompt: bool = True, **template_kwargs: Any) -> list[int]:
        """Token ids of a chat, rendered with the model's chat template."""

    def reasoning_parser(self) -> Parser | None: ...

    def tool_parser(self) -> Parser | None: ...


class EngineClient(Protocol):
    """What a frontend gets from the core: the engine behind a process boundary, in token ids."""

    def generate(self, token_ids: list[int], params: SamplingParams, rid: str | None = None
                 ) -> AsyncIterator[TokenDelta]:
        """Submit a request and yield its deltas until each of its `params.n` sequences has finished; leaving the
        iteration early aborts the request."""

    async def abort(self, rid: str) -> None: ...

    async def stats(self) -> dict[str, Any]: ...


class FrontendPlugin(Protocol):
    """A `frontend` slot plugin: serves an API (HTTP, …) over the engine client until cancelled."""

    async def serve(self, client: EngineClient, bundle: FrontendBundle, host: str, port: int) -> None: ...
