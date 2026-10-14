"""Frontend ↔ engine boundary (design §7.8).

The frontend speaks HTTP and text; the engine speaks token ids. These records are the language-neutral IPC schema:
plain fields only, so a native frontend can implement the same protocol later.
"""
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Protocol

from .batch import SamplingParams

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@dataclasses.dataclass(slots=True)
class EngineRequest:
    rid: str
    token_ids: list[int]
    params: SamplingParams
    resume_output: int = 0             # trailing tokens of `token_ids` that are output already (a resubmission)


@dataclasses.dataclass(slots=True)
class Logprobs:
    """Per position: the chosen token's logprob and the ranked top-k (ids and logprobs as parallel lists), from the
    raw logits (before temperature and filters) — masked for a constrained request, so a token it may not take has
    logprob −inf (frontends clamp it to −9999: JSON has no −inf). Prompt logprobs cover prompt positions 1 … n-1
    (the first token has no context)."""

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
    kind: str | None = None            # and the error's kind: "client" | "busy" | "engine" (`RequestError`)


@dataclasses.dataclass(slots=True)
class Abort:
    rid: str


class Tokenizer(Protocol):
    eos_token_ids: frozenset[int]

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...


@dataclasses.dataclass(slots=True)
class Text:
    text: str


@dataclasses.dataclass(slots=True)
class Reasoning:
    text: str


@dataclasses.dataclass(slots=True)
class ToolCallStart:
    index: int                         # the call's place in the response
    name: str


@dataclasses.dataclass(slots=True)
class ToolArgsDelta:
    index: int
    text: str                          # the next piece of the call's arguments as JSON (values typed by the schema)


@dataclasses.dataclass(slots=True)
class ToolCallEnd:
    index: int


ParseEvent = Text | Reasoning | ToolCallStart | ToolArgsDelta | ToolCallEnd


class FrontendBundle(Protocol):
    """Shipped by the model plugin (`ModelPlugin.bundle(source)`): tokenizer and chat template. The formats its
    template writes reasoning and tool calls in are named with versions (e.g. "hermes@1", "think@1"); the frontend
    implements each named format once — its parser and its constraint builder."""

    model_name: str
    tokenizer: Tokenizer
    tool_format: str | None            # None: the template has no tool calls
    reasoning_format: str | None

    def render_chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
                    add_generation_prompt: bool = True, **template_kwargs: Any) -> list[int]:
        """Token ids of a chat, rendered with the model's chat template."""


class EngineClient(Protocol):
    """What a frontend gets from the core: the engine behind a process boundary, in token ids."""

    def generate(self, token_ids: list[int], params: SamplingParams, rid: str | None = None, resume_output: int = 0
                 ) -> AsyncIterator[TokenDelta]:
        """Submit a request and yield its deltas until each of its `params.n` sequences has finished (an "error"
        delta finishes them all); leaving the iteration early aborts the request. `resume_output`: the trailing
        tokens of `token_ids` that are output already (`EngineRequest`)."""

    async def abort(self, rid: str) -> None: ...

    async def stats(self) -> dict[str, Any]: ...


class FrontendPlugin(Protocol):
    """A `frontend` slot plugin: serves an API (HTTP, …) over the engine client until cancelled; `options` are the
    deployment's plugin options (`EngineConfig.plugin_options`), of which it reads its own namespace."""

    async def serve(self, client: EngineClient, bundle: FrontendBundle, host: str, port: int,
                    options: dict[str, object]) -> None: ...
