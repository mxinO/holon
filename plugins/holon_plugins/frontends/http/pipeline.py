"""The chat pipeline every dialect's router shares (D7): a `Conversation` (the router's normalized request) is rendered
with the bundle's chat template, constrained as its tool choice or response format asks (in the bundle's formats,
`formats.py`), generated, and parsed into typed events per choice; `Blocks` groups an answer's events into the
reasoning, text and tool-call blocks the dialects return."""
from __future__ import annotations

import dataclasses
from collections.abc import AsyncIterator
from contextlib import aclosing
from typing import Any

from holon.api import (
    EngineClient,
    FrontendBundle,
    ParseEvent,
    Reasoning,
    SamplingParams,
    Text,
    ToolArgsDelta,
    ToolCallEnd,
    ToolCallStart,
)

from . import formats
from .stream import Piece, generate
from .web import BadRequest

KEY = "grammar.constraint@1"        # the constraint resource (D6), served by the selected logits processor
_TAIL = 16                          # prompt tokens whose text tells whether the template opened the reasoning


@dataclasses.dataclass
class Conversation:
    """A chat request in the form every router builds: OpenAI chat messages (assistant tool calls with JSON-string
    arguments, results as role "tool", reasoning as `reasoning_content`) and OpenAI function tools."""

    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    tool_choice: str | dict[str, str] = "auto"         # "auto" | "none" | "required" | {"name": …}
    parallel_tool_calls: bool = True
    response_format: dict[str, Any] | None = None       # a json_schema or json_object constraint
    template_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)


class Chat:
    """A conversation rendered for the bundle: its prompt, the constraint (if any) and a parser per choice.

    Tools are parsed in the bundle's `tool_format` unless `tool_choice` is "none" (then the template does not see
    them either). The output is constrained (D8) for a named or required tool choice, `parallel_tool_calls: false`
    or a strict tool — the call envelope, and each call's arguments by its schema (strict for a strict tool) — or for
    a response format; the two together are refused."""

    def __init__(self, bundle: FrontendBundle, conv: Conversation) -> None:
        tools = [] if conv.tool_choice == "none" else conv.tools
        names = [t["function"]["name"] for t in tools]
        named = conv.tool_choice.get("name") if isinstance(conv.tool_choice, dict) else None
        if conv.tool_choice not in ("auto", "none") and not tools:
            raise BadRequest(f"tool_choice {conv.tool_choice!r} needs tools")
        if named is not None and named not in names:
            raise BadRequest(f"tool_choice names {named!r}, which is not among the tools")
        try:
            self.ids = bundle.render_chat(conv.messages, tools=tools or None, **conv.template_kwargs)
        except Exception as e:  # noqa: BLE001 — a template that rejects the conversation is a client error
            raise BadRequest(f"chat template: {e}") from e
        self.tail = bundle.tokenizer.decode(self.ids[-_TAIL:])
        self.reasoning = formats.REASONING.get(bundle.reasoning_format or "")
        self.tool_format = bundle.tool_format if tools and bundle.tool_format in formats.TOOLS else None
        self.tools = tools if self.tool_format else []
        self.ends = (formats.CALL,) if self.tool_format else ()      # every tool format's calls open with it

        calls = None
        strict = any(t["function"].get("strict") for t in tools)
        if tools and (named or conv.tool_choice == "required" or not conv.parallel_tool_calls or strict):
            if self.tool_format is None:
                raise BadRequest(f"tool_choice / parallel_tool_calls / strict need a known tool format; this model's "
                                 f"is {bundle.tool_format!r}")
            call = formats.CALLS[self.tool_format]
            calls = call(tools[names.index(named)]) if named else {
                "type": "triggered_tags", "triggers": [formats.CALL], "tags": [call(t) for t in tools],
                "at_least_one": conv.tool_choice == "required", "stop_after_first": not conv.parallel_tool_calls}
        if calls and conv.response_format:
            raise BadRequest("response_format cannot be combined with a constrained tool choice")
        body = calls or conv.response_format
        if body and self.reasoning is not None:
            body = self.reasoning.constraint(self.tail, body, self.ends, calls is not None)
        self.constraint = body

    @property
    def ext(self) -> dict[str, Any]:
        return {KEY: self.constraint} if self.constraint else {}

    def parser(self) -> Parse:
        reasoning = self.reasoning(self.tail, self.ends) if self.reasoning else None
        return Parse(reasoning, formats.TOOLS[self.tool_format](self.tools) if self.tool_format else None)


class Parse:
    """One choice's parser: reasoning first, then tool calls in what is not reasoning; each run of text or reasoning
    is stripped at both ends (the formats put newlines around their markers) — holding back only trailing
    whitespace, so the events still do not depend on how the text is cut."""

    def __init__(self, reasoning: Any, tools: Any) -> None:
        self.reasoning, self.tools = reasoning, tools
        self.kind: type | None = None         # the current run's event type
        self.started, self.space = False, ""  # whether the run has text yet; its held-back trailing whitespace

    def feed(self, text: str, final: bool = False) -> list[ParseEvent]:
        events = self.reasoning.feed(text, final) if self.reasoning else [Text(text)]
        if self.tools:
            events = [x for e in events for x in (self.tools.feed(e.text) if isinstance(e, Text) else [e])]
            events += self.tools.feed("", True) if final else []
        out: list[ParseEvent] = []
        for e in events:
            if not isinstance(e, Text | Reasoning):
                self.kind = None
                out.append(e)
                continue
            if type(e) is not self.kind:
                self.kind, self.started, self.space = type(e), False, ""
            text = self.space + e.text if self.started else e.text.lstrip()
            body = text.rstrip()
            self.space = text[len(body):]
            if body:
                out.append(type(e)(body))
                self.started = True
        return out


async def pieces(client: EngineClient, bundle: FrontendBundle, chat: Chat, params: SamplingParams, rid: str,
                 stop: list[str]) -> AsyncIterator[Piece]:
    """A chat's pieces (`stream.generate`), each with the parse events of its text."""
    parsers = [chat.parser() for _ in range(params.n)]
    async with aclosing(generate(client, bundle.tokenizer, chat.ids, params, rid, stop)) as gen:
        async for p in gen:
            if p.finish_reason != "error":
                p.events = parsers[p.index].feed(p.text, final=p.finish_reason is not None)
            yield p


@dataclasses.dataclass
class Block:
    kind: str                           # "reasoning" | "text" | "tool"
    text: str = ""                      # the reasoning, the text, or a tool call's arguments (JSON)
    name: str = ""                      # a tool call's function
    id: str = ""                        # a tool call's id


class Blocks:
    """An answer's events as blocks — reasoning, text and tool calls in their order — each opened, extended and
    closed: the shape of Anthropic's content blocks and of the Responses API's output items. `push` returns the
    steps taken: ("start" | "delta" | "stop", block index, delta text)."""

    def __init__(self, call_id: Any) -> None:
        self.call_id = call_id              # () -> a new tool call id, in the dialect's form
        self.blocks: list[Block] = []
        self.open = False

    @property
    def calls(self) -> list[Block]:
        return [b for b in self.blocks if b.kind == "tool"]

    def push(self, events: list[ParseEvent]) -> list[tuple[str, int, str]]:
        steps: list[tuple[str, int, str]] = []
        for e in events:
            if isinstance(e, ToolCallStart):
                steps += self.close()
                steps += self._start(Block("tool", name=e.name, id=self.call_id()))
            elif isinstance(e, ToolArgsDelta):
                steps += self._delta(e.text)
            elif isinstance(e, ToolCallEnd):
                steps += self.close()
            else:
                kind = "reasoning" if isinstance(e, Reasoning) else "text"
                if not (self.open and self.blocks[-1].kind == kind):
                    steps += self.close() + self._start(Block(kind))
                steps += self._delta(e.text)
        return steps

    def close(self) -> list[tuple[str, int, str]]:
        was, self.open = self.open, False
        return [("stop", len(self.blocks) - 1, "")] if was else []

    def _start(self, block: Block) -> list[tuple[str, int, str]]:
        self.blocks.append(block)
        self.open = True
        return [("start", len(self.blocks) - 1, "")]

    def _delta(self, text: str) -> list[tuple[str, int, str]]:
        self.blocks[-1].text += text
        return [("delta", len(self.blocks) - 1, text)]
