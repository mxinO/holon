"""Anthropic Messages (`/v1/messages`, `/v1/messages/count_tokens`) over the chat pipeline (R §5.3): content blocks
become OpenAI messages (`tool_use` → tool calls, `tool_result` → tool messages ahead of the turn's text, `thinking` →
reasoning), tools become function tools; the answer's blocks come back as thinking, text and tool_use blocks, or as
Anthropic's server-sent events.

Claude Code's first system block, `x-anthropic-billing-header: …`, changes with every request and would defeat
prefix caching (R §5.4): it is dropped unless the app is built with `drop_billing_header=False` (counted in /metrics;
`CLAUDE_CODE_ATTRIBUTION_HEADER=0` stops the client sending it)."""
from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from contextlib import aclosing
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
from holon.api import EngineClient, FrontendBundle, SamplingParams
from pydantic import BaseModel, Field

from .pipeline import Block, Blocks, Chat, Conversation, pieces
from .stream import Piece
from .web import BadRequest, EventStream, collect, failure, rejected, unless_disconnected

BILLING = "x-anthropic-billing-header:"
_CHOICES = {"auto": "auto", "any": "required", "none": "none"}


class CountRequest(BaseModel):
    model: str | None = None
    messages: list[dict[str, Any]] = Field(min_length=1)
    system: str | list[dict[str, Any]] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: dict[str, Any] | None = None
    thinking: dict[str, Any] | None = None


class MessagesRequest(CountRequest):
    max_tokens: int = Field(ge=1)
    stop_sequences: list[str] | None = None
    stream: bool = False
    temperature: float = Field(1.0, ge=0)
    top_p: float = Field(1.0, gt=0, le=1)
    top_k: int = Field(0, ge=0)


def error_body(status: int, message: str) -> dict[str, Any]:
    kind = {400: "invalid_request_error", 503: "overloaded_error"}.get(status, "api_error")
    return {"type": "error", "error": {"type": kind, "message": message}}


def router(client: EngineClient, bundle: FrontendBundle, drop_billing_header: bool, counters: dict[str, int]
           ) -> APIRouter:
    r = APIRouter()

    def prepare(req: CountRequest) -> Chat:
        conv, dropped = conversation(req, drop_billing_header)
        counters["billing_headers_dropped"] += dropped
        return Chat(bundle, conv)

    @r.post("/v1/messages/count_tokens")
    async def count_tokens(req: CountRequest) -> dict[str, int]:
        return {"input_tokens": len(prepare(req).ids)}

    @r.post("/v1/messages")
    async def messages(req: MessagesRequest, request: Request) -> Response:
        chat = prepare(req)
        params = SamplingParams(temperature=req.temperature, top_p=req.top_p, top_k=req.top_k,
                                max_tokens=req.max_tokens, ext=chat.ext)
        rid = f"msg_{uuid.uuid4().hex}"
        gen = pieces(client, bundle, chat, params, rid, [s for s in req.stop_sequences or [] if s])
        message = {"id": rid, "type": "message", "role": "assistant", "model": bundle.model_name}
        if req.stream:
            first = await unless_disconnected(request, anext(gen))
            if first is None:
                await gen.aclose()
                return Response(status_code=499)
            if first.finish_reason == "error":
                raise rejected(first)
            return EventStream(_events(message, len(chat.ids), first, gen), _sse("ping", {"type": "ping"}))
        results = await unless_disconnected(request, collect([gen]))
        if results is None:
            return Response(status_code=499)
        if (failed := failure(results)) is not None:      # after the first token: a server error, never a 400
            raise BadRequest(failed.error or "generation failed", 500)
        (ps,) = results
        blocks = Blocks(_tool_id)
        blocks.push([e for p in ps for e in p.events])
        return JSONResponse({**message, "content": [_block(b) for b in blocks.blocks] or [_block(Block("text"))],
                             "stop_reason": _stop(ps[-1].finish_reason, blocks), "stop_sequence": None,
                             "usage": {"input_tokens": len(chat.ids),
                                       "output_tokens": sum(len(p.token_ids) for p in ps)}})

    return r


def conversation(req: CountRequest, drop_billing_header: bool) -> tuple[Conversation, bool]:
    """The request as the pipeline's conversation, and whether a billing-header block was dropped."""
    system = [req.system] if isinstance(req.system, str) else [_text(b) for b in req.system or []]
    kept = [s for s in system if not (drop_billing_header and s.startswith(BILLING))]
    messages: list[dict[str, Any]] = [{"role": "system", "content": "\n".join(kept)}] if kept else []
    for m in req.messages:
        blocks = m.get("content")
        blocks = [{"type": "text", "text": blocks}] if isinstance(blocks, str) else blocks
        if not isinstance(blocks, list) or m.get("role") not in ("user", "assistant"):
            raise BadRequest("messages are {role: user | assistant, content: text or blocks}")
        messages += _user(blocks) if m["role"] == "user" else [_assistant(blocks)]
    choice = req.tool_choice or {"type": "auto"}
    if choice.get("type") == "tool" and isinstance(choice.get("name"), str):
        tool_choice: str | dict[str, str] = {"name": choice["name"]}
    elif choice.get("type") in _CHOICES:
        tool_choice = _CHOICES[choice["type"]]
    else:
        raise BadRequest(f"tool_choice {choice!r}: auto, any, tool or none")
    kwargs = {"enable_thinking": False} if (req.thinking or {}).get("type") == "disabled" else {}
    conv = Conversation(messages, [_tool(t) for t in req.tools or []], tool_choice,
                        not choice.get("disable_parallel_tool_use", False), None, kwargs)
    return conv, len(kept) < len(system)


def _text(block: Any) -> str:
    if not isinstance(block, dict) or block.get("type") != "text" or not isinstance(block.get("text"), str):
        raise BadRequest(f"a text block, not {block!r:.80}: this server reads text only")
    return block["text"]


def _user(blocks: list[Any]) -> list[dict[str, Any]]:
    """Tool results become tool messages, ahead of the turn's text (as vLLM and SGLang)."""
    results, texts = [], []
    for b in blocks:
        if isinstance(b, dict) and b.get("type") == "tool_result":
            content = b.get("content", "")
            content = content if isinstance(content, str) else "\n".join(_text(c) for c in content)
            results.append({"role": "tool", "tool_call_id": b.get("tool_use_id"), "content": content})
        else:
            texts.append(_text(b))
    return results + ([{"role": "user", "content": "\n".join(texts)}] if texts else [])


def _assistant(blocks: list[Any]) -> dict[str, Any]:
    texts, reasoning, calls = [], [], []
    for b in blocks:
        kind = b.get("type") if isinstance(b, dict) else None
        if kind == "thinking":
            reasoning.append(b.get("thinking", ""))
        elif kind == "tool_use":
            calls.append({"id": b.get("id"), "type": "function", "function": {
                "name": b.get("name"), "arguments": json.dumps(b.get("input", {}), ensure_ascii=False)}})
        elif kind != "redacted_thinking":
            texts.append(_text(b))
    message = {"role": "assistant", "content": "\n".join(texts)}
    return message | ({"reasoning_content": "\n".join(reasoning)} if reasoning else {}) | (
        {"tool_calls": calls} if calls else {})


def _tool(tool: dict[str, Any]) -> dict[str, Any]:
    if tool.get("type") not in (None, "custom") or not isinstance(tool.get("name"), str) \
            or not isinstance(tool.get("input_schema") or {}, dict):
        raise BadRequest(f"tool {tool.get('name') or tool.get('type')!r}: only client tools {{name, input_schema}}")
    fn = {"name": tool["name"]} | {k: tool[k] for k in ("description", "strict") if k in tool}
    return {"type": "function", "function": fn | {"parameters": tool.get("input_schema") or {"type": "object"}}}


def _tool_id() -> str:
    return f"toolu_{uuid.uuid4().hex[:24]}"


def _stop(reason: str | None, blocks: Blocks) -> str:
    return "tool_use" if blocks.calls and reason == "stop" else "max_tokens" if reason == "length" else "end_turn"


def _block(b: Block) -> dict[str, Any]:
    if b.kind == "reasoning":
        return {"type": "thinking", "thinking": b.text, "signature": ""}
    if b.kind == "tool":
        return {"type": "tool_use", "id": b.id, "name": b.name, "input": json.loads(b.text)}
    return {"type": "text", "text": b.text}


async def _events(message: dict[str, Any], n_prompt: int, first: Piece, gen: AsyncIterator[Piece]
                  ) -> AsyncIterator[str]:
    """message_start; each block's content_block_start, _delta… and _stop; message_delta (the stop reason, the
    output tokens); message_stop. A failure after the first token is an `error` event."""
    yield _sse("message_start", {"type": "message_start", "message": {
        **message, "content": [], "stop_reason": None, "stop_sequence": None,
        "usage": {"input_tokens": n_prompt, "output_tokens": 0}}})
    blocks, n_output, reason = Blocks(_tool_id), 0, None
    async with aclosing(gen):
        p: Piece | None = first
        while p is not None:
            if p.finish_reason == "error":
                yield _sse("error", error_body(500, p.error or "generation failed"))
                return
            n_output, reason = n_output + len(p.token_ids), p.finish_reason
            for step, i, text in blocks.push(p.events) + (blocks.close() if reason else []):
                b = blocks.blocks[i]
                if step == "start":                     # empty: `b` may hold this piece's deltas already
                    start = _block(Block(b.kind, b.name, b.id, ["{}"] if b.kind == "tool" else []))
                    yield _sse("content_block_start", {"type": "content_block_start", "index": i,
                                                       "content_block": start})
                elif step == "delta":
                    delta = {"reasoning": {"type": "thinking_delta", "thinking": text},
                             "text": {"type": "text_delta", "text": text},
                             "tool": {"type": "input_json_delta", "partial_json": text}}[b.kind]
                    yield _sse("content_block_delta", {"type": "content_block_delta", "index": i, "delta": delta})
                else:
                    if b.kind == "reasoning":
                        yield _sse("content_block_delta", {"type": "content_block_delta", "index": i,
                                                           "delta": {"type": "signature_delta", "signature": ""}})
                    yield _sse("content_block_stop", {"type": "content_block_stop", "index": i})
            p = await anext(gen, None)
    yield _sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": _stop(reason, blocks),
                                                                    "stop_sequence": None},
                                 "usage": {"output_tokens": n_output}})
    yield _sse("message_stop", {"type": "message_stop"})


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
