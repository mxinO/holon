"""OpenAI Responses (`/v1/responses`), the minimal subset of R §5.2 over the chat pipeline: `input` as text or items
(messages, `function_call`, `function_call_output`, `reasoning`), `instructions`, function tools, `tool_choice`,
`parallel_tool_calls`, `text.format`, `max_output_tokens`, sampling and `stream`. Nothing is stored: `store`,
`previous_response_id` and `background` are refused. The answer is a list of output items — reasoning, message,
function_call — or their server-sent events."""
from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import aclosing
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
from holon.api import EngineClient, FrontendBundle, SamplingParams
from pydantic import BaseModel, Field, field_validator, model_validator

from .pipeline import Block, Blocks, Chat, Conversation, pieces
from .schemas import response_format, tool_choice
from .stream import Piece
from .web import BadRequest, EventStream, collect, failure, rejected, unless_disconnected


class ResponsesRequest(BaseModel):
    model: str | None = None
    input: str | list[dict[str, Any]]
    instructions: str | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool = True
    text: dict[str, Any] | None = None
    max_output_tokens: int | None = Field(None, ge=1)
    temperature: float = Field(1.0, ge=0)
    top_p: float = Field(1.0, gt=0, le=1)
    stream: bool = False

    @model_validator(mode="before")
    @classmethod
    def _stateless(cls, data: Any) -> Any:
        if isinstance(data, dict) and (data.get("store") or data.get("previous_response_id") or data.get("background")):
            raise ValueError("store, previous_response_id and background need stored responses: none are kept")
        return data

    @field_validator("tool_choice")
    @classmethod
    def _choice(cls, choice: Any) -> str | dict[str, str] | None:
        return tool_choice(choice)

    def conversation(self) -> Conversation:
        tools = [_tool(t) for t in self.tools or []]
        fmt = response_format((self.text or {}).get("format"))
        return Conversation(_messages(self.instructions, self.input), tools,
                            self.tool_choice or ("auto" if tools else "none"), self.parallel_tool_calls, fmt)


def router(client: EngineClient, bundle: FrontendBundle, model_id: str) -> APIRouter:
    r = APIRouter()

    @r.post("/v1/responses")
    async def responses(req: ResponsesRequest, request: Request) -> Response:
        try:
            conv = req.conversation()
        except ValueError as e:
            raise BadRequest(str(e)) from e
        chat = Chat(bundle, conv)
        params = SamplingParams(temperature=req.temperature, top_p=req.top_p, max_tokens=req.max_output_tokens,
                                ext=chat.ext)
        rid = f"resp_{uuid.uuid4().hex}"
        gen = pieces(client, bundle, chat, params, rid, [])
        head = {"id": rid, "object": "response", "created_at": int(time.time()), "model": model_id,
                "error": None, "incomplete_details": None, "parallel_tool_calls": req.parallel_tool_calls}
        if req.stream:
            first = await unless_disconnected(request, anext(gen))
            if first is None:
                await gen.aclose()
                return Response(status_code=499)
            if first.finish_reason == "error":
                raise rejected(first)
            return EventStream(_events(head, len(chat.ids), first, gen))
        results = await unless_disconnected(request, collect([gen]))
        if results is None:
            return Response(status_code=499)
        if (failed := failure(results)) is not None:      # after the first token: a server error, never a 400
            raise BadRequest(failed.error or "generation failed", 500)
        (ps,) = results
        blocks = Blocks(_call_id)
        blocks.push([e for p in ps for e in p.events])
        return JSONResponse(_response(head, blocks, ps[-1].finish_reason, len(chat.ids),
                                      sum(len(p.token_ids) for p in ps)))

    return r


def _messages(instructions: str | None, items: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The input items as chat messages: a function call joins the assistant message before it (or starts one), a
    reasoning item becomes the next assistant message's reasoning."""
    messages: list[dict[str, Any]] = [{"role": "system", "content": instructions}] if instructions else []
    reasoning = ""
    for item in [{"role": "user", "content": items}] if isinstance(items, str) else items:
        kind = item.get("type", "message")
        if kind == "message":
            role = {"developer": "system"}.get(item.get("role"), item.get("role"))
            messages.append({"role": role, "content": _text(item.get("content"))})
        elif kind == "function_call":
            if not (messages and messages[-1]["role"] == "assistant"):
                messages.append({"role": "assistant", "content": ""})
            call = {"id": item.get("call_id"), "type": "function",
                    "function": {"name": item.get("name"), "arguments": item.get("arguments", "{}")}}
            messages[-1]["tool_calls"] = [*messages[-1].get("tool_calls", []), call]
        elif kind == "function_call_output":
            messages.append({"role": "tool", "tool_call_id": item.get("call_id"), "content": _text(item.get("output"))})
        elif kind == "reasoning":
            reasoning += "".join(part.get("text", "") for part in item.get("content") or item.get("summary") or [])
            continue
        else:
            raise ValueError(f"input item {kind!r}: message, function_call, function_call_output or reasoning")
        if reasoning and messages[-1]["role"] == "assistant":
            messages[-1]["reasoning_content"], reasoning = reasoning, ""
    return messages


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list) and all(isinstance(p, dict) and isinstance(p.get("text"), str) for p in content):
        return "".join(p["text"] for p in content)           # input_text / output_text parts
    raise ValueError(f"content {content!r:.80}: text or text parts only")


def _tool(tool: dict[str, Any]) -> dict[str, Any]:
    """A Responses function tool as a chat one. An absent `strict` is false, as in chat (D8: true would constrain
    every request with tools)."""
    if tool.get("type") != "function" or not isinstance(tool.get("name"), str) \
            or not isinstance(tool.get("parameters") or {}, dict):
        raise ValueError(f"tool {tool.get('name') or tool.get('type')!r}: function tools only")
    return {"type": "function", "function": {k: tool[k] for k in ("name", "description", "parameters", "strict")
                                             if k in tool}}


def _call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


def _item(b: Block, i: int, done: bool = True) -> dict[str, Any]:
    status = "completed" if done else "in_progress"
    if b.kind == "reasoning":
        return {"type": "reasoning", "id": f"rs_{i}", "summary": [],
                "content": [{"type": "reasoning_text", "text": b.text}] if done else []}
    if b.kind == "tool":
        return {"type": "function_call", "id": f"fc_{i}", "call_id": b.id, "name": b.name,
                "arguments": b.text if done else "", "status": status}
    return {"type": "message", "id": f"msg_{i}", "role": "assistant", "status": status,
            "content": [_part(b.text)] if done else []}


def _part(text: str) -> dict[str, Any]:
    return {"type": "output_text", "text": text, "annotations": []}


def _response(head: dict[str, Any], blocks: Blocks, reason: str | None, n_prompt: int, n_output: int
              ) -> dict[str, Any]:
    incomplete = reason == "length"
    return {**head, "status": "incomplete" if incomplete else "completed",
            "incomplete_details": {"reason": "max_output_tokens"} if incomplete else None,
            "output": [_item(b, i) for i, b in enumerate(blocks.blocks)],
            "usage": {"input_tokens": n_prompt, "output_tokens": n_output, "total_tokens": n_prompt + n_output}}


async def _events(head: dict[str, Any], n_prompt: int, first: Piece, gen: AsyncIterator[Piece]) -> AsyncIterator[str]:
    """response.created and .in_progress; per item output_item.added, its deltas (output_text, reasoning_text,
    function_call_arguments) and their .done, output_item.done; response.completed (or .incomplete). A failure after
    the first token is response.failed. Every event has its sequence number."""
    seq = iter(range(1 << 62))

    def sse(kind: str, **data: Any) -> str:
        return f"event: {kind}\ndata: {json.dumps({'type': kind, 'sequence_number': next(seq), **data})}\n\n"

    created = {**head, "status": "in_progress", "output": []}
    yield sse("response.created", response=created)
    yield sse("response.in_progress", response=created)
    blocks, n_output, reason = Blocks(_call_id), 0, None
    async with aclosing(gen):
        p: Piece | None = first
        while p is not None:
            if p.finish_reason == "error":
                error = {"code": "server_error", "message": p.error or "generation failed"}
                yield sse("response.failed", response={**created, "status": "failed", "error": error})
                return
            n_output, reason = n_output + len(p.token_ids), p.finish_reason
            for step, i, text in blocks.push(p.events) + (blocks.close() if reason else []):
                b, at = blocks.blocks[i], {"output_index": i}
                ids = at | {"item_id": _item(b, i)["id"]} | ({"content_index": 0} if b.kind != "tool" else {})
                if step == "start":
                    yield sse("response.output_item.added", **at, item=_item(b, i, done=False))
                    if b.kind == "text":
                        yield sse("response.content_part.added", **ids, part=_part(""))
                elif step == "delta":
                    kind = {"text": "output_text", "reasoning": "reasoning_text", "tool": "function_call_arguments"}
                    yield sse(f"response.{kind[b.kind]}.delta", **ids, delta=text)
                else:
                    if b.kind == "tool":
                        yield sse("response.function_call_arguments.done", **ids, arguments=b.text)
                    else:
                        yield sse(f"response.{'output' if b.kind == 'text' else 'reasoning'}_text.done", **ids,
                                  text=b.text)
                    if b.kind == "text":
                        yield sse("response.content_part.done", **ids, part=_part(b.text))
                    yield sse("response.output_item.done", **at, item=_item(b, i))
            p = await anext(gen, None)
    final = _response(head, blocks, reason, n_prompt, n_output)
    yield sse(f"response.{final['status']}", response=final)
