"""Qwen3.8-Flash-Next's frontend bundle, copied from `models.qwen3_5`'s (`derived_from`): the checkpoint's tokenizer
and chat template, from whichever source holds the weights (a GGUF file), and the formats the template writes
reasoning and tool calls in (docs/m05-grammar.md §3.8; the frontend parses and constrains each format); the tool
format is read off the template."""
from __future__ import annotations

import json
from typing import Any


class Tokenizer:
    def __init__(self, source: Any) -> None:
        self._tok = source.tokenizer()

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=True)


class Bundle:
    reasoning_format = "think@1"

    def __init__(self, source: Any) -> None:
        self.model_name = source.name
        self.tokenizer = Tokenizer(source)
        self.tool_format = _tool_format(self._render)

    def render_chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
                    add_generation_prompt: bool = True, **template_kwargs: Any) -> list[int]:
        return self.tokenizer.encode(self._render(_parsed(messages), tools, add_generation_prompt, **template_kwargs))

    def _render(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None,
                add_generation_prompt: bool, **template_kwargs: Any) -> str:
        return self.tokenizer._tok.apply_chat_template(messages, tools=tools, tokenize=False,
                                                       add_generation_prompt=add_generation_prompt, **template_kwargs)


def _parsed(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tool-call arguments sent as JSON text (OpenAI's wire form) as objects: the XML templates iterate them (the
    Qwen3.8 GGUF one raises on text, R §4.1)."""
    def parsed(call: dict[str, Any]) -> dict[str, Any]:
        fn = call.get("function")
        if not isinstance(fn, dict) or not isinstance(fn.get("arguments"), str):
            return call
        try:
            args = json.loads(fn["arguments"] or "{}")
        except ValueError:
            return call
        return {**call, "function": {**fn, "arguments": args}} if isinstance(args, dict) else call
    return [{**m, "tool_calls": [parsed(c) for c in m["tool_calls"]]} if m.get("tool_calls") else m for m in messages]


def _tool_format(render: Any) -> str | None:
    """The tool-call format the template writes, read off a rendered call with a boolean argument: Qwen's XML with
    values written by Jinja's `| string` (Python's True: qwen_xml@1, Qwen3.5) or as JSON (qwen_xml@2, Qwen3.6+)."""
    call = {"type": "function", "function": {"name": "f", "arguments": {"b": True}}}
    try:
        text = render([{"role": "user", "content": "x"}, {"role": "assistant", "content": "", "tool_calls": [call]}],
                      [{"type": "function", "function": {"name": "f", "parameters": {}}}], False)
    except Exception:  # noqa: BLE001 — a template without tool calls
        return None
    if "<function=f>" not in text:
        return None
    return "qwen_xml@1" if "\nTrue\n" in text else "qwen_xml@2"
