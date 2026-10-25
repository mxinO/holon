"""GLM-5.3-Flash's frontend bundle (docs/v1-glm.md D13): the GGUF's tokenizer and chat template, its reasoning in
`<think>` (the template opens it: `think@1`) and its tool calls as `glm_xml@1` (`<tool_call>NAME<arg_key>K</arg_key>
<arg_value>V</arg_value>…</tool_call>`). The GGUF's stop tokens (`<|endoftext|>`, `<|user|>` as eot, `<|observation|>`
as eom) come from the weight source.

`Tokenizer` and `_parsed` are carried over from `models.qwen4_exp`'s bundle (`derived_from`); the recipe is written
fresh."""
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
    tool_format = "glm_xml@1"

    def __init__(self, source: Any) -> None:
        self.model_name = source.name
        self.tokenizer = Tokenizer(source)

    def render_chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
                    add_generation_prompt: bool = True, **template_kwargs: Any) -> list[int]:
        text = self.tokenizer._tok.apply_chat_template(_parsed(messages), tools=tools, tokenize=False,
                                                       add_generation_prompt=add_generation_prompt, **template_kwargs)
        return self.tokenizer.encode(text)


def _parsed(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tool-call arguments sent as JSON text (OpenAI's wire form) as objects: the template iterates their items."""
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
