"""Qwen3's frontend bundle: the checkpoint's tokenizer and chat template, from whichever source holds the weights
(HF directory or GGUF file). Reasoning and tool-call parsers arrive in M0.5."""
from __future__ import annotations

from typing import Any


class Tokenizer:
    def __init__(self, source: Any) -> None:
        self._tok = source.tokenizer()
        self.eos_token_ids = source.eos_token_ids

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=True)


class Bundle:
    tool_format = reasoning_format = None          # named (and their parsers built) in M0.5 step 3

    def __init__(self, source: Any) -> None:
        self.model_name = source.name
        self.tokenizer = Tokenizer(source)

    def render_chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
                    add_generation_prompt: bool = True, **template_kwargs: Any) -> list[int]:
        text = self.tokenizer._tok.apply_chat_template(messages, tools=tools, tokenize=False,
                                                       add_generation_prompt=add_generation_prompt, **template_kwargs)
        return self.tokenizer.encode(text)

    def reasoning_parser(self) -> None:
        return None

    def tool_parser(self, tools: list[dict[str, Any]]) -> None:
        return None
