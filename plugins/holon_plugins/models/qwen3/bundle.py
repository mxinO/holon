"""Qwen3's frontend bundle: the checkpoint's tokenizer and chat template (via transformers). Reasoning and tool-call
parsers arrive in M0.5."""
from __future__ import annotations

import json
import pathlib
from typing import Any

from transformers import AutoTokenizer


class Tokenizer:
    def __init__(self, checkpoint: str) -> None:
        self._tok = AutoTokenizer.from_pretrained(checkpoint)
        gen = pathlib.Path(checkpoint) / "generation_config.json"
        ids = json.loads(gen.read_text()).get("eos_token_id", []) if gen.is_file() else []
        ids = ids if isinstance(ids, list) else [ids]
        if self._tok.eos_token_id is not None:
            ids = [*ids, self._tok.eos_token_id]
        self.eos_token_ids = frozenset(ids)

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=True)


class Bundle:
    def __init__(self, checkpoint: str) -> None:
        self.model_name = _model_name(pathlib.Path(checkpoint))
        self.tokenizer = Tokenizer(checkpoint)

    def render_chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
                    add_generation_prompt: bool = True, **template_kwargs: Any) -> list[int]:
        text = self.tokenizer._tok.apply_chat_template(messages, tools=tools, tokenize=False,
                                                       add_generation_prompt=add_generation_prompt, **template_kwargs)
        return self.tokenizer.encode(text)

    def reasoning_parser(self) -> None:
        return None

    def tool_parser(self) -> None:
        return None


def _model_name(path: pathlib.Path) -> str:
    """`Qwen/Qwen3-8B` for a Hugging Face cache snapshot (models--Qwen--Qwen3-8B/snapshots/<hash>), else the dir."""
    for part in reversed(path.resolve().parts):
        if part.startswith("models--"):
            return part.removeprefix("models--").replace("--", "/")
    return path.name
