"""Text-mode OpenAI completions, for servers that take no token-id prompts (FreeToken) and for driving every server
the same way against them: a trace's token-id prompt is decoded to text with the model's own tokenizer, so each
server tokenizes the same text; the streamed text chunks count one token each, and the final usage chunk (asked for)
carries the server's own count, which settles the total. Generated tokens come back as a fixed stand-in id (no
server returns ids here), so a multi-turn history holds the same text for every server."""
from __future__ import annotations

import pathlib

from tokenizers import Tokenizer


def headroom(max_model_len: int) -> int:
    """A context limit for text prompts: decoded and re-encoded, a trace's token ids can come back longer."""
    return -(-max_model_len * 3 // 2 // 256) * 256


class Prompts:
    def __init__(self, model: str) -> None:
        path = pathlib.Path(model)
        self.tok = Tokenizer.from_file(str((path if path.is_dir() else path.parent) / "tokenizer.json"))
        self.stand_in = self.tok.encode(" a", add_special_tokens=False).ids[-1]

    def body(self, prompt: list[int], max_tokens: int) -> dict:
        return {"model": "m", "prompt": self.tok.decode(prompt), "max_tokens": max_tokens, "temperature": 0.0,
                "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}

    def parse(self, event: dict, state: dict) -> tuple[list[int], bool]:
        choices, usage, n = event.get("choices") or [], event.get("usage"), 0
        if choices and choices[0].get("text"):
            n = 1
        if usage:                                   # the last chunk: the server's own count settles the total
            n = max(n, usage["completion_tokens"] - state.get("seen", 0))
        state["seen"] = state.get("seen", 0) + n
        return [self.stand_in] * n, bool(usage)
