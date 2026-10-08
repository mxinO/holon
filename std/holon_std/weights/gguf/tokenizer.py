"""The tokenizer of a GGUF file, built from the file alone as llama.cpp does — not with transformers' GGUF converter,
which drops the USER_DEFINED tokens and the eos.

Byte-level BPE from `tokenizer.ggml.tokens` + `merges`; CONTROL tokens are special added tokens, USER_DEFINED ones
atomic added tokens, UNUSED ones (padding entries) unreachable. The pre-tokenizer comes from `tokenizer.ggml.pre`:
llama.cpp keeps one table of them, and so do we. Each entry is the original tokenizer's normalizer and split regex —
including NFC for `qwen2`, which llama.cpp skips, so decomposed input encodes as the model was trained.
"""
from __future__ import annotations

from typing import Any

from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers

CONTROL, USER_DEFINED = 3, 4                     # llama.cpp token types

_PRE = {   # tokenizer.ggml.pre -> (normalizer, split regex), as in the original tokenizer.json
    "qwen2": (normalizers.NFC(), r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}"
                                 r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"),
}


def build(tokens: list[str], merges: list[str], types: list[int], pre: str, chat_template: str | None,
          eos: str | None) -> Any:
    """A transformers fast tokenizer (so frontends keep `apply_chat_template`) over the GGUF table."""
    from transformers import PreTrainedTokenizerFast  # only frontends and tests build tokenizers

    if pre not in _PRE:
        raise ValueError(f"GGUF pre-tokenizer {pre!r} is not supported (known: {sorted(_PRE)})")
    normalizer, regex = _PRE[pre]
    tok = Tokenizer(models.BPE({t: i for i, t in enumerate(tokens)}, [tuple(m.split(" ", 1)) for m in merges]))
    tok.normalizer = normalizer
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(regex), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens([AddedToken(t, special=True, normalized=False)
                            for t, k in zip(tokens, types, strict=True) if k == CONTROL])
    tok.add_tokens([AddedToken(t, special=False, normalized=False)
                    for t, k in zip(tokens, types, strict=True) if k == USER_DEFINED])
    return PreTrainedTokenizerFast(tokenizer_object=tok, chat_template=chat_template, eos_token=eos)
