"""Invariant 1: the core never names a model, layer type, quant format, feature or model hub."""
from __future__ import annotations

import re

from _common import CORE, fail, py_files

DENY = re.compile(
    r"\b\w*(mamba|gdn|deltanet|mla|dsa|kda|qsa|ngram|indexer|nvfp4|mxfp4|gguf|ggml|gptq|awq|marlin|eagle|mtp|"
    r"qwen|deepseek|llama|mistral|gemma|lora|moe|expert|grammar|xgrammar|tool_call|structural|huggingface)\w*\b",
    re.IGNORECASE,
)

msgs = []
for p in py_files(CORE):
    for i, line in enumerate(p.read_text().splitlines(), 1):
        m = DENY.search(line)
        if m:
            msgs.append(f"{p.relative_to(CORE.parent.parent)}:{i}: '{m.group(0)}'")
fail(msgs, "core denylist (no model/layer/format/feature names in holon.*)")
