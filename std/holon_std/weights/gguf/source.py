"""A llama.cpp GGUF file, self-contained as in llama.cpp: tensors, configuration, tokenizer, chat template and stop
set all come from the file (M0.1 D3). Quantized tensors stay in their stored blocks (`gguf.native`) and are read
lazily from a copy-on-write memory map."""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import tempfile
from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np
import torch
from gguf import GGMLQuantizationType
from holon.api import Layout, QTensor, Scheme

from . import reader

_CACHE = pathlib.Path.home() / ".cache" / "holon" / "weights"     # file digests remembered across restarts
_DENSE = {GGMLQuantizationType.F32: torch.float32, GGMLQuantizationType.F16: torch.float16,
          GGMLQuantizationType.BF16: torch.bfloat16}
# llama.cpp keys every text architecture shares ("<arch>.<key>") -> the HF configuration key recipes read.
_KEYS = {"block_count": "num_hidden_layers", "embedding_length": "hidden_size",
         "feed_forward_length": "intermediate_size", "attention.head_count": "num_attention_heads",
         "attention.head_count_kv": "num_key_value_heads", "attention.key_length": "head_dim",
         "rope.freq_base": "rope_theta", "attention.layer_norm_rms_epsilon": "rms_norm_eps",
         "context_length": "max_position_embeddings"}
# Control tokens llama.cpp treats as end of generation by their text (llama-vocab.cpp; its FIM additions excluded).
_END_OF_TURN = {"<|eot_id|>", "<|im_end|>", "<|end|>", "<end_of_turn>", "<|endoftext|>", "<|eom_id|>", "<EOT>",
                "<|end_of_text|>"}


class GGUFSource:
    kind = "gguf"

    def __init__(self, path: str | pathlib.Path, device: str = "cpu") -> None:
        self.path = pathlib.Path(path)
        if not self.path.is_file():
            raise ValueError(f"{self.path}: a GGUF weight source reads one .gguf file")
        self._fields, self._tensors = reader.read(self.path)
        hub = [p for p in self.path.parts if p.startswith("models--")]
        # The display name: `<hub repo>/<file stem>` for a hub-cache file, else the stem.
        self.name = (f"{hub[-1].removeprefix('models--').replace('--', '/')}/{self.path.stem}" if hub
                     else self.path.stem)
        self._arch = self._value("general.architecture")
        self.arch = f"gguf.{self._arch}"
        tokens, types = self._value("tokenizer.ggml.tokens"), self._value("tokenizer.ggml.token_type")
        ends = {i for i, (t, k) in enumerate(zip(tokens, types, strict=True)) if k == 3 and t in _END_OF_TURN}
        named = (self._value(f"tokenizer.ggml.{k}_token_id") for k in ("eos", "eot"))
        self.eos_token_ids = frozenset(ends | {i for i in named if i is not None})

    def _value(self, key: str) -> Any:
        return self._fields.get(key)

    def config(self) -> Mapping[str, Any]:
        raw = {k.removeprefix(f"{self._arch}."): v for k, v in self._fields.items() if k.startswith(f"{self._arch}.")}
        cfg: dict[str, Any] = {hf: _shortest(raw[k]) for k, hf in _KEYS.items() if k in raw}
        names = {t.name for t in self._tensors}
        cfg |= {"vocab_size": len(self._value("tokenizer.ggml.tokens")),
                "tie_word_embeddings": "output.weight" not in names, "gguf": raw}
        return cfg

    def tensors(self) -> Iterator[tuple[str, torch.Tensor | QTensor]]:
        for t, data in reader.views(self.path, self._tensors):
            if t.type in _DENSE:
                yield t.name, torch.from_numpy(data).view(_DENSE[t.type]).reshape(t.shape)
            else:
                layout = Layout(Scheme(f"ggml.{t.type.name.lower()}"), "gguf.native")
                yield t.name, QTensor(layout, t.shape, {"blocks": torch.from_numpy(data)})

    def tokenizer(self) -> Any:
        from .tokenizer import build

        tokens, eos = self._value("tokenizer.ggml.tokens"), self._value("tokenizer.ggml.eos_token_id")
        return build(tokens, self._value("tokenizer.ggml.merges"), self._value("tokenizer.ggml.token_type"),
                     self._value("tokenizer.ggml.pre"), self._value("tokenizer.chat_template"),
                     None if eos is None else tokens[eos])

    def identity(self) -> str:
        """`<hub repo>/<file stem>@<16 hex of the file's sha256>` for a hub-cache file (the blob is named by its
        sha256), else `<file stem>@…`; a re-upload is a new model id."""
        hub_blob = self.path.is_symlink() and any(p.startswith("models--") for p in self.path.parts)
        digest = os.path.basename(os.readlink(self.path)) if hub_blob else _file_digest(self.path)
        return f"{self.name}@{digest[:16]}"


def _shortest(v: Any) -> Any:
    """GGUF floats are float32: take their shortest decimal form (1e-06, not 9.999999974752427e-07)."""
    return float(str(np.float32(v))) if isinstance(v, float) else v


def _file_digest(path: pathlib.Path) -> str:
    """sha256 of a file, remembered in the user's cache by the file's identity and change times."""
    st = path.stat()
    key = [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]
    cache = _CACHE / f"{hashlib.sha256(str(path.resolve()).encode()).hexdigest()}.json"
    try:
        known = json.loads(cache.read_text())
        if known[:-1] == key:
            return known[-1]
    except (OSError, ValueError):
        pass
    with open(path, "rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    try:                                   # atomically: engines may start side by side
        cache.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=cache.parent, delete=False) as tmp:
            tmp.write(json.dumps([*key, digest]))
        os.replace(tmp.name, cache)
    except OSError:
        pass
    return digest
