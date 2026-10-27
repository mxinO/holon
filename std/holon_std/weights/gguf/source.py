"""A llama.cpp GGUF file, self-contained as in llama.cpp: tensors, configuration, tokenizer, chat template and stop
set all come from the file (M0.1 D3). Quantized tensors stay in their stored blocks (`gguf.native`) and are read
lazily from a copy-on-write memory map; each says where its blocks lie in the file (`QTensor.extents`).

A split file (`<name>-00001-of-0000N.gguf` … beside it, llama.cpp's `gguf-split`) is one checkpoint (FN D14), opened by
its first file: the metadata is the first's, the tensors every shard's in order, each extent in its own shard. A file
without a tokenizer (e.g. a draft head's weights) has an empty vocabulary."""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import tempfile
from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np
import torch
from gguf import GGMLQuantizationType
from holon.api import Extent, Layout, QTensor, Scheme

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
            raise ValueError(f"{self.path}: a GGUF weight source reads a .gguf file")
        self._fields, first = reader.read(self.path)
        self.files = shards(self.path, self._fields.get("split.count", 1))
        if len(self.files) > 1 and self._fields.get("split.no") != 0:
            raise ValueError(f"{self.path}: split.no {self._fields.get('split.no')}, not the first shard its name says")
        self._shards = [(self.path, first)] + [(p, _shard(p, i, len(self.files))) for i, p in enumerate(self.files)
                                                if i]
        self._tensors = [t for _, infos in self._shards for t in infos]
        if len(self.files) > 1 and len(self._tensors) != self._fields.get("split.tensors.count"):
            raise ValueError(f"{self.path}: its {len(self.files)} files hold {len(self._tensors)} tensors, not the "
                             f"{self._fields.get('split.tensors.count')} the first declares")
        if nvfp4 := [t.name for t in self._tensors if t.type == GGMLQuantizationType.NVFP4]:
            raise ValueError(f"{self.path}: NVFP4 tensors ({nvfp4[0]}, …) are not served from GGUF: llama.cpp keeps "
                             "their per-tensor scales in separate `.scale` tensors no recipe maps; serve the HF "
                             "checkpoint instead (docs/dev/m06-lowp.md D1)")
        hub = [p for p in self.path.parts if p.startswith("models--")]
        # The display name: `<hub repo>/<file stem>` for a hub-cache file, else the stem.
        stem = SPLIT.sub("", self.path.stem)
        self.name = f"{hub[-1].removeprefix('models--').replace('--', '/')}/{stem}" if hub else stem
        self._arch = self._value("general.architecture")
        self.arch = f"gguf.{self._arch}"
        tokens, types = self._value("tokenizer.ggml.tokens") or [], self._value("tokenizer.ggml.token_type") or []
        ends = {i for i, (t, k) in enumerate(zip(tokens, types, strict=True)) if k == 3 and t in _END_OF_TURN}
        named = (self._value(f"tokenizer.ggml.{k}_token_id") for k in ("eos", "eot"))
        self.eos_token_ids = frozenset(ends | {i for i in named if i is not None})

    def _value(self, key: str) -> Any:
        return self._fields.get(key)

    def config(self) -> Mapping[str, Any]:
        raw = {k.removeprefix(f"{self._arch}."): v for k, v in self._fields.items() if k.startswith(f"{self._arch}.")}
        cfg: dict[str, Any] = {hf: _shortest(raw[k]) for k, hf in _KEYS.items() if k in raw}
        names = {t.name for t in self._tensors}
        cfg |= {"vocab_size": len(self._value("tokenizer.ggml.tokens") or []),
                "tie_word_embeddings": "output.weight" not in names, "gguf": raw}
        if (mask := self._value("tokenizer.ggml.mask_token_id")) is not None:      # e.g. a block drafter's mask row
            cfg["mask_token_id"] = mask
        return cfg

    def tensors(self) -> Iterator[tuple[str, torch.Tensor | QTensor]]:
        for path, infos in self._shards:
            for t, data in reader.views(path, infos):
                if t.type in _DENSE:
                    yield t.name, torch.from_numpy(data).view(_DENSE[t.type]).reshape(t.shape)
                    continue
                if t.type == GGMLQuantizationType.MXFP4 and (data.reshape(t.rows, -1, 17)[..., 0] == 0xFF).any():
                    raise ValueError(f"{t.name}: an MXFP4 scale is NaN (e8m0 0xFF)")     # as std.weights.hf's
                layout = Layout(Scheme(f"ggml.{t.type.name.lower()}"), "gguf.native")
                yield t.name, QTensor(layout, t.shape, {"blocks": torch.from_numpy(data)},
                                      {"blocks": Extent(str(path.resolve()), t.offset, data.nbytes)})

    def tokenizer(self) -> Any:
        from .tokenizer import build

        tokens, eos = self._value("tokenizer.ggml.tokens"), self._value("tokenizer.ggml.eos_token_id")
        if not tokens:
            raise ValueError(f"{self.path}: no tokenizer in this file (e.g. a draft head's weights): it is not a model "
                             "to serve")
        return build(tokens, self._value("tokenizer.ggml.merges"), self._value("tokenizer.ggml.token_type"),
                     self._value("tokenizer.ggml.pre"), self._value("tokenizer.chat_template"),
                     None if eos is None else tokens[eos])

    def identity(self) -> str:
        """`<hub repo>/<file stem>@<16 hex of the file's sha256>` for a hub-cache file (the blob is named by its
        sha256), else `<file stem>@…`; a split file's: the sha256 of its shards' digests in order. A re-upload is a
        new model id."""
        digests = [_digest(p) for p in self.files]
        digest = digests[0] if len(digests) == 1 else hashlib.sha256("".join(digests).encode()).hexdigest()
        return f"{self.name}@{digest[:16]}"


SPLIT = re.compile(r"-\d{5}-of-\d{5}$")          # a split file's suffix: `<stem>-00001-of-00003`


def shards(path: pathlib.Path, count: int) -> list[pathlib.Path]:
    """A split GGUF's files in order, given its first (whose `split.count` is `count`); a whole file alone."""
    if count == 1:
        return [path]
    m = re.fullmatch(rf"(.*)-00001-of-{count:05d}\.gguf", path.name)
    if m is None:
        raise ValueError(f"{path}: open a file split in {count} by its first, <name>-00001-of-{count:05d}.gguf")
    files = [path.with_name(f"{m[1]}-{i:05d}-of-{count:05d}.gguf") for i in range(1, count + 1)]
    if missing := [f.name for f in files if not f.is_file()]:
        raise ValueError(f"{path}: its split files {missing} are missing")
    return files


def _shard(path: pathlib.Path, index: int, count: int) -> list[reader.TensorInfo]:
    """Shard `index`'s tensors, its own `split.no` and `split.count` checked against its place in the set."""
    fields, infos = reader.read(path)
    if (fields.get("split.no"), fields.get("split.count")) != (index, count):
        raise ValueError(f"{path}: split.no {fields.get('split.no')} of {fields.get('split.count')}, not shard "
                         f"{index} of {count} as its name says")
    return infos


def _digest(path: pathlib.Path) -> str:
    hub_blob = path.is_symlink() and any(p.startswith("models--") for p in path.parts)
    return os.path.basename(os.readlink(path)) if hub_blob else _file_digest(path)


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
