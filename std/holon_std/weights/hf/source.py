"""Hugging Face checkpoint directory: config.json + one or more .safetensors files."""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
from collections.abc import Iterator, Mapping
from typing import Any

import torch
from safetensors import safe_open


class HFSafetensors:
    kind = "hf_safetensors"

    def __init__(self, path: str | pathlib.Path, device: str = "cpu") -> None:
        self.path, self.device = pathlib.Path(path), device
        if not (self.path / "config.json").is_file():
            raise FileNotFoundError(f"{self.path}: no config.json")
        self._config = json.loads((self.path / "config.json").read_text())
        self.arch = self._config["architectures"][0]
        # generation_config.json may add end-of-turn tokens (e.g. Qwen3 instruct: <|im_end|> and <|endoftext|>).
        gen = self.path / "generation_config.json"
        ids = [self._config.get("eos_token_id"), json.loads(gen.read_text()).get("eos_token_id") if gen.is_file()
               else None]
        self.eos_token_ids = frozenset(i for v in ids if v is not None for i in (v if isinstance(v, list) else [v]))

    def config(self) -> Mapping[str, Any]:
        return self._config

    def tensors(self) -> Iterator[tuple[str, torch.Tensor]]:
        for file in sorted(self.path.glob("*.safetensors")):
            with safe_open(file, "pt", device=str(self.device)) as f:
                for name in f.keys():
                    yield name, f.get_tensor(name)

    def identity(self) -> str:
        """`<name>@<build>`: the hub id for a hub-cache snapshot, else the directory name, and a hash of config.json
        and every weight file — content, not the hub revision, which also moves with README edits."""
        name = self.path.name
        if self.path.parent.name == "snapshots" and self.path.parent.parent.name.startswith("models--"):
            name = self.path.parent.parent.name.removeprefix("models--").replace("--", "/")
        files = {f.name: _sha256(f) for f in [self.path / "config.json", *sorted(self.path.glob("*.safetensors"))]}
        return f"{name}@{hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()[:16]}"


def _sha256(f: pathlib.Path) -> str:
    """A file's sha256. Free for large hub-cache files: their symlink target is the blob named by it (small files
    are named by another hash and are read)."""
    if f.is_symlink() and re.fullmatch(r"[0-9a-f]{64}", blob := f.readlink().name):
        return blob
    with open(f, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()
