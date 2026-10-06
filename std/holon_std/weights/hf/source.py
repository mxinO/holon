"""Hugging Face checkpoint directory: config.json + one or more .safetensors files."""
from __future__ import annotations

import json
import pathlib
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
