"""Hugging Face checkpoint directory: config.json + one or more .safetensors files, dense or quantized (a dialect
of `quant/` reads its `quantization_config`, docs/m06-lowp.md D2)."""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import pathlib
import re
import tempfile
from collections.abc import Iterator, Mapping
from typing import Any

import torch
from holon.api import QTensor
from safetensors import safe_open

from . import quant

_CACHE = pathlib.Path.home() / ".cache" / "holon" / "weights"     # weight-file digests remembered across restarts


class HFSafetensors:
    kind = "hf_safetensors"

    def __init__(self, path: str | pathlib.Path, device: str = "cpu") -> None:
        self.path, self.device = pathlib.Path(path), device
        if not (self.path / "config.json").is_file():
            raise FileNotFoundError(f"{self.path}: no config.json")
        self._config = json.loads((self.path / "config.json").read_text())
        self.arch = self._config["architectures"][0]
        self._hub = self.path.parent.name == "snapshots" and self.path.parent.parent.name.startswith("models--")
        # The display name: the hub id for a hub-cache snapshot, else the directory name.
        self.name = self.path.parent.parent.name.removeprefix("models--").replace("--", "/") if self._hub \
            else self.path.name
        # generation_config.json may add end-of-turn tokens (e.g. Qwen3 instruct: <|im_end|> and <|endoftext|>).
        gen = self.path / "generation_config.json"
        ids = [self._config.get("eos_token_id"), json.loads(gen.read_text()).get("eos_token_id") if gen.is_file()
               else None]
        self.eos_token_ids = frozenset(i for v in ids if v is not None for i in (v if isinstance(v, list) else [v]))
        # The quantization config: config.json's, else ModelOpt's hf_quant_config.json, then hashed into the identity.
        qc, hq = self._config.get("quantization_config"), self.path / "hf_quant_config.json"
        self._hashed = ["config.json"]
        if qc is None and hq.is_file():
            raw = json.loads(hq.read_text())
            qc = {"quant_method": raw.get("producer", {}).get("name"), **raw["quantization"]}
            self._hashed.append(hq.name)
        self._dialect = quant.dialect(qc) if qc else None

    def config(self) -> Mapping[str, Any]:
        return self._config

    def tokenizer(self) -> Any:
        from transformers import AutoTokenizer  # only frontends and tests need it

        return AutoTokenizer.from_pretrained(self.path)

    def tensors(self) -> Iterator[tuple[str, torch.Tensor | QTensor]]:
        """Every tensor in shard order. Quantized: every shard's header read first, each module's parts grouped by
        the dialect into one `QTensor` under `<module>.weight` at its primary part's place, each part read once."""
        if self._dialect is None:
            for file in sorted(self.path.glob("*.safetensors")):
                with safe_open(file, "pt", device=str(self.device)) as f:
                    for name in f.keys():
                        yield name, f.get_tensor(name)
            return
        with contextlib.ExitStack() as stack:
            where = {}
            for file in sorted(self.path.glob("*.safetensors")):
                f = stack.enter_context(safe_open(file, "pt", device=str(self.device)))
                if twice := set(where) & set(f.keys()):
                    raise ValueError(f"{file}: tensors also in an earlier shard: {sorted(twice)[:4]}")
                where |= dict.fromkeys(f.keys(), f)
            get = lambda name: where[name].get_tensor(name)                   # noqa: E731
            plan, checks = quant.group(self._dialect, [(n, f.get_slice(n).get_dtype()) for n, f in where.items()])
            for name, want in checks.items():
                if not bool((value := get(name)).eq(want).all()):
                    raise ValueError(f"{name} = {value.tolist()}: must be {want} (the engine does not apply it)")
            for item in plan:
                if isinstance(item, str):
                    yield item, get(item)
                else:
                    yield f"{item.name}.weight", quant.load(self._dialect, item, get)

    def identity(self) -> str:
        """`<name>@<build>`: the hub id for a hub-cache snapshot, else the directory name, and a hash of config.json
        and every weight file — content, not the hub revision, which also moves with README edits."""
        files = self._digests([*(self.path / m for m in self._hashed), *sorted(self.path.glob("*.safetensors"))],
                              remember=not self._hub)
        return f"{self.name}@{hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()[:16]}"

    def _digests(self, files: list[pathlib.Path], remember: bool) -> dict[str, str]:
        """sha256 per file. Large hub-cache files are free: their symlink target is the blob named by it. Others are
        read; with `remember` (not in the shared hub cache, whose weights are free anyway) their digests are kept in
        the user's cache, never in the checkpoint, keyed by the file's identity and change times (ctime moves on any
        replacement and cannot be set back), so restarts do not rehash a checkpoint."""
        cache = _CACHE / f"{hashlib.sha256(str(self.path.resolve()).encode()).hexdigest()}.json"
        try:
            known = json.loads(cache.read_text()) if remember else {}
        except (OSError, ValueError):
            known = {}
        out, fresh = {}, False
        for f in files:
            if f.is_symlink() and re.fullmatch(r"[0-9a-f]{64}", blob := f.readlink().name):
                out[f.name] = blob
                continue
            st = f.stat()
            key = [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]
            if known.get(f.name, [])[:-1] != key:
                with open(f, "rb") as fh:
                    known[f.name], fresh = [*key, hashlib.file_digest(fh, "sha256").hexdigest()], True
            out[f.name] = known[f.name][-1]
        if fresh and remember:
            try:                           # atomically: engines may start side by side
                cache.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile("w", dir=cache.parent, delete=False) as tmp:
                    tmp.write(json.dumps(known, sort_keys=True))
                os.replace(tmp.name, cache)
            except OSError:                # no writable cache: hash again next time
                pass
        return out

