"""Per-architecture hooks of the golden recorder (`golden`): what a model needs beyond transformers' own forward,
kept out of the generic driver. A model type without an entry needs nothing."""
from __future__ import annotations

import importlib
import pathlib
from types import ModuleType

import torch

MODULES = {"qwen4_exp_text": ".qwen4exp"}       # HF text model type -> its module (`prepare`, `final_norm`)


def _module(model: torch.nn.Module) -> ModuleType | None:
    name = MODULES.get(model.config.model_type)
    return importlib.import_module(name, __package__) if name else None


def prepare(model: torch.nn.Module, path: pathlib.Path | None, device: str, dtype: torch.dtype) -> dict:
    """Ready `model` (a skeleton or a loaded HF model; `path`: its GGUF, if any) for recording; returns the metadata
    its changes add (e.g. a documented deviation from HF)."""
    m = _module(model)
    return m.prepare(model, path, device, dtype) if m else {}


def final_norm(model: torch.nn.Module) -> torch.nn.Module:
    """The module between the last decoder layer and the head."""
    m = _module(model)
    return m.final_norm(model) if m else model.model.norm
