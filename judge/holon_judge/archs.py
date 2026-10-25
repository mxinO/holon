"""Per-architecture hooks of the golden recorder (`golden`): what a model needs beyond transformers' own forward,
kept out of the generic driver. A model type without an entry needs nothing."""
from __future__ import annotations

import importlib
import pathlib
from types import ModuleType

import torch

MODULES = {"qwen4_exp_text": ".qwen4exp", "glm5_next_text": ".glm5next"}   # HF text model type -> its module


def _by_type(model_type: str) -> ModuleType | None:
    name = MODULES.get(model_type)
    return importlib.import_module(name, __package__) if name else None


def _module(model: torch.nn.Module) -> ModuleType | None:
    return _by_type(model.config.model_type)


def register(model_type: str) -> None:
    """Before an HF model of `model_type` is built: its module's `register` (e.g. a causal LM transformers lacks)."""
    m = _by_type(model_type)
    if m is not None and hasattr(m, "register"):
        m.register()


def prepare(model: torch.nn.Module, path: pathlib.Path | None, device: str, dtype: torch.dtype) -> dict:
    """Ready `model` (a skeleton or a loaded HF model; `path`: its GGUF, if any) for recording; returns the metadata
    its changes add (e.g. a documented deviation from HF)."""
    m = _module(model)
    return m.prepare(model, path, device, dtype) if m else {}


def final_norm(model: torch.nn.Module) -> torch.nn.Module:
    """The module between the last decoder layer and the head."""
    m = _module(model)
    return m.final_norm(model) if m else model.model.norm


def stand_in(model: torch.nn.Module, hidden: torch.Tensor):
    """What a decoder layer returns for `hidden` (the streamed driver's stand-in): the hidden states, or the
    architecture's own tuple."""
    m = _module(model)
    return m.stand_in(hidden) if m and hasattr(m, "stand_in") else hidden


def layer_out(model: torch.nn.Module, out) -> torch.Tensor:
    """A decoder layer's output hidden states (its own tuple's first, where it returns one)."""
    m = _module(model)
    return m.streams(out) if m and hasattr(m, "streams") else out
