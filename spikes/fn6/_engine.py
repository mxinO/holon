"""Shared by the step-6 spikes: an engine from `holon serve`-style flags (spikes/fn7/tasks/run.py's parser). Run from
the repository root."""
from __future__ import annotations

import sys

sys.path.insert(0, "spikes/fn7/tasks")

import run  # noqa: E402
from holon.core.cli import layered  # noqa: E402
from holon.core.engine import Engine, EngineConfig  # noqa: E402

STORE = "std.moe.offload"


def engine(model: str, flags: str, **base) -> Engine:
    """`base` settings under the flags (theirs win), then `plugin_options` merged on top if given there."""
    extra = base.pop("plugin_options", None)
    cfg = layered(base, run.flags(flags))
    if extra:
        cfg = layered(cfg, {"plugin_options": extra})
    return Engine(EngineConfig(model, **cfg))
