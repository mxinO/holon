"""`holon_tools profile`: a routing profile for the offload store's warm start (docs/v1-offload.md F2, §14) — the
routes of a prompt set's decode rows per (layer, expert), counted through the offload store's route taps (its
`after_commit` counts), as `{"layers": {"<layer>": [count per expert]}}`. It lies beside the checkpoint, under a name
relative to its directory: the `moe.placement.profile` option's value."""
from __future__ import annotations

import dataclasses
import json
import pathlib

import numpy as np
from holon.api import SamplingParams
from holon.core.engine import Engine, EngineConfig

KEY = "std.moe.offload"           # the offload store's shared-object key


def profile(config: EngineConfig, prompts: list[list[int]], max_tokens: int) -> dict:
    """Greedy generations of `max_tokens` from `prompts` under the offload store; their decode rows' routes."""
    engine = Engine(dataclasses.replace(config, plugin_options={**config.plugin_options,
                                                                "moe.store": "std.moe.offload@1"}))
    try:
        store = engine.ctx.shared_objects[KEY]
        total = np.zeros(store.residency.host.shape[:2], dtype=np.int64)
        store.taps.append(lambda step, telemetry: np.add(total, telemetry["counts"].numpy(), out=total))
        engine.generate(prompts, SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True))
        return {"layers": {str(layer): total[g].tolist() for g, layer in enumerate(sorted(store._order))}}
    finally:
        engine.close()


def write(checkpoint: pathlib.Path, name: str, counts: dict) -> pathlib.Path:
    """`counts` as the profile `name` beside the checkpoint (a file's directory, or a checkpoint directory)."""
    path = pathlib.PurePath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"profile {name!r}: a name relative to the checkpoint's directory")
    out = (checkpoint if checkpoint.is_dir() else checkpoint.parent) / path
    out.write_text(json.dumps(counts) + "\n")
    return out
