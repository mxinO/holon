"""`holon_tools profile`: a routing profile for the offload store's warm start (docs/v1-offload.md F2, §14) — the
routes of a prompt set's decode rows per (layer, expert), counted through the offload store's route taps (its
`after_commit` counts), as `{"checkpoint": [[file, bytes], …], "layers": {"<layer>": [count per expert]}}` (the store
refuses it on other files). It lies beside the checkpoint, under a name relative to its directory (default: the
checkpoint's stem): the `moe.placement.profile` option's value."""
from __future__ import annotations

import dataclasses
import json
import pathlib

import numpy as np
from holon.api import SamplingParams
from holon.core.engine import Engine, EngineConfig
from holon_std.moe.offload import KEY


def profile(config: EngineConfig, prompts: list[list[int]], max_tokens: int) -> tuple[dict, pathlib.Path]:
    """Greedy generations of `max_tokens` from `prompts` under the offload store; their decode rows' routes, and the
    directory the store reads profiles from (beside the checkpoint's file)."""
    engine = Engine(dataclasses.replace(config, plugin_options={**config.plugin_options,
                                                                "moe.store": "std.moe.offload@1"}))
    try:
        store = engine.ctx.shared_objects[KEY]
        total = np.zeros(store.residency.host.shape[:2], dtype=np.int64)
        store.taps.append(lambda step, record: np.add(total, record["counts"], out=total))
        engine.generate(prompts, SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True))
        return {"checkpoint": store.checkpoint,
                "layers": {str(layer): total[g].tolist() for g, layer in enumerate(store.layers)}}, store.profile_dir
    finally:
        engine.close()


def default_name(checkpoint: pathlib.Path) -> str:
    """`<stem>.profile.json`: two files' profiles in one directory (a hub cache's blobs) do not share a name."""
    return f"{checkpoint.stem}.profile.json"


def write(directory: pathlib.Path, name: str, counts: dict) -> pathlib.Path:
    """`counts` as the profile `name` in `directory` (`profile`'s: where the store reads it)."""
    path = pathlib.PurePath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"profile {name!r}: a name relative to the checkpoint's directory")
    out = directory / path
    out.write_text(json.dumps(counts) + "\n")
    return out
