"""`python -m holon serve`: the engine in its own process behind a frontend plugin (design §7.8)."""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import typing

import torch

from .engine import EngineConfig, resolve_model
from .ipc import engine_process
from .plugins import PluginError, PluginSet, discover


def _engine_flags(parser: argparse.ArgumentParser) -> None:
    """One flag per EngineConfig field (bool fields as --x / --no-x; dtype by name)."""
    hints = typing.get_type_hints(EngineConfig)
    for f in dataclasses.fields(EngineConfig):
        if f.name == "model":
            continue
        flag, kind = "--" + f.name.replace("_", "-"), hints[f.name]
        if kind is bool:
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=f.default)
        elif kind is torch.dtype:
            parser.add_argument(flag, type=lambda s: getattr(torch, s), default=f.default)
        elif kind in (int, float, str) or kind == int | None:
            parser.add_argument(flag, type=int if kind == int | None else kind, default=f.default)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="holon")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="serve a model over a frontend plugin")
    serve.add_argument("model", help="checkpoint directory")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--frontend", default=None, help="frontend plugin name (default: the one installed)")
    _engine_flags(serve)
    args = parser.parse_args(argv)
    fields = {f.name for f in dataclasses.fields(EngineConfig)}
    config = EngineConfig(model=args.model, **{k: v for k, v in vars(args).items() if k in fields and k != "model"})

    plugins = PluginSet(discover())
    _, model_plugin = resolve_model(plugins, config)
    bundle = model_plugin.bundle(config.model)
    names = [m.name for m in plugins.by_slot("frontend")]
    name = args.frontend or (names[0] if len(names) == 1 else None)
    if name is None:
        raise PluginError(f"choose a frontend with --frontend among {names}")
    frontend = plugins.load(name)

    async def run() -> None:
        async with engine_process(config) as client:
            await frontend.serve(client, bundle, args.host, args.port)

    asyncio.run(run())
