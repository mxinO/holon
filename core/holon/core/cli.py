"""`python -m holon serve`: the engine in its own process, kept serving by the supervisor, behind a frontend plugin
(design §7.7, §7.8)."""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import typing

import torch

from .assembly import resolve_model
from .engine import EngineConfig
from .ipc import engine_process
from .plugins import PluginError, PluginSet, discover
from .supervisor import supervised


def engine_flags(parser: argparse.ArgumentParser) -> None:
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


def engine_config(model: str, args: argparse.Namespace) -> EngineConfig:
    """The EngineConfig of `model` from the flags `engine_flags` added."""
    fields = {f.name for f in dataclasses.fields(EngineConfig)} - {"model"}
    return EngineConfig(model=model, **{k: v for k, v in vars(args).items() if k in fields})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="holon")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="serve a model over a frontend plugin")
    serve.add_argument("model", help="checkpoint directory")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--frontend", default=None, help="frontend plugin name (default: the one installed)")
    serve.add_argument("--supervised", action=argparse.BooleanOptionalAction, default=True,
                       help="restart a failed engine and resubmit its requests (default: on)")
    engine_flags(serve)
    args = parser.parse_args(argv)
    config = engine_config(args.model, args)

    plugins = PluginSet(discover())
    _, model_plugin = resolve_model(plugins, config)
    bundle = model_plugin.bundle(config.model)
    names = [m.name for m in plugins.by_slot("frontend")]
    name = args.frontend or (names[0] if len(names) == 1 else None)
    if name is None:
        raise PluginError(f"choose a frontend with --frontend among {names}")
    frontend = plugins.load(name)

    async def run() -> None:
        if not args.supervised:
            async with engine_process(config) as client:
                await frontend.serve(client, bundle, args.host, args.port)
            return
        async with supervised([config]) as supervisor:      # the lock history arrives with the registry (§4.4)
            serving = asyncio.ensure_future(frontend.serve(supervisor, bundle, args.host, args.port))
            gave_up = asyncio.ensure_future(supervisor.failed())
            done, _ = await asyncio.wait({serving, gave_up}, return_when=asyncio.FIRST_COMPLETED)
            serving.cancel()
            gave_up.cancel()
            await asyncio.gather(serving, gave_up, return_exceptions=True)   # the frontend stops before the engine
            if gave_up in done:
                raise SystemExit(f"holon: {gave_up.result()}")
            serving.result()                                  # re-raises whatever stopped the frontend

    asyncio.run(run())
