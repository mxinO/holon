"""The `holon` command. `holon serve`: the engine in its own process, kept serving by the supervisor, behind a
frontend plugin (design §7.7, §7.8). `holon prebuild`: build the installed plugins' native code for the visible GPUs
ahead of time (release design D1).

`holon serve` starts from the checkpoint's lock (release design D2): the newest lock of its build for this GPU class
(or borrowed from the nearest, unverified), its settings under the flags given — flags win; dict flags merge per key
and `key=` deletes one of the lock's — and its kernel pins as preferences. `--no-lock` serves defaults and flags."""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import pathlib
import subprocess
import sys
import typing

import torch

from . import plans, platform
from .assembly import resolve_model
from .engine import EngineConfig
from .ipc import engine_process
from .plugins import PluginError, PluginSet, discover
from .supervisor import supervised

AUTO_CONTEXT = 32768      # `--max-model-len auto`: the model's trained context, at most this


def engine_flags(parser: argparse.ArgumentParser, given_only: bool = False) -> None:
    """One flag per EngineConfig field (bool fields as --x / --no-x; dtype by name; dict fields as key=value
    lists, e.g. `--plugin-options key=value other.key=value`). `given_only`: flags not given stay out of the parsed
    arguments (they fall to a lock's settings), and `--max-model-len` also takes `auto`."""
    hints = typing.get_type_hints(EngineConfig)
    for f in dataclasses.fields(EngineConfig):
        if f.name == "model":
            continue
        flag, kind = "--" + f.name.replace("_", "-"), hints[f.name]
        default = argparse.SUPPRESS if given_only else [] if typing.get_origin(kind) is dict else f.default
        if kind is bool:
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=default)
        elif kind is torch.dtype:
            parser.add_argument(flag, type=lambda s: getattr(torch, s), default=default)
        elif f.name == "max_model_len" and given_only:
            parser.add_argument(flag, type=lambda s: s if s == "auto" else int(s), default=default)
        elif kind in (int, float, str, int | None, str | None):
            parser.add_argument(flag, type={int | None: int, str | None: str}.get(kind, kind), default=default)
        elif typing.get_origin(kind) is dict:
            parser.add_argument(flag, nargs="*", default=default, metavar="KEY=VALUE")


def engine_settings(args: argparse.Namespace) -> dict[str, object]:
    """The EngineConfig fields among the parsed flags (dict fields' key=value lists as dicts)."""
    fields = {f.name for f in dataclasses.fields(EngineConfig)} - {"model"}
    values = {k: v for k, v in vars(args).items() if k in fields}
    for k, v in values.items():
        if isinstance(v, list):                       # a dict field's key=value list
            if any("=" not in item for item in v):
                raise SystemExit(f"--{k.replace('_', '-')}: expected KEY=VALUE items, got {v}")
            values[k] = dict(item.split("=", 1) for item in v)
    return values


def engine_config(model: str, args: argparse.Namespace) -> EngineConfig:
    """The EngineConfig of `model` from the flags `engine_flags` added."""
    return EngineConfig(model=model, **engine_settings(args))


def layered(base: dict[str, object], flags: dict[str, object]) -> dict[str, object]:
    """`flags` over `base` (a lock's settings): dict settings merge per key, and `key=` (empty) deletes a key."""
    out = dict(base)
    for k, v in flags.items():
        out[k] = {key: x for key, x in ((out.get(k) or {}) | v).items() if x != ""} if isinstance(v, dict) else v
    return out


def serve_history(model: str, flags: dict[str, object], found: tuple | None, auto: int
                  ) -> list[tuple[EngineConfig, str | None]]:
    """The supervisor's history, oldest first: defaults + flags, then the lock (`plans.find`'s result) + flags. The
    requested `max_model_len` is a flag's, else the lock's, else `auto`; every config clamps it to what fits."""
    def config(base: dict[str, object]) -> EngineConfig:
        settings = {"clamp_context": True} | layered(base, flags)
        if settings.get("max_model_len", "auto") == "auto":
            settings["max_model_len"] = auto
        return EngineConfig(model=model, **settings)

    history = [(config({}), None)]
    if found is not None:
        history.append((config(plans.lock_overrides(found[0])), str(found[2])))
    return history


def auto_context(model_plugin: object, source: object) -> int:
    """`--max-model-len auto`: the context the checkpoint was trained for (the model plugin's optional hook), at most
    AUTO_CONTEXT."""
    trained = getattr(model_plugin, "trained_context", None)
    return min((trained and trained(source.config(), source.kind)) or AUTO_CONTEXT, AUTO_CONTEXT)


def find_lock(directory: pathlib.Path, source: object, plugins: PluginSet, device: str) -> tuple | None:
    """The lock `holon serve` starts from (loudly: which, or why none)."""
    found = None
    if not directory.is_dir():
        _say(f"WARNING: no lock directory {directory}: serving defaults (name one with --plans)")
    elif (found := plans.find(directory, source.identity(), platform.fingerprint(torch.device(device)),
                              plans.current_srcs(plugins.installed))) is None:
        _say(f"no current lock for {source.identity()} in {directory}: serving defaults")
    else:
        _say(f"starting from {found[2]}" + (f" (unverified: {found[0].plan.fingerprint_class}'s)" if found[1] else ""))
    return found


def _say(text: str) -> None:
    print(f"holon: {text}", file=sys.stderr, flush=True)


def prebuild(sms: list[int]) -> None:
    """Run every installed plugin's build module (manifest `[build] module`) once per SM, each in its own process:
    `python -m <module> --sm N`. The modules choose their caches; the core resolves none."""
    sms = sms or platform.visible_sms()
    if not sms:
        raise SystemExit("holon prebuild: no visible GPU; name the targets with --sm")
    failed = []
    for name, inst in sorted(discover().items()):
        for sm in sms if inst.manifest.build else ():
            print(f"holon prebuild: {name} for sm {sm}", flush=True)
            module = f"{inst.package}.{inst.manifest.build}"
            if subprocess.run([sys.executable, "-m", module, "--sm", str(sm)]).returncode:
                failed.append(f"{name} (sm {sm})")
    if failed:
        raise SystemExit(f"holon prebuild: failed: {', '.join(failed)}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="holon")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("prebuild", help="build the plugins' native code for the visible GPUs ahead of time")
    build.add_argument("--sm", type=int, nargs="+", default=[], help="target SMs, e.g. 86 89 120 (default: visible)")
    serve = sub.add_parser("serve", help="serve a model over a frontend plugin")
    serve.add_argument("model", help="checkpoint directory or file")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--frontend", default=None, help="frontend plugin name (default: the one installed)")
    serve.add_argument("--supervised", action=argparse.BooleanOptionalAction, default=True,
                       help="restart a failed engine and resubmit its requests (default: on)")
    serve.add_argument("--plans", type=pathlib.Path, default=plans.TREE / "plans", help="the lock histories")
    serve.add_argument("--no-lock", action="store_true", help="serve defaults and flags, not the model's lock")
    engine_flags(serve, given_only=True)
    args = parser.parse_args(argv)
    if args.command == "prebuild":
        return prebuild(args.sm)
    flags = engine_settings(args)

    plugins = PluginSet(discover())
    source, model_plugin = resolve_model(plugins, EngineConfig(args.model, source=flags.get("source")))
    found = None if args.no_lock else find_lock(args.plans, source, plugins, flags.get("device", "cuda"))
    history = serve_history(args.model, flags, found, auto_context(model_plugin, source))
    config, lock = history[-1]
    bundle = model_plugin.bundle(source)
    names = [m.name for m in plugins.by_slot("frontend")]
    name = args.frontend or (names[0] if len(names) == 1 else None)
    if name is None:
        raise PluginError(f"choose a frontend with --frontend among {names}")
    frontend = plugins.load(name)

    async def run() -> None:
        if not args.supervised:
            async with engine_process(config, lock=lock) as client:
                await frontend.serve(client, bundle, args.host, args.port, config.plugin_options)
            return
        async with supervised(history) as supervisor:
            serving = asyncio.ensure_future(frontend.serve(supervisor, bundle, args.host, args.port,
                                                           config.plugin_options))
            gave_up = asyncio.ensure_future(supervisor.failed())
            done, _ = await asyncio.wait({serving, gave_up}, return_when=asyncio.FIRST_COMPLETED)
            serving.cancel()
            gave_up.cancel()
            await asyncio.gather(serving, gave_up, return_exceptions=True)   # the frontend stops before the engine
            if gave_up in done:
                raise SystemExit(f"holon: {gave_up.result()}")
            serving.result()                                  # re-raises whatever stopped the frontend

    asyncio.run(run())
