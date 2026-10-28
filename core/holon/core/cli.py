"""The `holon` command. `holon serve`: the engine in its own process, kept serving by the supervisor, behind a
frontend plugin (design §7.7, §7.8). `holon models`: the models locked for the visible GPU and how `holon serve` starts
each. `holon prebuild`: build the installed plugins' native code for the visible GPUs ahead of time (release design
D1).

`holon serve <model>` takes a local path or a reference a weight-source plugin resolves (e.g. a hub id; release design
§14.3): an existing path first; then a repository reference becomes the one model under it with a verified lock for
this GPU; then the weight sources' claims, the one claim logged, refused before fetching when larger than the GPU.

`holon serve` starts from the checkpoint's lock (release design D2): the newest lock of its build for this GPU class
(or borrowed from the nearest, unverified), its settings under the flags given — flags win; dict flags merge per key
and `key=` deletes one of the lock's — and its kernel pins as preferences. `--no-lock` serves defaults and flags."""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import fcntl
import functools
import os
import pathlib
import re
import subprocess
import sys
import textwrap
import time
import tomllib
import typing
from collections.abc import Iterator, Mapping

import torch

from holon.api import SLOTS, Cardinality, Fingerprint, Option

from . import plans, platform
from .assembly import claim_ref, resolve_model
from .engine import EngineConfig
from .ipc import StartFailed, engine_process
from .plugins import Installed, PluginError, PluginSet, discover
from .supervisor import Supervisor, _say

AUTO_CONTEXT = 32768      # `--max-model-len auto`: the model's trained context, at most this
SERVE_MAX_SEQS = 8        # `holon serve` without a lock (EngineConfig's 64 leaves a large hybrid no KV on 24 GB)
SERVE_DEFAULTS = {"max_seqs": f"`holon serve` without a lock: {SERVE_MAX_SEQS}",
                  "max_model_len": f"`holon serve`: auto = min(the model's trained context, {AUTO_CONTEXT})"}
SERVABLE = re.compile(r"[^/@\s]+/[^/@\s]+(/[^/@\s]+)?")    # a locked model's name `holon serve` accepts as it is
CACHE = pathlib.Path.home() / ".cache" / "holon"            # files derived on this machine (sources/: a lock's files)


def engine_flags(parser: argparse.ArgumentParser | argparse._ArgumentGroup, given_only: bool = False) -> None:
    """One flag per EngineConfig field, with the field's help line (bool fields as --x / --no-x; dtype by name; dict
    fields as key=value lists, e.g. `--plugin-options key=value other.key=value`). `given_only`: flags not given stay
    out of the parsed arguments (they fall to a lock's settings), and `--max-model-len` also takes `auto`."""
    hints = typing.get_type_hints(EngineConfig)
    for f in dataclasses.fields(EngineConfig):
        if f.name in ("model", "clamp_context") or (f.name == "kernel_preferences" and not given_only):
            continue                                  # `holon serve` clamps; preferences come from locks
        if not f.metadata.get("help"):
            raise TypeError(f"EngineConfig.{f.name} has no help line (`engine._setting`)")
        flag, kind = "--" + f.name.replace("_", "-"), hints[f.name]
        dict_kind = typing.get_origin(kind) is dict
        notes = [f"default: {str(f.default).removeprefix('torch.')}" if f.default is not None and not dict_kind else "",
                 "KEY= deletes the lock's" if given_only and dict_kind else "",
                 SERVE_DEFAULTS.get(f.name, "") if given_only else ""]
        notes = "; ".join(n for n in notes if n)
        add = functools.partial(parser.add_argument, flag, help=f.metadata["help"].replace("%", "%%")
                                + (f" ({notes})" if notes else ""),
                                default=argparse.SUPPRESS if given_only else [] if dict_kind else f.default)
        if kind is bool:
            add(action=argparse.BooleanOptionalAction)
        elif kind is torch.dtype:
            add(type=lambda s: getattr(torch, s))
        elif f.name == "max_model_len" and given_only:
            add(type=lambda s: s if s == "auto" else int(s))
        elif kind in (int, float, str, int | None, str | None):
            add(type={int | None: int, str | None: str}.get(kind, kind))
        elif dict_kind:
            add(nargs="*", metavar="KEY=VALUE")


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


def serve_defaults(directory: pathlib.Path) -> dict[str, object]:
    """The engine settings `holon serve` starts from without a current lock: `<plans>/defaults.toml` (written by hand,
    unlike the locks: the plugins every lock selects, so serving between propagations keeps them), else none."""
    path = directory / "defaults.toml"
    return tomllib.loads(path.read_text()) if path.is_file() else {}


def serve_history(model: str, flags: dict[str, object], found: tuple | None, auto: int,
                  defaults: Mapping[str, object] | None = None, *, lock_extra: Mapping[str, object] | None = None,
                  spill: Mapping[str, str] | None = None, keep_defaults: bool = True
                  ) -> list[tuple[EngineConfig, str | None]]:
    """The supervisor's history, oldest first: defaults (`serve_defaults`) + flags unless not `keep_defaults`, then
    the lock (`plans.find`'s result) + `lock_extra` + flags, preceded by the same with the `spill` options if any (the
    supervisor's memory-only entry). The requested `max_model_len` is a flag's, else the lock's, else `auto`; every
    config clamps it to what fits and derives its GPU memory's slack unless `--memory-fraction` is given. The defaults
    serve at most SERVE_MAX_SEQS sequences unless `--max-seqs` says otherwise."""
    def config(base: dict[str, object]) -> EngineConfig:
        settings = {"clamp_context": True, "memory_fraction": 0.0} | layered(base, flags)
        if settings.get("max_model_len", "auto") == "auto":
            settings["max_model_len"] = auto
        return EngineConfig(model=model, **settings)

    if "max_seqs" not in flags and (keep_defaults or found is None):
        _say(f"{'the fallback' if found else 'defaults'} without a lock: max_seqs {SERVE_MAX_SEQS} (--max-seqs)")
    history = [(config({"max_seqs": SERVE_MAX_SEQS} | dict(defaults or {})), None)] if keep_defaults else []
    if found is not None:
        lock = layered(plans.lock_overrides(found[0]), dict(lock_extra or {}))
        history += [(config(layered(lock, {"plugin_options": dict(spill)})), str(found[2]))] if spill else []
        history.append((config(lock), str(found[2])))
    return history


def host_budget(flags: Mapping[str, object]) -> int | None:
    """`holon serve`'s host memory (docs/dev/v1-auto-config.md D1): the RAM available less the headroom, read once
    and logged; None under `--host-memory`, warned about where it cuts into the headroom."""
    available, total, why = platform.host_ram()
    headroom, given = platform.host_headroom(total), flags.get("host_memory")
    budget = max(0, available - headroom)
    _say(f"host memory: {'--host-memory ' if given is not None else ''}budget {(given or budget) / 2**30:.1f} GiB; "
         f"{available / 2**30:.1f} GiB available ({why}), {headroom / 2**30:.1f} kept free (max 6 GiB, 5% of "
         f"{total / 2**30:.1f})" + (": pinning past that may swap" if (given or 0) > budget else ""))
    return None if given is not None else budget


@contextlib.contextmanager
def pin_lock(wait: int = 900) -> Iterator[None]:
    """One `holon serve` of this user at a time reads the RAM and pins host memory, so a second start's reading sees
    the first one's pins (docs/dev/v1-auto-config.md D1). Never fails a start: a cache it cannot write, or a lock held
    past `wait` seconds, is warned about."""
    fd = None
    with contextlib.suppress(OSError):
        CACHE.mkdir(parents=True, exist_ok=True)
        fd = os.open(CACHE / "pin.lock", os.O_RDWR | os.O_CREAT, 0o600)     # a concurrent create opens the same file
    for waited in range(wait if fd is not None else 0):
        with contextlib.suppress(BlockingIOError):
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        if not waited:
            _say("waiting for another `holon serve` to pin its host memory")
        time.sleep(1)
    else:
        _say("WARNING: no host-memory lock (no cache, or held 900 s): a concurrent start may pin past the headroom")
    try:
        yield
    finally:
        if fd is not None:
            os.close(fd)


@contextlib.contextmanager
def admit_restart(config: EngineConfig, ready: Mapping[str, object] | None, derived: bool) -> Iterator[EngineConfig]:
    """A restart of a history entry, under the pin lock: refused (kind "memory") when the host memory the entry pinned
    at its last start no longer fits above half the headroom; an entry never started has its derived cap lowered to
    what is free above that."""
    with pin_lock():
        available, total, _ = platform.host_ram()
        room = available - platform.host_headroom(total) // 2
        if derived and ready and int(ready.get("host_bytes", 0)) > room:
            raise StartFailed("memory", f"host memory: a restart needs {int(ready['host_bytes']) / 2**30:.1f} GiB, "
                              f"{room / 2**30:.1f} GiB free above half the headroom")
        yield config if ready or not derived else dataclasses.replace(config, host_memory=max(0, min(
            config.host_memory, room)))


def checkpoint_bytes(path: str) -> int:
    """A checkpoint's bytes: a directory's files, a split file's shards (`-0000N-of-`), or the file."""
    p = pathlib.Path(path)
    files = [*p.iterdir()] if p.is_dir() else p.parent.glob(re.sub(r"-\d{5}-of-", "-*-of-", p.name))
    return sum(f.stat().st_size for f in files if f.is_file())


def fit_lock(found: tuple | None, flags: Mapping[str, object], owners: list, checkpoint: str
             ) -> tuple[dict[str, object], dict[str, str], list[str], list[str]]:
    """What fits the lock to this machine (docs/dev/v1-auto-config.md D1, D5), asking `owners` (the model plugin, the
    plugins declaring the lock's options and sources): settings giving the lock's `[sources]` no flag gives
    (`MakesSources`, into CACHE; when one is unavailable or the cache cannot be written, speculation off instead: they
    are drafters' heads), the spill (`Spills`; one setting an option a flag gives is dropped) with its notes, and the
    notes of manual alternatives."""
    if found is None:
        return {}, {}, [], []
    given, extra, spill, notes = flags.get("plugin_options", {}), {"plugin_options": {}}, {}, []
    for key, identity in found[0].plan.sources.items() if flags.get("max_draft_tokens") != 0 else ():
        try:
            if key not in given:
                extra["plugin_options"][key] = source_file(identity, owners)
        except (ValueError, OSError) as e:
            _say(f"WARNING: {key} {identity} unavailable ({e}): speculation off")
            extra = {"max_draft_tokens": 0}
            break
    options = layered(plans.lock_overrides(found[0]), dict(flags)).get("plugin_options", {})
    manual = []
    for got in (getattr(o, "spill", lambda *a: None)(options, checkpoint) for o in owners):
        if got and not set(got[0]) & set(given):
            spill |= got[0]
            (notes if got[0] else manual).append("".join(f"{k}={v} " for k, v in got[0].items()) + got[1])
    return extra, spill, notes, manual


def source_file(identity: str, owners: list) -> str:
    """The file `identity` names, in or into `CACHE/sources/<identity>/`, from the first owner whose `make_source`
    claims it; ValueError why not."""
    if not re.fullmatch(r"[A-Za-z0-9._+-]+@[0-9a-f]{16}", identity):
        raise ValueError("not a file identity")
    (directory := CACHE / "sources" / identity).mkdir(parents=True, exist_ok=True)
    path = next((p for o in owners if (p := getattr(o, "make_source", lambda *a: None)(identity, str(directory)))),
                None)
    if path is None:
        raise ValueError("no installed plugin makes it")
    return path


def auto_context(model_plugin: object, source: object) -> int:
    """`--max-model-len auto`: the context the checkpoint was trained for (the model plugin's optional hook), at most
    AUTO_CONTEXT."""
    trained = getattr(model_plugin, "trained_context", None)
    return min((trained and trained(source.config(), source.kind)) or AUTO_CONTEXT, AUTO_CONTEXT)


def find_lock(directory: pathlib.Path, source: object, fp: Fingerprint, current: Mapping[str, str]) -> tuple | None:
    """The lock `holon serve` starts from (loudly: which, or why none). A history it cannot read (two model ids for one
    build, a lock edited by hand) is warned about, and defaults serve."""
    found = None
    if not directory.is_dir():
        _say(f"WARNING: no lock directory {directory}: serving defaults (name one with --plans)")
        return None
    try:
        found = plans.find(directory, source.identity(), fp, current)
    except ValueError as e:
        _say(f"WARNING: unreadable lock history: {e}; serving defaults")
        return None
    if found is None:
        _say(f"no current lock for {source.identity()} in {directory}: serving defaults")
    else:
        _say(f"starting from {found[2]}" + (f" (unverified: {found[0].plan.fingerprint_class}'s)" if found[1] else ""))
    return found


def locked_models(directory: pathlib.Path, fp: Fingerprint, current: Mapping[str, str]) -> list[tuple[str, object]]:
    """Every model id with lock histories under `directory`, as `holon serve` would start it on `fp` (the same lookup:
    `plans.history_id`, `nearest_class`): with its result (None: outdated), or the ValueError that stops `holon serve`
    reading its history (e.g. two model ids for one build)."""
    rows: list[tuple[str, object]] = []
    for model_id in plans.model_ids(directory) if directory.is_dir() else ():
        try:
            rows.append((model_id, plans.nearest_class(directory, plans.history_id(directory, model_id), fp, current)))
        except ValueError as e:
            rows.append((model_id, e))
    return rows


def models_table(rows: list[tuple[str, object]]) -> list[str]:
    """`holon models`' lines, by name: this card's locks first (the files to use here), then borrowed ones, outdated
    ones, unreadable histories."""
    def row(model_id: str, found: object) -> tuple[int, str, str]:
        name = model_id.rpartition("@")[0]
        if isinstance(found, ValueError):
            rank, status = 3, f"unreadable: {found}"
        elif found is None:
            rank, status = 2, "outdated (pins other code): serves defaults"
        else:
            lock, borrowed, path = found
            whose = f"{lock.plan.fingerprint_class.partition('/')[0]}'s lock {path.stem}, unverified" if borrowed \
                else f"this card's lock {path.stem}"
            settings = lock.plan.engine
            rank, status = int(borrowed), (f"{whose}; context {settings.get('max_model_len')}, "
                                           f"{settings.get('max_seqs')} sequences")
        return rank, name, status + ("" if SERVABLE.fullmatch(name) else "; not servable by name")
    return [f"  {name:<50} {status}" for _, name, status in sorted(row(*r) for r in rows)]


def repo_ref(ref: str, rows: list[tuple[str, object]]) -> str:
    """A repository reference (`<org>/<repo>`) under which exactly one locked model has a verified lock for this GPU
    names that model (whole path segments; release design §14.3); several are an error; none leaves `ref` as it is."""
    ref = ref.rstrip("/")
    verified = sorted(m.rpartition("@")[0] for m, found in rows if isinstance(found, tuple) and not found[1])
    under = [name for name in verified if ref.count("/") == 1 and name.startswith(ref + "/")]
    if len(under) > 1:
        raise SystemExit(f"holon: {ref}: several models have a lock for this GPU, name one: {', '.join(under)}")
    if under:
        _say(f"{ref} -> {under[0]}, this card's lock")
    return under[0] if under else ref


def local_model(ref: str, plugins: PluginSet, kind: str | None, rows: list[tuple[str, object]],
                fp: Fingerprint | None, host_memory: int = 0) -> str:
    """The local path `holon serve` serves `ref` from: an existing path as it is; else the one weight source's claim
    of `ref` (after `repo_ref`), logged, refused before fetching when larger than the GPU's memory (`fp`; None: no
    check) unless this card has a lock for it (`rows`: the lock decides what fits) or the deployment gives plugins host
    memory (`host_memory`; the build's per-pool report judges the fit then), then fetched."""
    if pathlib.Path(ref).exists():
        return ref
    try:
        claim = claim_ref(plugins, name := repo_ref(ref, rows), kind)
    except PluginError as e:                                  # a user's mistake: the message, not a traceback
        raise SystemExit(f"holon: {e}") from None
    size = f"{claim.nbytes / 2**30:.1f} GiB" if claim.nbytes else "size unknown"
    _say(f"{ref}: {claim.what} ({size})")
    locked = name in {m.rpartition("@")[0] for m, found in rows if isinstance(found, tuple)}
    if fp is not None and fp.vram_gib > 0 and not host_memory and not locked and claim.nbytes and \
            claim.nbytes > fp.vram_gib * 2**30:
        raise SystemExit(f"holon: {claim.what} is {size}, more than this GPU's {fp.vram_gib} GiB: it cannot fit "
                         "(`holon models` lists the models locked for it; --no-size-check skips this check)")
    return claim.fetch()


def declared_options(installed: Mapping[str, Installed]) -> dict[str, dict[str, Option]]:
    """Plugin option -> {declaring plugin: its declaration}, from the manifests (manifest `[options]`)."""
    out: dict[str, dict[str, Option]] = {}
    for name, inst in sorted(installed.items()):
        for key, option in inst.manifest.options.items():
            out.setdefault(key, {})[name] = option
    return out


def outside_choices(options: Mapping[str, object], declared: Mapping[str, Mapping[str, Option]]) -> list[str]:
    """The options whose value a declaring plugin's `choices` exclude (an empty value deletes and is not checked)."""
    return [f"{key}={value}: not one of {', '.join(o.choices)} ({name})" for key, value in sorted(options.items())
            if value != "" for name, o in declared.get(key, {}).items() if o.choices and str(value) not in o.choices]


def check_flags(flags: Mapping[str, object], declared: Mapping[str, Mapping[str, Option]]) -> None:
    """A flag's option value outside a declaring plugin's `choices` is an error, before anything is fetched."""
    if bad := outside_choices(flags.get("plugin_options", {}), declared):
        raise SystemExit("holon: --plugin-options " + "; ".join(bad))


def checked_options(history: list[tuple[EngineConfig, str | None]],
                    declared: Mapping[str, Mapping[str, Option]]) -> list[tuple[EngineConfig, str | None]]:
    """The option checks before any start (the flags passed `check_flags`): a key no plugin declares is warned about
    (it is still served and planned); a lock whose value is outside a declaring plugin's `choices` is warned about
    and dropped from the history, so defaults and flags serve."""
    for key in sorted({k for cfg, _ in history for k in cfg.plugin_options} - declared.keys()):
        _say(f"WARNING: no installed plugin declares the option {key} (`holon serve --help` lists them)")
    kept = []
    for cfg, lock in history:
        if lock is not None and (bad := outside_choices(cfg.plugin_options, declared)):
            _say(f"WARNING: {lock}: {'; '.join(bad)}; serving defaults")
        else:
            kept.append((cfg, lock))
    return kept


def plugins_help(installed: Mapping[str, Installed]) -> str:
    """`holon serve --help`'s epilog: the options the installed plugins declare, and the plugins `--slots` selects
    among (the slots neither bound per site nor chosen otherwise: the model by architecture, the frontend by flag)."""
    lines = ["plugin options (--plugin-options KEY=VALUE; KEY= deletes the lock's):"]
    for key, by in sorted(declared_options(installed).items()):
        for name, o in by.items():
            extra = "; ".join(x for x in (o.default and f"default: {o.default}",
                                          o.choices and f"one of: {', '.join(o.choices)}") if x)
            text = f"{o.help}" + (f" ({extra})" if extra else "") + f" [{name}]"
            lines += textwrap.wrap(text, 100, initial_indent=f"  {key:<26} ", subsequent_indent=" " * 29)
    lines.append("selectable plugins (--slots SLOT=PLUGIN):")
    for slot, cardinality in SLOTS.items():
        names = sorted(n for n, i in installed.items() if i.manifest.slot == slot)
        if names and cardinality is not Cardinality.PER_SITE and slot not in ("model", "frontend"):
            lines.append(f"  {slot:<26} {', '.join(names)}")
    return "\n".join(lines)


def list_models(directory: pathlib.Path, installed: Mapping[str, Installed]) -> None:
    """`holon models`: the locked models for the first visible GPU, as `holon serve` would start each."""
    if not platform.visible_sms():
        raise SystemExit("holon models: no visible GPU (CUDA_VISIBLE_DEVICES names the one to list for)")
    fp = platform.fingerprint(torch.device("cuda"))
    rows = locked_models(directory, fp, plans.current_srcs(installed))
    print(f"{fp.cls} (the first visible GPU); locks in {directory}")
    print("\n".join(models_table(rows)) or "  (none)")
    print("serve one: holon serve <name>")


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
    installed = discover()
    parser = argparse.ArgumentParser(prog="holon")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("prebuild", help="build the plugins' native code for the visible GPUs ahead of time")
    build.add_argument("--sm", type=int, nargs="+", default=[], help="target SMs, e.g. 86 89 120 (default: visible)")
    models = sub.add_parser("models", help="the models locked for the visible GPU, and how `holon serve` starts each")
    models.add_argument("--plans", type=pathlib.Path, default=plans.TREE / "plans", help="the lock histories")
    serve = sub.add_parser("serve", help="serve a model over a frontend plugin", epilog=plugins_help(installed),
                           formatter_class=argparse.RawDescriptionHelpFormatter)
    launch = serve.add_argument_group("launch (never in a lock)")
    launch.add_argument("model", help="checkpoint directory or file, or a reference a weight source resolves "
                                      "(e.g. a hub id; `holon models` lists the locked ones)")
    launch.add_argument("--host", default="127.0.0.1", help="address to listen on (default: 127.0.0.1)")
    launch.add_argument("--port", type=int, default=8000, help="port to listen on (default: 8000)")
    launch.add_argument("--frontend", default=None, help="frontend plugin name (default: the one installed)")
    launch.add_argument("--supervised", action=argparse.BooleanOptionalAction, default=True,
                        help="restart a failed engine and resubmit its requests (default: on)")
    launch.add_argument("--plans", type=pathlib.Path, default=plans.TREE / "plans",
                        help="the lock histories (default: plans/ of this tree)")
    launch.add_argument("--no-lock", action="store_true", help="serve defaults and flags, not the model's lock")
    launch.add_argument("--no-size-check", action="store_true",
                        help="fetch a model reference even if it is larger than the GPU's memory")
    engine_flags(serve.add_argument_group("serving settings (flags override the model's lock, which overrides these "
                                          "defaults)"), given_only=True)
    args = parser.parse_args(argv)
    if args.command == "prebuild":
        return prebuild(args.sm)
    if args.command == "models":
        return list_models(args.plans, installed)
    flags = engine_settings(args)

    plugins, declared = PluginSet(installed), declared_options(installed)
    check_flags(flags, declared)
    device = torch.device(flags.get("device", "cuda"))
    fp, current = platform.fingerprint(device), plans.current_srcs(installed)
    rows = locked_models(args.plans, fp, current)          # names this card's file even under --no-lock
    sized = None if args.no_size_check or device.type != "cuda" else fp
    model = local_model(args.model, plugins, flags.get("source"), rows, sized, flags.get("host_memory", 0))
    source, model_plugin = resolve_model(plugins, EngineConfig(model, source=flags.get("source")))
    found = None if args.no_lock else find_lock(args.plans, source, fp, current)
    # Fitted to this machine (docs/dev/v1-auto-config.md): the lock's files, a spill, the host budget (pin lock held
    # from the reading until an engine is ready), no defaults entry for a lock whose defaults cannot hold the model.
    keys = [*found[0].plan.engine.get("plugin_options", {}), *found[0].plan.sources] if found else []
    owners = [model_plugin]
    for n in sorted({n for k in keys for n in declared.get(k, {}) if installed[n].manifest.slot != "kernel"}):
        with contextlib.suppress(PluginError):           # a plugin that cannot load fails its engine's start instead
            owners.append(plugins.load(n))
    extra, spill, notes, manual = fit_lock(found, flags, owners, model)
    pin = contextlib.ExitStack()
    pin.enter_context(pin_lock())
    budget = host_budget(flags)
    keep = not (found and fp.vram_gib and checkpoint_bytes(model) > fp.vram_gib * 2**30 and "host_memory" not in flags)
    history = serve_history(model, flags | ({"host_memory": budget} if budget is not None else {}), found,
                            auto_context(model_plugin, source), serve_defaults(args.plans), lock_extra=extra,
                            spill=spill, keep_defaults=keep)
    spilled = history[-2][0] if spill else None
    history = checked_options(history, declared)
    if not history:
        raise SystemExit("holon: no configuration left to serve (the lock's was refused above, and the defaults "
                         "cannot hold this model): --no-lock serves defaults and your flags")
    config, lock = history[-1]
    memory_only = {i: "(spilled: " + "; ".join(notes) + ")" for i, (c, _) in enumerate(history) if c is spilled}
    admit = functools.partial(admit_restart, derived=budget is not None)

    def refused(e: object) -> SystemExit:
        hints = [*manual, "`holon models` lists the models locked for this card"] * ("host memory" in str(e))
        return SystemExit(f"holon: {e}" + "".join(f"\n  {h}" for h in hints))

    bundle = model_plugin.bundle(source)
    names = [m.name for m in plugins.by_slot("frontend")]
    name = args.frontend or (names[0] if len(names) == 1 else None)
    if name is None:
        raise PluginError(f"choose a frontend with --frontend among {names}")
    frontend = plugins.load(name)

    async def run() -> None:
        if not args.supervised:
            async with engine_process(config, lock=lock) as client:
                pin.close()
                await frontend.serve(client, bundle, args.host, args.port, config.plugin_options)
            return
        supervisor = Supervisor(history, memory_only=memory_only, admit=admit)
        try:
            await supervisor.start()
        except RuntimeError as e:
            raise refused(e) from None
        finally:
            pin.close()                                       # the first engine has pinned its host memory
        try:
            serving = asyncio.ensure_future(frontend.serve(supervisor, bundle, args.host, args.port,
                                                           config.plugin_options))
            gave_up = asyncio.ensure_future(supervisor.failed())
            done, _ = await asyncio.wait({serving, gave_up}, return_when=asyncio.FIRST_COMPLETED)
            serving.cancel()
            gave_up.cancel()
            await asyncio.gather(serving, gave_up, return_exceptions=True)   # the frontend stops before the engine
            if gave_up in done:
                raise refused(gave_up.result())
            serving.result()                                  # re-raises whatever stopped the frontend
        finally:
            await supervisor.close()

    asyncio.run(run())
