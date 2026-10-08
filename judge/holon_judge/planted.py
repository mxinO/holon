"""Planted bugs that calibrate the judge: each breaks one path a check guards, and the check must fail.

- The bounded check (`bounded --plant`, judge/tests/test_bounded.py): the configurations using the broken path fail.
- GGUF Tier A (`l2 --plant`, M0.1 design §4.6): bugs in what a GGUF file means (configuration, names, layouts),
  planted in the weights the engine loads; L2 fails.
- GGUF Tier B (`l2 --plant`): bugs in the quantized kernels' use (`kernels.ggml_cuda`); L2 fails, except the rare
  stale workspace, which measures Tier B's sensitivity (predicted before running: fails the decode run, likely passes
  the looser prefill run).

The GGUF plants patch names, so they rely on how the engine reaches them: `_loaded` replaces `load_weights` and
`resolve_model` in `holon.core.engine`'s namespace (it imports them by name), and the kernel plants wrap
`QLinear` in Python, so they act on eagerly run steps — all of L2's (prompt-logprob steps are never graphed); in
a graphed decode step they would act only while the graph is captured.

They patch engine classes, so they apply to every engine built inside the context, graph captures included. The
reference re-scoring is untouched by all three: prompt-logprob requests never adopt cached blocks, and it samples one
token per request, which is never fed back.
"""
from __future__ import annotations

import collections
import contextlib
import dataclasses
import functools
import itertools
import sys
from collections.abc import Callable, Iterator
from unittest import mock

import torch


@contextlib.contextmanager
def adoption_off_by_one() -> Iterator[None]:
    """A prefix hit claims one token more than it adopted, so that token's state is never computed."""
    from holon.core.state import StateManager

    admit = StateManager.admit

    def planted(self, seq):
        ok = admit(self, seq)
        if ok and seq.num_computed and seq.num_computed + 2 <= len(seq.tokens):
            seq.num_computed += 1
        return ok
    with mock.patch.object(StateManager, "admit", planted):
        yield


@contextlib.contextmanager
def wrong_block() -> Iterator[None]:
    """A prefix hit of two or more blocks adopts its first block's state in place of its last block's. (Swapping two
    adopted blocks would be no bug: keys are cached after RoPE, and attention over a prefix ignores its order.)"""
    from holon.core.prefix import PrefixIndex

    match = PrefixIndex.match

    def planted(self, keys, tokens):
        entries = match(self, keys, tokens)
        if len(entries) >= 2:
            entries = [*entries[:-1], dataclasses.replace(entries[-1], units=entries[0].units)]
        return entries
    with mock.patch.object(PrefixIndex, "match", planted):
        yield


@contextlib.contextmanager
def stale_device_token(every: int = 7) -> Iterator[None]:
    """After a sampled token id divisible by `every`, the next step's device-sourced input (depth 1) is the previous
    token."""
    from holon.core.executor import Executor

    store = Executor._store

    def planted(self, ctx, tokens, accepts=None):
        if accepts is not None:
            return store(self, ctx, tokens, accepts)
        slots = ctx.batch.dev.slots[ctx.batch.dev.logits_rows]
        before = self.last_token[slots]
        out = store(self, ctx, tokens, accepts)
        self.last_token[slots] = torch.where(tokens % every == 0, before, tokens)
        return out
    with mock.patch.object(Executor, "_store", planted):
        yield


@contextlib.contextmanager
def _loaded(edit: Callable[..., tuple[str, torch.Tensor]] | None = None, cfg: dict | None = None,
            rename: Callable[[str], str] | None = None) -> Iterator[None]:
    """Every engine built inside loads `edit(name, tensor, config)` for each dense checkpoint tensor, `rename(name)`
    for every tensor (quantized ones included), with `cfg` over its weight source's configuration."""
    from holon.core import engine

    load_weights, resolve = engine.load_weights, engine.resolve_model
    config: dict = {}                                   # of the engine being built: resolved, then loaded

    def planted_resolve(plugins, c):
        source, plugin = resolve(plugins, c)
        config.clear()
        config.update(source.config(), **(cfg or {}))
        source.config = lambda: config
        return source, plugin

    def planted_load(model, ctx, tensors):
        renamed = ((rename(n) if rename else n, t) for n, t in tensors)
        return load_weights(model, ctx, (edit(n, t, config) if edit and isinstance(t, torch.Tensor) else (n, t)
                                         for n, t in renamed))
    with mock.patch.object(engine, "load_weights", planted_load), \
            mock.patch.object(engine, "resolve_model", planted_resolve):
        yield


def _transposed_attn_output(name, t, config):
    return name, (t.T.contiguous() if name.endswith(".attn_output.weight") else t)


def _gate_up_swapped(name, t, config):
    swap = {".ffn_gate.": ".ffn_up.", ".ffn_up.": ".ffn_gate."}
    return next((name.replace(a, b) for a, b in swap.items() if a in name), name), t


def _qk_permuted(name, t, config):
    """llama.cpp's HF -> GGUF rope permutation (convert_hf_to_gguf.py `permute`), applied where it must not be: Qwen
    files keep NEOX order."""
    if name.endswith((".attn_q.weight", ".attn_k.weight")):
        t = t.reshape(-1, 2, config["head_dim"] // 2, t.shape[-1]).transpose(1, 2).reshape(t.shape)
    return name, t


@contextlib.contextmanager
def ggml_wrong_type() -> Iterator[None]:
    """The first Q4_K `qlinear` built reads its blocks as Q4_0 (the same 144 bytes per 256 values)."""
    from holon_plugins.kernels.ggml_cuda import plugin

    init, done = plugin.QLinear.__init__, []

    def planted(self, ext, ctx, **facts):
        init(self, ext, ctx, **facts)
        if not done and self.type == plugin.TYPES["q4_k"]:
            self.type = plugin.TYPES["q4_0"]
            done.append(facts)
    with mock.patch.object(plugin.QLinear, "__init__", planted):
        yield
    print(f"plant ggml-wrong-type: {done}", file=sys.stderr)


@contextlib.contextmanager
def ggml_stale_q8_1(every: int = 97) -> Iterator[None]:
    """Every `every`-th `qlinear` call reads the activations of the previous call sharing its q8_1 workspace (same
    `in`) — what a workspace left stale by a skipped quantization holds. Reports how often it fired on each path."""
    from holon_plugins.kernels.ggml_cuda import plugin

    call, last, calls, fired = plugin.QLinear.__call__, {}, itertools.count(1), collections.Counter()

    def planted(self, x, blocks):
        prev, last[self.k] = last.get(self.k), x.clone()
        if next(calls) % every == 0 and prev is not None and len(prev) >= len(x):
            fired["mmvq" if len(x) <= self.mmvq_rows else "mmq"] += 1
            x = prev[:len(x)]
        return call(self, x, blocks)
    with mock.patch.object(plugin.QLinear, "__call__", planted):
        yield
    print(f"plant ggml-stale-q8_1 fired: {dict(fired)}", file=sys.stderr)


@contextlib.contextmanager
def _moe(layer: int | None = None, init: Callable[[dict], dict] | None = None,
         topk: Callable | None = None, combine: Callable | None = None, merge_swapped: bool = False) -> Iterator[None]:
    """MoE bugs in the engine's std.moe unit, in every layer or only `layer`: its constructor arguments changed by
    `init`, its routing or combine kernel's arguments or results changed, or its gate/up merge given up first."""
    from holon_std.moe import units

    construct, merge = units.MoE.__init__, units.ExpertLinear.merge

    def planted_init(self, ctx, **kw):
        here = layer is None or kw.get("layer") == layer
        construct(self, ctx, **(init(kw) if init and here else kw))
        if here and topk:
            self._topk = functools.partial(lambda f, logits: topk(*f(logits), kw["experts"]), self._topk)
        if here and combine:
            self._combine = functools.partial(lambda f, y, w, out: f(*combine(y, w), out), self._combine)
    with mock.patch.object(units.MoE, "__init__", planted_init), \
            mock.patch.object(units.ExpertLinear, "merge",
                              staticmethod(lambda gate, up: merge(up, gate) if merge_swapped else merge(gate, up))):
        yield


def _swap_gate_up_experts(name: str) -> str:
    swap = {".ffn_gate_exps.": ".ffn_up_exps.", ".ffn_up_exps.": ".ffn_gate_exps."}
    return next((name.replace(a, b) for a, b in swap.items() if a in name), name)


def moe_plants(layer: int) -> dict[str, Callable[[], contextlib.AbstractContextManager[None]]]:
    """Tier A's MoE plants (docs/m02-moe.md §4.6), the one-layer ones at `layer`."""
    return {
        "moe-renormalize-off": lambda: _moe(init=lambda kw: kw | {"renormalize": False}),
        "moe-renormalize-off-one-layer": lambda: _moe(layer, init=lambda kw: kw | {"renormalize": False}),
        "moe-ids-shifted-one-layer": lambda: _moe(layer, topk=lambda w, ids, experts: (w, (ids + 1) % experts)),
        "moe-gate-up-swapped": lambda: _loaded(rename=_swap_gate_up_experts),
        "moe-top-k-7": lambda: _moe(init=lambda kw: kw | {"top_k": kw["top_k"] - 1}),
        "moe-combine-slots-reversed": lambda: _moe(combine=lambda y, w: (y, w.flip(-1).contiguous())),
        "moe-merge-up-first": lambda: _moe(merge_swapped=True)}


PLANTS: dict[str, Callable[[], contextlib.AbstractContextManager[None]]] = {
    "adoption-off-by-one": adoption_off_by_one, "wrong-block": wrong_block,
    "stale-device-token": stale_device_token, "rare-stale-device-token": lambda: stale_device_token(every=101),
    # GGUF Tier A: S5's calibration bug on the file's rope base, and three misreadings of its tensors.
    "gguf-rope-theta-9e5": lambda: _loaded(cfg={"rope_theta": 9e5}),
    "gguf-transposed-attn-output": lambda: _loaded(_transposed_attn_output),
    "gguf-gate-up-swapped": lambda: _loaded(_gate_up_swapped),
    "gguf-qk-permuted": lambda: _loaded(_qk_permuted),
    # GGUF Tier B
    "ggml-wrong-type": ggml_wrong_type, "ggml-stale-q8_1": ggml_stale_q8_1,
    "rare-ggml-stale-q8_1": lambda: ggml_stale_q8_1(every=9973),      # ~ one linear in 55 forwards
    # MoE Tier A (Qwen3-30B-A3B: the one-layer plants at layer 24)
    **moe_plants(24)}
