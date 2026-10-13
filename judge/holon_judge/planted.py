"""Planted bugs that calibrate the judge: each breaks one path a check guards, and the check must fail.

- The bounded check (`bounded --plant`, judge/tests/test_bounded.py): the configurations using the broken path fail.
- GGUF Tier A (`l2 --plant`, M0.1 design §4.6): bugs in what a GGUF file means (configuration, names, layouts),
  planted in the weights the engine loads; L2 fails.
- Graphs vs eager (`bounded --graphs-vs-eager --plant moe-padding-duplicate-ids`): a bug only graph padding rows
  reach; the graphed configuration fails where the padded bucket's expert path is mmq.
- Speculative decoding on a recurrent mixer (`bounded --spec-vs-plain --plant replay-off-by-one`, …): the GDN
  rollback broken; the speculating configurations fail.
- Logits processors' masks inside graphs (`mask-rows-not-reset`): the toy processor's graphs-vs-eager proof
  (plugins/tests/test_ngram.py) fails.
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
def replay_off_by_one() -> Iterator[None]:
    """Speculation on a recurrent mixer: the GDN rewind keeps the state after n accepted tokens, not 1 + n (the
    verify row's own input token is lost whenever a draft was accepted)."""
    from holon_std.mixers.gdn.units import GDN

    rewind = GDN.rewind

    def off_by_one(self, ctx):
        n_accept = ctx.seq_table.n_accept
        before = n_accept.clone()
        n_accept.sub_(1).clamp_(min=0)
        rewind(self, ctx)
        n_accept.copy_(before)

    with mock.patch.object(GDN, "rewind", off_by_one):
        yield


@contextlib.contextmanager
def verify_writes_state() -> Iterator[None]:
    """Speculation on a recurrent mixer: a verify row's delta rule writes its final state over the live one (no
    sink), so the replay starts from the state after every draft, rejected ones included."""
    from holon_std.mixers.gdn.units import GDN

    call = GDN._rule_call

    def live(self, rule, handle, mixed, a, b, ssm_state, slots, has_init, snapshot=None, out_slots=None):
        return call(self, rule, handle, mixed, a, b, ssm_state, slots, has_init, snapshot, None)

    with mock.patch.object(GDN, "_rule_call", live):
        yield


@contextlib.contextmanager
def unmasked_padding() -> Iterator[None]:
    """Speculative verification ignores `n_draft`: every column of a verify row counts as a real draft, so padding
    and a drafter's stale rows past its real drafts (tokens and laws) can be accepted (docs/m04-spec.md §5)."""
    from holon_std.spec.greedy import verifier as greedy
    from holon_std.spec.rejection import verifier as rejection

    def unmasked(verify):
        def planted(logits, ctx):
            n_draft = ctx.seq_table.n_draft
            before = n_draft.clone()
            n_draft.fill_(ctx.seq_table.draft_tokens.shape[1])
            try:
                return verify(logits, ctx)
            finally:
                n_draft.copy_(before)
        return planted

    with contextlib.ExitStack() as stack:
        for module in (greedy, rejection):
            planted = unmasked(module.verify)
            stack.enter_context(mock.patch.object(module, "verify", planted))
            stack.enter_context(mock.patch.object(module.Plugin, "verify", staticmethod(planted)))
        yield


@contextlib.contextmanager
def dflash_tap_offset() -> Iterator[None]:
    """DFlash reads the target one layer late: the residual stream after layer id + 1 for each `target_layer_ids`
    entry (llama.cpp's stored numbering, the next layer's input, taken as HF's)."""
    from holon_plugins.drafters.dflash import plugin

    taps = plugin.tap_layers
    with mock.patch.object(plugin, "tap_layers", lambda cfg: tuple(i + 1 for i in taps(cfg))):
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


@contextlib.contextmanager
def ggml_experts_wrong_type() -> Iterator[None]:
    """The first expert stack of a type with a same-size twin is read as the twin (IQ3_S as Q3_K, Q4_K as Q4_0: 110
    and 144 bytes per 256 values)."""
    from holon_plugins.kernels.ggml_cuda import plugin

    twin = {plugin.TYPES["iq3_s"]: plugin.TYPES["q3_k"], plugin.TYPES["q4_k"]: plugin.TYPES["q4_0"]}
    init, done = plugin.LinearExperts.__init__, []

    def planted(self, ext, ctx, **facts):
        init(self, ext, ctx, **facts)
        if not done and self.type in twin:
            self.type = twin[self.type]
            done.append(facts)
    with mock.patch.object(plugin.LinearExperts, "__init__", planted):
        yield
    print(f"plant ggml-experts-wrong-type: {done}", file=sys.stderr)


@contextlib.contextmanager
def ggml_experts_stale_q8_1(every: int = 97) -> Iterator[None]:
    """Every `every`-th expert call reads the activations of the previous call with the same `in` and rows per token
    (what a stale q8_1 input would hold; it models a stale per-token input, not a stale compacted buffer). Reports
    how often it fired on each path."""
    from holon_plugins.kernels.ggml_cuda import plugin

    call, last, calls, fired = plugin.LinearExperts.__call__, {}, itertools.count(1), collections.Counter()

    def planted(self, x, parts, ids):
        key = (self.k, x.shape[1])
        prev, last[key] = last.get(key), x.clone()
        if next(calls) % every == 0 and prev is not None and len(prev) >= len(x):
            fired["mmvq" if len(x) <= self.mmvq_tokens else "mmq"] += 1
            x = prev[:len(x)]
        return call(self, x, parts, ids)
    with mock.patch.object(plugin.LinearExperts, "__call__", planted):
        yield
    print(f"plant ggml-experts-stale-q8_1 fired: {dict(fired)}", file=sys.stderr)


@contextlib.contextmanager
def moe_padding_duplicate_ids() -> Iterator[None]:
    """Graph padding rows route all k slots to one expert (E − 2), and every expert workspace starts the layer filled
    with a real row's compacted index (0): mmq's compaction then leaves k − 1 stale entries per padding row that write
    over real row 0, while mmvq applies each slot on its own (docs/m02-moe.md §4.6; L0's plant, in the engine). Only
    padded decode steps — graph replays — have padding rows; the step's device slots mark them."""
    from holon.core import executor
    from holon_std.moe import units

    context, forward, step = executor.Executor._context, units.MoE.forward, {}

    def planted_context(self, plan, *args, **kwargs):
        ctx = context(self, plan, *args, **kwargs)
        step.update(slots=ctx.batch.dev.slots, padding=self.state.padding_slot)
        return ctx

    def planted_forward(self, x):
        slots, route = step.get("slots"), self._topk
        if slots is None or len(slots) != len(x):            # not one token per row: no padding rows
            return forward(self, x)

        def padded_route(logits):
            weights, ids = route(logits)
            for kernel in (self._gate_up._kernel, self.down._kernel):
                if hasattr(kernel, "workspace"):
                    kernel.workspace.view(torch.int32).fill_(0)
            pad = (slots == step["padding"])[:, None]
            return weights, torch.where(pad, logits.shape[1] - 2, ids).to(ids.dtype)
        self._topk = padded_route
        try:
            return forward(self, x)
        finally:
            self._topk = route
    with mock.patch.object(executor.Executor, "_context", planted_context), \
            mock.patch.object(units.MoE, "forward", planted_forward):
        yield


def _hf_experts(layer: int, edit: Callable[[int, str], tuple[int, str]]) -> Callable[[str], str]:
    """A rename of HF per-expert tensor names in `layer`: (expert, projection) -> (expert, projection)."""
    import re

    pattern = re.compile(rf"model\.layers\.{layer}\.mlp\.experts\.(\d+)\.(\w+)_proj\.weight")

    def rename(name: str) -> str:
        m = pattern.fullmatch(name)
        if m is None:
            return name
        e, proj = edit(int(m[1]), m[2])
        return f"model.layers.{layer}.mlp.experts.{e}.{proj}_proj.weight"
    return rename


def hf_moe_plants(layer: int, experts: int) -> dict[str, Callable[[], contextlib.AbstractContextManager[None]]]:
    """The bf16 HF path's plants (docs/m02-moe.md §4.6), in `layer` of a model with `experts` experts."""
    swap = {"gate": "up", "up": "gate"}
    return {
        "hf-stack-off-by-one": lambda: _loaded(rename=_hf_experts(layer, lambda e, p: ((e + 1) % experts, p))),
        "hf-gate-up-swapped": lambda: _loaded(rename=_hf_experts(layer, lambda e, p: (e, swap.get(p, p)))),
        "hf-two-experts-swapped": lambda: _loaded(rename=_hf_experts(layer, lambda e, p: ({0: 1, 1: 0}.get(e, e), p))),
        "hf-renormalize-ignored": lambda: _moe(init=lambda kw: kw | {"renormalize": False})}


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


@contextlib.contextmanager
def _qwen35_facts(**changes) -> Iterator[None]:
    """models.qwen3_5's recipe reading `changes` into its facts (a misread configuration or file)."""
    from holon_plugins.models.qwen3_5 import recipe

    facts = recipe._facts
    with mock.patch.object(recipe, "_facts", lambda c, s: facts(c, s) | changes):
        yield


@contextlib.contextmanager
def _qwen35_transform(name: str, fn: Callable[[torch.Tensor], torch.Tensor]) -> Iterator[None]:
    """One of models.qwen3_5's weight-map value transforms replaced (a misread GGUF tensor)."""
    from holon.core.plugins import Registry

    get = Registry.get

    def planted(self, key):
        return fn if key == name else get(self, key)
    with mock.patch.object(Registry, "get", planted):
        yield


@contextlib.contextmanager
def qwen35_gate_one_block() -> Iterator[None]:
    """The attention output gate split as q's whole block, then the gate's (not per head)."""
    from holon_std.mixers.gqa import units

    def planted(self, q, t):
        q, gate = q.split(self.heads * self.head_dim, dim=-1)
        return q.contiguous(), gate.reshape(t, self.heads, self.head_dim)
    with mock.patch.object(units.GQA, "_split_gate", planted):
        yield


@contextlib.contextmanager
def qwen35_stale_state() -> Iterator[None]:
    """Every row reads its recurrent-state unit, also a row that starts a request (has-initial-state forced)."""
    from holon_std.mixers.gdn import units

    prepare = units.Planner.prepare_device

    def planted(self, meta, ctx):
        prepare(self, meta, ctx)
        meta.has_init = torch.ones_like(meta.has_init)
    with mock.patch.object(units.Planner, "prepare_device", planted):
        yield


@contextlib.contextmanager
def qwen35_qk_unnormalized() -> Iterator[None]:
    """The delta rule without its q/k l2 normalisation."""
    from holon_plugins.kernels.fla import plugin

    def without(fn):
        return functools.wraps(fn)(lambda *a, **kw: fn(*a, **(kw | {"use_qk_l2norm_in_kernel": False})))
    with mock.patch.object(plugin, "chunk_gated_delta_rule", without(plugin.chunk_gated_delta_rule)), \
            mock.patch.object(plugin, "fused_sigmoid_gating_delta_rule_update",
                              without(plugin.fused_sigmoid_gating_delta_rule_update)):
        yield


@contextlib.contextmanager
def qwen35_conv_state_dropped() -> Iterator[None]:
    """The causal conv forgets its carried inputs between prefill chunks (a later chunk restarts its window); decode
    rows keep theirs, so one-shot prefill and decode stay correct and only chunked prefill differs."""
    from holon_plugins.kernels.gdn_triton import plugin

    call = plugin.CausalConv1d.__call__

    def planted(self, x, weight, state, slots, query_start, has_init):
        decode = (query_start[1:] - query_start[:-1]) == 1
        return call(self, x, weight, state, slots, query_start, has_init & decode.to(has_init.dtype))
    with mock.patch.object(plugin.CausalConv1d, "__call__", planted):
        yield


@contextlib.contextmanager
def payload_one_block_early() -> Iterator[None]:
    """A published payload filed at the block before its position (a hit there resumes from a later state)."""
    from holon.core import state

    register = state.StateManager.register

    def planted(self, seq, spec, at, payload):
        register(self, seq, spec, at - self.index.block, payload)
    with mock.patch.object(state.StateManager, "register", planted):
        yield


@contextlib.contextmanager
def padding_on_a_real_slot() -> Iterator[None]:
    """Graph padding rows on request slot 0 instead of the padding slot (their writes land in a real request)."""
    from holon.core import executor

    padded = executor.padded
    with mock.patch.object(executor, "padded", lambda plan, rows, slot: padded(plan, rows, 0)):
        yield


@contextlib.contextmanager
def mask_rows_not_reset() -> Iterator[None]:
    """A step with constrained rows stages their logits rows without the padding (an index past the step's rows) up
    to its count of logits rows. An eager step reads only the staged ones, but a graph replay reads as many as it was
    captured with, so it masks the rows past them by indices earlier steps left: a free request on a row a
    constrained one held is masked as that one (docs/m05-grammar.md §3.6). Signal: the graphed run's tokens differ from
    the eager run's on the same padded plans (plugins/tests/test_ngram.py)."""
    from holon.core.inputs import Uploader

    put = Uploader.put

    def planted(self, name, values):
        if name == "mask_rows" and (real := values[values < len(values)]).size:
            values = real
        return put(self, name, values)
    with mock.patch.object(Uploader, "put", planted):
        yield


QWEN35_PLANTS: dict[str, Callable[[], contextlib.AbstractContextManager[None]]] = {
    "qwen35-tiled-as-grouped": lambda: _qwen35_facts(order="grouped"),
    "qwen35-rotary-all-dims": lambda: _qwen35_facts(rotary_dim=256, sections=[]),     # partial_rotary_factor read as 1
    "qwen35-norm-offset-twice": lambda: _qwen35_transform("qwen3_5.minus_one", lambda t: t.float()),
    "qwen35-a-log-sign": lambda: _qwen35_transform("qwen3_5.a_log", lambda t: -torch.log(-t.float())),
    "qwen35-gate-one-block": qwen35_gate_one_block, "qwen35-stale-state": qwen35_stale_state,
    "qwen35-qk-unnormalized": qwen35_qk_unnormalized, "qwen35-conv-state-dropped": qwen35_conv_state_dropped,
    "payload-one-block-early": payload_one_block_early, "padding-on-a-real-slot": padding_on_a_real_slot}


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
    # MoE Tier A and the bf16 HF path (Qwen3-30B-A3B: the one-layer plants at layer 24)
    **moe_plants(24), **hf_moe_plants(24, 128),
    # MoE Tier B
    "ggml-experts-wrong-type": ggml_experts_wrong_type, "ggml-experts-stale-q8_1": ggml_experts_stale_q8_1,
    "rare-ggml-experts-stale-q8_1": lambda: ggml_experts_stale_q8_1(every=9973),
    "moe-padding-duplicate-ids": moe_padding_duplicate_ids, **QWEN35_PLANTS,
    # Speculative decoding on a recurrent mixer (bounded --spec-vs-plain; docs/m04-spec.md §5)
    "replay-off-by-one": replay_off_by_one, "verify-writes-state": verify_writes_state,
    # Speculative verification (L0-dist, plugins/tests/test_rejection.py; the n_draft cap, test_spec.py)
    "unmasked-padding": unmasked_padding,
    # Draft heads (L0-drafter, `holon_judge drafter`)
    "dflash-tap-offset": dflash_tap_offset,
    # Logits processors (docs/m05-grammar.md §5): the graphs-vs-eager toy proof (plugins/tests/test_ngram.py)
    "mask-rows-not-reset": mask_rows_not_reset}
