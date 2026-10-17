"""Planted bugs that calibrate the judge: each breaks one path a check guards, and the check must fail.

- The bounded check (`bounded --plant`, judge/tests/test_bounded.py): the configurations using the broken path fail.
- GGUF Tier A (`l2 --plant`, M0.1 design §4.6): bugs in what a GGUF file means (configuration, names, layouts),
  planted in the weights the engine loads; L2 fails.
- Speculative decoding on a recurrent mixer (`bounded --spec-vs-plain --plant replay-off-by-one`, …): the GDN
  rollback broken; the speculating configurations fail.
- Logits processors' masks inside graphs (`mask-rows-not-reset`): the toy processor's graphs-vs-eager proof
  (plugins/tests/test_ngram.py) fails.
- grammar.xgrammar (docs/m05-grammar.md §5; plugins/tests/test_grammar.py): a verify row's probe shifted by one
  column or never rolled back (the masked L0-dist and the CPU spec identity fail), the apply kernel's bit order
  reversed (L0-mask fails), a completed constraint's stop tokens or every closing bracket masked (the finish reasons
  of constrained generation change).
- GGUF Tier B (`l2 --plant`): bugs in the quantized kernels' use (`kernels.ggml_cuda`); L2 fails, except the rare
  stale workspace, which measures Tier B's sensitivity (predicted before running: fails the decode run, likely passes
  the looser prefill run).
- FP4 in `kernels.ggml_cuda` (docs/m06-lowp.md §4.8: N2–N6, N8, N8b, M1, M4): its repack, `scale2` paths, mmq
  precision and zeroed tail broken, and a checkpoint's `scale2` misread; each also fails L0 where L0 reaches it
  (plugins/tests/test_ggml_cuda*.py), N6 as the pre-launch precision check's error on sm_120.
- FP8 in `kernels.triton_fp8` (§4.8: P1–P4, P7): its block scales, activation scales and weight decode misread, and a
  checkpoint's block scales misread; P1–P4 also fail its L0 (plugins/tests/test_triton_fp8.py).
- The fused q/k norm + rope + KV write (docs/m07-perf.md D3, §6): a token that opens a page written one slot late
  (L2 fails; so would its L0), and decode rows rotated one position late (`bounded`: the configurations' decode
  rows against the reference's teacher-forced prefill rows).
- Expert offload (docs/v1-offload.md §6, the snapshots with adaptive placement on: spikes/x3_policies): a staged
  miss read from the neighbouring slot; a two-phase swap's copy unordered and unwaited (with an unseeded delay on
  the copy stream), or its map to another swap's slot; a host edit in F11's device-owned cache (with the
  precondition check, and without it); F11's victims chosen without the step's needs, or two admissions into one
  victim's slot (the kernel's plant bits); a lent slot returned while still mapped to its expert. Each reports how
  often it fired.

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

import numpy as np
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


def _when() -> str:
    """Where a planted path runs: in a graph capture (then on every replay) or eagerly."""
    return "captured" if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing() else "eager"


@contextlib.contextmanager
def replay_off_by_one() -> Iterator[None]:
    """Speculation on a recurrent mixer: the GDN rewind keeps the state after n accepted tokens, not 1 + n (the
    verify row's own input token is lost whenever a draft was accepted). Reports how often it fired: verify steps
    (the group's first layer, whose rewind writes every layer's windows), eagerly and in graph captures (a captured
    rewind fires on every replay); the startup profile and graph warm-up are included."""
    from holon_std.mixers.gdn.units import GDN, GROUP

    rewind, fired = GDN.rewind, collections.Counter()

    def off_by_one(self, ctx):
        n_accept = ctx.seq_table.n_accept
        before = n_accept.clone()
        n_accept.sub_(1).clamp_(min=0)
        if self._state is not None and self._state[2] == 0 and ctx.meta(GROUP).verify:
            fired[_when()] += 1
        rewind(self, ctx)
        n_accept.copy_(before)

    with mock.patch.object(GDN, "rewind", off_by_one):
        yield
    print(f"plant replay-off-by-one fired: {dict(fired)}", file=sys.stderr)


@contextlib.contextmanager
def verify_writes_state() -> Iterator[None]:
    """Speculation on a recurrent mixer: a verify row's delta rule writes its final state over the live one (no
    sink), so the replay starts from the state after every draft, rejected ones included. Reports how often it fired:
    verify steps (the group's first layer's verify call; the rewind's `out_slots` are its slots), eagerly and in graph
    captures; the startup profile and graph warm-up are included."""
    from holon_std.mixers.gdn.units import GDN

    call, fired = GDN._rule_call, collections.Counter()

    def live(self, rule, handle, mixed, a, b, ssm_state, slots, has_init, snapshot=None, out_slots=None):
        if out_slots is not None and out_slots is not slots and self._state[2] == 0:
            fired[_when()] += 1
        return call(self, rule, handle, mixed, a, b, ssm_state, slots, has_init, snapshot, None)

    with mock.patch.object(GDN, "_rule_call", live):
        yield
    print(f"plant verify-writes-state fired: {dict(fired)}", file=sys.stderr)


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
    `in`) — what a workspace left stale by a skipped quantization holds; a call given a prepared input (D6, shared
    quantization) is handed the stale rows instead of it. Reports how often it fired on each path."""
    from holon_plugins.kernels.ggml_cuda import plugin

    call, last, calls, fired = plugin.QLinear.__call__, {}, itertools.count(1), collections.Counter()

    def planted(self, x, parts):
        rows = x.x if isinstance(x, plugin.Prepared) else x
        prev, last[self.k] = last.get(self.k), rows.clone()
        if next(calls) % every == 0 and prev is not None and len(prev) >= len(rows):
            fired[("mmvq" if len(rows) <= self.mmvq_rows else "mmq") + ("" if x is rows else ", prepared")] += 1
            x = prev[:len(rows)]
        return call(self, x, parts)
    with mock.patch.object(plugin.QLinear, "__call__", planted):
        yield
    print(f"plant ggml-stale-q8_1 fired: {dict(fired)}", file=sys.stderr)


@contextlib.contextmanager
def qk_rope_kv_page_slot() -> Iterator[None]:
    """D3: `attention.qk_rope_kv@1` writes each token that opens a page (offset 0) into the page's second slot, only
    at page boundaries; the first slot keeps what it held. Reports how many calls it changed."""
    from holon_plugins.kernels.triton_attn import plugin

    call, fired = plugin.QkRopeKv.__call__, collections.Counter()

    def planted(self, q, k, v, positions, q_weight, k_weight, kv, unit, off):
        fired["calls"] += 1
        return call(self, q, k, v, positions, q_weight, k_weight, kv, unit, torch.where(off == 0, 1, off))
    with mock.patch.object(plugin.QkRopeKv, "__call__", planted):
        yield
    print(f"plant qk-rope-kv-page-slot fired: {dict(fired)}", file=sys.stderr)


@contextlib.contextmanager
def qk_rope_kv_decode_position() -> Iterator[None]:
    """D3: in decode steps (every row one token) the fused kernel rotates q and k at position + 1; prefill and verify
    rows are right. Reports how many decode calls it changed (graph captures included: the + 1 is captured)."""
    from holon_std.mixers.gqa import units

    project, fired = units.GQA._project_and_write, collections.Counter()

    def planted(self, x, ctx, kv):
        fused = self._fused
        if fused is None or not (ctx.batch.plan.n_query == 1).all():
            return project(self, x, ctx, kv)
        fired["decode"] += 1
        self._fused = lambda q, k, v, positions, *rest: fused(q, k, v, positions + 1, *rest)
        try:
            return project(self, x, ctx, kv)
        finally:
            self._fused = fused
    with mock.patch.object(units.GQA, "_project_and_write", planted):
        yield
    print(f"plant qk-rope-kv-decode-position fired: {dict(fired)}", file=sys.stderr)


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
def _fp4_blocks_edited(elem: str, edit: Callable[[torch.Tensor], torch.Tensor]) -> Iterator[None]:
    """kernels.ggml_cuda's repack of `elem` (what `prepare` holds) with its ggml blocks edited: `edit(blocks [rows,
    blocks per row, block bytes])`."""
    from holon_plugins.kernels.ggml_cuda import plugin

    blocks = plugin._blocks

    def planted(e, w, s):
        out = blocks(e, w, s)
        return out if e != elem else edit(out.view(len(out), -1, plugin.BLOCKS[e][1]).clone()).view(len(out), -1)
    with mock.patch.object(plugin, "_blocks", planted):
        yield


def _nibbles_swapped(b: torch.Tensor) -> torch.Tensor:
    b[..., 4:] = (b[..., 4:] >> 4) | (b[..., 4:] << 4)
    return b


def _sub_scales_permuted(b: torch.Tensor) -> torch.Tensor:
    b[..., :4] = b[..., :4].flip(-1)
    return b


def _e8m0_off_by_one(b: torch.Tensor) -> torch.Tensor:
    b[..., 0] += 1
    return b


@contextlib.contextmanager
def scale2_by_slot() -> Iterator[None]:
    """N4: mmq with ids (down) scales each slot by `scale2[slot]`, not by its expert's."""
    from holon_plugins.kernels.ggml_cuda import plugin

    def planted(y, scale, ids):
        y.mul_(scale[torch.arange(ids.shape[1], device=ids.device).expand_as(ids)][..., None])
    with mock.patch.object(plugin.LinearExperts, "_scale_slots", staticmethod(planted)):
        yield


@contextlib.contextmanager
def scale2_after_swiglu() -> Iterator[None]:
    """N5: mmq's gate/up SwiGLU takes `scale2` after the product (silu(g)·u·s_g·s_u), not on each half before it."""
    from holon_plugins.kernels.ggml_cuda import plugin

    def planted(self, gate_up, y, gate_scale, up_scale, ids):
        self.lib.swiglu(gate_up, y, None, None, None)
        if up_scale is not None:
            y.mul_((gate_scale * up_scale)[ids][..., None])
    with mock.patch.object(plugin.LinearExperts, "_swiglu", planted):
        yield


@contextlib.contextmanager
def ggml_prec_default() -> Iterator[None]:
    """N6, N6b: every mmq call leaves dst's `op_params[3]` zero (llama.cpp's default precision: native W4A4 for FP4
    types on Blackwell), whatever its spec's mode."""
    from holon_plugins.kernels.ggml_cuda import plugin

    init = plugin._Matmul.__init__

    def planted(self, *args, **kwargs):
        init(self, *args, **kwargs)
        self.prec = 0
    with mock.patch.object(plugin._Matmul, "__init__", planted):
        yield


@contextlib.contextmanager
def fp4_tail_not_zeroed() -> Iterator[None]:
    """M4: the `fp4` spec's tail after each FP4 tensor left as found (here 0xFF: NaN scales), not zeroed."""
    from holon_plugins.kernels.ggml_cuda import plugin

    def planted(blocks, nbytes, copy=True):
        buf = torch.full((blocks.numel() + nbytes,), 0xFF, dtype=torch.uint8, device=blocks.device)
        out = buf[:blocks.numel()].view(blocks.shape)
        return out.copy_(blocks) if copy else out
    with mock.patch.object(plugin, "with_tail", planted):
        yield


@contextlib.contextmanager
def _part_scaled(module: str, role: str, factor: float) -> Iterator[None]:
    """N8, N8b, P7: `module`'s `role` part (NVFP4 `scale2`, FP8 block `scale`) read × `factor` by std.weights.hf
    (the judge reads the checkpoint itself)."""
    from holon_std.weights.hf import quant

    load = quant.load

    def planted(d, m, get):
        q = load(d, m, get)
        if m.name == module and role in q.parts:
            q.parts[role] = q.parts[role] * factor
        return q
    with mock.patch.object(quant, "load", planted):
        yield


def fp4_plants(layer: int) -> dict[str, Callable[[], contextlib.AbstractContextManager[None]]]:
    """kernels.ggml_cuda's FP4 plants (docs/m06-lowp.md §4.8), the scale2 ones at `layer`'s o_proj."""
    o_proj = f"model.layers.{layer}.self_attn.o_proj"
    return {
        "nvfp4-repack-nibbles-swapped": lambda: _fp4_blocks_edited("nvfp4", _nibbles_swapped),            # N2
        "nvfp4-repack-scales-permuted": lambda: _fp4_blocks_edited("nvfp4", _sub_scales_permuted),        # N3
        "nvfp4-scale2-by-slot": scale2_by_slot,                                                            # N4
        "nvfp4-scale2-after-swiglu": scale2_after_swiglu,                                                  # N5
        "ggml-prec-default": ggml_prec_default,                                                            # N6, N6b
        "nvfp4-o-proj-scale2-1.25": lambda: _part_scaled(o_proj, "scale2", 1.25),                         # N8
        "nvfp4-o-proj-scale2-1.1": lambda: _part_scaled(o_proj, "scale2", 1.1),                           # N8b
        "mxfp4-repack-scale-off-by-one": lambda: _fp4_blocks_edited("mxfp4", _e8m0_off_by_one),           # M1
        "fp4-tail-not-zeroed": fp4_tail_not_zeroed}                                                         # M4


@contextlib.contextmanager
def _fp8_kernel(scale: Callable | None = None, weight: Callable | None = None,
                act_quant: Callable | None = None) -> Iterator[None]:
    """kernels.triton_fp8 misreading its operands: every call's block scales (`scale(s)`, both paths), W8A16's weight
    bytes (`weight(w)`, decode) or W8A8's activation quantizer (`act_quant`, prefill) replaced."""
    from holon_plugins.kernels.triton_fp8 import kernel

    decode, prefill = kernel.decode, kernel.prefill

    def planted_decode(x, w, s, out_dtype, cfg):
        return decode(x, weight(w) if weight else w, scale(s) if scale else s, out_dtype, cfg)

    def planted_prefill(x, w, s, out_dtype, cfg):
        return prefill(x, w, scale(s) if scale else s, out_dtype, cfg)
    with (mock.patch.object(kernel, "decode", planted_decode), mock.patch.object(kernel, "prefill", planted_prefill),
          mock.patch.object(kernel, "act_quant", act_quant or kernel.act_quant)):
        yield


def _column_major(s: torch.Tensor) -> torch.Tensor:
    """P1: the block scales `[out / 128, in / 128]` read as if stored column-major (a transposed index)."""
    return s.reshape(s.shape[1], s.shape[0]).T.contiguous()


def _subnormals_as_normals(w: torch.Tensor) -> torch.Tensor:
    """P4: e4m3 subnormals (exponent field 0, mantissa m > 0: m · 2^-9) decoded as normals ((1 + m / 8) · 2^-6)."""
    mag = w & 0x7F
    return torch.where((mag > 0) & (mag < 8), w | 0x08, w)


def _act_scale_per_token(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """P3: W8A8's activation scale per row, not per row and 128 columns."""
    a_s = x.float().abs().amax(-1, keepdim=True).clamp_min(1e-10) / 448
    q = (x.float() / a_s).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q, a_s.expand(-1, x.shape[1] // 128).contiguous()


def fp8_plants(layer: int) -> dict[str, Callable[[], contextlib.AbstractContextManager[None]]]:
    """kernels.triton_fp8's plants (docs/m06-lowp.md §4.8, P1–P4) and a checkpoint's FP8 scales misread at
    `layer`'s o_proj (P7). The kernel ones wrap its calls in Python: eager steps only (module docstring)."""
    return {
        "fp8-scale-transposed": lambda: _fp8_kernel(scale=_column_major),                                  # P1
        "fp8-scale-k-shifted": lambda: _fp8_kernel(scale=lambda s: s.roll(-1, 1)),                         # P2
        "fp8-act-scale-per-token": lambda: _fp8_kernel(act_quant=_act_scale_per_token),                    # P3
        "fp8-subnormals-as-normals": lambda: _fp8_kernel(weight=_subnormals_as_normals),                   # P4
        "fp8-o-proj-scales-1.25": lambda: _part_scaled(f"model.layers.{layer}.self_attn.o_proj", "scale", 1.25)}  # P7


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


@contextlib.contextmanager
def _hf_dialect(**changes) -> Iterator[None]:
    """Every dialect std.weights.hf picks, with `changes` (a misread producer convention)."""
    from holon_std.weights.hf import quant

    pick = quant.dialect
    with mock.patch.object(quant, "dialect", lambda qc: dataclasses.replace(pick(qc), **changes)):
        yield


@contextlib.contextmanager
def _hf_grouped(edit: Callable[[dict], dict]) -> Iterator[None]:
    """std.weights.hf's grouping with modules re-pointed: `edit(quantized modules by name)` returns the ones it
    changes (a dialect pairing a part with the wrong module)."""
    from holon_std.weights.hf import quant

    group = quant.group

    def planted(d, header):
        plan, checks = group(d, header)
        changed = edit({m.name: m for m in plan if isinstance(m, quant.Module)})
        return [changed.get(m.name, m) if isinstance(m, quant.Module) else m for m in plan], checks
    with mock.patch.object(quant, "group", planted):
        yield


def _part_from(modules: dict, name: str, other: str, role: str = "scale") -> dict:
    """Module `name` with `other`'s `role` part."""
    m = modules[name]
    return {name: dataclasses.replace(m, parts=m.parts | {role: modules[other].parts[role]})}


def _expert_scales_shifted(layer: int, role: str = "scale") -> Callable[[dict], dict]:
    """Each expert of `layer` takes the next expert's `role` part (that stack off by one)."""
    def edit(modules: dict) -> dict:
        prefix, changed = f"model.layers.{layer}.mlp.experts.", {}
        experts = len({n.split(".")[5] for n in modules if n.startswith(prefix)})
        for name in modules:
            if name.startswith(prefix):
                e, proj = name.removeprefix(prefix).split(".")
                changed |= _part_from(modules, name, f"{prefix}{(int(e) + 1) % experts}.{proj}", role)
        return changed
    return edit


@contextlib.contextmanager
def _gate_up_merge_gate_scales() -> Iterator[None]:
    """A dense gate/up merge that keeps gate's scale rows for up's half too."""
    from holon_std.ops import units

    merged_parts = units._merged_parts

    def planted(us):
        parts = merged_parts(us)
        if parts is not None and len(us) == 2 and "scale" in parts:
            parts["scale"] = torch.cat([us[0].parts["scale"]] * 2)
        return parts
    with mock.patch.object(units, "_merged_parts", planted):
        yield


@contextlib.contextmanager
def _gate_up_scale2_swapped() -> Iterator[None]:
    """An expert gate/up merge whose per-tensor `scale2` rows are swapped (up's for gate's half, and back)."""
    from holon_std.moe import units

    merge = units.ExpertLinear.merge

    def planted(gate, up):
        merged = merge(gate, up)
        if merged is not None and "scale2" in merged.parts:
            merged.parts["scale2"] = merged.parts["scale2"].flip(0)
        return merged
    with mock.patch.object(units.ExpertLinear, "merge", staticmethod(planted)):
        yield


@contextlib.contextmanager
def _mxfp4_scales_edited(edit: Callable[[torch.Tensor], torch.Tensor]) -> Iterator[None]:
    """The dialects' MXFP4 normalization yielding `edit(scale)`."""
    from holon_std.weights.hf.quant import base

    normalize = base.NORMALIZE["mxfp4"]
    with mock.patch.dict(base.NORMALIZE, {"mxfp4": lambda weight, scale: normalize(weight, edit(scale))}):
        yield


def hf_quant_plants(layer: int) -> dict[str, Callable[[], contextlib.AbstractContextManager[None]]]:
    """Quantized HF checkpoints' Tier A and structural plants (docs/m06-lowp.md §4.8: N1, T1, M3, M5, P5, P6; M2 is
    `hf-stack-off-by-one`; and the per-tensor `scale2` stack off by one or its gate/up rows swapped in the merge), the
    one-layer ones in `layer`."""
    attn = f"model.layers.{layer}.self_attn"
    return {
        "ct-global-not-inverted": lambda: _hf_dialect(inverted=frozenset()),                          # N1
        "dialect-k-scale-on-v": lambda: _hf_grouped(                                                   # T1
            lambda m: _part_from(m, f"{attn}.v_proj", f"{attn}.k_proj")),
        "mxfp4-scale-as-e4m3": lambda: mock.patch(                                                     # M3
            "holon_std.weights.hf.reference.e8m0", lambda s: s.view(torch.float8_e4m3fn).float()),
        # M5; a 0 byte would wrap to 0xFF (NaN, refused at load): none in the real files or the fixtures
        "mxfp4-scale-halved": lambda: _mxfp4_scales_edited(lambda s: s - 1),
        "gate-up-merge-gate-scales": _gate_up_merge_gate_scales,                                       # P5
        "hf-scale-stack-off-by-one": lambda: _hf_grouped(_expert_scales_shifted(layer)),               # P6
        "hf-scale2-stack-off-by-one": lambda: _hf_grouped(_expert_scales_shifted(layer, "scale2")),
        "hf-gate-up-scale2-swapped": _gate_up_scale2_swapped}


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


@contextlib.contextmanager
def mask_column_shift() -> Iterator[None]:
    """grammar.xgrammar masks verify column j with the matcher's state after j + 1 drafts instead of j."""
    from holon_plugins.grammar.xgrammar import plugin

    probe = plugin.probe

    def planted(matcher, drafts, out, stop_row):
        if not drafts or not matcher.accept_token(drafts[0]):
            return probe(matcher, drafts, out, stop_row)
        refused = probe(matcher, drafts[1:], out, stop_row)
        matcher.rollback(1)
        return refused
    with mock.patch.object(plugin, "probe", planted):
        yield


@contextlib.contextmanager
def no_rollback_after_probe() -> Iterator[None]:
    """grammar.xgrammar's probe leaves the matcher at the drafts it accepted (xgrammar's rollback does nothing)."""
    import xgrammar

    with mock.patch.object(xgrammar.GrammarMatcher, "rollback", lambda self, n: None):
        yield


def _int32(words: torch.Tensor) -> torch.Tensor:
    """int64 values in [0, 2³²) as the int32 words with the same bits."""
    return torch.where(words >= 1 << 31, words - (1 << 32), words).to(torch.int32)


@contextlib.contextmanager
def mask_bit_order() -> Iterator[None]:
    """grammar.xgrammar's apply reads bit 31 − t of a mask word for token 32·w + t."""
    from holon_plugins.grammar.xgrammar import kernel

    apply = kernel.apply

    def planted(logits, mask, rows):
        bits = torch.arange(32, device=mask.device)
        flipped = (((mask.long()[..., None] >> bits) & 1) << (31 - bits)).sum(-1)
        apply(logits, _int32(flipped), rows)
    with mock.patch.object(kernel, "apply", planted):
        yield


@contextlib.contextmanager
def stop_tokens_masked() -> Iterator[None]:
    """A completed constraint's mask omits the stop tokens (xgrammar's fill, planted)."""
    import xgrammar

    fill = xgrammar.GrammarMatcher.fill_next_token_bitmask

    def planted(self, bitmask, index=0, **kwargs):
        out = fill(self, bitmask, index, **kwargs)
        if self.is_completed():
            for t in self.stop_token_ids:
                bitmask[index, t >> 5] &= int(_int32(torch.tensor(~(1 << (t & 31)) & 0xFFFFFFFF)))
        return out
    with mock.patch.object(xgrammar.GrammarMatcher, "fill_next_token_bitmask", planted):
        yield


@contextlib.contextmanager
def closing_tokens_masked() -> Iterator[None]:
    """grammar.xgrammar never allows a token holding `}` or `]` (no JSON can close)."""
    from holon_plugins.grammar.xgrammar import plugin

    fill, closing = plugin.XGrammar.fill, {}

    def planted(self, rows, buf):
        fill(self, rows, buf)
        if id(self) not in closing:
            ids = torch.tensor([i for i, t in enumerate(self.info.decoded_vocab) if b"}" in t or b"]" in t])
            words = torch.zeros(buf.shape[1], dtype=torch.int64).index_add_(0, ids >> 5, 1 << (ids & 31))
            closing[id(self)] = _int32(~words & 0xFFFFFFFF).numpy()
        buf &= closing[id(self)]
    with mock.patch.object(plugin.XGrammar, "fill", planted):
        yield


def _store_tap(tap: Callable) -> contextlib.AbstractContextManager:
    """`tap(store, step, record)` on every tiered store built inside the context (as a route tap)."""
    from holon_std.tiered import store as tiered

    init = tiered.TieredStore.__init__

    def planted(self, ctx, **kw):
        init(self, ctx, **kw)
        self.taps.append(functools.partial(tap, self))
    return mock.patch.object(tiered.TieredStore, "__init__", planted)


@contextlib.contextmanager
def resolve_off_by_one() -> Iterator[None]:
    """Every staged miss reads the next staging slot (the last one the first): a miss maps to the neighbouring slot.
    Reports the resolves it wrapped (captured ones fire on every replay) and the steps with misses."""
    from holon_std.tiered.resolver import DeviceResolver

    call, fired = DeviceResolver.__call__, collections.Counter()

    def planted(self, ids, first):
        slots = call(self, ids, first)
        k, staging = self.store.top_k, self.store.staging
        fired[_when()] += 1
        return torch.where((slots >= k) & (slots < k + staging), k + (slots - k + 1) % staging, slots)

    def missed(store, step, record):
        fired["steps with misses"] += bool((record["routed"] & ~(store.residency.host >= 0).all(-1)).any())
    with mock.patch.object(DeviceResolver, "__call__", planted), _store_tap(missed):
        yield
    print(f"plant resolve-off-by-one fired: {dict(fired)}", file=sys.stderr)


@contextlib.contextmanager
def map_before_landing(most: float = 2e-3) -> Iterator[None]:
    """A two-phase swap's copy keeps neither its ordering after the unmap step nor the map step's wait: it starts on
    the copy stream at once (while the steps launched before it may still read the victim's slots), after an unseeded
    delay of 0–`most` seconds, and the map step does not wait for it. Reports the copies."""
    import random

    from holon.core import memory
    from holon_std.tiered import store as tiered

    copy, run = tiered.SlotClass.copy, memory.Copies._run
    fired, rng = collections.Counter(), random.SystemRandom()
    cycles = int(most * torch.cuda.get_device_properties(torch.cuda.current_device()).clock_rate * 1e3) \
        if torch.cuda.is_available() else 0

    def planted_copy(self, unit, slot, after, lands_by):
        fired["copies"] += 1
        return dataclasses.replace(copy(self, unit, slot, after, lands_by), after=None)

    def planted_run(self, ops, after):
        self.stream.run(lambda: torch.cuda._sleep(rng.randrange(cycles)) if cycles else None)
        return run(self, ops, after)
    unwaited = mock.patch.object(memory.Copies, "launching", lambda self, step: self.landing.pop(step, None))
    with mock.patch.object(tiered.SlotClass, "copy", planted_copy), unwaited, \
            mock.patch.object(memory.Copies, "_run", planted_run):
        yield
    print(f"plant map-before-landing fired: {dict(fired)}", file=sys.stderr)


@contextlib.contextmanager
def map_stale_slot() -> Iterator[None]:
    """A two-phase swap maps each incoming item to the slot the previous swap of its role and class freed (the first
    to the last one's): a deterministic map to a slot already reassigned. Reports the rotated maps."""
    from holon_std.tiered import store as tiered

    swap, fired = tiered.TieredStore._swap, collections.Counter()

    def planted(self, commit, swaps):
        edit = self.residency.edit

        def rotated(commit, entries, later=0):
            if later:
                groups = self.layers
                by: dict = {}
                for n, (g, _, r, _, _) in enumerate(entries):
                    by.setdefault((r, self.groups[groups[g]][r].name), []).append(n)
                entries = list(entries)
                for ns in (ns for ns in by.values() if len(ns) > 1):
                    slots = [entries[n][4] for n in ns]
                    for n, slot in zip(ns, slots[-1:] + slots[:-1], strict=True):
                        entries[n] = (*entries[n][:4], slot)
                    fired["maps"] += len(ns)
            return edit(commit, entries, later)
        self.residency.edit = rotated
        try:
            return swap(self, commit, swaps)
        finally:
            del self.residency.edit
    with mock.patch.object(tiered.TieredStore, "_swap", planted):
        yield
    print(f"plant map-stale-slot fired: {dict(fired)}", file=sys.stderr)


@contextlib.contextmanager
def owner_race(checked: bool = True) -> Iterator[None]:
    """At every commit of a step with decode rows, a host lending edit in F11's device-owned cache: the coldest
    resident of the layer with the most admissions that step (by the device's scores; the store's view of it, one
    step behind the device at the edit's launch) is unmapped at launch c + 2 and its slots filled with another
    piece's bytes (a ring would), copied after that step. The edit's precondition fails, and stops the engine, once
    the device has moved that expert in between; `checked=False` removes the check (the edit applies regardless).
    Reports the edits issued."""
    from holon_std.tiered import residency as maps
    from holon_std.tiered.store import LAND

    fired, apply = collections.Counter(), maps.ResidencyMap.apply

    def lend(store, step, record):
        records = record["records"]
        if not len(records):
            return
        g = int(collections.Counter(records[:, 0].tolist()).most_common(1)[0][0])
        view, groups = store.residency.host, store.layers
        whole = (view[g] >= 0).all(-1)
        if not whole.any():                                  # every expert of the layer lent or gone: none to take
            return
        score = store.state["score"][g].float().cpu().numpy()
        x = int(min(whole.nonzero()[0], key=lambda e: (score[e], e)))
        store.residency.edit(step.step, [(g, x, r, int(view[g, x, r]), -1) for r in range(store.roles)])
        copies = store._copies.setdefault(step.step + maps.LAG, [])
        for r, c in store.groups[groups[g]].items():
            copies.append(c.copy(0, int(view[g, x, r]), step.step + maps.LAG, step.step + maps.LAG + LAND))
        fired["edits"] += 1

    def unchecked(self, errors):
        index, _, value = self._list
        self._flat.scatter_(0, index, value.int())
        self._clear()
    with _store_tap(lend), mock.patch.object(maps.ResidencyMap, "apply", apply if checked else unchecked):
        yield
    print(f"plant owner-race{'' if checked else '-unchecked'} fired: {dict(fired)}", file=sys.stderr)


@contextlib.contextmanager
def _lfu_kernel(bit: int, name: str, fires: Callable) -> Iterator[None]:
    """kernels.expert_io's resolve with plant bit `bit` set (the variants read it); `fires(records, counts)` counts a
    step's firings from its delta log, at its commit before the store checks the step's error words."""
    from holon_plugins.kernels.expert_io import plugin
    from holon_std.tiered import store as tiered

    fired, commit = collections.Counter(), tiered.TieredStore.after_commit

    def counted(self, step, telemetry):
        delta, n = telemetry["delta"].numpy(), telemetry["n_delta"].numpy()
        records = np.concatenate([delta[g, :n[g]] for g in range(len(n))])
        fired["admissions"] += len(records)
        fired["fired"] += fires(records, telemetry["counts"].numpy())
        return commit(self, step, telemetry)
    with mock.patch.object(plugin.MoeResolve, "plant", bit), mock.patch.object(tiered.TieredStore, "after_commit",
                                                                               counted):
        yield
    print(f"plant {name} fired: {dict(fired)}", file=sys.stderr)


@contextlib.contextmanager
def lent_slot_still_mapped() -> Iterator[None]:
    """A prefill step's lending (D4) logs its experts unmapped but leaves them mapped: their slots serve the step's
    ring or staging and come back as empty slots while the experts still name them. Reports the experts lent."""
    from holon_std.tiered import store as tiered

    lend, fired = tiered.TieredStore.lend, collections.Counter()

    def planted(self, *args):
        before = self.residency.device.clone()
        picked = lend(self, *args)
        self.residency.device.copy_(before)                     # the unmap undone
        fired["lent"] += picked.sum()                          # a tensor: the step goes on without a host sync
        return picked
    with mock.patch.object(tiered.TieredStore, "lend", planted):
        yield
    print(f"plant lent-slot-still-mapped fired: { {k: int(v) for k, v in fired.items()} }", file=sys.stderr)


def _shared(records, counts) -> int:
    """Admissions into a victim slot an earlier admission of the step took."""
    seen, n = set(), 0
    for g, _, b in records.tolist():
        n += (g, b) in seen
        seen.add((g, b))
    return n


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
    # Quantized HF checkpoints (docs/m06-lowp.md §4.8)
    **hf_quant_plants(18), **fp4_plants(18), **fp8_plants(18),
    # MoE Tier B
    "ggml-experts-wrong-type": ggml_experts_wrong_type, "ggml-experts-stale-q8_1": ggml_experts_stale_q8_1,
    "rare-ggml-experts-stale-q8_1": lambda: ggml_experts_stale_q8_1(every=9973),
    **QWEN35_PLANTS,
    # The fused q/k norm + rope + KV write (docs/m07-perf.md D3)
    "qk-rope-kv-page-slot": qk_rope_kv_page_slot, "qk-rope-kv-decode-position": qk_rope_kv_decode_position,
    # Expert offload with adaptive placement (docs/v1-offload.md §6; spikes/x3_policies)
    "resolve-off-by-one": resolve_off_by_one, "map-before-landing": map_before_landing,
    "map-stale-slot": map_stale_slot, "owner-race": owner_race,
    "owner-race-unchecked": lambda: owner_race(checked=False),
    "lfu-victim-needed": lambda: _lfu_kernel(1, "lfu-victim-needed",           # victims the step's decode rows route
                                             lambda r, c: int(sum(c[g, b] > 0 for g, _, b in r.tolist()))),
    "lfu-shared-victim": lambda: _lfu_kernel(2, "lfu-shared-victim", _shared),
    "lent-slot-still-mapped": lent_slot_still_mapped,
    # Speculative decoding on a recurrent mixer (bounded --spec-vs-plain; docs/m04-spec.md §5)
    "replay-off-by-one": replay_off_by_one, "verify-writes-state": verify_writes_state,
    # Speculative verification (L0-dist, plugins/tests/test_rejection.py; the n_draft cap, test_spec.py)
    "unmasked-padding": unmasked_padding,
    # Draft heads (L0-drafter, `holon_judge drafter`)
    "dflash-tap-offset": dflash_tap_offset,
    # Logits processors (docs/m05-grammar.md §5): the graphs-vs-eager toy proof (plugins/tests/test_ngram.py), and
    # grammar.xgrammar's (plugins/tests/test_grammar.py)
    "mask-rows-not-reset": mask_rows_not_reset, "mask-column-shift": mask_column_shift,
    "no-rollback-after-probe": no_rollback_after_probe, "mask-bit-order": mask_bit_order,
    "stop-tokens-masked": stop_tokens_masked, "closing-tokens-masked": closing_tokens_masked}
