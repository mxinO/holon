"""The engine: builds a model with its state, and runs steps (design §7).

v0 runs depth 0 (commit before the next schedule) and eager; the overlap loop, device SeqTable and CUDA graphs are
added on the same structure. One step: schedule (reserving state) → step context with planner metadata → forward →
sample → commit.

Startup: resolve the weight source and the model plugin for its architecture → bind each mixer group's state specs
to ledger pools and state kinds → build and load the model → size the pools: either a configured byte budget, or
`memory_fraction` of the memory left after weights minus the peak of a worst-case step measured on minimum pools.
"""
from __future__ import annotations

import collections
import dataclasses
import gc
import itertools
import math
import secrets
from typing import Any

import numpy as np
import torch

from holon.api import (
    Batch,
    BatchPlan,
    DeviceBatch,
    Logprobs,
    Paged,
    Ref,
    SampleOut,
    SamplingParams,
    Seq,
    SeqTable,
    StateSpec,
    Status,
    TokenDelta,
    TokenSrc,
    canonical_hash,
)

from . import platform
from .batching import host_plan, padded
from .build import BuildContext, StepContext, build_model, load_weights
from .inputs import Uploader
from .memory import Ledger
from .plugins import PluginError, PluginSet, discover
from .prefix import PrefixIndex
from .sched import Scheduler
from .state import BoundSpec, StateManager

PROMPT_LOGPROB_CHUNK = 256     # rows of full-vocabulary logits materialized at once for prompt logprobs
_SEED_MASK = 2**63 - 1


def _splitmix64(x: int) -> int:
    """A fixed 64-bit mixer (sibling seeds must not depend on the interpreter's hash function)."""
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    x = (x ^ (x >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    return x ^ (x >> 31)


@dataclasses.dataclass
class EngineConfig:
    model: str                               # checkpoint directory
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    source: str = "hf_safetensors"           # weight source kind
    page_size: int = 16
    max_seqs: int = 64
    max_model_len: int = 8192                # prompt + generated tokens per request; pools always hold one
    max_step_tokens: int = 4096
    memory_fraction: float = 0.9             # of the memory left after weights, for activations and state
    state_memory: int | None = None          # bytes for state pools; None: measure (GPU only)
    prefix_cache: bool = True                # reuse cached prompt blocks across requests (paged state only)
    overlap: bool = True                     # schedule step N while step N-1 runs (depth 1); False: depth 0
    cuda_graphs: bool = True                 # replay decode steps from CUDA graphs, per batch-size bucket
    max_draft_tokens: int = 0                # speculative decoding: drafts per verify row at most (0: off)
    kernels: dict[str, str] = dataclasses.field(default_factory=dict)   # op name -> pinned kernel provider

    def __post_init__(self) -> None:
        for name in ("page_size", "max_seqs", "max_model_len", "max_step_tokens"):
            if getattr(self, name) < 1:
                raise ValueError(f"EngineConfig.{name} must be >= 1")
        if self.max_draft_tokens < 0:
            raise ValueError("EngineConfig.max_draft_tokens must be >= 0")


# Sampler variants captured at startup: greedy, sampled, sampled with filters (logprob steps run eagerly).
_VARIANTS = (SamplingParams(temperature=0.0, seed=0), SamplingParams(temperature=1.0, seed=0),
             SamplingParams(temperature=1.0, top_p=0.9, seed=0))


def _last_tokens(qstart: np.ndarray, n_logits: np.ndarray) -> np.ndarray:
    """Flat indices of each row's last `n_logits` tokens (the tokens whose logits a step computes)."""
    ends = np.repeat(qstart[1:], n_logits)
    first = np.repeat(np.cumsum(n_logits) - n_logits, n_logits)          # where each row's run starts in `ends`
    return ends - n_logits.repeat(n_logits) + (np.arange(len(ends)) - first)


@dataclasses.dataclass
class _Prep:
    """Static device inputs of a step's device prep (`Engine._prepare`)."""

    row_of_token: torch.Tensor     # [T] batch row of each token
    offset: torch.Tensor           # [T] index of each token within its row
    n_query: torch.Tensor          # [B]
    exact: torch.Tensor            # [B] the host knows this row's start exactly
    start: torch.Tensor            # [B] that start (ignored where not exact)
    from_device: torch.Tensor      # [B] the row's first input token is the device's last token
    lens: torch.Tensor             # [B] out: tokens held after the step (shared with every page group's Pages)
    draft_index: torch.Tensor | None = None   # [R*k] verify rows: input positions of the drafts
    draft_from: torch.Tensor | None = None    # [R*k] where each draft is in the SeqTable (flat)


@dataclasses.dataclass
class _Inflight:
    """A launched step: per-row snapshots (seq, tokens computed, sampled) taken at launch, the sampled tokens'
    pinned host copy and the event that says it has landed."""

    rows: list[tuple[Seq, int, int]]          # (seq, tokens computed, logits rows: 0 | 1 | 1 + k)
    out: SampleOut | None
    tokens: torch.Tensor | None
    accepts: torch.Tensor | None             # verify rows' accepted draft counts (pinned host copy)
    done: platform.Event
    sync: bool                               # commit at once (its logprobs need the host)


def resolve_model(plugins: PluginSet, cfg: EngineConfig) -> tuple[Any, Any]:
    """The checkpoint's weight source and the model plugin that provides its architecture."""
    plugins.load(plugins.owner(f"weight_source.{cfg.source}")[0])
    source = plugins.regs.weight_sources.get(cfg.source)(cfg.model)
    return source, plugins.load(plugins.owner(f"arch.{source.arch}")[0])


class _PagesPlanner:
    """Default planner: kernels read the core's `Pages` directly."""

    def plan_bounds(self, batch: Batch, pages: Any) -> Any:
        return pages

    def prepare_device(self, meta: Any, ctx: Any) -> None:
        pass


class Engine:
    def __init__(self, cfg: EngineConfig, plugins: PluginSet | None = None) -> None:
        self.cfg, self.device = cfg, torch.device(cfg.device)
        platform.configure_allocator(self.device)
        self.plugins = p = plugins or PluginSet(discover())
        source, model_plugin = resolve_model(p, cfg)
        self.spec = model_plugin.recipe(source.config(), source.kind)
        self.eos = source.eos_token_ids

        self.ledger = Ledger(self.device)
        options = {"page_size": cfg.page_size, "max_seqs": cfg.max_seqs, "max_model_len": cfg.max_model_len,
                   "kernels": cfg.kernels}
        self.ctx = BuildContext(p, self.device, cfg.dtype, options)
        bound, self.planners = self._bind_state()
        self.model = build_model(self.spec, self.ctx)
        load_weights(self.model, self.ctx, source.tensors())

        # One extra slot (max_seqs) is the padding slot of graph-captured steps.
        self.sampler = sampler = p.load(p.single("sampler")).sampler(cfg.max_seqs + 1, self.device)
        self.state = StateManager(bound, cfg.max_seqs, cfg.max_model_len, self.device,
                                  on_admit=lambda seq: sampler.admit(seq.slot, seq.params), index=self._index(bound))
        # Speculative decoding (design §7.6): a draft policy picks k per request, one drafter proposes, the verifier
        # checks; all three are plugins, used only when drafts are enabled and a policy is installed.
        self.policy = self.drafter = self.verifier = None
        if cfg.max_draft_tokens and any(True for _ in p.by_slot("draft_policy")):
            self.policy, self.drafter = p.load(p.single("draft_policy")), p.load(p.single("drafter"))
            self.verifier = p.load(p.single("verifier"))
        host_lens = any(getattr(planner, "needs_host_lens", False) for planner in self.planners.values())
        self.sched = Scheduler(self.state, cfg.max_step_tokens, cfg.max_model_len,
                               draft_k=self._draft_k if self.policy is not None else None, host_lens_only=host_lens)
        seqs = cfg.max_seqs
        tokens = max(cfg.max_step_tokens, seqs * (1 + cfg.max_draft_tokens))     # a verify step: 1 + k per row
        self.uploader = Uploader(self.device, self.state.upload_fields(seqs, tokens) | {
            "input_ids": (tokens, torch.int64), "positions": (tokens, torch.int64),
            "query_start": (seqs + 1, torch.int64), "slots": (seqs, torch.int64),
            "logits_index": (tokens, torch.int64), "logits_rows": (tokens, torch.int64),   # verify rows: 1 + k each
            "row_of_token": (tokens, torch.int64), "offset": (tokens, torch.int64), "n_query": (seqs, torch.int64),
            "exact": (seqs, torch.bool), "start": (seqs, torch.int64), "from_device": (seqs, torch.bool),
            "draft_index": (tokens, torch.int64), "draft_from": (tokens, torch.int64)})
        # The device SeqTable (one row per slot, plus the padding slot): authoritative while steps are in flight.
        self.last_token = torch.zeros(seqs + 1, dtype=torch.int64, device=self.device)
        self.num_computed = torch.zeros(seqs + 1, dtype=torch.int64, device=self.device)
        self.seq_table = SeqTable(self.num_computed, self.last_token,
                                  n_accept=torch.zeros(seqs + 1, dtype=torch.int64, device=self.device),
                                  draft_tokens=torch.zeros(seqs + 1, max(cfg.max_draft_tokens, 1), dtype=torch.int64,
                                                           device=self.device))
        self.graphs = platform.Graphs(self.device) if cfg.cuda_graphs and self.device.type == "cuda" else None
        self._buckets = sorted({min(b, seqs) for b in (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256)} | {seqs})
        width = 1 + cfg.max_draft_tokens
        self._out_tokens = [platform.pinned_empty(seqs * width, torch.int64, self.device) for _ in range(2)]
        self._out_accepts = [platform.pinned_empty(seqs, torch.int64, self.device) for _ in range(2)]
        self._out_turn = 0
        self._inflight: _Inflight | None = None
        self.ledger.materialize(cfg.state_memory if cfg.state_memory is not None else self._measure_state_memory())
        self.state.reset_padding()
        for pool in self.ledger.pools.values():          # fresh segments, so they show the allocator setting
            platform.check_expandable(pool.buffer)
        if self.graphs is not None:          # captured graphs bake in pool addresses: after the final materialize
            self._capture_graphs()
        self._ids = itertools.count()
        self._prompt_lp: dict[Seq, Logprobs] = {}         # sequences still accumulating prompt logprobs
        self._siblings: dict[Seq, list[Seq]] = {}         # n > 1: waiting for the first sequence's prefill
        self._live: dict[str, list[Seq]] = {}              # request id -> its unfinished sequences

    def _draft_k(self, seq: Seq) -> int:
        """Drafts the policy wants for `seq` this step. Only greedy requests without logprobs are drafted: the
        verifier is greedy-exact and logprob steps run synchronously."""
        if self.policy is None or seq.params.temperature != 0 or seq.params.logprobs is not None:
            return 0
        return min(self.policy.choose(seq)[1], self.cfg.max_draft_tokens)

    def _index(self, bound: list[BoundSpec]) -> PrefixIndex | None:
        """A prefix index over blocks every paged spec can share whole pages of; None when disabled or when some
        state is not paged (checkpointed recurrent state joins in M0.3)."""
        if not self.cfg.prefix_cache or not bound or not all(isinstance(b.spec.shape, Paged) for b in bound):
            return None
        block = math.lcm(*(b.spec.shape.tokens_per_unit for b in bound), *(b.spec.alignment for b in bound))
        # Model + recipe identity. Enough within one process; weight updates (v1) and persistent tiers must add the
        # checkpoint identity and weight version (design §6.3).
        namespace = int(canonical_hash(self.spec).removeprefix("sha256:")[:16], 16)
        return PrefixIndex(namespace, block, {b.spec.name: b.pool for b in bound})

    def _bind_state(self) -> tuple[list[BoundSpec], dict[str, Any]]:
        """Each mixer group's state specs → pools and bound state kinds; one planner per state group."""
        by_mixer: dict[Ref, list[int]] = collections.defaultdict(list)
        for i, layer in enumerate(self.spec.layers):
            by_mixer[layer.mixer].append(i)
        bound, planners = [], {}
        for ref, layers in by_mixer.items():
            cls = self.ctx.unit_factory(ref)
            specs = cls.state_specs(dict(ref.params), tuple(layers), self.ctx) if hasattr(cls, "state_specs") else []
            bound += [self._bind(spec) for spec in specs]
            for group in dict.fromkeys(s.group for s in specs):
                if group in planners:
                    raise PluginError(f"state group {group!r} declared by two mixers")
                members = [s for s in specs if s.group == group]
                planners[group] = (cls.planner(dict(ref.params), members, self.ctx) if hasattr(cls, "planner")
                                   else _PagesPlanner())
        return bound, planners

    def _bind(self, spec: StateSpec) -> BoundSpec:
        cfg, shape = self.cfg, spec.shape
        if isinstance(shape, Paged):
            sizing = {"min_units": -(-cfg.max_model_len // shape.tokens_per_unit),
                      "tokens_per_unit": shape.tokens_per_unit}
        else:
            n = cfg.max_seqs * (1 + shape.draft_slots)
            sizing = {"min_units": n, "pref_units": n}
        pool = self.ledger.request(f"state.{spec.name}", unit_bytes=shape.unit_bytes, segments=spec.segments,
                                   null_unit=isinstance(shape, Paged),
                                   **sizing)
        kind = Ref(spec.kind)
        self.plugins.load(self.plugins.provider("state_kind", kind))
        self.ctx.bind_state(spec, pool)
        return BoundSpec(spec, pool, self.plugins.regs.state_kinds.get(kind.name)(spec, pool, self.ctx))

    def _measure_state_memory(self) -> int:
        """`memory_fraction` of the free memory, minus the activation peak of a worst-case step: `max_step_tokens`
        prompt tokens in rows of at most `max_model_len`, each sampled, run on minimum-size pools."""
        cfg = self.cfg
        rows = [cfg.max_model_len] * (cfg.max_step_tokens // cfg.max_model_len)
        rows = (rows + [cfg.max_step_tokens % cfg.max_model_len])[:cfg.max_seqs]
        n = np.array([r for r in rows if r], dtype=np.int32)
        plan = host_plan(0, np.arange(len(n), dtype=np.int32), np.zeros_like(n), n, np.ones_like(n),
                         np.zeros(int(n.sum()), dtype=np.int64))
        self.ledger.materialize(None)
        self.state.reset_padding()
        with torch.no_grad():
            peak = platform.peak_memory(self.device, lambda: self._worst_step(plan))
            if self.graphs is not None:
                # The graphs' private pool comes on top. Captured largest bucket first, the smaller ones reuse its
                # blocks, so the pool is what the largest bucket's variants leave reserved.
                graphs, self.graphs = self.graphs, platform.Graphs(self.device)
                peak += platform.reserved_growth(self.device, lambda: [
                    self._graph_step(self._buckets[-1], params) for params in _VARIANTS])
                self.graphs = graphs
        self.ledger.free()
        return int(platform.free_memory(self.device) * cfg.memory_fraction) - peak

    def _worst_step(self, plan: BatchPlan) -> None:
        """The largest prefill, a prompt-logprob chunk, a filtered sampling step over every slot with logprobs, and
        (when drafting) a verify step's logits over every slot's 1 + k rows."""
        ctx = self._context(plan)
        self._prepare(ctx)
        h = self.model.trunk(ctx)
        self.model.head(h[:PROMPT_LOGPROB_CHUNK]).float().log_softmax(-1)
        del h
        n = self.cfg.max_seqs
        for slot in range(n):
            self.sampler.admit(slot, SamplingParams(temperature=1.0, top_p=0.9, seed=0, logprobs=20))
        ones = np.ones(n, dtype=np.int32)
        decode = host_plan(0, np.arange(n, dtype=np.int32), np.zeros(n, dtype=np.int32), ones, ones,
                           np.zeros(n, dtype=np.int64))
        ctx = self._context(decode)
        self._prepare(ctx)
        self.sampler.sample(torch.zeros(n, self.spec.vocab, device=self.device), ctx)
        if self.cfg.max_draft_tokens:
            rows = n * (1 + self.cfg.max_draft_tokens)
            torch.zeros(rows, self.spec.vocab, device=self.device).argmax(-1)

    def _capture_graphs(self) -> None:
        """Capture every decode bucket for each sampler variant at startup, so serving never stops to capture."""
        for rows in reversed(self._buckets):         # largest first: smaller graphs reuse its pool blocks
            for params in _VARIANTS:
                if not self._graph_step(rows, params):
                    self.graphs = None              # some planner's kernel cannot be captured: stay eager
                    return
        self.sampler.admit(self.state.padding_slot, _VARIANTS[0])

    @torch.no_grad()
    def _graph_step(self, rows: int, params: SamplingParams) -> bool:
        """Warm up and capture one decode graph with every row on the padding slot (null unit, `params`)."""
        self.sampler.admit(self.state.padding_slot, params)
        empty = np.zeros(0, dtype=np.int32)
        plan = padded(host_plan(0, empty, empty, empty, empty, np.zeros(0, dtype=np.int64)), rows,
                      self.state.padding_slot)
        ctx = self._context(plan)
        keys = [getattr(meta, "graph_key", None) for meta in ctx.metas.values()]
        if None in keys:
            return False
        key = (rows, self.sampler.variant(plan.slots), *keys)
        for _ in range(2):                          # warm-up, then capture
            self.graphs.run(key, lambda: self._decode(ctx))
        return True

    # -- requests -------------------------------------------------------------------------------------------------
    def add(self, token_ids: list[int], params: SamplingParams | None = None, rid: str | None = None) -> str:
        """Queue a request; its `params.n` sequences stream as TokenDeltas with `index` 0 … n-1. `rid` defaults to a
        fresh id; callers that route results themselves (frontends) choose it."""
        p = params or SamplingParams()
        if not token_ids:
            raise ValueError("empty prompt")
        if len(token_ids) >= self.cfg.max_model_len:
            raise ValueError(f"prompt of {len(token_ids)} tokens leaves no room under max_model_len "
                             f"{self.cfg.max_model_len}")
        if p.n < 1:
            raise ValueError(f"n must be >= 1, got {p.n}")
        if p.presence_penalty or p.frequency_penalty or p.repetition_penalty != 1:
            raise NotImplementedError("sampling penalties are not supported yet")
        rid = rid if rid is not None else str(next(self._ids))
        if rid in self._live:
            raise ValueError(f"request id {rid!r} is already in flight")
        base = (p.seed if p.seed is not None else secrets.randbits(63)) & _SEED_MASK
        for i in range(p.n):
            # A concrete seed per sequence keeps sampling reproducible across batching and preemption.
            seed = base if i == 0 else _splitmix64(base + i) & _SEED_MASK
            child = dataclasses.replace(p, seed=seed, prompt_logprobs=p.prompt_logprobs if i == 0 else None)
            seq = Seq(rid=rid, tokens=list(token_ids), params=child, index=i)
            self._live.setdefault(rid, []).append(seq)       # siblings too: abort must account for them
            if child.prompt_logprobs is not None:
                self._prompt_lp[seq] = Logprobs([], [], [])
            if i == 0 or self.state.index is None:
                self.sched.add(seq)
                first = seq
            else:   # siblings wait until the first sequence's prompt is cached, then adopt it
                self._siblings.setdefault(first, []).append(seq)
        return rid

    def has_work(self) -> bool:
        return bool(self.sched.waiting or self.sched.running or self._inflight)

    # -- control (design §7.7) -------------------------------------------------------------------------------------
    def abort(self, rid: str, reason: str = "abort", error: str | None = None) -> list[TokenDelta]:
        """Finish every sequence of request `rid` now — one delta each, siblings that never started included; a row
        already in flight is discarded at its commit."""
        deltas = []
        for seq in self._live.pop(rid, []):
            for sibling in self._siblings.pop(seq, []):             # never started: nothing to release
                sibling.status = Status.FINISHED
                deltas.append(TokenDelta(rid, [], index=sibling.index, finish_reason=reason, error=error))
            if seq.status is Status.FINISHED:
                continue
            self.sched.finish(seq, reason)
            self._prompt_lp.pop(seq, None)
            deltas.append(TokenDelta(rid, [], index=seq.index, finish_reason=reason, error=error))
        return deltas

    def fail_all(self, error: str) -> list[TokenDelta]:
        """An error delta for every live request (the engine is going down)."""
        return [d for rid in list(self._live) for d in self.abort(rid, "error", error)]

    def drain(self) -> list[TokenDelta]:
        """Commit the step in flight (the quiesce point: nothing runs on the device afterwards)."""
        prev, self._inflight = self._inflight, None
        return self._commit(prev) if prev is not None else []

    def flush_prefix(self) -> list[TokenDelta]:
        """Drop the prefix cache (e.g. after a weight update); requests keep the blocks they hold."""
        deltas = self.drain()
        if self.state.index is not None:
            self.state.index.clear()
        return deltas

    def close(self) -> None:
        """Release the device memory now (graphs, state pools, weights) instead of whenever the garbage collector
        breaks the engine's reference cycles; the engine is unusable afterwards."""
        self.drain()
        self.graphs = self.model = self._inflight = None
        self.ledger.free()
        gc.collect()
        platform.release_cache(self.device)

    def stats(self) -> dict[str, int]:
        return {"waiting": len(self.sched.waiting), "running": len(self.sched.running),
                "preemptions": self.sched.preemptions, "prefix_tokens_saved": self.state.tokens_saved,
                "cached_blocks": len(self.state.index.entries) if self.state.index is not None else 0,
                "graphs": len(self.graphs) if self.graphs is not None else 0}

    def generate(self, prompts: list[list[int]], params: SamplingParams | None = None) -> list[list[int]]:
        """Token ids generated for each prompt (one sequence per prompt)."""
        if params is not None and params.n != 1:
            raise ValueError("generate() returns one sequence per prompt; step() streams n > 1")
        rids = [self.add(p, params) for p in prompts]
        out: dict[str, list[int]] = {r: [] for r in rids}
        pending = set(rids)
        while pending:
            for d in self.step():
                if d.rid in out:                     # other requests may be in flight too
                    out[d.rid].extend(d.token_ids)
                    if d.finish_reason:
                        pending.discard(d.rid)
        return [out[r] for r in rids]

    # -- one step -------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def step(self) -> list[TokenDelta]:
        """Schedule and launch one step; commit the previous one while this one runs (overlap), or this one at
        once (depth 0, and steps that report logprobs, whose extraction needs the host)."""
        prev, self._inflight = self._inflight, None
        deltas: list[TokenDelta] = []
        scheduled = self.sched.schedule()
        if scheduled is None and prev is not None:      # what is left waits on the in-flight step's tokens
            deltas, prev = self._commit(prev), None
            scheduled = self.sched.schedule()
        if scheduled is None:
            if self.has_work():
                raise RuntimeError("the scheduler made no progress with requests pending")
            return deltas
        step = self._launch(*scheduled)
        if prev is not None:
            deltas += self._commit(prev)
        if self.cfg.overlap and not step.sync:
            self._inflight = step
        else:
            deltas += self._commit(step)
        return deltas

    def _launch(self, plan: BatchPlan, seqs: list[Seq]) -> _Inflight:
        """Enqueue the step's device work without waiting for the device, then advance the host mirrors."""
        rows = [(seq, int(n), int(m)) for seq, n, m in zip(seqs, plan.n_query, plan.n_logits, strict=True)]
        sync = any(seq in self._prompt_lp or (m and seq.params.logprobs is not None) for seq, _, m in rows)
        decode = bool((plan.n_query == 1).all())
        exact = np.array([seq.inexact == 0 for seq in seqs], dtype=bool)
        graphed = self.graphs is not None and decode and not sync and len(seqs) <= self._buckets[-1]   # all captured
        if graphed:
            plan = padded(plan, next(b for b in self._buckets if b >= len(seqs)), self.state.padding_slot)
            exact = np.concatenate([exact, np.ones(len(plan.slots) - len(seqs), dtype=bool)])   # padding: start 0
        ctx = self._context(plan, exact)
        out = sampled = accepts = None
        if graphed:
            keys = [meta.graph_key for meta in ctx.metas.values()]
            key = (len(plan.slots), self.sampler.variant(plan.slots), *keys)
            sampled = self.graphs.run(key, lambda: self._decode(ctx))
        else:
            self._prepare(ctx)
            if any(seq in self._prompt_lp for seq in seqs):
                h = self.model.trunk(ctx)
                logits = self.model.head(h[ctx.batch.dev.logits_index])
                self._record_prompt_logprobs(h, plan, seqs)
            else:
                logits = self.model(ctx)            # always: chunks without sampled rows still write their state
            if (plan.n_logits > 1).any():           # verify rows
                out, accepts = self.verifier.verify(logits.float(), ctx)
                sampled = self._store(ctx, out.tokens, accepts)
            elif plan.n_logits.any():
                out = self.sampler.sample(logits.float(), ctx)
                sampled = self._store(ctx, out.tokens)
        if sampled is not None and self.drafter is not None:
            self.drafter.propose(ctx)               # drafts for the next step, from the advanced SeqTable
        tokens = None
        if sampled is not None:
            n_sampled = sum(m for _, _, m in rows)
            tokens = self._out_tokens[self._out_turn][:n_sampled]
            tokens.copy_(sampled[:n_sampled], non_blocking=True)
            if accepts is not None:
                accepts = self._out_accepts[self._out_turn][: len(accepts)].copy_(accepts, non_blocking=True)
            self._out_turn ^= 1
        done = platform.Event(self.device)
        done.record()
        running = Status.RUNNING
        for seq, n, m in rows:
            seq.num_computed += n                   # an upper bound for verify rows until their commit
            if m:
                seq.pending += m
                if m > 1:
                    seq.inexact += 1
                if seq.status is not running:
                    self.sched.finished_prefill(seq)
        return _Inflight(rows, out, tokens, accepts, done, sync)

    def _decode(self, ctx: StepContext) -> torch.Tensor:
        """A decode step's device work from inputs to sampled tokens — the part captured in a graph."""
        self._prepare(ctx)
        return self._store(ctx, self.sampler.sample(self.model(ctx).float(), ctx).tokens)

    def _store(self, ctx: StepContext, tokens: torch.Tensor, accepts: torch.Tensor | None = None) -> torch.Tensor:
        """Advance the device SeqTable for the sampled rows: the next input token and the computed length — for a
        verify row, only up to the drafts it accepted (the state its rejected drafts wrote is past that length)."""
        dev = ctx.batch.dev
        if accepts is None:
            slots = dev.slots[dev.logits_rows]
            self.last_token[slots] = tokens
            self.num_computed[slots] = ctx.prep.lens[dev.logits_rows].long()
            return tokens
        width = int(ctx.batch.plan.n_logits.max())
        rows = dev.logits_rows.view(-1, width)[:, 0]
        slots = dev.slots[rows]
        self.last_token[slots] = tokens.view(-1, width).gather(1, accepts[:, None])[:, 0]
        self.num_computed[slots] = ctx.prep.lens[rows].long() - (width - 1) + accepts
        self.seq_table.n_accept[slots] = accepts
        return tokens

    def _context(self, plan: BatchPlan, exact: np.ndarray | None = None) -> StepContext:
        """The host half of a step's inputs: index arrays and the starts the host knows exactly (`exact` rows; all by
        default), staged into static device buffers with asynchronous copies, plus host-side planning. Positions,
        lengths and write indices are derived on the device by `_prepare`."""
        n = plan.n_query
        rows, tokens = len(n), int(n.sum())
        qstart = np.concatenate([[0], np.cumsum(n)])
        up = self.uploader
        up.begin()
        batch = Batch(plan, DeviceBatch(
            input_ids=up.put("input_ids", plan.host_tokens),
            positions=up.dev["positions"][:tokens],
            query_start=up.put("query_start", qstart),
            slots=up.put("slots", plan.slots.astype(np.int64)),
            logits_index=up.put("logits_index", _last_tokens(qstart, plan.n_logits)),
            logits_rows=up.put("logits_rows", np.repeat(np.arange(rows), plan.n_logits)),
        ))
        prep = _Prep(row_of_token=up.put("row_of_token", np.repeat(np.arange(rows), n)),
                     offset=up.put("offset", np.arange(tokens) - np.repeat(qstart[:-1], n)),
                     n_query=up.put("n_query", n.astype(np.int64)),
                     exact=up.put("exact", np.ones(rows, dtype=bool) if exact is None else exact),
                     start=up.put("start", plan.start_upper.astype(np.int64)),
                     from_device=up.put("from_device", plan.token_src == TokenSrc.DEVICE_LAST),
                     lens=up.dev["pages.lens"][:rows])
        verify = np.flatnonzero(plan.n_logits > 1)
        if len(verify):                          # a verify row's inputs after the first are its drafts (on device)
            k = int(n[verify[0]]) - 1
            j = np.tile(np.arange(k), len(verify))
            prep.draft_index = up.put("draft_index", np.repeat(qstart[verify] + 1, k) + j)
            cols = self.seq_table.draft_tokens.shape[1]
            prep.draft_from = up.put("draft_from", np.repeat(plan.slots[verify].astype(np.int64) * cols, k) + j)
        pages = self.state.pages(plan, up)
        up.end()
        metas = {g: p.plan_bounds(batch, pages.get(g)) for g, p in self.planners.items()}
        return StepContext(batch, metas, seq_table=self.seq_table, pages=pages, prep=prep)

    def _prepare(self, ctx: StepContext) -> None:
        """The device half: the SeqTable's starts for host-exact rows, then positions, lengths, write indices and
        inputs still on the device — all into static buffers, so this runs inside a captured graph too."""
        dev, prep = ctx.batch.dev, ctx.prep
        self.num_computed[dev.slots] = torch.where(prep.exact, prep.start, self.num_computed[dev.slots])
        start = self.num_computed[dev.slots]
        dev.positions.copy_(start[prep.row_of_token] + prep.offset)
        prep.lens.copy_(start + prep.n_query)
        first = dev.query_start[:-1]
        dev.input_ids[first] = torch.where(prep.from_device, self.last_token[dev.slots], dev.input_ids[first])
        if prep.draft_index is not None:
            dev.input_ids[prep.draft_index] = self.seq_table.draft_tokens.view(-1)[prep.draft_from]
        slot_of_token = dev.slots[prep.row_of_token]
        for pages in ctx.pages.values():
            unit = pages.table[slot_of_token, dev.positions // pages.page].long()
            pages.write_index.copy_(unit * pages.page + dev.positions % pages.page)
        for group, planner in self.planners.items():
            planner.prepare_device(ctx.metas[group], ctx)

    def _record_prompt_logprobs(self, h: torch.Tensor, plan: BatchPlan, seqs: list[Seq],
                                chunk: int = PROMPT_LOGPROB_CHUNK) -> None:
        """Logprobs of the prompt tokens this step's rows predict, in row chunks (no full-vocab logits for long
        prompts); positions recomputed after a preemption are not recorded twice."""
        qstart = np.concatenate([[0], np.cumsum(plan.n_query)])
        for i, seq in enumerate(seqs):
            acc = self._prompt_lp.get(seq)
            if acc is None:
                continue
            start = int(plan.start_upper[i])
            lo, hi = max(start, len(acc.token)), min(start + int(plan.n_query[i]), seq.prompt_len - 1)
            k = max(seq.params.prompt_logprobs, 1)
            for a in range(lo, hi, chunk):
                b = min(a + chunk, hi)
                lp = self.model.head(h[qstart[i] + a - start: qstart[i] + b - start]).float().log_softmax(-1)
                target = torch.as_tensor(seq.tokens[a + 1: b + 1], device=lp.device)
                top = lp.topk(k)
                acc.token += lp.gather(-1, target[:, None])[:, 0].tolist()
                acc.top_ids += top.indices.tolist()
                acc.top += top.values.tolist()

    def _commit(self, step: _Inflight) -> list[TokenDelta]:
        """Host bookkeeping once a step's results are on the host: tokens, finishing, publishing, streaming."""
        step.done.synchronize()
        tokens = step.tokens.tolist() if step.tokens is not None else []
        accepts = iter(step.accepts.tolist() if step.accepts is not None else [])
        lps = None if step.out is None or step.out.logprobs is None else (
            step.out.logprobs.tolist(), step.out.top_ids.tolist(), step.out.top_logprobs.tolist())
        deltas, row = [], 0
        for seq, _, m in step.rows:
            if not m:
                if seq.status is not Status.FINISHED and self._completes_block(seq):
                    self.state.publish(seq)
                continue
            at = row
            if m > 1:                                # a verify row: the accepted drafts, then the target's token
                accepted = next(accepts)
                emitted = tokens[row: row + accepted + 1]
                seq.inexact -= 1
                if seq.status is Status.RUNNING:     # the host length is exact again (preempted ones restart at 0)
                    seq.num_computed -= m - 1 - accepted
            row += m
            seq.pending -= m
            if seq.status is Status.FINISHED:      # finished at an earlier commit; this row was extra
                continue
            if m == 1:                               # the common case, without the per-token loop
                out = [tokens[at]]
                seq.tokens.append(out[0])
                reason = self._finish_reason(seq, out[0])
            else:
                reason, out = None, []
                for token in emitted:
                    seq.tokens.append(token)
                    out.append(token)
                    if reason := self._finish_reason(seq, token):
                        break
            if self._completes_block(seq):
                self.state.publish(seq)
            if self._siblings:
                for sibling in reversed(self._siblings.pop(seq, [])):     # its prompt is cached now
                    self.sched.add(sibling, front=True)
            if reason:
                self.sched.finish(seq, reason)
                live = self._live.get(seq.rid, [])
                if seq in live:
                    live.remove(seq)
                if not live:
                    self._live.pop(seq.rid, None)
            k = seq.params.logprobs                  # logprob requests are never drafted: one token per row
            logprobs = None if k is None else Logprobs([lps[0][at]], [lps[1][at][:k]], [lps[2][at][:k]])
            deltas.append(TokenDelta(seq.rid, out, index=seq.index, logprobs=logprobs,
                                     prompt_logprobs=self._prompt_lp.pop(seq, None), finish_reason=reason))
        return deltas

    def _completes_block(self, seq: Seq) -> bool:
        """Whether `seq` has a computed block it has not published yet (a cheap check for the common no case)."""
        index, chain = self.state.index, self.state.chain.get(seq.slot)
        return (index is not None and chain is not None
                and min(seq.num_computed, len(seq.tokens)) // index.block > len(chain))

    def _finish_reason(self, seq: Seq, token: int) -> str | None:
        p = seq.params
        if (token in self.eos and not p.ignore_eos) or token in p.stop_token_ids:
            return "stop"
        generated = len(seq.tokens) - seq.prompt_len
        if (p.max_tokens is not None and generated >= p.max_tokens) or len(seq.tokens) >= self.cfg.max_model_len:
            return "length"
        return None
