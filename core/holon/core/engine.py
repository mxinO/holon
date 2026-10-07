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
    kernels: dict[str, str] = dataclasses.field(default_factory=dict)   # op name -> pinned kernel provider

    def __post_init__(self) -> None:
        for name in ("page_size", "max_seqs", "max_model_len", "max_step_tokens"):
            if getattr(self, name) < 1:
                raise ValueError(f"EngineConfig.{name} must be >= 1")


@dataclasses.dataclass
class _Inflight:
    """A launched step: per-row snapshots (seq, tokens computed, sampled) taken at launch, the sampled tokens'
    pinned host copy and the event that says it has landed."""

    rows: list[tuple[Seq, int, bool]]
    out: SampleOut | None
    tokens: torch.Tensor | None
    done: platform.Event
    sync: bool                               # commit at once (its logprobs need the host)


class _PagesPlanner:
    """Default planner: kernels read the core's `Pages` directly."""

    def plan_bounds(self, batch: Batch, pages: Any) -> Any:
        return pages

    def prepare_device(self, meta: Any, ctx: Any) -> None:
        pass


class Engine:
    def __init__(self, cfg: EngineConfig, plugins: PluginSet | None = None) -> None:
        self.cfg, self.device = cfg, torch.device(cfg.device)
        self.plugins = p = plugins or PluginSet(discover())
        p.load(p.owner(f"weight_source.{cfg.source}")[0])
        source = p.regs.weight_sources.get(cfg.source)(cfg.model)
        self.spec = p.load(p.owner(f"arch.{source.arch}")[0]).recipe(source.config(), source.kind)
        self.eos = source.eos_token_ids

        self.ledger = Ledger(self.device)
        self.ctx = BuildContext(p, self.device, cfg.dtype, {"page_size": cfg.page_size, "max_seqs": cfg.max_seqs,
                                                            "kernels": cfg.kernels})
        bound, self.planners = self._bind_state()
        self.model = build_model(self.spec, self.ctx)
        load_weights(self.model, self.ctx, source.tensors())

        # One extra slot (max_seqs) is the padding slot of graph-captured steps.
        self.sampler = sampler = p.load(p.single("sampler")).sampler(cfg.max_seqs + 1, self.device)
        self.state = StateManager(bound, cfg.max_seqs, cfg.max_model_len, self.device,
                                  on_admit=lambda seq: sampler.admit(seq.slot, seq.params), index=self._index(bound))
        self.sched = Scheduler(self.state, cfg.max_step_tokens, cfg.max_model_len)
        tokens, seqs = max(cfg.max_step_tokens, cfg.max_seqs), cfg.max_seqs
        self.uploader = Uploader(self.device, self.state.upload_fields(seqs, tokens) | {
            "input_ids": (tokens, torch.int64), "positions": (tokens, torch.int64),
            "query_start": (seqs + 1, torch.int64), "slots": (seqs, torch.int64), "logits_index": (seqs, torch.int64),
            "logits_rows": (seqs, torch.int64), "device_tokens": (seqs, torch.int64),
            "device_slots": (seqs, torch.int64)})
        self.last_token = torch.zeros(seqs + 1, dtype=torch.int64, device=self.device)  # device SeqTable column
        self.graphs = platform.Graphs(self.device) if cfg.cuda_graphs else None
        self._buckets = sorted({min(b, seqs) for b in (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256)} | {seqs})
        self._out_tokens = [platform.pinned_empty(seqs, torch.int64, self.device) for _ in range(2)]
        self._out_turn = 0
        self._inflight: _Inflight | None = None
        self.ledger.materialize(cfg.state_memory if cfg.state_memory is not None else self._measure_state_memory())
        self.state.reset_padding()
        self._ids = itertools.count()
        self._prompt_lp: dict[Seq, Logprobs] = {}         # sequences still accumulating prompt logprobs
        self._siblings: dict[Seq, list[Seq]] = {}         # n > 1: waiting for the first sequence's prefill

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
        with torch.no_grad():
            peak = platform.peak_memory(self.device, lambda: self._worst_step(plan))
        self.ledger.free()
        return int(platform.free_memory(self.device) * cfg.memory_fraction) - peak

    def _worst_step(self, plan: BatchPlan) -> None:
        """The largest prefill, a prompt-logprob chunk, and a filtered sampling step over every slot with logprobs."""
        h = self.model.trunk(self._context(plan))
        self.model.head(h[:PROMPT_LOGPROB_CHUNK]).float().log_softmax(-1)
        del h
        n = self.cfg.max_seqs
        for slot in range(n):
            self.sampler.admit(slot, SamplingParams(temperature=1.0, top_p=0.9, seed=0, logprobs=20))
        ones = np.ones(n, dtype=np.int32)
        decode = host_plan(0, np.arange(n, dtype=np.int32), np.zeros(n, dtype=np.int32), ones, ones,
                           np.zeros(n, dtype=np.int64))
        self.sampler.sample(torch.zeros(n, self.spec.vocab, device=self.device), self._context(decode))

    # -- requests -------------------------------------------------------------------------------------------------
    def add(self, token_ids: list[int], params: SamplingParams | None = None) -> str:
        """Queue a request; its `params.n` sequences stream as TokenDeltas with `index` 0 … n-1."""
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
        rid = str(next(self._ids))
        base = (p.seed if p.seed is not None else secrets.randbits(63)) & _SEED_MASK
        for i in range(p.n):
            # A concrete seed per sequence keeps sampling reproducible across batching and preemption.
            seed = base if i == 0 else _splitmix64(base + i) & _SEED_MASK
            child = dataclasses.replace(p, seed=seed, prompt_logprobs=p.prompt_logprobs if i == 0 else None)
            seq = Seq(rid=rid, tokens=list(token_ids), params=child, index=i)
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
        rows = [(seq, int(n), bool(k)) for seq, n, k in zip(seqs, plan.n_query, plan.n_logits, strict=True)]
        sync = any(seq in self._prompt_lp or seq.params.logprobs is not None for seq, _, k in rows if k) or any(
            seq in self._prompt_lp for seq, _, _ in rows)
        decode = bool((plan.n_query == 1).all())
        graphed = self.graphs is not None and decode and not sync and len(seqs) <= self._buckets[-1]
        if graphed:
            plan = padded(plan, next(b for b in self._buckets if b >= len(seqs)), self.state.padding_slot)
        ctx = self._context(plan)
        keys = [getattr(meta, "graph_key", None) for meta in ctx.metas.values()]
        out = sampled = tokens = None
        if graphed and None not in keys:
            key = (len(plan.slots), self.sampler.variant(plan.slots), *keys)
            sampled = self.graphs.run(key, lambda: self._decode(ctx))
        else:
            if any(seq in self._prompt_lp for seq in seqs):
                h = self.model.trunk(ctx)
                logits = self.model.head(h[ctx.batch.dev.logits_index])
                self._record_prompt_logprobs(h, plan, seqs)
            else:
                logits = self.model(ctx)            # always: chunks without sampled rows still write their state
            if plan.n_logits.any():
                out = self.sampler.sample(logits.float(), ctx)
                sampled = self._store(ctx, out.tokens)
        if sampled is not None:
            n_sampled = sum(k for _, _, k in rows)
            tokens = self._out_tokens[self._out_turn][:n_sampled]
            tokens.copy_(sampled[:n_sampled], non_blocking=True)
            self._out_turn ^= 1
        done = platform.Event(self.device)
        done.record()
        for seq, n, sampled in rows:
            seq.num_computed += n
            if sampled:
                seq.pending += 1
                if seq.status is not Status.RUNNING:
                    self.sched.finished_prefill(seq)
        return _Inflight(rows, out, tokens, done, sync)

    def _decode(self, ctx: StepContext) -> torch.Tensor:
        """A decode step's device work from inputs to sampled tokens — the part captured in a graph."""
        return self._store(ctx, self.sampler.sample(self.model(ctx).float(), ctx).tokens)

    def _store(self, ctx: StepContext, tokens: torch.Tensor) -> torch.Tensor:
        dev = ctx.batch.dev
        self.last_token[dev.slots[dev.logits_rows]] = tokens               # the device SeqTable
        return tokens

    def _context(self, plan: BatchPlan) -> StepContext:
        """The step's device inputs, staged through pinned memory with asynchronous copies (no host sync)."""
        n = plan.n_query
        qstart = np.concatenate([[0], np.cumsum(n)])
        positions = np.arange(qstart[-1]) + np.repeat(plan.start_upper - qstart[:-1], n)
        up = self.uploader
        up.begin()
        on_device = np.flatnonzero(plan.token_src == TokenSrc.DEVICE_LAST)
        batch = Batch(plan, DeviceBatch(
            input_ids=up.put("input_ids", plan.host_tokens),
            positions=up.put("positions", positions),
            query_start=up.put("query_start", qstart),
            slots=up.put("slots", plan.slots.astype(np.int64)),
            logits_index=up.put("logits_index", qstart[1:][plan.n_logits > 0] - 1),
            logits_rows=up.put("logits_rows", np.flatnonzero(plan.n_logits > 0)),
        ))
        if len(on_device):        # decode inputs sampled by the previous step, still only on the device
            idx = up.put("device_tokens", qstart[on_device])
            batch.dev.input_ids[idx] = self.last_token[up.put("device_slots", plan.slots[on_device].astype(np.int64))]
        pages = self.state.pages(plan, positions, up)
        up.end()
        ctx = StepContext(batch, {g: p.plan_bounds(batch, pages.get(g)) for g, p in self.planners.items()})
        for group, planner in self.planners.items():
            planner.prepare_device(ctx.metas[group], ctx)
        return ctx

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
        lps = None if step.out is None or step.out.logprobs is None else (
            step.out.logprobs.tolist(), step.out.top_ids.tolist(), step.out.top_logprobs.tolist())
        deltas, row = [], 0
        for seq, _, sampled in step.rows:
            if not sampled:
                if seq.status is not Status.FINISHED:
                    self.state.publish(seq)
                continue
            token, row = tokens[row], row + 1
            seq.pending -= 1
            if seq.status is Status.FINISHED:          # finished at an earlier commit; this row was extra
                continue
            seq.tokens.append(token)
            self.state.publish(seq)
            for sibling in reversed(self._siblings.pop(seq, [])):     # its prompt is cached now
                self.sched.add(sibling, front=True)
            reason = self._finish_reason(seq, token)
            if reason:
                self.sched.finish(seq, reason)
            k = seq.params.logprobs
            logprobs = None if k is None else Logprobs([lps[0][row - 1]], [lps[1][row - 1][:k]],
                                                       [lps[2][row - 1][:k]])
            deltas.append(TokenDelta(seq.rid, [token], index=seq.index, logprobs=logprobs,
                                     prompt_logprobs=self._prompt_lp.pop(seq, None), finish_reason=reason))
        return deltas

    def _finish_reason(self, seq: Seq, token: int) -> str | None:
        p = seq.params
        if (token in self.eos and not p.ignore_eos) or token in p.stop_token_ids:
            return "stop"
        generated = len(seq.tokens) - seq.prompt_len
        if (p.max_tokens is not None and generated >= p.max_tokens) or len(seq.tokens) >= self.cfg.max_model_len:
            return "length"
        return None
