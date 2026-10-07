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
    canonical_hash,
)

from . import platform
from .batching import host_plan
from .build import BuildContext, StepContext, build_model, load_weights
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
    kernels: dict[str, str] = dataclasses.field(default_factory=dict)   # op name -> pinned kernel provider

    def __post_init__(self) -> None:
        for name in ("page_size", "max_seqs", "max_model_len", "max_step_tokens"):
            if getattr(self, name) < 1:
                raise ValueError(f"EngineConfig.{name} must be >= 1")


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

        self.sampler = sampler = p.load(p.single("sampler")).sampler(cfg.max_seqs, self.device)
        self.state = StateManager(bound, cfg.max_seqs, cfg.max_model_len, self.device,
                                  on_admit=lambda seq: sampler.admit(seq.slot, seq.params), index=self._index(bound))
        self.sched = Scheduler(self.state, cfg.max_step_tokens)
        self.ledger.materialize(cfg.state_memory if cfg.state_memory is not None else self._measure_state_memory())
        self._ids = itertools.count()
        self._prompt_lp: dict[Seq, Logprobs] = {}         # sequences still accumulating prompt logprobs
        self._siblings: dict[Seq, list[Seq]] = {}         # n > 1: waiting for the first sequence's prefill

    def _index(self, bound: list[BoundSpec]) -> PrefixIndex | None:
        """A prefix index over blocks every paged spec can share whole pages of; None when disabled or when some
        state is not paged (checkpointed recurrent state joins in M0.3)."""
        if not self.cfg.prefix_cache or not bound or not all(isinstance(b.spec.shape, Paged) for b in bound):
            return None
        block = math.lcm(*(b.spec.shape.tokens_per_unit for b in bound), *(b.spec.alignment for b in bound))
        namespace = int(canonical_hash(self.spec).removeprefix("sha256:")[:16], 16)   # model + recipe identity
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
        return bool(self.sched.waiting or self.sched.running)

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
        if not self.has_work():
            return []
        scheduled = self.sched.schedule()
        if scheduled is None:
            raise RuntimeError("the scheduler made no progress with requests pending")
        plan, seqs = scheduled
        ctx = self._context(plan)
        if any(seq in self._prompt_lp for seq in seqs):
            h = self.model.trunk(ctx)
            logits = self.model.head(h[ctx.batch.dev.logits_index])
            self._record_prompt_logprobs(h, plan, seqs)
        else:
            logits = self.model(ctx)
        out = self.sampler.sample(logits.float(), ctx) if plan.n_logits.any() else None
        return self._commit(plan, seqs, out)

    def _context(self, plan: BatchPlan) -> StepContext:
        n = plan.n_query
        qstart = np.concatenate([[0], np.cumsum(n)])
        positions = np.arange(qstart[-1]) + np.repeat(plan.start_upper - qstart[:-1], n)
        d = self.device
        batch = Batch(plan, DeviceBatch(
            input_ids=torch.as_tensor(plan.host_tokens, device=d),
            positions=torch.as_tensor(positions, device=d),
            query_start=torch.as_tensor(qstart, device=d),
            slots=torch.as_tensor(plan.slots, device=d),
            logits_index=torch.as_tensor(qstart[1:][plan.n_logits > 0] - 1, device=d),
            logits_rows=torch.as_tensor(np.flatnonzero(plan.n_logits > 0), device=d),
        ))
        pages = self.state.pages(plan, positions)
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

    def _commit(self, plan: BatchPlan, seqs: list[Seq], out: SampleOut | None) -> list[TokenDelta]:
        deltas, row = [], 0
        tokens = out.tokens.tolist() if out is not None else []
        lps = None if out is None or out.logprobs is None else (
            out.logprobs.tolist(), out.top_ids.tolist(), out.top_logprobs.tolist())     # one transfer per step
        for seq, n, has_logits in zip(seqs, plan.n_query, plan.n_logits, strict=True):
            seq.num_computed += int(n)
            self.state.publish(seq)
            if not has_logits:
                continue
            token = tokens[row]
            seq.tokens.append(token)
            if seq.status is not Status.RUNNING:
                self.sched.finished_prefill(seq)
                for sibling in reversed(self._siblings.pop(seq, [])):
                    self.sched.add(sibling, front=True)
            reason = self._finish_reason(seq, token)
            if reason:
                self.sched.finish(seq, reason)
            k = seq.params.logprobs
            logprobs = None if k is None else Logprobs([lps[0][row]], [lps[1][row][:k]], [lps[2][row][:k]])
            deltas.append(TokenDelta(seq.rid, [token], index=seq.index, logprobs=logprobs,
                                     prompt_logprobs=self._prompt_lp.pop(seq, None), finish_reason=reason))
            row += 1
        return deltas

    def _finish_reason(self, seq: Seq, token: int) -> str | None:
        p = seq.params
        if (token in self.eos and not p.ignore_eos) or token in p.stop_token_ids:
            return "stop"
        generated = len(seq.tokens) - seq.prompt_len
        if (p.max_tokens is not None and generated >= p.max_tokens) or len(seq.tokens) >= self.cfg.max_model_len:
            return "length"
        return None
