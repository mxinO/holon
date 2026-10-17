"""The engine (design §7): requests in, token deltas out. It owns the host side of serving — requests, scheduling
with state reservation, and the commit that turns a step's results into streamed tokens, finished requests and
published prefix blocks — and runs at overlap depth 1 by default: step N is scheduled and launched while step N-1
runs, which is committed meanwhile. Its `Executor` owns the device side of a step (inputs, forward, sampling, CUDA
graphs).

Startup (`assembly`): resolve the weight source and the model plugin for its architecture → bind each mixer group's
state specs to ledger pools and state kinds → build and load the model → size the pools: either a configured byte
budget, or `memory_fraction` of the memory left after weights minus the peak of a worst-case step measured on
minimum pools → capture the decode graphs.
"""
from __future__ import annotations

import collections
import dataclasses
import functools
import gc
import itertools
import secrets
import sys
import time
from typing import Any

import numpy as np
import torch

from holon.api import (
    BatchPlan,
    Logprobs,
    MaskRow,
    RequestError,
    SamplingParams,
    Seq,
    Status,
    TokenDelta,
    VocabView,
    rewind_modules,
    sequence_seed,
)

from . import platform
from .assembly import AuxModels, bind_state, prefix_index, resolve_model
from .build import BuildContext, build_model, load_weights
from .executor import Executor, Launched, PromptRows
from .memory import Ledger
from .plugins import PluginError, PluginSet, discover
from .sched import Scheduler
from .state import StateManager

BLOCKED_POLL_S = 0.005       # how often a server whose only requests wait for their logits processor steps


def _setting(default: object, help: str) -> Any:
    """An EngineConfig field with its help line (`holon serve --help`); a dict default is a fresh dict per config."""
    if isinstance(default, dict):
        return dataclasses.field(default_factory=dict, metadata={"help": help})
    return dataclasses.field(default=default, metadata={"help": help})


@dataclasses.dataclass
class EngineConfig:
    """Every engine setting: one flag each in `holon serve` and the tools, recorded in plans and locks unless
    `plans.NOT_PLANNED` (release design §14.2). A field needs a help line (`_setting`)."""

    model: str = dataclasses.field(metadata={"help": "checkpoint directory or file"})
    device: str = _setting("cuda", "torch device to serve on")
    dtype: torch.dtype = _setting(torch.bfloat16, "activation dtype, by torch name (e.g. bfloat16)")
    source: str | None = _setting(None, "weight-source kind; default: detected from the checkpoint")
    page_size: int = _setting(16, "tokens per page of paged state")
    max_seqs: int = _setting(64, "sequences served at once")
    max_model_len: int = _setting(8192, "prompt + generated tokens per request; pools always hold one")
    clamp_context: bool = _setting(False, "pools may hold less: serve at most what they hold (`holon serve`)")
    max_step_tokens: int = _setting(4096, "tokens one step computes at most")
    chunk_with_decode: int = _setting(0, "prefill tokens a step with decode rows may also carry (0: none)")
    memory_fraction: float = _setting(0.9, "of the memory left after weights, for activations and state")
    state_memory: int | None = _setting(None, "bytes for state pools; default: measured (GPU only)")
    prefix_cache: bool = _setting(True, "reuse cached prompt blocks across requests (and payloads)")
    overlap: bool = _setting(True, "schedule step N while step N-1 runs (depth 1); off: depth 0")
    cuda_graphs: bool = _setting(True, "replay decode steps from CUDA graphs, per batch-size bucket")
    max_draft_tokens: int = _setting(0, "speculative decoding: drafts per verify row at most (0: off)")
    kernels: dict[str, str] = _setting({}, "op name -> pinned kernel provider")
    kernel_preferences: dict[str, str] = _setting(
        {}, "op name -> preferred provider (a lock's pins): used where it supports the site, else default priority")
    slots: dict[str, str] = _setting({}, "slot -> plugin, where several are installed")
    plugin_options: dict[str, object] = _setting({}, "plugin options, read by plugins, never the core")
    host_memory: int = _setting(0, "bytes of page-locked host memory plugin pools may take (0: none)")
    paged_min_tokens: int | None = _setting(
        None, "tokens of paged state per pool sized before plugin fill pools grow; default: max_model_len")
    loop_core: int | None = _setting(None, "CPU core to pin the engine loop to (e.g. one kept free of CPU workers)")

    def __post_init__(self) -> None:
        for name in ("page_size", "max_seqs", "max_model_len", "max_step_tokens"):
            if getattr(self, name) < 1:
                raise ValueError(f"EngineConfig.{name} must be >= 1")
        for name in ("max_draft_tokens", "chunk_with_decode", "host_memory"):
            if getattr(self, name) < 0:
                raise ValueError(f"EngineConfig.{name} must be >= 0")


@dataclasses.dataclass
class _Inflight:
    """A launched step: per-row snapshots (seq, tokens computed, sampled) taken at launch, and its results."""

    rows: list[tuple[Seq, int, int]]          # (seq, tokens computed, logits rows: 0 | 1 | 1 + k)
    plan: BatchPlan
    out: Launched
    sync: bool                               # commit at once (prompt logprobs were extracted on the host)
    publishes: list = dataclasses.field(default_factory=list)   # payloads to register at the commit


class _Target:
    """The TargetView a drafter builds against: the target's spec, its built shared modules, its checkpoint."""

    def __init__(self, spec, model, source, plugin) -> None:
        self.spec, self._model, self._source, self._plugin = spec, model, source, plugin

    def shared(self, name: str):
        if name not in ("embed", "norm", "lm_head"):
            raise KeyError(f"the target shares embed, norm and lm_head, not {name!r}")
        return getattr(self._model, name)

    def tensors(self):
        return self._source.tensors()

    def draft_head(self, name: str):
        declare = getattr(self._plugin, "draft_head", None)
        return declare(self._source.config(), self._source.kind, name) if declare else None


class Engine:
    def __init__(self, cfg: EngineConfig, plugins: PluginSet | None = None) -> None:
        cfg = dataclasses.replace(cfg)             # its own copy: the clamp may lower max_model_len
        self.cfg, self.device = cfg, torch.device(cfg.device)
        platform.configure_allocator(self.device)
        self.plugins = p = plugins or PluginSet(discover())
        source, model_plugin = resolve_model(p, cfg)
        self.source = source                       # what the engine serves (plans name it; bundles read it)
        self.spec = model_plugin.recipe(source.config(), source.kind)
        self.eos = source.eos_token_ids

        floor = cfg.max_model_len if cfg.paged_min_tokens is None else cfg.paged_min_tokens
        self.ledger = Ledger(self.device, host_memory=cfg.host_memory, paged_min_tokens=floor)
        # Speculative decoding (design §7.6) needs drafts enabled and a draft policy installed: then a drafter is built.
        # Units see `max_draft_tokens` 0 without one, and `chunk_with_decode` 0 with one (steps then alternate and never
        # mix), so they and the scheduler share this one predicate.
        drafting = bool(cfg.max_draft_tokens) and any(True for _ in p.by_slot("draft_policy"))
        mixing = cfg.chunk_with_decode if not drafting else 0
        if mixing and cfg.max_step_tokens < cfg.max_seqs + mixing:
            raise ValueError(f"chunk_with_decode {mixing}: max_step_tokens must hold max_seqs decode rows and the "
                             f"chunk ({cfg.max_seqs} + {mixing}), not {cfg.max_step_tokens}")
        options = {"page_size": cfg.page_size, "max_seqs": cfg.max_seqs, "max_model_len": cfg.max_model_len,
                   "max_step_tokens": cfg.max_step_tokens, "max_draft_tokens": cfg.max_draft_tokens if drafting else 0,
                   "chunk_with_decode": mixing,
                   "kernels": cfg.kernels, "kernel_preferences": cfg.kernel_preferences, "plugins": cfg.plugin_options}
        self.ctx = BuildContext(p, self.device, cfg.dtype, options, ledger=self.ledger)
        bound, planners = bind_state(self.spec, self.ctx, p, self.ledger, cfg)
        if mixing and (still := [g for g, planner in planners.items() if not getattr(planner, "mixes_row_kinds", 0)]):
            raise PluginError(f"chunk_with_decode > 0: the planners of {still} cannot run decode rows and prefill "
                              "chunks in one step (`mixes_row_kinds`)")
        model = build_model(self.spec, self.ctx)
        load_weights(model, self.ctx, source.tensors())

        # One extra slot (max_seqs) is the padding slot of graph-captured steps.
        sampler = p.load(p.single("sampler", cfg.slots.get("sampler"))).sampler(cfg.max_seqs + 1, self.device)
        # Speculative decoding (design §7.6): a draft policy picks k per step, one drafter proposes, the verifier
        # checks; all three are plugins, used only when drafts are enabled and a policy is installed.
        self.policy = drafter = verifier = aux = None
        # Speculation counters (stats()): verify steps (graphed, launched at depth 1), real drafts verified, and
        # verify rows by (k, drafts accepted).
        self.spec_counts = collections.Counter()
        self.mixed_steps = 0                       # steps that carried decode rows and prefill chunks (stats())
        substeps, sub_planners = (0, 0), {}
        if drafting:
            pick = lambda slot: p.load(p.single(slot, cfg.slots.get(slot)))        # noqa: E731
            policy, verifier = pick("draft_policy"), pick("verifier")
            self.policy = policy.build(self.ctx) if hasattr(policy, "build") else policy
            self.ctx.aux = aux = AuxModels(self.ctx, p, self.ledger, cfg)
            drafter = pick("drafter").build(self.ctx, _Target(self.spec, model, source, model_plugin))
            if bad := sorted(set(self.ctx.taps) - set(range(len(self.spec.layers)))):
                raise PluginError(f"the drafter taps layers {bad}; the target has {len(self.spec.layers)}")
            bound += aux.bound                           # the drafter's auxiliary models' state and planners
            planners |= aux.planners
            widths = [getattr(drafter, "substeps", lambda k: [])(k) for k in range(cfg.max_draft_tokens + 1)]
            substeps = (max(map(len, widths)), max((q for w in widths for q in w), default=0))
            sub_planners = aux.sub_planners(substeps[0])
        admit_draft = getattr(drafter, "admit", None)
        # At most one logits processor, selected by the deployment (opt-in, like the drafter).
        self.processor = None
        if "logits_processor" in cfg.slots:
            self.processor = p.load(p.single("logits_processor", cfg.slots["logits_processor"])).build(
                self.ctx, VocabView(self.spec.vocab, self.eos, source.tokenizer))

        # Shared objects (`BuildCtx.shared`): their last pool requests, then a reset after every materialize. They are
        # all built by now: the hooks run for these.
        self.shared, self.ctx.sealed = list(self.ctx.shared_objects.values()), True
        for o in self.shared:
            if hasattr(o, "before_materialize"):
                o.before_materialize(self.ctx)
        self.ledger.after_materialize = [functools.partial(o.after_materialize, self.ctx) for o in self.shared
                                         if hasattr(o, "after_materialize")]

        self._hits: dict[Seq, int | None] = {}            # first admission's prefix hit, None once reported

        def admit(seq: Seq) -> None:
            self._hits.setdefault(seq, seq.num_computed)   # the prefix hit of its first admission only
            sampler.admit(seq.slot, seq.params)
            if admit_draft is not None:
                admit_draft(seq.slot, seq.tokens)

        self.state = StateManager(bound, cfg.max_seqs, cfg.max_model_len, self.device, on_admit=admit,
                                  index=prefix_index(self.spec, bound, cfg.prefix_cache, self.ledger.pools),
                                  overflow=self._overflow(drafter))
        host_lens = any(getattr(planner, "needs_host_lens", False) for planner in planners.values())
        self.sched = Scheduler(self.state, cfg.max_step_tokens, cfg.max_model_len,
                               draft_k=self._draft_k if self.policy is not None else None,
                               lookahead=getattr(drafter, "lookahead", None), host_lens_only=host_lens,
                               chunk_with_decode=cfg.chunk_with_decode)
        # What rolls back after a verify (api.Rewindable): the target's modules and the slot objects. An auxiliary
        # model runs inside `Drafter.run`, after this: rolling it back is the drafter's call.
        slot_objects = [o for o in (sampler, verifier, drafter, self.policy) if callable(getattr(o, "rewind", None))]
        # Slot objects with counters of their own (`stats()`), reported by `stats()` under their plugin's name.
        self._reporters = [(p.single(slot, cfg.slots.get(slot)), o) for slot, o in (
            ("sampler", sampler), ("verifier", verifier), ("drafter", drafter), ("draft_policy", self.policy),
            ("logits_processor", self.processor)) if callable(getattr(o, "stats", None))]
        rewinders = [functools.partial(rewind_modules, model), *(o.rewind for o in slot_objects)]
        self.executor = Executor(cfg, self.device, model, self.spec.vocab, sampler, self.state, planners,
                                 verifier, drafter, pools=self.ledger.pools, rewinders=rewinders,
                                 sub_planners=sub_planners, substeps=substeps,
                                 namespaces=[ns for ns, _ in aux.modules] if aux is not None else [],
                                 verify_widths=self._verify_widths(), processor=self.processor, shared=self.shared)
        self.ledger.materialize(cfg.state_memory if cfg.state_memory is not None
                                else self.executor.state_memory(self.ledger, cfg.memory_fraction))
        if cfg.clamp_context:
            self._clamp()
        self.state.reset_padding()
        for pool in self.ledger.pools.values():          # fresh segments, so they show the allocator setting
            if pool.buffer is not None and not pool.host:   # (a plugin pool may get no units)
                platform.check_expandable(pool.buffer)
        self.executor.capture_graphs()
        if cfg.loop_core is not None:
            platform.pin_loop(cfg.loop_core)

        self._inflight: _Inflight | None = None
        self._stopped: str | None = None                  # why the engine stopped (a shared object's commit failed)
        self._ids = itertools.count()
        self._prompt_lp: dict[Seq, Logprobs] = {}         # sequences still accumulating prompt logprobs
        self._siblings: dict[Seq, list[Seq]] = {}         # n > 1: waiting for the first sequence's prefill
        self._live: dict[str, list[Seq]] = {}              # request id -> its unfinished sequences
        self._blocked: dict[str, None] = {}                # constrained requests until their processor is ready
        # What a request's `SamplingParams.ext` may name: the resources of the plugins this engine runs.
        self._served = {str(r) for name in p.loaded for r in p.installed[name].manifest.provides}

    def _clamp(self) -> None:
        """Serve at most what the paged pools hold for one request (`clamp_context`; release design D2): after they
        are sized, before graphs are captured (they bake in the length limit). What was built for the requested
        length (static buffers, block tables, the profile) keeps it."""
        fits = min(((p.capacity - p.has_null) * p.tokens_per_unit for p in self.ledger.pools.values()
                    if p.tokens_per_unit), default=self.cfg.max_model_len)
        if fits < self.cfg.max_model_len:
            print(f"holon: max_model_len {self.cfg.max_model_len} clamped to {fits}, what the state pools hold",
                  file=sys.stderr, flush=True)
            self.cfg.max_model_len = self.sched.max_model_len = fits
            self.state.limit(fits)

    def _verify_widths(self) -> tuple[int, ...]:
        """The verify-row widths to capture: 1 + each k the policy declares (`ks()`), else every k up to the maximum."""
        if self.policy is None:
            return ()
        ks = self.policy.ks() if hasattr(self.policy, "ks") else range(1, self.cfg.max_draft_tokens + 1)
        return tuple(sorted({1 + k for k in ks if 0 < k <= self.cfg.max_draft_tokens}))

    def _overflow(self, drafter) -> int:
        """Positions past max_model_len a step may touch: a verify row padded from one below the limit, then what
        the drafter writes ahead of it."""
        k = self.cfg.max_draft_tokens if drafter is not None else 0
        if not k:
            return 0
        return 1 + 2 * k + (drafter.lookahead(k) if hasattr(drafter, "lookahead") else 0)

    def _draft_k(self, seqs: list[Seq]) -> int:
        """Drafts per row the policy wants for the next step of `seqs` (the verifier serves every request: greedy,
        sampled, with logprobs)."""
        return max(0, min(self.policy.k(seqs), self.cfg.max_draft_tokens))

    # -- requests -------------------------------------------------------------------------------------------------
    def add(self, token_ids: list[int], params: SamplingParams | None = None, rid: str | None = None,
            resume_output: int = 0) -> str:
        """Queue a request; its `params.n` sequences stream as TokenDeltas with `index` 0 … n-1. `rid` defaults to a
        fresh id; callers that route results themselves (frontends) choose it. `resume_output`: the last tokens of
        `token_ids` are output already (a resubmission): they count towards `max_tokens`, and a logits processor
        replays them. Raises ValueError (`RequestError` from the processor) for a request it rejects."""
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
        if not 0 <= resume_output < len(token_ids) or (p.max_tokens is not None and resume_output >= p.max_tokens):
            raise ValueError(f"resume_output {resume_output}: not within the {len(token_ids)} tokens and max_tokens")
        if unserved := set(p.ext) - self._served:     # never ignore what a request asks for
            raise ValueError(f"no plugin of this deployment serves {sorted(unserved)} (SamplingParams.ext)")
        rid = rid if rid is not None else str(next(self._ids))
        if rid in self._live:
            raise ValueError(f"request id {rid!r} is already in flight")
        prompt_len = len(token_ids) - resume_output
        constrained = self.processor is not None and self.processor.open(rid, p, list(token_ids[:prompt_len]))
        base = p.seed if p.seed is not None else secrets.randbits(63)
        for i in range(p.n):
            # A concrete seed per sequence keeps sampling reproducible across batching and preemption.
            child = dataclasses.replace(p, seed=sequence_seed(base, i),
                                        prompt_logprobs=p.prompt_logprobs if i == 0 else None)
            seq = Seq(rid=rid, tokens=list(token_ids), params=child, prompt_len=prompt_len, index=i,
                      constrained=constrained)
            self._live.setdefault(rid, []).append(seq)       # siblings too: abort must account for them
            if child.prompt_logprobs is not None:
                self._prompt_lp[seq] = Logprobs([], [], [])
            if i == 0 or self.state.index is None:
                self.sched.add(seq)
                if constrained:                               # until its processor is ready (`_unblock`)
                    seq.status = Status.BLOCKED
                first = seq
            else:   # siblings wait until the first sequence's prompt is cached, then adopt it
                self._siblings.setdefault(first, []).append(seq)
        if constrained:
            self._blocked[rid] = None
        return rid

    def has_work(self) -> bool:
        return bool(self.sched.waiting or self.sched.running or self._inflight)

    def idle_for(self) -> float | None:
        """How long a serving loop may wait for new requests before its next `step`: 0 while some request can run,
        BLOCKED_POLL_S while the only ones left wait for their logits processor (each step polls `ready`), None
        (indefinitely) when there is no work."""
        if not self.has_work():
            return None
        if self.sched.running or self._inflight is not None:
            return 0.0
        return BLOCKED_POLL_S if all(seq.status is Status.BLOCKED for seq in self.sched.waiting) else 0.0

    # -- control (design §7.7) -------------------------------------------------------------------------------------
    def abort(self, rid: str, reason: str = "abort", error: str | None = None, kind: str | None = None
              ) -> list[TokenDelta]:
        """Finish every sequence of request `rid` now — one delta each, siblings that never started included; a row
        already in flight is discarded at its commit. `error` and `kind` (a `RequestError`'s) say why it failed."""
        deltas, seqs = [], self._live.pop(rid, [])
        for seq in seqs:
            for sibling in self._siblings.pop(seq, []):             # never started: nothing to release
                sibling.status = Status.FINISHED
                deltas.append(TokenDelta(rid, [], sibling.index, finish_reason=reason, error=error, kind=kind))
            if seq.status is Status.FINISHED:
                continue
            self.sched.finish(seq, reason)
            self._prompt_lp.pop(seq, None)
            self._hits.pop(seq, None)
            deltas.append(TokenDelta(rid, [], seq.index, finish_reason=reason, error=error, kind=kind))
        if seqs and seqs[0].constrained:
            self._blocked.pop(rid, None)
            self.processor.close(rid)
        return deltas

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
        breaks the engine's reference cycles; the engine is unusable afterwards (closing again does nothing)."""
        if self.executor is None:
            return
        try:
            if not self._stopped:                        # a stopped engine's later steps are discarded
                self.drain()
        finally:                                         # a failing commit still stops the helpers
            platform.synchronize(self.device)            # (every stream) helpers and host pools go once it is idle
            for o in (self.processor, *self.shared):
                getattr(o, "shutdown", lambda: None)()
            self.executor = self._inflight = None
            self.shared, self.ctx.shared_objects = [], {}   # what they hold, host memory included, goes too
            self.ledger.free(host=True)
            gc.collect()
            platform.release_cache(self.device)

    def stats(self) -> dict[str, int]:
        out = {"waiting": len(self.sched.waiting), "running": len(self.sched.running),
               "preemptions": self.sched.preemptions, "prefix_tokens_saved": self.state.tokens_saved,
               "mixed_steps": self.mixed_steps,
               "cached_blocks": len(self.state.index.entries) if self.state.index is not None else 0,
               "graphs": len(self.executor.graphs) if self.executor and self.executor.graphs is not None else 0}
        if self.policy is not None:                  # the speculation counters (__init__)
            out |= {f"spec_{key}" if isinstance(key, str) else f"spec_rows_k{key[0]}_accepting_{key[1]}": n
                    for key, n in self.spec_counts.items()}
        for plugin, o in self._reporters:
            out |= {f"{plugin}.{key}": n for key, n in o.stats().items()}
        return out

    def generate(self, prompts: list[list[int]], params: SamplingParams | None = None) -> list[list[int]]:
        """Token ids generated for each prompt (one sequence per prompt)."""
        if params is not None and params.n != 1:
            raise ValueError("generate() returns one sequence per prompt; step() streams n > 1")
        rids = [self.add(p, params) for p in prompts]
        out: dict[str, list[int]] = {r: [] for r in rids}
        pending = set(rids)
        while pending:
            if wait := self.idle_for():              # only requests waiting for their logits processor
                time.sleep(wait)
            for d in self.step():
                if d.rid in out:                     # other requests may be in flight too
                    out[d.rid].extend(d.token_ids)
                    if d.finish_reason:
                        pending.discard(d.rid)
        return [out[r] for r in rids]

    # -- one step -------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def step(self) -> list[TokenDelta]:
        """Schedule and launch one step; commit the previous one while this one runs (overlap) — before scheduling
        if it holds a constrained row, whose processor needs that row's tokens before the sequence's next row (so a
        constrained sequence decodes every step and the step's other rows lose their overlap: M0.5 design D2) — or
        this one at once (depth 0, and steps that report prompt logprobs, whose extraction needs the host)."""
        if self._stopped:
            raise RuntimeError(f"the engine stopped: {self._stopped}")
        prev, self._inflight = self._inflight, None
        deltas = self._unblock() if self._blocked else []
        if prev is not None and self.processor is not None and any(m and seq.constrained for seq, _, m in prev.rows):
            deltas, prev = deltas + self._commit(prev), None
        scheduled = self.sched.schedule()
        if scheduled is None and prev is not None:      # what is left waits on the in-flight step's tokens
            deltas, prev = deltas + self._commit(prev), None
            scheduled = self.sched.schedule()
        if scheduled is None:
            if self.idle_for() == 0:                    # not only requests waiting for their logits processor
                raise RuntimeError("the scheduler made no progress with requests pending")
            return deltas
        step = self._launch(*scheduled)
        self.mixed_steps += scheduled[0].n_decode > 0
        if self.policy is not None and any(m > 1 for _, _, m in step.rows):
            self.spec_counts.update(verify_steps=1, verify_steps_graphed=step.out.graphed,
                             verify_steps_overlapped=prev is not None)
        keep = self.cfg.overlap and not step.sync
        if keep:                                         # tracked before `prev`'s commit, which may fail
            self._inflight = step
        if prev is not None:
            deltas += self._commit(prev)
        if not keep:
            deltas += self._commit(step)
        return deltas

    def _unblock(self) -> list[TokenDelta]:
        """Let the constrained requests whose processor is ready run — a resumed one after replaying its output into
        the processor (`commit`); end those the processor fails."""
        deltas = []
        for rid in list(self._blocked):
            try:
                if not self.processor.ready(rid):
                    continue
                for seq in self._live[rid]:
                    if len(seq.tokens) > seq.prompt_len:
                        self.processor.commit(seq, seq.tokens[seq.prompt_len:])
            except RequestError as e:
                deltas += self.abort(rid, "error", str(e), e.kind)
                continue
            del self._blocked[rid]
            for seq in self._live[rid]:
                if seq.status is Status.BLOCKED:
                    seq.status = Status.WAITING
        return deltas

    def _launch(self, plan: BatchPlan, seqs: list[Seq]) -> _Inflight:
        """Launch the step on the device, then advance the host mirrors."""
        rows = [(seq, int(n), int(m)) for seq, n, m in zip(seqs, plan.n_query, plan.n_logits, strict=True)]
        sync = any(seq in self._prompt_lp for seq in seqs)
        exact = np.array([seq.inexact == 0 for seq in seqs], dtype=bool)
        logprobs = any(m and seq.params.logprobs is not None for seq, _, m in rows)
        prompt = self._prompt_rows(plan, seqs) if sync else None
        ops = self.state.step_ops(plan, seqs)
        plan.mem_ops, plan.publishes = ops.mem_ops, ops.in_step
        masks = self._masks(rows) if self.processor is not None else None
        out = self.executor.launch(plan, exact, logprobs, prompt, masks)
        self.state.unpin(ops.pins)
        for r, got in zip(prompt or [], out.prompt_logprobs, strict=True):
            acc = self._prompt_lp[seqs[r.row]]
            acc.token += got.token
            acc.top_ids += got.top_ids
            acc.top += got.top
        running = Status.RUNNING
        for seq, n, m in rows:
            seq.num_computed += n                   # an upper bound for verify rows until their commit
            if m:
                seq.pending += m
                if m > 1:
                    seq.inexact += 1
                if seq.status is not running:
                    self.sched.finished_prefill(seq)
        return _Inflight(rows, plan, out, sync, ops.publishes)

    @staticmethod
    def _masks(rows: list[tuple[Seq, int, int]]) -> tuple[list[MaskRow], np.ndarray]:
        """The step's constrained rows — a verify row with the drafts it checks — and their logits rows (none: the
        step still applies, over padding only, so one graph serves steps with and without them)."""
        masks, index, at = [], [], 0
        for seq, _, m in rows:
            if m and seq.constrained:
                masks.append(MaskRow(seq, len(index), m, seq.drafts[: m - 1]))
                index += range(at, at + m)
            at += m
        return masks, np.array(index, dtype=np.int64)

    def _prompt_rows(self, plan: BatchPlan, seqs: list[Seq]) -> list[PromptRows]:
        """The prompt logprobs this step's rows owe: the prompt positions they compute that predict a prompt token,
        except those already recorded (a request preempted mid-prompt recomputes them)."""
        out = []
        for i, seq in enumerate(seqs):
            acc = self._prompt_lp.get(seq)
            if acc is None:
                continue
            start = int(plan.start_upper[i])
            lo, hi = max(start, len(acc.token)), min(start + int(plan.n_query[i]), seq.prompt_len - 1)
            if lo < hi:
                out.append(PromptRows(i, lo - start, hi - start, max(seq.params.prompt_logprobs, 1),
                                      seq.tokens[lo + 1: hi + 1]))
        return out

    def _commit(self, step: _Inflight) -> list[TokenDelta]:
        """Host bookkeeping once a step's results are on the host: tokens, finishing, publishing, streaming."""
        out = step.out
        out.done.synchronize()
        for observe, telemetry in out.observed:         # shared objects, before any token is emitted
            try:
                observe(step.plan, telemetry)
            except Exception as e:                       # the steps launched after this one are discarded
                self._stopped = f"after_commit of step {step.plan.step}: {type(e).__name__}: {e}"
                raise
        tokens = out.tokens.tolist() if out.tokens is not None else []
        accepts = iter(out.accepts.tolist() if out.accepts is not None else [])
        drafts = out.drafts.tolist() if out.drafts is not None else None
        draft_tokens = out.draft_tokens                # read only for constrained rows
        lps = None if out.logprobs is None else [t.tolist() for t in out.logprobs]
        deltas, row = [], 0
        index, chains, finished = self.state.index, self.state.chain, Status.FINISHED
        block = index.block if index is not None else 0
        for i, (seq, _, m) in enumerate(step.rows):
            if not m:
                if seq.status is not finished and self._completes_block(seq):
                    self.state.publish(seq)
                continue
            at, accepted = row, 0
            if m > 1:                                # a verify row: the accepted drafts, then the target's token
                accepted = next(accepts)
                emitted = tokens[row: row + accepted + 1]
                seq.inexact -= 1
                if seq.status is Status.RUNNING:     # the host length is exact again (preempted ones restart at 0)
                    seq.num_computed -= m - 1 - accepted
            row += m
            seq.pending -= m
            if seq.status is finished:             # finished at an earlier commit; this row was extra
                continue
            if drafts is not None:
                self.policy.observe(seq, step.plan.draft_k, drafts[i], accepted)
                if m > 1:                            # this row verified what the last commit's drafts were
                    self.spec_counts["drafted"] += min(seq.n_draft, m - 1)
                    self.spec_counts[m - 1, accepted] += 1
                seq.n_draft = drafts[i]
                if seq.constrained:                  # for the masks of its next verify row
                    seq.drafts = draft_tokens[i, : drafts[i]].tolist()
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
            if seq.constrained:
                try:
                    self.processor.commit(seq, out)
                except RequestError as e:            # the request fails: these tokens are not its own
                    del seq.tokens[len(seq.tokens) - len(out):]
                    deltas += self.abort(seq.rid, "error", str(e), e.kind)
                    continue
            # _completes_block, inline: this runs per row per step
            if block and (chain := chains.get(seq.slot)) is not None and \
                    min(seq.num_computed, len(seq.tokens)) // block > len(chain):
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
                    if seq.constrained:
                        self.processor.close(seq.rid)
            k = seq.params.logprobs                  # one entry per emitted token (a verify row emits several)
            logprobs = None if k is None else Logprobs(
                lps[0][at: at + len(out)], [t[:k] for t in lps[1][at: at + len(out)]],
                [t[:k] for t in lps[2][at: at + len(out)]])
            prompt_lp = self._prompt_lp.pop(seq, None) if self._prompt_lp else None
            cached = self._hits.pop(seq, None) if reason else self._hits.get(seq)
            if cached is not None and not reason:
                self._hits[seq] = None
            deltas.append(TokenDelta(seq.rid, out, seq.index, logprobs, prompt_lp, reason, cached_tokens=cached))
        for seq, spec, at, payload in step.publishes:    # after this commit's blocks are cached
            self.state.register(seq, spec, at, payload)
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
