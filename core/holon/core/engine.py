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

import dataclasses
import gc
import itertools
import secrets

import numpy as np
import torch

from holon.api import BatchPlan, Logprobs, SamplingParams, Seq, Status, TokenDelta, sequence_seed

from . import platform
from .assembly import bind_state, prefix_index, resolve_model
from .build import BuildContext, build_model, load_weights
from .executor import Executor, Launched, PromptRows
from .memory import Ledger
from .plugins import PluginSet, discover
from .sched import Scheduler
from .state import StateManager


@dataclasses.dataclass
class EngineConfig:
    model: str                               # checkpoint directory
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    source: str | None = None                # weight source kind; None: detected from the checkpoint
    page_size: int = 16
    max_seqs: int = 64
    max_model_len: int = 8192                # prompt + generated tokens per request; pools always hold one
    max_step_tokens: int = 4096
    memory_fraction: float = 0.9             # of the memory left after weights, for activations and state
    state_memory: int | None = None          # bytes for state pools; None: measure (GPU only)
    prefix_cache: bool = True                # reuse cached prompt blocks across requests (and payloads)
    overlap: bool = True                     # schedule step N while step N-1 runs (depth 1); False: depth 0
    cuda_graphs: bool = True                 # replay decode steps from CUDA graphs, per batch-size bucket
    max_draft_tokens: int = 0                # speculative decoding: drafts per verify row at most (0: off)
    kernels: dict[str, str] = dataclasses.field(default_factory=dict)   # op name -> pinned kernel provider
    plugin_options: dict[str, object] = dataclasses.field(default_factory=dict)   # read by plugins, never the core

    def __post_init__(self) -> None:
        for name in ("page_size", "max_seqs", "max_model_len", "max_step_tokens"):
            if getattr(self, name) < 1:
                raise ValueError(f"EngineConfig.{name} must be >= 1")
        if self.max_draft_tokens < 0:
            raise ValueError("EngineConfig.max_draft_tokens must be >= 0")


@dataclasses.dataclass
class _Inflight:
    """A launched step: per-row snapshots (seq, tokens computed, sampled) taken at launch, and its results."""

    rows: list[tuple[Seq, int, int]]          # (seq, tokens computed, logits rows: 0 | 1 | 1 + k)
    out: Launched
    sync: bool                               # commit at once (prompt logprobs were extracted on the host)
    publishes: list = dataclasses.field(default_factory=list)   # payloads to register at the commit


class Engine:
    def __init__(self, cfg: EngineConfig, plugins: PluginSet | None = None) -> None:
        self.cfg, self.device = cfg, torch.device(cfg.device)
        platform.configure_allocator(self.device)
        self.plugins = p = plugins or PluginSet(discover())
        source, model_plugin = resolve_model(p, cfg)
        self.source = source                       # what the engine serves (plans name it; bundles read it)
        self.spec = model_plugin.recipe(source.config(), source.kind)
        self.eos = source.eos_token_ids

        self.ledger = Ledger(self.device)
        options = {"page_size": cfg.page_size, "max_seqs": cfg.max_seqs, "max_model_len": cfg.max_model_len,
                   "max_step_tokens": cfg.max_step_tokens, "kernels": cfg.kernels, "plugins": cfg.plugin_options}
        self.ctx = BuildContext(p, self.device, cfg.dtype, options, ledger=self.ledger)
        bound, planners = bind_state(self.spec, self.ctx, p, self.ledger, cfg)
        model = build_model(self.spec, self.ctx)
        load_weights(model, self.ctx, source.tensors())

        # One extra slot (max_seqs) is the padding slot of graph-captured steps.
        sampler = p.load(p.single("sampler")).sampler(cfg.max_seqs + 1, self.device)
        self.state = StateManager(bound, cfg.max_seqs, cfg.max_model_len, self.device,
                                  on_admit=lambda seq: sampler.admit(seq.slot, seq.params),
                                  index=prefix_index(self.spec, bound, cfg.prefix_cache, self.ledger.pools))
        # Speculative decoding (design §7.6): a draft policy picks k per request, one drafter proposes, the verifier
        # checks; all three are plugins, used only when drafts are enabled and a policy is installed.
        self.policy = drafter = verifier = None
        if cfg.max_draft_tokens and any(True for _ in p.by_slot("draft_policy")):
            self.policy, drafter = p.load(p.single("draft_policy")), p.load(p.single("drafter"))
            verifier = p.load(p.single("verifier"))
        host_lens = any(getattr(planner, "needs_host_lens", False) for planner in planners.values())
        self.sched = Scheduler(self.state, cfg.max_step_tokens, cfg.max_model_len,
                               draft_k=self._draft_k if self.policy is not None else None, host_lens_only=host_lens)
        self.executor = Executor(cfg, self.device, model, self.spec.vocab, sampler, self.state, planners,
                                 verifier, drafter, pools=self.ledger.pools)
        self.ledger.materialize(cfg.state_memory if cfg.state_memory is not None
                                else self.executor.state_memory(self.ledger, cfg.memory_fraction))
        self.state.reset_padding()
        for pool in self.ledger.pools.values():          # fresh segments, so they show the allocator setting
            if pool.buffer is not None:                  # (a plugin pool may get no units)
                platform.check_expandable(pool.buffer)
        self.executor.capture_graphs()

        self._inflight: _Inflight | None = None
        self._ids = itertools.count()
        self._prompt_lp: dict[Seq, Logprobs] = {}         # sequences still accumulating prompt logprobs
        self._siblings: dict[Seq, list[Seq]] = {}         # n > 1: waiting for the first sequence's prefill
        self._live: dict[str, list[Seq]] = {}              # request id -> its unfinished sequences

    def _draft_k(self, seq: Seq) -> int:
        """Drafts the policy wants for `seq` this step. Only greedy requests without logprobs are drafted: the
        verifier is greedy-exact and reports no logprobs."""
        if self.policy is None or seq.params.temperature != 0 or seq.params.logprobs is not None:
            return 0
        return min(self.policy.choose(seq)[1], self.cfg.max_draft_tokens)

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
        base = p.seed if p.seed is not None else secrets.randbits(63)
        for i in range(p.n):
            # A concrete seed per sequence keeps sampling reproducible across batching and preemption.
            child = dataclasses.replace(p, seed=sequence_seed(base, i),
                                        prompt_logprobs=p.prompt_logprobs if i == 0 else None)
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
        self.executor = self._inflight = None
        self.ledger.free()
        gc.collect()
        platform.release_cache(self.device)

    def stats(self) -> dict[str, int]:
        return {"waiting": len(self.sched.waiting), "running": len(self.sched.running),
                "preemptions": self.sched.preemptions, "prefix_tokens_saved": self.state.tokens_saved,
                "cached_blocks": len(self.state.index.entries) if self.state.index is not None else 0,
                "graphs": len(self.executor.graphs) if self.executor and self.executor.graphs is not None else 0}

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
        once (depth 0, and steps that report prompt logprobs, whose extraction needs the host)."""
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
        """Launch the step on the device, then advance the host mirrors."""
        rows = [(seq, int(n), int(m)) for seq, n, m in zip(seqs, plan.n_query, plan.n_logits, strict=True)]
        sync = any(seq in self._prompt_lp for seq in seqs)
        exact = np.array([seq.inexact == 0 for seq in seqs], dtype=bool)
        logprobs = any(m and seq.params.logprobs is not None for seq, _, m in rows)
        prompt = self._prompt_rows(plan, seqs) if sync else None
        ops = self.state.step_ops(plan, seqs)
        plan.mem_ops = ops.mem_ops
        out = self.executor.launch(plan, exact, logprobs, prompt)
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
        return _Inflight(rows, out, sync, ops.publishes)

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
        tokens = out.tokens.tolist() if out.tokens is not None else []
        accepts = iter(out.accepts.tolist() if out.accepts is not None else [])
        lps = None if out.logprobs is None else [t.tolist() for t in out.logprobs]
        deltas, row = [], 0
        index, chains, finished = self.state.index, self.state.chain, Status.FINISHED
        block = index.block if index is not None else 0
        for seq, _, m in step.rows:
            if not m:
                if seq.status is not finished and self._completes_block(seq):
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
            if seq.status is finished:             # finished at an earlier commit; this row was extra
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
            k = seq.params.logprobs                  # logprob requests are never drafted: one token per row
            logprobs = None if k is None else Logprobs([lps[0][at]], [lps[1][at][:k]], [lps[2][at][:k]])
            prompt_lp = self._prompt_lp.pop(seq, None) if self._prompt_lp else None
            deltas.append(TokenDelta(seq.rid, out, seq.index, logprobs, prompt_lp, reason))
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
