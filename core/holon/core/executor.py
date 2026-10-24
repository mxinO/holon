"""The device side of a step (design §7.3, §7.4): inputs staged into static buffers and finished on the device from
the device SeqTable, the forward, sampling or verification, drafting, and CUDA graph replay of decode steps — plus
the startup work that needs the built model: the worst-case step's memory and graph capture.

The engine owns requests, scheduling and commits; it hands the executor a scheduled plan and gets back the pinned
host copies of the step's results and the event that says they have landed.
"""
from __future__ import annotations

import contextlib
import dataclasses
from typing import Any

import numpy as np
import torch

from holon.api import Batch, BatchPlan, DeviceBatch, Logprobs, MaskRow, Pages, SamplingParams, SeqTable, TokenSrc

from . import platform
from .batching import host_plan, padded
from .build import StepContext, local_batch
from .inputs import Uploader, Writes
from .memory import Copies, Ledger, run_mem_ops
from .state import StateManager

PROMPT_LOGPROB_CHUNK = 256     # rows of full-vocabulary logits materialized at once for prompt logprobs
LOGPROB_F32_ROWS = 64          # of them, rows taken to f32 log-probabilities at once (row-wise: the same bits)
BUCKETS = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256)      # decode batch sizes captured as graphs

# Sampler variants captured at startup: greedy, sampled, sampled with filters (logprob steps run eagerly).
_VARIANTS = (SamplingParams(temperature=0.0, seed=0), SamplingParams(temperature=1.0, seed=0),
             SamplingParams(temperature=1.0, top_p=0.9, seed=0))


@dataclasses.dataclass
class PromptRows:
    """Prompt logprobs a step owes one batch row: its query tokens `first` … `end`-1 predict `targets`; top `k`."""

    row: int
    first: int
    end: int
    k: int
    targets: list[int]


@dataclasses.dataclass
class Launched:
    """A launched step's results: pinned host copies, valid once `done` has fired."""

    tokens: torch.Tensor | None
    accepts: torch.Tensor | None             # verify rows' accepted draft counts
    drafts: torch.Tensor | None              # per plan row: the drafts written for its next step (with a drafter)
    logprobs: tuple[torch.Tensor, ...] | None   # sampled tokens' logprob, top ids, top logprobs
    done: platform.Event
    prompt_logprobs: list[Logprobs] = dataclasses.field(default_factory=list)   # per PromptRows, on the host now
    graphed: bool = False                    # replayed a captured graph
    draft_tokens: torch.Tensor | None = None   # per plan row: the drafts themselves (with a logits processor)
    observed: list = dataclasses.field(default_factory=list)   # (a shared object's `after_commit`, its telemetry)


@dataclasses.dataclass
class _Prep:
    """Static device inputs of a step's device prep (`Executor._prepare`)."""

    lens: torch.Tensor             # [B] tokens held after the step (shared with every page group's Pages)
    start: torch.Tensor            # [B] the row's start (where the host knows it exactly)
    inputs: torch.Tensor           # the step's input-ids buffer: `DeviceBatch.input_ids`, then a scratch element
    input_at: torch.Tensor         # [B] where the device's last token goes: the row's first input, or the scratch
    logits_slots: torch.Tensor     # [L] the slot of each logits row
    host_staged: bool = False                 # positions, lengths and write indices came from the host
    logits_end: torch.Tensor | None = None    # [L] host-staged steps: the row's length after the step
    # Steps whose starts the device may hold (verify rows): positions, lengths and write indices derived on the device.
    row_of_token: torch.Tensor | None = None  # [T] batch row of each token
    offset: torch.Tensor | None = None        # [T] index of each token within its row
    n_query: torch.Tensor | None = None       # [B]
    exact: torch.Tensor | None = None         # [B] the host knows this row's start
    draft_index: torch.Tensor | None = None   # [R*k] verify rows: input positions of the drafts
    draft_from: torch.Tensor | None = None    # [R*k] where each draft is in the SeqTable (flat)
    mask: torch.Tensor | None = None          # [M, mask_width] the logits processor's masks of the constrained rows
    mask_rows: torch.Tensor | None = None     # [M] their logits rows; past the step's logits rows: padding


class _Substeps:
    """A step's planned drafter sub-steps, handed out in order (`StepCtx.substep`); `reset` before each run of the
    drafter (a graph's warm-up and capture run it twice)."""

    def __init__(self, executor: Executor, subs: list[StepContext], ahead: int) -> None:
        self.executor, self.subs, self.ahead, self.cursor = executor, subs, ahead, 0

    def reset(self) -> None:
        self.cursor = 0

    def next(self, q: int, offset: int) -> StepContext:
        if self.cursor >= len(self.subs) or int(self.subs[self.cursor].batch.plan.n_query.max(initial=q)) != q:
            raise RuntimeError(f"sub-step {self.cursor} of width {q} was not declared (Drafter.substeps)")
        if offset + q > self.ahead:            # what the scheduler reserved past each row (Drafter.lookahead)
            raise RuntimeError(f"sub-step at offset {offset} of width {q} writes past the lookahead {self.ahead}")
        sub, i = self.subs[self.cursor], self.cursor
        self.cursor += 1
        self.executor._prepare_substep(sub, i, offset)
        return sub


def _last_tokens(qstart: np.ndarray, n_logits: np.ndarray) -> np.ndarray:
    """Flat indices of each row's last `n_logits` tokens (the tokens whose logits a step computes)."""
    ends = np.repeat(qstart[1:], n_logits)
    first = np.repeat(np.cumsum(n_logits) - n_logits, n_logits)          # where each row's run starts in `ends`
    return ends - n_logits.repeat(n_logits) + (np.arange(len(ends)) - first)


def profile_rows(cfg: Any, tokens: int = 0) -> list[int]:
    """The rows of the worst-case step `state_memory` profiles: `tokens` (default `max_step_tokens`) prompt tokens in
    rows of at most `max_model_len`, at most `max_seqs` of them."""
    tokens = tokens or cfg.max_step_tokens
    rows = [cfg.max_model_len] * (tokens // cfg.max_model_len)
    return [r for r in (rows + [tokens % cfg.max_model_len])[:cfg.max_seqs] if r]


class Executor:
    def __init__(self, cfg: Any, device: torch.device, model: Any, vocab: int, sampler: Any, state: StateManager,
                 planners: dict[str, Any], verifier: Any = None, drafter: Any = None,
                 pools: dict[str, Any] | None = None, rewinders: list[Any] = (),
                 sub_planners: dict[str, list[Any]] | None = None, substeps: tuple[int, int] = (0, 0),
                 namespaces: list[str] = (), verify_widths: tuple[int, ...] = (), processor: Any = None,
                 shared: list[Any] = ()) -> None:
        """`cfg`: the EngineConfig (sizes, graphs on or off); `pools`: every ledger pool, by name (for MemOps);
        `rewinders`: what rolls back past rejected drafts after each verify (`rewind(ctx)` callables);
        `sub_planners`: per auxiliary state group, one planner per drafter sub-step; `substeps`: how many sub-steps a
        step may have and their widest row (the drafter's declarations); `namespaces`: the auxiliary models';
        `verify_widths`: the verify-row widths (1 + k) the draft policy may ask for, captured as graphs; `processor`:
        the logits processor, if one is selected; `shared`: the engine's shared objects, whose step hooks run around
        each launch."""
        self.cfg, self.device, self.model, self.vocab, self.state = cfg, device, model, vocab, state
        self.pools = pools or {}
        self.sampler, self.planners, self.verifier, self.drafter = sampler, planners, verifier, drafter
        self.processor = processor
        self.rewinders = list(rewinders)
        self.sub_planners, self.substeps = sub_planners or {}, substeps
        self.namespaces = sorted(namespaces, key=len, reverse=True)       # the longest prefix names a group's
        self.verify_widths = tuple(verify_widths)
        seqs = cfg.max_seqs
        tokens = max(cfg.max_step_tokens, seqs * (1 + cfg.max_draft_tokens))     # a verify step: 1 + k per row
        # A step's logits rows are at most a verify row's 1 + k per sequence; masks are staged for constrained ones.
        logits_rows = seqs * (1 + cfg.max_draft_tokens)
        masks = {"mask": (logits_rows * processor.mask_width, torch.int32),
                 "mask_rows": (logits_rows, torch.int64)} if processor is not None else {}
        fields = lambda tokens: masks | state.upload_fields(seqs, tokens) | {    # noqa: E731
            "input_ids": (tokens + 1, torch.int64), "positions": (tokens, torch.int64),    # + the scratch element
            "query_start": (seqs + 1, torch.int64), "slots": (seqs, torch.int64),
            "logits_index": (tokens, torch.int64), "logits_rows": (tokens, torch.int64),   # verify rows: 1 + k each
            "start": (seqs, torch.int64), "input_at": (seqs, torch.int64), "logits_slots": (tokens, torch.int64),
            "logits_end": (tokens, torch.int64), "row_of_token": (tokens, torch.int64),
            "offset": (tokens, torch.int64), "n_query": (seqs, torch.int64), "exact": (seqs, torch.bool),
            "draft_index": (tokens, torch.int64), "draft_from": (tokens, torch.int64)} | self._sub_fields(cfg)
        self.uploader = Uploader(device, fields(tokens))
        # Steps of at most a decode or verify step's tokens (every graphed one) stage into an arena of that size, so
        # their few bytes of inputs lie together and move in one copy.
        self.small_tokens = logits_rows
        self.small_uploader = Uploader(device, fields(logits_rows)) if logits_rows < tokens else self.uploader
        # Without a drafter no row verifies, so every row's start is the host's (design §7.3): a step's positions,
        # lengths and write indices are staged from the host, and the device moves only the tokens it holds.
        self.host_starts = drafter is None
        # The device SeqTable (one row per slot, plus the padding slot): authoritative while steps are in flight.
        rows, cols = seqs + 1, max(cfg.max_draft_tokens, 1)
        self.last_token = torch.zeros(rows, dtype=torch.int64, device=device)
        self.num_computed = torch.zeros(rows, dtype=torch.int64, device=device)
        kq = getattr(drafter, "q_support", 0)               # a sampling drafter's laws: Kq entries per draft
        laws = {"draft_q_ids": torch.zeros(rows, cols, kq, dtype=torch.int64, device=device),
                "draft_q": torch.zeros(rows, cols, kq, dtype=torch.float32, device=device)} if kq else {}
        self.seq_table = SeqTable(self.num_computed, self.last_token,
                                  n_accept=torch.zeros(rows, dtype=torch.int64, device=device),
                                  draft_tokens=torch.zeros(rows, cols, dtype=torch.int64, device=device),
                                  n_draft=torch.zeros(rows, dtype=torch.int64, device=device), **laws)
        self.graphs = platform.Graphs(device) if cfg.cuda_graphs and device.type == "cuda" else None
        self.buckets = sorted({min(b, seqs) for b in BUCKETS} | {seqs})
        width = 1 + cfg.max_draft_tokens
        self._out_tokens = [platform.pinned_empty(seqs * width, torch.int64, device) for _ in range(2)]
        self._out_accepts = [platform.pinned_empty(seqs, torch.int64, device) for _ in range(2)]
        self._out_drafts = [platform.pinned_empty(seqs, torch.int64, device) for _ in range(2)]
        k = self.seq_table.draft_tokens.shape[1]
        self._out_draft_tokens = [platform.pinned_empty(seqs * k, torch.int64, device).view(seqs, k)
                                  for _ in range(2)] if processor is not None and drafter is not None else None
        self._out_turn = 0
        self.copies = Copies(device, self.pools)               # MemOps with `after` or `lands_by`
        self.writes = Writes(device)                           # what shared objects write before a launch
        self._before = [o.before_launch for o in shared if hasattr(o, "before_launch")]
        self._lenders = [o for o in shared if hasattr(o, "lend_activations")]
        self._observers = [o for o in shared if hasattr(o, "after_commit")]
        self._telemetry: list[dict[str, tuple[torch.Tensor, list[torch.Tensor]]]] | None = None   # at first launch

    def _sub_fields(self, cfg: Any) -> dict[str, tuple[int, torch.dtype]]:
        """Static buffers of each drafter sub-step: up to `max_seqs` rows of the widest declared width."""
        count, width = self.substeps
        if not self.sub_planners or not count:
            return {}
        rows, tokens = cfg.max_seqs, cfg.max_seqs * width
        out = {}
        for i in range(count):
            out |= {f"sub{i}.query_start": (rows + 1, torch.int64), f"sub{i}.slots": (rows, torch.int64),
                    f"sub{i}.rows": (tokens, torch.int64), f"sub{i}.offset": (tokens, torch.int64),
                    f"sub{i}.logits": (tokens, torch.int64), f"sub{i}.input_ids": (tokens, torch.int64),
                    f"sub{i}.positions": (tokens, torch.int64), f"sub{i}.lens": (rows, torch.int32)}
            out |= {f"sub{i}.{g}.write": (tokens, torch.int64) for g in self.sub_planners}
        return out

    # -- startup --------------------------------------------------------------------------------------------------
    def state_memory(self, ledger: Ledger, fraction: float) -> int:
        """Bytes for the state pools: `fraction` of the free memory, minus the activation peak of a worst-case step
        (`max_step_tokens` prompt tokens in rows of at most `max_model_len`, each sampled) run on minimum-size pools,
        and minus the graphs' private pool. Where a shared object lends the steps from some size their activations
        (`lend_activations`, told the worst step's peak), the worst step below that size (at least an eager step of
        every slot's decode or verify rows) is the one reserved."""
        ledger.materialize(None)
        self.state.reset_padding()
        with torch.no_grad():
            peak = platform.peak_memory(self.device, lambda: self._worst_step(self._profile_plan()))
            lends = min((t for o in self._lenders if (t := o.lend_activations(peak))), default=0)
            small = max(lends - 1, self.cfg.max_seqs * (1 + self.cfg.max_draft_tokens))   # (eager decode rows too)
            if lends and small < self.cfg.max_step_tokens:
                peak = platform.peak_memory(self.device, lambda: self._worst_step(self._profile_plan(small)))
            if self.graphs is not None:
                # Captured largest bucket first, the smaller ones reuse its blocks, so the pool is what the largest
                # bucket's variants leave reserved; every bucket's execs take the device memory the largest's do.
                graphs, self.graphs = self.graphs, platform.Graphs(self.device)
                peak += platform.reserved_growth(self.device, lambda: [
                    self._graph_step(self.buckets[-1], *variant) for variant in self._graph_variants()],
                    len(self.buckets))
                self.graphs = graphs                     # the profile's graphs go before the free memory is read
        ledger.free()
        return int(platform.free_memory(self.device) * fraction) - peak

    def _profile_plan(self, tokens: int = 0) -> BatchPlan:
        n = np.array(profile_rows(self.cfg, tokens), dtype=np.int32)
        return host_plan(0, np.arange(len(n), dtype=np.int32), np.zeros_like(n), n, np.ones_like(n),
                         np.zeros(int(n.sum()), dtype=np.int64))

    def _worst_step(self, plan: BatchPlan) -> None:
        """The largest prefill, a prompt-logprob chunk, a filtered sampling step over every slot with logprobs,
        (with mixed steps) every slot but one decoding beside a full chunk, sampled with the chunk's activations alive,
        (when drafting) a verify step over every slot's 1 + k rows, then its rewind and the drafter's run with its
        sub-steps (eager: logprob steps are never graphed), and the logits processor's masks on every one of those
        rows."""
        ctx = self._context(plan)
        self._prepare(ctx)
        h = self.model.trunk(ctx)
        width, dtype = self.model.residual
        if h.shape[1:] != (width,) or h.dtype != dtype:     # what taps and the drafter are built for (below)
            raise ValueError(f"the trunk returns [T, {h.shape[1]}] {h.dtype}, not the [T, {width}] {dtype} the model "
                             "declares (ModelSpec.residual)")
        self.model.head(h[:PROMPT_LOGPROB_CHUNK])[:LOGPROB_F32_ROWS].float().log_softmax(-1)
        if self.drafter is not None:                 # its pass over the step's tokens
            self._run_drafter(ctx, h)
        del h
        n = self.cfg.max_seqs
        for slot in range(n):
            self.sampler.admit(slot, SamplingParams(temperature=1.0, top_p=0.9, seed=0, logprobs=20))
        ones = np.ones(n, dtype=np.int32)
        decode = host_plan(0, np.arange(n, dtype=np.int32), np.zeros(n, dtype=np.int32), ones, ones,
                           np.zeros(n, dtype=np.int64))
        ctx = self._context(decode)
        self._prepare(ctx)
        self.sampler.sample(torch.zeros(n, self.vocab, device=self.device), ctx)
        if self.cfg.chunk_with_decode and self.drafter is None and n > 1:
            rows = np.array([1] * (n - 1) + [min(self.cfg.chunk_with_decode, self.cfg.max_model_len)], np.int32)
            mixed = host_plan(0, np.arange(n, dtype=np.int32), np.zeros(n, dtype=np.int32), rows, np.ones_like(rows),
                              np.zeros(int(rows.sum()), dtype=np.int64))
            mixed.n_decode = n - 1
            ctx = self._context(mixed)
            self._prepare(ctx)
            h = self.model.trunk(ctx)
            self.sampler.sample(self._logits(ctx, h), ctx)
            del h
        if self.drafter is not None:                 # the verifier's own buffers (filtered, with logprobs)
            width = np.full(n, 1 + self.cfg.max_draft_tokens, dtype=np.int32)
            verify = host_plan(0, np.arange(n, dtype=np.int32), np.zeros(n, dtype=np.int32), width, width,
                               np.zeros(int(width.sum()), dtype=np.int64))
            verify.draft_k = self.cfg.max_draft_tokens
            ctx = self._context(verify)
            self._prepare(ctx)
            self.verifier.verify(torch.zeros(int(width.sum()), self.vocab, device=self.device), ctx)
            for rewind in self.rewinders:
                rewind(ctx)
            self._run_drafter(ctx, torch.zeros(int(width.sum()), self.model.residual[0], dtype=self.model.residual[1],
                                               device=self.device))
        if self.processor is not None:
            r = n * (1 + self.cfg.max_draft_tokens)
            self.processor.apply(torch.zeros(r, self.vocab, device=self.device), self.uploader.dev["mask"].view(r, -1),
                                 torch.arange(r, device=self.device))

    def _graph_variants(self) -> list[tuple[SamplingParams, int]]:
        """(sampler params, row width) of every graph per bucket: decode rows and verify rows of each width the policy
        may ask for, for each sampler variant."""
        return [(params, width) for width in (1, *self.verify_widths) for params in _VARIANTS]

    def capture_graphs(self) -> None:
        """Capture every bucket's decode and verify graphs (after the pools' final materialize: graphs bake in their
        addresses), so serving never stops to capture."""
        if self.graphs is None:
            return
        for rows in reversed(self.buckets):          # largest first: smaller graphs reuse its pool blocks
            for variant in self._graph_variants():
                if not self._graph_step(rows, *variant):
                    self.graphs = None              # some planner's kernel cannot be captured: stay eager
                    return
        self.sampler.admit(self.state.padding_slot, _VARIANTS[0])

    @torch.no_grad()
    def _graph_step(self, rows: int, params: SamplingParams, width: int = 1) -> bool:
        """Warm up and capture one graph with every row on the padding slot (null unit, `params`): decode rows, or
        verify rows of `width` tokens with their drafter sub-steps; with a logits processor, its `apply` (over padding
        rows only: every step applies, constrained rows or not)."""
        self.sampler.admit(self.state.padding_slot, params)
        empty = np.zeros(0, dtype=np.int32)
        plan = host_plan(0, empty, empty, empty, empty, np.zeros(0, dtype=np.int64))
        plan.draft_k = width - 1
        plan = padded(plan, rows, self.state.padding_slot, width)
        ctx = self._context(plan, masks=([], np.zeros(0, dtype=np.int64)) if self.processor is not None else None)
        key = self._graph_key(ctx, width)
        if key is None:
            return False
        for _ in range(2):                          # warm-up, then capture
            self.graphs.run(key, lambda: self._decode(ctx))
        return True

    def _graph_key(self, ctx: StepContext, width: int) -> tuple | None:
        """The graph a step replays: its size and width, the sampler's variant, every planner's and sub-step's key
        (None: some kernel cannot be captured)."""
        metas = list(ctx.metas.values()) + [m for sub in (ctx.substeps.subs if ctx.substeps else []) for m in
                                            sub.metas.values()]
        keys = [getattr(meta, "graph_key", None) for meta in metas]
        if None in keys:
            return None
        return (len(ctx.batch.plan.slots), width, self.sampler.variant(ctx.batch.plan.slots), *keys)

    # -- one step -------------------------------------------------------------------------------------------------
    def launch(self, plan: BatchPlan, exact: np.ndarray, logprobs: bool, prompt: list[PromptRows] | None = None,
               masks: tuple[list[MaskRow], np.ndarray] | None = None) -> Launched:
        """Enqueue the step's device work without waiting for the device. `exact`: the rows whose start the host
        knows; `logprobs`: some sampled row reports logprobs; `prompt` (not None: a prompt-logprob step) — those are
        computed now, on the host, so that step waits for the device; `masks`: the constrained rows (the logits
        processor fills their masks) and their logits rows."""
        rows, n_sampled, step = len(plan.slots), int(plan.n_logits.sum()), plan.step
        writes, mem_ops = [], list(plan.mem_ops)        # with the shared objects' (`before_launch`)
        for before in self._before:
            w, ops = before(plan)
            writes += w
            mem_ops += ops
        width = int(plan.n_query.max(initial=1))
        uniform = bool((plan.n_query == width).all() and (plan.n_logits == width).all())   # decode or verify rows
        graphed = (self.graphs is not None and prompt is None and not logprobs        # graphs return tokens only
                   and uniform and not plan.n_decode and rows <= self.buckets[-1])      # mixed steps run eagerly
        if graphed:
            plan = padded(plan, next(b for b in self.buckets if b >= rows), self.state.padding_slot, width)
            exact = np.concatenate([exact, np.ones(len(plan.slots) - rows, dtype=bool)])   # padding: start 0
        ctx = self._context(plan, exact, masks)
        key = self._graph_key(ctx, width) if graphed else None
        out = sampled = accepts = drafts = None
        recorded: list[Logprobs] = []
        # Asynchronous MemOps go to the copy stream; this step waits for the copies landing by it, then the writes.
        self.copies.add([op for op in mem_ops if op.after is not None or op.lands_by is not None], step)
        mem_ops = [op for op in mem_ops if op.after is None and op.lands_by is None]
        self.copies.launching(step)
        self.writes.run(writes)
        # MemOps run eagerly in stream order around the forward, never inside a graph: their count varies per step.
        run_mem_ops(mem_ops, self.pools, "pre_forward")
        if key is not None:
            sampled, accepts = self.graphs.run(key, lambda: self._decode(ctx))   # verify, rewind, drafter included
        else:
            loan = next((c for o in self._lenders if (c := o.activations(plan)) is not None), None)
            with loan or contextlib.nullcontext():          # its activations a shared object's loan (if any)
                self._prepare(ctx)
                h = self.model.trunk(ctx)               # always: chunks without sampled rows still write their state
                # The head only when a row samples: a step of mid-prompt chunks has no logits rows at all.
                logits = self._logits(ctx, h) if n_sampled else None
                if prompt is not None:
                    recorded = self._prompt_logprobs(h, plan, prompt)
                if (plan.n_logits > 1).any():           # verify rows
                    out, accepts = self.verifier.verify(logits, ctx)
                    sampled = self._store(ctx, out.tokens, accepts)
                    for rewind in self.rewinders:       # past the rejected drafts, before anything drafts from it
                        rewind(ctx)
                elif plan.n_logits.any():
                    out = self.sampler.sample(logits, ctx)
                    sampled = self._store(ctx, out.tokens)
                if self.drafter is not None:
                    self._run_drafter(ctx, h)           # its state for this step's tokens, then the next step's drafts
        run_mem_ops(mem_ops, self.pools, "post_sample")
        draft_tokens = None
        if sampled is not None and self.drafter is not None:
            slots = ctx.batch.dev.slots[:rows]
            drafts = self._out_drafts[self._out_turn][:rows].copy_(self.seq_table.n_draft[slots], non_blocking=True)
            if self._out_draft_tokens is not None:   # the masks of a constrained row's next verify need the drafts
                draft_tokens = self._out_draft_tokens[self._out_turn][:rows]
                draft_tokens.copy_(self.seq_table.draft_tokens[slots], non_blocking=True)
        tokens = lps = None
        if out is not None and out.logprobs is not None:   # pinned copies, read at commit: no host sync at launch
            lps = tuple(platform.pinned_empty(t.numel(), t.dtype, self.device).view(t.shape).copy_(t, non_blocking=True)
                        for t in (out.logprobs, out.top_ids, out.top_logprobs))
        if sampled is not None:
            tokens = self._out_tokens[self._out_turn][:n_sampled]
            tokens.copy_(sampled[:n_sampled], non_blocking=True)
            if accepts is not None:
                accepts = self._out_accepts[self._out_turn][: len(accepts)].copy_(accepts, non_blocking=True)
            self._out_turn ^= 1
        observed = self._observe(step)
        done = platform.Event(self.device)
        done.record()
        self.copies.launched(step, done)
        return Launched(tokens, accepts, drafts, lps, done, recorded, key is not None, draft_tokens, observed)

    def _observe(self, step: int) -> list[tuple[Any, dict[str, torch.Tensor]]]:
        """Copy the shared objects' telemetry to this launch's pinned host buffers (two per buffer, alternating: a
        step's are read at its commit, which precedes the launch two steps later)."""
        if self._telemetry is None:
            self._telemetry = [{name: (t, [platform.pinned_empty(t.numel(), t.dtype, self.device).view(t.shape)
                                           for _ in range(2)]) for name, t in getattr(o, "telemetry", {}).items()}
                               for o in self._observers]
        return [(o.after_commit, {name: host[step % 2].copy_(t, non_blocking=True) for name, (t, host) in tel.items()})
                for o, tel in zip(self._observers, self._telemetry, strict=True)]

    def _decode(self, ctx: StepContext) -> tuple[torch.Tensor, torch.Tensor | None]:
        """A decode or verify step's device work from inputs to tokens (and accepted drafts), the rewind and the
        drafter's run — the part captured in a graph."""
        self._prepare(ctx)
        h = self.model.trunk(ctx)
        logits = self._logits(ctx, h)
        accepts = None
        if (ctx.batch.plan.n_logits > 1).any():
            out, accepts = self.verifier.verify(logits, ctx)
            tokens = self._store(ctx, out.tokens, accepts)
            for rewind in self.rewinders:
                rewind(ctx)
        else:
            tokens = self._store(ctx, self.sampler.sample(logits, ctx).tokens)
        if self.drafter is not None:
            self._run_drafter(ctx, h)
        return tokens, accepts

    def _logits(self, ctx: StepContext, h: torch.Tensor) -> torch.Tensor:
        """The step's logits rows (fp32), masked where the logits processor constrains them: every path that samples
        or verifies runs this (prefill, decode and verify; graphed and eager)."""
        plan = ctx.batch.plan
        # Every token a logits row (decode and verify steps): the head reads the trunk's output itself, which it
        # returns new tensors from (a drafter reads `h` after it).
        rows = h if (plan.n_logits == plan.n_query).all() else h[ctx.batch.dev.logits_index]
        logits = self.model.head(rows).float()
        if ctx.prep.mask is not None:
            self.processor.apply(logits, ctx.prep.mask, ctx.prep.mask_rows)
        return logits

    def _run_drafter(self, ctx: StepContext, h: torch.Tensor) -> None:
        if ctx.substeps is not None:
            ctx.substeps.reset()
        self.drafter.run(ctx, h)

    def _store(self, ctx: StepContext, tokens: torch.Tensor, accepts: torch.Tensor | None = None) -> torch.Tensor:
        """Advance the device SeqTable for the sampled rows: the next input token and the computed length — for a
        verify row, only up to the drafts it accepted (the state its rejected drafts wrote is past that length)."""
        dev, prep = ctx.batch.dev, ctx.prep
        if accepts is None:
            slots = prep.logits_slots
            self.last_token[slots] = tokens
            self.num_computed[slots] = prep.logits_end if prep.host_staged else \
                prep.lens[dev.logits_rows].long()
            if self.drafter is not None:             # the drafter reads it after every step
                self.seq_table.n_accept.index_fill_(0, slots, 0)     # device-only: capturable
            return tokens
        width = int(ctx.batch.plan.n_logits.max())
        rows = dev.logits_rows.view(-1, width)[:, 0]
        slots = dev.slots[rows]
        self.last_token[slots] = tokens.view(-1, width).gather(1, accepts[:, None])[:, 0]
        self.num_computed[slots] = ctx.prep.lens[rows].long() - (width - 1) + accepts
        self.seq_table.n_accept[slots] = accepts
        return tokens

    def _context(self, plan: BatchPlan, exact: np.ndarray | None = None,
                 masks: tuple[list[MaskRow], np.ndarray] | None = None) -> StepContext:
        """The host half of a step's inputs: index arrays, the starts the host knows exactly (`exact` rows; all by
        default) and the constrained rows' masks, staged into static device buffers with asynchronous copies, plus
        host-side planning. Positions, lengths and write indices are staged too when every start is the host's (no
        drafter), else derived on the device by `_prepare`."""
        n = plan.n_query
        rows, tokens = len(n), int(n.sum())
        qstart = np.concatenate([[0], np.cumsum(n)])
        offset = np.arange(tokens) - np.repeat(qstart[:-1], n)
        start, logits_rows = plan.start_upper.astype(np.int64), np.repeat(np.arange(rows), plan.n_logits)
        up = self.small_uploader if tokens <= self.small_tokens else self.uploader
        up.begin()
        batch = Batch(plan, DeviceBatch(
            input_ids=up.put("input_ids", plan.host_tokens),
            positions=up.dev["positions"][:tokens],
            query_start=up.put("query_start", qstart),
            slots=up.put("slots", plan.slots.astype(np.int64)),
            logits_index=up.put("logits_index", _last_tokens(qstart, plan.n_logits)),
            logits_rows=up.put("logits_rows", logits_rows),
        ))
        from_device = plan.token_src == TokenSrc.DEVICE_LAST
        inputs = up.dev["input_ids"]
        prep = _Prep(lens=up.dev["pages.lens"][:rows], start=up.put("start", start), inputs=inputs,
                     input_at=up.put("input_at", np.where(from_device, qstart[:-1], len(inputs) - 1)),
                     logits_slots=up.put("logits_slots", plan.slots[logits_rows].astype(np.int64)))
        positions = None
        if self.host_starts:
            if exact is not None and not exact.all():
                raise RuntimeError("a step without drafts has rows whose start only the device knows")
            positions = np.repeat(start, n) + offset
            up.put("positions", positions)
            up.put("pages.lens", (start + n).astype(np.int32))
            prep.host_staged, prep.logits_end = True, up.put("logits_end", (start + n)[logits_rows])
        else:
            prep.row_of_token = up.put("row_of_token", np.repeat(np.arange(rows), n))
            prep.offset = up.put("offset", offset)
            prep.n_query = up.put("n_query", n.astype(np.int64))
            prep.exact = up.put("exact", np.ones(rows, dtype=bool) if exact is None else exact)
        verify = np.flatnonzero(plan.n_logits > 1)
        if len(verify):                          # a verify row's inputs after the first are its drafts (on device)
            k = int(n[verify[0]]) - 1
            j = np.tile(np.arange(k), len(verify))
            prep.draft_index = up.put("draft_index", np.repeat(qstart[verify] + 1, k) + j)
            cols = self.seq_table.draft_tokens.shape[1]
            prep.draft_from = up.put("draft_from", np.repeat(plan.slots[verify].astype(np.int64) * cols, k) + j)
        if masks is not None:
            # Only the constrained rows' masks are staged. Their logits rows are padded with an index past the step's
            # last row up to the step's count of logits rows, which a captured graph reads in full: no index of an
            # earlier step may survive in the graph's view.
            constrained, index = masks
            width, total = self.processor.mask_width, int(plan.n_logits.sum())
            self.processor.fill(constrained, up.stage("mask", len(index) * width)[0].reshape(-1, width))
            prep.mask_rows = up.put("mask_rows", np.concatenate([index, np.full(total - len(index), total)]))
            prep.mask = up.dev["mask"][: len(prep.mask_rows) * width].view(-1, width)
        pages = self.state.pages(plan, up, positions)
        drafting = plan.draft_k and plan.n_logits.any() and hasattr(self.drafter, "substeps")
        widths = self.drafter.substeps(plan.draft_k) if drafting else []
        subs = [self._substep(up, plan, pages, i, q) for i, q in enumerate(widths)]
        up.end()
        # An auxiliary model's planners see the step's in-step publishes of its own specs only, under local names.
        local = {ns: local_batch(batch, ns) for ns in self.namespaces}
        metas = {g: p.plan_bounds(local.get(self._namespace(g), batch), pages.get(g)) for g, p in self.planners.items()}
        for i, sub in enumerate(subs):
            sub.metas = {g: p[i].plan_bounds(sub.batch, sub.pages[g]) for g, p in self.sub_planners.items()}
        ahead = self.drafter.lookahead(plan.draft_k) if subs and hasattr(self.drafter, "lookahead") else 0
        return StepContext(batch, metas, seq_table=self.seq_table, pages=pages, prep=prep, pools=self.pools,
                           substeps=_Substeps(self, subs, ahead) if subs else None, sampler=self.sampler)

    def _namespace(self, group: str) -> str | None:
        return next((ns for ns in self.namespaces if group.startswith(ns + ".")), None)

    def _substep(self, up: Uploader, plan: BatchPlan, pages: dict[str, Any], i: int, q: int) -> StepContext:
        """The host half of drafter sub-step `i`: q tokens for each sampled row of `plan`, over the auxiliary state
        groups; positions, lengths and write indices are derived on the device when the drafter asks for it."""
        if i >= self.substeps[0] or q > self.substeps[1]:
            raise RuntimeError(f"sub-step {i} of width {q}: beyond what the drafter declared ({self.substeps})")
        f = f"sub{i}."
        slots = plan.slots[plan.n_logits > 0]
        rows, tokens = len(slots), len(slots) * q
        sub_plan = host_plan(plan.step, slots, plan.start_upper[plan.n_logits > 0] + int(plan.n_query.max()) + q
                             + (self.drafter.lookahead(plan.draft_k) if hasattr(self.drafter, "lookahead") else 0),
                             np.full(rows, q, dtype=np.int32), np.full(rows, q, dtype=np.int32),
                             np.zeros(tokens, dtype=np.int64))
        batch = Batch(sub_plan, DeviceBatch(
            input_ids=up.dev[f + "input_ids"][:tokens], positions=up.dev[f + "positions"][:tokens],
            query_start=up.put(f + "query_start", np.arange(rows + 1) * q),
            slots=up.put(f + "slots", slots.astype(np.int64)),
            logits_index=up.put(f + "logits", np.arange(tokens)),
            logits_rows=up.put(f + "rows", np.repeat(np.arange(rows), q))))
        sub_pages = {g: Pages(pages[g].page, pages[g].table, pages[g].host_table, up.dev[f + "lens"][:rows],
                              up.dev[f"{f}{g}.write"][:tokens]) for g in self.sub_planners}
        offset = up.put(f + "offset", np.tile(np.arange(q), rows))
        return StepContext(batch, {}, seq_table=self.seq_table, pages=sub_pages, prep=offset, pools=self.pools,
                           sampler=self.sampler)

    def _prepare_substep(self, sub: StepContext, i: int, offset: int) -> None:
        """The device half of a sub-step (inside the step's graph): positions from the advanced SeqTable."""
        dev, q = sub.batch.dev, int(sub.batch.plan.n_query[0]) if len(sub.batch.plan.n_query) else 0
        start = self.num_computed[dev.slots] + offset
        positions = start[dev.logits_rows] + sub.prep
        dev.positions.copy_(positions.clamp(max=self.cfg.max_model_len - 1))
        slot_of_token = dev.slots[dev.logits_rows]
        for g, pages in sub.pages.items():
            pages.lens.copy_(start + q)
            unit = pages.table[slot_of_token, positions // pages.page].long()
            pages.write_index.copy_(unit * pages.page + positions % pages.page)
            self.sub_planners[g][i].prepare_device(sub.metas[g], sub)

    def _prepare(self, ctx: StepContext) -> None:
        """The device half: inputs still on the device and the SeqTable's starts, then (unless the host staged them)
        positions, lengths and write indices — all into static buffers, so this runs inside a captured graph too."""
        dev, prep = ctx.batch.dev, ctx.prep
        # A row's first input is the device's last token where it holds it (the other rows write the scratch element).
        prep.inputs[prep.input_at] = self.last_token[dev.slots]
        if prep.host_staged:
            self.num_computed[dev.slots] = prep.start
        else:
            self._derive(ctx)
        for group, planner in self.planners.items():
            planner.prepare_device(ctx.metas[group], ctx)

    def _derive(self, ctx: StepContext) -> None:
        """Positions, lengths and write indices from the SeqTable's starts, for steps whose starts the device may hold
        (the host's where `exact`), and a verify row's drafts."""
        dev, prep = ctx.batch.dev, ctx.prep
        self.num_computed[dev.slots] = torch.where(prep.exact, prep.start, self.num_computed[dev.slots])
        start = self.num_computed[dev.slots]
        # Columns past the length limit (a verify row's padding near it) read and write the block tables' overflow
        # columns, which point at the null unit, and take the last position for position-indexed caches (rope); they
        # are never accepted.
        limit = self.cfg.max_model_len
        positions = start[prep.row_of_token] + prep.offset
        dev.positions.copy_(positions.clamp(max=limit - 1) if self.drafter is not None else positions)
        prep.lens.copy_(start + prep.n_query)
        if prep.draft_index is not None:
            # A verify row's drafts past its `n_draft` (capped so its tokens stay within the limit) are token 0.
            table, cols = self.seq_table, self.seq_table.draft_tokens.shape[1]
            slot, j = prep.draft_from // cols, prep.draft_from % cols
            n_draft = torch.minimum(table.n_draft[slot], (limit - 2 - self.num_computed[slot]).clamp(min=0))
            table.n_draft[slot] = n_draft
            drafts = table.draft_tokens.view(-1)[prep.draft_from]
            dev.input_ids[prep.draft_index] = torch.where(j < n_draft, drafts, torch.zeros_like(drafts))
        slot_of_token = dev.slots[prep.row_of_token]
        for pages in ctx.pages.values():
            unit = pages.table[slot_of_token, positions // pages.page].long()
            pages.write_index.copy_(unit * pages.page + positions % pages.page)

    def _prompt_logprobs(self, h: torch.Tensor, plan: BatchPlan, prompt: list[PromptRows],
                         chunk: int = PROMPT_LOGPROB_CHUNK) -> list[Logprobs]:
        """The prompt logprobs `prompt` asks for, from the trunk's hidden states in row chunks (no full-vocab logits
        for long prompts)."""
        qstart = np.concatenate([[0], np.cumsum(plan.n_query)])
        out = []
        for r in prompt:
            acc, base = Logprobs([], [], []), int(qstart[r.row])
            for a in range(r.first, r.end, chunk):
                logits = self.model.head(h[base + a: base + min(a + chunk, r.end)])
                for c in range(a, min(a + chunk, r.end), LOGPROB_F32_ROWS):
                    b = min(c + LOGPROB_F32_ROWS, a + chunk, r.end)
                    lp = logits[c - a: b - a].float().log_softmax(-1)
                    target = torch.as_tensor(r.targets[c - r.first: b - r.first], device=lp.device)
                    top = lp.topk(r.k)
                    acc.token += lp.gather(-1, target[:, None])[:, 0].tolist()
                    acc.top_ids += top.indices.tolist()
                    acc.top += top.values.tolist()
                    del lp, top                     # (each piece freed before the next: the profile's peak)
                del logits
            out.append(acc)
        return out
