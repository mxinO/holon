"""The device side of a step (design §7.3, §7.4): inputs staged into static buffers and finished on the device from
the device SeqTable, the forward, sampling or verification, drafting, and CUDA graph replay of decode steps — plus
the startup work that needs the built model: the worst-case step's memory and graph capture.

The engine owns requests, scheduling and commits; it hands the executor a scheduled plan and gets back the pinned
host copies of the step's results and the event that says they have landed.
"""
from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import torch

from holon.api import Batch, BatchPlan, DeviceBatch, Logprobs, SamplingParams, SeqTable, TokenSrc

from . import platform
from .batching import host_plan, padded
from .build import StepContext
from .inputs import Uploader
from .memory import Ledger
from .state import StateManager

PROMPT_LOGPROB_CHUNK = 256     # rows of full-vocabulary logits materialized at once for prompt logprobs
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
    logprobs: tuple[torch.Tensor, ...] | None   # sampled tokens' logprob, top ids, top logprobs
    done: platform.Event
    prompt_logprobs: list[Logprobs] = dataclasses.field(default_factory=list)   # per PromptRows, on the host now


@dataclasses.dataclass
class _Prep:
    """Static device inputs of a step's device prep (`Executor._prepare`)."""

    row_of_token: torch.Tensor     # [T] batch row of each token
    offset: torch.Tensor           # [T] index of each token within its row
    n_query: torch.Tensor          # [B]
    exact: torch.Tensor            # [B] the host knows this row's start exactly
    start: torch.Tensor            # [B] that start (ignored where not exact)
    from_device: torch.Tensor      # [B] the row's first input token is the device's last token
    lens: torch.Tensor             # [B] out: tokens held after the step (shared with every page group's Pages)
    draft_index: torch.Tensor | None = None   # [R*k] verify rows: input positions of the drafts
    draft_from: torch.Tensor | None = None    # [R*k] where each draft is in the SeqTable (flat)


def _last_tokens(qstart: np.ndarray, n_logits: np.ndarray) -> np.ndarray:
    """Flat indices of each row's last `n_logits` tokens (the tokens whose logits a step computes)."""
    ends = np.repeat(qstart[1:], n_logits)
    first = np.repeat(np.cumsum(n_logits) - n_logits, n_logits)          # where each row's run starts in `ends`
    return ends - n_logits.repeat(n_logits) + (np.arange(len(ends)) - first)


class Executor:
    def __init__(self, cfg: Any, device: torch.device, model: Any, vocab: int, sampler: Any, state: StateManager,
                 planners: dict[str, Any], verifier: Any = None, drafter: Any = None) -> None:
        """`cfg`: the EngineConfig (sizes, graphs on or off)."""
        self.cfg, self.device, self.model, self.vocab, self.state = cfg, device, model, vocab, state
        self.sampler, self.planners, self.verifier, self.drafter = sampler, planners, verifier, drafter
        seqs = cfg.max_seqs
        tokens = max(cfg.max_step_tokens, seqs * (1 + cfg.max_draft_tokens))     # a verify step: 1 + k per row
        self.uploader = Uploader(device, state.upload_fields(seqs, tokens) | {
            "input_ids": (tokens, torch.int64), "positions": (tokens, torch.int64),
            "query_start": (seqs + 1, torch.int64), "slots": (seqs, torch.int64),
            "logits_index": (tokens, torch.int64), "logits_rows": (tokens, torch.int64),   # verify rows: 1 + k each
            "row_of_token": (tokens, torch.int64), "offset": (tokens, torch.int64), "n_query": (seqs, torch.int64),
            "exact": (seqs, torch.bool), "start": (seqs, torch.int64), "from_device": (seqs, torch.bool),
            "draft_index": (tokens, torch.int64), "draft_from": (tokens, torch.int64)})
        # The device SeqTable (one row per slot, plus the padding slot): authoritative while steps are in flight.
        rows = seqs + 1
        self.last_token = torch.zeros(rows, dtype=torch.int64, device=device)
        self.num_computed = torch.zeros(rows, dtype=torch.int64, device=device)
        self.seq_table = SeqTable(self.num_computed, self.last_token,
                                  n_accept=torch.zeros(rows, dtype=torch.int64, device=device),
                                  draft_tokens=torch.zeros(rows, max(cfg.max_draft_tokens, 1), dtype=torch.int64,
                                                           device=device))
        self.graphs = platform.Graphs(device) if cfg.cuda_graphs and device.type == "cuda" else None
        self.buckets = sorted({min(b, seqs) for b in BUCKETS} | {seqs})
        width = 1 + cfg.max_draft_tokens
        self._out_tokens = [platform.pinned_empty(seqs * width, torch.int64, device) for _ in range(2)]
        self._out_accepts = [platform.pinned_empty(seqs, torch.int64, device) for _ in range(2)]
        self._out_turn = 0

    # -- startup --------------------------------------------------------------------------------------------------
    def state_memory(self, ledger: Ledger, fraction: float) -> int:
        """Bytes for the state pools: `fraction` of the free memory, minus the activation peak of a worst-case step
        (`max_step_tokens` prompt tokens in rows of at most `max_model_len`, each sampled) run on minimum-size pools,
        and minus the graphs' private pool."""
        cfg = self.cfg
        rows = [cfg.max_model_len] * (cfg.max_step_tokens // cfg.max_model_len)
        rows = (rows + [cfg.max_step_tokens % cfg.max_model_len])[:cfg.max_seqs]
        n = np.array([r for r in rows if r], dtype=np.int32)
        plan = host_plan(0, np.arange(len(n), dtype=np.int32), np.zeros_like(n), n, np.ones_like(n),
                         np.zeros(int(n.sum()), dtype=np.int64))
        ledger.materialize(None)
        self.state.reset_padding()
        with torch.no_grad():
            peak = platform.peak_memory(self.device, lambda: self._worst_step(plan))
            if self.graphs is not None:
                # Captured largest bucket first, the smaller ones reuse its blocks, so the pool is what the largest
                # bucket's variants leave reserved.
                graphs, self.graphs = self.graphs, platform.Graphs(self.device)
                peak += platform.reserved_growth(self.device, lambda: [
                    self._graph_step(self.buckets[-1], params) for params in _VARIANTS])
                self.graphs = graphs
        ledger.free()
        return int(platform.free_memory(self.device) * fraction) - peak

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
        self.sampler.sample(torch.zeros(n, self.vocab, device=self.device), ctx)
        if self.cfg.max_draft_tokens:
            torch.zeros(n * (1 + self.cfg.max_draft_tokens), self.vocab, device=self.device).argmax(-1)

    def capture_graphs(self) -> None:
        """Capture every decode bucket for each sampler variant (after the pools' final materialize: graphs bake in
        their addresses), so serving never stops to capture."""
        if self.graphs is None:
            return
        for rows in reversed(self.buckets):          # largest first: smaller graphs reuse its pool blocks
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

    # -- one step -------------------------------------------------------------------------------------------------
    def launch(self, plan: BatchPlan, exact: np.ndarray, logprobs: bool, prompt: list[PromptRows] | None = None
               ) -> Launched:
        """Enqueue the step's device work without waiting for the device. `exact`: the rows whose start the host
        knows; `logprobs`: some sampled row reports logprobs; `prompt` (not None: a prompt-logprob step) — those are
        computed now, on the host, so that step waits for the device."""
        rows, n_sampled = len(plan.slots), int(plan.n_logits.sum())
        graphed = (self.graphs is not None and prompt is None and not logprobs        # graphs return tokens only
                   and bool((plan.n_query == 1).all()) and rows <= self.buckets[-1])
        if graphed:
            plan = padded(plan, next(b for b in self.buckets if b >= rows), self.state.padding_slot)
            exact = np.concatenate([exact, np.ones(len(plan.slots) - rows, dtype=bool)])   # padding: start 0
        ctx = self._context(plan, exact)
        out = sampled = accepts = None
        recorded: list[Logprobs] = []
        if graphed:
            keys = [meta.graph_key for meta in ctx.metas.values()]
            key = (len(plan.slots), self.sampler.variant(plan.slots), *keys)
            sampled = self.graphs.run(key, lambda: self._decode(ctx))
        else:
            self._prepare(ctx)
            if prompt is not None:
                h = self.model.trunk(ctx)
                logits = self.model.head(h[ctx.batch.dev.logits_index])
                recorded = self._prompt_logprobs(h, plan, prompt)
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
        done = platform.Event(self.device)
        done.record()
        return Launched(tokens, accepts, lps, done, recorded)

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

    def _prompt_logprobs(self, h: torch.Tensor, plan: BatchPlan, prompt: list[PromptRows],
                         chunk: int = PROMPT_LOGPROB_CHUNK) -> list[Logprobs]:
        """The prompt logprobs `prompt` asks for, from the trunk's hidden states in row chunks (no full-vocab logits
        for long prompts)."""
        qstart = np.concatenate([[0], np.cumsum(plan.n_query)])
        out = []
        for r in prompt:
            acc, base = Logprobs([], [], []), int(qstart[r.row])
            for a in range(r.first, r.end, chunk):
                b = min(a + chunk, r.end)
                lp = self.model.head(h[base + a: base + b]).float().log_softmax(-1)
                target = torch.as_tensor(r.targets[a - r.first: b - r.first], device=lp.device)
                top = lp.topk(r.k)
                acc.token += lp.gather(-1, target[:, None])[:, 0].tolist()
                acc.top_ids += top.indices.tolist()
                acc.top += top.values.tolist()
            out.append(acc)
        return out
