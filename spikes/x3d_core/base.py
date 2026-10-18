"""The step loop's device prep as it was before step 3d (bdd8422's `core/holon/core/executor.py`, verbatim): the
`base` arm of `ab.py`, bound onto a live executor with its own uploader of the old layout.

Every step's positions, lengths and write indices are derived on the device from the SeqTable, its first inputs are
chosen by a `where` over the step's first tokens, the store gathers the logits rows' slots, and the head reads the
logits rows through a gather.
Also bdd8422's `Uploader` (fields laid out in the order given, one arena for every step), for the arms before the
`upload` change: `uploader` (bdd8422's fields) and `prep_uploader` (the `prep` change's fields).
"""
from __future__ import annotations

import dataclasses
import types

import numpy as np
import torch
from holon.api import Batch, BatchPlan, DeviceBatch, MaskRow, TokenSrc
from holon.core.build import StepContext, local_batch
from holon.core.executor import _last_tokens, _Substeps
from collections.abc import Callable

from holon.core import platform


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
    mask: torch.Tensor | None = None          # [M, mask_width] the logits processor's masks of the constrained rows
    mask_rows: torch.Tensor | None = None     # [M] their logits rows; past the step's logits rows: padding


_ALIGN = 64
_MERGE_GAP = 4096        # bytes: copying a small unwritten gap is cheaper than another copy launch


class Uploader:
    def __init__(self, device: torch.device, fields: dict[str, tuple[int, torch.dtype]]) -> None:
        """`fields`: name -> (maximum length, dtype)."""
        self._layout: dict[str, tuple[int, int, torch.dtype]] = {}       # name -> (byte offset, length, dtype)
        size = 0
        for name, (n, dtype) in fields.items():
            self._layout[name] = (size, n, dtype)
            size += -(-n * dtype.itemsize // _ALIGN) * _ALIGN
        self._device_arena = torch.zeros(size, dtype=torch.uint8, device=device)
        self._host_arenas = [platform.pinned_empty(size, torch.uint8, device) for _ in range(2)]
        self.dev = {name: self._view(self._device_arena, name) for name in fields}
        self._host = [{name: self._view(arena, name).numpy() for name in fields} for arena in self._host_arenas]
        self._done = [platform.Event(device), platform.Event(device)]
        self._after: list[Callable[[], None]] = []
        self._written: list[tuple[int, int]] = []
        self._turn = 0

    def _view(self, arena: torch.Tensor, name: str) -> torch.Tensor:
        offset, n, dtype = self._layout[name]
        return arena[offset: offset + n * dtype.itemsize].view(dtype)

    def begin(self) -> None:
        """Start a step: switch to the other host arena once its previous copy has landed."""
        self._turn ^= 1
        self._done[self._turn].synchronize()
        self._written.clear()

    def put(self, name: str, values: np.ndarray) -> torch.Tensor:
        """Stage `values`; returns the device view that holds them once `end()` has run."""
        host, dev = self.stage(name, len(values))
        host[:] = values
        return dev

    def stage(self, name: str, n: int) -> tuple[np.ndarray, torch.Tensor]:
        """The host view of `name`'s first `n` values, for the caller to write in full before `end()`, and the
        device view that holds them afterwards."""
        offset, _, dtype = self._layout[name]
        if n:
            self._written.append((offset, offset + n * dtype.itemsize))
        return self._host[self._turn][name][:n], self.dev[name][:n]

    def after(self, fn: Callable[[], None]) -> None:
        """Run `fn` (device work reading staged inputs) right after this step's copy is enqueued."""
        self._after.append(fn)

    def end(self) -> None:
        host = self._host_arenas[self._turn]
        for lo, hi in _merged(self._written):
            self._device_arena[lo:hi].copy_(host[lo:hi], non_blocking=True)
        self._done[self._turn].record()
        for fn in self._after:
            fn()
        self._after.clear()


def _merged(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for lo, hi in sorted(ranges):
        if out and lo - out[-1][1] <= _MERGE_GAP:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return [(lo, hi) for lo, hi in out]


def uploader(ex) -> Uploader:
    """An uploader with bdd8422's fields, in its order (so its copies are the ones it made)."""
    cfg, seqs = ex.cfg, ex.cfg.max_seqs
    if ex.processor is not None:
        raise NotImplementedError("the base arm stages no masks")
    tokens = max(cfg.max_step_tokens, seqs * (1 + cfg.max_draft_tokens))
    return Uploader(ex.device, ex.state.upload_fields(seqs, tokens) | {
        "input_ids": (tokens, torch.int64), "positions": (tokens, torch.int64),
        "query_start": (seqs + 1, torch.int64), "slots": (seqs, torch.int64),
        "logits_index": (tokens, torch.int64), "logits_rows": (tokens, torch.int64),
        "row_of_token": (tokens, torch.int64), "offset": (tokens, torch.int64), "n_query": (seqs, torch.int64),
        "exact": (seqs, torch.bool), "start": (seqs, torch.int64), "from_device": (seqs, torch.bool),
        "draft_index": (tokens, torch.int64), "draft_from": (tokens, torch.int64)} | ex._sub_fields(cfg))


def prep_uploader(ex) -> Uploader:
    """An uploader with the `prep` change's fields (b60328f), in its order, one arena for every step."""
    cfg, seqs = ex.cfg, ex.cfg.max_seqs
    tokens = max(cfg.max_step_tokens, seqs * (1 + cfg.max_draft_tokens))
    return Uploader(ex.device, ex.state.upload_fields(seqs, tokens) | {
        "input_ids": (tokens + 1, torch.int64), "positions": (tokens, torch.int64),
        "query_start": (seqs + 1, torch.int64), "slots": (seqs, torch.int64),
        "logits_index": (tokens, torch.int64), "logits_rows": (tokens, torch.int64),
        "start": (seqs, torch.int64), "input_at": (seqs, torch.int64), "logits_slots": (tokens, torch.int64),
        "logits_end": (tokens, torch.int64), "row_of_token": (tokens, torch.int64),
        "offset": (tokens, torch.int64), "n_query": (seqs, torch.int64), "exact": (seqs, torch.bool),
        "draft_index": (tokens, torch.int64), "draft_from": (tokens, torch.int64)} | ex._sub_fields(cfg))


def methods(ex) -> dict[str, types.MethodType]:
    """bdd8422's `_context`, `_prepare`, `_store` and `_logits`, bound to `ex`."""
    return {name: types.MethodType(fn, ex) for name, fn in
            (("_context", _context), ("_prepare", _prepare), ("_store", _store), ("_logits", _logits))}


def _logits(self, ctx: StepContext, h: torch.Tensor) -> torch.Tensor:
    """The step's logits rows (fp32), masked where the logits processor constrains them: every path that samples
    or verifies runs this (prefill, decode and verify; graphed and eager)."""
    logits = self.model.head(h[ctx.batch.dev.logits_index]).float()
    if ctx.prep.mask is not None:
        self.processor.apply(logits, ctx.prep.mask, ctx.prep.mask_rows)
    return logits


def _store(self, ctx: StepContext, tokens: torch.Tensor, accepts: torch.Tensor | None = None) -> torch.Tensor:
    """Advance the device SeqTable for the sampled rows: the next input token and the computed length — for a
    verify row, only up to the drafts it accepted (the state its rejected drafts wrote is past that length)."""
    dev = ctx.batch.dev
    if accepts is None:
        slots = dev.slots[dev.logits_rows]
        self.last_token[slots] = tokens
        self.num_computed[slots] = ctx.prep.lens[dev.logits_rows].long()
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
    host-side planning. Positions, lengths and write indices are derived on the device by `_prepare`."""
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
    if masks is not None:
        # Only the constrained rows' masks are staged. Their logits rows are padded with an index past the step's
        # last row up to the step's count of logits rows, which a captured graph reads in full: no index of an
        # earlier step may survive in the graph's view.
        constrained, index = masks
        width, total = self.processor.mask_width, int(plan.n_logits.sum())
        self.processor.fill(constrained, up.stage("mask", len(index) * width)[0].reshape(-1, width))
        prep.mask_rows = up.put("mask_rows", np.concatenate([index, np.full(total - len(index), total)]))
        prep.mask = up.dev["mask"][: len(prep.mask_rows) * width].view(-1, width)
    pages = self.state.pages(plan, up)
    drafting = plan.draft_k and plan.n_logits.any() and hasattr(self.drafter, "substeps")
    widths = self.drafter.substeps(plan.draft_k) if drafting else []
    subs = [self._substep(plan, pages, i, q) for i, q in enumerate(widths)]
    up.end()
    # An auxiliary model's planners see the step's in-step publishes of its own specs only, under local names.
    local = {ns: local_batch(batch, ns) for ns in self.namespaces}
    metas = {g: p.plan_bounds(local.get(self._namespace(g), batch), pages.get(g)) for g, p in self.planners.items()}
    for i, sub in enumerate(subs):
        sub.metas = {g: p[i].plan_bounds(sub.batch, sub.pages[g]) for g, p in self.sub_planners.items()}
    ahead = self.drafter.lookahead(plan.draft_k) if subs and hasattr(self.drafter, "lookahead") else 0
    return StepContext(batch, metas, seq_table=self.seq_table, pages=pages, prep=prep, pools=self.pools,
                       substeps=_Substeps(self, subs, ahead) if subs else None, sampler=self.sampler)


def _prepare(self, ctx: StepContext) -> None:
    """The device half: the SeqTable's starts for host-exact rows, then positions, lengths, write indices and
    inputs still on the device — all into static buffers, so this runs inside a captured graph too."""
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
    first = dev.query_start[:-1]
    dev.input_ids[first] = torch.where(prep.from_device, self.last_token[dev.slots], dev.input_ids[first])
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
    for group, planner in self.planners.items():
        planner.prepare_device(ctx.metas[group], ctx)
