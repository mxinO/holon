"""`qwen4_exp.ple@1`: Qwen3.8-Flash-Next's PLE (docs/v1-flash-next.md D4), the `pre` unit of `std.hyper.gr_block@1`
at layer 1: it updates the residual R [T, n·H] in place (`R += G + silu(conv(norm(G)))`, the ops in `kernels.py`).

Its state is declared by the block (C2, C3): per request, the last two token ids before the step (`ple.history`,
int64) and the conv history (`ple.conv`, the last (K−1)·dilation rows of C, float32), two Slot specs of kind
`recurrent@1` in one group, whose `alignment` and `in_step` the recipe sets to GDN's 64 so PLE's checkpoints sit on
GDN's blocks. In-step checkpoints are filled here, from the state before the step and the step's own ids and C (the
ops return C). Verify rows (GDN's way, D4) run on a scratch copy of their rows' state, so the real state stays as it
was before the step; `rewind` writes the window and the two ids after each row's accepted tokens — the same bits 1 + n
one-token steps would leave (every op is per token)."""
from __future__ import annotations

import dataclasses
from typing import Any

import torch
from holon.api import Ref, Slot, StateSpec
from torch import nn

from . import kernels

LINEAR = "std.ops.linear@1"
GROUP, HISTORY, CONV = "ple", "ple.history", "ple.conv"
TABLE = Ref("std.ops.embedding@1")      # the n-gram table (std.tiered.rows@1 serves it from RAM or SSD later: D5)


@dataclasses.dataclass(eq=False)
class PLEMeta:
    slots: torch.Tensor
    query_start: torch.Tensor
    verify: bool
    width: int
    graph_key: Any
    publishes: list = dataclasses.field(default_factory=list)   # (spec, row, its first token, offset, pool, unit)
    has_init: torch.Tensor | None = None


class PLEPlanner:
    mixes_row_kinds = True

    def plan_bounds(self, batch, pages) -> PLEMeta:
        plan = batch.plan
        lens = plan.n_query.tolist()
        decode = bool(lens) and all(t == 1 for t in lens) and not plan.n_decode
        verify = bool(lens) and not decode and not plan.n_decode and bool((plan.n_logits == plan.n_query).all())
        key = ("ple", len(lens)) if decode else ("ple.verify", len(lens), lens[0]) if verify else None
        meta = PLEMeta(batch.dev.slots, batch.dev.query_start, verify, lens[0] if verify else 0, key)
        starts = [0, *plan.n_query.cumsum().tolist()]
        meta.publishes = [(p.spec, p.row, starts[p.row], p.offset, p.payload.pool, p.payload.units[0])
                          for p in plan.publishes if p.spec in (HISTORY, CONV)]
        return meta

    def prepare_device(self, meta: PLEMeta, ctx) -> None:
        dev = ctx.batch.dev
        meta.has_init = dev.positions[dev.query_start[:-1]] > 0


class PLE(nn.Module):
    @classmethod
    def state_specs(cls, params: dict, layers: tuple[int, ...], ctx) -> list[StateSpec]:
        if len(layers) != 1:
            raise ValueError("qwen4_exp.ple: one layer")
        width = params["width"]
        window = (params.get("kernel", 4) - 1) * params["dilation"]
        common = {"alignment": params.get("alignment", 64), "page_group": GROUP, "in_step": params.get("in_step", 64)}
        return [StateSpec(HISTORY, "recurrent@1", Slot(16), layers, "ple.history.int64", **common),
                StateSpec(CONV, "recurrent@1", Slot(window * width * 4), layers, "ple.conv.f32", **common)]

    @classmethod
    def planner(cls, params: dict, specs: list[StateSpec], ctx) -> PLEPlanner:
        return PLEPlanner()

    def __init__(self, ctx, *, layer: int, hidden: int, n: int, width: int, embed_dim: int, ngram: int,
                 heads_per_ngram: int, eos: int, multipliers: tuple[int, ...], sizes: tuple[int, ...],
                 offsets: tuple[int, ...], table_rows: int, dilation: int, kernel: int = 4, eps: float = 1e-6,
                 alignment: int = 64, in_step: int = 64, table: Ref = TABLE) -> None:
        super().__init__()
        heads = (ngram - 1) * heads_per_ngram
        if width != n * hidden:
            raise ValueError(f"qwen4_exp.ple: width {width} is not n · hidden ({n} · {hidden})")
        self.n, self.hidden, self.heads, self.window, self.eps = n, hidden, heads, (kernel - 1) * dilation, eps
        self.table = ctx.unit(table, vocab=table_rows, hidden=embed_dim // heads)
        self.key_proj = ctx.unit(Ref(LINEAR, in_features=embed_dim, out_features=width))
        self.value_proj = ctx.unit(Ref(LINEAR, in_features=embed_dim, out_features=hidden))
        self.norm_key, self.norm_query, self.norm_conv = (
            nn.Parameter(torch.empty(width, dtype=torch.float32), requires_grad=False) for _ in range(3))
        self.conv = nn.Parameter(torch.empty(width, kernel, dtype=torch.float32), requires_grad=False)
        site = f"qwen4_exp.ple[layer={layer}]"
        hashes = {"ngram": ngram, "heads_per_ngram": heads_per_ngram, "eos": eos}
        self._rows = ctx.select_kernel(kernels.ROWS, site=site, **hashes).impl(ctx, **hashes)
        facts = {"streams": n, "hidden": hidden, "kernel": kernel, "dilation": dilation, "eps": eps,
                 "dtype": str(ctx.dtype).removeprefix("torch.")}
        self._block = ctx.select_kernel(kernels.BLOCK, site=site, **facts).impl(ctx, **facts)
        self.hashes = ctx.static("qwen4_exp.ple.hashes", (3, heads), torch.int64)    # multipliers (padded), sizes,
        with torch.device("cpu"):                                                    # offsets (a build runs under
            hashes = torch.zeros(3, heads, dtype=torch.int64)                        # the meta device)
            hashes[0, : len(multipliers)] = torch.tensor(multipliers, dtype=torch.int64)
            hashes[1], hashes[2] = torch.tensor(sizes), torch.tensor(offsets)
        self.hashes.copy_(hashes)
        self.ngram, self.eos = ngram, eos
        history, conv = ctx.state(HISTORY, layer), ctx.state(CONV, layer)
        self._state = None if history is None else (history[1], conv[1])
        if self._state is not None:
            rows_max = ctx.options["max_seqs"] + 1
            tokens = max(ctx.options["max_step_tokens"], rows_max * (1 + ctx.options["max_draft_tokens"]))
            self._saved = (ctx.static("qwen4_exp.ple.saved.ids", (tokens,), torch.int64),
                           ctx.static("qwen4_exp.ple.saved.c", (tokens, width), torch.float32),
                           ctx.static("qwen4_exp.ple.saved.history", (rows_max, 2), torch.int64),
                           ctx.static("qwen4_exp.ple.saved.window", (rows_max, self.window, width), torch.float32))

    def _views(self, history_pool=None, conv_pool=None) -> tuple[torch.Tensor, torch.Tensor]:
        """[units, 2] int64 histories and [units, window, n·H] f32 conv windows: the specs' pools, or payload pools of
        their layout."""
        h, c = history_pool or self._state[0], conv_pool or self._state[1]
        return (h.view(torch.int64, (h.capacity, 2)),
                c.view(torch.float32, (c.capacity, self.window, self.n * self.hidden)))

    def _run(self, r, ids, history, state, slots, query_start, has_init) -> torch.Tensor:
        """Both ops over the step's rows; returns C."""
        rows = self._rows(ids, history, slots, query_start, has_init, self.hashes[0, : self.ngram], self.hashes[1],
                          self.hashes[2])
        e = self.table(rows.reshape(-1)).reshape(len(ids), -1)
        return self._block(self.key_proj(e), self.value_proj(e), r, self.norm_key, self.norm_query, self.norm_conv,
                           self.conv, state, slots, query_start, has_init)

    def forward(self, r: torch.Tensor, ctx) -> torch.Tensor:
        dev = ctx.batch.dev
        if self._state is None:                    # a model-only build: every sequence from a fresh state
            rows = len(dev.query_start) - 1
            history = torch.zeros(rows, 2, dtype=torch.int64, device=r.device)
            state = torch.zeros(rows, self.window, r.shape[1], dtype=torch.float32, device=r.device)
            fresh = torch.zeros(rows, dtype=torch.bool, device=r.device)
            self._run(r, dev.input_ids, history, state, torch.arange(rows, device=r.device), dev.query_start, fresh)
            return r
        meta = ctx.meta(GROUP)
        history, state = self._views()
        slots, t = meta.slots.long(), r.shape[0]
        if meta.verify:                            # a scratch copy of the rows' state; the real one stays S0
            ids_s, c_s, history_s, window_s = self._saved
            rows = len(slots)
            history_s[:rows].copy_(history[slots])
            window_s[:rows].copy_(state[slots])
            scratch_h, scratch_c = history_s[:rows].clone(), window_s[:rows].clone()
            ids_s[:t].copy_(dev.input_ids)
            c_s[:t].copy_(self._run(r, dev.input_ids, scratch_h, scratch_c, torch.arange(rows, device=r.device),
                                    meta.query_start, meta.has_init))
            return r
        if meta.publishes:                         # the state before the step: in-step checkpoints start from it
            before_h, before_c = history[slots].clone(), state[slots].clone()
        c = self._run(r, dev.input_ids, history, state, meta.slots, meta.query_start, meta.has_init)
        for spec, row, start, offset, pool, unit in meta.publishes:      # a prefill step's checkpoints (eager)
            ph, pc = self._views(ctx.pool(pool) if spec == HISTORY else None, ctx.pool(pool) if spec == CONV else None)
            if spec == HISTORY:
                ph[unit] = torch.cat([before_h[row], dev.input_ids[start: start + offset]])[-2:]
            else:
                pc[unit] = torch.cat([before_c[row], c[start: start + offset]])[-self.window:]
        return r

    def rewind(self, ctx) -> None:
        """After a verify step: each row's window and two ids after its first 1 + n tokens (n its accepted drafts)."""
        if self._state is None or not (meta := ctx.meta(GROUP)).verify:
            return
        ids_s, c_s, history_s, window_s = self._saved
        history, state = self._views()
        rows, w, slots = len(meta.slots), meta.width, meta.slots.long()
        end = 1 + ctx.seq_table.n_accept[slots]                             # tokens that stand, per row
        ids = torch.cat([history_s[:rows], ids_s[: rows * w].view(rows, w)], dim=1)
        history[slots] = ids.gather(1, (end[:, None] + torch.arange(2, device=end.device)))
        c = torch.cat([window_s[:rows], c_s[: rows * w].view(rows, w, -1)], dim=1)
        at = end[:, None] + torch.arange(self.window, device=end.device)
        state[slots] = c.gather(1, at[:, :, None].expand(-1, -1, c.shape[2]))
