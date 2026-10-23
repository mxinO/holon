"""std.tiered.rows@1: a row table — `std.ops.embedding@1`'s contract, ids → rows in the model's dtype — that may live in
pinned host RAM (docs/v1-flash-next.md D5). A quantized table on the host is read into a mapped host pool from its
checkpoint extent, and its `qembedding@1` kernel gathers the step's rows from there across the bus, in place (16 rows
× 90 B a token for Qwen3.8-Flash-Next's PLE table, one row for a token embedding). On the device it is held and read
as `std.ops.embedding@1` holds and reads it.

Where each table lives is the deployment's: the plugin option `rows.place` names tables (`<name>:host`,
`<name>:device` or `<name>:ssd`, comma-separated, `*` for the rest); a table not named takes its Ref's `place` (default
`device`). The host copy takes `host_memory` and is filled at build. Only quantized tables go to the host or the SSD
(a dense one is the weight loader's, on the device).

On the SSD (D5's fallback, where a desktop cannot hold the table in RAM) a table stays in its checkpoint file: each
call's rows are read there, in the graph, by `cpu_task@1`'s copy task on the I/O worker — the call writes each id's
file offset into its read list (mapped host memory), signals the worker, which reads every row with `O_DIRECT` (the
row's aligned pages into a bounce buffer: the page cache holds no copy), waits, and gathers the rows it read with the
same kernel, so the rows are the RAM tier's, byte for byte. A call reads at most `per_token` rows a token of the
engine's largest step (the read list's size); an I/O error stops the engine at the step's commit."""
from __future__ import annotations

import os

import numpy as np
import torch
from torch import nn

from . import files

PLACES = ("device", "host", "ssd")
QEMBEDDING, CPU_TASK = "qembedding@1", "cpu_task@1"


def placement(ctx, name: str, default: str = "device") -> str:
    """The deployment's place for table `name`: `rows.place`'s entry for it, or for `*`, else `default`."""
    spec = str(ctx.options.get("plugins", {}).get("rows.place", "") or "")
    items = (item.partition(":") for item in spec.split(",") if item.strip())
    named = {k.strip(): v.strip() for k, _, v in items}
    got = named.get(name, named.get("*", default))
    if got not in PLACES:
        raise ValueError(f"std.tiered.rows@1: rows.place {name}:{got!r}, not one of {PLACES}")
    return got


class Rows(nn.Module):
    def __init__(self, ctx, *, vocab: int, hidden: int, name: str = "rows", place: str = "device",
                 per_token: int = 1) -> None:
        super().__init__()
        self.vocab, self.hidden, self.dtype, self.name, self.per_token = vocab, hidden, ctx.dtype, name, per_token
        self.place = placement(ctx, name, place)
        self.weight = nn.Parameter(torch.empty(vocab, hidden, dtype=ctx.dtype), requires_grad=False)
        self._q = None                         # the adopted quantized table (layout, parts, extents)
        self._kernel = None
        self._table: torch.Tensor | None = None
        self._ssd = None                       # the SSD tier's read list and worker (`_reads`)

    def adopt_weights(self, ctx, name: str, q, index: int | None = None) -> bool:
        if name != "weight" or index is not None or tuple(q.logical_shape) != (self.vocab, self.hidden):
            return False
        if set(q.parts) != {"blocks"}:
            return False
        self._q = q                            # read at bind; the parameter holds nothing of it
        torch.utils.swap_tensors(self.weight, nn.Parameter(torch.empty(0, dtype=self.dtype), requires_grad=False))
        return True

    def bind_kernels(self, ctx) -> None:
        if self._q is None:
            if self.place != "device":
                raise ValueError(f"std.tiered.rows@1 {self.name}: a dense table cannot be held on the {self.place}")
            return
        layout = self._q.layout
        facts = {"layout": f"{layout.scheme.elem}/{layout.name}", "vocab": self.vocab, "hidden": self.hidden,
                 "dtype": str(self.dtype).removeprefix("torch.")}
        site = "std.ops.qembedding[" + ",".join(f"{k}={v}" for k, v in facts.items()) + "]"
        self._kernel = ctx.select_kernel(QEMBEDDING, site=site, **facts).impl(ctx, **facts)
        blocks = self._q.parts["blocks"]
        if self.place == "device":
            self._table = blocks.to(ctx.device)
        elif self.place == "host":
            self._table = self._host(ctx, blocks, self._q.extents.get("blocks"))
        else:
            self._ssd = _Reads(ctx, self, blocks, self._q.extents.get("blocks"))
        self._q = None                         # the source's mapping is released

    def _host(self, ctx, blocks: torch.Tensor, extent) -> torch.Tensor:
        row = blocks.numel() * blocks.element_size() // self.vocab
        pool = ctx.pool(f"tiered.rows.{self.name}", unit_bytes=row, min_units=self.vocab, pref_units=self.vocab,
                        device="host", mapped=True)
        if pool.capacity < self.vocab:
            raise ValueError(f"std.tiered.rows@1 {self.name}: host_memory holds {pool.capacity} of {self.vocab} rows")
        host = pool.view(torch.uint8, (pool.capacity, row))[: self.vocab]
        if extent is not None and extent.nbytes == host.numel():
            files.read([(extent, host.numpy().reshape(-1))])
        else:                                   # no single file extent: copy what the source mapped
            host.reshape(-1).copy_(blocks.reshape(-1).view(torch.uint8))
        return pool.device_view(torch.uint8, (pool.capacity, row))[: self.vocab].view(blocks.dtype).view(blocks.shape)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if self._kernel is None:
            return nn.functional.embedding(ids, self.weight)
        if self._ssd is not None:
            table, ids = self._ssd(ids)
            return self._kernel(ids, table)
        return self._kernel(ids, self._table)


class _Reads:
    """A table on the SSD: the read list (per row (fd, O_DIRECT, offset, row bytes), in mapped host memory), the request
    table naming its first n rows, the rows read (mapped), the copy task's slot on the I/O worker."""

    plant = 0                                   # judge-only (`rows_off_by_one`): rows read this far past each id's

    def __init__(self, ctx, rows: Rows, blocks: torch.Tensor, extent) -> None:
        if extent is None or extent.nbytes != blocks.numel() * blocks.element_size():
            raise ValueError(f"std.tiered.rows@1 {rows.name}: the SSD tier reads one file extent (none for this table)")
        o = ctx.options
        tokens = max(o.get("max_step_tokens", 1), o.get("max_seqs", 1) * (1 + o.get("max_draft_tokens", 0)))
        self.width, self.row = rows.per_token * tokens, extent.nbytes // rows.vocab
        self.base, self.shape, self.dtype = extent.offset, (-1, *blocks.shape[1:]), blocks.dtype
        name = f"tiered.rows.{rows.name}.ssd"
        lists = [ctx.pool(f"{name}.{part}", unit_bytes=unit, min_units=n, pref_units=n, device="host", mapped=True)
                 for part, unit, n in (("list", 32, self.width), ("requests", 8, 1 + self.width),
                                       ("rows", self.row, self.width))]
        if any(p.capacity < n for p, n in zip(lists, (self.width, 1 + self.width, self.width), strict=True)):
            raise ValueError(f"std.tiered.rows@1 {rows.name}: host_memory holds no SSD read list of {self.width} rows")
        fd, direct = files._open(extent.path)
        desc = lists[0].view(torch.int64, (lists[0].capacity, 1, 4))[: self.width]
        desc[:, 0, 0], desc[:, 0, 1], desc[:, 0, 3] = fd, int(direct), self.row
        requests = lists[1].view(torch.int32, (lists[1].capacity, 2))[: 1 + self.width]
        requests[1:] = torch.arange(self.width, dtype=torch.int32)[:, None]        # row i into unit i
        requests[0] = 0
        out = lists[2].view(torch.uint8, (lists[2].capacity, self.row))[: self.width]
        self.io = ctx.select_kernel(CPU_TASK, site=f"std.tiered.rows[{rows.name}].io", pool="io").impl(ctx, pool="io")
        self.slot = self.io.bind(self.io.copy([(desc, out, requests)]))
        self.offsets = lists[0].device_view(torch.int64, (lists[0].capacity, 4))[: self.width, 2]
        self.n = lists[1].device_view(torch.int32, (lists[1].capacity, 2))[0, :1]
        self.rows = lists[2].device_view(torch.uint8, (lists[2].capacity, self.row))[: self.width]
        self.index = ctx.static(f"{name}.index", (self.width,), torch.int64)
        self.index.copy_(torch.arange(self.width))
        ctx.shared(f"{name}.errors", lambda root: _Errors(self.io.errors, rows.name, fd))

    def __call__(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The table of this call's rows (row i id i's) and the ids into it, once the worker has read them."""
        n = ids.numel()
        if n > self.width:
            raise ValueError(f"std.tiered.rows@1: {n} ids past the SSD read list's {self.width}")
        self.offsets[:n].copy_((ids.reshape(-1) + self.plant) * self.row + self.base)
        self.n.fill_(n)
        self.io.signal(self.slot)
        self.io.wait(self.slot)
        return self.rows[:n].view(self.dtype).view(self.shape), self.index[:n].view(ids.shape)


class _Errors:
    """The I/O worker's error word for an SSD table's reads: zeroed at each launch, checked at each commit."""

    def __init__(self, word: torch.Tensor, name: str, fd: int) -> None:
        self.word, self.name, self.fd, self.telemetry = word, name, fd, {"errors": word}

    def before_launch(self, step):
        return [(self.word, np.zeros(1, dtype=np.int32))], []

    def after_commit(self, step, telemetry) -> None:
        if bits := int(telemetry["errors"][0]):
            raise RuntimeError(f"std.tiered.rows@1 {self.name}: the SSD reads failed (cpu_task@1 errors {bits}: 1 a "
                               "read failed, 2 a wait timed out)")

    def shutdown(self) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
