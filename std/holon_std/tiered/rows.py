"""std.tiered.rows@1: a row table — `std.ops.embedding@1`'s contract, ids → rows in the model's dtype — that may live in
pinned host RAM (docs/v1-flash-next.md D5). A quantized table on the host is read into a mapped host pool from its
checkpoint extent, and its `qembedding@1` kernel gathers the step's rows from there across the bus, in place (16 rows
× 90 B a token for Qwen3.8-Flash-Next's PLE table, one row for a token embedding). On the device it is held and read
as `std.ops.embedding@1` holds and reads it.

Where each table lives is the deployment's: the plugin option `rows.place` names tables (`<name>:host` or
`<name>:device`, comma-separated, `*` for the rest); a table not named takes its Ref's `place` (default `device`). The
host copy takes `host_memory` and is filled at build. Only quantized tables go to the host (a dense one is the weight
loader's, on the device)."""
from __future__ import annotations

import torch
from torch import nn

from . import files

PLACES = ("device", "host")
QEMBEDDING = "qembedding@1"


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
    def __init__(self, ctx, *, vocab: int, hidden: int, name: str = "rows", place: str = "device") -> None:
        super().__init__()
        self.vocab, self.hidden, self.dtype, self.name = vocab, hidden, ctx.dtype, name
        self.place = placement(ctx, name, place)
        self.weight = nn.Parameter(torch.empty(vocab, hidden, dtype=ctx.dtype), requires_grad=False)
        self._q = None                         # the adopted quantized table (layout, parts, extents)
        self._kernel = None
        self._table: torch.Tensor | None = None

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
            if self.place == "host":
                raise ValueError(f"std.tiered.rows@1 {self.name}: a dense table cannot be held on the host")
            return
        layout = self._q.layout
        facts = {"layout": f"{layout.scheme.elem}/{layout.name}", "vocab": self.vocab, "hidden": self.hidden,
                 "dtype": str(self.dtype).removeprefix("torch.")}
        site = "std.ops.qembedding[" + ",".join(f"{k}={v}" for k, v in facts.items()) + "]"
        self._kernel = ctx.select_kernel(QEMBEDDING, site=site, **facts).impl(ctx, **facts)
        blocks = self._q.parts["blocks"]
        if self.place == "device":
            self._table = blocks.to(ctx.device)
        else:
            self._table = self._host(ctx, blocks, self._q.extents.get("blocks"))
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
        return self._kernel(ids, self._table)
