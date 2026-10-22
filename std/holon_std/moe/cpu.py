"""F5b, the CPU's share of a layer's misses (docs/v1-offload.md D4–D7): the engine's one `CpuShare` (a shared object of
the offload store's layers), present when the deployment gives a split table.

Per expert pass of at most `tokens` tokens — a graphed step's most (its decode and verify rows), so a decode step of
any batch, and a prompt chunk as short as that —, a layer's graph runs

    moe_resolve(plan, jobs) → [publish: its CPU rows' inputs; signal] → expert_fetch → linear_experts_arena ×2
        → [wait] → moe_patch_rows → moe_combine

`moe_resolve@1` takes `plan[n]` of the layer's n CPU candidates (the lowest-ranked) and writes their jobs; the CPU
experts (`cpu_experts@1`, a native task the `cpu_task@1` worker runs: one slot per layer) compute them from the host
tier's blocks while the GPU fetches and computes the rest; their output rows replace the null slots' zeros before the
combine. Larger passes take no CPU share (a zero plan): the rule is the pass's size, deployment-level (D4), never the
row kind, so a short prompt chunk takes the share a decode step of its size takes (the L2 run at one token per step
measures the split that way); F8, a separate share for larger prefill chunks, is not built. The job table, the input
rows and the output rows are mapped host pools, written and read in place by both sides (zero-copy, S6).

Options (EngineConfig.plugin_options, from `holon_tools tune`): `moe.cpu_plan` — `plan[1], plan[2], …, plan[N]`, the
CPU's share of n misses (D4: deployment-level, never per request); past N the last entry's share, plan[n] =
⌊n · plan[N] / N⌋ —, and `moe.cpu_isa`, the CPU tier it was measured on: on a host of another tier the split is off
(S5: the store forces F5a), and `moe.cpu_min_tokens`, the fewest tokens of a pass that takes the share (default 1; 2
at batch 1: verify passes only, a decode pass's misses all fetched — FN step 7: a verify window's misses are 87%
one-token experts, which the CPU computes as fast as a decode's, against 3.6 times the bytes to fetch), and
`moe.cpu_admit` (`on`: the CPU's experts are admitted as any miss would be, under a variant that `defers`, their
pieces copied after the layer on side streams, so the cache evolves as without the split; default `off`: never
admitted). The worker's threads are `cpu_task@1`'s options."""
from __future__ import annotations

import torch

from . import kernels


def plan_table(text: str, experts: int) -> list[int] | None:
    """`moe.cpu_plan` as plan[0 … experts] (plan[0] = 0); None when it is empty or all zero."""
    given = [int(v) for v in text.replace(" ", "").split(",") if v]
    if not any(given):
        return None
    if any(not 0 <= p <= n for n, p in enumerate(given, 1)):
        raise ValueError(f"moe.cpu_plan {text!r}: plan[n] must lie in [0, n]")
    n_last, last = len(given), given[-1]
    return [0, *given[:experts], *(n * last // n_last for n in range(n_last + 1, experts + 1))][:experts + 1]


class CpuShare:
    def __init__(self, ctx, store, *, plan: list[int], tokens: int, min_tokens: int = 1) -> None:
        self.ctx, self.store, self.tokens, self.min_tokens = ctx, store, tokens, min_tokens
        self.top_k, self.jobs_max = store.top_k, min(store.items, tokens * store.top_k)
        self.plan = ctx.static("moe.cpu.plan", (store.items + 1,), torch.int32)
        self.plan.copy_(torch.tensor(plan, dtype=torch.int32, device="cpu"))       # (units are built on meta)
        self.task = None              # `cpu_task@1`'s, at the first bind: its worker (a shared object) is built after
        #                               this one, so it starts after this one's `after_materialize` has bound the tasks
        self.layers: list[_Layer] = []
        self.hidden, self.patch, self.pools = None, None, None

    def bind(self, ctx, gate_up, down) -> int:
        """A layer's CPU experts (its gate/up and down stacks, `ArenaExperts`): its slot."""
        hidden, inter = gate_up.in_features, gate_up.out // 2
        if self.hidden is None:
            self.hidden = hidden
            self.task = ctx.select_kernel(kernels.CPU_TASK, site="std.moe.cpu", pool="compute").impl(ctx,
                                                                                                    pool="compute")
            self.store.watch(self.task.errors)             # the worker's error word: checked at each commit
            facts = {"hidden": hidden, "top_k": self.top_k, "dtype": str(ctx.dtype).removeprefix("torch.")}
            self.patch = ctx.select_kernel(kernels.MOE_PATCH_ROWS, site="std.moe.cpu", **facts).impl(ctx, **facts)
        elif hidden != self.hidden:
            raise ValueError(f"std.moe.cpu: layers of hidden {self.hidden} and {hidden}")
        facts = {"gate_up": gate_up.layout_fact(ctx), "down": down.layout_fact(ctx), "hidden": hidden, "inter": inter,
                 "top_k": self.top_k, "tokens": self.tokens, "jobs": self.jobs_max}
        site = "std.moe.cpu[" + ",".join(f"{k}={v}" for k, v in facts.items()) + "]"
        layer = _Layer(ctx.select_kernel(kernels.CPU_EXPERTS, site=site, **facts).impl(ctx, **facts), gate_up.cls,
                       down.cls)
        self.layers.append(layer)
        return self.task.bind(layer)

    def before_materialize(self, ctx) -> None:
        """The job table, the CPU's input rows and its output rows: mapped host pools."""
        shapes = {"jobs": (1 + self.jobs_max, 4 + self.tokens), "x": (self.tokens, self.hidden),
                  "rows": (self.tokens * self.top_k, self.hidden)}
        self.pools = {name: (ctx.pool(f"moe.cpu.{name}", unit_bytes=4 * shape[1], min_units=shape[0],
                                      pref_units=shape[0], device="host", mapped=True), shape)
                      for name, shape in shapes.items()}

    def after_materialize(self, ctx) -> None:
        if self.layers and self.layers[0].task is None:     # host pools are allocated once: bind once
            dtype = {"jobs": torch.int32, "x": torch.float32, "rows": torch.float32}
            host = {n: p.view(dtype[n], shape) for n, (p, shape) in self.pools.items()}
            self.device = {n: p.device_view(dtype[n], shape) for n, (p, shape) in self.pools.items()}
            host["jobs"].zero_()
            for layer in self.layers:
                layer.task = layer.kernel.task(host["jobs"], host["x"], layer.units(layer.gate_up),
                                               layer.units(layer.down), host["rows"])

    # -- a pass ------------------------------------------------------------------------------------------------------
    def takes(self, tokens: int) -> bool:
        """Whether a pass of `tokens` tokens takes the CPU's share (by its size: D4)."""
        return self.min_tokens <= tokens <= self.tokens

    def args(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The resolve's `plan` and `jobs` for a pass with a CPU share."""
        return self.plan, self.device["jobs"]

    def publish(self, slot: int, x: torch.Tensor, cpu: torch.Tensor) -> None:
        """After the layer's resolve: its CPU tokens' inputs to the CPU, then its jobs run."""
        self.patch.publish(x, cpu, self.device["x"])
        self.task.signal(slot)

    def collect(self, slot: int, y: torch.Tensor, cpu: torch.Tensor) -> None:
        """Before the combine: the CPU's rows into the experts' output `y` [T, k, H]."""
        self.task.wait(slot)
        self.patch(y, self.device["rows"], cpu)


class _Layer:
    """A layer's `cpu_task@1` task: its kernel's, over the shared buffers and its classes' host pools (bound after the
    first materialize; the worker reads `native` after it too)."""

    def __init__(self, kernel, gate_up, down) -> None:
        self.kernel, self.gate_up, self.down, self.task = kernel, gate_up, down, None

    @staticmethod
    def units(cls) -> torch.Tensor:
        return cls.host_view()

    @property
    def native(self):
        return self.task.native

    def __call__(self) -> None:
        self.task()
