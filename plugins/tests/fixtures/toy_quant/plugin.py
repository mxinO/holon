"""A toy quantized layout and its kernels, for the `qlinear@2` / `linear_experts@2` contracts (docs/dev/m06-lowp.md D4,
D5): `toy.int8/rows` holds int8 rows `q` [out, in], an f32 scale per G rows `scale` [out/G] and a per-tensor `scale2`
([] or [1]; per expert: [E, out, in], [E, out/G], E values; a merged gate/up stack's `scale2` is [2, E]). Every
kernel's `prepare` folds `scale2` into `scale` and copies `q`, so the weight it reads (`toy.int8/folded`) shares no
tensor with its source; `PREPARED` records each source it was handed and a copy of its parts, to check it was not
modified.

Matmuls register one spec per activation mode, as kernels.ggml_cuda does (`config["act"]`, path -> mode): one mode
on both paths for `act_quant` a8 or a4, and, preferred under a4, the chunk path at 4 bits (docs/dev/m07-perf.md §3.1).
As kernels.ggml_cuda's (docs/dev/m07-perf.md D6), `qlinear` lists `qlinear_glu@1` in `config["fused"]` — which fuses one
row and returns None above — and has `prepare_input`, whose handle (`Handle`) its calls take; `CALLS` logs both."""
import dataclasses

import torch
from holon.api import Constraint, KernelSpec, Layout, QTensor, Scheme
from holon_std.moe import reference as moe

G = 2                                              # rows per scale
ROWS = Layout(Scheme("toy.int8"), "rows")
FOLDED = Layout(Scheme("toy.int8"), "folded")
PREPARED: list[tuple[QTensor, dict[str, torch.Tensor]]] = []
CALLS: list[str] = []


@dataclasses.dataclass(frozen=True)
class Handle:
    x: torch.Tensor


def dequantize(parts: dict[str, torch.Tensor]) -> torch.Tensor:
    """The folded layout's weight, [out, in] or [E, out, in]."""
    return parts["q"].float() * parts["scale"].repeat_interleave(G, dim=-1)[..., None]


def _record(q: QTensor) -> None:
    assert q.layout == ROWS
    PREPARED.append((q, {r: t.clone() for r, t in q.parts.items()}))


class QLinear:
    def __init__(self, ctx, **facts) -> None:
        self.facts = facts

    def prepare(self, q: QTensor) -> QTensor:
        _record(q)
        parts = {"q": q.parts["q"].clone(), "scale": q.parts["scale"] * q.parts["scale2"]}
        return QTensor(FOLDED, q.logical_shape, parts)

    def prepare_input(self, x: torch.Tensor) -> Handle:
        CALLS.append("prepare_input")
        return Handle(x)

    def __call__(self, x, parts: dict[str, torch.Tensor]) -> torch.Tensor:
        handle = not isinstance(x, torch.Tensor)              # ours or another provider's: its `x`
        CALLS.append("qlinear(handle)" if handle else "qlinear")
        x = x.x if handle else x
        return (x.float() @ dequantize(parts).T).to(getattr(torch, self.facts["out_dtype"]))


class QLinearGlu(QLinear):
    def __call__(self, x: torch.Tensor, parts: dict[str, torch.Tensor]) -> torch.Tensor | None:
        if len(x) != 1:
            return None
        CALLS.append("qlinear_glu")
        gate, up = (x.float() @ dequantize(parts).T).chunk(2, dim=-1)
        return (torch.nn.functional.silu(gate) * up).to(getattr(torch, self.facts["out_dtype"]))


class QEmbedding:
    def __init__(self, ctx, **facts) -> None:
        pass

    def prepare(self, q: QTensor) -> QTensor:
        raise AssertionError("qembedding@1 has no prepare: a unit must not call one")

    def __call__(self, ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("a toy table is only read by a tied head")


class LinearExperts:
    def __init__(self, ctx, *, glu: str, **facts) -> None:
        self.glu = glu

    def prepare(self, q: QTensor) -> QTensor:
        _record(q)
        scale, scale2 = q.parts["scale"], q.parts["scale2"]
        per_row = scale2.T.repeat_interleave(scale.shape[1] // 2, dim=1) if self.glu == "swiglu" else \
            scale2.reshape(-1, 1)
        return QTensor(FOLDED, q.logical_shape, {"q": q.parts["q"].clone(), "scale": scale * per_row})

    def __call__(self, x: torch.Tensor, parts: dict[str, torch.Tensor], ids: torch.Tensor) -> torch.Tensor:
        return moe.linear_experts(x, dequantize(parts), ids, self.glu)


ONE_MODE, TWO_MODES = {"rows": "q8", "chunk": "q8"}, {"rows": "q8", "chunk": "q4"}


class Plugin:
    def register(self, regs) -> None:
        def add(op: str, impl, act_quant=None, priority: int = 10, act=None) -> None:
            def supports(f: dict) -> bool:
                return f["layout"] == "toy.int8/rows" and (act_quant is None or f["act_quant"] in act_quant)
            regs.kernels.add(op, KernelSpec(
                op=op, provider="tests.toy_quant", impl=impl, requires=Constraint(devices=frozenset({"cpu"})),
                supports=supports, props=frozenset({"deterministic", "graph_safe"}), priority=priority,
                config=({"act": act} if act else {}) | ({"fused": ["qlinear_glu@1"]} if op == "qlinear" else {})))

        add("qembedding", QEmbedding)
        for op, impl in (("qlinear", QLinear), ("qlinear_glu", QLinearGlu), ("linear_experts", LinearExperts)):
            add(op, impl, ("a4", "a8"), 10, ONE_MODE)
            add(op, impl, ("a4",), 11, TWO_MODES)
