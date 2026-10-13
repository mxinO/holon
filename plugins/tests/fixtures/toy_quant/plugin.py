"""A toy quantized layout and its kernels, for the `qlinear@2` / `linear_experts@2` contracts (docs/m06-lowp.md D4,
D5): `toy.int8/rows` holds int8 rows `q` [out, in], an f32 scale per G rows `scale` [out/G] and a per-tensor `scale2`
([] or [1]; per expert: [E, out, in], [E, out/G], E values; a merged gate/up stack's `scale2` is [2, E]). Every
kernel's `prepare` folds `scale2` into `scale` and copies `q`, so the weight it reads (`toy.int8/folded`) shares no
tensor with its source; `PREPARED` records each source it was handed and a copy of its parts, to check it was not
modified."""
import torch
from holon.api import Constraint, KernelSpec, Layout, QTensor, Scheme
from holon_std.moe import reference as moe

G = 2                                              # rows per scale
ROWS = Layout(Scheme("toy.int8"), "rows")
FOLDED = Layout(Scheme("toy.int8"), "folded")
PREPARED: list[tuple[QTensor, dict[str, torch.Tensor]]] = []


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

    def __call__(self, x: torch.Tensor, parts: dict[str, torch.Tensor]) -> torch.Tensor:
        return (x.float() @ dequantize(parts).T).to(getattr(torch, self.facts["out_dtype"]))


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


class Plugin:
    def register(self, regs) -> None:
        for op, impl in (("qlinear", QLinear), ("qembedding", QEmbedding), ("linear_experts", LinearExperts)):
            regs.kernels.add(op, KernelSpec(op=op, provider="tests.toy_quant", impl=impl,
                                            requires=Constraint(devices=frozenset({"cpu"})),
                                            supports=lambda f: f["layout"] == "toy.int8/rows",
                                            props=frozenset({"deterministic", "graph_safe"})))
