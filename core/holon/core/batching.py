"""Host-side batch helpers shared by the scheduler, the engine and reference runs."""
from __future__ import annotations

import numpy as np

from holon.api import BatchPlan, RowKind, TokenSrc


def padded(plan: BatchPlan, rows: int, slot: int, width: int = 1) -> BatchPlan:
    """`plan` with rows of `width` tokens, all producing logits (decode rows, or verify rows), appended up to `rows`,
    all on `slot` at position 0 (graph padding)."""
    extra = rows - len(plan.slots)
    if extra == 0:
        return plan
    pad = lambda a, v, n=extra: np.concatenate([a, np.full(n, v, dtype=a.dtype)])   # noqa: E731
    out = host_plan(plan.step, pad(plan.slots, slot), pad(plan.start_upper, 0), pad(plan.n_query, width),
                    pad(plan.n_logits, width), pad(plan.host_tokens, 0, extra * width),
                    pad(plan.token_src, TokenSrc.HOST))
    out.mem_ops, out.publishes, out.draft_k = plan.mem_ops, plan.publishes, plan.draft_k
    if plan.row_kind is not None:
        out.row_kind = pad(plan.row_kind, RowKind.VERIFY if width > 1 else RowKind.DECODE)
    return out


def host_plan(step: int, slots: np.ndarray, starts: np.ndarray, n_query: np.ndarray, n_logits: np.ndarray,
              tokens: np.ndarray, token_src: np.ndarray | None = None) -> BatchPlan:
    """A step's plan; input tokens come from the host unless `token_src` says otherwise per row."""
    src = np.full(len(slots), TokenSrc.HOST, dtype=np.int8) if token_src is None else token_src
    return BatchPlan(step=step, slots=slots, n_query=n_query, n_logits=n_logits, start_upper=starts,
                     token_src=src, host_tokens=tokens)
