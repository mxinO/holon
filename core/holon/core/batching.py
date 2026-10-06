"""Host-side batch helpers shared by the scheduler, the engine and reference runs."""
from __future__ import annotations

import numpy as np

from holon.api import BatchPlan, ShapeKey, TokenSrc


def host_plan(step: int, slots: np.ndarray, starts: np.ndarray, n_query: np.ndarray, n_logits: np.ndarray,
              tokens: np.ndarray) -> BatchPlan:
    """A plan whose input tokens all come from the host."""
    return BatchPlan(step=step, slots=slots, n_query=n_query, n_logits=n_logits, start_upper=starts,
                     token_src=np.full(len(slots), TokenSrc.HOST, dtype=np.int8), host_tokens=tokens,
                     shape=ShapeKey(int(n_query.sum()), len(slots)))
