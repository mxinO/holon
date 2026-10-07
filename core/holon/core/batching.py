"""Host-side batch helpers shared by the scheduler, the engine and reference runs."""
from __future__ import annotations

import numpy as np

from holon.api import BatchPlan, ShapeKey, TokenSrc


def host_plan(step: int, slots: np.ndarray, starts: np.ndarray, n_query: np.ndarray, n_logits: np.ndarray,
              tokens: np.ndarray, token_src: np.ndarray | None = None) -> BatchPlan:
    """A step's plan; input tokens come from the host unless `token_src` says otherwise per row."""
    src = np.full(len(slots), TokenSrc.HOST, dtype=np.int8) if token_src is None else token_src
    return BatchPlan(step=step, slots=slots, n_query=n_query, n_logits=n_logits, start_upper=starts,
                     token_src=src, host_tokens=tokens, shape=ShapeKey(int(n_query.sum()), len(slots)))
