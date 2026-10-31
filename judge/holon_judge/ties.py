"""The judge's scoped patch for a tie rule (D2 of each model's design): a module's call runs with one attribute (e.g.
`torch.Tensor.topk`) replaced, restored when the call returns, so only that module's own top-k breaks ties the
rule's way (docs/dev/v1-flash-next.md D2, docs/dev/v1-glm.md D2)."""
from __future__ import annotations

from collections.abc import Callable


def scoped(forward: Callable, owner: object, name: str, replacement: Callable) -> Callable:
    """`forward`, with `owner.name` replaced by `replacement` for the duration of each call."""
    def call(*args, **kwargs):
        saved = getattr(owner, name)
        setattr(owner, name, replacement)
        try:
            return forward(*args, **kwargs)
        finally:
            setattr(owner, name, saved)
    return call
