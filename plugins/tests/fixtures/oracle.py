"""Shared state of the test draft policy and drafter: the full expected sequence of each prompt (prompt + greedy
continuation), and how the drafter corrupts it. Tests set ORACLE before building an engine."""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class Oracle:
    sequences: dict[tuple[int, ...], list[int]]    # prompt -> prompt + expected continuation
    k: int = 3                                     # drafts requested per step
    mode: str = "right"                            # right | wrong | mixed | alternate
    vocab: int = 512
    by_slot: dict[int, list[int]] = dataclasses.field(default_factory=dict)
    steps: int = 0


ORACLE = Oracle({})
