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
    ahead: bool = False                            # the aux drafter writes KV ahead through sub-steps
    hidden: set[tuple[int, ...]] = dataclasses.field(default_factory=set)   # prompts drafted right, but n_draft 0
    hidden_slots: set[int] = dataclasses.field(default_factory=set)
    real: int | None = None                        # drafts per row that are real (n_draft); the rest invalid ids
    rewinds: int = 0                               # verify steps the drafter saw rewound before it ran
    max_position: int = -1                         # the largest position any step computed at
    max_computed: int = -1                         # the largest length a step left in the SeqTable
    history: object = None                         # the drafter's per-slot token history (a [slots, positions] tensor)
    absorbed: object = None                        # the part of it that came from steps' inputs
    observed: list[tuple[tuple[int, ...], int, int, int]] = dataclasses.field(default_factory=list)
    # (prompt, k, n_draft, n_accept) per committed sampled row


ORACLE = Oracle({})
