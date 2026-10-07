"""Planted bugs that calibrate the bounded check (`bounded --plant`, judge/tests/test_bounded.py): each breaks one
path the check guards, and the configurations using that path must fail.

They patch engine classes, so they apply to every engine built inside the context, graph captures included. The
reference re-scoring is untouched by all three: prompt-logprob requests never adopt cached blocks, and it samples one
token per request, which is never fed back.
"""
from __future__ import annotations

import contextlib
import dataclasses
from collections.abc import Callable, Iterator
from unittest import mock

import torch


@contextlib.contextmanager
def adoption_off_by_one() -> Iterator[None]:
    """A prefix hit claims one token more than it adopted, so that token's state is never computed."""
    from holon.core.state import StateManager

    admit = StateManager.admit

    def planted(self, seq):
        ok = admit(self, seq)
        if ok and seq.num_computed and seq.num_computed + 2 <= len(seq.tokens):
            seq.num_computed += 1
        return ok
    with mock.patch.object(StateManager, "admit", planted):
        yield


@contextlib.contextmanager
def wrong_block() -> Iterator[None]:
    """A prefix hit of two or more blocks adopts its first block's state in place of its last block's. (Swapping two
    adopted blocks would be no bug: keys are cached after RoPE, and attention over a prefix ignores its order.)"""
    from holon.core.prefix import PrefixIndex

    match = PrefixIndex.match

    def planted(self, keys, tokens):
        entries = match(self, keys, tokens)
        if len(entries) >= 2:
            entries = [*entries[:-1], dataclasses.replace(entries[-1], units=entries[0].units)]
        return entries
    with mock.patch.object(PrefixIndex, "match", planted):
        yield


@contextlib.contextmanager
def stale_device_token() -> Iterator[None]:
    """After a sampled token id divisible by 7, the next step's device-sourced input (depth 1) is the previous
    token."""
    from holon.core.engine import Engine

    store = Engine._store

    def planted(self, ctx, tokens, accepts=None):
        if accepts is not None:
            return store(self, ctx, tokens, accepts)
        slots = ctx.batch.dev.slots[ctx.batch.dev.logits_rows]
        before = self.last_token[slots]
        out = store(self, ctx, tokens, accepts)
        self.last_token[slots] = torch.where(tokens % 7 == 0, before, tokens)
        return out
    with mock.patch.object(Engine, "_store", planted):
        yield


PLANTS: dict[str, Callable[[], contextlib.AbstractContextManager[None]]] = {
    "adoption-off-by-one": adoption_off_by_one, "wrong-block": wrong_block,
    "stale-device-token": stale_device_token}
