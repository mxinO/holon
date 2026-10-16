"""A request's prefix-cache hit reaches the client once per sequence (`TokenDelta.cached_tokens`): the supervisor
keeps the first engine's count when it resumes a sequence on another (the engine's side: plugins/tests/test_engine.py,
on a tiny model)."""
from holon.api import SamplingParams, TokenDelta
from holon.core.supervisor import _Request


def test_a_resumed_sequence_keeps_its_first_count():
    req = _Request("r", [1, 2, 3], SamplingParams(n=2, seed=1))
    first = [TokenDelta("r#1", [7], 0, cached_tokens=16), TokenDelta("r#1", [8], 1, cached_tokens=16),
             TokenDelta("r#1", [9], 0)]
    resumed = [TokenDelta("r#2.0", [5], 0, cached_tokens=20), TokenDelta("r#2.1", [6], 1, cached_tokens=20)]
    for d in first + resumed:
        req.first_count(d)
    assert [d.cached_tokens for d in first + resumed] == [16, 16, None, None, None]
