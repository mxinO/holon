"""`std.spec.fixed_k` (docs/dev/m04-spec.md D7): k by the number of running requests; with a fallback, plain decoding
while no request has had a real draft lately, and speculation again from the first one."""
import types

import pytest
from holon_std.spec.fixed_k.plugin import Plugin


def _policy(**opts):
    return Plugin().build(types.SimpleNamespace(options={"plugins": opts}))


def test_k_by_running_requests():
    policy = _policy(**{"spec.k": "1:3,3:2"})
    assert [policy.k([object()] * n) for n in (1, 2, 3, 8)] == [3, 3, 2, 2]
    assert _policy().k([object()]) == 3
    with pytest.raises(ValueError):
        _policy(**{"spec.k": "2:3"})


def test_fallback_switches_off_and_back_on():
    policy, seqs = _policy(**{"spec.k": "4", "spec.fallback": 2}), [object(), object()]
    for _ in range(2):
        assert policy.k(seqs) == 4
        for s in seqs:
            policy.observe(s, 4, 0, 0)                     # no real drafts
    assert policy.k(seqs) == 0                             # plain decoding: the drafter still looks up
    policy.observe(seqs[1], 0, 3, 0)                       # a hit
    assert policy.k(seqs) == 4
