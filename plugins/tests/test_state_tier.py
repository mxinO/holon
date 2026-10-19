"""std.state_tier (docs/v1-kv-tiers.md §5) on a tiny Qwen3.5/3.8-architecture checkpoint on the CPU: pages and
recurrent checkpoints in its host level, with the `prompt_end`, `branch` and `answer` policies. Tier invariance: four
sessions sharing a system prompt, interleaved through a device pool too small for them, give the token streams,
sampled logprobs and `cached_tokens` of the same sessions run back to back in a roomy one — the hybrid state
resumed from host copies is bitwise the state that never left the device. The host level's own rules on a bare tier:
leaf-first page eviction, the admission options, payload stores that always succeed, nothing being loaded evicted
before `loaded`, the clock and the in-flight floor."""
import random
import types

import pytest
import torch
from holon.api import BlockRecord, Payload, SamplingParams
from holon.core import platform
from holon.core.engine import Engine, EngineConfig
from holon.core.memory import Ledger
from holon_std.state_tier.tier import HostTier
from holon_testing.qwen3_5_files import save, tiny_hf

SAMPLED = SamplingParams(temperature=1.0, seed=3, max_tokens=8, ignore_eos=True, logprobs=0)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return save(tiny_hf(4, seed=1), tmp_path_factory.mktemp("tiny_qwen3_5_tier"))


def _engine(path, state_memory: int, policy: str = "prompt_end,branch,answer:1") -> Engine:
    return Engine(EngineConfig(
        model=str(path), device="cpu", dtype=torch.float32, page_size=4, max_seqs=2, max_step_tokens=64,
        max_model_len=256, state_memory=state_memory, slots={"tier": "std.state_tier"}, host_memory=96 << 20,
        plugin_options={"recurrent.policy": policy, "recurrent.checkpoints": "3", "state_tier.host.bytes": "64MiB",
                        "state_tier.host.pages": "0.5", "state_tier.step_cost": "1,0,0"}))


def _ask(engine: Engine, prompt: list[int]) -> tuple:
    rid, tokens, lps, cached = engine.add(prompt, SAMPLED), [], [], None
    while engine.has_work():
        for d in engine.step():
            if d.rid == rid:
                tokens += d.token_ids
                lps += d.logprobs.token if d.logprobs is not None else []
                cached = d.cached_tokens if d.cached_tokens is not None else cached
    return tokens, lps, cached


def _turn(engine, history: dict, s: int, rng: random.Random) -> tuple:
    prompt = history[s] + [rng.randrange(320) for _ in range(rng.randrange(20, 40))]
    out = _ask(engine, prompt)
    history[s] = prompt + out[0]
    return out


def test_hybrid_state_resumes_bitwise_through_the_host_level(tiny):
    system = [random.Random(9).randrange(320) for _ in range(70)]
    starts = {s: system + [random.Random(s).randrange(320) for _ in range(40)] for s in range(4)}
    roomy = _engine(tiny, 64 << 20)
    reference = {}
    for s in starts:                                         # back to back: every return hits on the device
        history, rng = {s: list(starts[s])}, random.Random(s)
        reference |= {(s, t): _turn(roomy, history, s, rng) for t in range(3)}
    roomy.close()
    tight = _engine(tiny, 360 << 10)                      # ≈ 72 pages of 4 tokens: under one session each
    history, rngs, tested = {s: list(starts[s]) for s in starts}, {s: random.Random(s) for s in starts}, {}
    for t in range(3):                                       # interleaved: returns come through the host level
        for s in starts:
            tested[s, t] = _turn(tight, history, s, rngs[s])
    stats = tight.stats()
    tight.close()
    assert tested == reference
    assert stats["std.state_tier.loaded_blocks"] > 0 and stats["std.state_tier.payload_stores"] > 0
    assert sum(cached for _, _, cached in tested.values()) > 0 and stats["dropped_publishes"] == 0


class _Done:
    def wait(self) -> None:
        pass

    synchronize = wait


def _bare(admit: str = "publish", pages: int = 6, payloads: int = 3):
    """A host level over one 2-segment page pool and one payload pool, room for `pages` blocks and `payloads`."""
    ledger = Ledger(torch.device("cpu"), host_memory=1 << 20)
    kv = ledger.request("state.kv", unit_bytes=64, segments=2, min_units=16, tokens_per_unit=16)
    ckpt = ledger.plugin_pool("ckpt", unit_bytes=32, min_units=2, pref_units=2)
    ledger.materialize(None)
    for p in (kv, ckpt):
        p.buffer.copy_(torch.arange(p.buffer.numel(), dtype=torch.uint8))
    tier = HostTier(types.SimpleNamespace(options={"page_size": 16, "plugins": {
        "state_tier.host.bytes": str(pages * 64 + payloads * 32), "state_tier.host.pages": str(pages * 64 / (
            pages * 64 + payloads * 32)), "state_tier.host.admit": admit, "state_tier.step_cost": "1,0,0",
        "state_tier.stall_bytes": "0"}}, candidates=lambda op, facts: []))
    tier.attach(types.SimpleNamespace(
        pools={"kv": kv, "plugin.ckpt": ckpt}, block=16, max_seqs=1, max_model_len=64, stream=lambda:
        platform.Stream(torch.device("cpu")), host_pool=lambda name, *, unit_bytes, segments, units, mapped=False:
        ledger.plugin_pool(name, unit_bytes=unit_bytes, segments=segments, min_units=units, pref_units=units,
                           device="host", mapped=mapped, late=True)))
    return tier, kv, ckpt


def _chain(tier, keys, evicting=False):
    accepted = set()
    for i, key in enumerate(keys):
        accepted |= tier.store([BlockRecord(key, keys[i - 1] if i else None, (key,), {"kv": (i % 16,)}, {})], [],
                               evicting)[0]
    return accepted


def test_the_host_level_evicts_leaves_first_and_never_what_it_is_loading():
    tier, kv, _ = _bare(pages=4)
    _chain(tier, [1, 2, 3])                                   # a chain 1 -> 2 -> 3
    _chain(tier, [10, 11])                                    # and 10 -> 11: 5 blocks, room for 4
    assert tier.dropped() == [(3, None)]                      # 3 is the least recent leaf; 1 and 2 are interior
    load = tier.load([10], {"kv": [5]}, [], [])
    _chain(tier, [20, 21])                                    # 10 is being loaded: the leaves 2, then 1, go
    assert tier.dropped() == [(2, None), (1, None)] and (10, None) in tier.copies
    assert torch.equal(kv.units()[:, 5], kv.units()[:, 0])   # 10's pages: what unit 0 held when it was stored
    tier.loaded(load, _Done())
    _chain(tier, [30])                                        # loaded: 10 may go once a leaf, after its child 11
    assert tier.dropped() == [(11, None)]
    assert load.due == tier.t == pytest.approx(64 / 43e9)     # charged on the H2D lane; `loaded` brought the clock


def test_a_parent_stored_after_its_child_is_not_a_leaf():
    tier, _, _ = _bare("evict", pages=4)                     # write-back stores a chain's leaves first
    tier.store([BlockRecord(2, 1, (2,), {"kv": (1,)}, {})], [], True)
    tier.store([BlockRecord(1, None, (1,), {"kv": (0,)}, {})], [], True)
    tier.hit([1, 2], [])                                      # 1 is the least recent copy, but not a leaf
    _chain(tier, [5, 6, 7], evicting=True)
    assert tier.dropped() == [(2, None)]


@pytest.mark.parametrize("admit, stored", [("publish", {1, 2}), ("hit:2", {2}), ("evict", set())])
def test_page_admission_options_and_mandatory_payloads(admit, stored):
    tier, _, ckpt = _bare(admit)
    tier.hit([2], [])
    tier.hit([2], [])
    got = {k for k, s in _chain(tier, [1, 2]) if s is None}
    assert got == stored
    assert _chain(tier, [7], evicting=True) == ({(7, None)} if admit != "hit:2" else set())
    for key in range(4):                                       # 3 payload units: the fourth evicts the oldest
        tier.store([BlockRecord(key, None, (key,), {}, {"c": Payload((key % 2,), "plugin.ckpt", key == 3)})], [],
                   False)
    assert ((0, "c") not in tier.copies and (0, "c") in tier.dropped() and tier.tentative == {(3, "c")})
    tier.settled([(3, "c")])
    assert not tier.tentative


def test_attach_refuses_a_host_level_below_the_loads_in_flight():
    with pytest.raises(RuntimeError, match="too small for the loads in flight"):
        _bare(pages=3)                                        # one max_model_len load is 4 blocks
