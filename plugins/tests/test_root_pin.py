"""evict.root_pin (slot `eviction_policy`): hit roots stay while anything else is eligible, and go when nothing else
is; an engine binds it by slot name with its options."""
import collections

import torch
from holon.core.engine import Engine, EngineConfig
from holon_plugins.evict.root_pin.plugin import RootPin
from holon_testing.qwen3_5_files import save, tiny_hf

E = collections.namedtuple("E", "key depth hits")


def test_hit_roots_stay_while_anything_else_can_go(tmp_path):
    pin = RootPin(depth=0, hits=2)
    assert pin.victim(iter([E(1, 0, 5), E(2, 1, 0), E(3, 0, 1)])).key == 2   # 1: a root matched 5 times
    assert pin.victim(iter([E(3, 0, 1), E(1, 0, 5)])).key == 3               # a root matched once is not pinned
    assert pin.victim(iter([E(1, 0, 5), E(4, 0, 2)])).key == 1               # only pinned ones: the oldest goes
    assert pin.victim(iter([])) is None
    engine = Engine(EngineConfig(model=str(save(tiny_hf(4, seed=1), tmp_path)), device="cpu", dtype=torch.float32,
                                 slots={"eviction_policy": "evict.root_pin"}, max_model_len=256, state_memory=8 << 20,
                                 plugin_options={"root_pin.depth": "1", "root_pin.hits": "3"}))
    assert vars(engine.state.index.policy) == {"depth": 1, "hits": 3}
    engine.close()
