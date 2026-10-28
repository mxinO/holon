"""state_tiers.zlib, the held-out `Level` (docs/dev/v1-kv-tiers.md §1): blobs round-trip bitwise and its capacity is
counted in uncompressed units; through std.state_tier on the tiny hybrid it passes tier invariance as the SSD level
does, its pages coming back from the level."""
import random

import numpy as np
import pytest
import torch
from holon.core.engine import Engine, EngineConfig
from holon_judge import bounded, tiers
from holon_plugins.state_tiers.zlib.plugin import ZlibLevel
from holon_testing.qwen3_5_files import save, tiny_hf


def test_blobs_round_trip_and_capacity_counts_uncompressed_units():
    level = ZlibLevel(None, {"zlib.bytes": "64KiB"})
    assert level.bind(None, {"kv": (4096, 1, 0.5), "c": (1000, 2, 0.5)}, None) == {"kv": 8, "c": 32}
    data = np.zeros(4096, np.uint8)
    data[::7] = 3
    level.write(1, None, {"kv": [[memoryview(data)]]})
    out = np.ones(4096, np.uint8)
    level.read(1, None, {"kv": [[memoryview(out)]]})
    assert np.array_equal(out, data) and level.stored() < 4096
    level.drop(1, None)
    assert level.stored() == 0


@pytest.mark.slow
def test_tier_invariance_through_the_zlib_level(tmp_path):
    model = tiny_hf(4, seed=1)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("A_log"):
                p.fill_(-6.0)
    path = save(model, tmp_path)

    def build(tier=True, options=None, **overrides):
        opts = {"state_tier.host.bytes": "2177928", "state_tier.host.pages": str(2105344 / 2177928),
                "recurrent.checkpoints": "6", "recurrent.policy": "branch", "state_tier.levels": "state_tiers.zlib@1",
                "zlib.bytes": "32MiB"} | (options or {})
        cfg = dict(model=str(path), device="cpu", dtype=torch.float32, page_size=16, max_seqs=4, max_step_tokens=64,
                   max_model_len=512, state_memory=920 << 10, host_memory=64 << 20,
                   slots={"tier": "std.state_tier"} if tier else {}, plugin_options=opts) | overrides
        return Engine(EngineConfig(**cfg))
    rng = random.Random(0)
    trace = bounded.Trace([rng.randrange(320) for _ in range(140)],
                          [[[rng.randrange(320) for _ in range(rng.randrange(20, 60))] for _ in range(4)]
                           for _ in range(8)], 8, ())
    got = tiers.invariance(build, trace, lambda i=0: tiers.Deferred(i))
    assert got["verdict"] == "pass", got["failures"]
    assert got["tested"]["tier.level_reads"] >= tiers.MIN_LOADS
