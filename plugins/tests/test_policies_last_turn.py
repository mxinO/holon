"""policies.last_turn (held out after `v1-kv-tiers-api`): its rule, and on the tiny hybrid a second turn checkpointed
where its admission's untrimmed match ends (the field the proof's finding added); preemption is
`test_qwen3_5_engine.py::test_a_branch_point_outlives_preemption`."""
import types

import torch
from holon.api import SamplingParams
from holon.core.engine import Engine, EngineConfig
from holon_plugins.policies.last_turn.plugin import LastTurn
from holon_testing.qwen3_5_files import save, tiny_hf


def test_it_checkpoints_where_the_untrimmed_match_ends():
    unit = LastTurn(None)
    assert unit.next(types.SimpleNamespace(matched=192, num_computed=128)) == (192, False)
    assert unit.next(types.SimpleNamespace(matched=128, num_computed=128)) is None


def test_a_turn_extending_its_history_is_checkpointed_at_its_start(tmp_path, monkeypatch):
    """A second turn resends the first turn's prompt and output: its admission matches their blocks (192 tokens), the
    hit trims to the prompt-end checkpoint (128) and `branch_at` is 0 (the first turn was the chain's leaf); the
    policy sees `matched` 192 and the turn checkpoints there."""
    from holon_std.state.recurrent import kind

    path = save(tiny_hf(4, seed=1), tmp_path)
    engine = Engine(EngineConfig(model=str(path), device="cpu", dtype=torch.float32, page_size=4, max_seqs=2,
                                 max_step_tokens=256, max_model_len=512, state_memory=8 << 20,
                                 plugin_options={"recurrent.policy": "prompt_end,policies.last_turn@1"}))
    gen = torch.Generator().manual_seed(0)
    first = torch.randint(0, 320, (131,), generator=gen).tolist()
    out = engine.generate([first], SamplingParams(temperature=0.0, max_tokens=70, ignore_eos=True))[0]
    turn = first + out + [7, 8, 9]
    seen = []
    policy_next = kind.PromptEnd.next
    monkeypatch.setattr(kind.PromptEnd, "next", lambda self, view: seen.append(
        (view.num_computed, view.branch_at, view.matched)) or policy_next(self, view))
    engine.generate([turn], SamplingParams(temperature=0.0, max_tokens=1))
    assert seen[0] == (128, 0, 192)
    idx = engine.state.index
    assert "ssm_state" in idx.match(idx.keys(turn, 0, 3, None), turn)[2].payloads    # the block ending at 192
    engine.close()
