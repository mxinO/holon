"""policies.last_turn (held out after `v1-kv-tiers-api`): its rule on a view carrying the untrimmed match length, the
proof's finding on a real engine (a turn that extends its history admits with that length nowhere in `SeqView`), and
the unit's refusal on a core without the field."""
import types

import pytest
import torch
from holon.api import SamplingParams
from holon.core.engine import Engine, EngineConfig
from holon_plugins.policies.last_turn.plugin import LastTurn, core_has_matched
from holon_testing.qwen3_5_files import save, tiny_hf


def test_it_checkpoints_where_the_untrimmed_match_ends():
    unit = LastTurn.__new__(LastTurn)                                   # the rule, apart from the core's check
    assert unit.next(types.SimpleNamespace(matched=192, num_computed=128)) == (192, False)
    assert unit.next(types.SimpleNamespace(matched=128, num_computed=128)) is None


def test_the_finding_seqview_lacks_the_untrimmed_match(tmp_path, monkeypatch):
    """A second turn resends the first turn's prompt and output: its admission matches their blocks (192 tokens),
    the hit trims to the prompt-end checkpoint (128), and the view shows 128 with `branch_at` 0 (the first turn was
    the chain's leaf). Building the unit on this core refuses, naming the field."""
    from holon_std.state.recurrent import kind

    path = save(tiny_hf(4, seed=1), tmp_path)
    cfg = dict(model=str(path), device="cpu", dtype=torch.float32, page_size=4, max_seqs=2, max_step_tokens=256,
               max_model_len=512, state_memory=8 << 20)
    engine = Engine(EngineConfig(**cfg))
    gen = torch.Generator().manual_seed(0)
    first = torch.randint(0, 320, (131,), generator=gen).tolist()
    params = SamplingParams(temperature=0.0, max_tokens=70, ignore_eos=True)
    out = engine.generate([first], params)[0]
    seen = []
    policy_next = kind.PromptEnd.next
    monkeypatch.setattr(kind.PromptEnd, "next", lambda self, view: seen.append(
        (view.num_computed, view.branch_at, len(view.tokens))) or policy_next(self, view))
    turn = first + out + [7, 8, 9]
    idx = engine.state.index
    matched = len(idx.match(idx.keys(turn, 0, (len(turn) - 1) // idx.block, None), turn)) * idx.block
    assert matched == 192                                              # what the turn's admission matches
    engine.generate([turn], SamplingParams(temperature=0.0, max_tokens=1))
    assert seen[0][:2] == (128, 0)                                     # all a policy sees of it
    engine.close()
    assert not core_has_matched()
    with pytest.raises(RuntimeError, match="needs SeqView.matched"):
        Engine(EngineConfig(**cfg, plugin_options={"recurrent.policy": "prompt_end,policies.last_turn@1"}))
