"""The GPU prompt-lookup drafter (`drafters.ngram`): the lookup itself on synthetic histories, and lossless
speculation with it (and `std.spec.fixed_k`'s fallback) on a tiny Qwen3, on CPU."""
import pathlib
import types

import pytest
import torch
from holon.api import SamplingParams
from holon.core.engine import Engine, EngineConfig
from holon_plugins.drafters.ngram.plugin import NGramDrafter

from .test_engine import GREEDY, tiny_case, tiny_checkpoint  # noqa: F401 — fixtures


def _drafter(k=3, n_max=3, n_min=2, length=32):
    ctx = types.SimpleNamespace(options={"max_draft_tokens": k, "max_model_len": length, "max_seqs": 3},
                                static=lambda name, shape, dtype: torch.zeros(shape, dtype=dtype),
                                device=torch.device("cpu"))
    return NGramDrafter(ctx, n_max, n_min)


def test_lookup_takes_the_latest_longest_match():
    d = _drafter()
    hist = torch.full((4, 32), -1)
    rows = [[5, 6, 7, 8, 9, 5, 6, 7, 1, 2, 5, 6, 7],      # last 3 (5 6 7) last seen at 5: drafts 1 2 5
            [1, 2, 3, 4, 9, 9, 3, 4],                      # no 3-gram (9 3 4); 2-gram (3 4) at 2: drafts 9 9 3
            [1, 2, 3, 4, 5],                               # nothing repeats
            [7, 8, 9, 7, 8]]                               # (7 8) at 0, followed by 9 7 8: drafts 9 7 8
    for r, toks in enumerate(rows):
        hist[r, : len(toks)] = torch.tensor(toks)
    drafts, n = d._lookup(hist, torch.tensor([len(t) for t in rows]))
    assert n.tolist() == [3, 3, 0, 3]
    assert drafts[0].tolist() == [1, 2, 5] and drafts[1].tolist() == [9, 9, 3] and drafts[3].tolist() == [9, 7, 8]


def test_lookup_stops_at_the_current_length():
    d = _drafter(k=4)
    hist = torch.full((1, 32), -1)
    hist[0, :6] = torch.tensor([3, 4, 5, 6, 3, 4])                 # (3 4) at 0, then 5 6 3 4: only 4 known tokens
    hist[0, 6:9] = torch.tensor([7, 7, 7])                         # stale tokens past the length: never drafted
    drafts, n = d._lookup(hist, torch.tensor([6]))
    assert n.tolist() == [4] and drafts[0].tolist() == [5, 6, 3, 4]


def test_ngram_speculation_is_plain_greedy(tiny_checkpoint, tiny_case):  # noqa: F811
    prompts, expected = tiny_case
    prompts = [p + p[:6] for p in prompts]                         # repeats to find
    cfg = dict(model=str(tiny_checkpoint), device="cpu", dtype=torch.float32, page_size=4, max_seqs=3,
               max_step_tokens=10, max_model_len=64, state_memory=1 << 20)
    plain = Engine(EngineConfig(**cfg)).generate(prompts, GREEDY)
    for overlap in (False, True):
        engine = Engine(EngineConfig(**cfg, overlap=overlap, max_draft_tokens=3,
                                     slots={"drafter": "drafters.ngram", "draft_policy": "std.spec.fixed_k",
                      "verifier": "std.spec.greedy"},
                                     plugin_options={"spec.k": "3", "spec.fallback": 2}))
        seen, observe = [], engine.policy.observe

        def record(seq, k, n_draft, n_accept, seen=seen, observe=observe):
            seen.append((n_draft, n_accept))
            observe(seq, k, n_draft, n_accept)

        engine.policy.observe = record
        assert engine.generate(prompts, GREEDY) == plain
        assert any(n for n, _ in seen) and all(a <= 3 for _, a in seen)       # lookups hit (the prompts repeat)


SNAP = max(pathlib.Path.home().glob(".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*"), default=None)


@pytest.mark.skipif(SNAP is None or not torch.cuda.is_available(), reason="needs Qwen3-0.6B + GPU")
def test_ngram_speculation_captures_graphs():
    # The drafter runs inside the captured decode and verify graphs: nothing in it may sync with the host. Greedy
    # output stays plain greedy's up to bf16 batch-shape rounding (an agreement bound).
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(SNAP)
    text = "def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a - b\n\ndef mul(a, b):\n"
    prompts = [tok(text)["input_ids"], tok("1, 2, 3, 4, 5, 6, 7, 8,")["input_ids"]]
    params = SamplingParams(temperature=0.0, max_tokens=32, ignore_eos=True)
    cfg = dict(model=str(SNAP), dtype=torch.bfloat16, max_seqs=4, max_model_len=256)
    plain = Engine(EngineConfig(**cfg)).generate(prompts, params)
    engine = Engine(EngineConfig(**cfg, max_draft_tokens=3, plugin_options={"spec.k": "3", "spec.fallback": 2},
                                 slots={"drafter": "drafters.ngram", "draft_policy": "std.spec.fixed_k",
                                        "verifier": "std.spec.greedy"}))
    assert engine.executor.graphs is not None and len(engine.executor.graphs) > 0
    spec = engine.generate(prompts, params)
    agree = [next((i for i, (x, y) in enumerate(zip(a, b, strict=True)) if x != y), len(a))
             for a, b in zip(plain, spec, strict=True)]
    assert sum(agree) >= 0.75 * len(agree) * params.max_tokens, agree
