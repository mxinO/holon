"""The `causal` and `window` facts of `attention.paged_verify@1` (M0.4 D4) on the std oracle, pinned by which keys can
change a query's output: with a window W, the query at position p sees the key at p − W + 1 and not the one at p − W
(llama.cpp's convention); not causal, the first of a row's new tokens sees the last. The kernels are held to the
oracle by L0 (test_attention_kernels, test_attention_kv_quant). End to end, a sliding-window target served on paged KV
decodes what its model-only build does."""
import dataclasses

import pytest
import torch
from holon_std.mixers.gqa.reference import paged_attention

from .test_engine import GREEDY, tiny_case, tiny_checkpoint  # noqa: F401 — fixtures

PAGE, N, NQ, W = 4, 40, 8, 8


def _case():
    g = torch.Generator().manual_seed(0)
    table = torch.randperm(12, generator=g)[:10].view(1, 10)          # one sequence of 40 tokens in 10 units
    kv = torch.randn(12, 2, 2, PAGE, 16, generator=g)
    q = torch.randn(NQ, 4, 16, generator=g)
    return q, kv, table


def _first_row(q, kv, table, **masks) -> torch.Tensor:
    """The output of the row's first new token (position N − NQ)."""
    return paged_attention(q, kv, table, torch.tensor([0]), torch.tensor([N]), torch.tensor([0, NQ]), **masks)[0]


def _with_key(kv, table, position: int) -> torch.Tensor:
    """`kv` with the key at `position` replaced."""
    out = kv.clone()
    out[table[0, position // PAGE], 0, :, position % PAGE] = 10 * torch.randn(2, 16)
    return out


@pytest.mark.parametrize("causal", [True, False])
def test_the_window_admits_the_last_w_positions(causal):
    q, kv, table = _case()
    p = N - NQ
    base = _first_row(q, kv, table, causal=causal, window=W)
    assert torch.equal(_first_row(q, _with_key(kv, table, p - W), table, causal=causal, window=W), base)
    assert not torch.allclose(_first_row(q, _with_key(kv, table, p - W + 1), table, causal=causal, window=W), base)
    assert not torch.allclose(_first_row(q, _with_key(kv, table, p - W), table, causal=causal), _first_row(
        q, kv, table, causal=causal))                                 # no window: every earlier key counts


@pytest.mark.parametrize("window", [0, W])
def test_not_causal_rows_see_their_later_tokens(window):
    q, kv, table = _case()
    late = _with_key(kv, table, N - 1)                                # the row's last new token
    assert torch.equal(_first_row(q, late, table, window=window), _first_row(q, kv, table, window=window))
    assert not torch.allclose(_first_row(q, late, table, causal=False, window=window),
                              _first_row(q, kv, table, causal=False, window=window))


def test_a_sliding_window_target_end_to_end(tiny_checkpoint, tiny_case, monkeypatch):  # noqa: F811
    # The GQA `window` knob reaches every attention kernel it selects: a tiny Qwen3 with a 6-token window decodes on
    # paged KV (chunked prefill, decode rows) what a model-only run of the same windowed model does — not what the
    # unwindowed one does.
    from holon.core.build import BuildContext, StepContext, build_model, load_weights
    from holon.core.engine import Engine, EngineConfig
    from holon_plugins.models.qwen3 import plugin

    prompts, plain = tiny_case
    recipe = plugin.Plugin.recipe

    def windowed(c, source):
        spec = recipe(c, source)
        return dataclasses.replace(spec, layers=tuple(dataclasses.replace(d, mixer=d.mixer.with_(window=6))
                                                      for d in spec.layers))

    monkeypatch.setattr(plugin.Plugin, "recipe", staticmethod(windowed))
    engine = Engine(EngineConfig(model=str(tiny_checkpoint), device="cpu", dtype=torch.float32, page_size=4, max_seqs=3,
                                 max_step_tokens=10, max_model_len=40, state_memory=1 << 20))
    ctx = BuildContext(engine.plugins, torch.device("cpu"), torch.float32)
    model = build_model(engine.spec, ctx)
    load_weights(model, ctx, engine.source.tensors())
    expected = []
    with torch.no_grad():
        for p in prompts:
            ids = list(p)
            for _ in range(GREEDY.max_tokens):
                ids.append(int(model(StepContext.whole_sequences([ids], "cpu"))[-1].argmax()))
            expected.append(ids[len(p):])
    assert engine.generate(prompts, GREEDY) == expected != plain
