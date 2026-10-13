"""The `causal` and `window` facts of `attention.paged_verify@1` (M0.4 D4) on the std oracle, pinned by which keys can
change a query's output: with a window W, the query at position p sees the key at p − W + 1 and not the one at p − W
(llama.cpp's convention); not causal, the first of a row's new tokens sees the last. The kernels are held to the
oracle by L0 (test_attention_kernels, test_attention_kv_quant)."""
import pytest
import torch
from holon_std.mixers.gqa.reference import paged_attention

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
