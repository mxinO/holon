"""The Gated DeltaNet oracles (std.mixers.gdn reference) against transformers' own torch code for Qwen3.5/3.8: the
recurrence (`torch_recurrent_gated_delta_rule`), the gated norm (`Qwen3_5RMSNormGated`) and the convolution (HF's
`nn.Conv1d` path), on the CPU in f32 — the oracle is checked against an independent source before any kernel is."""
import pytest
import torch
import torch.nn.functional as F
from holon_std.mixers.gdn import reference as ref

hf = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")


def _inputs(t: int, hk: int, hv: int, dk: int, dv: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    q, k = torch.randn(t, hk, dk, generator=g), torch.randn(t, hk, dk, generator=g)
    v = torch.randn(t, hv, dv, generator=g)
    a, b = torch.randn(t, hv, generator=g), torch.randn(t, hv, generator=g)
    a_log, dt_bias = torch.randn(hv, generator=g) * 0.5, torch.randn(hv, generator=g)
    return q, k, v, a, b, a_log, dt_bias


@pytest.mark.parametrize(("hk", "hv"), [(2, 4), (2, 6)])          # head ratios 2 (Qwen3.5) and 3 (Qwen3.8)
def test_delta_rule_matches_transformers(hk, hv):
    dk = dv = 16
    q, k, v, a, b, a_log, dt_bias = _inputs(11, hk, hv, dk, dv, seed=hv)
    h0 = torch.randn(hv, dv, dk, generator=torch.Generator().manual_seed(1))
    state = torch.zeros(3, hv, dv, dk)
    state[2] = h0
    o = ref.gated_delta_rule(q, k, v, a, b, a_log, dt_bias, state, torch.tensor([2]), torch.tensor([0, 11]),
                             torch.tensor([True]), "grouped")
    g = -a_log.exp() * F.softplus(a + dt_bias)
    r = hv // hk
    want, s = hf.torch_recurrent_gated_delta_rule(
        q.repeat_interleave(r, 1)[None], k.repeat_interleave(r, 1)[None], v[None], g[None], torch.sigmoid(b)[None],
        initial_state=h0.transpose(-1, -2)[None], output_final_state=True, use_qk_l2norm_in_kernel=True)
    torch.testing.assert_close(o, want[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(state[2], s[0].transpose(-1, -2), rtol=1e-5, atol=1e-5)
    assert not state[0].any() and not state[1].any()                  # other units untouched


def test_tiled_order_is_a_permutation_of_grouped():
    hk, hv, dk = 2, 6, 8
    q, k, v, a, b, a_log, dt_bias = _inputs(5, hk, hv, dk, dk, seed=3)
    perm = torch.argsort(ref.head_map(hk, hv, "tiled") * hv + torch.arange(hv))     # v-heads sorted by their k-head
    grouped = ref.gated_delta_rule(q, k, v[:, perm], a[:, perm], b[:, perm], a_log[perm], dt_bias[perm],
                                   torch.zeros(1, hv, dk, dk), torch.tensor([0]), torch.tensor([0, 5]),
                                   torch.tensor([False]), "grouped")
    tiled = ref.gated_delta_rule(q, k, v, a, b, a_log, dt_bias, torch.zeros(1, hv, dk, dk), torch.tensor([0]),
                                 torch.tensor([0, 5]), torch.tensor([False]), "tiled")
    torch.testing.assert_close(tiled[:, perm], grouped)


def test_no_initial_state_ignores_the_unit_and_rows_continue():
    hk, hv, d = 2, 4, 8
    q, k, v, a, b, a_log, dt_bias = _inputs(9, hk, hv, d, d, seed=5)
    stale = torch.full((1, hv, d, d), 7.0)
    one = ref.gated_delta_rule(q, k, v, a, b, a_log, dt_bias, stale.clone(), torch.tensor([0]), torch.tensor([0, 9]),
                               torch.tensor([False]), "grouped")
    clean = ref.gated_delta_rule(q, k, v, a, b, a_log, dt_bias, torch.zeros(1, hv, d, d), torch.tensor([0]),
                                 torch.tensor([0, 9]), torch.tensor([True]), "grouped")
    torch.testing.assert_close(one, clean)
    state = torch.zeros(1, hv, d, d)                                    # 9 tokens as 4 + 5 continue exactly
    first = ref.gated_delta_rule(q[:4], k[:4], v[:4], a[:4], b[:4], a_log, dt_bias, state, torch.tensor([0]),
                                 torch.tensor([0, 4]), torch.tensor([False]), "grouped")
    second = ref.gated_delta_rule(q[4:], k[4:], v[4:], a[4:], b[4:], a_log, dt_bias, state, torch.tensor([0]),
                                  torch.tensor([0, 5]), torch.tensor([True]), "grouped")
    torch.testing.assert_close(torch.cat([first, second]), clean)


def test_causal_conv1d_matches_conv1d_and_carries_state():
    c, kernel, t = 6, 4, 10
    gen = torch.Generator().manual_seed(7)
    x, w = torch.randn(t, c, generator=gen), torch.randn(c, kernel, generator=gen)
    conv = torch.nn.Conv1d(c, c, kernel, groups=c, padding=kernel - 1, bias=False)
    conv.weight.data = w[:, None, :]
    want = F.silu(conv(x.T[None])[0, :, :t]).T
    state = torch.full((2, c, kernel - 1), 3.0)
    got = ref.causal_conv1d(x, w, state, torch.tensor([1]), torch.tensor([0, t]), torch.tensor([False]))
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(state[1], x[-(kernel - 1):].T)
    assert (state[0] == 3.0).all()
    split = torch.zeros(1, c, kernel - 1)                               # 10 tokens as 3 + 7 continue exactly
    a = ref.causal_conv1d(x[:3], w, split, torch.tensor([0]), torch.tensor([0, 3]), torch.tensor([False]))
    b = ref.causal_conv1d(x[3:], w, split, torch.tensor([0]), torch.tensor([0, 7]), torch.tensor([True]))
    torch.testing.assert_close(torch.cat([a, b]), want, rtol=1e-5, atol=1e-5)


def test_gated_rmsnorm_matches_transformers():
    gen = torch.Generator().manual_seed(9)
    o, z = torch.randn(5, 3, 16, generator=gen).bfloat16(), torch.randn(5, 3, 16, generator=gen).bfloat16()
    norm = hf.Qwen3_5RMSNormGated(16, eps=1e-6)
    norm.weight.data = torch.randn(16, generator=gen).bfloat16()
    want = norm(o.reshape(-1, 16), z.reshape(-1, 16)).reshape(5, 3, 16)
    assert torch.equal(ref.gated_rmsnorm(o, z, norm.weight.data, 1e-6), want)
