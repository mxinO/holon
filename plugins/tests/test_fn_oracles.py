"""FN's op oracles against HF's modules (transformers 5.19.0, `models/qwen4_exp/modeling_qwen4_exp.py`), in the golden
venv (`local/envs/golden-qwen4`, with the plugins installed), on the CPU in float32 with the tiny model's shapes
(`holon_testing.qwen4exp_files.tiny_config`) and random weights. The oracles keep float32 where HF rounds to the
model dtype (the contracts' choice, docs/dev/v1-flash-next.md D4, D6), so in float32 the two compute the same function:

- `hc_gr_mix@1` / `hc_gr_combine@1`: `Qwen4ExpTextGatedResidual` (a block's and the output mixer's) and the decoder
  layer's `hyper_input + injection`.
- `qwen4_exp.ple_rows@1`: `Qwen4ExpTextNGramEmbedding`'s table rows (its own multipliers, primes and offsets), with
  EOS at the start, inside and twice in a row; bitwise. Rows split into chunks carry the history: bitwise one shot.
- `qwen4_exp.ple_block@1`: `Qwen4ExpTextPLELayer` given the same embeddings (dilation 3); in two chunks with the
  state carried as well.
- `qsa_index@1` + `qsa_select@1`: `Qwen4ExpTextQSAIndexer`'s selected-token mask, under the judge's tie rule (D2),
  for every query of a 150-token prompt at budget 16 (sparse past 19 tokens; boundary ties occur), with the keys
  indexed in two chunks through a shuffled page table;
  `qsa_attend@1`: HF's eager attention under that mask.

Skips without transformers' qwen4_exp.
"""
import math
import warnings

import pytest
import torch
import transformers
from holon_judge import l0

SKIP = ("FN's op oracles against HF (acceptance 1(a)) did not run: they need transformers with qwen4_exp (5.19.0, "
        "the golden venv): run `pytest plugins/tests/test_fn_oracles.py` there (docs/dev/release.md §22)")
if not hasattr(transformers, "Qwen4ExpTextConfig"):
    warnings.warn(SKIP, stacklevel=1)       # loud: a skipped proof is not a passed one
pytestmark = pytest.mark.skipif(not hasattr(transformers, "Qwen4ExpTextConfig"), reason=SKIP)

L, EPS = 150, 1e-6


@pytest.fixture(scope="module")
def setup():
    from holon_testing import qwen4exp_files as q
    from transformers.models.qwen4_exp import modeling_qwen4_exp as hf

    return q, hf, q.tiny_config()


def _randomize(module: torch.nn.Module, seed: int) -> torch.nn.Module:
    g = torch.Generator().manual_seed(seed)
    for p in module.parameters():
        fan_in = p.shape[-1] if p.dim() > 1 else 1
        p.data = torch.randn(p.shape, generator=g) * (0.1 if p.dim() == 1 else 1 / math.sqrt(fan_in))
    return module.float().eval()


def _check(ours, ref, n, what):
    report = l0.check(ours, ref, torch.float32, n)
    assert report.passed, (what, report)


@torch.no_grad()
def test_hc(setup):
    from holon_std.hyper import reference as ref

    _, hf, cfg = setup
    n, h = cfg.hc_count, cfg.hidden_size
    r = torch.randn(L, n * h) * torch.linspace(0.5, 4, n).repeat_interleave(h)
    for combine in (True, False):
        gr = _randomize(hf.Qwen4ExpTextGatedResidual(cfg, use_combine=combine), 1)
        out = gr(r)
        inject = gr.block_inject_weight.weight if combine else None
        x, w = ref.hc_gr_mix(r, gr.hc_norm.weight, gr.input_mix_weight_down.weight, gr.input_mix_weight_up.weight,
                             inject, None, None, n, cfg.rms_norm_eps, torch.float32)
        _check(x, out[0] if combine else out, n * h, "mix x")
        if combine:
            _check(w, out[2], n * h, "mix w")
            y = torch.randn(L, h)
            ours = r.clone()
            ref.hc_gr_combine(ours, y, out[2])
            _check(ours, out[1] + (y.unsqueeze(-2) * out[2].unsqueeze(-1)).flatten(-2), 1, "combine")


def _ids(eos: int) -> torch.Tensor:
    ids = torch.randint(1000, 200000, (L,), generator=torch.Generator().manual_seed(2))
    ids[[0, 40, 41, 97]] = eos                                       # at the start, twice in a row, inside
    return ids


def _rows(n: int, has_init: bool):
    return (torch.zeros(1, dtype=torch.int32), torch.tensor([0, n], dtype=torch.int32),
            torch.tensor([has_init]))


@torch.no_grad()
def test_ple_rows(setup):
    from holon_plugins.models.qwen4_exp import reference as ref

    q, hf, cfg = setup
    emb = hf.Qwen4ExpTextNGramEmbedding(cfg, cfg.ple_embed_dim, layer_idx=1, ple_layer_index=0)
    seen = []

    class Capture(torch.nn.Module):                                  # the table lookup: keep the rows
        weight = torch.zeros(1)

        def forward(self, rows):
            seen.append(rows)
            return torch.zeros(*rows.shape, 1)

    emb.ngram_embedding = Capture()
    ids = _ids(q.EOS)
    emb(ids[None], None)
    tables = emb.layer_multipliers, emb.ngram_heads_vocab_sizes, emb.ngram_heads_offsets
    facts = cfg.ngram_size, cfg.heads_per_ngram, q.EOS
    history = torch.zeros(1, 2, dtype=torch.long)
    one = ref.ple_rows(ids, history, *_rows(L, False), *tables, *facts)
    assert torch.equal(one, seen[0][0])
    chunked, done = [], 0
    for n in (41, 1, 1, 57, L - 100):                                # the history carried across chunks
        chunked.append(ref.ple_rows(ids[done:done + n], history, *_rows(n, done > 0), *tables, *facts))
        done += n
    assert torch.equal(torch.cat(chunked), one) and torch.equal(history[0], ids[-2:])


@torch.no_grad()
def test_ple_block(setup):
    from holon_plugins.models.qwen4_exp import reference as ref

    q, hf, cfg = setup
    layer = _randomize(hf.Qwen4ExpTextPLELayer(cfg, layer_idx=1, ple_layer_index=0), 3)
    g = torch.Generator().manual_seed(4)
    e = torch.randn(1, L, cfg.ple_embed_dim, generator=g)
    layer.ple_embedding.forward = lambda *args: e
    width = cfg.hc_count * cfg.hidden_size
    r = torch.randn(L, width, generator=g) * 2
    expected = layer(r[None], None, None)[0]
    key, value = layer.key_proj(e)[0], layer.value_proj(e)[0]
    weights = layer.norm_key.weight, layer.norm_query.weight, layer.norm_conv.weight, layer.conv1d.weight[:, 0]
    state = torch.full((1, layer.short_conv_state_len, width), float("nan"))     # without has_init: never read
    ours = r.clone()
    ref.ple_block(key, value, ours, *weights, state, *_rows(L, False), cfg.hc_count, cfg.ngram_size, cfg.rms_norm_eps)
    _check(ours - r, expected, cfg.hidden_size, "ple_block")
    chunked, first = r.clone(), 61
    for a, b in ((0, first), (first, L)):
        part = chunked[a:b].clone()
        ref.ple_block(key[a:b], value[a:b], part, *weights, state, *_rows(b - a, a > 0), cfg.hc_count,
                      cfg.ngram_size, cfg.rms_norm_eps)
        chunked[a:b] = part
    assert torch.equal(chunked, ours)


@torch.no_grad()
def test_qsa(setup):
    from holon_std.mixers.qsa import reference as ref

    _, hf, cfg = setup
    from holon_judge.qwen4exp import wrap_indexers

    indexer = _randomize(hf.Qwen4ExpTextQSAIndexer(cfg, layer_idx=3), 5)
    ties = []
    assert wrap_indexers(torch.nn.ModuleList([indexer]), ties.append) == 1     # D2's rule: ties to the lower block
    rotary = hf.Qwen4ExpTextRotaryEmbedding(cfg)
    g = torch.Generator().manual_seed(6)
    hidden = torch.randn(1, L, cfg.hidden_size, generator=g)
    cos, sin = rotary(hidden, torch.arange(L)[None])
    causal = torch.ones(L, L, dtype=torch.bool).tril()[None, None]
    selected = indexer(hidden, (cos, sin), causal, None)[0, 0]

    hi, dk, ratio, page = indexer.index_n_heads, indexer.index_head_dim, indexer.compress_ratio, 8
    table_rows = ref.cos_sin(L, cos.shape[-1], cfg.rope_parameters["rope_theta"])
    assert torch.allclose(table_rows[:, : cos.shape[-1] // 2], cos[0, :, : cos.shape[-1] // 2], atol=1e-6)
    qk = indexer.index_qk_proj(hidden)[0]
    q = torch.stack([ref.rope(x, p, table_rows) for p, x in
                     enumerate(indexer.q_layernorm(qk[:, : hi * dk].view(L, hi, dk)))])
    k = qk[:, hi * dk:]
    pages = -(-L // page)
    table = torch.randperm(pages + 3, generator=g)[:pages].int()[None]
    raw = torch.zeros(pages + 3, page, dk)
    pooled = torch.zeros(pages + 3, page // ratio, dk)
    seq, pos = torch.zeros(L, dtype=torch.int32), torch.arange(L)
    for a, b in ((0, 37), (37, L)):                                  # the second chunk completes a block of the first
        ref.qsa_index(k[a:b], raw, pooled, table, seq[a:b], pos[a:b], indexer.k_layernorm.weight, table_rows, ratio,
                      cfg.rms_norm_eps)
    blocks = ref.qsa_select(q, pooled, table, seq, pos, ratio, indexer.token_budget)
    ours = torch.zeros(L, L, dtype=torch.bool)
    for t in range(L):
        tokens = [ratio * int(b) + i for b in blocks[t] if b >= 0 for i in range(ratio) if ratio * int(b) + i <= t]
        ours[t, tokens] = True
    assert torch.equal(ours, selected)
    assert (ours.sum(-1) < torch.arange(1, L + 1)).any() and any(ties)   # sparse rows, boundary ties (ReLU's zeros)

    hq, hkv, d = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    qa, ka, va = (torch.randn(L, h, d, generator=g) for h in (hq, hkv, hkv))
    kv = torch.zeros(pages + 3, 2, hkv, page, d)
    for t in range(L):
        kv[table[0, t // page], :, :, t % page] = torch.stack((ka[t], va[t]))
    attended = ref.qsa_attend(qa, kv, blocks, table, seq, pos, None, ratio)

    class Module:
        num_key_value_groups, training = hq // hkv, False

    heads_first = [x.transpose(0, 1)[None] for x in (qa, ka, va)]
    expected, _ = hf.eager_attention_forward(Module, *heads_first, torch.where(selected, 0.0, -torch.inf)[None, None],
                                             scaling=d ** -0.5)
    _check(attended, expected[0], L, "qsa_attend")
