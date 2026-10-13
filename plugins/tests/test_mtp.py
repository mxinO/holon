"""The MTP drafter (docs/m04-spec.md D2, D3) on a tiny Qwen3.5 with an MTP block, on CPU.

- Lossless: speculative greedy decoding with `drafters.mtp` emits what plain greedy decoding does (depth 0 and 1,
  prefix hits).
- L0-drafter: the drafter's draft logits after a prefill — the first from the new token and the target's final-norm
  hidden of the position before it, the next ones chained on the head's own output — equal those of transformers' own
  Qwen3.5 decoder layer loaded with the MTP block's weights (llama.cpp's pairing: the head's row at position p takes
  token p and the target's hidden of p − 1, zeros for the first), to fp32 rounding.
"""
import pytest
import torch
from holon.api import SamplingParams
from holon.core.engine import Engine, EngineConfig
from holon_testing.qwen3_5_files import can_convert, convert, save, tiny_hf
from safetensors.torch import load_file
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf

GREEDY = SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    model = tiny_hf(4, seed=1)
    path = save(model, tmp_path_factory.mktemp("tiny_qwen3_5_mtp"), mtp=True)
    g = torch.Generator().manual_seed(2)
    prompts = [torch.randint(0, 320, (n,), generator=g).tolist() for n in (5, 70, 131, 30)]
    plain = _engine(path, max_draft_tokens=0).generate(prompts, GREEDY)
    return model, path, prompts, plain


def _engine(path, **kw) -> Engine:
    cfg = dict(model=str(path), device="cpu", dtype=torch.float32, page_size=4, max_seqs=3, max_step_tokens=256,
               max_model_len=256, state_memory=8 << 20, max_draft_tokens=3, plugin_options={"spec.k": "3"},
               slots={"drafter": "drafters.mtp", "draft_policy": "std.spec.fixed_k",
                      "verifier": "std.spec.greedy"}) | kw
    return Engine(EngineConfig(**cfg))


@pytest.mark.parametrize("overlap", [False, True], ids=["depth0", "depth1"])
def test_mtp_speculation_is_plain_greedy(tiny, overlap):
    _, path, prompts, plain = tiny
    engine = _engine(path, overlap=overlap)
    assert "mtp.kv" in engine.executor.planners
    assert engine.generate(prompts, GREEDY) == plain
    assert engine.generate(prompts, GREEDY) == plain and engine.state.tokens_saved > 0
    stats = engine.stats()                       # every verify row counted once, by k and the drafts it accepted
    rows = sum(n for key, n in stats.items() if key.startswith("spec_rows_k3_accepting_"))
    assert 0 < rows and stats["spec_drafted"] == 3 * rows      # MTP drafts every row at the maximum k
    steps, overlapped = stats["spec_verify_steps"], stats["spec_verify_steps_overlapped"]
    assert 0 < overlapped <= steps if overlap else overlapped == 0      # a step after a forced commit is at depth 0


def _reference(model, weights, tokens: list[int], hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """transformers' MTP forward over rows (tokens[p], hidden[p − 1]) at positions 0 … n−1: logits and head outputs."""
    cfg = model.config
    full = next(i for i, t in enumerate(cfg.layer_types) if t == "full_attention")
    norm = lambda name: hf.Qwen3_5RMSNorm(cfg.hidden_size, cfg.rms_norm_eps).requires_grad_(False)   # noqa: E731
    enorm, hnorm, mnorm = norm("e"), norm("h"), norm("m")
    enorm.weight.copy_(weights["mtp.pre_fc_norm_embedding.weight"])
    hnorm.weight.copy_(weights["mtp.pre_fc_norm_hidden.weight"])
    mnorm.weight.copy_(weights["mtp.norm.weight"])
    layer = hf.Qwen3_5DecoderLayer(cfg, layer_idx=full).eval()
    layer.load_state_dict({k.removeprefix("mtp.layers.0."): v for k, v in weights.items()
                           if k.startswith("mtp.layers.0.")})
    n = len(tokens)
    prev = torch.cat([torch.zeros(1, cfg.hidden_size), hidden[: n - 1]])
    x = torch.nn.functional.linear(torch.cat([enorm(model.model.embed_tokens(torch.tensor(tokens))), hnorm(prev)], -1),
                                   weights["mtp.fc.weight"])[None]
    positions = torch.arange(n)[None]
    pe = model.model.rotary_emb(x, positions[None].expand(3, 1, n))
    mask = torch.full((n, n), float("-inf")).triu(1)[None, None]
    out = mnorm(layer(x, position_embeddings=pe, attention_mask=mask, position_ids=positions))[0]
    return model.lm_head(out), out


@torch.no_grad()
def test_mtp_draft_logits_match_transformers(tiny):
    model, path, prompts, _ = tiny
    engine = _engine(path, overlap=False)
    drafter, k = engine.executor.drafter, 3
    seen, head = [], drafter.lm_head
    drafter.lm_head = lambda x: seen.append(head(x)) or seen[-1]
    prompt = prompts[1]
    engine.add(prompt, SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True))
    engine.step()                                                # the prefill, then the drafter's k drafts
    assert len(seen) == k

    weights = load_file(str(path / "model.safetensors"))
    hidden = model.model(torch.tensor([prompt])).last_hidden_state[0]           # final-norm, the lm_head input
    tokens = prompt + [int(model.lm_head(hidden[-1]).argmax())]                  # + the token the target sampled
    for i in range(k):                  # chained rows pair each draft with the head's output of the row before it
        logits, out = _reference(model, weights, tokens, hidden)
        torch.testing.assert_close(seen[i][0], logits[-1], rtol=1e-4, atol=1e-4)
        tokens, hidden = tokens + [int(logits[-1].argmax())], torch.cat([hidden, out[-1:]])


@pytest.mark.skipif(not can_convert(), reason="needs llama.cpp's converter and the Qwen3.8 tokenizer")
@torch.no_grad()
def test_mtp_from_gguf_drafts_as_from_hf(tmp_path):
    # The same tiny checkpoint, converted by llama.cpp (the MTP block as `blk.4.nextn.*` and `blk.4.*`, norms w + 1):
    # its draft logits equal the HF checkpoint's, and speculation stays lossless.
    model = tiny_hf(4, seed=1, vocab=248320)
    path = save(model, tmp_path, mtp=True)
    gguf = convert(path)
    prompt = torch.randint(0, 248320, (40,), generator=torch.Generator().manual_seed(5)).tolist()
    seen = {}
    for name, source, kind in (("hf", path, "hf_safetensors"), ("gguf", gguf, "gguf")):
        engine = _engine(source, overlap=False, source=kind)
        drafter, logits = engine.executor.drafter, []
        head = drafter.lm_head
        drafter.lm_head = lambda x, head=head, logits=logits: logits.append(head(x)) or logits[-1]
        engine.add(prompt, SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True))
        engine.step()
        seen[name] = torch.cat(logits)
        plain = _engine(source, max_draft_tokens=0, source=kind).generate([prompt], GREEDY)
        assert _engine(source, source=kind).generate([prompt], GREEDY) == plain
    torch.testing.assert_close(seen["gguf"], seen["hf"], rtol=1e-4, atol=1e-4)
