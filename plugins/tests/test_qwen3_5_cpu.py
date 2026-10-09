"""CPU check without downloads: a tiny random Qwen3.5/3.8-architecture model built from our recipe equals HF
transformers in fp32 (the HF checkpoint layout of `Qwen3_5ForConditionalGeneration`, text only), at both value/key
head ratios — Qwen3.5's 2 and Qwen3.8's 3."""
import pytest
import torch
from holon.core.build import BuildContext, StepContext, build_model, load_weights
from holon.core.plugins import PluginSet, discover

hf = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
from transformers import Qwen3_5TextConfig  # noqa: E402

assert hf.chunk_gated_delta_rule is None and hf.FusedRMSNormGated is None, "HF must run its torch GDN path"


def tiny_config(heads_v: int) -> Qwen3_5TextConfig:
    return Qwen3_5TextConfig(
        vocab_size=320, hidden_size=64, intermediate_size=96, num_hidden_layers=4, layer_types=None,
        full_attention_interval=2, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        linear_num_key_heads=2, linear_num_value_heads=heads_v, linear_key_head_dim=16, linear_value_head_dim=16,
        linear_conv_kernel_dim=4, tie_word_embeddings=False, attn_implementation="eager",
        rope_parameters={"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25,
                         "mrope_section": [2, 1, 1], "mrope_interleaved": True})


def tiny_hf(heads_v: int, seed: int = 0):
    torch.manual_seed(seed)
    model = hf.Qwen3_5ForCausalLM(tiny_config(heads_v)).eval()
    with torch.no_grad():                        # zero-initialised norms and A_log ones would hide mistakes
        for name, p in model.named_parameters():
            if name.endswith(("norm.weight", "A_log", "dt_bias")):
                p.copy_(torch.randn_like(p) * 0.5)
    return model


def checkpoint(model):
    """The state dict as `Qwen3_5ForConditionalGeneration` stores it (language model under `model.language_model.`)."""
    return [(k.replace("model.", "model.language_model.", 1), v) for k, v in model.state_dict().items()]


@pytest.mark.parametrize("heads_v", [4, 6])
def test_tiny_qwen3_5_matches_hf_fp32(heads_v):
    model_hf = tiny_hf(heads_v)
    plugins = PluginSet(discover())
    spec = plugins.load("models.qwen3_5").recipe({"text_config": model_hf.config.to_dict()}, "hf_safetensors")
    assert [d.kind for d in spec.layers] == ["linear_attention", "full_attention"] * 2
    ctx = BuildContext(plugins, torch.device("cpu"), torch.float32)
    model = build_model(spec, ctx)
    load_weights(model, ctx, checkpoint(model_hf))

    ids = [torch.randint(0, 320, (n,)).tolist() for n in (5, 70, 9)]
    with torch.no_grad():
        ours = model(StepContext.whole_sequences(ids, "cpu"))
        ref = torch.cat([model_hf(torch.tensor([s])).logits[0] for s in ids])
    torch.testing.assert_close(ours, ref, rtol=1e-4, atol=1e-4)
