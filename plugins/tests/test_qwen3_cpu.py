"""CPU check without downloads: a tiny random Qwen3 built from our recipe equals HF transformers in fp32."""
import torch
from holon.core.build import BuildContext, StepContext, build_model, load_weights
from holon.core.plugins import PluginSet, discover
from transformers import Qwen3Config, Qwen3ForCausalLM


def test_tiny_qwen3_matches_hf_fp32():
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, tie_word_embeddings=False,
                      attn_implementation="eager")
    hf = Qwen3ForCausalLM(cfg).eval()
    plugins = PluginSet(discover())
    spec = plugins.load("models.qwen3").recipe(cfg.to_dict(), "hf_safetensors")
    ctx = BuildContext(plugins, torch.device("cpu"), torch.float32)
    model = build_model(spec, ctx)
    load_weights(model, ctx, hf.state_dict().items())

    ids = [torch.randint(0, 512, (n,)).tolist() for n in (5, 9)]
    with torch.no_grad():
        ours = model(StepContext.whole_sequences(ids, "cpu"))
        ref = torch.cat([hf(torch.tensor([s])).logits[0] for s in ids])
    torch.testing.assert_close(ours, ref, rtol=1e-4, atol=1e-4)
