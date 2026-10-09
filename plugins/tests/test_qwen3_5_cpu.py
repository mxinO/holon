"""CPU check without downloads: a tiny random Qwen3.5/3.8-architecture model built from our recipe equals HF
transformers in fp32 (the HF checkpoint layout of `Qwen3_5ForConditionalGeneration`, text only), at both value/key
head ratios — Qwen3.5's 2 and Qwen3.8's 3."""
import pytest
import torch
from holon.core.build import BuildContext, StepContext, build_model, load_weights
from holon.core.plugins import PluginSet, discover

hf = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
from holon_testing.qwen3_5_files import checkpoint, tiny_hf  # noqa: E402

assert hf.chunk_gated_delta_rule is None and hf.FusedRMSNormGated is None, "HF must run its torch GDN path"


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
