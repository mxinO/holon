"""Walking skeleton: Qwen3-0.6B built from its recipe on the reference units matches HF transformers."""
import json
import pathlib

import pytest
import torch
from holon.api import canonical_hash
from holon.core.build import EagerStepCtx, build_model, load_weights
from holon.core.plugins import PluginSet, discover

SNAP = max(pathlib.Path.home().glob(".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*"), default=None)
pytestmark = pytest.mark.skipif(SNAP is None or not torch.cuda.is_available(), reason="needs Qwen3-0.6B + GPU")


def test_qwen3_matches_hf():
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM

    device = torch.device("cuda")
    cfg = json.loads((SNAP / "config.json").read_text())
    plugins = PluginSet(discover())
    model_plugin = plugins.load("models.qwen3")
    spec = model_plugin.recipe(cfg, "hf_safetensors")
    assert canonical_hash(spec) == canonical_hash(model_plugin.recipe(cfg, "hf_safetensors"))

    model, ctx = build_model(spec, plugins, device)
    with safe_open(SNAP / "model.safetensors", "pt", device="cuda") as f:
        load_weights(model, ctx, ((k, f.get_tensor(k)) for k in f.keys()))

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(SNAP)
    texts = ["The quick brown fox jumps over the lazy dog because it wanted to see what was on the other side.",
             "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n"]
    ids = [tok(t)["input_ids"] for t in texts]

    def hf_logits(dtype):
        hf = AutoModelForCausalLM.from_pretrained(SNAP, dtype=dtype).to(device).eval()
        out = torch.cat([hf(torch.tensor([s], device=device)).logits[0].float() for s in ids])
        del hf
        return out

    with torch.no_grad():
        ours = model(EagerStepCtx.whole_sequences(ids, device))
        gold, hf_bf16 = hf_logits(torch.float32), hf_logits(torch.bfloat16)

    def metrics(x):
        agree = (x.argmax(-1) == gold.argmax(-1)).float().mean().item()
        kl = torch.nn.functional.kl_div(x.log_softmax(-1), gold.log_softmax(-1), log_target=True,
                                        reduction="none").sum(-1).mean().item()
        return agree, kl

    (a_ours, kl_ours), (a_hf, kl_hf) = metrics(ours), metrics(hf_bf16)
    print(f"vs HF fp32 — ours: top-1 {a_ours:.4f} KL {kl_ours:.2e} | HF bf16: top-1 {a_hf:.4f} KL {kl_hf:.2e}")
    # Not worse than HF's own bf16 path (the noise floor), with a small margin.
    assert a_ours >= a_hf - 0.02 and kl_ours <= 2 * kl_hf + 1e-4
