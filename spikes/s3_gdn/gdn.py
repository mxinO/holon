"""S3 harness: one real Qwen3.5/3.8 Gated DeltaNet layer, the HF pure-torch reference and the vendored kernels.

State layouts differ: HF keeps the recurrent state as [B, H, K, V], the vendored kernels as [N, H, V, K]
(`to_fla` / `to_hf` transpose the last two dims). All references run in fp32 from the same bf16 inputs.
"""

import glob
import json
import os
import sys
from pathlib import Path

import torch
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "third_party"))

import fla_ops  # noqa: E402
from transformers import AutoConfig  # noqa: E402
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf  # noqa: E402

assert hf.chunk_gated_delta_rule is None and hf.FusedRMSNormGated is None, "HF must run its torch path (no fla)"

MODELS = {"4b": "Qwen/Qwen3.5-4B", "9b": "Qwen/Qwen3.5-9B", "27b": "Qwen/Qwen3.8-27B"}
HF_HUB = Path(os.path.expanduser("~/.cache/huggingface/hub"))


def snapshot(model: str) -> Path:
    repo = MODELS[model].replace("/", "--")
    return Path(glob.glob(str(HF_HUB / f"models--{repo}" / "snapshots" / "*"))[0])


def text_config(model: str):
    return AutoConfig.from_pretrained(snapshot(model)).text_config


def gdn_shape(model: str) -> dict:
    """Kernel-level shape: Hg q/k heads, H value heads, head dims, number of GDN layers."""
    c = json.loads((snapshot(model) / "config.json").read_text())["text_config"]
    return dict(
        Hg=c["linear_num_key_heads"],
        H=c["linear_num_value_heads"],
        K=c["linear_key_head_dim"],
        V=c["linear_value_head_dim"],
        gdn_layers=sum(t == "linear_attention" for t in c["layer_types"]),
    )


def load_layer(model: str, layer: int = 0, dtype=torch.bfloat16, device="cuda"):
    """Real weights of one GDN layer: (HF Qwen3_5GatedDeltaNet, HF input RMSNorm)."""
    cfg = text_config(model)
    assert cfg.layer_types[layer] == "linear_attention"
    root = snapshot(model)
    wmap = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"model.language_model.layers.{layer}."
    tensors = {}
    for name, shard in wmap.items():
        if name.startswith(prefix) and (".linear_attn." in name or ".input_layernorm." in name):
            with safe_open(root / shard, "pt") as f:
                tensors[name[len(prefix):]] = f.get_tensor(name)
    with torch.device(device):
        mixer = hf.Qwen3_5GatedDeltaNet(cfg, layer)
        norm = hf.Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    mixer.load_state_dict({k[len("linear_attn."):]: v for k, v in tensors.items() if k.startswith("linear_attn.")})
    norm.load_state_dict({"weight": tensors["input_layernorm.weight"]})
    return mixer.to(dtype).eval(), norm.to(dtype).eval()


def random_hidden(model_norm, batch: int, length: int, seed: int) -> torch.Tensor:
    """Random residual stream passed through the layer's real input RMSNorm (what the mixer sees)."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    w = model_norm.weight
    x = torch.randn(batch, length, w.numel(), generator=gen, device="cuda", dtype=torch.float32)
    return model_norm(x.to(w.dtype))


@torch.no_grad()
def capture_core_inputs(mixer, x: torch.Tensor) -> dict:
    """Run the HF mixer up to the core GDN call and return its inputs (q/k before the head repeat).

    q/k come out of HF `repeat_interleave`d to H heads; taking every (H // Hg)-th head undoes it.
    """
    seen = {}

    def grab(query, key, value, g, beta, **_):
        seen.update(q=query, k=key, v=value, g=g, beta=beta)
        return torch.zeros_like(value), None

    saved = mixer.chunk_gated_delta_rule
    mixer.chunk_gated_delta_rule = grab
    try:
        mixer(x)
    finally:
        mixer.chunk_gated_delta_rule = saved
    rep = mixer.num_v_heads // mixer.num_k_heads
    return dict(
        q=seen["q"][:, :, ::rep].contiguous(),
        k=seen["k"][:, :, ::rep].contiguous(),
        v=seen["v"].contiguous(),
        g=seen["g"].float().contiguous(),
        beta=seen["beta"].contiguous(),
    )


def to_fla(state_hf: torch.Tensor) -> torch.Tensor:
    return state_hf.transpose(-1, -2).contiguous()


def to_hf(state_fla: torch.Tensor) -> torch.Tensor:
    return state_fla.transpose(-1, -2).contiguous()


def _hf_inputs(q, k, v, beta, H, golden):
    # golden: fp32 from the bf16 inputs on (l2norm and output in fp32). Otherwise exactly what HF does in a bf16
    # model: l2norm in bf16, fp32 core, bf16 output — the "HF bf16" noise floor.
    rep = H // q.shape[2]
    q, k = q.repeat_interleave(rep, dim=2), k.repeat_interleave(rep, dim=2)
    if golden:
        q, k, v, beta = q.float(), k.float(), v.float(), beta.float()
    return q, k, v, beta


@torch.no_grad()
def ref_recurrent(q, k, v, g, beta, h0_hf=None, golden=True):
    """HF `torch_recurrent_gated_delta_rule` (exact token-by-token recurrence). Returns (o, state fp32)."""
    q, k, v, beta = _hf_inputs(q, k, v, beta, v.shape[2], golden)
    return hf.torch_recurrent_gated_delta_rule(
        q, k, v, g, beta, initial_state=h0_hf, output_final_state=True, use_qk_l2norm_in_kernel=True
    )


@torch.no_grad()
def ref_chunk(q, k, v, g, beta, h0_hf=None, golden=True):
    """HF `torch_chunk_gated_delta_rule` (the HF prefill path)."""
    q, k, v, beta = _hf_inputs(q, k, v, beta, v.shape[2], golden)
    return hf.torch_chunk_gated_delta_rule(
        q, k, v, g, beta, initial_state=h0_hf, output_final_state=True, use_qk_l2norm_in_kernel=True
    )


def cu_seqlens_of(lengths, device="cuda") -> torch.Tensor:
    return torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device=device)


@torch.no_grad()
def kernel_chunk(q, k, v, g, beta, h0_fla, cu_seqlens, v_head_tiled=False):
    """Vendored chunked prefill over a packed varlen batch ([1, sum(T), ...]); h0_fla is [N, H, V, K] fp32."""
    return fla_ops.chunk_gated_delta_rule(
        q, k, v, g, beta,
        initial_state=h0_fla,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
        v_head_tiled=v_head_tiled,
    )


def errors(out: torch.Tensor, ref: torch.Tensor) -> dict:
    """max |out - ref|, that normalised by max |ref|, and the relative Frobenius error."""
    d = (out.double() - ref.double()).abs()
    scale = ref.double().abs().max().clamp_min(1e-30)
    return dict(
        max_abs=d.max().item(),
        max_rel=(d.max() / scale).item(),
        fro_rel=(d.norm() / ref.double().norm().clamp_min(1e-30)).item(),
        finite=bool(torch.isfinite(out).all().item()),
    )


def device_info() -> dict:
    p = torch.cuda.get_device_properties(0)
    import triton

    return dict(
        gpu=p.name,
        sm=f"sm_{p.major}{p.minor}",
        torch=torch.__version__,
        triton=triton.__version__,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
    )
