"""The tiny GLM-5.3-Flash (`glm5_next`) models (docs/v1-glm.md D2, §6.1): random weights in transformers 5.19.0's text
model (`holon_judge.glm5next`'s causal LM over `Glm5NextTextModel`; run in the golden venv, `local/envs/golden-glm`),
saved as the released checkpoint stores them (its config and names, no vision tower, no MTP layer: D7 defers MTP);
its GGUF as llama.cpp's own converter writes it (an oracle for the conversion's conventions that shares neither the
engine's nor the judge's beliefs: `ssm_a`, the MLA split, the conv layouts, the mHC names), and for the GPU fixture
its Q8_0 copy by `llama-quantize`.

Two fixtures (D2, lead call 3): `cpu` (f32, H = 64, MLA dims distinct powers of two: q_lora 32, kv_lora 64, nope 32,
v 16) and `gpu` (H = I = 512, kv_lora 128, nope 64, v 32, KDA 2 x 128, indexer 4 x 64, the head scaled to a logit
std >= 2). Both: 8 layers (DSA at 3 and 7, the dense FFN at 0, KDA elsewhere), 4 streams, 8 experts top-2 with a
shared expert, the indexer's pools of 4 with top-k 8 (sparse from t + 1 > 11), the real 154,880-token vocabulary.
Random init hides conventions: `A_log`, `dt_bias`, the selection bias, mHC `base` / `scale`, the pooling gate and
APE, every norm weight and LayerNorm bias and the pad row are randomised and asserted non-default; the comb logits
are scaled so that most rows are peaked (|s2 * m| >= 8); the FFN weights so that the clamp binds on >= 5% of the rows
of every path (dense, routed, shared; the gate's bound and both of the up projection's). `stats` measures both.

The GGUFs (≈ 160 MB / 0.6 GB at this vocabulary) are not committed: this script rebuilds them byte for byte and
checks their sha256 against `GGUF_SHA256`. What is committed is the small CPU fixture (`FIXTURES`, written with
`--fixtures`): per test prompt HF's ids (the prompt and its 24 greedy tokens), the head's input x, and each DSA
layer's selected tokens per position, from HF's one-shot forward under D2's tie rule; the boundary ties, clamp bind
rates and comb peaking in its metadata. Importing this module needs no transformers with `glm5_next`; building does.

    python -m holon_testing.glm5next_files --out <dir> --llama <llama.cpp 27c7a76> --quantize <its llama-quantize> \
        [--kind cpu|gpu|both] [--fixtures]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

import torch
from safetensors.torch import save_file

from . import gguf_files

TOKENIZER = ("zai-org/GLM-5.3-Flash", "46a050bec13002ad29ee795c16e61cf4fa3c4a9c")
EOS = PAD = 154820
NAME = "tiny-glm5next"
GGUF_SHA256 = {"cpu": "7114e563b206c3aaccfae604f73158eb3514147b11f72be0b087a7c90ec64e79",       # f32
               "gpu": "8f439ffad222f0fb7284510dc5219f5fde1eb032fc6e446bfe001bba8b1f7d86"}       # Q8_0
FIXTURES = pathlib.Path(__file__).with_name("glm5next_fixtures.safetensors")
PROMPT, DECODE, EOS_AT = 100, 24, (40, 71)           # §6.1: 100-token prompts with an EOS, then 24 decode steps
GPU_PROMPT, GPU_PROMPTS = 300, 3                     # §6.1: 3 prompts of 300 tokens
SIZES = {
    "cpu": dict(hidden_size=64, intermediate_size=64, moe_intermediate_size=64, num_attention_heads=2,
                q_lora_rank=32, kv_lora_rank=64, qk_nope_head_dim=32, v_head_dim=16, linear_head_dim=32,
                index_n_heads=4, index_head_dim=16),
    "gpu": dict(hidden_size=512, intermediate_size=512, moe_intermediate_size=512, num_attention_heads=2,
                q_lora_rank=128, kv_lora_rank=128, qk_nope_head_dim=64, v_head_dim=32, linear_head_dim=128,
                index_n_heads=4, index_head_dim=64)}
LAYERS = 8
CLAMP_SIGMA = {"cpu": 3.3, "gpu": 2.9}   # an FFN pre-activation's std: the clamp at ±10 binds on 10-20% of rows
COMB_SIGMA = 12.0            # s2 * m's std: |s2 * m| >= 8 on about half the entries
LOGIT_STD = 2.5              # the GPU fixture's head (A-N2)
# What the release keeps in F32 (transformers' `_keep_in_fp32_modules_strict` and the mHC base and scale): left at full
# f32 in both fixtures, so a recipe that loads one of them in the model dtype shows (step 2's review A-m1)
F32 = ("e_score_correction_bias", "conv1d", "dt_bias", "A_log", "hc.base", "hc.scale")


def tokenizer_dir() -> pathlib.Path:
    return gguf_files.tokenizer_dir(*TOKENIZER)


def checkpoint_config(kind: str) -> dict:
    """The released `config.json` (its keys, the converter reads them) at the tiny sizes: no MTP layer, the vision
    tower's config left as released (no tensors of it are saved)."""
    cfg = json.loads((tokenizer_dir() / "config.json").read_text())
    cfg.pop("quantization_config", None)
    text, s = cfg["text_config"], SIZES[kind]
    full = [i for i in range(LAYERS) if i % 4 == 3]
    text.update(s, vocab_size=154880, num_hidden_layers=LAYERS, num_key_value_heads=s["num_attention_heads"],
                n_routed_experts=8, num_experts_per_tok=2, n_shared_experts=1, first_k_dense_replace=1,
                qk_head_dim=s["qk_nope_head_dim"], index_topk=8, index_kpool=4, num_nextn_predict_layers=0,
                layer_types=["deepseek_sparse_attention" if i in full else "linear_attention" for i in range(LAYERS)],
                mlp_layer_types=["dense"] + ["sparse"] * (LAYERS - 1), indexer_types=["full"] * LAYERS,
                linear_attn_config=text["linear_attn_config"] | {
                    "num_heads": s["num_attention_heads"], "head_dim": s["linear_head_dim"],
                    "kda_layers": [i for i in range(LAYERS) if i not in full], "full_attn_layers": full})
    return cfg


def text_config(kind: str):
    from transformers import Glm5NextConfig

    return Glm5NextConfig.from_dict(checkpoint_config(kind)).text_config


def tiny_hf(kind: str = "cpu", seed: int = 0) -> torch.nn.Module:
    """The random model (D2's randomised init, the FFN and comb scales above), its values bf16 as the released
    checkpoint holds them."""
    from holon_judge.glm5next import register
    from transformers.models.glm5_next import modeling_glm5_next as hf

    torch.manual_seed(seed)
    config = text_config(kind)
    model = register()(config).eval()
    h = config.hidden_size
    picked = []

    def randomise(p: torch.Tensor, value: torch.Tensor) -> None:
        p.copy_(value)
        picked.append(p)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, (hf.Glm5NextTextRMSNorm, hf.Glm5NextTextRMSNormGated)):
                randomise(m.weight, 1 + 0.3 * torch.randn_like(m.weight))
            elif isinstance(m, torch.nn.LayerNorm):
                randomise(m.weight, 1 + 0.3 * torch.randn_like(m.weight))
                randomise(m.bias, 0.1 * torch.randn_like(m.bias))
            elif isinstance(m, hf.Glm5NextTextForgetGate):
                randomise(m.A_log, 0.5 * torch.randn_like(m.A_log))
                randomise(m.dt_bias, 0.5 * torch.randn_like(m.dt_bias))
            elif isinstance(m, hf.Glm5NextTextTopkRouter):
                randomise(m.e_score_correction_bias, 0.1 * torch.randn_like(m.e_score_correction_bias))
            elif isinstance(m, hf.Glm5NextTextHyperConnection):
                n = m.hc_mult
                comb = COMB_SIGMA / (m.fn[2 * n:].std() * (n * h) ** 0.5)
                randomise(m.base, 0.5 * torch.randn_like(m.base))
                randomise(m.scale, torch.stack([1 + 0.3 * torch.randn(()), 1 + 0.3 * torch.randn(()), comb]))
            elif isinstance(m, hf.Glm5NextTextIndexer):
                randomise(m.index_kpool_compress_gate, torch.randn_like(m.index_kpool_compress_gate) / h ** 0.5)
                randomise(m.index_kpool_compress_ape, 0.5 * torch.randn_like(m.index_kpool_compress_ape))
            elif isinstance(m, hf.Glm5NextTextExperts):
                m.gate_up_proj.normal_(0, CLAMP_SIGMA[kind] / h ** 0.5)
            elif isinstance(m, hf.Glm5NextTextMLP):
                m.gate_proj.weight.normal_(0, CLAMP_SIGMA[kind] / h ** 0.5)
                m.up_proj.weight.normal_(0, CLAMP_SIGMA[kind] / h ** 0.5)
        emb = model.model.embed_tokens.weight
        randomise(emb[PAD], emb[0].std() * torch.randn(h))
        if kind == "gpu":
            model.lm_head.weight.normal_(0, LOGIT_STD / h ** 0.5)
        for name, t in [*model.named_parameters(), *model.named_buffers()]:   # bf16 values, as the release holds
            if t.is_floating_point() and not any(n in name for n in F32):    # them, but its F32 tensors
                t.copy_(t.bfloat16().float())
    defaults = (torch.zeros(()), torch.ones(()))
    assert all(not any(torch.equal(p, d.expand_as(p)) for d in defaults) for p in picked)
    return model


def checkpoint(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """The state dict under the released checkpoint's names (transformers' conversion mapping undone: `hc_attn_fn`,
    the KDA forget gate's tensors in `self_attn`, per-expert projections, separate q / k / v convs), bf16 but the
    tensors the release keeps f32."""
    from transformers.models.glm5_next import modeling_glm5_next as hf

    f32 = (*hf.Glm5NextPreTrainedModel._keep_in_fp32_modules_strict, "hc_attn_base", "hc_attn_scale",
           "hc_ffn_base", "hc_ffn_scale")
    out = {}
    for k, v in model.state_dict().items():
        k = k.replace(".forget_gate.", ".")
        for part, name in (("attn_hc.", "hc_attn_"), ("ffn_hc.", "hc_ffn_")):
            k = k.replace(part, name)
        if k.endswith(".conv1d.weight"):
            for x, conv in zip(("q", "k", "v"), v.chunk(3), strict=True):
                out[k.replace(".conv1d.", f".{x}_conv1d.")] = conv.contiguous()
            continue
        if k.endswith(".experts.gate_up_proj"):
            for e, w in enumerate(v):
                gate, up = w.chunk(2)
                out[k.replace("gate_up_proj", f"{e}.gate_proj.weight")] = gate.contiguous()
                out[k.replace("gate_up_proj", f"{e}.up_proj.weight")] = up.contiguous()
            continue
        if k.endswith(".experts.down_proj"):
            for e, w in enumerate(v):
                out[k.replace("down_proj", f"{e}.down_proj.weight")] = w.contiguous()
            continue
        out[k] = v.contiguous()
    return {k: v if any(n in k for n in f32) else v.bfloat16() for k, v in out.items()}


def save(model: torch.nn.Module, path: pathlib.Path, kind: str = "cpu") -> pathlib.Path:
    """`model` as a checkpoint directory: the released config at the tiny sizes, the tensors, the tokenizer."""
    path.mkdir(parents=True, exist_ok=True)
    save_file(checkpoint(model), str(path / "model.safetensors"))
    (path / "config.json").write_text(json.dumps(checkpoint_config(kind), indent=1))
    gguf_files.copy_tokenizer(tokenizer_dir(), path, ("tokenizer.json", "tokenizer_config.json",
                                                      "chat_template.jinja", "generation_config.json"))
    return path


def convert(path: pathlib.Path, llama: pathlib.Path) -> pathlib.Path:
    """`path` converted by llama.cpp's `convert_hf_to_gguf.py` to f32."""
    return gguf_files.convert(path, llama, NAME)


def quantize(f32: pathlib.Path, quantizer: pathlib.Path) -> pathlib.Path:
    """The f32 file as Q8_0 by llama.cpp's `llama-quantize` (single-threaded: the same bytes on every host)."""
    out = f32.with_name(f"{NAME}-Q8_0.gguf")
    subprocess.run([str(quantizer), f32, out, "Q8_0", "1"], check=True, capture_output=True)
    return out


QUANTIZED = {"attn_k_b.weight": "Q6_K", "attn_v_b.weight": "Q6_K", "indexer.attn_q_b.weight": "Q8_0",
             "indexer.attn_k.weight": "Q8_0", "indexer.proj.weight": "Q8_0", "indexer_compressor_gate.weight": "Q8_0",
             "ssm_f_a.weight": "Q8_0", "ssm_f_b.weight": "Q8_0", "ssm_g_a.weight": "Q8_0", "ssm_g_b.weight": "Q8_0",
             "ssm_beta.weight": "Q8_0"}
LATENT, NOPE = 512, 256               # the real file's widths of `attn_v_b` and `attn_k_b` rows (Q6_K's 256 blocks)


def quantized_copy(f32: pathlib.Path, out: pathlib.Path) -> tuple[pathlib.Path, set[str]]:
    """§6.1's A-m1 file: the CPU fixture's GGUF with the tensors `llama-quantize` leaves f32 quantized as the release
    ships them (`QUANTIZED`: Q6_K `attn_k_b` / `attn_v_b`, Q8_0 indexer and KDA-gate tensors), its DSA layers' MLA at
    the real row widths (kv_lora 512, nope 256: Q6_K needs rows of 256) with random values; everything else as it is.
    Returns the file and the quantized tensors' names: Q8_0 by gguf-py, Q6_K by `q6_k` (gguf-py quantizes no
    Q6_K, and llama.cpp's `llama-quantize` leaves these tensors f32 for `glm5-next`)."""
    import gguf
    import numpy as np

    reader = gguf.GGUFReader(f32)
    arch = "glm5-next"
    heads = int(reader.fields[f"{arch}.attention.head_count"].contents())
    q_lora = int(reader.fields[f"{arch}.attention.q_lora_rank"].contents())
    hidden = int(reader.fields[f"{arch}.embedding_length"].contents())
    v = int(reader.fields[f"{arch}.attention.value_length_mla"].contents())
    shapes = {"attn_kv_a_mqa.weight": (LATENT, hidden), "attn_kv_a_norm.weight": (LATENT,),
              "attn_q_b.weight": (heads * NOPE, q_lora), "attn_k_b.weight": (heads, LATENT, NOPE),
              "attn_v_b.weight": (heads, v, LATENT)}
    keys = {f"{arch}.attention.kv_lora_rank": LATENT, f"{arch}.attention.key_length": LATENT,
            f"{arch}.attention.value_length": LATENT, f"{arch}.attention.key_length_mla": NOPE}
    rng = np.random.default_rng(5)
    path = out / f"{NAME}-quantized.gguf"
    writer = gguf.GGUFWriter(path, arch)
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        kind = field.types[0]
        sub = field.types[-1] if kind == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, keys.get(field.name, field.contents()), kind, sub_type=sub)
    tensors, quantized = [], set()
    for t in reader.tensors:
        part = t.name.split(".", 2)[2] if t.name.startswith("blk.") else t.name
        x = np.asarray(t.data, dtype=np.float32).reshape([int(n) for n in reversed(t.shape)])
        if part in shapes and int(t.name.split(".")[1]) % 4 == 3:
            x = (rng.standard_normal(shapes[part]) * (0.05 if part != "attn_kv_a_norm.weight" else 1)).astype(
                np.float32) + (1 if part == "attn_kv_a_norm.weight" else 0)
        kind = QUANTIZED.get(part)
        if kind is None:
            tensors.append((t.name, x, None))
            continue
        qtype = getattr(gguf.GGMLQuantizationType, kind)
        tensors.append((t.name, q6_k(x) if kind == "Q6_K" else gguf.quants.quantize(x, qtype), qtype))
        quantized.add(t.name)
    for name, data, qtype in tensors:
        writer.add_tensor(name, data, raw_dtype=qtype)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path, quantized


def q6_k(x):
    """`x` [..., n·256] f32 as ggml's Q6_K blocks [..., n·210] uint8 (16 sub-blocks of 16 values a block: an int8
    scale each, an f16 block scale, 6-bit values), laid out as gguf-py's `Q6_K.dequantize_blocks` reads them."""
    import numpy as np

    shape = x.shape
    v = x.reshape(-1, 16, 16).astype(np.float32)                         # [blocks, sub-blocks, values]
    sub = np.abs(v).max(-1) / 31
    d = (np.abs(sub).max(-1) / 127).astype(np.float16)                 # [blocks]
    df = d.astype(np.float32)
    sc = np.where(df[:, None] > 0, np.rint(sub / np.where(df[:, None] > 0, df[:, None], 1)), 0).clip(-128, 127)
    step = df[:, None, None] * sc[:, :, None]
    q = np.where(step != 0, np.rint(v / np.where(step != 0, step, 1)), 0).clip(-32, 31).astype(np.int16) + 32
    q = q.reshape(-1, 256).astype(np.uint8)
    p = np.arange(256)
    h, r = p // 128, p % 128
    ql = np.zeros((len(q), 128), dtype=np.uint8)
    qh = np.zeros((len(q), 64), dtype=np.uint8)
    for s in range(2):                                                    # low nibbles: byte h·64 + r % 64
        at = r // 64 == s
        np.bitwise_or.at(ql, (slice(None), h[at] * 64 + r[at] % 64), (q[:, at] & 0x0F) << (4 * s))
    for s in range(4):                                                    # high bits: byte h·32 + r % 32
        at = r // 32 == s
        np.bitwise_or.at(qh, (slice(None), h[at] * 32 + r[at] % 32), ((q[:, at] >> 4) & 0x03) << (2 * s))
    blocks = np.concatenate([ql, qh, sc.astype(np.int8).view(np.uint8), d.reshape(-1, 1).view(np.uint8)], axis=1)
    return blocks.reshape(*shape[:-1], shape[-1] // 256 * 210)


def prompts(kind: str = "cpu") -> list[list[int]]:
    """The test prompts: random ids below the special tokens, each CPU prompt with an EOS."""
    g = torch.Generator().manual_seed(11)
    if kind == "gpu":
        return [torch.randint(0, PAD, (GPU_PROMPT,), generator=g).tolist() for _ in range(GPU_PROMPTS)]
    out = []
    for at in EOS_AT:
        ids = torch.randint(0, PAD, (PROMPT,), generator=g).tolist()
        ids[at] = EOS
        out.append(ids)
    return out


@torch.inference_mode()
def stats(model: torch.nn.Module, ids: list[list[int]]) -> dict[str, float]:
    """Over HF's one-shot forward of `ids`: per FFN path (dense, routed, shared), the fraction of its rows (a token's,
    or a token's at one routed expert) with some element at the clamp's bound — the gate's upper (`gate`), the up
    projection's upper and lower (`up+`, `up-`) — and, `... elements`, the fraction of their elements there (the clamp's
    effect size); and `comb_peaked`, the fraction of comb rows (a token's source stream) whose largest |s2 * m| is
    >= 8."""
    from transformers.models.glm5_next import modeling_glm5_next as hf

    lim = model.config.swiglu_limit
    seen: dict[str, list[torch.Tensor]] = {}

    def bound(path: str, gate: torch.Tensor, up: torch.Tensor) -> None:
        for side, hit in (("gate", gate > lim), ("up+", up > lim), ("up-", up < -lim)):
            seen.setdefault(f"{path} {side}", []).append(hit.any(-1).float())
            seen.setdefault(f"{path} {side} elements", []).append(hit.float().flatten())

    def mlp_hook(m, args, path):
        x = args[0]
        bound(path, m.gate_proj(x), m.up_proj(x))

    def experts(m, gate_up):
        gate, up = gate_up.chunk(2, dim=-1)
        bound("routed", gate, up)
        return applied(m, gate_up)

    def comb(m, args):
        flat = m.input_norm(args[0].reshape(*args[0].shape[:2], -1).float())
        w = torch.nn.functional.linear(flat, m.fn.float())[..., 2 * m.hc_mult:] * m.scale[2]
        rows = w.reshape(-1, m.hc_mult, m.hc_mult).abs().amax(-1)
        seen.setdefault("comb_peaked", []).append((rows >= 8).float().flatten())
    hooks = []
    applied = hf.Glm5NextTextExperts._apply_gate
    for layer in model.model.layers:
        mlp = layer.mlp
        if isinstance(mlp, hf.Glm5NextTextMoE):
            hooks.append(mlp.shared_experts.register_forward_pre_hook(lambda m, a: mlp_hook(m, a, "shared")))
        else:
            hooks.append(mlp.register_forward_pre_hook(lambda m, a: mlp_hook(m, a, "dense")))
        for hc in (layer.attn_hc, layer.ffn_hc):
            hooks.append(hc.register_forward_pre_hook(comb))
    hf.Glm5NextTextExperts._apply_gate = experts
    try:
        for x in ids:
            model(torch.tensor([x]))
    finally:
        hf.Glm5NextTextExperts._apply_gate = applied
        for h in hooks:
            h.remove()
    return {k: round(float(torch.cat(v).mean()), 4) for k, v in sorted(seen.items())}


@torch.inference_mode()
def fixtures(model: torch.nn.Module, gguf: pathlib.Path) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """HF's reference per prompt `p<i>`, over the weights the CPU fixture's GGUF `gguf` holds (its A_log as read back,
    as the engine reads it; step 2's review A-m2): `ids` [T] (the prompt and its greedy tokens, recomputed from
    scratch), `x` [T, H] (the head's input, one-shot) and `sets.<layer>` [T, top-k + pool - 1] (each position's
    selected tokens as the indexer returns them, -1 padded); the indexer's and the router's boundary ties per prompt
    and `stats` in the metadata. Wraps `model`'s indexers and routers with the judge's tie rule
    (`holon_judge.glm5next`), so pass a model of its own."""
    from holon_judge import gguf as judge_gguf
    from holon_judge.glm5next import wrap_topk

    model.load_state_dict(dict(judge_gguf.hf_state(gguf)))
    seen: dict[str, torch.Tensor] = {}
    ties = [0, 0]
    wrap_topk(model, lambda n: ties.__setitem__(0, ties[0] + n), lambda n: ties.__setitem__(1, ties[1] + n))
    hooks = [layer.self_attn.indexer.register_forward_hook(
        lambda m, a, out, i=i: seen.__setitem__(f"sets.{i}", out[0])) for i, layer in enumerate(model.model.layers)
        if getattr(layer.self_attn, "indexer", None) is not None]
    tensors, meta = {}, {}
    try:
        for p, ids in enumerate(prompts()):
            for _ in range(DECODE):
                ids = ids + [int(model(torch.tensor([ids])).logits[0, -1].argmax())]
            ties[:] = [0, 0]
            x = model.model(torch.tensor([ids])).last_hidden_state[0]
            tensors[f"p{p}.ids"], tensors[f"p{p}.x"] = torch.tensor(ids), x.float().contiguous()
            for name, sets in seen.items():
                tensors[f"p{p}.{name}"] = sets.int().contiguous()
            meta[f"p{p}.boundary_ties"], meta[f"p{p}.router_ties"] = str(ties[0]), str(ties[1])
    finally:
        for h in hooks:
            h.remove()
    meta["stats"] = json.dumps(stats(model, [tensors[f"p{p}.ids"].tolist() for p in range(len(EOS_AT))]))
    return tensors, meta


def load_fixtures() -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    return gguf_files.load_fixtures(FIXTURES)


def build(out: pathlib.Path, llama: pathlib.Path, kind: str, quantizer: pathlib.Path | None = None) -> pathlib.Path:
    """The fixture's GGUF (`cpu`: f32; `gpu`: Q8_0 of the f32 conversion, by `quantizer`) under `out`."""
    model = tiny_hf(kind)
    f32 = convert(save(model, out / f"{NAME}-{kind}", kind), llama)
    if kind == "cpu":
        return f32
    if quantizer is None:
        raise ValueError("the GPU fixture needs llama.cpp's llama-quantize")
    return quantize(f32, quantizer)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--llama", required=True, type=pathlib.Path, help="a llama.cpp checkout at 27c7a76 (built)")
    ap.add_argument("--quantize", type=pathlib.Path, help="llama.cpp 27c7a76's llama-quantize (the GPU fixture)")
    ap.add_argument("--kind", choices=("cpu", "gpu", "both"), default="both")
    ap.add_argument("--fixtures", action="store_true", help=f"(re)write {FIXTURES.name}")
    a = ap.parse_args()
    import transformers

    bad = False
    for kind in ("cpu", "gpu") if a.kind == "both" else (a.kind,):
        f = build(a.out, a.llama, kind, a.quantize)
        digest = hashlib.sha256(f.read_bytes()).hexdigest()
        ok = digest == GGUF_SHA256[kind]
        bad |= not ok
        print(f"{digest}  {f.name}  {'ok' if ok else 'DIFFERS from the pin ' + GGUF_SHA256[kind]}")
    if a.fixtures:
        tensors, meta = fixtures(tiny_hf("cpu"), a.out / f"{NAME}-cpu" / f"{NAME}-f32.gguf")
        meta |= {"transformers": transformers.__version__, "llama_cpp": "27c7a76", "gguf_sha256": GGUF_SHA256["cpu"]}
        save_file(tensors, str(FIXTURES), metadata=meta)
        print(f"wrote {FIXTURES.name}: {meta}")
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
