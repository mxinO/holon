"""The tiny Qwen3.8-Flash-Next (`qwen4_exp`) for CPU tests (docs/v1-flash-next.md D2, §6.1): random weights in
transformers' `Qwen4ExpForCausalLM` (5.19.0; run in the golden venv, `local/envs/golden-qwen4`), saved as
`Qwen4ExpForConditionalGeneration` stores them (the n-gram table in `split_ngram_parts` shards, a random MTP block
under `mtp.`); its GGUF as llama.cpp's own converter writes it (an oracle for the conversion's conventions that does
not share the engine's or the judge's beliefs: norm + 1, `ssm_a`, the conv layouts, the HC reshapes); and its MTP
block as Strata's tools see it: `tools/mtp_fetch.py`'s layout (one raw bf16 file per tensor and `mtp-manifest.json`),
packed by `tools/mtp_pack.py`.

The GGUF (145 MB at the 248,320-token vocabulary) and the pack are not committed: this script rebuilds them, byte for
byte, and checks their sha256 against `GGUF_SHA256` / `MTP_PACK_SHA256` (D2's deviation: no host to fetch them from).
What is committed are the small fixtures (`FIXTURES`, written with `--fixtures`): for each test prompt, HF's ids
(the prompt and its 24 greedy tokens), the output mixer's x, the indexer's selected token sets per QSA layer and
position, and PLE's row ids, all from HF's one-shot forward under D2's tie rule. Importing this module needs no
transformers with `qwen4_exp`; building does (the golden venv).

    python -m holon_testing.qwen4exp_files --out <dir> --llama <llama.cpp 27c7a76> --strata <Strata 296b59d> \
        [--fixtures]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys

import torch
from safetensors.torch import load_file, save_file

TOKENIZER = ("Qwen/Qwen3.8-Flash-Next", "de4b8e4d43b917e7706784d8bb445c9af86a3540")
EOS = 248044
NAME = "tiny-qwen4exp"                               # the converter's general.name (it defaults to the directory's)
GGUF_SHA256 = "eb8a60380b850383c931eee5e156a7d468ff89393bc9ddfb77fe52d975183998"
MTP_PACK_SHA256 = "37ab79892e8d4140eaba19ad45560cf92c925f0c149340eb5e90caa0b1b1ff65"
FIXTURES = pathlib.Path(__file__).with_name("qwen4exp_fixtures.safetensors")
PROMPT, DECODE, EOS_AT = 100, 24, (40, 71)           # §6.1: 100-token prompts with an EOS, then 24 decode steps


def tiny_config():
    """§6.1: 8 layers (QSA at 3 and 7, PLE at layer 1), H = 64, four streams of rank 16, the real vocabulary; attention
    4 / 2 heads x 64; indexer 2 x 32, ratio 4, budget 16 (so selection is sparse past 16 + 3 tokens); GDN 2 / 6 heads x
    32; 8 experts top-2 with I = 64 and a shared expert of 64 (Q2_0's 64-value blocks); PLE 4 n-gram heads x 16 over
    1,000-row hash tables; one MTP layer."""
    from transformers import Qwen4ExpTextConfig

    return Qwen4ExpTextConfig(
        vocab_size=248320, hidden_size=64, num_hidden_layers=8, layer_types=None, full_attention_interval=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=64, linear_num_key_heads=2, linear_num_value_heads=6,
        linear_key_head_dim=32, linear_value_head_dim=32, linear_conv_kernel_dim=4, moe_intermediate_size=64,
        shared_expert_intermediate_size=64, num_experts=8, num_experts_per_tok=2, hc_count=4, hc_lowrank=16,
        ple_layer_ids=[2], ple_embed_dim=64, heads_per_ngram=2, ngram_size=3, ngram_vocab_size_base=1000,
        split_ngram_parts=4, indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=32, indexer_budget=16,
        indexer_compress_ratio=4, output_gate_type="sigmoid", eos_token_id=EOS, bos_token_id=EOS,
        tie_word_embeddings=False, mtp_num_hidden_layers=1,
        rope_parameters={"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25,
                         "mrope_section": [3, 3, 2], "mrope_interleaved": True})


def tiny_hf(seed: int = 0) -> torch.nn.Module:
    """The random model; every norm, the PLE conv, `A_log` and `dt_bias` randomised too (HF zero-initialises the norms
    and the PLE conv, and zeros would hide a wrong convention)."""
    from transformers.models.qwen4_exp import modeling_qwen4_exp as hf

    torch.manual_seed(seed)
    model = hf.Qwen4ExpForCausalLM(tiny_config()).eval()
    norms = (hf.Qwen4ExpTextRMSNorm, hf.Qwen4ExpTextRMSNormGated)
    picked = [m.weight for m in model.modules() if isinstance(m, norms)] + [
        p for name, p in model.named_parameters() if name.endswith(("A_log", "dt_bias", "ple.conv1d.weight"))]
    with torch.no_grad():
        for p in picked:
            p.copy_(torch.randn_like(p) * 0.5)
        for p in model.parameters():                     # bf16 values, as the real checkpoint holds: the converter's
            p.copy_(p.bfloat16().float())                # w + 1 and the judge's - 1 are then exact in f32
    assert all(p.abs().min() > 0 for p in picked)
    return model


def mtp_tensors(model: torch.nn.Module, seed: int = 3) -> dict[str, torch.Tensor]:
    """A random MTP block (bf16) under the real checkpoint's names: the projections from e and h, the pre-fc norms,
    one QSA + MoE layer shaped like the trunk's (layer 3's) and its output mixer."""
    g = torch.Generator().manual_seed(seed)
    rand = lambda *s: (torch.randn(*s, generator=g) * 0.05).to(torch.bfloat16)            # noqa: E731
    c = model.config
    h, n = c.hidden_size, c.hc_count * c.hidden_size
    out = {"mtp.fc_embedding.weight": rand(h, h), "mtp.fc_hidden.weight": rand(h, h),
           "mtp.pre_fc_norm_embedding.weight": rand(h), "mtp.pre_fc_norm_hidden.weight": rand(n)}
    qsa = next(i for i, t in enumerate(c.layer_types) if t == "indexed_attention")
    for k, v in model.model.layers[qsa].state_dict().items():
        out[f"mtp.layers.0.{k}"] = rand(*v.shape)
    for k, v in model.model.hyper_connection_mixer.state_dict().items():
        out[f"mtp.hyper_connection_mixer.{k}"] = rand(*v.shape)
    return out


def checkpoint(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """The state dict as `Qwen4ExpForConditionalGeneration` stores it: bf16, the language model under
    `model.language_model.`, the n-gram table split row-wise into `split_ngram_parts` shards, the int64 hash buffers."""
    out = {}
    parts = model.config.split_ngram_parts
    for k, v in model.state_dict().items():
        k = k.replace("model.", "model.language_model.", 1)
        v = v.bfloat16() if v.is_floating_point() else v
        if k.endswith("ngram_embedding.weight"):
            for i, shard in enumerate(v.chunk(parts)):
                out[k.replace("ngram_embedding.weight", f"ngram_embedding.shard_{i}.weight")] = shard.contiguous()
        else:
            out[k] = v.contiguous()
    return out


def save(model: torch.nn.Module, path: pathlib.Path) -> pathlib.Path:
    """`model` (with its random MTP block) as an HF checkpoint directory."""
    path.mkdir(parents=True, exist_ok=True)
    save_file(checkpoint(model) | mtp_tensors(model), str(path / "model.safetensors"))
    text = model.config.to_dict() | {"model_type": "qwen4_exp_text"}
    (path / "config.json").write_text(json.dumps({"architectures": ["Qwen4ExpForConditionalGeneration"],
                                                  "model_type": "qwen4_exp", "tie_word_embeddings": False,
                                                  "language_model_only": True, "text_config": text}))
    return path


def tokenizer_dir() -> pathlib.Path:
    hub = pathlib.Path(os.environ.get("HF_HUB_CACHE") or pathlib.Path(os.environ.get("HF_HOME") or
                                                                       "~/.cache/huggingface").expanduser() / "hub")
    return hub / f"models--{TOKENIZER[0].replace('/', '--')}" / "snapshots" / TOKENIZER[1]


def convert(path: pathlib.Path, llama: pathlib.Path) -> pathlib.Path:
    """`path` (the checkpoint directory) converted by llama.cpp's `convert_hf_to_gguf.py` to f32, with the model's
    tokenizer (the converter needs the full 248,320-token vocabulary)."""
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        (path / name).write_bytes((tokenizer_dir() / name).read_bytes())
    out = path / f"{NAME}-f32.gguf"
    subprocess.run([sys.executable, str(llama / "convert_hf_to_gguf.py"), str(path), "--outtype", "f32",
                    "--model-name", NAME, "--outfile", str(out)],
                   env=os.environ | {"PYTHONPATH": str(llama / "gguf-py")}, check=True)
    return out


def mtp_pack(model: torch.nn.Module, path: pathlib.Path, strata: pathlib.Path, llama: pathlib.Path) -> pathlib.Path:
    """The MTP block in `tools/mtp_fetch.py`'s layout (St tools/mtp_fetch.py:1-12), then Strata's own
    `tools/mtp_pack.py --experts q2_0` (HF names, fused gate_up_proj, BF16 dense parts, F32 norms, MSE-scaled Q2_0)."""
    src = path / "mtp"
    (src / "tensors").mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, t in sorted(mtp_tensors(model).items()):
        raw = t.contiguous().view(torch.int16).numpy().tobytes()
        (src / "tensors" / f"{name}.bin").write_bytes(raw)
        manifest.append({"name": name, "dtype": "BF16", "shape": list(t.shape), "file": f"tensors/{name}.bin",
                         "sha256": hashlib.sha256(raw).hexdigest()})
    (src / "mtp-manifest.json").write_text(json.dumps(manifest, indent=1))
    out = path / f"{NAME}-mtp-q2_0.gguf"
    subprocess.run([sys.executable, str(strata / "tools" / "mtp_pack.py"), "--src", str(src), "--experts", "q2_0",
                    "--out", str(out)], env=os.environ | {"STRATA_GGUF_PY": str(llama / "gguf-py")}, check=True)
    return out


def prompts() -> list[list[int]]:
    """The test prompts: random ids below the special tokens, each with an EOS (PLE's window cuts there)."""
    g = torch.Generator().manual_seed(11)
    out = []
    for at in EOS_AT:
        ids = torch.randint(0, 248000, (PROMPT,), generator=g).tolist()
        ids[at] = EOS
        out.append(ids)
    return out


def stable_topk(x: torch.Tensor, k: int, dim: int = -1, largest: bool = True, sorted: bool = True):
    """D2's tie rule: a stable sort, so equal scores go to the lower index."""
    values, indices = torch.sort(x, dim=dim, descending=largest, stable=True)
    return torch.return_types.topk((values.narrow(dim, 0, k), indices.narrow(dim, 0, k)))


@torch.inference_mode()
def fixtures(model: torch.nn.Module) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """HF's reference per prompt `p<i>`: `ids` [T] (the prompt and its greedy tokens), `x` [T, H] (the output mixer's
    output, one-shot), `sets.<layer>` [T, budget + ratio - 1] (each position's selected tokens, ascending, -1 padded)
    and `ple_ids` [T, heads]; the boundary ties per prompt in the metadata."""
    c = model.config
    width = c.indexer_budget + c.indexer_compress_ratio - 1
    seen: dict[str, torch.Tensor] = {}
    ties = [0]
    hooks = []
    for i, layer in enumerate(model.model.layers):
        if layer.layer_type == "indexed_attention":
            hooks.append(layer.self_attn.indexer.register_forward_hook(
                lambda m, a, out, i=i: seen.__setitem__(f"sets.{i}", out[0, 0])))
        if layer.ple is not None:
            hooks.append(layer.ple.ple_embedding.ngram_embedding.register_forward_hook(
                lambda m, a, out: seen.__setitem__("ple_ids", a[0][0])))

    def counting(x, k, dim=-1, largest=True, sorted=True):
        if x.dim() == 1 and k < x.numel():
            values = torch.sort(x, descending=True, stable=True).values
            ties[0] += int(values[k - 1] == values[k])
        return stable_topk(x, k, dim, largest, sorted)
    topk, torch.Tensor.topk = torch.Tensor.topk, counting       # only the indexer calls topk in this model's forward
    tensors, meta = {}, {}
    try:
        for p, ids in enumerate(prompts()):
            for _ in range(DECODE):                                           # greedy, recomputed from scratch
                ids = ids + [int(model(torch.tensor([ids])).logits[0, -1].argmax())]
            ties[0] = 0
            x = model.model(torch.tensor([ids])).last_hidden_state[0]
            tensors[f"p{p}.ids"], tensors[f"p{p}.x"] = torch.tensor(ids), x.float().contiguous()
            for name, mask in seen.items():
                if name == "ple_ids":
                    tensors[f"p{p}.ple_ids"] = mask.contiguous()
                    continue
                sets = torch.full((len(ids), width), -1, dtype=torch.int32)
                for t in range(len(ids)):
                    sel = mask[t].nonzero().flatten()
                    sets[t, :len(sel)] = sel.int()
                tensors[f"p{p}.{name}"] = sets
            meta[f"p{p}.boundary_ties"] = str(ties[0])
    finally:
        torch.Tensor.topk = topk
        for h in hooks:
            h.remove()
    return tensors, meta


def load_fixtures() -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    from safetensors import safe_open

    with safe_open(str(FIXTURES), "pt") as f:
        return load_file(str(FIXTURES)), f.metadata()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--llama", required=True, type=pathlib.Path, help="a llama.cpp checkout at 27c7a76")
    ap.add_argument("--strata", required=True, type=pathlib.Path, help="a Strata checkout at 296b59d (the MTP pack)")
    ap.add_argument("--fixtures", action="store_true", help=f"(re)write {FIXTURES.name}")
    a = ap.parse_args()
    import transformers

    model = tiny_hf()
    path = save(model, a.out / NAME)
    files = {convert(path, a.llama): GGUF_SHA256, mtp_pack(model, path, a.strata, a.llama): MTP_PACK_SHA256}
    digests = {f: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    for f, digest in digests.items():
        print(f"{digest}  {f.name}  {'ok' if digest == files[f] else 'DIFFERS from the pin ' + files[f]}")
    if a.fixtures:
        tensors, meta = fixtures(model)
        meta |= {"transformers": transformers.__version__, "llama_cpp": "27c7a76",
                 "gguf_sha256": digests[next(iter(files))]}
        save_file(tensors, str(FIXTURES), metadata=meta)
        print(f"wrote {FIXTURES.name}: {meta}")
    if any(d != files[f] for f, d in digests.items()):
        sys.exit(1)


if __name__ == "__main__":
    main()
