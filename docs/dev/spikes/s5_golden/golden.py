"""L2 recorder prototype: teacher-forced top-k logprobs over a frozen prompt set, stored as one safetensors file.

Layout of a recording (all positions of all sequences concatenated; position t of a sequence predicts ids[t+1]):
    topk_ids   int32 [N, k]   top-k token ids, logprob-descending
    topk_lp    fp32  [N, k]   their log-softmax values (fp32 log-softmax of the model's logits)
    target_lp  fp32  [N]      logprob of the actual next token
    offsets    int64 [S+1]    sequence s owns rows offsets[s]:offsets[s+1]
    full_pos   int64 [M]      (optional) rows for which the full-vocab logprobs are kept, for validating top-k KL
    full_lp    fp32  [M, V]
Metadata (safetensors header, key "meta"): JSON with model, revision, weight blob hashes, dtype, device, versions,
prompt-set hash, k, wall time.

    python golden.py --model Qwen/Qwen3-0.6B --device cpu --dtype float32 --threads <n> --out out/0.6b-fp32.st
    CUDA_VISIBLE_DEVICES=0 python golden.py --model Qwen/Qwen3-8B --device cuda --dtype bfloat16 --out ...
"""

import argparse
import glob
import hashlib
import json
import os
import time
from dataclasses import dataclass

import torch
import transformers
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


@dataclass
class Recording:
    topk_ids: torch.Tensor
    topk_lp: torch.Tensor
    target_lp: torch.Tensor
    offsets: torch.Tensor
    meta: dict
    full_pos: torch.Tensor | None = None
    full_lp: torch.Tensor | None = None

    def save(self, path: str) -> None:
        t = dict(topk_ids=self.topk_ids, topk_lp=self.topk_lp, target_lp=self.target_lp, offsets=self.offsets)
        if self.full_pos is not None:
            t.update(full_pos=self.full_pos, full_lp=self.full_lp)
        save_file({k: v.contiguous() for k, v in t.items()}, path, metadata={"meta": json.dumps(self.meta)})

    @classmethod
    def load(cls, path: str, full: bool = True) -> "Recording":
        with safe_open(path, "pt") as f:
            keys = set(f.keys())
            get = lambda k: f.get_tensor(k) if k in keys and (full or not k.startswith("full_")) else None
            return cls(get("topk_ids"), get("topk_lp"), get("target_lp"), get("offsets"),
                       json.loads(f.metadata()["meta"]), get("full_pos"), get("full_lp"))


def load_prompts(path: str) -> tuple[list[dict], str]:
    data = open(path, "rb").read()
    return [json.loads(line) for line in data.splitlines()], hashlib.sha256(data).hexdigest()


@torch.inference_mode()
def teacher_force(model, ids: list[int], k: int, chunk: int = 0) -> tuple[torch.Tensor, ...]:
    """One sequence → (topk_ids, topk_lp, target_lp on CPU; full logprobs [T-1, V] on the model's device).

    `chunk` > 0 runs it as chunked prefill through the KV cache (an engine-like path with different GEMM shapes)."""
    x = torch.tensor([ids], device=model.device)
    if chunk:
        logits, past = [], None
        for a in range(0, x.shape[1], chunk):
            out = model(x[:, a:a + chunk], past_key_values=past, use_cache=True)
            logits.append(out.logits[0])
            past = out.past_key_values
        logits = torch.cat(logits)
    else:
        logits = model(x).logits[0]
    lp = logits[:-1].float().log_softmax(-1)
    tgt = x[0, 1:]
    top_lp, top_ids = lp.topk(k, dim=-1)
    return top_ids.int().cpu(), top_lp.cpu(), lp.gather(1, tgt[:, None])[:, 0].cpu(), lp


def cpu_name() -> str:
    return "cpu"            # host details stay out of records


def model_meta(model_id: str) -> dict:
    snap = snapshot_download(model_id, local_files_only=True)
    blobs = {os.path.basename(p): os.path.basename(os.readlink(p)) for p in sorted(glob.glob(f"{snap}/*.safetensors"))}
    return dict(model=model_id, revision=os.path.basename(snap), weight_sha256=blobs)


def record(model_id: str, prompts_path: str, device: str, dtype: str, k: int = 20, full_stride: int = 0,
           attn: str = "sdpa", override: dict | None = None, chunk: int = 0) -> Recording:
    """`override` sets config fields (used only to inject known bugs when testing the comparator)."""
    prompts, prompts_hash = load_prompts(prompts_path)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id, dtype=DTYPES[dtype], device_map=device, attn_implementation=attn, **(override or {})).eval()
    ids, lps, tgts, offsets, full_pos, full_lp = [], [], [], [0], [], []
    t0 = time.perf_counter()
    for p in prompts:
        i, l, t, full = teacher_force(model, p["ids"], k, chunk)
        if full_stride:
            rows = torch.arange(0, len(t), full_stride)
            full_pos.append(rows + offsets[-1])
            full_lp.append(full[rows.to(full.device)].cpu())
        ids.append(i), lps.append(l), tgts.append(t), offsets.append(offsets[-1] + len(t))
        print(f"  {p['id']:22s} {len(p['ids']):5d} tok  {time.perf_counter() - t0:7.1f}s", flush=True)
    wall = time.perf_counter() - t0
    meta = dict(model_meta(model_id), dtype=dtype, device=device, attn=attn, override=override or {}, chunk=chunk,
                k=k, full_stride=full_stride, torch=torch.__version__, transformers=transformers.__version__,
                threads=torch.get_num_threads(),
                hw=torch.cuda.get_device_name() if device.startswith("cuda") else cpu_name(),
                prompts=os.path.basename(prompts_path), prompts_sha256=prompts_hash, wall_s=round(wall, 2),
                n_positions=offsets[-1])
    cat = lambda xs: torch.cat(xs) if xs else None
    return Recording(torch.cat(ids), torch.cat(lps), torch.cat(tgts), torch.tensor(offsets), meta,
                     cat(full_pos), cat(full_lp))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", default=os.path.join(os.path.dirname(__file__), "prompts/v1.jsonl"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32", choices=DTYPES)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--full-stride", type=int, default=0, help="keep full-vocab logprobs every N-th position")
    ap.add_argument("--chunk", type=int, default=0, help="chunked-prefill size (0 = one forward)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--override", nargs="*", default=[], help="config key=json_value (bug injection)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    override = {kv.split("=", 1)[0]: json.loads(kv.split("=", 1)[1]) for kv in a.override}
    rec = record(a.model, a.prompts, a.device, a.dtype, a.k, a.full_stride, a.attn, override, a.chunk)
    rec.save(a.out)
    print(json.dumps(dict(rec.meta, bytes=os.path.getsize(a.out)), indent=1))


if __name__ == "__main__":
    main()
