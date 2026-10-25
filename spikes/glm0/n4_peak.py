"""GLM step 0, N4's memory term (docs/v1-glm.md D9, §7.1): the live set of a prompt step per token, with and without
D9's row sub-batching, at 4K / 8K / 16K / 32K-token steps — one KDA layer and one DSA layer of GLM-5.3-Flash's shapes,
each inside its mHC block, in torch with the KDA chunk kernel at the FLA pin (vLLM 6dd813e0's `ops/kda.py`, importable
via `--fla <dir>`: the step-0 spike's copy with vLLM's imports stubbed, not Holon's vendored tree), the peak of
`torch.cuda.max_memory_allocated` above the step's inputs (R and the step's ids) divided by the step's tokens.

What spans the step either way: R [T, 4·4096] f32, the block input x and output y (bf16). With `--sub S` every other
op runs on ≤ S rows at a time: the mHC mix's coefficients, the norms, the projections, KDA (its chunk kernel over
1,024-token sub-chunks, the state carried, as `kernels.fla`'s `SUB_TOKENS`), MLA's absorbed queries q̃ [S, 64, 512]
and its attention over the latent (a dense `scaled_dot_product_attention` stand-in over ≤ 2,051 selected keys per
query: the sparse kernel's live set is its q̃, the gathered keys' tile and the output), the indexer's scores (512
queries a pass), the dense / shared FFN. The routed experts' buffers are the store's (lent) and left out. Without
`--sub` each op runs on the whole step (rev 1's 270 KB/token estimate).

    CUDA_VISIBLE_DEVICES=0 python spikes/glm0/n4_peak.py --fla local/tmp/glm/kda_spike [--sub 2048] --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys

import torch
import torch.nn.functional as F

H, N = 4096, 4
KH, KD = 64, 128                                     # KDA heads x dim
MH, NOPE, V, KV, QL = 64, 256, 256, 512, 1536         # MLA
IH, ID, POOL, TOPK = 32, 128, 4, 2048                 # indexer
SH = 2048                                             # shared expert (and dense FFN at layers 0-2: 12,288)
dev = torch.device("cuda")


def w(*shape, dtype=torch.bfloat16):
    return torch.randn(*shape, device=dev, dtype=dtype) * 0.02


class Weights:
    def __init__(self):
        self.fn = w(24, N * H, dtype=torch.float32)
        self.qkv, self.fa, self.fb, self.ga, self.gb = w(3 * KH * KD, H), w(128, H), w(KH * KD, 128), w(128, H), w(KH * KD, 128)
        self.beta, self.ko = w(KH, H), w(H, KH * KD)
        self.qa, self.qb, self.kva = w(QL, H), w(MH * NOPE, QL), w(KV, H)
        self.kb, self.vb, self.mo = w(MH, KV, NOPE), w(MH, V, KV), w(H, MH * V)
        self.iq, self.ik, self.iw, self.ig = w(IH * ID, QL), w(ID, H), w(IH, H), w(ID, H)
        self.router, self.su, self.sd = w(288, H, dtype=torch.float32), w(2 * SH, H), w(H, SH)


def rows(t: int, sub: int | None):
    s = sub or t
    return [(a, min(a + s, t)) for a in range(0, t, s)]


def mix(R: torch.Tensor, W: Weights, sub: int | None) -> torch.Tensor:
    x = torch.empty(R.shape[0], H, device=dev, dtype=torch.bfloat16)
    for a, b in rows(R.shape[0], sub):
        r = R[a:b]
        m = (r * torch.rsqrt(r.pow(2).mean(-1, keepdim=True) + 1e-5)) @ W.fn.T
        pre = torch.sigmoid(m[:, :4]) + 1e-6
        x[a:b] = (pre[:, :, None] * r.view(-1, N, H)).sum(1).to(torch.bfloat16)
    return x


def writeback(R: torch.Tensor, y: torch.Tensor, sub: int | None) -> None:
    for a, b in rows(R.shape[0], sub):
        r = R[a:b].view(-1, N, H)
        R[a:b] = (r + y[a:b, None].float()).view(-1, N * H)            # same live set as post·y + combᵀ·R


def kda(x: torch.Tensor, W: Weights, sub: int | None, chunk_kda) -> torch.Tensor:
    t = x.shape[0]
    y = torch.empty_like(x)
    state = torch.zeros(1, KH, KD, KD, device=dev, dtype=torch.float32)
    for a, b in rows(t, sub):
        h = F.rms_norm(x[a:b], (H,))
        q, k, v = (h @ W.qkv.T).view(b - a, 3, KH, KD).unbind(1)
        g = -5 * torch.sigmoid((h @ W.fa.T @ W.fb.T).float()).view(1, b - a, KH, KD)
        beta = torch.sigmoid((h @ W.beta.T).float()).view(1, b - a, KH)
        outs = []
        for c in range(0, b - a, 1024):                             # kernels.fla's SUB_TOKENS
            e = min(c + 1024, b - a)
            o, state = chunk_kda(q[None, c:e], k[None, c:e], v[None, c:e], g[:, c:e], beta[:, c:e],
                                 initial_state=state, output_final_state=True, use_qk_l2norm_in_kernel=True)
            outs.append(o[0])
        o = torch.cat(outs).reshape(b - a, KH * KD)
        z = torch.sigmoid(h @ W.ga.T @ W.gb.T)
        y[a:b] = (F.rms_norm(o.float().view(-1, KH, KD), (KD,)).view(-1, KH * KD) * z).to(torch.bfloat16) @ W.ko.T
    return y


def dsa(x: torch.Tensor, W: Weights, sub: int | None, latent: torch.Tensor) -> torch.Tensor:
    t = x.shape[0]
    y = torch.empty_like(x)
    for a, b in rows(t, sub):
        h = F.rms_norm(x[a:b], (H,))
        qr = F.rms_norm(h @ W.qa.T, (QL,))
        latent[a:b] = F.rms_norm(h @ W.kva.T, (KV,))
        keys = h @ W.ik.T                                           # cached raw keys (the idx page)
        # indexer: 512 queries a pass against every pool so far; the top-512 ids are what stays
        qi = (qr @ W.iq.T).view(b - a, IH, ID)
        wt = (h @ W.iw.T).float()
        pools = latent.new_zeros(0)
        npool = max(1, b // POOL)
        kbar = torch.zeros(npool, ID, device=dev, dtype=torch.float32)
        sel = torch.empty(b - a, min(TOPK // POOL, npool), device=dev, dtype=torch.int32)
        for c in range(0, b - a, 512):
            e = min(c + 512, b - a)
            s = (torch.relu(torch.einsum("thd,pd->thp", qi[c:e].float(), kbar)) * wt[c:e, :, None]).sum(1)
            sel[c:e] = s.topk(sel.shape[1], dim=-1).indices.int()
            del s
        # absorbed MLA: q̃ [S, 64, 512], attention over <= 2,051 gathered latent rows a query (stand-in)
        q = (qr @ W.qb.T).view(b - a, MH, NOPE)
        qt = torch.einsum("thn,hkn->thk", q, W.kb)
        kv = latent[:b][-min(b, TOPK + 3):]                        # the gathered tile (a shared stand-in)
        ctx = F.scaled_dot_product_attention(qt.transpose(0, 1)[None], kv[None, None].expand(1, MH, -1, -1),
                                             kv[None, None].expand(1, MH, -1, -1), scale=1 / 16)[0].transpose(0, 1)
        o = torch.einsum("thk,hvk->thv", ctx, W.vb).reshape(b - a, MH * V)
        y[a:b] = o @ W.mo.T
        del q, qt, ctx, o, keys, pools
    return y


def moe_glue(x: torch.Tensor, W: Weights, sub: int | None) -> torch.Tensor:
    y = torch.empty_like(x)
    for a, b in rows(x.shape[0], sub):
        h = F.rms_norm(x[a:b], (H,))
        logits = h.float() @ W.router.T
        top = torch.sigmoid(logits).topk(8, dim=-1)
        gu = h @ W.su.T
        s = (F.silu(gu[:, :SH].clamp(max=10)) * gu[:, SH:].clamp(-10, 10)) @ W.sd.T
        y[a:b] = s
        del logits, top, gu
    return y


def step(t: int, sub: int | None, W: Weights, chunk_kda) -> dict:
    R = torch.randn(t, N * H, device=dev, dtype=torch.float32)
    latent = torch.empty(t, KV, device=dev, dtype=torch.bfloat16)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    for mixer in ("kda", "dsa"):
        x = mix(R, W, sub)
        y = kda(x, W, sub, chunk_kda) if mixer == "kda" else dsa(x, W, sub, latent)
        writeback(R, y, sub)
        del x, y
        x = mix(R, W, sub)
        y = moe_glue(x, W, sub)
        writeback(R, y, sub)
        del x, y
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    return {"tokens": t, "sub": sub, "peak_gib": round(peak / 2**30, 3), "kib_per_token": round(peak / t / 1024, 1),
            "residual_kib_per_token": 64}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fla", required=True, help="dir holding fla_ops/ (kda.py) and the vllm import stubs")
    ap.add_argument("--sub", type=int, nargs="*", default=[0, 2048])
    ap.add_argument("--tokens", type=int, nargs="+", default=[4096, 8192, 16384, 32768])
    ap.add_argument("--out")
    a = ap.parse_args()
    sys.path.insert(0, a.fla)
    from fla_ops.kda import chunk_kda
    W = Weights()
    out = {"gpu": torch.cuda.get_device_name(), "rows": []}
    for sub in a.sub:
        for t in a.tokens:
            try:
                r = step(t, sub or None, W, chunk_kda)
            except torch.cuda.OutOfMemoryError:
                r = {"tokens": t, "sub": sub or None, "oom": True}
            torch.cuda.empty_cache()
            print(json.dumps(r), flush=True)
            out["rows"].append(r)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
