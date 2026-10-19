"""FN step 0, N3's memory term (docs/v1-flash-next.md §7): the non-lendable step peak per token, from a trunk-only
allocation profile — one GDN layer and one QSA layer of Qwen3.8-Flash-Next's shapes run in torch with the GDN chunk
kernel Holon ships (`kernels.fla`'s vendored FLA), the peak of `torch.cuda.max_memory_allocated` above the step's
inputs measured per token at 2K / 4K / 8K-token steps. The routed experts' buffers (the lendable part, D12c) are left
out; everything else a layer holds at once is in: the fp32 residual R and its grouped norm, the HC mix's projections
(unfused, so an upper bound for D10's fused kernels), the mixer's projections and internals, the router's f32 logits,
the shared expert, the combine.

    CUDA_VISIBLE_DEVICES=0 python spikes/fn0/n3_peak.py --out <record.json>
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import sys

import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "plugins")]
import run  # noqa: E402
from holon_plugins.kernels.fla.vendor import chunk_gated_delta_rule  # noqa: E402

H, N, RANK, E, TOPK, INTER = 2560, 4, 320, 512, 10, 640
GDN_K, GDN_V, DK = 16, 48, 128                          # GDN heads (k / v) x 128
Q_HEADS, KV_HEADS, HEAD, IDX_HEADS, IDX_DIM = 24, 2, 256, 4, 128
SUB = 512                                               # selection scores: 512 queries per pass (D6)
dev = torch.device("cuda")


def w(*shape, dtype=torch.bfloat16):
    return torch.randn(*shape, device=dev, dtype=dtype) * 0.02


class Layer:
    """One layer's weights (bf16 dense, as the GGUF's dequantized projections would feed torch here)."""

    def __init__(self, qsa: bool):
        self.qsa = qsa
        self.hc = [(w(RANK, N * H), w(N * H, RANK), w(N, N * H)) for _ in range(2)]
        if qsa:
            self.q, self.k, self.v = w(2 * Q_HEADS * HEAD, H), w(KV_HEADS * HEAD, H), w(KV_HEADS * HEAD, H)
            self.o, self.idx = w(H, Q_HEADS * HEAD), w((IDX_HEADS + 1) * IDX_DIM, H)
        else:
            self.qkv, self.z = w(2 * GDN_K * DK + GDN_V * DK, H), w(GDN_V * DK, H)
            self.ab, self.out = w(2 * GDN_V, H), w(H, GDN_V * DK)
            self.conv = w(2 * GDN_K * DK + GDN_V * DK, 4)
        self.router, self.shared, self.sdown, self.sgate = w(E, H), w(2 * INTER, H), w(H, INTER), w(1, H)


def mix(r: torch.Tensor, ws) -> tuple[torch.Tensor, torch.Tensor]:
    down, up, inj = ws
    rn = (r.view(-1, N, H) * torch.rsqrt(r.view(-1, N, H).pow(2).mean(-1, keepdim=True) + 1e-6)).view(-1, N * H)
    u = torch.sigmoid(F.silu(rn @ down.float().T / N) @ up.float().T)
    x = (u.view(-1, N, H) * rn.view(-1, N, H)).mean(1)
    return x.to(torch.bfloat16), 2 * torch.sigmoid(rn @ inj.float().T / N)


def ffn(r: torch.Tensor, layer: Layer) -> None:
    x, inj = mix(r, layer.hc[1])
    logits = (x.float() @ layer.router.float().T)                       # the router's f32 logits
    weights, ids = torch.topk(torch.softmax(logits, -1), TOPK)
    gu = x @ layer.shared.T
    shared = (F.silu(gu[:, :INTER]) * gu[:, INTER:]) @ layer.sdown.T
    y = shared.float() * torch.sigmoid(x.float() @ layer.sgate.float().T)    # + the routed sum (lent buffers)
    r.view(-1, N, H).add_(inj[..., None] * y[:, None, :])


def gdn(r: torch.Tensor, layer: Layer) -> None:
    x, inj = mix(r, layer.hc[0])
    t = x.shape[0]
    qkv = x @ layer.qkv.T
    z, ab = x @ layer.z.T, (x @ layer.ab.T).float()
    qkv = F.silu(F.conv1d(F.pad(qkv.T[None], (3, 0)), layer.conv[:, None, :], groups=qkv.shape[1]))[0].T
    q, k, v = qkv.split([GDN_K * DK, GDN_K * DK, GDN_V * DK], -1)
    rep = GDN_V // GDN_K
    q = q.reshape(1, t, GDN_K, DK).repeat_interleave(rep, 2)
    k = k.reshape(1, t, GDN_K, DK).repeat_interleave(rep, 2)
    g = -F.softplus(ab[:, :GDN_V])[None]
    beta = torch.sigmoid(ab[:, GDN_V:])[None]
    o, _ = chunk_gated_delta_rule(q.contiguous(), k.contiguous(), v.reshape(1, t, GDN_V, DK).contiguous(), g, beta,
                                  use_qk_l2norm_in_kernel=True, output_final_state=True)
    o = (o[0].float() * torch.sigmoid(z.view(t, GDN_V, DK).float())).to(torch.bfloat16)
    y = (o.reshape(t, -1) @ layer.out.T).float()
    r.view(-1, N, H).add_(inj[..., None] * y[:, None, :])


def qsa(r: torch.Tensor, layer: Layer) -> None:
    x, inj = mix(r, layer.hc[0])
    t = x.shape[0]
    qg = (x @ layer.q.T).view(t, Q_HEADS, 2 * HEAD)
    q, gate = qg[..., :HEAD], qg[..., HEAD:]
    k, v = (x @ layer.k.T).view(t, KV_HEADS, HEAD), (x @ layer.v.T).view(t, KV_HEADS, HEAD)
    idx = (x @ layer.idx.T).float()
    iq, ik = idx[:, :IDX_HEADS * IDX_DIM].view(t, IDX_HEADS, IDX_DIM), idx[:, IDX_HEADS * IDX_DIM:]
    pooled = ik[: t // 4 * 4].view(-1, 4, IDX_DIM).mean(1)              # the pooled keys (the cache holds them)
    for a in range(0, t, SUB):                                            # sub-batched selection scores
        s = torch.relu(torch.einsum("qhd,bd->qhb", iq[a:a + SUB], pooled)).sum(1)
        torch.topk(s, min(512, s.shape[1]), dim=1)
    o = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None],
                                       is_causal=True, enable_gqa=True)
    o = (o[0].transpose(0, 1) * torch.sigmoid(gate)).reshape(t, -1)
    y = (o @ layer.o.T).float()
    r.view(-1, N, H).add_(inj[..., None] * y[:, None, :])


def peak(layer: Layer, tokens: int) -> float:
    """Bytes per token above the step's inputs (R and the ids) while the layer runs."""
    r = torch.randn(tokens, N * H, device=dev)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    (qsa if layer.qsa else gdn)(r, layer)
    ffn(r, layer)
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / tokens


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(0)
    layers = {"gdn": Layer(False), "qsa": Layer(True)}
    for layer in layers.values():                                         # warm up (Triton compiles)
        peak(layer, 1024)
    rows = []
    for tokens in (2048, 4096, 8192):
        row = {"tokens": tokens}
        for name, layer in layers.items():
            row[f"{name}_kb_per_token"] = round(peak(layer, tokens) / 1e3, 1)
        row["residual_kb_per_token"] = N * H * 4 / 1e3
        row["non_lendable_kb_per_token"] = round(row["residual_kb_per_token"] + max(
            row["gdn_kb_per_token"], row["qsa_kb_per_token"]), 1)
        rows.append(row)
        print(row, flush=True)
    rec = {"name": "fn0_n3_peak", "date": datetime.date.today().isoformat(), "gpu": run.gpu_class(0),
           "commit": run.git("rev-parse", "HEAD"), "command": "python spikes/fn0/n3_peak.py",
           "method": __doc__.split("\n\n")[1].replace("\n", " "), "rows": rows}
    pathlib.Path(a.out).write_text(json.dumps(rec, indent=1) + "\n")


if __name__ == "__main__":
    main()
