"""The DFlash2 published forward in plain torch (docs/m04-spec.md §5, L0-drafter): research note §1.4–1.5 and §2, after
SGLang's `models/dflash.py` (studied, not copied). One sequence: the context features of positions 0 … P − 1 (the
target's residual streams after the tapped layers, concatenated) and the anchor token at P; the block's rows
1 … B − 1 predict positions P + 1 … P + B − 1.

Weights are the checkpoint's own (HF names); the target's embedding and head are given. `dtype` is the arithmetic
(fp32: the golden; the model's dtype: the floor); `matmul(name, x, w)` may be replaced to emulate a kernel path (the
judge's q8_1 rounding for GGUF weights); `device` is where it runs.
"""
from __future__ import annotations

from collections.abc import Callable

import torch


class Reference:
    def __init__(self, state: dict[str, torch.Tensor], cfg: dict, embed: torch.Tensor, head: torch.Tensor,
                 dtype: torch.dtype = torch.float32,
                 matmul: Callable[[str, torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
                 device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self.w = {k: v.to(self.device, dtype) for k, v in state.items()}
        self.cfg, self.dtype = cfg, dtype
        self.embed, self.head = embed, head.to(self.device, dtype)          # the embedding: two rows read
        self.matmul = matmul or (lambda name, x, w: x @ w.T)

    def _linear(self, name: str, x: torch.Tensor) -> torch.Tensor:
        return self.matmul(name, x, self.w[name]).to(self.dtype)

    def _norm(self, name: str, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        y = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.cfg["eps"])
        return (y * self.w[name].float()).to(self.dtype)

    def _rope(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1]
        inv = 1.0 / self.cfg["theta"] ** (torch.arange(0, d, 2, dtype=torch.float64) / d)
        angle = positions.double()[:, None] * inv[None]
        cos, sin = angle.cos()[:, None].float(), angle.sin()[:, None].float()
        a, b = x.float().chunk(2, -1)
        return torch.cat([a * cos - b * sin, b * cos + a * sin], -1).to(self.dtype)

    def _kv(self, i: int, x: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c, p = self.cfg, f"layers.{i}.self_attn."
        k = self._linear(p + "k_proj.weight", x).view(len(x), c["kv_heads"], c["head_dim"])
        v = self._linear(p + "v_proj.weight", x).view(len(x), c["kv_heads"], c["head_dim"])
        return self._rope(self._norm(p + "k_norm.weight", k), positions), v

    def _conv(self, name: str, x: torch.Tensor, delta: torch.Tensor, side: int) -> torch.Tensor:
        c = self.cfg
        groups = c["hidden"] // c["group"]
        blocks = x.view(len(x), groups, -1)
        coef = self.w[name][side].view(1, c["conv"], groups, -1) + delta[..., None]
        out = coef[:, 0] * blocks
        for tap in range(1, c["conv"]):                     # within the block: rows before `tap` have no predecessor
            out[tap:] = out[tap:] + coef[tap:, tap] * blocks[:-tap]
        return out.reshape(len(x), -1)

    def _wrapped(self, conv: str, x: torch.Tensor, sublayer: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        delta = self._linear(conv + ".kernel_projection.weight", x).view(len(x), 2, self.cfg["conv"], -1)
        y = sublayer(self._conv(conv + ".base_kernel", x, delta[:, 0], 0))
        return self._conv(conv + ".base_kernel", y, delta[:, 1], 1)

    def block(self, features: torch.Tensor, anchor: int) -> torch.Tensor:
        """The final-norm hidden states of block rows 1 … B − 1 [B − 1, H]."""
        with torch.device(self.device):
            return self._block(features.to(self.device), anchor)

    def _block(self, features: torch.Tensor, anchor: int) -> torch.Tensor:
        c = self.cfg
        n, b, heads, dim = len(features), c["block"], c["heads"], c["head_dim"]
        context = self._norm("hidden_norm.weight", self._linear("fc.weight", features.to(self.dtype)))
        positions = torch.arange(n + b)
        keys = positions[None]
        queries = positions[n:, None]
        visible = keys <= queries if c["causal"] else torch.ones(b, n + b, dtype=torch.bool)
        if c["window"]:
            visible = visible & (queries - keys < c["window"])
        ids = torch.tensor([anchor] + [c["mask"]] * (b - 1), device="cpu")
        x, residual = self.embed[ids].to(self.device, self.dtype), None
        for i in range(c["layers"]):
            p = f"layers.{i}."
            residual = x if residual is None else residual + x
            h = self._norm(p + "input_layernorm.weight", residual)

            def attention(h: torch.Tensor, i: int = i, p: str = p) -> torch.Tensor:
                ck, cv = self._kv(i, context, positions[:n])
                bk, bv = self._kv(i, h, positions[n:])
                q = self._linear(p + "self_attn.q_proj.weight", h).view(b, heads, dim)
                q = self._rope(self._norm(p + "self_attn.q_norm.weight", q), positions[n:])
                k, v = (torch.cat(t).repeat_interleave(heads // c["kv_heads"], 1) for t in ((ck, bk), (cv, bv)))
                scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) / dim ** 0.5
                probs = scores.masked_fill(~visible, float("-inf")).softmax(-1)
                o = torch.einsum("hqk,khd->qhd", probs, v.float()).to(self.dtype).reshape(b, -1)
                return self._linear(p + "self_attn.o_proj.weight", o)

            def mlp(h: torch.Tensor, p: str = p) -> torch.Tensor:
                g, u = self._linear(p + "mlp.gate_proj.weight", h), self._linear(p + "mlp.up_proj.weight", h)
                return self._linear(p + "mlp.down_proj.weight", (torch.nn.functional.silu(g.float()) * u.float())
                                    .to(self.dtype))

            residual = residual + self._wrapped(p + "attention_conv", h, attention)
            x = self._wrapped(p + "mlp_conv", self._norm(p + "post_attention_layernorm.weight", residual), mlp)
        return self._norm("norm.weight", residual + x)[1:]

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """The target head's logits of draft rows [B − 1, V] (fp32)."""
        return self.matmul("lm_head", hidden, self.head).float()

    def scores(self, cand: torch.Tensor, unary: torch.Tensor, hidden: torch.Tensor, anchor: int) -> torch.Tensor:
        """The selector's lattice [B − 1, K, K] for one block's candidates (ids and logits [B − 1, K])."""
        projected = self._linear("candidate_selector.hidden_projection.weight", hidden).float()
        before = torch.cat([torch.full((1, cand.shape[1]), anchor, device=cand.device), cand[:-1]])
        a = self.w["candidate_selector.predecessor_codebook"][before].float()
        s = self.w["candidate_selector.successor_codebook"][cand].float()
        return unary[:, None, :] + torch.einsum("epr,ecr->epc", a * projected[:, None], s)

    def greedy(self, hidden: torch.Tensor, anchor: int, k: int) -> torch.Tensor:
        """The greedy walk's drafts [B − 1]."""
        unary, cand = self.logits(hidden).topk(k)
        scores = self.scores(cand, unary, hidden, anchor)
        prev, out = 0, []
        for e in range(len(cand)):
            prev = int(scores[e, prev].argmax())
            out.append(int(cand[e, prev]))
        return torch.tensor(out)
