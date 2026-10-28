"""Qwen3.8-Flash-Next's PLE op contracts (docs/dev/v1-flash-next.md D4, D13′: in the model plugin until a second user
appears, prefixed like the unit). Oracles in `reference.py`; kernels come from kernel plugins (`kernels.ple_triton`).
Kernel instances are stateless across steps and safe to call inside CUDA graphs. Per-request state lives in units
addressed by `slots` (one per batch row); `has_init[r]` False: row r starts its sequence and reads no state. Rows are
given by `query_start` [R+1] (int32 prefix sums of each row's token count); T = query_start[-1].

- `qwen4_exp.ple_rows@1`: instance `(ids [T] int64, history [units, 2] int64, slots [R], query_start [R+1],
  has_init [R], multipliers [ngram] int64, sizes [heads] int64, offsets [heads] int64) -> rows [T, heads] int64`: each
  token's n-gram table rows (HF's `Qwen4ExpTextNGramEmbedding`, `modeling_qwen4_exp.py:1123-1172`). Over the row's
  history (its last two ids before this step; EOS EOS without has_init) followed by its ids, the token i and its
  s-th predecessor for s < ngram, the predecessor replaced by EOS when an EOS lies at or after it before i (the window
  is cut at EOS) or before the history; for n = 2 … ngram, `mixed = ⊕_{s<n} id_{i−s} · multipliers[s]` (int64,
  wrapping), and for each of that n-gram's `heads_per_ngram` heads h: `floor_mod(mixed, sizes[h]) + offsets[h]`.
  Then `history[slots[r]]` holds the last two ids of (history ‖ row). Facts: ngram, heads_per_ngram, eos.
- `qwen4_exp.ple_block@1`: instance `(key [T, n·H], value [T, H], r [T, n·H] f32, w_key, w_query, w_conv [n·H],
  conv [n·H, K], state [units, (K−1)·dilation, n·H] f32, slots [R], query_start [R+1], has_init [R])
  -> C [T, n·H] f32`: PLE's
  block (`modeling_qwen4_exp.py:1227-1247`), float32 throughout:
  - `K = norm(key) · (1 + w_key)` and `Q = norm(r) · (1 + w_query)`, each RMS-normed per stream of H (eps);
  - `u_s = ⟨K_s, Q_s⟩ / √H`; `g_s = sigmoid(sign(u_s) · √max(|u_s|, 1e-6))`; `G_s = g_s · value`;
  - `C = norm(G) · (1 + w_conv)` per stream; `conv` depthwise over time with `dilation`: at token t,
    `Σ_k conv[:, k] · C[t − (K − 1 − k) · dilation]`, the positions before the row's first token read from
    `state[slots[r]]` (zeros without has_init);
  - in place `r += G + silu(conv)`; then `state[slots[r]]` holds the last (K−1)·dilation rows of (state ‖ the row's C);
  - returns C, the conv's inputs: the unit builds in-step checkpoints (the window before a position) and a verify
    row's rewind (the window after its accepted tokens) from it; a verify step passes a scratch copy of its rows' state
    (their real state stays as before the step until the rewind).
  Every reduction (the norms, ⟨K_s, Q_s⟩) is per token, in an order independent of the token count and launch shape,
  and the conv is a per-channel sum, so a row split into chunks (the same key and value rows in) is bitwise one shot
  (acceptance 2's unit test). Facts: streams, hidden, kernel, dilation, eps, dtype (key's and value's).
"""
from __future__ import annotations

ROWS, BLOCK = "qwen4_exp.ple_rows@1", "qwen4_exp.ple_block@1"
