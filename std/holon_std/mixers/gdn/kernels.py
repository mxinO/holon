"""The op contracts of the Gated DeltaNet mixer (docs/m03-qwen38.md D3). Their oracles are in `reference.py`; kernels
come from kernel plugins. Kernel instances are stateless across steps and safe to call inside CUDA graphs.

Per-sequence state lives in a pool of units addressed by `slots` (one unit per batch row; a row with no tokens leaves
its unit untouched). `has_init[r]` is False for a row that starts a sequence (its start position is 0): it then starts
from zero state and never reads its unit, so a unit reused by a new sequence needs no clearing. Rows are given by
`query_start` [R+1] (int32, the prefix sums of each row's token count); T = query_start[-1].

- `causal_conv1d@1`: instance `(x [T, C], weight [C, K], state [units, C, K-1], slots [R], query_start [R+1],
  has_init [R]) -> y [T, C]`: per row, a depthwise causal convolution of the row's tokens continuing the K-1 inputs in
  `state[slots[r]]` (zeros without `has_init`), `y = silu(conv)` in f32, cast to x's dtype; then `state[slots[r]]`
  holds the last K-1 inputs of (previous ‖ row). The weights are used in their stored dtype (f32 for KDA's convs,
  docs/v1-glm.md D4; the model dtype for GDN's). Facts: channels, kernel_size, dtype[, weight_dtype: only when the
  weights' dtype is not `dtype`].
- `gated_delta_rule@1`: instance with `plan(lens: list[int], snapshots=(), per_token=False) -> handle` (host, once
  per step: per-row token counts; `snapshots` (row, offset, unit): every provider writes row's state after its first
  `offset` tokens, a multiple of 64 with 0 < offset < the row's length, to `snapshot[unit]` — an in-step checkpoint;
  `per_token`: each row is computed token by token exactly as that many one-token rows would be, no snapshots —
  verify rows and their replays, M0.4 D6) and `(handle, q [T, Hk, Dk], k [T, Hk, Dk], v [T, Hv, Dv], a [T, Hv],
  b [T, Hv], a_log [Hv] f32, dt_bias [Hv] f32, state [units, Hv, Dv, Dk] f32, slots [R], has_init [R],
  snapshot=None [units', Hv, Dv, Dk] f32, out_slots=None [R]) -> o [T, Hv, Dv]` of v's dtype (`out_slots`: the units
  each row's final state is written to instead of its `slots` unit; a deployment that passes it or plans `per_token`
  selects with the fact `out_slots=True`): q and k l2-normalised
  (`x · rsqrt(Σx² + 1e-6)`) in f32, q scaled by Dk^-½, `g = −exp(a_log) · softplus(a + dt_bias)`, `β = sigmoid(b)`;
  per row from `S = state[slots[r]]` (zeros without `has_init`), for each token `S ← e^g S; S += k (β(v − Sk))ᵀ;
  o = S q` (S is [Dv, Dk] per v-head), and the final S written back. V-head j reads q/k head `j // (Hv/Hk)`
  (`v_head_order = "grouped"`, HF's) or `j % Hk` (`"tiled"`, GGUF's). Facts: heads_k, heads_v, head_k_dim,
  head_v_dim, v_head_order, dtype.
- `kda@1` (GLM-5.3-Flash's Kimi delta attention, docs/v1-glm.md D4): `gated_delta_rule@1`'s `plan` and call terms
  (rows, slots, `has_init`, in-step snapshots, `per_token`, `out_slots`) with a per-channel gate: instance `(handle,
  q [T, H, Dk], k [T, H, Dk], v [T, H, Dv], f [T, H·Dk], b [T, H], a_log [H] f32, dt_bias [H·Dk] f32, state [units, H,
  Dv, Dk] f32, slots [R], has_init [R], snapshot=None, out_slots=None) -> o [T, H, Dv]` of v's dtype: q and k
  l2-normalised (`x / sqrt(Σx² + 1e-6)`, HF's and FLA's form) in f32, q scaled by Dk^-½; the gate per channel
  `g = lower_bound · σ(exp(a_log) · (f + dt_bias))` (`a_log` per head, broadcast over its Dk channels), `β = σ(b)`;
  per row from `S = state[slots[r]]` (zeros without `has_init`), for each token `S ← S · diag(e^g)` (each key channel
  decayed), `S += (β(v − S k)) kᵀ; o = S q`, and the final S written back. Facts: heads, head_k_dim, head_v_dim,
  lower_bound, dtype[, out_slots].
- `gated_rmsnorm@1`: instance `(o [T, H, D], z [T, H, D], weight [D]) -> [T, H, D]`: per head
  `weight · (o · rsqrt(mean(o²) + eps)) · act(z)`, the norm in f32 cast to o's dtype before the weight (HF's
  `RMSNormGated`), the gate in f32, act = silu, or sigmoid with the fact `out_gate="sigmoid"` (Qwen3.8-Flash-Next's
  GDN, docs/v1-flash-next.md D8; given only when not silu, so the facts every other site records stay as they were);
  with the fact `round = "none"` (KDA's o_norm, docs/v1-glm.md D4; given only then) the norm stays f32 through the
  weight and the gate, rounded once at the end. Facts: dim, eps, dtype[, out_gate, round].
- `gdn_rewind@1` (after a verify step, every layer of a state group at once; docs/m07-perf.md D5): instance
  `(state [L, units, C, K-1], window [L, R', C, K-1], x [L, T', C], a [L, T', W], b [L, T', Hv], slots [R],
  accepted [S], width) -> None`, with R ≤ R' rows of `width` tokens (row r's at r·width in each layer's x, a and b; its
  conv inputs before them in window[:, r]) and n_r = accepted[slots[r]] (int64) of its drafts accepted: in place,
  a and b ← −inf (in their dtype) at each row's tokens past its first 1 + n_r, and state[:, slots[r]] ← the last K-1
  inputs of (window[:, r] ‖ the row's first 1 + n_r tokens of x). The gate width W is Hv (GDN's a) or any other
  per-row width (KDA's per-channel gate f, H·Dk: the fact `gate_width`, given only when it is not `heads_v`);
  −inf is the delta rules' identity step for both (GDN: softplus(−inf) = 0, σ(−inf) = 0 — the bits −1e4 gave; KDA:
  −5 · σ(−inf) = −0, β = 0; docs/v1-glm.md D4). A unit shared by several rows (graph padding) is left unspecified
  (deterministic otherwise). Facts: channels, kernel_size, heads_v, dtype[, gate_width].
"""
from __future__ import annotations

CAUSAL_CONV1D, GATED_DELTA_RULE, GATED_RMSNORM = "causal_conv1d@1", "gated_delta_rule@1", "gated_rmsnorm@1"
KDA = "kda@1"
GDN_REWIND = "gdn_rewind@1"
MASKED = float("-inf")             # gdn_rewind's gates past the accepted tokens: the delta rules' identity step
ORDERS = ("grouped", "tiled")
OUT_GATES = ("silu", "sigmoid")                # gated_rmsnorm's activation of z (`out_gate`; silu by default)
