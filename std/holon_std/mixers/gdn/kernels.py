"""The op contracts of the Gated DeltaNet mixer (docs/m03-qwen38.md D3). Their oracles are in `reference.py`; kernels
come from kernel plugins. Kernel instances are stateless across steps and safe to call inside CUDA graphs.

Per-sequence state lives in a pool of units addressed by `slots` (one unit per batch row; a row with no tokens leaves
its unit untouched). `has_init[r]` is False for a row that starts a sequence (its start position is 0): it then starts
from zero state and never reads its unit, so a unit reused by a new sequence needs no clearing. Rows are given by
`query_start` [R+1] (int32, the prefix sums of each row's token count); T = query_start[-1].

- `causal_conv1d@1`: instance `(x [T, C], weight [C, K], state [units, C, K-1], slots [R], query_start [R+1],
  has_init [R]) -> y [T, C]`: per row, a depthwise causal convolution of the row's tokens continuing the K-1 inputs in
  `state[slots[r]]` (zeros without `has_init`), `y = silu(conv)` in f32, cast to x's dtype; then `state[slots[r]]`
  holds the last K-1 inputs of (previous ‖ row). Facts: channels, kernel_size, dtype.
- `gated_delta_rule@1`: instance with `plan(lens: list[int], snapshots=(), per_token=False) -> handle` (host, once
  per step: per-row token counts; `snapshots` (row, offset, unit): every provider writes row's state after its first
  `offset` tokens, a multiple of 64 with 0 < offset < the row's length, to `snapshot[unit]` — an in-step checkpoint;
  `per_token`: each row is computed token by token exactly as that many one-token rows would be, no snapshots —
  verify rows and their replays, M0.4 D6) and `(handle, q [T, Hk, Dk], k [T, Hk, Dk], v [T, Hv, Dv], a [T, Hv],
  b [T, Hv], a_log [Hv] f32, dt_bias [Hv] f32, state [units, Hv, Dv, Dk] f32, slots [R], has_init [R],
  snapshot=None [units', Hv, Dv, Dk] f32, out_slots=None [R]) -> o [T, Hv, Dv]` of v's dtype (`out_slots`: the units
  each row's final state is written to instead of its `slots` unit; a deployment that passes it selects with the fact
  `out_slots=True`): q and k l2-normalised
  (`x · rsqrt(Σx² + 1e-6)`) in f32, q scaled by Dk^-½, `g = −exp(a_log) · softplus(a + dt_bias)`, `β = sigmoid(b)`;
  per row from `S = state[slots[r]]` (zeros without `has_init`), for each token `S ← e^g S; S += k (β(v − Sk))ᵀ;
  o = S q` (S is [Dv, Dk] per v-head), and the final S written back. V-head j reads q/k head `j // (Hv/Hk)`
  (`v_head_order = "grouped"`, HF's) or `j % Hk` (`"tiled"`, GGUF's). Facts: heads_k, heads_v, head_k_dim,
  head_v_dim, v_head_order, dtype.
- `gated_rmsnorm@1`: instance `(o [T, H, D], z [T, H, D], weight [D]) -> [T, H, D]`: per head
  `weight · (o · rsqrt(mean(o²) + eps)) · silu(z)`, the norm in f32 cast to o's dtype before the weight (HF's
  `RMSNormGated`), the gate in f32. Facts: dim, eps, dtype.
"""
from __future__ import annotations

CAUSAL_CONV1D, GATED_DELTA_RULE, GATED_RMSNORM = "causal_conv1d@1", "gated_delta_rule@1", "gated_rmsnorm@1"
ORDERS = ("grouped", "tiled")
