# S3 — Gated DeltaNet kernels (Qwen3.5 / Qwen3.8)

## Question

Can the flash-linear-attention ops copied into vLLM be vendored without any `vllm.*` import and serve as the
plug-serve GDN kernel plugin on sm_86 and sm_120 — chunked prefill, recurrent decode and spec verify — within
budget against the HF pure-torch reference, with no NaN/Inf for T = 1..8192, graph-capturable decode with state
indices in device tensors, and run-to-run byte identity? And can the same kernels consume GGUF's tiled
value-head order through a `v_head_order` knob instead of repacking weights?

**Answer: yes on all counts** (details and caveats below).

## Setup

| | |
|---|---|
| sm_86 | RTX A5000 24 GB |
| sm_120 | RTX PRO 6000 Blackwell Max-Q, peak torch allocation < 4 GiB |
| software | `.venv` python: torch 2.11.0+cu130, Triton 3.6.0, transformers 5.6.0.dev0; `import fla` fails (ModuleNotFoundError), and `gdn.py` asserts HF runs its torch path |
| weights | layer 0 (a GDN layer) of `Qwen/Qwen3.5-4B` (Hg=16, H=32) and `Qwen/Qwen3.8-27B` (Hg=16, H=48), from the HF cache; random inputs through the layer's real input RMSNorm |
| perf shapes | `Qwen3.5-9B` (Hg=16, H=32, 24 GDN layers), `Qwen3.8-27B` (Hg=16, H=48, 48 GDN layers), K=V=128, read from the cached configs |

Files (all scripts take the result path; run from this directory):

| file | job |
|---|---|
| `vendor.sh`, `vendor.patch` | recreate `third_party/fla_ops/` (gitignored) from a vLLM checkout: `./vendor.sh /tmp/cmp/vllm` |
| `gdn.py` | harness: load one real layer, capture its core inputs, HF references, kernel wrapper, error metrics |
| `correctness.py` | task 3 → `results/correctness_{sm86,sm120}.json` |
| `tune.py` | search vLLM's autotune space → `results/tune_*.json` (source of the pinned configs) |
| `bench.py` | task 4 → `results/perf_*.json`, `results/perf_*_autotune.json` (`bench.py out.json autotune`) |
| `graph_determinism.py` | task 5 → `results/graph_det_*.json` |
| `tiled_order.py` | task 6 → `results/tiled_order_*.json` |
| `report.py` | prints the tables below from `results/` |

Commands: `CUDA_VISIBLE_DEVICES=0 ../../.venv/bin/python <script>.py results/<name>_sm86.json`, and the same over
on the sm_120 GPU for `_sm120`. Logs in `out/`.

## Vendored subset and provenance

- **Source:** https://github.com/vllm-project/vllm, commit `6dd813e0b7ca8314fed28c13a73cfb5fc5310163`
  (2026-10-05), `vllm/third_party/flash_linear_attention/`. Files carry `SPDX-License-Identifier: Apache-2.0`
  (vLLM) over code copied from flash-linear-attention (MIT, © 2023-2025 Songlin Yang, Yu Zhang; the directory's
  `LICENSE` is that MIT text and is vendored with it). Both are compatible (design §5.6 rule 5).
- **Files (14 + LICENSE):** `chunk.py` (prefill entry), `chunk_delta_h.py`, `chunk_o.py`, `chunk_scaled_dot_kkt.py`,
  `cumsum.py`, `solve_tril.py`, `wy_fast.py`, `l2norm.py`, `index.py`, `op.py`, `utils.py`, `fused_recurrent.py`
  (generic recurrent + `packed_decode`), `fused_sigmoid_gating.py` (recurrent with fused gating — vLLM's decode/verify
  path), `__init__.py`. **3201 lines** (≈ 2750 non-blank, non-comment; upstream 3374 for the same files).
  Excluded: `kda.py` (1702), `layernorm_guard.py` (615), `fused_norm_gate.py` (413),
  `fused_gdn_prefill_post_conv.py` (248).
- **Changes (`vendor.patch`, 419 changed lines):**
  - every `vllm.*` import removed: `vllm.triton_utils` → `triton`/`triton.language`; `vllm.platforms` → NVIDIA-only
    constants; `gpu_sync_allowed` / `async_tensor_h2d` → plain `.tolist()` / `.to()`; ROCm-only branches dropped.
  - env switches removed (`FLA_USE_FAST_OPS`, `FLA_TRIL_PRECISION`=ieee, `USE_DEFAULT_FLA_NORM` → only the
    non-autotuned l2norm kernel kept, `FLA_USE_CUDA_GRAPH`, `FLA_USE_TMA`, `GDN_RECOMPUTE_SUPPRESS_LEVEL`).
  - `tensor_cache` removed: it memoises chunk indices by tensor identity, which returns stale indices when a static
    `cu_seqlens` buffer is rewritten in place — exactly what plug-serve's static buffers do.
  - `@triton.autotune` → `utils.autotune`: one config per kernel (`utils.PINNED`, else the first listed) — §5.6 rule 7.
    `S3_FLA_AUTOTUNE=1` (spike only) restores vLLM's search space for `tune.py` and the comparison runs.
  - `v_head_tiled` knob (constexpr `V_HEAD_TILED`) in all 6 kernels that map value heads to q/k heads: q/k head =
    `v_head % Hg` instead of `v_head // (H / Hg)` (task 6).
- `grep -r vllm third_party/fla_ops` finds only SPDX/comment lines.

## Correctness

Golden: HF `torch_recurrent_gated_delta_rule` (token-by-token) in fp32 from the same bf16 q/k/v/g/beta the layer
produces. Noise floor "HF bf16": what HF computes in a bf16 model (its chunk path, l2norm in bf16, bf16 output; at
layer level its cache also keeps the state in bf16). "HF fp32 chunk" shows the reference choice does not matter
(≤ 2.9e-5). Cells are `max_abs / max_rel / fro_rel` with `max_rel = max|Δ| / max|ref|`. Kernel state is fp32
(`[slots, H, V, K]`); HF's is `[B, H, K, V]` (transposed for comparison).

Spike budget (until S5 calibrates real ones): kernel-level `max_rel ≤ 1e-2` and `fro_rel ≤ 1e-2`; layer-level
`fro_rel` ≤ 1.5 × HF bf16's. **Measured worst case over both archs and both models:** kernel prefill o
`max_rel 8.4e-3 / fro_rel 4.9e-3`, state `7.7e-3 / 4.3e-3`; decode/verify o `5.1e-3 / 3.6e-3`, state
`5.3e-3 / 3.9e-3`; HF bf16 itself `6.9e-3 / 3.9e-3` (o), `5.0e-3 / 3.2e-3` (state). Error does not grow with T.
Layer-level ratio ours / HF bf16: 0.59–1.34. **Every output and state tensor is finite** (T = 1..8192, both archs).

Highlights: chunk-aligned resume (cuts at multiples of 64) is **bitwise identical** to one-shot prefill; unaligned
resume moves the chunk grid, so it is not bitwise but has the same error vs golden. Verify-k with 2-D slot indices is
**bitwise identical** to k+1 sequential decode steps (outputs and every per-token slot state), as is the replay commit.

#### NVIDIA RTX A5000 (sm86); every tensor finite: True


**4b** (Hg=16, H=32) — prefill one shot vs golden; cells: max_abs / max_rel / fro_rel

| T | kernel o | kernel state | HF bf16 o | HF bf16 state | HF fp32 chunk o |
|---|---|---|---|---|---|
| 1 | 3.5e-06 / 2.2e-03 / 2.3e-03 | 2.2e-04 / 2.6e-03 / 2.1e-03 | 2.2e-06 / 1.5e-03 / 1.6e-03 | 3.0e-04 / 3.4e-03 / 2.7e-03 | 2.9e-10 / 1.9e-07 / 1.9e-07 |
| 16 | 1.6e-04 / 3.4e-03 / 3.2e-03 | 2.8e-03 / 2.6e-03 / 3.5e-03 | 1.6e-04 / 3.4e-03 / 3.6e-03 | 1.7e-03 / 1.6e-03 / 2.3e-03 | 1.5e-08 / 3.2e-07 / 2.7e-07 |
| 64 | 2.3e-04 / 4.3e-03 / 3.8e-03 | 2.9e-03 / 4.6e-03 / 3.8e-03 | 2.3e-04 / 4.3e-03 / 3.4e-03 | 1.4e-03 / 2.2e-03 / 3.0e-03 | 1.5e-08 / 2.8e-07 / 3.4e-07 |
| 512 | 8.3e-04 / 7.5e-03 / 3.7e-03 | 5.8e-03 / 4.0e-03 / 3.5e-03 | 5.7e-04 / 5.1e-03 / 3.4e-03 | 3.3e-03 / 2.3e-03 / 2.5e-03 | 2.3e-07 / 2.1e-06 / 1.1e-06 |
| 2048 | 6.8e-04 / 3.7e-03 / 3.3e-03 | 4.3e-03 / 2.7e-03 / 3.3e-03 | 7.6e-04 / 4.2e-03 / 3.2e-03 | 2.5e-03 / 1.5e-03 / 2.5e-03 | 1.3e-06 / 7.0e-06 / 3.8e-06 |
| 8192 | 1.3e-03 / 4.1e-03 / 3.0e-03 | 5.8e-03 / 2.5e-03 / 3.3e-03 | 1.9e-03 / 6.2e-03 / 3.1e-03 | 2.8e-03 / 1.2e-03 / 2.3e-03 | 6.5e-06 / 2.1e-05 / 1.4e-05 |

| batch-4 varlen T | h0≠0 | o | state |
|---|---|---|---|
| 1 | False | 2.7e-05 / 6.1e-03 / 3.8e-03 | 4.1e-04 / 3.5e-03 / 2.3e-03 |
| 63 | False | 2.6e-04 / 4.3e-03 / 3.7e-03 | 7.0e-03 / 5.2e-03 / 3.7e-03 |
| 777 | True | 6.5e-04 / 4.3e-03 / 3.3e-03 | 3.4e-03 / 2.4e-03 / 3.2e-03 |
| 2048 | True | 1.2e-03 / 5.2e-03 / 3.1e-03 | 5.4e-03 / 2.8e-03 / 3.3e-03 |

| resume cuts | bitwise = one shot | resumed vs one shot o | resumed o vs golden | resumed state vs golden |
|---|---|---|---|---|
| [0, 37, 600, 1500, 2048] | False | 9.8e-04 / 5.6e-03 / 3.6e-03 | 8.3e-04 / 4.7e-03 / 3.3e-03 | 4.0e-03 / 2.7e-03 / 3.5e-03 |
| [0, 64, 1024, 1536, 2048] | True | 0.0e+00 / 0.0e+00 / 0.0e+00 | 7.8e-04 / 4.4e-03 / 3.3e-03 | 6.4e-03 / 4.2e-03 / 3.7e-03 |

| decode kernel (32 steps after prefill) | worst o | worst state | other slots untouched |
|---|---|---|---|
| fused_recurrent | 1.3e-04 / 3.0e-03 / 2.7e-03 | 3.6e-03 / 1.9e-03 / 2.9e-03 | True |
| fused_sigmoid_gating | 1.3e-04 / 3.0e-03 / 2.8e-03 | 2.8e-03 / 2.7e-03 / 3.1e-03 | True |
| packed_decode | 1.3e-04 / 3.0e-03 / 2.8e-03 | 2.8e-03 / 2.7e-03 / 3.1e-03 | True |

verify k=3, n_accept=[1, 2, 3, 4]: checkpoint_slots step A o/slot states bitwise = sequential: True/True; step B (reads slot n_accept-1) bitwise = sequential: True, worst o vs golden 1.9e-04 / 4.7e-03 / 3.2e-03; replay: verify o bitwise = checkpoint_slots True, committed state bitwise = sequential True.

| layer level T (+16 decode) | prefill ours | prefill HF bf16 | ratio | decode ours | decode HF bf16 | ratio |
|---|---|---|---|---|---|---|
| 16 | 2.1e-02 / 6.0e-03 / 7.6e-03 | 8.8e-03 / 2.5e-03 / 6.4e-03 | 1.20 | 2.2e-02 / 1.5e-03 / 4.6e-03 | 2.2e-02 / 1.5e-03 / 5.0e-03 | 0.93 |
| 512 | 1.6e-01 / 4.7e-03 / 5.5e-03 | 9.7e-02 / 2.9e-03 / 5.3e-03 | 1.04 | 2.7e-02 / 1.6e-03 / 4.8e-03 | 3.5e-02 / 2.1e-03 / 7.0e-03 | 0.69 |
| 2048 | 1.9e-01 / 3.9e-03 / 5.3e-03 | 1.4e-01 / 2.8e-03 / 5.0e-03 | 1.06 | 3.3e-02 / 2.2e-03 / 5.5e-03 | 3.2e-02 / 2.1e-03 / 6.4e-03 | 0.85 |
| 8192 | 2.4e-01 / 3.8e-03 / 5.2e-03 | 2.0e-01 / 3.1e-03 / 5.0e-03 | 1.04 | 3.9e-02 / 3.2e-03 / 4.8e-03 | 3.9e-02 / 3.2e-03 / 4.9e-03 | 0.98 |

**27b** (Hg=16, H=48) — prefill one shot vs golden; cells: max_abs / max_rel / fro_rel

| T | kernel o | kernel state | HF bf16 o | HF bf16 state | HF fp32 chunk o |
|---|---|---|---|---|---|
| 1 | 5.4e-06 / 1.8e-03 / 2.4e-03 | 7.4e-04 / 3.5e-03 / 2.6e-03 | 5.4e-06 / 1.8e-03 / 2.0e-03 | 9.8e-04 / 4.7e-03 / 2.4e-03 | 2.3e-10 / 7.8e-08 / 1.3e-07 |
| 16 | 2.3e-04 / 4.9e-03 / 4.1e-03 | 6.7e-03 / 4.7e-03 / 3.6e-03 | 2.3e-04 / 4.9e-03 / 3.3e-03 | 3.8e-03 / 2.7e-03 / 2.7e-03 | 1.1e-08 / 2.4e-07 / 2.6e-07 |
| 64 | 6.4e-04 / 4.6e-03 / 3.6e-03 | 7.8e-03 / 3.8e-03 / 3.8e-03 | 3.9e-04 / 2.8e-03 / 3.4e-03 | 6.8e-03 / 3.3e-03 / 2.7e-03 | 7.5e-08 / 5.3e-07 / 3.7e-07 |
| 512 | 1.1e-03 / 4.1e-03 / 3.5e-03 | 6.2e-03 / 2.7e-03 / 4.0e-03 | 1.2e-03 / 4.5e-03 / 3.3e-03 | 5.3e-03 / 2.3e-03 / 3.2e-03 | 4.8e-07 / 1.8e-06 / 1.7e-06 |
| 2048 | 1.4e-03 / 4.5e-03 / 3.2e-03 | 9.6e-03 / 2.2e-03 / 3.5e-03 | 1.9e-03 / 5.9e-03 / 3.2e-03 | 5.5e-03 / 1.3e-03 / 2.7e-03 | 3.3e-06 / 1.0e-05 / 5.8e-06 |
| 8192 | 2.5e-03 / 4.5e-03 / 2.9e-03 | 9.9e-03 / 1.3e-03 / 3.3e-03 | 3.6e-03 / 6.4e-03 / 3.0e-03 | 6.5e-03 / 8.3e-04 / 2.4e-03 | 1.6e-05 / 2.8e-05 / 1.5e-05 |

| batch-4 varlen T | h0≠0 | o | state |
|---|---|---|---|
| 1 | False | 8.0e-06 / 3.7e-03 / 3.3e-03 | 3.3e-04 / 2.3e-03 / 2.1e-03 |
| 63 | False | 4.5e-04 / 4.9e-03 / 4.5e-03 | 6.9e-03 / 6.8e-03 / 4.3e-03 |
| 777 | True | 1.1e-03 / 4.8e-03 / 3.2e-03 | 5.9e-03 / 2.0e-03 / 3.6e-03 |
| 2048 | True | 1.6e-03 / 4.3e-03 / 3.1e-03 | 7.5e-03 / 1.7e-03 / 3.6e-03 |

| resume cuts | bitwise = one shot | resumed vs one shot o | resumed o vs golden | resumed state vs golden |
|---|---|---|---|---|
| [0, 37, 600, 1500, 2048] | False | 2.0e-03 / 5.7e-03 / 3.5e-03 | 1.5e-03 / 4.4e-03 / 3.2e-03 | 6.8e-03 / 1.7e-03 / 3.5e-03 |
| [0, 64, 1024, 1536, 2048] | True | 0.0e+00 / 0.0e+00 / 0.0e+00 | 1.5e-03 / 4.3e-03 / 3.2e-03 | 7.2e-03 / 1.8e-03 / 3.5e-03 |

| decode kernel (32 steps after prefill) | worst o | worst state | other slots untouched |
|---|---|---|---|
| fused_recurrent | 2.7e-04 / 4.4e-03 / 3.5e-03 | 6.1e-03 / 2.1e-03 / 3.2e-03 | True |
| fused_sigmoid_gating | 2.7e-04 / 4.4e-03 / 3.6e-03 | 5.8e-03 / 2.0e-03 / 3.3e-03 | True |
| packed_decode | 2.7e-04 / 4.4e-03 / 3.6e-03 | 5.8e-03 / 2.0e-03 / 3.3e-03 | True |

verify k=3, n_accept=[1, 2, 3, 4]: checkpoint_slots step A o/slot states bitwise = sequential: True/True; step B (reads slot n_accept-1) bitwise = sequential: True, worst o vs golden 5.3e-05 / 4.0e-03 / 3.4e-03; replay: verify o bitwise = checkpoint_slots True, committed state bitwise = sequential True.

| layer level T (+16 decode) | prefill ours | prefill HF bf16 | ratio | decode ours | decode HF bf16 | ratio |
|---|---|---|---|---|---|---|
| 16 | 2.7e-01 / 2.2e-03 / 3.4e-03 | 3.5e-01 / 2.8e-03 / 3.9e-03 | 0.86 | 2.4e-01 / 3.7e-03 / 4.9e-03 | 2.4e-01 / 3.7e-03 / 5.4e-03 | 0.91 |
| 512 | 2.3e+00 / 5.1e-03 / 4.6e-03 | 2.3e+00 / 5.1e-03 / 4.5e-03 | 1.02 | 3.9e-01 / 1.5e-03 / 3.1e-03 | 4.7e-01 / 1.8e-03 / 3.7e-03 | 0.83 |
| 2048 | 2.7e+00 / 5.7e-03 / 4.1e-03 | 1.7e+00 / 3.6e-03 / 3.8e-03 | 1.07 | 2.5e-01 / 3.4e-03 / 4.7e-03 | 5.7e-01 / 7.8e-03 / 7.4e-03 | 0.63 |
| 8192 | 2.6e+00 / 4.2e-03 / 3.6e-03 | 2.4e+00 / 3.9e-03 / 3.5e-03 | 1.03 | 3.7e-01 / 1.6e-03 / 4.5e-03 | 1.4e+00 / 6.0e-03 / 7.6e-03 | 0.59 |

#### NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (sm120); every tensor finite: True


**4b** (Hg=16, H=32) — prefill one shot vs golden; cells: max_abs / max_rel / fro_rel

| T | kernel o | kernel state | HF bf16 o | HF bf16 state | HF fp32 chunk o |
|---|---|---|---|---|---|
| 1 | 3.1e-06 / 2.0e-03 / 2.1e-03 | 2.9e-04 / 3.3e-03 / 2.1e-03 | 6.5e-06 / 4.2e-03 / 3.5e-03 | 2.0e-04 / 2.3e-03 / 2.4e-03 | 4.7e-10 / 3.0e-07 / 2.3e-07 |
| 16 | 1.6e-04 / 3.4e-03 / 3.2e-03 | 2.8e-03 / 2.6e-03 / 3.5e-03 | 1.6e-04 / 3.4e-03 / 3.6e-03 | 1.7e-03 / 1.6e-03 / 2.3e-03 | 9.7e-09 / 2.0e-07 / 2.1e-07 |
| 64 | 2.6e-04 / 4.7e-03 / 3.7e-03 | 4.8e-03 / 5.8e-03 / 3.7e-03 | 2.3e-04 / 4.3e-03 / 3.4e-03 | 1.8e-03 / 2.2e-03 / 2.6e-03 | 1.9e-08 / 3.4e-07 / 3.7e-07 |
| 512 | 7.9e-04 / 6.3e-03 / 3.6e-03 | 7.9e-03 / 4.5e-03 / 3.6e-03 | 6.9e-04 / 5.5e-03 / 3.4e-03 | 2.8e-03 / 1.6e-03 / 2.6e-03 | 2.2e-07 / 1.7e-06 / 1.1e-06 |
| 2048 | 8.1e-04 / 4.1e-03 / 3.3e-03 | 6.4e-03 / 4.0e-03 / 3.4e-03 | 7.8e-04 / 3.9e-03 / 3.2e-03 | 3.4e-03 / 2.1e-03 / 2.7e-03 | 2.2e-06 / 1.1e-05 / 5.0e-06 |
| 8192 | 1.2e-03 / 4.1e-03 / 3.0e-03 | 4.8e-03 / 2.2e-03 / 3.2e-03 | 1.8e-03 / 6.3e-03 / 3.1e-03 | 3.1e-03 / 1.4e-03 / 2.4e-03 | 6.3e-06 / 2.2e-05 / 1.4e-05 |

| batch-4 varlen T | h0≠0 | o | state |
|---|---|---|---|
| 1 | False | 2.8e-05 / 6.3e-03 / 4.2e-03 | 4.3e-04 / 3.6e-03 / 2.3e-03 |
| 63 | False | 3.5e-04 / 6.0e-03 / 3.8e-03 | 2.9e-03 / 3.2e-03 / 3.8e-03 |
| 777 | True | 5.8e-04 / 4.5e-03 / 3.3e-03 | 4.9e-03 / 2.8e-03 / 3.5e-03 |
| 2048 | True | 7.3e-04 / 3.4e-03 / 3.2e-03 | 5.2e-03 / 2.7e-03 / 3.2e-03 |

| resume cuts | bitwise = one shot | resumed vs one shot o | resumed o vs golden | resumed state vs golden |
|---|---|---|---|---|
| [0, 37, 600, 1500, 2048] | False | 9.8e-04 / 5.5e-03 / 3.7e-03 | 1.1e-03 / 6.3e-03 / 3.4e-03 | 5.7e-03 / 3.1e-03 / 3.4e-03 |
| [0, 64, 1024, 1536, 2048] | True | 0.0e+00 / 0.0e+00 / 0.0e+00 | 9.3e-04 / 5.2e-03 / 3.4e-03 | 4.0e-03 / 2.2e-03 / 3.3e-03 |

| decode kernel (32 steps after prefill) | worst o | worst state | other slots untouched |
|---|---|---|---|
| fused_recurrent | 1.8e-04 / 2.3e-03 / 2.5e-03 | 2.8e-03 / 1.8e-03 / 2.8e-03 | True |
| fused_sigmoid_gating | 1.8e-04 / 2.3e-03 / 2.6e-03 | 2.9e-03 / 1.8e-03 / 3.0e-03 | True |
| packed_decode | 1.8e-04 / 2.3e-03 / 2.6e-03 | 2.9e-03 / 1.8e-03 / 3.0e-03 | True |

verify k=3, n_accept=[1, 2, 3, 4]: checkpoint_slots step A o/slot states bitwise = sequential: True/True; step B (reads slot n_accept-1) bitwise = sequential: True, worst o vs golden 1.2e-04 / 1.8e-03 / 2.8e-03; replay: verify o bitwise = checkpoint_slots True, committed state bitwise = sequential True.

| layer level T (+16 decode) | prefill ours | prefill HF bf16 | ratio | decode ours | decode HF bf16 | ratio |
|---|---|---|---|---|---|---|
| 16 | 2.1e-02 / 6.0e-03 / 7.7e-03 | 2.1e-02 / 6.0e-03 / 6.8e-03 | 1.14 | 1.7e-02 / 1.2e-03 / 4.3e-03 | 1.7e-02 / 1.2e-03 / 5.3e-03 | 0.81 |
| 512 | 1.1e-01 / 3.4e-03 / 5.4e-03 | 1.1e-01 / 3.4e-03 / 5.2e-03 | 1.04 | 5.0e-02 / 5.1e-03 / 6.5e-03 | 5.0e-02 / 5.1e-03 / 8.0e-03 | 0.81 |
| 2048 | 1.9e-01 / 3.2e-03 / 5.3e-03 | 1.9e-01 / 3.1e-03 / 5.1e-03 | 1.03 | 1.0e-01 / 2.5e-03 / 5.2e-03 | 1.0e-01 / 2.5e-03 / 4.7e-03 | 1.10 |
| 8192 | 1.9e-01 / 3.2e-03 / 5.3e-03 | 1.7e-01 / 2.9e-03 / 5.1e-03 | 1.03 | 3.6e-02 / 4.6e-03 / 6.5e-03 | 4.3e-02 / 5.5e-03 / 8.3e-03 | 0.78 |

**27b** (Hg=16, H=48) — prefill one shot vs golden; cells: max_abs / max_rel / fro_rel

| T | kernel o | kernel state | HF bf16 o | HF bf16 state | HF fp32 chunk o |
|---|---|---|---|---|---|
| 1 | 1.6e-05 / 5.3e-03 / 4.0e-03 | 7.5e-04 / 3.6e-03 / 2.7e-03 | 5.2e-06 / 1.7e-03 / 2.2e-03 | 1.0e-03 / 4.9e-03 / 2.7e-03 | 7.0e-10 / 2.3e-07 / 2.1e-07 |
| 16 | 2.4e-04 / 5.2e-03 / 4.5e-03 | 5.4e-03 / 3.8e-03 / 3.6e-03 | 2.4e-04 / 5.2e-03 / 3.6e-03 | 2.5e-03 / 1.8e-03 / 2.5e-03 | 1.3e-08 / 2.9e-07 / 2.6e-07 |
| 64 | 3.5e-04 / 4.5e-03 / 4.0e-03 | 1.2e-02 / 7.7e-03 / 4.2e-03 | 5.4e-04 / 6.9e-03 / 3.9e-03 | 3.4e-03 / 2.1e-03 / 3.2e-03 | 4.1e-08 / 5.3e-07 / 3.9e-07 |
| 512 | 1.1e-03 / 5.4e-03 / 3.7e-03 | 8.8e-03 / 3.4e-03 / 4.1e-03 | 9.2e-04 / 4.7e-03 / 3.4e-03 | 4.5e-03 / 1.8e-03 / 3.0e-03 | 4.2e-07 / 2.1e-06 / 1.8e-06 |
| 2048 | 1.2e-03 / 4.1e-03 / 3.2e-03 | 6.8e-03 / 1.6e-03 / 3.5e-03 | 1.9e-03 / 6.3e-03 / 3.1e-03 | 5.2e-03 / 1.3e-03 / 2.8e-03 | 2.6e-06 / 8.8e-06 / 5.4e-06 |
| 8192 | 3.7e-03 / 6.0e-03 / 2.9e-03 | 1.4e-02 / 1.6e-03 / 3.2e-03 | 4.1e-03 / 6.6e-03 / 3.0e-03 | 5.6e-03 / 6.5e-04 / 2.4e-03 | 1.8e-05 / 2.9e-05 / 1.5e-05 |

| batch-4 varlen T | h0≠0 | o | state |
|---|---|---|---|
| 1 | False | 1.8e-05 / 8.4e-03 / 4.9e-03 | 3.4e-04 / 2.4e-03 / 2.1e-03 |
| 63 | False | 3.3e-04 / 3.2e-03 / 3.9e-03 | 7.4e-03 / 4.7e-03 / 3.9e-03 |
| 777 | True | 1.0e-03 / 4.7e-03 / 3.3e-03 | 7.0e-03 / 2.5e-03 / 3.6e-03 |
| 2048 | True | 1.1e-03 / 3.6e-03 / 3.1e-03 | 7.1e-03 / 1.5e-03 / 3.5e-03 |

| resume cuts | bitwise = one shot | resumed vs one shot o | resumed o vs golden | resumed state vs golden |
|---|---|---|---|---|
| [0, 37, 600, 1500, 2048] | False | 2.0e-03 / 5.7e-03 / 3.5e-03 | 1.4e-03 / 4.1e-03 / 3.2e-03 | 7.4e-03 / 1.8e-03 / 3.6e-03 |
| [0, 64, 1024, 1536, 2048] | True | 0.0e+00 / 0.0e+00 / 0.0e+00 | 1.4e-03 / 4.2e-03 / 3.2e-03 | 6.5e-03 / 1.6e-03 / 3.5e-03 |

| decode kernel (32 steps after prefill) | worst o | worst state | other slots untouched |
|---|---|---|---|
| fused_recurrent | 2.8e-04 / 2.7e-03 / 2.9e-03 | 5.7e-03 / 2.5e-03 / 3.5e-03 | True |
| fused_sigmoid_gating | 3.0e-04 / 2.9e-03 / 2.9e-03 | 6.0e-03 / 2.6e-03 / 3.5e-03 | True |
| packed_decode | 3.0e-04 / 2.9e-03 / 2.9e-03 | 6.0e-03 / 2.6e-03 / 3.5e-03 | True |

verify k=3, n_accept=[1, 2, 3, 4]: checkpoint_slots step A o/slot states bitwise = sequential: True/True; step B (reads slot n_accept-1) bitwise = sequential: True, worst o vs golden 4.6e-04 / 4.1e-03 / 3.6e-03; replay: verify o bitwise = checkpoint_slots True, committed state bitwise = sequential True.

| layer level T (+16 decode) | prefill ours | prefill HF bf16 | ratio | decode ours | decode HF bf16 | ratio |
|---|---|---|---|---|---|---|
| 16 | 1.5e-01 / 1.2e-03 / 2.9e-03 | 3.5e-01 / 2.8e-03 / 3.4e-03 | 0.85 | 5.0e-01 / 7.7e-03 / 4.6e-03 | 5.0e-01 / 7.7e-03 / 5.4e-03 | 0.84 |
| 512 | 2.6e+00 / 6.2e-03 / 5.1e-03 | 2.3e+00 / 5.3e-03 / 5.2e-03 | 0.98 | 3.2e-01 / 4.0e-03 / 4.5e-03 | 4.5e-01 / 5.5e-03 / 5.6e-03 | 0.79 |
| 2048 | 1.8e+00 / 3.7e-03 / 4.1e-03 | 1.8e+00 / 3.7e-03 / 4.0e-03 | 1.02 | 7.5e-01 / 5.2e-03 / 5.9e-03 | 3.4e-01 / 2.3e-03 / 4.4e-03 | 1.34 |
| 8192 | 2.6e+00 / 4.7e-03 / 3.7e-03 | 2.8e+00 / 5.2e-03 / 3.6e-03 | 1.03 | 2.6e-01 / 2.8e-03 / 3.4e-03 | 2.6e-01 / 2.8e-03 / 3.9e-03 | 0.89 |

### Kernel semantics the plugin must respect (found while testing)

- **Slot 0 is NULL.** The recurrent kernels skip any Seq whose slot index is `<= 0`; the state pool must reserve
  slot 0. For skipped (padded) Seqs `fused_recurrent` / `fused_sigmoid_gating` leave their output rows
  **uninitialized**; `packed_decode` writes zeros.
- **One state store per token.** With `ssm_state_indices`, the kernels store the state after token `t` into column
  `t` of the Seq's index row. A 1-D index (one slot per Seq) is therefore only valid for one token per Seq; more
  tokens write out of bounds (hit as an illegal memory access in the first smoke test). Verify uses `[N, k+1]` indices.
- **`checkpoint_slots` is native:** column `j` receives the state after token `j`; the next step reads column
  `n_accept − 1` (`num_accepted_tokens`). Commit costs nothing, but the live slot rotates among the Seq's k+1 slots,
  so the StateKind must carry `n_accept` into the next step and publish from column `n_accept − 1`.
- **`replay` works with the same kernel:** verify with index row `[S, 0, …, 0]` (only the state after token 0, which
  is always accepted, is stored); commit re-runs the accepted drafts' cached core inputs with row `[0, …, 0, S]`
  (S in column n−1) and `num_accepted_tokens = n`. Measured bitwise equal to sequential decode. Needs the verify
  step's per-layer q/k/v/a/b kept until commit (or recomputed).
- **The chunk kernel has no state indices:** the caller gathers `h0 = pool[slots]` and scatters the final state (vLLM
  does the same). Its `prepare_chunk_indices` calls `.tolist()` (host sync) unless `chunk_indices` /
  `chunk_offsets` are passed — they belong in `plan_bounds`.

## Performance

Kernel level: core GDN only (in-kernel q/k l2norm, gating, recurrence); projections, conv and gated norm excluded.
Prefill: one `chunk_gated_delta_rule` call with host-precomputed chunk indices; eager is what v0 runs (prefill is
EAGER), graph replay shows GPU time without launch overhead; "(autotuned)" = vLLM's original search space tuned at
T=8192 (its keys exclude T). "ktok/s all GDN layers" = tokens / (eager time × GDN layer count).

| model | lens | sm86 eager ms/layer (autotuned) | sm86 graph ms/layer | sm86 Mtok/s/layer eager | sm86 ktok/s all GDN layers eager | sm120 eager ms/layer (autotuned) | sm120 graph ms/layer | sm120 Mtok/s/layer eager | sm120 ktok/s all GDN layers eager |
|---|---|---|---|---|---|---|---|---|---|
| 9b | 64 | 0.580 (0.749) | 0.066 | 0.11 | 5 | 0.458 (0.495) | 0.047 | 0.14 | 6 |
| 9b | 512 | 0.627 (0.718) | 0.171 | 0.82 | 34 | 0.435 (0.495) | 0.074 | 1.18 | 49 |
| 9b | 2048 | 0.819 (0.891) | 0.613 | 2.50 | 104 | 0.475 (0.516) | 0.221 | 4.31 | 179 |
| 9b | 8192 | 2.673 (2.671) | 2.595 | 3.07 | 128 | 1.063 (1.085) | 0.932 | 7.71 | 321 |
| 9b | 32768 | 11.002 (10.579) | 11.048 | 2.98 | 124 | 4.140 (4.158) | 4.092 | 7.92 | 330 |
| 9b | 4x2048 | 2.640 (2.688) | 2.670 | 3.10 | 129 | 1.115 (0.959) | 0.983 | 7.35 | 306 |
| 27b | 64 | 0.621 (0.507) | 0.072 | 0.10 | 2 | 0.445 (0.496) | 0.053 | 0.14 | 3 |
| 27b | 512 | 0.644 (0.534) | 0.245 | 0.80 | 17 | 0.439 (0.520) | 0.106 | 1.17 | 24 |
| 27b | 2048 | 1.099 (1.040) | 0.854 | 1.86 | 39 | 0.548 (0.560) | 0.344 | 3.74 | 78 |
| 27b | 8192 | 3.948 (3.885) | 3.843 | 2.07 | 43 | 1.692 (1.526) | 1.557 | 4.84 | 101 |
| 27b | 32768 | 16.904 (17.188) | 16.620 | 1.94 | 40 | 6.623 (5.933) | 6.547 | 4.95 | 103 |
| 27b | 4x2048 | 3.900 (3.836) | 4.001 | 2.10 | 44 | 1.607 (1.712) | 1.436 | 5.10 | 106 |

| model | kernel | Seqs | tok/Seq | sm86 graph µs/layer | sm86 eager µs/layer | sm86 state GB/s | sm120 graph µs/layer | sm120 eager µs/layer | sm120 state GB/s |
|---|---|---|---|---|---|---|---|---|---|
| 9b | fused_recurrent | 1 | 1 | 10.1 | 93.1 | 414 | 4.8 | 87.1 | 882 |
| 9b | fused_sigmoid_gating | 1 | 1 | 8.0 | 54.7 | 526 | 4.3 | 55.7 | 967 |
| 9b | packed_decode | 1 | 1 | 9.6 | 41.3 | 437 | 5.2 | 39.9 | 808 |
| 9b | fused_recurrent | 4 | 1 | 29.7 | 85.8 | 564 | 11.7 | 85.8 | 1434 |
| 9b | fused_sigmoid_gating | 4 | 1 | 28.1 | 55.0 | 597 | 11.9 | 54.1 | 1404 |
| 9b | packed_decode | 4 | 1 | 25.5 | 40.8 | 659 | 12.1 | 40.3 | 1391 |
| 9b | fused_recurrent | 16 | 1 | 105.9 | 111.6 | 634 | 45.2 | 87.7 | 1486 |
| 9b | fused_sigmoid_gating | 16 | 1 | 103.3 | 107.7 | 649 | 44.8 | 57.8 | 1497 |
| 9b | packed_decode | 16 | 1 | 101.3 | 105.1 | 663 | 46.1 | 47.4 | 1455 |
| 9b | fused_recurrent | 64 | 1 | 404.7 | 424.4 | 663 | 184.7 | 202.0 | 1454 |
| 9b | fused_sigmoid_gating | 64 | 1 | 401.9 | 416.9 | 668 | 183.7 | 196.9 | 1462 |
| 9b | packed_decode | 64 | 1 | 400.1 | 412.2 | 671 | 185.0 | 194.2 | 1451 |
| 27b | fused_recurrent | 1 | 1 | 13.4 | 131.1 | 470 | 6.4 | 88.1 | 982 |
| 27b | fused_sigmoid_gating | 1 | 1 | 12.2 | 87.7 | 515 | 6.5 | 56.1 | 972 |
| 27b | packed_decode | 1 | 1 | 12.2 | 62.3 | 516 | 7.4 | 40.4 | 850 |
| 27b | fused_recurrent | 4 | 1 | 41.7 | 86.3 | 603 | 15.8 | 86.6 | 1593 |
| 27b | fused_sigmoid_gating | 4 | 1 | 41.9 | 54.8 | 601 | 15.9 | 56.3 | 1582 |
| 27b | packed_decode | 4 | 1 | 40.8 | 43.3 | 617 | 16.4 | 40.7 | 1530 |
| 27b | fused_recurrent | 16 | 1 | 155.4 | 163.5 | 648 | 67.7 | 91.0 | 1486 |
| 27b | fused_sigmoid_gating | 16 | 1 | 154.3 | 160.4 | 653 | 68.0 | 72.1 | 1481 |
| 27b | packed_decode | 16 | 1 | 151.6 | 156.8 | 664 | 68.4 | 71.0 | 1472 |
| 27b | fused_recurrent | 64 | 1 | 602.5 | 629.2 | 668 | 276.6 | 301.0 | 1456 |
| 27b | fused_sigmoid_gating | 64 | 1 | 602.1 | 622.1 | 669 | 276.8 | 294.4 | 1455 |
| 27b | packed_decode | 64 | 1 | 598.7 | 615.4 | 673 | 275.5 | 291.2 | 1462 |

| model | kernel | Seqs | tok/Seq | sm86 graph µs/layer | sm86 eager µs/layer | sm86 state GB/s | sm120 graph µs/layer | sm120 eager µs/layer | sm120 state GB/s |
|---|---|---|---|---|---|---|---|---|---|
| 9b | fused_sigmoid_gating | 1 | 4 | 21.4 | 60.1 | 196 | 12.6 | 56.4 | 334 |
| 9b | fused_sigmoid_gating | 4 | 4 | 67.9 | 72.1 | 247 | 26.8 | 56.3 | 627 |
| 9b | fused_sigmoid_gating | 16 | 4 | 257.3 | 280.1 | 261 | 117.2 | 130.9 | 573 |
| 27b | fused_sigmoid_gating | 1 | 4 | 31.1 | 55.6 | 202 | 14.4 | 57.7 | 435 |
| 27b | fused_sigmoid_gating | 4 | 4 | 101.8 | 107.3 | 247 | 40.5 | 59.2 | 622 |
| 27b | fused_sigmoid_gating | 16 | 4 | 384.9 | 403.3 | 262 | 178.0 | 198.2 | 566 |

Decode / verify: per-layer time, CUDA graph of one call per state pool, pools rotated so the states do not stay
resident in L2 (≥ 512 MiB or one pool per GDN layer). "state GB/s" = read + write of one fp32 state per Seq.
Caveat: for 9B at 1 Seq the 24 pools (96 MiB) fit sm_120's 128 MiB L2, so that cell is optimistic.

Prefill workspace (temporaries + outputs allocated inside the call, measured): ≈ 59 KB/token at 9B shapes and
≈ 84 KB/token at 27B shapes per layer call (e.g. 692 MiB for T=8192 at 27B).

Reading: decode is bandwidth-bound from 4–16 Seqs on (87 % of the A5000's 768 GB/s, ≈ 1.5 TB/s on sm_120); the three
decode kernels are equal under graphs. Eager prefill has a **host floor of ≈ 0.44–0.65 ms per GDN layer** (≈ 10
Triton launches through Python), i.e. 11–15 ms per prefill step for 9B and 21–31 ms for 27B regardless of length,
while the GPU work at T=64 is 0.05–0.07 ms (eager times at T ≤ 512 are host-bound and noisy). By GPU (graph) time the
single pinned table is 0.96–1.07× the autotuned time, except sm_120 at 27B shapes (up to 1.13×; both from
`results/perf_*.json` `graph_ms_per_layer`).

## CUDA graphs and determinism

Decode and verify graphs: static q/k/v/a/b, slot indices, accepted counts and the pool in device tensors; each replay
rewrites inputs, slot assignment (including NULL-padded Seqs) and accepted counts, and is compared to eager on a pool
copy. Chunked prefill also captures once its chunk indices are precomputed (informational; v0 prefill is eager).
Byte identity: SHA-256 of chunk (T=8192 and 4-Seq varlen), decode and verify outputs + states.

| check | sm86 | sm120 |
|---|---|---|
| graph_decode_fused_recurrent: 5 replays, new inputs/slots/accepted counts, = eager bitwise | True | True |
| graph_decode_fused_sigmoid: 5 replays, new inputs/slots/accepted counts, = eager bitwise | True | True |
| graph_verify_k3_fused_sigmoid: 5 replays, new inputs/slots/accepted counts, = eager bitwise | True | True |
| graph_chunk_prefill (precomputed chunk indices) = eager bitwise | True | True |
| 10 runs in-process identical (all 5 outputs) | True | True |
| 2 fresh processes identical | True | True |
| autotuned process = pinned | True | True |
| sm_86 hashes = sm_120 hashes | False | |

## GGUF tiled value-head order

From transformers' GGUF loader (`src/transformers/integrations/gguf/gguf_conversion_mapping.py`, `_qwen35`,
huggingface/transformers `14e738b5d0cc`): llama.cpp stores value heads **tiled** (`v0k0 v0k1 v1k0 v1k1`, value head
j belongs to key head `j % Hg`) instead of grouped (`v0k0 v1k0 v0k1 v1k1`, key head `j // (H/Hg)`); the loader
permutes back with `TiledToGroupedRows` on **in_proj_qkv (value block rows only, offset 2·key_dim), conv1d (same
rows), in_proj_z (rows), in_proj_a, in_proj_b, A_log, dt_bias (per head)** and `TiledToGroupedInputs` on **out_proj
input columns** — the only permute that crosses quantization blocks, so transformers dequantizes `out_proj` (or
permutes its input when packed). Unrelated to head order, the same mapping also undoes `w+1` norms (except
`ssm_norm`), `ssm_a = −exp(A_log)` and the squeezed conv1d.

Test (`tiled_order.py`): HF weights rewritten into tiled order with the inverse permutation; the mixer runs with all
tensors left tiled (no repacking — `out_proj` consumes the tiled columns directly), the kernels with
`v_head_tiled=True` (q/k `repeat` semantics, `i_hk = i_h % Hg`), prefill 1000 tokens + 8 decode steps; the control
runs the tiled weights through the grouped kernels.

| arch | model | knob | core out bitwise | state bitwise | out_proj input bitwise | layer output (max_abs / max_rel / fro_rel) |
|---|---|---|---|---|---|---|
| sm86 | 4b | v_head_tiled=True | True | True | True | 2.5e-01 / 5.0e-03 / 2.7e-03 |
| sm86 | 4b | v_head_tiled=False | False | False | False | 1.2e+01 / 2.5e-01 / 7.8e-01 |
| sm86 | 27b | v_head_tiled=True | True | True | True | 5.0e-01 / 1.3e-03 / 2.7e-04 |
| sm86 | 27b | v_head_tiled=False | False | False | False | 1.6e+02 / 4.0e-01 / 5.7e-01 |
| sm120 | 4b | v_head_tiled=True | True | True | True | 1.2e-01 / 2.6e-03 / 2.5e-03 |
| sm120 | 4b | v_head_tiled=False | False | False | False | 1.7e+01 / 3.5e-01 / 8.2e-01 |
| sm120 | 27b | v_head_tiled=True | True | True | True | 2.0e+00 / 3.0e-03 / 4.3e-03 |
| sm120 | 27b | v_head_tiled=False | False | False | False | 2.2e+02 / 3.3e-01 / 5.7e-01 |

sm86: fused_recurrent / fused_sigmoid_gating / packed_decode on tiled inputs + knob, 4 decode steps, outputs and state bitwise = grouped: {'4b': True, '27b': True}

sm120: fused_recurrent / fused_sigmoid_gating / packed_decode on tiled inputs + knob, 4 decode steps, outputs and state bitwise = grouped: {'4b': True, '27b': True}

So the knob makes the core outputs, the recurrent state and the `out_proj` input **bitwise identical** (up to the
head permutation) to the grouped layout on both archs, for chunk prefill and all three recurrent kernels; the only
layer-output difference is `out_proj`'s GEMM summing the same products in a different column order (bf16 rounding).
The state pool simply lives in tiled head order. Cost: one constexpr per kernel, no runtime overhead.

## Decision

**Adopt** the vendored subset as the `mixers/gdn` kernel plugin for v0 on sm_86 and sm_120 (exit criteria met: within
the spike budget, no NaN/Inf at T = 1..8192, run-to-run byte identical in and across processes, no `vllm` imports):

- prefill: `chunk_gated_delta_rule` with planner-supplied `chunk_indices`/`chunk_offsets`, state gathered/scattered
  by the mixer; decode and verify: `fused_sigmoid_gating_delta_rule_update` (gating fused, varlen, 2-D slot indices,
  `num_accepted_tokens`); `packed_decode` is equal under graphs and has less eager overhead — keep as an alternative.
- state: fp32 `[slots, H, V, K]` Slot pool, slot 0 reserved; `Rewindable` `checkpoint_slots` uses the kernels'
  native per-token slot columns, `replay` uses the index-row trick above — no kernel change for either mode.
- configs: one pinned table (`utils.PINNED`) as the deterministic default; per-fingerprint-class tables are an
  optional ≤ 13 % on sm_120 at 27B shapes.
- GGUF: `v_head_order="tiled"` maps to the kernels' `v_head_tiled` — no weight repacking, `out_proj` stays packed.

## Open issues

1. **Rule 1 (our buffers):** the wrappers allocate outputs, final state and all prefill temporaries internally
   (≈ 59 / 84 KB per token per layer call at 9B / 27B). The plugin needs `out=` buffers and a declared workspace,
   and the prefill token budget per step must account for it.
2. **Eager prefill launch floor** (0.44–0.65 ms per GDN layer, ≈ 11–31 ms per step): matters for short agent turns.
   Options: bucketed graphs for short prefills (capture works with precomputed indices), PIECEWISE (v1), or a
   thinner launch path (call the compiled kernels directly instead of through `heuristics`/`autotune` wrappers).
3. **Chunk kernel state indices:** a small kernel change (index `h0`/`ht` by slot like the recurrent kernels) would
   remove the gather/scatter copies per prefill.
4. **Not covered by S3:** the causal conv1d kernels (vLLM `mamba/ops/causal_conv1d.py`, 1288 lines, imports
   `vllm.platforms` and `NULL_BLOCK_ID`), the gated RMSNorm, and vLLM's fused post-conv prep — the spike used HF's
   torch code for them. The conv state also needs `checkpoint_slots`/`replay` handling for spec verify.
5. **GGUF:** `ssm_a` holds `−exp(A_log)` while the fused-gating kernels take `A_log` — either load `log(−ssm_a)` or
   add an `A` input variant (plan §6 already computes `−exp(A_log)` in `finalize_weights`). The tiled order was
   verified against transformers' mapping, not a real Qwen3.5/3.8 GGUF file (none cached) — check in M0.3b with the
   transformers-GGUF golden as planned.
6. **Padded rows** in the output of `fused_recurrent`/`fused_sigmoid_gating` are uninitialized; zero or mask them
   before anything that snapshots whole buffers.
7. **Trim:** the subset still carries dead paths (cumsum vector kernel, 16/32 solve_tril variants, TMA and KDA
   branches, non-inplace per-token state store); ≈ 2k lines is reachable when it becomes the plugin.
8. Byte identity holds per fingerprint class only (sm_86 and sm_120 hashes differ); budgets here are spike-local
   until S5. The chunk kernel keeps per-chunk states in bf16 for its output kernel — its state error is ≈ 1.3× HF
   bf16's, revisit only if an fp32-golden budget gets tighter.
