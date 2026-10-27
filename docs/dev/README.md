# Development records

One record per milestone: its design, the review rounds that challenged it, the decisions taken, and the build record
step by step, with links to the measurements in `evidence/` and the experiments in `spikes/`. They explain why the
code is the way it is; code comments cite them by section (e.g. "v1-offload.md D8"). They are history, not required
reading: start from [the README](../../README.md), [the design](../design.md) and [the roadmap](../plan.md).

| Record | Milestone |
|---|---|
| [m01-gguf.md](m01-gguf.md) | M0.1: GGUF quantized models |
| [m02-moe.md](m02-moe.md) | M0.2: mixture of experts on a 16 GB card |
| [m03-qwen38.md](m03-qwen38.md) | M0.3: Qwen3.8-27B, heterogeneous state, prefix reuse, quantized KV |
| [m03-in-step-snapshot.md](m03-in-step-snapshot.md) | M0.3b: in-step recurrent checkpoints |
| [m04-spec.md](m04-spec.md) | M0.4: speculative decoding (MTP, DFlash2, n-gram) |
| [m05-grammar.md](m05-grammar.md) | M0.5: structured output and tool calls |
| [m06-lowp.md](m06-lowp.md) | M0.6: FP8, NVFP4 and MXFP4 checkpoints |
| [m07-perf.md](m07-perf.md) | M0.7: performance |
| [release.md](release.md) | v0 release: packaging, locks, benchmarks, ABI freeze; the release and test process (§22) |
| [agent-test-pi-rtx5090.md](agent-test-pi-rtx5090.md) | A real coding agent (pi) served by Holon on an RTX 5090 |
| [v1-offload.md](v1-offload.md) | v1 X: expert offload, MoE models larger than VRAM |
| [v1-kv-tiers.md](v1-kv-tiers.md) | v1 K: KV and checkpoint tiers under the prefix index |
| [v1-moe-prefill.md](v1-moe-prefill.md) | v1: MoE prefill speed |
| [v1-flash-next.md](v1-flash-next.md) | v1 FN: Qwen3.8-Flash-Next and its offload, compared with Strata |

Paths under `local/` (e.g. `local/fn/`, `local/models/`, `local/tmp/`) name the authors' untracked working files:
downloads of public models, clones of the reference engines at their public commits, files built from those with the
engines' own public tools, and scratch output. They are not part of the repository and hold no other source of
information.
