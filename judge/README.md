# holon_judge — the judge harness (M0)

The judge runs outside the engine and drives it only as a client (design §10.3). M0 covers:

| module | tier | what |
|---|---|---|
| `golden` | L2 | HF recorder: teacher-forced top-20 logprobs, the target token's logprob and the tail mass, one forward per sequence |
| `compare` | L2 | candidate vs golden: KL p50/p99, top-1 agreement, ΔPPL, per category; pass/fail against the floor |
| `engine` | L2 | the candidate side: scores the prompt set through a Holon engine (`prompt_logprobs=20`) |
| `l0` | L0 | `budget(dtype, reduction_len)` and `check(candidate, reference, dtype, reduction_len)` for op oracles |
| `snapshot` | snapshot | records a fixed trace's token streams and a sha256 per delta (request, sequence, step); compares runs for byte identity; `trace(prompts)` is the snapshot trace of design §7.10 (long context, a prefix hit, the decode batch shrinking through every graph bucket up to 64, seeded n > 1, logprobs, prompt logprobs) |
| `recording` | | prompt sets and the recording format (safetensors + JSON provenance) |

The prompt set is `prompts/v1.jsonl` (24 sequences, 32,807 tokens; chat, code, recall, agent; built in spike S5), frozen
by `prompts/v1.jsonl.sha256`, which the loader checks. Goldens are written by the tool, never by hand.

## Commands

```
# golden: fp32 on CPU, thread count pinned (bitwise reproducible only at the same count)
python -m holon_judge golden --model Qwen/Qwen3-0.6B --prompts judge/prompts/v1.jsonl \
    --out judge/goldens/qwen3-0.6b/v1-fp32.safetensors --threads 44
# floor: the same recorder in the model's activation dtype, on the GPU class the engine targets
CUDA_VISIBLE_DEVICES=<gpu> python -m holon_judge golden --model Qwen/Qwen3-0.6B --prompts judge/prompts/v1.jsonl \
    --out judge/goldens/qwen3-0.6b/v1-bfloat16-floor.safetensors --dtype bfloat16 --device cuda --threads 2
# floor report and budgets (machine-readable)
python -m holon_judge compare --golden judge/goldens/qwen3-0.6b/v1-fp32.safetensors \
    --candidate judge/goldens/qwen3-0.6b/v1-bfloat16-floor.safetensors \
    --floor judge/goldens/qwen3-0.6b/v1-bfloat16-floor.safetensors --json judge/goldens/qwen3-0.6b/v1-bfloat16-floor.json
# L2 check of the engine (exit status 1 on FAIL)
CUDA_VISIBLE_DEVICES=<gpu> python -m holon_judge l2 --model Qwen/Qwen3-0.6B --prompts judge/prompts/v1.jsonl \
    --golden judge/goldens/qwen3-0.6b/v1-fp32.safetensors \
    --floor judge/goldens/qwen3-0.6b/v1-bfloat16-floor.safetensors [--save cand.safetensors] [--json result.json]
# any two recordings (e.g. a saved candidate)
python -m holon_judge compare --golden <golden> --candidate <recording> [--floor <floor>] [--json <file>]
python -m holon_judge snapshot-diff a.json b.json
```

Qwen3-8B uses the same commands with `--model Qwen/Qwen3-8B` and `judge/goldens/qwen3-8b/`. Its fp32 golden needs
~49 GB of host RAM.

## Goldens (prompt set v1)

| model | file | recorded on | wall | size |
|---|---|---|---|---|
| Qwen3-0.6B @ c1899de2 | `goldens/qwen3-0.6b/v1-fp32.safetensors` | CPU fp32, 44 threads (<cpu>) | 53 s | 5.5 MB |
| | `goldens/qwen3-0.6b/v1-bfloat16-floor.safetensors` | HF bf16 sdpa, RTX A5000 (sm_86) | 3.4 s | 5.5 MB |
| Qwen3-8B @ b968826d | `goldens/qwen3-8b/v1-fp32.safetensors` | CPU fp32, 44 threads (<cpu>), 49 GB peak RSS | 282 s | 5.5 MB |
| | `goldens/qwen3-8b/v1-bfloat16-floor.safetensors` | HF bf16 sdpa, RTX A5000 (sm_86) | 9.8 s | 5.5 MB |

Floors and budgets (`v1-bfloat16-floor.json`; budgets: KL ≤ 2× floor, top-1 ≥ floor − 0.25 points,
|ΔPPL| ≤ |floor| + 0.25%):

| model | top-1 | KL p50 | KL p99 | ΔPPL ± SE | budget top-1 | budget KL p50 | budget KL p99 | budget \|ΔPPL\| |
|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | 98.447% | 2.30e-04 | 4.47e-03 | +0.079% ± 0.022% | ≥ 98.197% | ≤ 4.61e-04 | ≤ 8.94e-03 | ≤ 0.329% |
| Qwen3-8B | 98.929% | 4.48e-05 | 5.76e-03 | +0.090% ± 0.030% | ≥ 98.679% | ≤ 8.95e-05 | ≤ 1.15e-02 | ≤ 0.340% |

These reproduce spike S5. Re-scored with this comparator, S5's HF variants keep their verdicts: eager attention,
chunked prefill (256), fp16 and the rms_norm eps bug pass; the rope-theta bug fails (8B: top-1 98.57%, KL p99
1.75e-2).

Differences from the S5 prototype: the log-softmax is normalized in fp64 before it is stored in fp32 (an fp32
log-softmax over the 152k vocabulary on CPU is mis-normalized by up to 1.4e-4 nats, the size of the 8B floor's KL p50),
and the tail mass is stored exactly instead of being recomputed as 1 − Σ top-20. The golden's top-20 ids are bitwise
identical to S5's at the same thread count.

### First Holon engine run (E3, commit e85cef9; eager, bf16, RTX A5000 sm_86)

`python -m holon_judge l2 --model <model> ... --json judge/results/l2-e85cef9-<model>.json`, both PASS (raw results in
`judge/results/`):

| model | top-1 | top-1 conf | KL p50 | KL p99 (× floor) | ΔPPL ± SE | missing | wall |
|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | 98.539% | 99.800% | 1.92e-04 | 3.60e-03 (0.81×) | +0.021% ± 0.027% | 1.43% | 8.3 s |
| Qwen3-8B | 99.021% | 99.831% | 4.07e-05 | 5.15e-03 (0.89×) | +0.130% ± 0.037% | 1.33% | 18.8 s |

The engine is closer to fp32 than HF bf16 on every budgeted metric, so the S5 δ needs no widening for this path
(design §10.3: "δ is re-checked on the first real-engine run"). Agent transcripts carry the largest ΔPPL (8B
+0.346%; floor +0.228%), as in S5.

## GGUF files (M0.1)

`gguf` builds the dense model a GGUF file stands for: `Qwen3ForCausalLM` from the HF config the file was made from,
filled with the file's tensors dequantized by gguf-py through the judge's own name map (never the engine's loader or
weight map). Goldens and floors record it with `--config`; meta = the file's sha256, the gguf-py version and the
name-map hash.

```
# per-tensor gate: NMSE(dequantized, HF bf16) <= 0.5 for every tensor
python -c "from holon_judge import gguf; print(max(gguf.gate('<file>.gguf', '<hf dir>').values()))"
# golden (CPU fp32) and Tier A floor (HF bf16 of the dequantized weights, A5000)
python -m holon_judge golden --model <file>.gguf --config Qwen/Qwen3-8B --prompts judge/prompts/v1.jsonl \
    --out judge/goldens/qwen3-8b-q4_k_m/v1-fp32.safetensors --threads 44
CUDA_VISIBLE_DEVICES=<gpu> python -m holon_judge golden --model <file>.gguf --config Qwen/Qwen3-8B \
    --prompts judge/prompts/v1.jsonl --out judge/goldens/qwen3-8b-q4_k_m/v1-bfloat16-floor.safetensors \
    --dtype bfloat16 --device cuda --threads 2
# Tier A: the engine on the dequantized weights (the test-only `gguf_dense` source), and its planted bugs
CUDA_VISIBLE_DEVICES=<gpu> python -m holon_judge l2 --model <file>.gguf --source gguf_dense \
    --prompts judge/prompts/v1.jsonl --golden <golden> --floor <floor> [--plant gguf-qk-permuted]
```

## Engine contract (L2 candidate)

`SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=20)` per sequence; the `TokenDelta.prompt_logprobs`
(`holon.api.Logprobs`) must cover prompt positions 1 … n-1 with the fp32 log-softmax of the engine's own logits
(they may arrive over several deltas). The normalization matters: a log-softmax off by a uniform 1e-5 nats raises
the 8B KL p50 by 20%, so it must be accurate to ≲ 1e-6 nats (GPU fp32 reductions are; a CPU fp32 log-softmax over a
152k vocabulary is not). `Logprobs` has no tail mass, so the judge derives it from the top-20 (accurate to ~1e-7).

## L0 budgets

`budget(dtype, K)` models an output reduced over K terms with fp32 accumulation and stored in `dtype` (u = unit
roundoff, s = u + u32·√K): per element |cand − ref| ≤ 3u·|ref| + 6s·rms(ref), and mean |cand − ref| ≤ s·rms(ref).
On matmul (K = 1 … 14336), softmax, silu·mul and rmsnorm, on CPU and sm_86, legitimate bf16/fp16/fp32 results use
≤ 0.55 of the budget; a 1% scale error in bf16 uses 2.0 and 0.1% in fp16 1.6. See the `l0` docstring.
