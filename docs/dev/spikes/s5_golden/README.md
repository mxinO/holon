# S5 — golden recording and noise floor

## Question

Can the judge's L2 tier (design §10.3: teacher-forced logits vs an HF fp32 recording on frozen prompts) be built
cheaply for Qwen3-8B, and what are the calibrated budgets? Specifically: what does the fp32 golden cost in time and
storage? Is a top-k recording enough to compute KL? How large is the HF bf16-vs-fp32 noise floor? Which δ separates
legitimate bf16 implementations from real bugs?

## Setup

- fp32 recording uses a fixed CPU thread count (recorded in each golden), each GPU process 2.
  GPUs: two RTX A5000 (sm_86).
- torch 2.11.0+cu130, transformers 5.6.0.dev0, `.venv`.
- Models: `Qwen/Qwen3-8B@b968826d`, `Qwen/Qwen3-0.6B@c1899de2`. Weight blob sha256s are in each recording's
  metadata.
- Golden: HF `AutoModelForCausalLM`, `dtype=float32`, `device_map="cpu"`, sdpa, one forward per sequence.
- Candidates, all HF on GPU and recorded with the same recorder:

| run | what | purpose |
|---|---|---|
| `bf16` | bf16, sdpa, one forward per sequence | **the floor** |
| `bf16-r2` | same as `bf16`, run again | bf16 run-to-run variation |
| `bf16-eager` | bf16, eager attention | legitimate kernel change |
| `bf16-chunk256` | bf16, chunked prefill through the KV cache (256-token chunks) | legitimate path change (engine-like) |
| `fp16` | fp16, sdpa | completeness |
| `bug-eps` | bf16 with `rms_norm_eps` 1e-6 → 1e-5 | injected subtle bug |
| `bug-rope` | bf16 with `rope_theta` 1e6 → 9e5 | injected subtle bug |
| `fp32-bug-*` | the same bugs in fp32 on CPU (0.6B only) | size of each bug without bf16 noise |

Reproduce:

```
CUDA_VISIBLE_DEVICES=0 python build_prompts.py --out prompts/v1.jsonl   # once; already frozen
./run_all.sh Qwen/Qwen3-8B qwen3-8b                                      # golden + candidates + noise_floor.py
./run_all.sh Qwen/Qwen3-0.6B qwen3-0.6b
python golden.py --model Qwen/Qwen3-0.6B --threads <n> --override rms_norm_eps=1e-5 \
       --out out/qwen3-0.6b/fp32-bug-eps.safetensors                     # fp32 bugs (likewise for rope)
python noise_floor.py --dir out/qwen3-8b --out results/noise_floor_qwen3-8b.json
python tables.py results/noise_floor_qwen3-8b.json > results/tables_qwen3-8b.md
```

Files: `build_prompts.py` builds the prompt set. `golden.py` is the recorder and storage format. `compare.py` is the
comparator and budgets. `noise_floor.py` produces the tables and validates the KL approximation. `tables.py` renders
the markdown. Raw numbers are in `results/noise_floor_*.json` and `results/fp32_reproducibility_*.json`. Rendered
tables are in `results/tables_*.md`.

## Prompt set v1

`prompts/v1.jsonl` has 24 sequences and 32,807 tokens (32,783 scored positions). Its sha256 is
`5157b507f2553584a5d44ca1ef902043e7a1f5ad5a3deecd3038b6eb5fadb9ae` (also in `prompts/v1.jsonl.sha256`, and every
recording's metadata stores it). Each sequence is 607–2048 tokens, rendered with the Qwen3 chat template.

| category | n | content |
|---|---|---|
| chat | 6 | explanation, probability math (thinking), Chinese, story, DB advice (thinking), multi-turn |
| code | 6 | LRU cache, debug a buggy function, CUDA softmax kernel, Rust parser (thinking), 2 LiveCodeBench problems |
| recall | 6 | ~1.4k-token wikitext-103 articles with an inserted needle fact, needle question, or summary/years question |
| agent | 6 | 5 tool schemas in the system prompt, hand-written tool calls and tool responses (incl. errors and retries), then the model's next turn |

The final assistant turn of every sequence was generated once, greedily, by Qwen3-8B bf16. Each sequence therefore
mixes off-policy context (14.5k positions: user text, documents, tool output) with on-policy model text (18.3k
positions). Thinking is on for 6 sequences. Each row stores
`{id, category, think, n_tokens, gen_start, ids, text}`. The token ids are the frozen artifact; `text` is only there
to make the file readable.

## Storage format

There is one safetensors file per (model, run). All positions of all sequences are concatenated, and position *t*
predicts `ids[t+1]`.

| tensor | dtype | shape | |
|---|---|---|---|
| `topk_ids` | int32 | [N, 20] | top-20 token ids, logprob-descending |
| `topk_lp` | fp32 | [N, 20] | fp32 log-softmax of the model's logits |
| `target_lp` | fp32 | [N] | logprob of the actual next token |
| `offsets` | int64 | [S+1] | rows of sequence *s* |
| `full_pos`, `full_lp` | int64, fp32 | [M], [M, V] | optional full-vocab rows (every 32nd position); used only to validate the KL approximation |

The safetensors header key `meta` holds JSON with: model, revision, weight blob sha256s, dtype, device, attention
implementation, overrides, chunking, k, torch and transformers versions, thread count, hardware, prompt-set name and
sha256, wall time, and N. The comparator refuses to compare recordings whose prompt hash or offsets differ.

## Timing and storage

| | Qwen3-8B | Qwen3-0.6B |
|---|---|---|
| fp32 CPU golden, wall (44 threads) | 458 s (host busy with other GPU runs) / 248 s (rerun) | 48 s / 38 s |
| fp32 CPU peak RSS | 49.0 GB | — |
| bf16 GPU candidate, wall (1× A5000) | 8.6 s (eager 11.6 s, chunk256 10.6 s) | 2–4 s |
| golden file (top-20 only) | **5.38 MB** (164 B/position; 3.6 MB gzipped) | 5.38 MB |
| with the full-vocab validation subset | 633 MB | 633 MB |
| total `out/` for this spike | 9.6 GB, gitignored (mostly validation subsets) | |

The fp32 CPU golden is bitwise reproducible run to run when the thread count is the same (verified for 8B and 0.6B).
It is **not** bitwise identical across thread counts: 24 vs 44 threads on 0.6B gives max |Δlogprob| 6.9e-5, KL p99
1.6e-9, top-1 agreement 100%. The thread count is recorded in the metadata and must be pinned to re-record
byte-identically.

## Noise floor (raw)

### Qwen3-8B: each run vs the fp32 golden (32,783 positions)

`top-1 (gap≥0.1)` is top-1 agreement on positions where the golden's top-2 logprob gap is ≥ 0.1 nat (97.5% of
positions). `imputed` is the share of golden top-20 ids that are missing from the candidate's top-20. ΔPPL is
relative, with a standard error clustered by sequence.

| run | top-1 | top-1 (gap≥0.1) | KL mean | KL p50 | KL p99 | KL max | ΔPPL ± SE | imputed |
|---|---|---|---|---|---|---|---|---|
| **bf16 (floor)** | **98.929%** | **99.794%** | 6.27e-04 | **4.51e-05** | **5.76e-03** | 1.091 | **+0.090% ± 0.030%** | 1.40% |
| bf16-r2 | 98.929% | 99.794% | 6.27e-04 | 4.51e-05 | 5.76e-03 | 1.091 | +0.090% ± 0.030% | 1.40% |
| bf16-eager | 98.838% | 99.706% | 9.57e-04 | 5.79e-05 | 9.11e-03 | 3.208 | +0.099% ± 0.044% | 1.66% |
| bf16-chunk256 | 98.899% | 99.759% | 5.85e-04 | 4.52e-05 | 6.01e-03 | 0.340 | +0.134% ± 0.031% | 1.41% |
| fp16 | 99.878% | 100.000% | 8.77e-06 | 1.16e-06 | 9.54e-05 | 0.002 | +0.003% ± 0.003% | 0.19% |
| bug-eps | 98.954% | 99.784% | 5.84e-04 | 4.54e-05 | 6.01e-03 | 0.422 | +0.161% ± 0.041% | 1.40% |
| bug-rope | 98.572% | 99.453% | 1.66e-03 | 8.19e-05 | 1.75e-02 | 2.421 | +0.020% ± 0.053% | 2.21% |

Sequence-bootstrap SE of the floor (sampling over prompts, not a paired comparison): top-1 0.09 pp, KL p50 2.5e-5,
KL p99 6.6e-4, ΔNLL 3.2e-4.

bf16 vs bf16:

| pair | top-1 | top-1 (gap≥0.1) | KL p50 | KL p99 | ΔPPL | bitwise |
|---|---|---|---|---|---|---|
| bf16 vs bf16-r2 | 100% | 100% | 0 | 0 | 0 | **identical** |
| bf16 vs bf16-eager | 98.948% | 99.443% | 3.11e-05 | 1.01e-02 | +0.009% | no |
| bf16 vs bf16-chunk256 | 99.012% | 99.465% | 2.10e-05 | 7.77e-03 | +0.043% | no |

Breakdown of fp32 vs bf16:

| subset | positions | top-1 | top-1 (gap≥0.1) | KL p50 | KL p99 | ΔPPL ± SE |
|---|---|---|---|---|---|---|
| agent | 6943 | 99.237% | 99.782% | 1.35e-07 | 9.88e-03 | +0.228% ± 0.082% |
| chat | 8061 | 99.107% | 99.797% | 3.90e-05 | 4.58e-03 | +0.057% ± 0.031% |
| code | 8930 | 99.015% | 99.714% | 2.91e-05 | 6.70e-03 | +0.026% ± 0.070% |
| recall | 8849 | 98.441% | 99.882% | 2.46e-04 | 3.54e-03 | +0.077% ± 0.014% |
| context (off-policy) | 14515 | 98.588% | 99.786% | 1.24e-04 | 6.56e-03 | +0.160% ± 0.062% |
| generated (on-policy) | 18268 | 99.201% | 99.799% | 1.35e-05 | 5.49e-03 | +0.035% ± 0.015% |

### Qwen3-0.6B: each run vs the fp32 golden

| run | top-1 | top-1 (gap≥0.1) | KL mean | KL p50 | KL p99 | KL max | ΔPPL ± SE |
|---|---|---|---|---|---|---|---|
| **bf16 (floor)** | **98.447%** | **99.721%** | 5.68e-04 | **2.31e-04** | **4.47e-03** | 0.191 | **+0.081% ± 0.023%** |
| bf16-eager | 98.371% | 99.622% | 8.73e-04 | 3.16e-04 | 7.63e-03 | 0.533 | +0.083% ± 0.030% |
| bf16-chunk256 | 98.508% | 99.759% | 5.73e-04 | 2.29e-04 | 4.33e-03 | 0.051 | +0.113% ± 0.028% |
| fp16 | 99.860% | 100.000% | 9.38e-06 | 3.88e-06 | 7.25e-05 | 0.001 | +0.005% ± 0.003% |
| bug-eps | 98.511% | 99.768% | 6.35e-04 | 2.29e-04 | 4.72e-03 | 1.136 | +0.094% ± 0.028% |
| bug-rope | 98.081% | 99.423% | 1.35e-03 | 3.84e-04 | 1.29e-02 | 1.574 | −0.076% ± 0.049% |
| fp32-bug-eps | 99.851% | 99.975% | 5.71e-05 | 2.58e-06 | 2.68e-04 | 0.312 | +0.012% ± 0.010% |
| fp32-bug-rope | 98.877% | 99.740% | 7.83e-04 | 9.50e-05 | 9.04e-03 | 1.602 | −0.156% ± 0.043% |

bf16 vs bf16-r2 is bitwise identical here too. Full tables, including the fp16 breakdown and bootstrap SEs, are in
`results/tables_*.md`.

### KL from top-k: approximation vs exact

The KL is computed over k+1 buckets: the golden's top-k ids, plus one tail bucket that holds the rest of each
distribution's mass. By the log-sum inequality this is a lower bound on the full-vocab KL. The gap equals the KL
inside the tail, weighted by the golden's tail mass. To validate the approximation, both runs stored full-vocab
logprobs at every 32nd position (1009 positions). Sequence starts are excluded because the stride over-samples them:
the position after a lone `<|im_start|>` is near-uniform, with 99% of its mass outside the top-100.

| model / run | k | exact p50 | approx p50 | exact p99 | approx p99 | median ratio | golden tail mass p50 / p99 |
|---|---|---|---|---|---|---|---|
| 8B bf16 | 5 | 6.70e-05 | 5.24e-05 | 5.17e-03 | 4.88e-03 | 0.96 | 8.6e-05 / 0.51 |
| 8B bf16 | **20** | 6.70e-05 | **6.34e-05** | 5.17e-03 | **5.16e-03** | **1.00** | 1.2e-06 / 0.22 |
| 8B bf16 | 100 | 6.70e-05 | 6.43e-05 | 5.17e-03 | 5.16e-03 | 1.00 | 7.0e-08 / 0.09 |
| 8B fp16 | 20 | 1.74e-06 | 1.38e-06 | 1.25e-04 | 1.24e-04 | 0.77 | |
| 0.6B bf16 | 20 | 3.31e-04 | 2.74e-04 | 4.81e-03 | 4.79e-03 | 0.94 | 1.9e-03 / 0.44 |
| 0.6B fp16 | 20 | 1.44e-05 | 4.35e-06 | 1.06e-04 | 6.86e-05 | 0.33 | |

At k=20 the approximation captures 95–100% of the bf16-scale KL, with p99 within 1%. Going to k=100 gains almost
nothing. It under-reads only when the discrepancy is spread thinly over the tail (fp16, KL ~1e-6), far below any
budget. Because the floor and the candidate are measured with the same estimator, the pass rule stays consistent.
**k = 20 is sufficient.**

## Findings

1. **HF bf16 is run-to-run bitwise identical** for both models (same GPU, same shapes). For HF, the floor is a single
   number, not a distribution.
2. **The floor is dominated by near-ties.** Restricted to confident positions (golden gap ≥ 0.1 nat), bf16 agrees
   99.79% (8B); about 80% of top-1 flips are near-ties. The rope bug raises confident flips from 66 to 175 (2.7×),
   while eager attention raises them to 94 (1.4×). That makes confident agreement a sharper discriminator than raw
   top-1.
3. **KL p99 is the most discriminative metric.** Legitimate variants reach 1.0–1.6× the floor (chunked 1.04×,
   eager 1.58×). The rope bug reaches 3.0× (8B) / 2.9× (0.6B). **ΔPPL does not discriminate**: the rope bug moves
   PPL less than the legitimate chunked path does. KL max is heavy-tailed (single context positions at 1–5 nats for
   legitimate variants) and unusable as a budget.
4. **L2 resolution is about the bf16 floor itself.** The eps bug has KL p99 2.7e-4 in fp32, 17× below the bf16 floor,
   so it is invisible in bf16 (verdict PASS). Errors of that size are L0's job: the op oracle with the config's
   eps. The same rope bug in an otherwise perfect fp32 engine fails only marginally (KL p99 9.04e-3 vs a budget of
   8.94e-3).
5. **Agent transcripts carry the heaviest tail.** They have the largest KL p99 (9.9e-3) and ΔPPL (+0.23%), from tool
   JSON and schema tokens, while their KL p50 is about 0 (deterministic tokens). Off-policy context positions are
   noisier than on-policy ones. Keep all four categories.
6. fp16 is about 60× closer to fp32 than bf16 (KL p99 9.5e-5) and shows no overflow on Qwen3-8B.

## Proposed L2 budgets (bf16 engines)

The floor is HF bf16 (sdpa, one forward) vs the HF fp32 CPU golden, on prompt set v1, recorded once per model.
`compare.proposed_budgets(floor)`:

| metric | rule | δ | Qwen3-8B budget | Qwen3-0.6B budget |
|---|---|---|---|---|
| top-1 agreement | ≥ floor − δ | 0.25 pp (≈ 2.5 bootstrap SE) | ≥ 98.679% | ≥ 98.197% |
| KL p50 | ≤ floor + δ | δ = floor (budget 2× floor) | ≤ 9.02e-05 | ≤ 4.61e-04 |
| KL p99 | ≤ floor + δ | δ = floor (budget 2× floor) | ≤ 1.15e-02 | ≤ 8.94e-03 |
| \|ΔPPL\| (relative) | ≤ \|floor\| + δ | 0.25% (coarse guard, ~8 SE) | ≤ 0.340% | ≤ 0.331% |

Verdicts (`results/noise_floor_*.json → budgets`):

- 8B: bf16, bf16-r2, bf16-eager, bf16-chunk256, fp16 and bug-eps **PASS**. bug-rope **FAIL** (top-1 98.57% <
  98.68%; KL p99 1.75e-2 > 1.15e-2).
- 0.6B: same pattern. bug-rope FAILs (top-1 and KL p99); fp32-bug-rope FAILs on KL p99 by 1%.

The KL δ is relative because the floor varies 5× across models (KL p50: 4.5e-5 for 8B, 2.3e-4 for 0.6B). At 2× floor,
eager attention uses 79% of the KL p99 budget, so the margin for real engine kernels is thin. Recalibrate once the
first plug-serve or SGLang run exists (open issue 1). For the "Approve" path (§4.6, no regression beyond δ vs the
current snapshot), use the same δ against the snapshot's own L2 report. Optional addition: confident top-1 agreement
≥ floor − 0.1 pp (8B: 99.69%), which rejects rope (99.45%) and passes eager (99.706%), though tightly.

### Data needed per model

| item | size / cost |
|---|---|
| prompt set (shared, versioned by sha256) | 318 KB |
| `golden.safetensors` (top-20 + target + offsets + meta) | 5.4 MB per 32.8k positions |
| floor report + budgets (JSON) | ~1 KB; one HF run in the activation dtype (8B bf16: ~9 s on one A5000) |
| recording | fp32 on CPU, about 4 bytes/param RAM plus logits: 8B → 49 GB peak, 4–8 min at 44 threads. Larger models (estimated, not measured): 32B ≈ 130 GB RAM; beyond host RAM use `device_map` over GPUs in fp32 with TF32 off |
| pinned | model revision and blob hashes, transformers/torch versions, thread count, prompt-set hash |

### Quantized models (short note)

- The golden is the dense HF class loaded with `dequant_ref` weights, in fp32 on CPU. L2 then measures the engine's
  numerics against the *same quantized model*, not the quantization loss. Quantization loss (vs the unquantized fp32
  golden) is reported as information only, not as a pass rule.
- The floor is HF bf16 with the same dequantized weights. Dequantized values are not always bf16-representable; that
  rounding becomes part of the floor.
- Integer kernels (mmq/mmvq) also quantize activations (q8_1), an error source the bf16 floor does not contain. The
  floor for those types must come from a second legitimate reference that does the same: llama.cpp logits on the
  prompt set, or a torch reference with simulated q8_1 activations. Expect a larger floor per quant type, calibrated
  per type (IQ3_XXS ≠ Q8_0).
- Dequant correctness itself belongs to L0 (`dequant_ref` vs gguf-py), not to L2.

## Comparator API sketch

```python
# judge recorder (CPU fp32), once per (model revision, prompt-set version)
gold = record("Qwen/Qwen3-8B", "prompts/v1.jsonl", device="cpu", dtype="float32", k=20)   # golden.record
gold.save("judge/goldens/qwen3-8b/v1/golden.safetensors")         # written by the tool, never by hand

# calibration, once per (model, activation dtype): HF in that dtype
floor = compare(gold, record(..., device="cuda", dtype="bfloat16"))
budgets = proposed_budgets(floor)                                 # stored next to the golden

# L2 check of an engine: prompt logprobs (top-k ≥ 20, plus the target token) for every prompt in the set,
# packed into a Recording with the same offsets / prompt hash
report = compare(gold, candidate, budgets)
report.passed, report.failures       # e.g. ("top1 0.9857 < 0.9868", "KL p99 1.75e-02 > 1.15e-02")
report.top1, report.top1_confident, report.kl_p50, report.kl_p99, report.dppl, report.dppl_se, report.missing
```

Engine contract: the fp32 log-softmax of the engine's own logits, for top-k ids and for the target id at every
prompt position. Golden ids missing from the candidate's top-k (1.4% for bf16) are imputed as
`min(q_k, leftover/(m+1))`. An engine that can return logprobs at requested ids (the golden's top-20) removes the
imputation.

## Open issues

1. **δ is calibrated only on HF variants** (eager, chunked). Before M0 acceptance, add the first real-engine run
   (SGLang/vLLM prompt logprobs, then plug-serve) as a calibration point. If it exceeds 2× floor on KL p99 for a
   legitimate reason, widen δ_KL to 3×; the rope bug (3.0×) would then sit on the boundary.
2. The floor was measured only on sm_86 (A5000). Other architectures use different bf16 GEMM kernels, so expect a
   similar magnitude but not identical numbers. Verify on sm_120 before using one floor for all classes.
3. Bugs smaller than the bf16 noise (bug-eps) pass L2 by construction. L0 op oracles must cover config-constant bugs.
4. Prompt set v1 is on-policy for Qwen3-8B only. For other models the generated turns are off-policy, which is
   acceptable for a numerics check, but the context/generated split means something different per model.
5. fp32 goldens depend on the CPU thread count at the 1e-5 logprob level (harmless for L2). Pin threads to reproduce
   bytes.
6. The full-vocab validation subset should be offset from sequence starts (`arange(1, T, stride)`) if it is kept;
   `noise_floor.py` currently masks the starts out after the fact.
7. Cost of goldens for models larger than host RAM (fp32 > 200 GB) is not measured.
