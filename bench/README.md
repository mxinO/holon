# bench — Holon vs reference engines (plan §3.2 G, acceptance §3.4 items 2–3)

Same GPU, same model, same requests; every engine in its own environment; client-side timing; interleaved rounds.

## Layout

| file | job |
|---|---|
| `workloads.py` | deterministic workload specs → `traces/*.jsonl` (lengths and turns only; token ids are drawn from a seeded stream at replay time) |
| `driver.py` | the `Backend` protocol and the trace replayer (arrivals, multi-turn dependencies, per-token client timestamps, per-step wall/CPU time) |
| `backends/{holon,minisgl,nanovllm}.py` | one adapter per engine: `add`, `step`, `has_work`, `info` |
| `backends/http.py`, `llamaserver.py`, `holon_http.py` | engines behind a server (llama.cpp's `llama-server`; Holon over its OpenAI frontend), streamed; their "steps" are client polls, so only request timings compare |
| `llama_bench.py` | Holon's in-process llama-bench (pp512, tg128 at depths), batch 1 |
| `worker.py` | runs one backend over a list of traces inside that backend's venv; writes raw JSON |
| `run.py` | orchestrator: ABAB rounds of fresh worker processes, summaries, evidence record |
| `setup_baselines.sh` | pinned baseline checkouts and venvs under `bench/.envs/` (gitignored) |
| `decode_kernels.py` | E2's per-kernel decode step micro-benchmark (separate from the harness) |

`workloads.py`, `driver.py` and `worker.py` are stdlib-only so they run unchanged in every engine's environment. The
HTTP backends fit the same protocol: `add` posts a streaming request, `step` returns the tokens that arrived.

## Baselines

| engine | commit | environment |
|---|---|---|
| [mini-sglang](https://github.com/sgl-project/mini-sglang) | `9a91cfafe754` (2026-05-17) | own venv: torch 2.9.1+cu128 (it pins torch < 2.10, so the shared 2.11 stack cannot be reused), flashinfer 0.7.0.post1, sgl-kernel 0.3.21; attention auto-selects FlashInfer on sm_86 |
| [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) | `bb823b3e0698` (2026-04-26) | own venv: torch 2.9.1+cu128, flash-attn 2.8.3 (prebuilt cu12/torch2.9 wheel) |

Both use their defaults otherwise (mini-sglang: overlap scheduling, CUDA graphs, radix cache; nano-vllm: CUDA
graphs, prefix cache, 256-token KV blocks). Every backend gets the same `--max-batch` (max running sequences) and
`--max-step-tokens` (prefill tokens per step); each sizes its KV cache from its own ~0.9 memory fraction. nano-vllm
rejects temperature 0, so it samples at 1e-5 (argmax in practice); the others are greedy.

## Workloads (`python bench/workloads.py`, seed 0)

- **decode_b{1,8,64,256}**: B requests at t=0, prompt 256, output 256 tokens. 256 + 256 (not 1024 + 256) because
  batch 64 of Qwen3-8B must fit the ~5 GiB of KV a 24 GB card has left after 16 GiB of weights (~37k tokens).
- **sharegpt**: 200 chat requests, ShareGPT-like log-normal lengths (prompt median ~200, output ~180), Poisson
  arrivals at 2 req/s. Synthesized: no ShareGPT dataset is cached locally and none is needed.
- **agent**: mixed agent multi-turn tool trace: 24 sessions starting as a Poisson process (0.2/s), sharing a 1536-
  token system prompt (instructions + tool schemas); 3–8 turns each; every later turn = whole history (including
  the tokens the engine generated) + a tool result (log-normal, median 256) and sometimes a user message, released
  after a tool latency (exponential, mean 0.5 s); short generations (24–192 tokens). 126 requests, contexts up to
  ~6.1k tokens, long shared prefixes.

## Metrics and method

- **TTFT** = first token − release (arrival time, or the previous turn's last token + gap); **TPOT** = (last − first)
  / (tokens − 1); p50/p90/p99 pooled over rounds. **Throughput** = output tokens / (last token − first release).
- **decode tok/s** = tokens / wall time over decode steps at peak batch (steps returning no first token).
- **CPU ms/step** (plan §3.4 item 3) = `time.process_time()` delta around `Backend.step()` on those decode steps.
  Run with `--blocking-sync`: the worker puts the CUDA primary context in blocking-sync mode (driver API, before
  the engine starts) and makes torch events blocking (event waits ignore the context flag), so a thread waiting on
  the GPU sleeps instead of spinning and the number is host work only. Checked: a 1.26 s GPU wait costs ~1 ms CPU
  through `synchronize()`, `.tolist()` and `Event.synchronize()` alike. Without the flag, spin-waits make CPU time
  ≈ wall time, so CPU time is only reported for `--blocking-sync` runs.
- Timestamps are client-side: a token counts when `step()` returns it, so overlap engines are charged their one-
  step reporting delay. Warmup (every session's first turn at t=0, 16 output tokens, other prompt tokens so nothing
  is cached for the measured run) is excluded.
- **ABAB**: every round runs each backend once in a fresh process (model load, warmup, all traces), rounds ≥ 2.
- Evidence: `evidence/bench_<workload>_<gpu-class>_<date>.json` with the command, commit, GPU class, baseline
  commits and package versions, engine configs and KV capacities, per-request rows and summaries. Raw per-step
  timings and worker logs stay in `bench/.out/` (gitignored).

## GGUF against llama.cpp (M0.1)

```bash
# in-process bars, compared with llama-bench -p 512 -n 128 -d 0,256 -ngl 99 -fa on -r 5 on the same file and card
python bench/llama_bench.py --model <file.gguf> -r 5 --blocking-sync --engine-args "--memory-fraction 0.98"
# serving: both behind HTTP, same settings; -c = Holon's measured KV tokens
LLAMA_SERVER=<llama-server> LLAMA_SERVER_CTX=65536 HOLON_SERVE_ARGS="--memory-fraction 0.98" \
python bench/run.py --name serve --gpu 0 --model <file.gguf> --backends holon_http llamaserver \
    --max-batch 48 --max-step-tokens 2048 --traces bench/traces/decode_b1.jsonl bench/traces/agent.jsonl
```

## Reproduce

```bash
bash bench/setup_baselines.sh                 # once: bench/.envs/{minisgl,nanovllm}
. .venv/bin/activate                          # Holon's env runs the orchestrator and the holon worker
M=<Qwen3-8B snapshot dir>; T=bench/traces
python bench/run.py --name decode --gpu 5 --model $M --backends holon minisgl \
    --traces $T/decode_b1.jsonl $T/decode_b8.jsonl $T/decode_b64.jsonl
python bench/run.py --name agent --gpu 5 --model $M --backends holon minisgl --traces $T/agent.jsonl
python bench/run.py --name cpu_step --gpu 5 --model <Qwen3-0.6B snapshot dir> --backends holon nanovllm \
    --max-batch 256 --blocking-sync --traces $T/decode_b256.jsonl
```

The holon worker imports Holon from this checkout (`run.py` puts `core/`, `std/`, `plugins/` first on its path).
Run one `run.py` per host at a time: both baselines bind local port 2333 for their process group, and the
rounds are meant to have the GPU to themselves (check `nvidia-smi` first).

## First results (RTX A5000, 2026-10-07)

Holon at commit 826e467 (E4): **eager, depth 0** — no CUDA graphs, no overlap — against mini-sglang's defaults, so
the gap is expected: these are baseline numbers for the harness, not a verdict on the design. 2 ABAB rounds each,
GPU idle before every run; pooled over rounds. Evidence: `evidence/bench_{decode,agent,cpu_step}_rtx-a5000_2026-10-07.json`.

**Decode throughput, Qwen3-8B bf16, prompt 256 + output 256** (acceptance §3.4-2: within 10% of mini-sglang)

| batch | Holon decode tok/s | mini-sglang decode tok/s | Holon / mini-sglang | Holon TPOT p50 | mini-sglang TPOT p50 |
|---|---|---|---|---|---|
| 1 | 19.5 | 41.6 | 0.47 | 51.4 ms | 24.1 ms |
| 8 | 161.3 | 315.7 | 0.51 | 49.6 ms | 25.3 ms |
| 64 | 1289.3 | 2047.0 | 0.63 | 53.1 ms | 34.0 ms |

mini-sglang's step at batch 1 (24 ms) is near the weight-read floor (16.4 GB at ~768 GB/s ≈ 21 ms); Holon's ~47–52
ms is host-bound (see CPU time below), flat from batch 1 to 64. Holon's two rounds differed by ~13% (47.8 vs 55.0 ms
TPOT at batch 1) — a host-bound engine feels host noise; mini-sglang's rounds agreed within 0.2%.

**Mixed agent trace, Qwen3-8B** (126 requests over 24 sessions; acceptance §3.4-2: TTFT/TPOT within 10%)

| engine | TTFT p50 / p90 / p99 ms | TPOT p50 / p90 / p99 ms | out tok/s |
|---|---|---|---|
| Holon | 130.5 / 271.7 / 458.1 | 49.4 / 54.9 / 57.4 | 64.8 |
| mini-sglang | 109.0 / 235.1 / 458.3 | 27.4 / 29.9 / 33.4 | 71.0 |

The trace is arrival-bound (output tok/s close for both); TTFT p50 is +20% and TPOT p50 +80% for Holon.

**Host CPU time per decode step at batch 256** (acceptance §3.4-3: ≤ nano-vllm's). Qwen3-0.6B, prompt 256 +
output 256, `--blocking-sync`: Qwen3-8B cannot hold 256 sequences on 24 GB (nano-vllm's 256-token KV blocks alone
would need 9.6 GB), and with CUDA graphs the baseline's host cost does not depend on model size.

| engine | CPU ms/step p50 (mean) | wall ms/step p50 | decode tok/s |
|---|---|---|---|
| Holon | 37.7 (39.7) | 39.4 | 6175 |
| nano-vllm | 2.8 (2.4) | 25.3 | 10059 |

Holon's step is CPU work end to end (eager launches plus host-side input preparation), ~13× nano-vllm's; this is the
gap E5/E6 (overlap, CUDA graphs) and §3.4-3's "move per-token input prep to GPU kernels" target.
