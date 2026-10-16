# Holon v0.1.0

The first release of Holon: an LLM inference engine for one consumer NVIDIA GPU, under Apache-2.0, installed from
source at this tag. Its core fixes only the timing of a step, memory accounting, the plugin resolver and a set of typed
contracts (4235 of at most 6000 lines); models, layer types, quantization formats, kernels, speculative decoding,
grammars and frontends are plugins, against a plugin ABI frozen at 1.0.

What this buys is isolation: adding a model touches no shared code, and each model has its own lock per GPU class —
the engine settings and kernel bindings tuned for that card, with the code they were verified against. `holon serve`
starts from the lock, and a lock changes only through the judge (outputs against fp32 goldens, or reproduced byte for
byte). Holon serves the OpenAI and Anthropic APIs, with tool calls, reasoning and grammar-constrained structured
output, so an OpenAI client, an Anthropic client or Claude Code can use it directly.

## Supported GPUs and models

| GPU | memory | SM | lock class |
|---|---|---|---|
| RTX 5080 | 16 GB | 120 | `rtx-5080` |
| RTX 4090 | 24 GB | 89 | `rtx-4090` |
| RTX 5090 | 32 GB | 120 | `rtx-5090` |
| RTX 3090 | 24 GB | 86 | `rtx-3090` |

Architectures: Qwen3, Qwen3-MoE and the Qwen3.5 hybrid (which also serves Qwen3.8), from HF safetensors or GGUF.
Locked files, written and verified on each card in the release environment (`holon models` lists them for the card it
runs on; the README's table gives every file per card):

| model file | RTX 5080 | RTX 4090 | RTX 5090 | RTX 3090 |
|---|---|---|---|---|
| Qwen3-8B bf16 | — | ✓ | ✓ | ✓ |
| Qwen3-8B Q4_K_M | ✓ | ✓ | ✓ | ✓ |
| Qwen3-8B UD-IQ3_XXS | ✓ | | | |
| Qwen3-8B NVFP4 | ✓ | ✓ | borrowed | borrowed |
| Qwen3-8B FP8 | ✓ | ✓ | borrowed | not served (SM 89+) |
| Qwen3-30B-A3B UD-IQ3_XXS | ✓ | | | |
| Qwen3-30B-A3B Q4_K_M | — | ✓ | ✓ | ✓ |
| Qwen3-30B-A3B NVFP4 | — | ✓ | borrowed | borrowed |
| Qwen3-30B-A3B-Instruct-2507 MXFP4_MOE | — | ✓ | borrowed | borrowed |
| Qwen3.8-27B UD-IQ3_S | ✓ | | | |
| Qwen3.8-27B UD-Q4_K_M | — | ✓ | ✓ | ✓ |

"—": does not fit; empty: another card's lock is borrowed (settings as they are, kernel pins as preferences),
unverified on this card; "borrowed": the same, for M0.6's low-precision files. Other checkpoints of the three
architectures serve with defaults.

## Results

Measured at release steps 8 and 9 on one clock-screened desktop per card; ratios are Holon's advantage (above 1, Holon
is faster), ranges over the four cards. Every number comes from an evidence record; the full tables, losses included,
are in `docs/release.md` §17 and the README's Performance section.

- **The release gate passes on every GGUF cell:** in-process decode (tg128) 1.04–1.11× `llama-bench` at the llama.cpp
  commit Holon vendors, prefill (pp512) 1.00–1.15× at depth 0 and 0.89–1.07× at depth 8192 (gate: ≥ 0.8×).
- **`holon serve` against `llama-server`** (GGUF, matched batch, step budget and KV): TPOT p50 1.07–1.15× at batch 1,
  1.20–1.66× at batch 8, 1.33–1.77× on the agent trace; output throughput 1.03–1.90×; the 30K-token prompt's own TTFT
  0.98–1.17×.
- **Against SGLang 0.5.21 and FreeToken** on the bf16 Qwen3-8B: parity, TPOT 0.99–1.02× (0.82× on the RTX 4090's long
  prompt), throughput 1.00–1.03×.
- **MTP on the Qwen3.8-27B** (off in the locks): 1.51–1.60× plain decoding's TPOT at batch 8, 1.27–1.88× on the agent
  trace (unpaired, greedy).
- **Agent demo:** Claude Code 2.1.300 on each card's 27B lock fixed a small seeded Python project in 3 of 3 runs on the
  RTX 5080 and 3 of 3 on the RTX 4090 (pre-registered; not a claim of general agent capability).

Caveats: `llama-server` saves its previous slot to its RAM prompt cache (`--cache-ram`, on by default) before starting
a request, so the TTFT ratios on short requests (up to 43× on the long-prompt trace's streams, 1.6–9.3× on the agent
trace) are mostly that wait or were not checked for it; the 30K prompt's own TTFT is the comparison quoted. MTP's
single-stream speedup is inflated by a looping prompt and not quoted.

Where Holon loses:
- the longest pause at batch 8 (0.15–0.63× llama.cpp's, 0.06–0.27× SGLang's): `holon serve` starts the first requests
  of a burst before the rest have prefilled;
- the streams behind a 30K prompt on the locks without mixed steps (every 8B and 30B lock): 0.09–0.14× llama.cpp's
  longest pause on the 8B Q4_K_M, 0.01–0.02× SGLang's and FreeToken's on the RTX 4090's bf16 8B;
- first tokens on the NVFP4 files against SGLang (0.22–0.69×), and small FP8 losses (0.89–0.97×);
- prefill at depth 8192 on the 8B / 30B cells, 0.89–0.99× the pinned llama.cpp.

## Known limitations

- One GPU per server, no offload.
- Context: 49,152 tokens on the Qwen3.8-27B locks, 8,192–16,384 on the 8B and 30B locks, at most 32,768 without a
  lock by default (`--max-model-len N`).
- Speculation (MTP) is off in every lock: its longest pauses failed the rule fixed before measuring. The README's
  Options section has the flags that turn it on.
- On the locks without mixed steps (every 8B and 30B lock) a long prompt stalls the streams already decoding;
  `--chunk-with-decode 512` mixes them.
- No authentication: do not expose the server beyond a trusted network.
- Every start compiles FlashInfer's CuTe DSL norms (about 1 s); the first start also compiles Triton kernels.
- M0.6's low-precision files run with borrowed, unverified locks on the RTX 5090 and RTX 3090.
- Grammar runaways (2.66% of constrained outputs truncated at `max_tokens` on the 27B with MTP) are a recorded
  shortfall; DFlash2 speculation is an opt-in in no lock.
- SHAs in older documents, evidence and lock histories refer to the private development history.

## Install

Linux x86-64, an NVIDIA driver for CUDA 13, the CUDA 13 toolkit, gcc / g++ and [uv](https://docs.astral.sh/uv/):

```bash
git clone --branch v0.1.0 <repository> holon && cd holon
uv sync --frozen
uv run holon prebuild
uv run holon serve unsloth/Qwen3.8-27B-GGUF
```

The README has the rest: the Docker image for hosts without a CUDA toolkit, the clients, Claude Code and the options.
