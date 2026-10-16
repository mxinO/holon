# Holon

Holon is an LLM inference engine for one consumer NVIDIA GPU, licensed under Apache-2.0. Its core fixes only the
timing of a step, memory accounting, the plugin resolver and a set of typed contracts, in at most 6000 lines; models,
layer types, quantization formats, kernels, speculative decoding, grammars and frontends are all plugins. Each
supported model ships a lock per GPU class: the engine settings and kernel bindings tuned for that card, with the
code they were verified against, and `holon serve` starts from it. A lock changes only through the judge, which
checks the model's outputs against fp32 goldens or reproduces them byte for byte, and every published performance
number comes from an evidence record. It serves the OpenAI and Anthropic APIs, with tool calls, reasoning and
grammar-constrained structured output, so an OpenAI client, an Anthropic client or Claude Code can use it directly;
the design is in [docs/design.md](docs/design.md).

## Supported GPUs and models

| GPU | memory | SM | lock class |
|---|---|---|---|
| RTX 5080 | 16 GB | 120 | `rtx-5080` |
| RTX 4090 | 24 GB | 89 | `rtx-4090` |
| RTX 5090 | 32 GB | 120 | `rtx-5090` |
| RTX 3090 | 24 GB | 86 | `rtx-3090` |

Other GPUs: GGUF files refuse at startup (the GGUF kernels are built for SM 86, 89 and 120 only); bf16 checkpoints
are untested.

Architectures: Qwen3 (`models.qwen3`), Qwen3-MoE (`models.qwen3_moe`) and the Qwen3.5 hybrid (`models.qwen3_5`, which
also serves Qwen3.8), from HF safetensors directories or GGUF files. The recommended file per card and its lock
(`holon models` lists them for the card it runs on):

| model file (hub repository) | RTX 5080 | RTX 4090 | RTX 5090 | RTX 3090 |
|---|---|---|---|---|
| Qwen3-8B bf16 (`Qwen/Qwen3-8B`) | does not fit | this card's lock | this card's lock | this card's lock |
| Qwen3-8B Q4_K_M (`unsloth/Qwen3-8B-GGUF`) | this card's lock | this card's lock | this card's lock | this card's lock |
| Qwen3-8B UD-IQ3_XXS (`unsloth/Qwen3-8B-GGUF`) | this card's lock | borrowed, unverified | borrowed, unverified | borrowed, unverified |
| Qwen3-8B NVFP4 (`nvidia/Qwen3-8B-NVFP4`) | this card's lock | this card's lock | borrowed, unverified | borrowed, unverified |
| Qwen3-8B FP8 (`Qwen/Qwen3-8B-FP8`) | this card's lock | this card's lock | borrowed, unverified | not served (FP8 needs SM 89 or newer) |
| Qwen3-30B-A3B UD-IQ3_XXS (`unsloth/Qwen3-30B-A3B-GGUF`) | this card's lock | borrowed, unverified | borrowed, unverified | borrowed, unverified |
| Qwen3-30B-A3B Q4_K_M (`unsloth/Qwen3-30B-A3B-GGUF`) | does not fit (17.3 GiB) | this card's lock | this card's lock | this card's lock |
| Qwen3-30B-A3B NVFP4 (`nvidia/Qwen3-30B-A3B-NVFP4`) | does not fit (16.9 GiB) | this card's lock | borrowed, unverified | borrowed, unverified |
| Qwen3-30B-A3B-Instruct-2507 MXFP4_MOE (`Face314/Qwen3-30B-A3B-Instruct-2507-MXFP4_MOE`) | does not fit (15.9 GiB) | this card's lock | borrowed, unverified | borrowed, unverified |
| Qwen3.8-27B UD-IQ3_S (`unsloth/Qwen3.8-27B-GGUF`) | this card's lock | borrowed, unverified | borrowed, unverified | borrowed, unverified |
| Qwen3.8-27B UD-Q4_K_M (`unsloth/Qwen3.8-27B-GGUF`) | does not fit (15.3 GiB) | this card's lock | this card's lock | this card's lock |

**This card's lock**: written and verified on this card (L2 against the judge's floor, the snapshot trace identical
over three runs), in the release environment (`docs/release.md` §16). Every lock selects the grammar processor. The
Qwen3.8-27B locks serve without speculation: MTP decodes 11–45% faster per token, but on a Claude-Code-shaped trace a
speculating stream's longest pauses grew 1.1–3.1× (`docs/release.md` §16.6); to serve with it, see Options.

**Borrowed, unverified.** Locks are found by the checkpoint's build hash, so a file downloaded anywhere, under any
name, finds its lock. Without a lock for this card, `holon serve` borrows the nearest class's: the same GPU on another
toolchain, then the same SM, then the same toolchain. A borrowed lock's settings apply as they are and its kernel pins
become preferences (a provider that cannot run here falls back to the default choice, logged). It was not verified on
this card. If it does not start with a plugin or memory error, the server falls back to defaults at once.

**Served without a lock** (defaults, logged): every other checkpoint of the three architectures, for example
Qwen3.5 4B / 9B and the bf16 Qwen3.8-27B and Qwen3-30B-A3B (large-memory paths). Defaults serve at most 8 sequences
(`holon serve <model> --max-seqs N` changes it) and a context of min(the model's trained context, 32768) tokens,
clamped to what fits.

## Install

Install the release tag. Older SHAs in docs, evidence and lock histories refer to the private development history.

Host requirements: Linux x86-64, an NVIDIA driver supporting CUDA 13, the CUDA 13 toolkit (`nvcc` on `PATH` or under
`CUDA_HOME`), gcc / g++, and [uv](https://docs.astral.sh/uv/). uv brings Python 3.12, torch 2.11 for CUDA 13.0 and
every pinned dependency from `uv.lock`.

```bash
git clone --branch v0.1.0 <repository> holon && cd holon
uv sync --frozen                     # Python 3.12, torch 2.11 + CUDA 13 and every pin from uv.lock
uv run holon prebuild                # GGUF kernels and FlashInfer modules for the visible GPUs, once
```

`holon prebuild` compiles the GGUF kernels and FlashInfer's modules for the SM of every visible GPU, once per install;
`holon prebuild --sm 86 89 120` names the targets instead, and `MAX_JOBS=N` caps the parallel compiles.

### Docker

The `Dockerfile` installs the same environment for hosts without a CUDA toolkit and prebuilds for SM 86, 89 and 120.
Build it from the release tag, never from a working directory:

```bash
git archive --format=tar v0.1.0 | docker build -t holon:v0.1.0 -           # --build-arg MAX_JOBS=N caps compiles
docker run --gpus all -p 8000:8000 -v <model dir>:/models holon:v0.1.0 serve /models/<file> --host 0.0.0.0
docker run --gpus all -p 8000:8000 -e HF_HOME=/hf -v <hf cache dir>:/hf holon:v0.1.0 \
  serve unsloth/Qwen3.8-27B-GGUF --host 0.0.0.0                            # a hub model, kept in <hf cache dir>
```

As another user, give the container a writable home, so the cache of weight-file hashes (`~/.cache/holon/weights`)
persists across starts instead of each start re-reading the file:

```bash
docker run --gpus all --user "$(id -u):$(id -g)" -e HOME=/cache -v <cache dir>:/cache \
  -p 8000:8000 -v <model dir>:/models holon:v0.1.0 serve /models/<file> --host 0.0.0.0
```

## Quickstart

```bash
uv run holon models                                   # the models locked for this card, by name
uv run holon serve unsloth/Qwen3.8-27B-GGUF           # this card's locked file, downloaded into the HF cache
```

`holon serve` takes a local file or directory, or a Hugging Face Hub reference, downloaded into the HF cache once:

| reference | serves |
|---|---|
| `unsloth/Qwen3.8-27B-GGUF:UD-IQ3_S` | the repository's GGUF file whose name ends in `-UD-IQ3_S` (any case; `:Q4_K_M` takes `…-Q4_K_M.gguf` over `…-UD-Q4_K_M.gguf`) |
| `unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_S` | that file, as `holon models` names it |
| `unsloth/Qwen3.8-27B-GGUF` | the file this card has a verified lock for (`holon models`: "this card's lock"); else the repository's only GGUF file, else an error listing them |
| `Qwen/Qwen3-8B` | an HF safetensors checkpoint |

The log names the file chosen and its size; a file larger than the GPU's memory is refused before downloading
(`--no-size-check` skips that). The hub's own settings apply: `hf auth login` for gated repositories, `HF_HOME` for the
cache, and `HF_HUB_OFFLINE=1` to serve cached models without the network (a tag then sees only the cached files).
Split GGUF files (`-00001-of-0000N.gguf`) are not supported.

`holon serve` listens on `127.0.0.1:8000` (`holon serve <model> --host H --port P`) and checks no API key: do not
expose it beyond a trusted network. Its log says which lock it starts from, or why none:

```text
holon: starting from …/plans/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_S@<build>/rtx-5080/cu130-torch2.11/<n>.lock
holon: engine ready, plan sha256:…, from …/<n>.lock
  differs from the lock: engine:max_model_len: …
```

A borrowed lock adds `(unverified: <class>'s)`; no lock logs
`no current lock for <model>@<build> in …/plans: serving defaults`. The line
`the fallback without a lock: max_seqs 8` describes the configuration used only if the lock fails to start.
The `differs from the lock` lines list every setting that differs from the lock: the flags you gave, and
`max_model_len` when the lock's context does not fit this card's memory. `/metrics` reports the same plan.

The served model id is in `/v1/models` (`curl http://127.0.0.1:8000/v1/models`): the checkpoint's name without its
repository, e.g. `Qwen3.8-27B-UD-IQ3_S` (`--plugin-options http.model_name=ID` sets another); the server ignores the
model a request names. Clients (`pip install openai anthropic` in your own environment):

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="holon")
model = client.models.list().data[0].id
r = client.chat.completions.create(model=model, max_tokens=512,
                                   messages=[{"role": "user", "content": "Name three prime numbers."}])
print(r.choices[0].message.content)        # the reasoning, if any, is in r.choices[0].message.reasoning_content
```

```python
import anthropic

client = anthropic.Anthropic(base_url="http://127.0.0.1:8000", api_key="holon")
msg = client.messages.create(model=model, max_tokens=512,
                             messages=[{"role": "user", "content": "Name three prime numbers."}])
print("".join(b.text for b in msg.content if b.type == "text"))   # thinking blocks come first
```

The server also answers `/v1/completions`, `/v1/responses`, `/v1/messages/count_tokens` and `/health`.

### Claude Code

Once it listens, the server prints the line that starts Claude Code against it, with its model id:

```text
holon: Claude Code:
  ANTHROPIC_BASE_URL=http://127.0.0.1:8000 ANTHROPIC_AUTH_TOKEN=holon ANTHROPIC_MODEL=Qwen3.8-27B-UD-IQ3_S ANTHROPIC_DEFAULT_HAIKU_MODEL=Qwen3.8-27B-UD-IQ3_S CLAUDE_CODE_SUBAGENT_MODEL=Qwen3.8-27B-UD-IQ3_S claude
```

The server drops Claude Code's per-request billing-header system block, which would otherwise defeat prefix caching
(`holon serve <model> --plugin-options http.drop_billing_header=false` keeps it; `/metrics` counts the drops).
Claude Code's auto-compaction assumes a far larger window than the 32K tokens served at most by default.

The release's agent demo (`bench/demo/`) is a small fix in a three-module Python project — a seeded bug and a missing
function — chosen to fit that 32K context. It shows the served model completing such a task through Claude Code; it
is not a claim of general agent capability. **Result** (pre-registered, Claude Code 2.1.300, each card's 27B lock,
scored by the harness's own copy of the tests): **3 of 3 runs passed on the RTX 5080 (UD-IQ3_S) and 3 of 3 on the
RTX 4090 (UD-Q4_K_M)**, in 13–16 turns and 2–5 minutes each (`evidence/agent_demo_rtx-5080_2026-10-16.json`,
`evidence/agent_demo_rtx-4090_2026-10-15.json`; `docs/release.md` §18). Claude Code sends no temperature, so the
server's default (1.0) applies. Claude Code 2.1.300 also sends side requests of ≈ 31K tokens, which fit the 32K
window with little room.

## Options

Flags override the lock, which overrides the defaults; `holon serve --help` lists every flag (one per engine
setting), every plugin option the installed plugins read, and the plugins `--slots` selects among. An option no plugin
declares is warned about; a flag's value outside an option's choices is refused before anything is downloaded, and a
lock's is warned about and that lock not used.

- **Speculation (MTP) on the Qwen3.8-27B:** `holon serve <model> --max-draft-tokens 3 --slots drafter=drafters.mtp
  draft_policy=std.spec.fixed_k verifier=std.spec.greedy --plugin-options spec.k=1:3,3:2`; `--max-draft-tokens 0`
  turns speculation off.
- **Mixed steps:** the plain Qwen3.8-27B locks set `--chunk-with-decode 512`: a step that decodes may also carry up to
  512 prompt tokens, so a long prompt does not stall the streams already decoding. Turn speculation on over such a
  lock (a drafter and `--max-draft-tokens N`) and the same setting instead alternates a prefill step of at most 512
  tokens with a verify step. `--chunk-with-decode 0` turns both off.
- **Ignore the lock:** `holon serve <model> --no-lock` serves defaults and your flags. Use it if a start hangs: the
  server retries a start that fails other than with a plugin or memory error 4 times, each up to 600 s, before falling
  back to defaults.
- **Change or delete one key of the lock:** dict flags merge per key, and `key=` (empty value) deletes the lock's
  setting: `holon serve <model> --plugin-options kv.format=` serves the default KV format, and
  `holon serve <model> --slots logits_processor=` deselects the grammar processor.
- **Mixed-GPU hosts:** Holon serves on one GPU, the first visible one, and picks the lock of its class. Pin the card,
  for the prebuild too: `CUDA_VISIBLE_DEVICES=1 uv run holon serve <model>`.
- **Other locks:** `holon serve <model> --plans DIR` reads lock histories from `DIR` (default: `plans/` of this tree).

### Structured output

The release locks select the grammar processor `grammar.xgrammar`. With `--no-lock` or without a lock,
`holon serve <model> --slots logits_processor=grammar.xgrammar` selects it; without it, constrained requests are
refused. Constrained:
`response_format` with `json_schema` or `json_object` (OpenAI chat and Responses), `tool_choice` `required` or a named
tool, `parallel_tool_calls: false`, and `strict: true` tools. `tool_choice: auto` stays unconstrained unless a tool is
strict.

```python
schema = {"type": "object", "properties": {"city": {"type": "string"}, "population": {"type": "integer"}},
          "required": ["city", "population"], "additionalProperties": False}
r = client.chat.completions.create(model=model, max_tokens=512,
    messages=[{"role": "user", "content": "The largest city in Japan, as JSON."}],
    response_format={"type": "json_schema", "json_schema": {"name": "city", "schema": schema, "strict": True}})
```

A strict schema that uses a feature the grammar cannot enforce gets a 400 naming it; non-strict schemas compile best
effort (the reject list: [docs/m05-grammar.md](docs/m05-grammar.md), D9). The processor runs two vetting workers,
which take 1.6–2.2 GB of host RAM and 3–4 s to start.

## First start

After `holon prebuild`, a first start compiles only Triton kernels (12–40 per model, cached on disk for later starts)
and FlashInfer's CuTe DSL norms, which compile in memory at every start, about 0.3 s per norm shape and about 1 s per
start. A file outside the HF cache is hashed once to find its lock; the hash is remembered in
`~/.cache/holon/weights`.

## Known limitations

- One GPU per server; no offload: the model and its KV cache must fit the card.
- The context is at most 32768 tokens by default (`holon serve <model> --max-model-len N`), and less when it does not
  fit.
- Under a grammar, an output can run on in whitespace, digits or a string the schema admits until `max_tokens`: on the
  27B with MTP, 2.66% of constrained outputs were truncated against 0.33% unconstrained, over a 2-point gate that was
  accepted as a recorded shortfall. Guards against such runaways are deferred; set `max_tokens`.
- DFlash2 speculation is an opt-in, in no lock ([docs/m04-spec.md](docs/m04-spec.md)).
- Stop the server before a `git pull`: the install is editable, so a supervisor restart would import the new code.

## Performance

Measured at release step 8 on one desktop per card, each clock-screened before and after, with Holon serving from that
card's lock. The method, every table below and the absolute numbers per board are in
[docs/release.md §17](docs/release.md#17-as-built-step-8). Each table is rendered by `bench/report.py` from the evidence
records it links, and losses are included. **Ratios are the claim**: absolute tokens per second belong to one board.

**The gate.** Holon's in-process prefill (pp512) and decode (tg128), at depth 0 and 8192, are compared with
`llama-bench` on the same GGUF file and card. They run at llama.cpp `cbe28bb3`, the commit Holon's GGUF kernels vendor,
and at upstream `f12aaa34` (2026-10-16), in 3 ABBA rounds. The release gate is ≥ 0.8× the pin on every metric, and
every cell passes it: decode is 1.04–1.11× the pin on every card. Prefill is 1.00–1.15× at depth 0 and 0.89–1.07× at
depth 8192, under the 1.04–1.20× predicted. "≈" marks a ratio within 2 sd of its per-round ratios of 1.

NVIDIA GeForce RTX 3090; Holon's ggml at cbe28bb3

| file (record) | lock | metric | Holon t/s | llama.cpp gate (cbe28bb3c) t/s | Holon / gate | llama.cpp upstream (f12aaa344) t/s | Holon / upstream |
|---|---|---|---|---|---|---|---|
| [qwen3-30b-a3b-q4_k_m](evidence/bars_release-qwen3-30b-a3b-q4_k_m_rtx-3090_2026-10-16.json) | rtx-3090 | pp512 | 5223.2 | 4775.0 | 1.09 | 4752.1 | 1.10 |
| [qwen3-30b-a3b-q4_k_m](evidence/bars_release-qwen3-30b-a3b-q4_k_m_rtx-3090_2026-10-16.json) | rtx-3090 | pp512 @ d8192 | 3335.4 | 3381.0 | 0.99 | 3366.2 | 0.99 |
| [qwen3-30b-a3b-q4_k_m](evidence/bars_release-qwen3-30b-a3b-q4_k_m_rtx-3090_2026-10-16.json) | rtx-3090 | tg128 | 234.4 | 214.4 | 1.09 | 214.9 | 1.09 |
| [qwen3-30b-a3b-q4_k_m](evidence/bars_release-qwen3-30b-a3b-q4_k_m_rtx-3090_2026-10-16.json) | rtx-3090 | tg128 @ d8192 | 193.1 | 177.7 | 1.09 | 177.5 | 1.09 |
| [qwen3.8-27b-ud-q4_k_m](evidence/bars_release-qwen3.8-27b-ud-q4_k_m_rtx-3090_2026-10-16.json) | rtx-3090 | pp512 | 1732.4 | 1594.6 | 1.09 | 1652.5 | 1.05 |
| [qwen3.8-27b-ud-q4_k_m](evidence/bars_release-qwen3.8-27b-ud-q4_k_m_rtx-3090_2026-10-16.json) | rtx-3090 | pp512 @ d8192 | 1558.4 | 1454.4 | 1.07 | 1500.9 | 1.04 |
| [qwen3.8-27b-ud-q4_k_m](evidence/bars_release-qwen3.8-27b-ud-q4_k_m_rtx-3090_2026-10-16.json) | rtx-3090 | tg128 | 47.2 | 44.7 | 1.06 | 44.7 | 1.06 |
| [qwen3.8-27b-ud-q4_k_m](evidence/bars_release-qwen3.8-27b-ud-q4_k_m_rtx-3090_2026-10-16.json) | rtx-3090 | tg128 @ d8192 | 45.9 | 43.6 | 1.05 | 43.6 | 1.05 |
| [qwen3-8b-q4_k_m](evidence/bars_release-qwen3-8b-q4_k_m_rtx-3090_2026-10-16.json) | rtx-3090 | pp512 | 5803.1 | 5307.3 | 1.09 | 5283.5 | 1.10 |
| [qwen3-8b-q4_k_m](evidence/bars_release-qwen3-8b-q4_k_m_rtx-3090_2026-10-16.json) | rtx-3090 | pp512 @ d8192 | 3822.6 | 3880.8 | 0.98 | 3902.1 | 0.98 |
| [qwen3-8b-q4_k_m](evidence/bars_release-qwen3-8b-q4_k_m_rtx-3090_2026-10-16.json) | rtx-3090 | tg128 | 154.2 | 142.1 | 1.08 | 142.2 | 1.08 |
| [qwen3-8b-q4_k_m](evidence/bars_release-qwen3-8b-q4_k_m_rtx-3090_2026-10-16.json) | rtx-3090 | tg128 @ d8192 | 128.0 | 118.9 | 1.08 | 118.8 | 1.08 |

NVIDIA GeForce RTX 4090; Holon's ggml at cbe28bb3

| file (record) | lock | metric | Holon t/s | llama.cpp gate (cbe28bb3c) t/s | Holon / gate | llama.cpp upstream (f12aaa344) t/s | Holon / upstream |
|---|---|---|---|---|---|---|---|
| [qwen3-30b-a3b-mxfp4_moe](evidence/bars_release-qwen3-30b-a3b-mxfp4_moe_rtx-4090_2026-10-16.json) | rtx-4090 | pp512 | 10794.1 | 10745.9 | 1.00 | 10682.6 | 1.01 |
| [qwen3-30b-a3b-mxfp4_moe](evidence/bars_release-qwen3-30b-a3b-mxfp4_moe_rtx-4090_2026-10-16.json) | rtx-4090 | pp512 @ d8192 | 7438.7 | 7944.1 | 0.94 | 7912.2 | 0.94 |
| [qwen3-30b-a3b-mxfp4_moe](evidence/bars_release-qwen3-30b-a3b-mxfp4_moe_rtx-4090_2026-10-16.json) | rtx-4090 | tg128 | 242.3 | 222.3 | 1.09 | 222.2 | 1.09 |
| [qwen3-30b-a3b-mxfp4_moe](evidence/bars_release-qwen3-30b-a3b-mxfp4_moe_rtx-4090_2026-10-16.json) | rtx-4090 | tg128 @ d8192 | 198.3 | 178.4 | 1.11 | 178.1 | 1.11 |
| [qwen3-30b-a3b-q4_k_m](evidence/bars_release-qwen3-30b-a3b-q4_k_m_rtx-4090_2026-10-16.json) | rtx-4090 | pp512 | 10155.9 | 10087.8 | 1.01 | 10026.6 | 1.01 |
| [qwen3-30b-a3b-q4_k_m](evidence/bars_release-qwen3-30b-a3b-q4_k_m_rtx-4090_2026-10-16.json) | rtx-4090 | pp512 @ d8192 | 7211.0 | 7663.5 | 0.94 | 7629.9 | 0.94 |
| [qwen3-30b-a3b-q4_k_m](evidence/bars_release-qwen3-30b-a3b-q4_k_m_rtx-4090_2026-10-16.json) | rtx-4090 | tg128 | 271.1 | 249.8 | 1.08 | 249.7 | 1.09 |
| [qwen3-30b-a3b-q4_k_m](evidence/bars_release-qwen3-30b-a3b-q4_k_m_rtx-4090_2026-10-16.json) | rtx-4090 | tg128 @ d8192 | 217.4 | 195.8 | 1.11 | 195.5 | 1.11 |
| [qwen3.8-27b-ud-q4_k_m](evidence/bars_release-qwen3.8-27b-ud-q4_k_m_rtx-4090_2026-10-16.json) | rtx-4090 | pp512 | 3758.2 | 3435.7 | 1.09 | 3500.0 | 1.07 |
| [qwen3.8-27b-ud-q4_k_m](evidence/bars_release-qwen3.8-27b-ud-q4_k_m_rtx-4090_2026-10-16.json) | rtx-4090 | pp512 @ d8192 | 3358.2 | 3190.3 | 1.05 | 3240.6 | 1.04 |
| [qwen3.8-27b-ud-q4_k_m](evidence/bars_release-qwen3.8-27b-ud-q4_k_m_rtx-4090_2026-10-16.json) | rtx-4090 | tg128 | 47.5 | 45.1 | 1.05 | 45.1 | 1.05 |
| [qwen3.8-27b-ud-q4_k_m](evidence/bars_release-qwen3.8-27b-ud-q4_k_m_rtx-4090_2026-10-16.json) | rtx-4090 | tg128 @ d8192 | 46.1 | 43.8 | 1.05 | 43.8 | 1.05 |
| [qwen3-8b-q4_k_m](evidence/bars_release-qwen3-8b-q4_k_m_rtx-4090_2026-10-16.json) | rtx-4090 | pp512 | 12892.7 | 12366.8 | 1.04 | 12300.6 | 1.05 |
| [qwen3-8b-q4_k_m](evidence/bars_release-qwen3-8b-q4_k_m_rtx-4090_2026-10-16.json) | rtx-4090 | pp512 @ d8192 | 8871.1 | 9307.8 | 0.95 | 9354.4 | 0.95 |
| [qwen3-8b-q4_k_m](evidence/bars_release-qwen3-8b-q4_k_m_rtx-4090_2026-10-16.json) | rtx-4090 | tg128 | 154.9 | 146.6 | 1.06 | 146.6 | 1.06 |
| [qwen3-8b-q4_k_m](evidence/bars_release-qwen3-8b-q4_k_m_rtx-4090_2026-10-16.json) | rtx-4090 | tg128 @ d8192 | 127.3 | 120.5 | 1.06 | 120.4 | 1.06 |

NVIDIA GeForce RTX 5080; Holon's ggml at cbe28bb3

| file (record) | lock | metric | Holon t/s | llama.cpp gate (cbe28bb3c) t/s | Holon / gate | llama.cpp upstream (f12aaa344) t/s | Holon / upstream |
|---|---|---|---|---|---|---|---|
| [qwen3-30b-a3b-ud-iq3_xxs](evidence/bars_release-qwen3-30b-a3b-ud-iq3_xxs_rtx-5080_2026-10-16.json) | rtx-5080 | pp512 | 8136.7 | 7364.8 | 1.10 | 7342.7 | 1.11 |
| [qwen3-30b-a3b-ud-iq3_xxs](evidence/bars_release-qwen3-30b-a3b-ud-iq3_xxs_rtx-5080_2026-10-16.json) | rtx-5080 | pp512 @ d8192 | 4961.4 | 5601.6 | 0.89 | 5487.6 | 0.90 |
| [qwen3-30b-a3b-ud-iq3_xxs](evidence/bars_release-qwen3-30b-a3b-ud-iq3_xxs_rtx-5080_2026-10-16.json) | rtx-5080 | tg128 | 261.9 | 246.1 | 1.06 | 246.1 | 1.06 |
| [qwen3-30b-a3b-ud-iq3_xxs](evidence/bars_release-qwen3-30b-a3b-ud-iq3_xxs_rtx-5080_2026-10-16.json) | rtx-5080 | tg128 @ d8192 | 211.8 | 197.2 | 1.07 | 197.3 | 1.07 |
| [qwen3.8-27b-ud-iq3_s](evidence/bars_release-qwen3.8-27b-ud-iq3_s_rtx-5080_2026-10-16.json) | rtx-5080 | pp512 | 2544.0 | 2219.4 | 1.15 | 2298.2 | 1.11 |
| [qwen3.8-27b-ud-iq3_s](evidence/bars_release-qwen3.8-27b-ud-iq3_s_rtx-5080_2026-10-16.json) | rtx-5080 | pp512 @ d8192 | 2153.9 | 2052.8 | 1.05 | 2127.9 | 1.01 |
| [qwen3.8-27b-ud-iq3_s](evidence/bars_release-qwen3.8-27b-ud-iq3_s_rtx-5080_2026-10-16.json) | rtx-5080 | tg128 | 63.8 | 60.1 | 1.06 | 60.1 | 1.06 |
| [qwen3.8-27b-ud-iq3_s](evidence/bars_release-qwen3.8-27b-ud-iq3_s_rtx-5080_2026-10-16.json) | rtx-5080 | tg128 @ d8192 | 61.6 | 56.8 | 1.08 | 56.8 | 1.08 |
| [qwen3-8b-q4_k_m](evidence/bars_release-qwen3-8b-q4_k_m_rtx-5080_2026-10-16.json) | rtx-5080 | pp512 | 8684.4 | 8015.6 | 1.08 | 7979.4 | 1.09 |
| [qwen3-8b-q4_k_m](evidence/bars_release-qwen3-8b-q4_k_m_rtx-5080_2026-10-16.json) | rtx-5080 | pp512 @ d8192 | 5546.6 | 6138.6 | 0.90 | 6004.8 | 0.92 |
| [qwen3-8b-q4_k_m](evidence/bars_release-qwen3-8b-q4_k_m_rtx-5080_2026-10-16.json) | rtx-5080 | tg128 | 166.3 | 158.4 | 1.05 | 158.3 | 1.05 |
| [qwen3-8b-q4_k_m](evidence/bars_release-qwen3-8b-q4_k_m_rtx-5080_2026-10-16.json) | rtx-5080 | tg128 @ d8192 | 136.0 | 129.4 | 1.05 | 129.3 | 1.05 |
| [qwen3-8b-ud-iq3_xxs](evidence/bars_release-qwen3-8b-ud-iq3_xxs_rtx-5080_2026-10-16.json) | rtx-5080 | pp512 | 8923.8 | 8103.7 | 1.10 | 8132.9 | 1.10 |
| [qwen3-8b-ud-iq3_xxs](evidence/bars_release-qwen3-8b-ud-iq3_xxs_rtx-5080_2026-10-16.json) | rtx-5080 | pp512 @ d8192 | 5625.6 | 6206.6 | 0.91 | 6107.4 | 0.92 |
| [qwen3-8b-ud-iq3_xxs](evidence/bars_release-qwen3-8b-ud-iq3_xxs_rtx-5080_2026-10-16.json) | rtx-5080 | tg128 | 219.3 | 209.1 | 1.05 | 209.1 | 1.05 |
| [qwen3-8b-ud-iq3_xxs](evidence/bars_release-qwen3-8b-ud-iq3_xxs_rtx-5080_2026-10-16.json) | rtx-5080 | tg128 @ d8192 | 168.7 | 161.8 | 1.04 | 161.6 | 1.04 |

NVIDIA GeForce RTX 5090; Holon's ggml at cbe28bb3

| file (record) | lock | metric | Holon t/s | llama.cpp gate (cbe28bb3c) t/s | Holon / gate | llama.cpp upstream (f12aaa344) t/s | Holon / upstream |
|---|---|---|---|---|---|---|---|
| [qwen3-30b-a3b-q4_k_m](evidence/bars_release-qwen3-30b-a3b-q4_k_m_rtx-5090_2026-10-16.json) | rtx-5090 | pp512 | 12736.4 | 11581.0 | 1.10 | 11540.7 | 1.10 |
| [qwen3-30b-a3b-q4_k_m](evidence/bars_release-qwen3-30b-a3b-q4_k_m_rtx-5090_2026-10-16.json) | rtx-5090 | pp512 @ d8192 | 8630.5 | 9026.6 | 0.96 | 8907.0 | 0.97 |
| [qwen3-30b-a3b-q4_k_m](evidence/bars_release-qwen3-30b-a3b-q4_k_m_rtx-5090_2026-10-16.json) | rtx-5090 | tg128 | 387.5 | 362.5 | 1.07 | 363.1 | 1.07 |
| [qwen3-30b-a3b-q4_k_m](evidence/bars_release-qwen3-30b-a3b-q4_k_m_rtx-5090_2026-10-16.json) | rtx-5090 | tg128 @ d8192 | 325.9 | 295.4 | 1.10 | 295.2 | 1.10 |
| [qwen3.8-27b-ud-q4_k_m](evidence/bars_release-qwen3.8-27b-ud-q4_k_m_rtx-5090_2026-10-15.json) | rtx-5090 | pp512 | 4763.2 | 4174.9 | 1.14 | 4321.3 | 1.10 |
| [qwen3.8-27b-ud-q4_k_m](evidence/bars_release-qwen3.8-27b-ud-q4_k_m_rtx-5090_2026-10-15.json) | rtx-5090 | pp512 @ d8192 | 4103.4 | 3953.1 | 1.04 | 4079.7 | 1.01 |
| [qwen3.8-27b-ud-q4_k_m](evidence/bars_release-qwen3.8-27b-ud-q4_k_m_rtx-5090_2026-10-15.json) | rtx-5090 | tg128 | 87.3 | 82.4 | 1.06 | 82.4 | 1.06 |
| [qwen3.8-27b-ud-q4_k_m](evidence/bars_release-qwen3.8-27b-ud-q4_k_m_rtx-5090_2026-10-15.json) | rtx-5090 | tg128 @ d8192 | 84.9 | 78.6 | 1.08 | 78.6 | 1.08 |
| [qwen3-8b-q4_k_m](evidence/bars_release-qwen3-8b-q4_k_m_rtx-5090_2026-10-15.json) | rtx-5090 | pp512 | 15458.6 | 14140.0 | 1.09 | 14085.5 | 1.10 |
| [qwen3-8b-q4_k_m](evidence/bars_release-qwen3-8b-q4_k_m_rtx-5090_2026-10-15.json) | rtx-5090 | pp512 @ d8192 | 10311.7 | 10920.8 | 0.94 | 10714.7 | 0.96 |
| [qwen3-8b-q4_k_m](evidence/bars_release-qwen3-8b-q4_k_m_rtx-5090_2026-10-15.json) | rtx-5090 | tg128 | 275.1 | 253.2 | 1.09 | 253.2 | 1.09 |
| [qwen3-8b-q4_k_m](evidence/bars_release-qwen3-8b-q4_k_m_rtx-5090_2026-10-15.json) | rtx-5090 | tg128 @ d8192 | 227.2 | 207.8 | 1.09 | 207.7 | 1.09 |

**Serving.** `holon serve` is compared with `llama-server` (the same pin, matched to the lock's batch, step budget, KV
format and KV size), SGLang 0.5.21 and FreeToken `74a38777`. All run greedy on natural-text prompts, 4 ABBA rounds
(decode-only rows: 3). Each ratio is Holon's advantage: the baseline's TTFT p50, TPOT p50 or longest-pause p99 over
Holon's, or Holon's output tok/s over the baseline's. Above 1, Holon is faster. "—" is a baseline that failed, with its
reason.
- `decode_b1` / `decode_b8`: 1 or 8 requests of 256 + 256 tokens.
- `agent`: 24 multi-turn tool sessions.
- `long_prompt`: a 30K-token prompt arriving behind three streams. Its TTFT p50 is the three streams'.

Against `llama-server` the long prompt's large TTFT ratios (4.7–43×) are mostly `llama-server` saving its previous slot
to its default RAM prompt cache before it starts the three streams. The 30K request's own TTFT is 0.98–1.17× llama.cpp's
(`docs/release.md` §17.4).

Where Holon loses:
- **The longest pause on `decode_b8`.** Of 8 requests sent together over HTTP, `holon serve` starts the first 1–3 at
  once. It pauses them for 57–320 ms while the rest prefill, on the 8B / 30B files. The baselines start all eight
  together.
- **The streams behind the 30K prompt** on every lock but the 27B's. Those locks do not mix decoding into prefill
  (`chunk_with_decode` 0, `docs/release.md` §16.5), so on the bf16 8B even TPOT falls to 0.82× on the RTX 4090.
- **First tokens on the NVFP4 files against SGLang:** 0.22–0.69× on the RTX 5080's 8B and the RTX 4090's 30B. The
  5080's 8B NVFP4 also decodes at 0.90× SGLang's speed at batch 8.
- **Smaller losses** on FP8 and NVFP4 (0.89–0.98×), in the tables.

rtx-3090: Holon's advantage (> 1: Holon faster or higher)

| record | model | trace (sampling, prompts; max seqs) | Holon served vs | TTFT p50 | TPOT p50 | max gap p99 | out tok/s | paired rounds |
|---|---|---|---|---|---|---|---|---|
| [release-qwen3.8-27b-ud-q4_k_m-mtp](evidence/bench_release-qwen3.8-27b-ud-q4_k_m-mtp_rtx-3090_2026-10-16.json) (KV 73088 tokens) | unsloth/Qwen3.8-27B-GGUF@4ca720788d1e/Qwen3.8-27B-UD-Q4_K_M.gguf | decode_b1 (greedy, v1-qwen3_5.jsonl; 4) | holon_http@mtp vs llamaserver@mtp3 | 2.87 | 1.05 | 1.20 | 1.16 | 4/4 |
| [release-qwen3.8-27b-ud-q4_k_m-mtp](evidence/bench_release-qwen3.8-27b-ud-q4_k_m-mtp_rtx-3090_2026-10-16.json) (KV 73088 tokens) | unsloth/Qwen3.8-27B-GGUF@4ca720788d1e/Qwen3.8-27B-UD-Q4_K_M.gguf | decode_b8 (greedy, v1-qwen3_5.jsonl; 4) | holon_http@mtp vs llamaserver@mtp3 | 1.62 | 1.29 | ≈ 1.97 | 1.39 | 4/4 |
| [release-qwen3.8-27b-ud-q4_k_m-mtp](evidence/bench_release-qwen3.8-27b-ud-q4_k_m-mtp_rtx-3090_2026-10-16.json) (KV 73088 tokens) | unsloth/Qwen3.8-27B-GGUF@4ca720788d1e/Qwen3.8-27B-UD-Q4_K_M.gguf | agent (greedy, v1-qwen3_5.jsonl; 4) | holon_http@mtp vs llamaserver@mtp3 | 1.51 | 1.31 | 6.74 | 1.22 | 4/4 |
| [release-qwen3.8-27b-ud-q4_k_m-mtp](evidence/bench_release-qwen3.8-27b-ud-q4_k_m-mtp_rtx-3090_2026-10-16.json) (KV 73088 tokens) | unsloth/Qwen3.8-27B-GGUF@4ca720788d1e/Qwen3.8-27B-UD-Q4_K_M.gguf | long_prompt (greedy, v1-qwen3_5.jsonl; 4) | holon_http@mtp vs llamaserver@mtp3 | ≈ 20.83 | ≈ 0.98 | 4.06 | 1.30 | 4/4 |
| [release-qwen3.8-27b-ud-q4_k_m](evidence/bench_release-qwen3.8-27b-ud-q4_k_m_rtx-3090_2026-10-15.json) (KV 86560 tokens) | unsloth/Qwen3.8-27B-GGUF@4ca720788d1e/Qwen3.8-27B-UD-Q4_K_M.gguf | decode_b1 (greedy, v1-qwen3_5.jsonl; 4) | holon_http vs llamaserver | 2.56 | 1.08 | 3.19 | 1.12 | 4/4 |
| [release-qwen3.8-27b-ud-q4_k_m](evidence/bench_release-qwen3.8-27b-ud-q4_k_m_rtx-3090_2026-10-15.json) (KV 86560 tokens) | unsloth/Qwen3.8-27B-GGUF@4ca720788d1e/Qwen3.8-27B-UD-Q4_K_M.gguf | decode_b8 (greedy, v1-qwen3_5.jsonl; 4) | holon_http vs llamaserver | 1.56 | 1.22 | 0.35 | 1.36 | 4/4 |
| [release-qwen3.8-27b-ud-q4_k_m](evidence/bench_release-qwen3.8-27b-ud-q4_k_m_rtx-3090_2026-10-15.json) (KV 86560 tokens) | unsloth/Qwen3.8-27B-GGUF@4ca720788d1e/Qwen3.8-27B-UD-Q4_K_M.gguf | agent (greedy, v1-qwen3_5.jsonl; 4) | holon_http vs llamaserver | 1.57 | 1.44 | 7.43 | 1.25 | 4/4 |
| [release-qwen3.8-27b-ud-q4_k_m](evidence/bench_release-qwen3.8-27b-ud-q4_k_m_rtx-3090_2026-10-15.json) (KV 86560 tokens) | unsloth/Qwen3.8-27B-GGUF@4ca720788d1e/Qwen3.8-27B-UD-Q4_K_M.gguf | long_prompt (greedy, v1-qwen3_5.jsonl; 4) | holon_http vs llamaserver | 30.57 | ≈ 0.96 | 4.05 | 1.19 | 4/4 |
| [release-qwen3-8b-bf16](evidence/bench_release-qwen3-8b-bf16_rtx-3090_2026-10-16.json) (KV 43360 tokens) | Qwen/Qwen3-8B@b968826d9c46 | decode_b1 (greedy, v1.jsonl; 64) | holon_http vs sglang_http | 1.26 | 1.00 | ≈ 1.00 | 1.01 | 4/4 |
| [release-qwen3-8b-bf16](evidence/bench_release-qwen3-8b-bf16_rtx-3090_2026-10-16.json) (KV 43360 tokens) | Qwen/Qwen3-8B@b968826d9c46 | decode_b8 (greedy, v1.jsonl; 64) | holon_http vs sglang_http | 0.98 | 1.01 | 0.06 | 1.01 | 4/4 |
| [release-qwen3-8b-bf16](evidence/bench_release-qwen3-8b-bf16_rtx-3090_2026-10-16.json) (KV 43360 tokens) | Qwen/Qwen3-8B@b968826d9c46 | agent (greedy, v1.jsonl; 64) | holon_http vs sglang_http | ≈ 0.99 | 0.99 | 1.07 | 1.00 | 4/4 |
| [release-qwen3-8b-bf16](evidence/bench_release-qwen3-8b-bf16_rtx-3090_2026-10-16.json) (KV 43360 tokens) | Qwen/Qwen3-8B@b968826d9c46 | long_prompt (greedy, v1.jsonl; 64) | holon_http vs sglang_http | ≈ 1.23 | 1.01 | 1.02 | 1.01 | 4/4 |
| [release-qwen3-8b-bf16](evidence/bench_release-qwen3-8b-bf16_rtx-3090_2026-10-16.json) (KV 43360 tokens) | Qwen/Qwen3-8B@b968826d9c46 | decode_b1 (greedy, v1.jsonl; 64) | holon_http vs freetoken_http | ≈ 1.01 | 1.01 | 1.02 | 1.01 | 4/4 |
| [release-qwen3-8b-bf16](evidence/bench_release-qwen3-8b-bf16_rtx-3090_2026-10-16.json) (KV 43360 tokens) | Qwen/Qwen3-8B@b968826d9c46 | decode_b8 (greedy, v1.jsonl; 64) | holon_http vs freetoken_http | ≈ 1.00 | 1.01 | ≈ 1.03 | 1.01 | 4/4 |
| [release-qwen3-8b-bf16](evidence/bench_release-qwen3-8b-bf16_rtx-3090_2026-10-16.json) (KV 43360 tokens) | Qwen/Qwen3-8B@b968826d9c46 | agent (greedy, v1.jsonl; 64) | holon_http vs freetoken_http | 1.12 | 1.01 | 1.10 | 1.00 | 4/4 |
| [release-qwen3-8b-bf16](evidence/bench_release-qwen3-8b-bf16_rtx-3090_2026-10-16.json) (KV 43360 tokens) | Qwen/Qwen3-8B@b968826d9c46 | long_prompt (greedy, v1.jsonl; 64) | holon_http vs freetoken_http | ≈ 1.09 | 1.00 | ≈ 1.00 | 1.00 | 4/4 |
| [release-qwen3-8b-q4_k_m](evidence/bench_release-qwen3-8b-q4_k_m_rtx-3090_2026-10-16.json) (KV 112912 tokens) | unsloth/Qwen3-8B-GGUF@a6adef130ffb/Qwen3-8B-Q4_K_M.gguf | decode_b1 (greedy, v1.jsonl; 48) | holon_http vs llamaserver | 1.59 | 1.14 | 1.32 | 1.15 | 4/4 |
| [release-qwen3-8b-q4_k_m](evidence/bench_release-qwen3-8b-q4_k_m_rtx-3090_2026-10-16.json) (KV 112912 tokens) | unsloth/Qwen3-8B-GGUF@a6adef130ffb/Qwen3-8B-Q4_K_M.gguf | decode_b8 (greedy, v1.jsonl; 48) | holon_http vs llamaserver | 1.84 | 1.24 | 0.16 | 1.30 | 4/4 |
| [release-qwen3-8b-q4_k_m](evidence/bench_release-qwen3-8b-q4_k_m_rtx-3090_2026-10-16.json) (KV 112912 tokens) | unsloth/Qwen3-8B-GGUF@a6adef130ffb/Qwen3-8B-Q4_K_M.gguf | agent (greedy, v1.jsonl; 48) | holon_http vs llamaserver | 3.82 | 1.50 | 5.79 | 1.04 | 4/4 |
| [release-qwen3-8b-q4_k_m](evidence/bench_release-qwen3-8b-q4_k_m_rtx-3090_2026-10-16.json) (KV 112912 tokens) | unsloth/Qwen3-8B-GGUF@a6adef130ffb/Qwen3-8B-Q4_K_M.gguf | long_prompt (greedy, v1.jsonl; 48) | holon_http vs llamaserver | 22.18 | 1.32 | 0.10 | 1.47 | 4/4 |

rtx-4090: Holon's advantage (> 1: Holon faster or higher)

| record | model | trace (sampling, prompts; max seqs) | Holon served vs | TTFT p50 | TPOT p50 | max gap p99 | out tok/s | paired rounds |
|---|---|---|---|---|---|---|---|---|
| [release-qwen3-30b-a3b-nvfp4](evidence/bench_release-qwen3-30b-a3b-nvfp4_rtx-4090_2026-10-16.json) (KV 34960 tokens) | nvidia/Qwen3-30B-A3B-NVFP4@2538ded2a4ed | decode_b1 (greedy, v1.jsonl; 32) | holon_http vs sglang_http | 0.69 | 1.14 | 1.14 | 1.12 | 3/3 |
| [release-qwen3-30b-a3b-nvfp4](evidence/bench_release-qwen3-30b-a3b-nvfp4_rtx-4090_2026-10-16.json) (KV 34960 tokens) | nvidia/Qwen3-30B-A3B-NVFP4@2538ded2a4ed | decode_b8 (greedy, v1.jsonl; 32) | holon_http vs sglang_http | 0.48 | 1.03 | 0.24 | 0.98 | 3/3 |
| [release-qwen3-30b-a3b-nvfp4](evidence/bench_release-qwen3-30b-a3b-nvfp4_rtx-4090_2026-10-16.json) (KV 34960 tokens) | nvidia/Qwen3-30B-A3B-NVFP4@2538ded2a4ed | decode_b1 (greedy, v1.jsonl; 32) | holon_http vs freetoken_http | — (exit code 1: [2026-10-16\|04:10:01\|FrontendAPI] ERROR    Backend supervisor: WeightLoadError: RuntimeError: chunk expects at least a 1-dimensional tensor) |  |  |  |  |
| [release-qwen3-30b-a3b-nvfp4](evidence/bench_release-qwen3-30b-a3b-nvfp4_rtx-4090_2026-10-16.json) (KV 34960 tokens) | nvidia/Qwen3-30B-A3B-NVFP4@2538ded2a4ed | decode_b8 (greedy, v1.jsonl; 32) | holon_http vs freetoken_http | — (exit code 1: [2026-10-16\|04:10:01\|FrontendAPI] ERROR    Backend supervisor: WeightLoadError: RuntimeError: chunk expects at least a 1-dimensional tensor) |  |  |  |  |
| [release-qwen3.8-27b-ud-q4_k_m-mtp](evidence/bench_release-qwen3.8-27b-ud-q4_k_m-mtp_rtx-4090_2026-10-16.json) (KV 49072 tokens) | unsloth/Qwen3.8-27B-GGUF@4ca720788d1e/Qwen3.8-27B-UD-Q4_K_M.gguf | decode_b1 (greedy, v1-qwen3_5.jsonl; 4) | holon_http@mtp vs llamaserver@mtp3 | 4.39 | 1.09 | 1.29 | 1.22 | 4/4 |
| [release-qwen3.8-27b-ud-q4_k_m-mtp](evidence/bench_release-qwen3.8-27b-ud-q4_k_m-mtp_rtx-4090_2026-10-16.json) (KV 49072 tokens) | unsloth/Qwen3.8-27B-GGUF@4ca720788d1e/Qwen3.8-27B-UD-Q4_K_M.gguf | decode_b8 (greedy, v1-qwen3_5.jsonl; 4) | holon_http@mtp vs llamaserver@mtp3 | 1.68 | 1.29 | 2.77 | 1.42 | 4/4 |
| [release-qwen3.8-27b-ud-q4_k_m-mtp](evidence/bench_release-qwen3.8-27b-ud-q4_k_m-mtp_rtx-4090_2026-10-16.json) (KV 49072 tokens) | unsloth/Qwen3.8-27B-GGUF@4ca720788d1e/Qwen3.8-27B-UD-Q4_K_M.gguf | agent (greedy, v1-qwen3_5.jsonl; 4) | holon_http@mtp vs llamaserver@mtp3 | 5.02 | 1.93 | 4.72 | 1.17 | 4/4 |
| [release-qwen3.8-27b-ud-q4_k_m-mtp](evidence/bench_release-qwen3.8-27b-ud-q4_k_m-mtp_rtx-4090_2026-10-16.json) (KV 49072 tokens) | unsloth/Qwen3.8-27B-GGUF@4ca720788d1e/Qwen3.8-27B-UD-Q4_K_M.gguf | long_prompt (greedy, v1-qwen3_5.jsonl; 4) | holon_http@mtp vs llamaserver@mtp3 | ≈ 14.40 | 1.09 | 3.83 | 1.32 | 4/4 |
| [release-qwen3.8-27b-ud-q4_k_m](evidence/bench_release-qwen3.8-27b-ud-q4_k_m_rtx-4090_2026-10-15.json) (KV 61312 tokens) | unsloth/Qwen3.8-27B-GGUF@4ca720788d1e/Qwen3.8-27B-UD-Q4_K_M.gguf | decode_b1 (greedy, v1-qwen3_5.jsonl; 4) | holon_http vs llamaserver | 3.84 | 1.07 | 3.18 | 1.12 | 4/4 |
| [release-qwen3.8-27b-ud-q4_k_m](evidence/bench_release-qwen3.8-27b-ud-q4_k_m_rtx-4090_2026-10-15.json) (KV 61312 tokens) | unsloth/Qwen3.8-27B-GGUF@4ca720788d1e/Qwen3.8-27B-UD-Q4_K_M.gguf | decode_b8 (greedy, v1-qwen3_5.jsonl; 4) | holon_http vs llamaserver | 1.69 | 1.23 | 0.63 | 1.40 | 4/4 |
| [release-qwen3.8-27b-ud-q4_k_m](evidence/bench_release-qwen3.8-27b-ud-q4_k_m_rtx-4090_2026-10-15.json) (KV 61312 tokens) | unsloth/Qwen3.8-27B-GGUF@4ca720788d1e/Qwen3.8-27B-UD-Q4_K_M.gguf | agent (greedy, v1-qwen3_5.jsonl; 4) | holon_http vs llamaserver | 9.29 | 1.77 | 11.58 | 1.30 | 4/4 |
| [release-qwen3.8-27b-ud-q4_k_m](evidence/bench_release-qwen3.8-27b-ud-q4_k_m_rtx-4090_2026-10-15.json) (KV 61312 tokens) | unsloth/Qwen3.8-27B-GGUF@4ca720788d1e/Qwen3.8-27B-UD-Q4_K_M.gguf | long_prompt (greedy, v1-qwen3_5.jsonl; 4) | holon_http vs llamaserver | 40.77 | ≈ 1.04 | ≈ 2.64 | 1.33 | 4/4 |
| [release-qwen3-8b-bf16](evidence/bench_release-qwen3-8b-bf16_rtx-4090_2026-10-16.json) (KV 32528 tokens) | Qwen/Qwen3-8B@b968826d9c46 | decode_b1 (greedy, v1.jsonl; 64) | holon_http vs sglang_http | 1.53 | 1.00 | 1.01 | 1.01 | 4/4 |
| [release-qwen3-8b-bf16](evidence/bench_release-qwen3-8b-bf16_rtx-4090_2026-10-16.json) (KV 32528 tokens) | Qwen/Qwen3-8B@b968826d9c46 | decode_b8 (greedy, v1.jsonl; 64) | holon_http vs sglang_http | 1.04 | 1.01 | 0.14 | 1.01 | 4/4 |
| [release-qwen3-8b-bf16](evidence/bench_release-qwen3-8b-bf16_rtx-4090_2026-10-16.json) (KV 32528 tokens) | Qwen/Qwen3-8B@b968826d9c46 | agent (greedy, v1.jsonl; 64) | holon_http vs sglang_http | 1.02 | 1.01 | 1.08 | 1.00 | 4/4 |
| [release-qwen3-8b-bf16](evidence/bench_release-qwen3-8b-bf16_rtx-4090_2026-10-16.json) (KV 32528 tokens) | Qwen/Qwen3-8B@b968826d9c46 | long_prompt (greedy, v1.jsonl; 64) | holon_http vs sglang_http | 1.15 | 0.82 | 0.01 | 1.03 | 4/4 |
| [release-qwen3-8b-bf16](evidence/bench_release-qwen3-8b-bf16_rtx-4090_2026-10-16.json) (KV 32528 tokens) | Qwen/Qwen3-8B@b968826d9c46 | decode_b1 (greedy, v1.jsonl; 64) | holon_http vs freetoken_http | 1.03 | 1.01 | 1.02 | 1.01 | 4/4 |
| [release-qwen3-8b-bf16](evidence/bench_release-qwen3-8b-bf16_rtx-4090_2026-10-16.json) (KV 32528 tokens) | Qwen/Qwen3-8B@b968826d9c46 | decode_b8 (greedy, v1.jsonl; 64) | holon_http vs freetoken_http | ≈ 1.00 | 1.01 | 1.08 | 1.01 | 4/4 |
| [release-qwen3-8b-bf16](evidence/bench_release-qwen3-8b-bf16_rtx-4090_2026-10-16.json) (KV 32528 tokens) | Qwen/Qwen3-8B@b968826d9c46 | agent (greedy, v1.jsonl; 64) | holon_http vs freetoken_http | 1.09 | 1.01 | 1.12 | 1.00 | 4/4 |
| [release-qwen3-8b-bf16](evidence/bench_release-qwen3-8b-bf16_rtx-4090_2026-10-16.json) (KV 32528 tokens) | Qwen/Qwen3-8B@b968826d9c46 | long_prompt (greedy, v1.jsonl; 64) | holon_http vs freetoken_http | ≈ 1.00 | 0.82 | 0.02 | 1.03 | 4/4 |
| [release-qwen3-8b-fp8](evidence/bench_release-qwen3-8b-fp8_rtx-4090_2026-10-16.json) (KV 78176 tokens) | Qwen/Qwen3-8B-FP8@220b46e3b218 | decode_b1 (greedy, v1.jsonl; 32) | holon_http vs sglang_http | 1.37 | 1.39 | 1.40 | 1.39 | 3/3 |
| [release-qwen3-8b-fp8](evidence/bench_release-qwen3-8b-fp8_rtx-4090_2026-10-16.json) (KV 78176 tokens) | Qwen/Qwen3-8B-FP8@220b46e3b218 | decode_b8 (greedy, v1.jsonl; 32) | holon_http vs sglang_http | 0.97 | 1.16 | 0.18 | 1.15 | 3/3 |
| [release-qwen3-8b-fp8](evidence/bench_release-qwen3-8b-fp8_rtx-4090_2026-10-16.json) (KV 78176 tokens) | Qwen/Qwen3-8B-FP8@220b46e3b218 | decode_b1 (greedy, v1.jsonl; 32) | holon_http vs freetoken_http | ≈ 1.01 | 1.06 | 1.08 | 1.06 | 3/3 |
| [release-qwen3-8b-fp8](evidence/bench_release-qwen3-8b-fp8_rtx-4090_2026-10-16.json) (KV 78176 tokens) | Qwen/Qwen3-8B-FP8@220b46e3b218 | decode_b8 (greedy, v1.jsonl; 32) | holon_http vs freetoken_http | 0.96 | 1.04 | 0.94 | 1.03 | 3/3 |
| [release-qwen3-8b-nvfp4](evidence/bench_release-qwen3-8b-nvfp4_rtx-4090_2026-10-16.json) (KV 95504 tokens) | nvidia/Qwen3-8B-NVFP4@ccd10a893cbc | decode_b1 (greedy, v1.jsonl; 32) | holon_http vs sglang_http | 1.32 | 1.09 | 1.09 | 1.09 | 3/3 |
| [release-qwen3-8b-nvfp4](evidence/bench_release-qwen3-8b-nvfp4_rtx-4090_2026-10-16.json) (KV 95504 tokens) | nvidia/Qwen3-8B-NVFP4@ccd10a893cbc | decode_b8 (greedy, v1.jsonl; 32) | holon_http vs sglang_http | ≈ 1.01 | ≈ 1.00 | 0.27 | ≈ 1.00 | 3/3 |
| [release-qwen3-8b-nvfp4](evidence/bench_release-qwen3-8b-nvfp4_rtx-4090_2026-10-16.json) (KV 95504 tokens) | nvidia/Qwen3-8B-NVFP4@ccd10a893cbc | decode_b1 (greedy, v1.jsonl; 32) | holon_http vs freetoken_http | — (exit code 1: [2026-10-16\|03:55:24\|FrontendAPI] ERROR    Backend supervisor: WeightLoadError: RuntimeError: chunk expects at least a 1-dimensional tensor) |  |  |  |  |
| [release-qwen3-8b-nvfp4](evidence/bench_release-qwen3-8b-nvfp4_rtx-4090_2026-10-16.json) (KV 95504 tokens) | nvidia/Qwen3-8B-NVFP4@ccd10a893cbc | decode_b8 (greedy, v1.jsonl; 32) | holon_http vs freetoken_http | — (exit code 1: [2026-10-16\|03:55:24\|FrontendAPI] ERROR    Backend supervisor: WeightLoadError: RuntimeError: chunk expects at least a 1-dimensional tensor) |  |  |  |  |
| [release-qwen3-8b-q4_k_m](evidence/bench_release-qwen3-8b-q4_k_m_rtx-4090_2026-10-16.json) (KV 102080 tokens) | unsloth/Qwen3-8B-GGUF@a6adef130ffb/Qwen3-8B-Q4_K_M.gguf | decode_b1 (greedy, v1.jsonl; 48) | holon_http vs llamaserver | 2.02 | 1.11 | 1.29 | 1.12 | 4/4 |
| [release-qwen3-8b-q4_k_m](evidence/bench_release-qwen3-8b-q4_k_m_rtx-4090_2026-10-16.json) (KV 102080 tokens) | unsloth/Qwen3-8B-GGUF@a6adef130ffb/Qwen3-8B-Q4_K_M.gguf | decode_b8 (greedy, v1.jsonl; 48) | holon_http vs llamaserver | 2.69 | 1.55 | 0.25 | 1.63 | 4/4 |
| [release-qwen3-8b-q4_k_m](evidence/bench_release-qwen3-8b-q4_k_m_rtx-4090_2026-10-16.json) (KV 102080 tokens) | unsloth/Qwen3-8B-GGUF@a6adef130ffb/Qwen3-8B-Q4_K_M.gguf | agent (greedy, v1.jsonl; 48) | holon_http vs llamaserver | 4.02 | 1.43 | 9.28 | 1.04 | 4/4 |
| [release-qwen3-8b-q4_k_m](evidence/bench_release-qwen3-8b-q4_k_m_rtx-4090_2026-10-16.json) (KV 102080 tokens) | unsloth/Qwen3-8B-GGUF@a6adef130ffb/Qwen3-8B-Q4_K_M.gguf | long_prompt (greedy, v1.jsonl; 48) | holon_http vs llamaserver | 43.26 | 1.65 | 0.09 | 1.90 | 4/4 |

rtx-5080: Holon's advantage (> 1: Holon faster or higher)

| record | model | trace (sampling, prompts; max seqs) | Holon served vs | TTFT p50 | TPOT p50 | max gap p99 | out tok/s | paired rounds |
|---|---|---|---|---|---|---|---|---|
| [release-qwen3.8-27b-ud-iq3_s-mtp](evidence/bench_release-qwen3.8-27b-ud-iq3_s-mtp_rtx-5080_2026-10-16.json) (KV 37344 tokens) | unsloth/Qwen3.8-27B-GGUF@4ca720788d1e/Qwen3.8-27B-UD-IQ3_S.gguf | decode_b1 (greedy, v1-qwen3_5.jsonl; 4) | holon_http@mtp vs llamaserver@mtp3 | 3.04 | 1.10 | 2.13 | 1.21 | 4/4 |
| [release-qwen3.8-27b-ud-iq3_s-mtp](evidence/bench_release-qwen3.8-27b-ud-iq3_s-mtp_rtx-5080_2026-10-16.json) (KV 37344 tokens) | unsloth/Qwen3.8-27B-GGUF@4ca720788d1e/Qwen3.8-27B-UD-IQ3_S.gguf | decode_b8 (greedy, v1-qwen3_5.jsonl; 4) | holon_http@mtp vs llamaserver@mtp3 | 1.50 | 1.20 | 1.98 | 1.40 | 4/4 |
| [release-qwen3.8-27b-ud-iq3_s-mtp](evidence/bench_release-qwen3.8-27b-ud-iq3_s-mtp_rtx-5080_2026-10-16.json) (KV 37344 tokens) | unsloth/Qwen3.8-27B-GGUF@4ca720788d1e/Qwen3.8-27B-UD-IQ3_S.gguf | agent (greedy, v1-qwen3_5.jsonl; 4) | holon_http@mtp vs llamaserver@mtp3 | 2.47 | 1.39 | 4.09 | 1.08 | 4/4 |
| [release-qwen3.8-27b-ud-iq3_s-mtp](evidence/bench_release-qwen3.8-27b-ud-iq3_s-mtp_rtx-5080_2026-10-16.json) (KV 37344 tokens) | unsloth/Qwen3.8-27B-GGUF@4ca720788d1e/Qwen3.8-27B-UD-IQ3_S.gguf | long_prompt (greedy, v1-qwen3_5.jsonl; 4) | holon_http@mtp vs llamaserver@mtp3 | 5.14 | 1.10 | 3.62 | 1.31 | 4/4 |
| [release-qwen3.8-27b-ud-iq3_s](evidence/bench_release-qwen3.8-27b-ud-iq3_s_rtx-5080_2026-10-16.json) (KV 56208 tokens) | unsloth/Qwen3.8-27B-GGUF@4ca720788d1e/Qwen3.8-27B-UD-IQ3_S.gguf | decode_b1 (greedy, v1-qwen3_5.jsonl; 4) | holon_http vs llamaserver | 2.71 | 1.09 | 3.23 | 1.13 | 4/4 |
| [release-qwen3.8-27b-ud-iq3_s](evidence/bench_release-qwen3.8-27b-ud-iq3_s_rtx-5080_2026-10-16.json) (KV 56208 tokens) | unsloth/Qwen3.8-27B-GGUF@4ca720788d1e/Qwen3.8-27B-UD-IQ3_S.gguf | decode_b8 (greedy, v1-qwen3_5.jsonl; 4) | holon_http vs llamaserver | 1.57 | 1.20 | 0.35 | 1.34 | 4/4 |
| [release-qwen3.8-27b-ud-iq3_s](evidence/bench_release-qwen3.8-27b-ud-iq3_s_rtx-5080_2026-10-16.json) (KV 56208 tokens) | unsloth/Qwen3.8-27B-GGUF@4ca720788d1e/Qwen3.8-27B-UD-IQ3_S.gguf | agent (greedy, v1-qwen3_5.jsonl; 4) | holon_http vs llamaserver | 4.94 | 1.53 | 6.30 | 1.25 | 4/4 |
| [release-qwen3.8-27b-ud-iq3_s](evidence/bench_release-qwen3.8-27b-ud-iq3_s_rtx-5080_2026-10-16.json) (KV 56208 tokens) | unsloth/Qwen3.8-27B-GGUF@4ca720788d1e/Qwen3.8-27B-UD-IQ3_S.gguf | long_prompt (greedy, v1-qwen3_5.jsonl; 4) | holon_http vs llamaserver | 4.72 | 1.23 | 3.61 | 1.26 | 4/4 |
| [release-qwen3-8b-fp8](evidence/bench_release-qwen3-8b-fp8_rtx-5080_2026-10-16.json) (KV 35536 tokens) | Qwen/Qwen3-8B-FP8@220b46e3b218 | decode_b1 (greedy, v1.jsonl; 32) | holon_http vs sglang_http | 1.19 | 1.07 | 1.06 | 1.07 | 3/3 |
| [release-qwen3-8b-fp8](evidence/bench_release-qwen3-8b-fp8_rtx-5080_2026-10-16.json) (KV 35536 tokens) | Qwen/Qwen3-8B-FP8@220b46e3b218 | decode_b8 (greedy, v1.jsonl; 32) | holon_http vs sglang_http | 0.89 | 1.03 | 0.10 | 1.02 | 3/3 |
| [release-qwen3-8b-fp8](evidence/bench_release-qwen3-8b-fp8_rtx-5080_2026-10-16.json) (KV 35536 tokens) | Qwen/Qwen3-8B-FP8@220b46e3b218 | decode_b1 (greedy, v1.jsonl; 32) | holon_http vs freetoken_http | 0.97 | 1.02 | 1.02 | 1.02 | 3/3 |
| [release-qwen3-8b-fp8](evidence/bench_release-qwen3-8b-fp8_rtx-5080_2026-10-16.json) (KV 35536 tokens) | Qwen/Qwen3-8B-FP8@220b46e3b218 | decode_b8 (greedy, v1.jsonl; 32) | holon_http vs freetoken_http | 0.94 | 1.10 | 1.10 | 1.09 | 3/3 |
| [release-qwen3-8b-nvfp4](evidence/bench_release-qwen3-8b-nvfp4_rtx-5080_2026-10-16.json) (KV 53024 tokens) | nvidia/Qwen3-8B-NVFP4@ccd10a893cbc | decode_b1 (greedy, v1.jsonl; 32) | holon_http vs sglang_http | 0.48 | 1.05 | 1.05 | 1.04 | 3/3 |
| [release-qwen3-8b-nvfp4](evidence/bench_release-qwen3-8b-nvfp4_rtx-5080_2026-10-16.json) (KV 53024 tokens) | nvidia/Qwen3-8B-NVFP4@ccd10a893cbc | decode_b8 (greedy, v1.jsonl; 32) | holon_http vs sglang_http | 0.22 | 0.90 | 0.06 | 0.82 | 3/3 |
| [release-qwen3-8b-nvfp4](evidence/bench_release-qwen3-8b-nvfp4_rtx-5080_2026-10-16.json) (KV 53024 tokens) | nvidia/Qwen3-8B-NVFP4@ccd10a893cbc | decode_b1 (greedy, v1.jsonl; 32) | holon_http vs freetoken_http | — (exit code 1: [2026-10-16\|09:00:04\|FrontendAPI] ERROR    Backend supervisor: WeightLoadError: RuntimeError: chunk expects at least a 1-dimensional tensor) |  |  |  |  |
| [release-qwen3-8b-nvfp4](evidence/bench_release-qwen3-8b-nvfp4_rtx-5080_2026-10-16.json) (KV 53024 tokens) | nvidia/Qwen3-8B-NVFP4@ccd10a893cbc | decode_b8 (greedy, v1.jsonl; 32) | holon_http vs freetoken_http | — (exit code 1: [2026-10-16\|09:00:04\|FrontendAPI] ERROR    Backend supervisor: WeightLoadError: RuntimeError: chunk expects at least a 1-dimensional tensor) |  |  |  |  |
| [release-qwen3-8b-q4_k_m](evidence/bench_release-qwen3-8b-q4_k_m_rtx-5080_2026-10-16.json) (KV 59296 tokens) | unsloth/Qwen3-8B-GGUF@a6adef130ffb/Qwen3-8B-Q4_K_M.gguf | decode_b1 (greedy, v1.jsonl; 48) | holon_http vs llamaserver | 1.61 | 1.09 | 1.18 | 1.10 | 4/4 |
| [release-qwen3-8b-q4_k_m](evidence/bench_release-qwen3-8b-q4_k_m_rtx-5080_2026-10-16.json) (KV 59296 tokens) | unsloth/Qwen3-8B-GGUF@a6adef130ffb/Qwen3-8B-Q4_K_M.gguf | decode_b8 (greedy, v1.jsonl; 48) | holon_http vs llamaserver | 1.89 | 1.41 | 0.15 | 1.46 | 4/4 |
| [release-qwen3-8b-q4_k_m](evidence/bench_release-qwen3-8b-q4_k_m_rtx-5080_2026-10-16.json) (KV 59296 tokens) | unsloth/Qwen3-8B-GGUF@a6adef130ffb/Qwen3-8B-Q4_K_M.gguf | agent (greedy, v1.jsonl; 48) | holon_http vs llamaserver | 3.53 | 1.35 | 6.80 | 1.03 | 4/4 |
| [release-qwen3-8b-q4_k_m](evidence/bench_release-qwen3-8b-q4_k_m_rtx-5080_2026-10-16.json) (KV 59296 tokens) | unsloth/Qwen3-8B-GGUF@a6adef130ffb/Qwen3-8B-Q4_K_M.gguf | long_prompt (greedy, v1.jsonl; 48) | holon_http vs llamaserver | 21.72 | 1.50 | 0.14 | 1.64 | 4/4 |

rtx-5090: Holon's advantage (> 1: Holon faster or higher)

| record | model | trace (sampling, prompts; max seqs) | Holon served vs | TTFT p50 | TPOT p50 | max gap p99 | out tok/s | paired rounds |
|---|---|---|---|---|---|---|---|---|
| [release-qwen3.8-27b-ud-q4_k_m-mtp](evidence/bench_release-qwen3.8-27b-ud-q4_k_m-mtp_rtx-5090_2026-10-16.json) (KV 175264 tokens) | unsloth/Qwen3.8-27B-GGUF@4ca720788d1e/Qwen3.8-27B-UD-Q4_K_M.gguf | decode_b1 (greedy, v1-qwen3_5.jsonl; 4) | holon_http@mtp vs llamaserver@mtp3 | 4.63 | 1.12 | 1.27 | 1.30 | 4/4 |
| [release-qwen3.8-27b-ud-q4_k_m-mtp](evidence/bench_release-qwen3.8-27b-ud-q4_k_m-mtp_rtx-5090_2026-10-16.json) (KV 175264 tokens) | unsloth/Qwen3.8-27B-GGUF@4ca720788d1e/Qwen3.8-27B-UD-Q4_K_M.gguf | decode_b8 (greedy, v1-qwen3_5.jsonl; 4) | holon_http@mtp vs llamaserver@mtp3 | 1.86 | 1.44 | 2.95 | 1.58 | 4/4 |
| [release-qwen3.8-27b-ud-q4_k_m-mtp](evidence/bench_release-qwen3.8-27b-ud-q4_k_m-mtp_rtx-5090_2026-10-16.json) (KV 175264 tokens) | unsloth/Qwen3.8-27B-GGUF@4ca720788d1e/Qwen3.8-27B-UD-Q4_K_M.gguf | agent (greedy, v1-qwen3_5.jsonl; 4) | holon_http@mtp vs llamaserver@mtp3 | 3.72 | 1.66 | 11.00 | 1.05 | 4/4 |
| [release-qwen3.8-27b-ud-q4_k_m-mtp](evidence/bench_release-qwen3.8-27b-ud-q4_k_m-mtp_rtx-5090_2026-10-16.json) (KV 175264 tokens) | unsloth/Qwen3.8-27B-GGUF@4ca720788d1e/Qwen3.8-27B-UD-Q4_K_M.gguf | long_prompt (greedy, v1-qwen3_5.jsonl; 4) | holon_http@mtp vs llamaserver@mtp3 | 10.61 | 1.17 | 4.00 | 1.47 | 4/4 |
| [release-qwen3.8-27b-ud-q4_k_m](evidence/bench_release-qwen3.8-27b-ud-q4_k_m_rtx-5090_2026-10-15.json) (KV 195056 tokens) | unsloth/Qwen3.8-27B-GGUF@4ca720788d1e/Qwen3.8-27B-UD-Q4_K_M.gguf | decode_b1 (greedy, v1-qwen3_5.jsonl; 4) | holon_http vs llamaserver | 3.87 | 1.10 | 3.24 | 1.16 | 4/4 |
| [release-qwen3.8-27b-ud-q4_k_m](evidence/bench_release-qwen3.8-27b-ud-q4_k_m_rtx-5090_2026-10-15.json) (KV 195056 tokens) | unsloth/Qwen3.8-27B-GGUF@4ca720788d1e/Qwen3.8-27B-UD-Q4_K_M.gguf | decode_b8 (greedy, v1-qwen3_5.jsonl; 4) | holon_http vs llamaserver | 1.79 | 1.22 | 0.44 | 1.44 | 4/4 |
| [release-qwen3.8-27b-ud-q4_k_m](evidence/bench_release-qwen3.8-27b-ud-q4_k_m_rtx-5090_2026-10-15.json) (KV 195056 tokens) | unsloth/Qwen3.8-27B-GGUF@4ca720788d1e/Qwen3.8-27B-UD-Q4_K_M.gguf | agent (greedy, v1-qwen3_5.jsonl; 4) | holon_http vs llamaserver | 2.83 | 1.47 | 13.40 | 1.06 | 4/4 |
| [release-qwen3.8-27b-ud-q4_k_m](evidence/bench_release-qwen3.8-27b-ud-q4_k_m_rtx-5090_2026-10-15.json) (KV 195056 tokens) | unsloth/Qwen3.8-27B-GGUF@4ca720788d1e/Qwen3.8-27B-UD-Q4_K_M.gguf | long_prompt (greedy, v1-qwen3_5.jsonl; 4) | holon_http vs llamaserver | 10.24 | 1.13 | 4.03 | 1.20 | 4/4 |
| [release-qwen3-8b-bf16](evidence/bench_release-qwen3-8b-bf16_rtx-5090_2026-10-16.json) (KV 92464 tokens) | Qwen/Qwen3-8B@b968826d9c46 | decode_b1 (greedy, v1.jsonl; 64) | holon_http vs sglang_http | 1.38 | 1.01 | ≈ 1.01 | 1.01 | 4/4 |
| [release-qwen3-8b-bf16](evidence/bench_release-qwen3-8b-bf16_rtx-5090_2026-10-16.json) (KV 92464 tokens) | Qwen/Qwen3-8B@b968826d9c46 | decode_b8 (greedy, v1.jsonl; 64) | holon_http vs sglang_http | 1.07 | 1.02 | 0.11 | 1.02 | 4/4 |
| [release-qwen3-8b-bf16](evidence/bench_release-qwen3-8b-bf16_rtx-5090_2026-10-16.json) (KV 92464 tokens) | Qwen/Qwen3-8B@b968826d9c46 | agent (greedy, v1.jsonl; 64) | holon_http vs sglang_http | ≈ 0.98 | 1.01 | 1.07 | 1.00 | 4/4 |
| [release-qwen3-8b-bf16](evidence/bench_release-qwen3-8b-bf16_rtx-5090_2026-10-16.json) (KV 92464 tokens) | Qwen/Qwen3-8B@b968826d9c46 | long_prompt (greedy, v1.jsonl; 64) | holon_http vs sglang_http | ≈ 1.09 | 1.02 | 1.02 | 1.02 | 4/4 |
| [release-qwen3-8b-bf16](evidence/bench_release-qwen3-8b-bf16_rtx-5090_2026-10-16.json) (KV 92464 tokens) | Qwen/Qwen3-8B@b968826d9c46 | decode_b1 (greedy, v1.jsonl; 64) | holon_http vs freetoken_http | 1.05 | 1.02 | 1.04 | 1.02 | 4/4 |
| [release-qwen3-8b-bf16](evidence/bench_release-qwen3-8b-bf16_rtx-5090_2026-10-16.json) (KV 92464 tokens) | Qwen/Qwen3-8B@b968826d9c46 | decode_b8 (greedy, v1.jsonl; 64) | holon_http vs freetoken_http | ≈ 1.00 | 1.02 | ≈ 1.07 | 1.02 | 4/4 |
| [release-qwen3-8b-bf16](evidence/bench_release-qwen3-8b-bf16_rtx-5090_2026-10-16.json) (KV 92464 tokens) | Qwen/Qwen3-8B@b968826d9c46 | agent (greedy, v1.jsonl; 64) | holon_http vs freetoken_http | 1.12 | 1.02 | 1.08 | 1.00 | 4/4 |
| [release-qwen3-8b-bf16](evidence/bench_release-qwen3-8b-bf16_rtx-5090_2026-10-16.json) (KV 92464 tokens) | Qwen/Qwen3-8B@b968826d9c46 | long_prompt (greedy, v1.jsonl; 64) | holon_http vs freetoken_http | ≈ 1.06 | 1.02 | 1.00 | 1.02 | 4/4 |
| [release-qwen3-8b-q4_k_m](evidence/bench_release-qwen3-8b-q4_k_m_rtx-5090_2026-10-16.json) (KV 162016 tokens) | unsloth/Qwen3-8B-GGUF@a6adef130ffb/Qwen3-8B-Q4_K_M.gguf | decode_b1 (greedy, v1.jsonl; 48) | holon_http vs llamaserver | 2.24 | 1.15 | 1.24 | 1.18 | 4/4 |
| [release-qwen3-8b-q4_k_m](evidence/bench_release-qwen3-8b-q4_k_m_rtx-5090_2026-10-16.json) (KV 162016 tokens) | unsloth/Qwen3-8B-GGUF@a6adef130ffb/Qwen3-8B-Q4_K_M.gguf | decode_b8 (greedy, v1.jsonl; 48) | holon_http vs llamaserver | 2.80 | 1.66 | 0.24 | 1.76 | 4/4 |
| [release-qwen3-8b-q4_k_m](evidence/bench_release-qwen3-8b-q4_k_m_rtx-5090_2026-10-16.json) (KV 162016 tokens) | unsloth/Qwen3-8B-GGUF@a6adef130ffb/Qwen3-8B-Q4_K_M.gguf | agent (greedy, v1.jsonl; 48) | holon_http vs llamaserver | 5.48 | 1.33 | 12.26 | 1.03 | 4/4 |
| [release-qwen3-8b-q4_k_m](evidence/bench_release-qwen3-8b-q4_k_m_rtx-5090_2026-10-16.json) (KV 162016 tokens) | unsloth/Qwen3-8B-GGUF@a6adef130ffb/Qwen3-8B-Q4_K_M.gguf | long_prompt (greedy, v1.jsonl; 48) | holon_http vs llamaserver | 36.62 | 1.51 | 0.09 | 1.74 | 4/4 |

**Speculation** (the Qwen3.8-27B with MTP; the locks serve without it, `docs/release.md` §16.6): greedy on natural text, each engine's
TPOT against its own plain run on the same desktop. "Inflated by repetition" marks a trace whose output repeats its
own 16-grams more than twice as often as the prompt text does: MTP accepts more there than on ordinary text.

rtx-3090: speculation

| record | trace (sampling, prompts) | backend | TPOT p50 ms | speedup over its plain (TPOT p50) | plain from | accepted / drafted | tokens per verify row | depth-1 share | repeated output |
|---|---|---|---|---|---|---|---|---|---|
| [release-qwen3.8-27b-ud-q4_k_m-mtp](evidence/bench_release-qwen3.8-27b-ud-q4_k_m-mtp_rtx-3090_2026-10-16.json) | decode_b1 (greedy, v1-qwen3_5.jsonl) | holon@mtp | 10.11 | 2.09 | [release-qwen3.8-27b-ud-q4_k_m](evidence/bench_release-qwen3.8-27b-ud-q4_k_m_rtx-3090_2026-10-15.json), unpaired | 0.836 | 3.51 | 1.000 | 0.019 |
| [release-qwen3.8-27b-ud-q4_k_m-mtp](evidence/bench_release-qwen3.8-27b-ud-q4_k_m-mtp_rtx-3090_2026-10-16.json) | decode_b1 (greedy, v1-qwen3_5.jsonl) | holon_http@mtp | 10.07 | 2.09 | [release-qwen3.8-27b-ud-q4_k_m](evidence/bench_release-qwen3.8-27b-ud-q4_k_m_rtx-3090_2026-10-15.json), unpaired | 0.836 | 3.51 | 1.000 | 0.019 |
| [release-qwen3.8-27b-ud-q4_k_m-mtp](evidence/bench_release-qwen3.8-27b-ud-q4_k_m-mtp_rtx-3090_2026-10-16.json) | decode_b1 (greedy, v1-qwen3_5.jsonl) | llamaserver@mtp3 | 10.54 | 2.16 (inflated by repetition) | [release-qwen3.8-27b-ud-q4_k_m](evidence/bench_release-qwen3.8-27b-ud-q4_k_m_rtx-3090_2026-10-15.json), unpaired | 0.859 | — | — | 0.238 |
| [release-qwen3.8-27b-ud-q4_k_m-mtp](evidence/bench_release-qwen3.8-27b-ud-q4_k_m-mtp_rtx-3090_2026-10-16.json) | decode_b8 (greedy, v1-qwen3_5.jsonl) | holon@mtp | 18.04 | 1.64 | [release-qwen3.8-27b-ud-q4_k_m](evidence/bench_release-qwen3.8-27b-ud-q4_k_m_rtx-3090_2026-10-15.json), unpaired | 0.752 | 2.51 | 1.000 | 0.088 |
| [release-qwen3.8-27b-ud-q4_k_m-mtp](evidence/bench_release-qwen3.8-27b-ud-q4_k_m-mtp_rtx-3090_2026-10-16.json) | decode_b8 (greedy, v1-qwen3_5.jsonl) | holon_http@mtp | 18.47 | 1.60 | [release-qwen3.8-27b-ud-q4_k_m](evidence/bench_release-qwen3.8-27b-ud-q4_k_m_rtx-3090_2026-10-15.json), unpaired | 0.747 | 2.51 | 0.999 | 0.088 |
| [release-qwen3.8-27b-ud-q4_k_m-mtp](evidence/bench_release-qwen3.8-27b-ud-q4_k_m-mtp_rtx-3090_2026-10-16.json) | decode_b8 (greedy, v1-qwen3_5.jsonl) | llamaserver@mtp3 | 23.2 | 1.56 | [release-qwen3.8-27b-ud-q4_k_m](evidence/bench_release-qwen3.8-27b-ud-q4_k_m_rtx-3090_2026-10-15.json), unpaired | 0.693 | — | — | 0.108 |
| [release-qwen3.8-27b-ud-q4_k_m-mtp](evidence/bench_release-qwen3.8-27b-ud-q4_k_m-mtp_rtx-3090_2026-10-16.json) | agent (greedy, v1-qwen3_5.jsonl) | holon@mtp | 39.19 | 1.27 | [release-qwen3.8-27b-ud-q4_k_m](evidence/bench_release-qwen3.8-27b-ud-q4_k_m_rtx-3090_2026-10-15.json), unpaired | 0.805 | 2.74 | 1.000 | 0.108 |
| [release-qwen3.8-27b-ud-q4_k_m-mtp](evidence/bench_release-qwen3.8-27b-ud-q4_k_m-mtp_rtx-3090_2026-10-16.json) | agent (greedy, v1-qwen3_5.jsonl) | holon_http@mtp | 39.15 | 1.27 | [release-qwen3.8-27b-ud-q4_k_m](evidence/bench_release-qwen3.8-27b-ud-q4_k_m_rtx-3090_2026-10-15.json), unpaired | 0.800 | 2.73 | 1.000 | 0.099 |
| [release-qwen3.8-27b-ud-q4_k_m-mtp](evidence/bench_release-qwen3.8-27b-ud-q4_k_m-mtp_rtx-3090_2026-10-16.json) | agent (greedy, v1-qwen3_5.jsonl) | llamaserver@mtp3 | 52.81 | 1.34 | [release-qwen3.8-27b-ud-q4_k_m](evidence/bench_release-qwen3.8-27b-ud-q4_k_m_rtx-3090_2026-10-15.json), unpaired | 0.711 | — | — | 0.134 |
| [release-qwen3.8-27b-ud-q4_k_m-mtp](evidence/bench_release-qwen3.8-27b-ud-q4_k_m-mtp_rtx-3090_2026-10-16.json) | long_prompt (greedy, v1-qwen3_5.jsonl) | holon@mtp | 35.76 | 1.38 (inflated by repetition) | [release-qwen3.8-27b-ud-q4_k_m](evidence/bench_release-qwen3.8-27b-ud-q4_k_m_rtx-3090_2026-10-15.json), unpaired | 0.893 | 2.82 | 1.000 | 0.226 |
| [release-qwen3.8-27b-ud-q4_k_m-mtp](evidence/bench_release-qwen3.8-27b-ud-q4_k_m-mtp_rtx-3090_2026-10-16.json) | long_prompt (greedy, v1-qwen3_5.jsonl) | holon_http@mtp | 35.72 | 1.38 (inflated by repetition) | [release-qwen3.8-27b-ud-q4_k_m](evidence/bench_release-qwen3.8-27b-ud-q4_k_m_rtx-3090_2026-10-15.json), unpaired | 0.893 | 2.82 | 1.000 | 0.241 |
| [release-qwen3.8-27b-ud-q4_k_m-mtp](evidence/bench_release-qwen3.8-27b-ud-q4_k_m-mtp_rtx-3090_2026-10-16.json) | long_prompt (greedy, v1-qwen3_5.jsonl) | llamaserver@mtp3 | 37.86 | 1.05 (inflated by repetition) | [release-qwen3.8-27b-ud-q4_k_m](evidence/bench_release-qwen3.8-27b-ud-q4_k_m_rtx-3090_2026-10-15.json), unpaired | 0.766 | — | — | 0.281 |

rtx-4090: speculation

| record | trace (sampling, prompts) | backend | TPOT p50 ms | speedup over its plain (TPOT p50) | plain from | accepted / drafted | tokens per verify row | depth-1 share | repeated output |
|---|---|---|---|---|---|---|---|---|---|
| [release-qwen3.8-27b-ud-q4_k_m-mtp](evidence/bench_release-qwen3.8-27b-ud-q4_k_m-mtp_rtx-4090_2026-10-16.json) | decode_b1 (greedy, v1-qwen3_5.jsonl) | holon@mtp | 8.23 | 2.56 (inflated by repetition) | [release-qwen3.8-27b-ud-q4_k_m](evidence/bench_release-qwen3.8-27b-ud-q4_k_m_rtx-4090_2026-10-15.json), unpaired | 0.878 | 3.63 | 1.000 | 0.238 |
| [release-qwen3.8-27b-ud-q4_k_m-mtp](evidence/bench_release-qwen3.8-27b-ud-q4_k_m-mtp_rtx-4090_2026-10-16.json) | decode_b1 (greedy, v1-qwen3_5.jsonl) | holon_http@mtp | 8.22 | 2.57 (inflated by repetition) | [release-qwen3.8-27b-ud-q4_k_m](evidence/bench_release-qwen3.8-27b-ud-q4_k_m_rtx-4090_2026-10-15.json), unpaired | 0.878 | 3.63 | 1.000 | 0.238 |
| [release-qwen3.8-27b-ud-q4_k_m-mtp](evidence/bench_release-qwen3.8-27b-ud-q4_k_m-mtp_rtx-4090_2026-10-16.json) | decode_b1 (greedy, v1-qwen3_5.jsonl) | llamaserver@mtp3 | 8.93 | 2.54 (inflated by repetition) | [release-qwen3.8-27b-ud-q4_k_m](evidence/bench_release-qwen3.8-27b-ud-q4_k_m_rtx-4090_2026-10-15.json), unpaired | 0.894 | — | — | 0.238 |
| [release-qwen3.8-27b-ud-q4_k_m-mtp](evidence/bench_release-qwen3.8-27b-ud-q4_k_m-mtp_rtx-4090_2026-10-16.json) | decode_b8 (greedy, v1-qwen3_5.jsonl) | holon@mtp | 15.53 | 1.53 | [release-qwen3.8-27b-ud-q4_k_m](evidence/bench_release-qwen3.8-27b-ud-q4_k_m_rtx-4090_2026-10-15.json), unpaired | 0.760 | 2.54 | 0.995 | 0.126 |
| [release-qwen3.8-27b-ud-q4_k_m-mtp](evidence/bench_release-qwen3.8-27b-ud-q4_k_m-mtp_rtx-4090_2026-10-16.json) | decode_b8 (greedy, v1-qwen3_5.jsonl) | holon_http@mtp | 15.59 | 1.51 | [release-qwen3.8-27b-ud-q4_k_m](evidence/bench_release-qwen3.8-27b-ud-q4_k_m_rtx-4090_2026-10-15.json), unpaired | 0.766 | 2.55 | 0.995 | 0.118 |
| [release-qwen3.8-27b-ud-q4_k_m-mtp](evidence/bench_release-qwen3.8-27b-ud-q4_k_m-mtp_rtx-4090_2026-10-16.json) | decode_b8 (greedy, v1-qwen3_5.jsonl) | llamaserver@mtp3 | 20.36 | 1.42 | [release-qwen3.8-27b-ud-q4_k_m](evidence/bench_release-qwen3.8-27b-ud-q4_k_m_rtx-4090_2026-10-15.json), unpaired | 0.709 | — | — | 0.142 |
| [release-qwen3.8-27b-ud-q4_k_m-mtp](evidence/bench_release-qwen3.8-27b-ud-q4_k_m-mtp_rtx-4090_2026-10-16.json) | agent (greedy, v1-qwen3_5.jsonl) | holon@mtp | 15.52 | 1.86 | [release-qwen3.8-27b-ud-q4_k_m](evidence/bench_release-qwen3.8-27b-ud-q4_k_m_rtx-4090_2026-10-15.json), unpaired | 0.766 | 2.92 | 0.996 | 0.109 |
| [release-qwen3.8-27b-ud-q4_k_m-mtp](evidence/bench_release-qwen3.8-27b-ud-q4_k_m-mtp_rtx-4090_2026-10-16.json) | agent (greedy, v1-qwen3_5.jsonl) | holon_http@mtp | 16.62 | 1.77 | [release-qwen3.8-27b-ud-q4_k_m](evidence/bench_release-qwen3.8-27b-ud-q4_k_m_rtx-4090_2026-10-15.json), unpaired | 0.757 | 2.90 | 0.994 | 0.084 |
| [release-qwen3.8-27b-ud-q4_k_m-mtp](evidence/bench_release-qwen3.8-27b-ud-q4_k_m-mtp_rtx-4090_2026-10-16.json) | agent (greedy, v1-qwen3_5.jsonl) | llamaserver@mtp3 | 31.98 | 1.63 | [release-qwen3.8-27b-ud-q4_k_m](evidence/bench_release-qwen3.8-27b-ud-q4_k_m_rtx-4090_2026-10-15.json), unpaired | 0.712 | — | — | 0.146 |
| [release-qwen3.8-27b-ud-q4_k_m-mtp](evidence/bench_release-qwen3.8-27b-ud-q4_k_m-mtp_rtx-4090_2026-10-16.json) | long_prompt (greedy, v1-qwen3_5.jsonl) | holon@mtp | 22.53 | 1.43 (inflated by repetition) | [release-qwen3.8-27b-ud-q4_k_m](evidence/bench_release-qwen3.8-27b-ud-q4_k_m_rtx-4090_2026-10-15.json), unpaired | 0.941 | 2.89 | 1.000 | 0.453 |
| [release-qwen3.8-27b-ud-q4_k_m-mtp](evidence/bench_release-qwen3.8-27b-ud-q4_k_m-mtp_rtx-4090_2026-10-16.json) | long_prompt (greedy, v1-qwen3_5.jsonl) | holon_http@mtp | 22.53 | 1.43 (inflated by repetition) | [release-qwen3.8-27b-ud-q4_k_m](evidence/bench_release-qwen3.8-27b-ud-q4_k_m_rtx-4090_2026-10-15.json), unpaired | 0.903 | 2.84 | 0.999 | 0.366 |
| [release-qwen3.8-27b-ud-q4_k_m-mtp](evidence/bench_release-qwen3.8-27b-ud-q4_k_m-mtp_rtx-4090_2026-10-16.json) | long_prompt (greedy, v1-qwen3_5.jsonl) | llamaserver@mtp3 | 24.84 | 1.27 (inflated by repetition) | [release-qwen3.8-27b-ud-q4_k_m](evidence/bench_release-qwen3.8-27b-ud-q4_k_m_rtx-4090_2026-10-15.json), unpaired | 0.776 | — | — | 0.302 |

rtx-5080: speculation

| record | trace (sampling, prompts) | backend | TPOT p50 ms | speedup over its plain (TPOT p50) | plain from | accepted / drafted | tokens per verify row | depth-1 share | repeated output |
|---|---|---|---|---|---|---|---|---|---|
| [release-qwen3.8-27b-ud-iq3_s-mtp](evidence/bench_release-qwen3.8-27b-ud-iq3_s-mtp_rtx-5080_2026-10-16.json) | decode_b1 (greedy, v1-qwen3_5.jsonl) | holon@mtp | 7.62 | 2.05 (inflated by repetition) | [release-qwen3.8-27b-ud-iq3_s](evidence/bench_release-qwen3.8-27b-ud-iq3_s_rtx-5080_2026-10-16.json), unpaired | 0.800 | 3.40 | 1.000 | 0.227 |
| [release-qwen3.8-27b-ud-iq3_s-mtp](evidence/bench_release-qwen3.8-27b-ud-iq3_s-mtp_rtx-5080_2026-10-16.json) | decode_b1 (greedy, v1-qwen3_5.jsonl) | holon_http@mtp | 7.61 | 2.05 (inflated by repetition) | [release-qwen3.8-27b-ud-iq3_s](evidence/bench_release-qwen3.8-27b-ud-iq3_s_rtx-5080_2026-10-16.json), unpaired | 0.800 | 3.40 | 1.000 | 0.227 |
| [release-qwen3.8-27b-ud-iq3_s-mtp](evidence/bench_release-qwen3.8-27b-ud-iq3_s-mtp_rtx-5080_2026-10-16.json) | decode_b1 (greedy, v1-qwen3_5.jsonl) | llamaserver@mtp3 | 8.38 | 2.03 (inflated by repetition) | [release-qwen3.8-27b-ud-iq3_s](evidence/bench_release-qwen3.8-27b-ud-iq3_s_rtx-5080_2026-10-16.json), unpaired | 0.789 | — | — | 0.227 |
| [release-qwen3.8-27b-ud-iq3_s-mtp](evidence/bench_release-qwen3.8-27b-ud-iq3_s-mtp_rtx-5080_2026-10-16.json) | decode_b8 (greedy, v1-qwen3_5.jsonl) | holon@mtp | 12.8 | 1.65 (inflated by repetition) | [release-qwen3.8-27b-ud-iq3_s](evidence/bench_release-qwen3.8-27b-ud-iq3_s_rtx-5080_2026-10-16.json), unpaired | 0.829 | 2.67 | 1.000 | 0.305 |
| [release-qwen3.8-27b-ud-iq3_s-mtp](evidence/bench_release-qwen3.8-27b-ud-iq3_s-mtp_rtx-5080_2026-10-16.json) | decode_b8 (greedy, v1-qwen3_5.jsonl) | holon_http@mtp | 13.58 | 1.54 (inflated by repetition) | [release-qwen3.8-27b-ud-iq3_s](evidence/bench_release-qwen3.8-27b-ud-iq3_s_rtx-5080_2026-10-16.json), unpaired | 0.810 | 2.64 | 0.996 | 0.281 |
| [release-qwen3.8-27b-ud-iq3_s-mtp](evidence/bench_release-qwen3.8-27b-ud-iq3_s-mtp_rtx-5080_2026-10-16.json) | decode_b8 (greedy, v1-qwen3_5.jsonl) | llamaserver@mtp3 | 16.26 | 1.54 (inflated by repetition) | [release-qwen3.8-27b-ud-iq3_s](evidence/bench_release-qwen3.8-27b-ud-iq3_s_rtx-5080_2026-10-16.json), unpaired | 0.726 | — | — | 0.259 |
| [release-qwen3.8-27b-ud-iq3_s-mtp](evidence/bench_release-qwen3.8-27b-ud-iq3_s-mtp_rtx-5080_2026-10-16.json) | agent (greedy, v1-qwen3_5.jsonl) | holon@mtp | 17.73 | 1.66 | [release-qwen3.8-27b-ud-iq3_s](evidence/bench_release-qwen3.8-27b-ud-iq3_s_rtx-5080_2026-10-16.json), unpaired | 0.773 | 2.88 | 0.997 | 0.081 |
| [release-qwen3.8-27b-ud-iq3_s-mtp](evidence/bench_release-qwen3.8-27b-ud-iq3_s-mtp_rtx-5080_2026-10-16.json) | agent (greedy, v1-qwen3_5.jsonl) | holon_http@mtp | 17.37 | 1.85 | [release-qwen3.8-27b-ud-iq3_s](evidence/bench_release-qwen3.8-27b-ud-iq3_s_rtx-5080_2026-10-16.json), unpaired | 0.773 | 2.89 | 0.997 | 0.091 |
| [release-qwen3.8-27b-ud-iq3_s-mtp](evidence/bench_release-qwen3.8-27b-ud-iq3_s-mtp_rtx-5080_2026-10-16.json) | agent (greedy, v1-qwen3_5.jsonl) | llamaserver@mtp3 | 24.53 | 2.00 | [release-qwen3.8-27b-ud-iq3_s](evidence/bench_release-qwen3.8-27b-ud-iq3_s_rtx-5080_2026-10-16.json), unpaired | 0.719 | — | — | 0.128 |
| [release-qwen3.8-27b-ud-iq3_s-mtp](evidence/bench_release-qwen3.8-27b-ud-iq3_s-mtp_rtx-5080_2026-10-16.json) | long_prompt (greedy, v1-qwen3_5.jsonl) | holon@mtp | 27.22 | 1.31 (inflated by repetition) | [release-qwen3.8-27b-ud-iq3_s](evidence/bench_release-qwen3.8-27b-ud-iq3_s_rtx-5080_2026-10-16.json), unpaired | 0.919 | 2.85 | 1.000 | 0.369 |
| [release-qwen3.8-27b-ud-iq3_s-mtp](evidence/bench_release-qwen3.8-27b-ud-iq3_s-mtp_rtx-5080_2026-10-16.json) | long_prompt (greedy, v1-qwen3_5.jsonl) | holon_http@mtp | 27.39 | 1.30 (inflated by repetition) | [release-qwen3.8-27b-ud-iq3_s](evidence/bench_release-qwen3.8-27b-ud-iq3_s_rtx-5080_2026-10-16.json), unpaired | 0.916 | 2.85 | 1.000 | 0.363 |
| [release-qwen3.8-27b-ud-iq3_s-mtp](evidence/bench_release-qwen3.8-27b-ud-iq3_s-mtp_rtx-5080_2026-10-16.json) | long_prompt (greedy, v1-qwen3_5.jsonl) | llamaserver@mtp3 | 29.98 | 1.46 (inflated by repetition) | [release-qwen3.8-27b-ud-iq3_s](evidence/bench_release-qwen3.8-27b-ud-iq3_s_rtx-5080_2026-10-16.json), unpaired | 0.733 | — | — | 0.302 |

rtx-5090: speculation

| record | trace (sampling, prompts) | backend | TPOT p50 ms | speedup over its plain (TPOT p50) | plain from | accepted / drafted | tokens per verify row | depth-1 share | repeated output |
|---|---|---|---|---|---|---|---|---|---|
| [release-qwen3.8-27b-ud-q4_k_m-mtp](evidence/bench_release-qwen3.8-27b-ud-q4_k_m-mtp_rtx-5090_2026-10-16.json) | decode_b1 (greedy, v1-qwen3_5.jsonl) | holon@mtp | 4.9 | 2.33 (inflated by repetition) | [release-qwen3.8-27b-ud-q4_k_m](evidence/bench_release-qwen3.8-27b-ud-q4_k_m_rtx-5090_2026-10-15.json), unpaired | 0.878 | 3.63 | 1.000 | 0.238 |
| [release-qwen3.8-27b-ud-q4_k_m-mtp](evidence/bench_release-qwen3.8-27b-ud-q4_k_m-mtp_rtx-5090_2026-10-16.json) | decode_b1 (greedy, v1-qwen3_5.jsonl) | holon_http@mtp | 4.89 | 2.34 (inflated by repetition) | [release-qwen3.8-27b-ud-q4_k_m](evidence/bench_release-qwen3.8-27b-ud-q4_k_m_rtx-5090_2026-10-15.json), unpaired | 0.878 | 3.63 | 1.000 | 0.238 |
| [release-qwen3.8-27b-ud-q4_k_m-mtp](evidence/bench_release-qwen3.8-27b-ud-q4_k_m-mtp_rtx-5090_2026-10-16.json) | decode_b1 (greedy, v1-qwen3_5.jsonl) | llamaserver@mtp3 | 5.48 | 2.29 (inflated by repetition) | [release-qwen3.8-27b-ud-q4_k_m](evidence/bench_release-qwen3.8-27b-ud-q4_k_m_rtx-5090_2026-10-15.json), unpaired | 0.859 | — | — | 0.238 |
| [release-qwen3.8-27b-ud-q4_k_m-mtp](evidence/bench_release-qwen3.8-27b-ud-q4_k_m-mtp_rtx-5090_2026-10-16.json) | decode_b8 (greedy, v1-qwen3_5.jsonl) | holon@mtp | 8.74 | 1.58 | [release-qwen3.8-27b-ud-q4_k_m](evidence/bench_release-qwen3.8-27b-ud-q4_k_m_rtx-5090_2026-10-15.json), unpaired | 0.758 | 2.52 | 0.995 | 0.107 |
| [release-qwen3.8-27b-ud-q4_k_m-mtp](evidence/bench_release-qwen3.8-27b-ud-q4_k_m-mtp_rtx-5090_2026-10-16.json) | decode_b8 (greedy, v1-qwen3_5.jsonl) | holon_http@mtp | 8.83 | 1.56 | [release-qwen3.8-27b-ud-q4_k_m](evidence/bench_release-qwen3.8-27b-ud-q4_k_m_rtx-5090_2026-10-15.json), unpaired | 0.765 | 2.54 | 0.995 | 0.108 |
| [release-qwen3.8-27b-ud-q4_k_m-mtp](evidence/bench_release-qwen3.8-27b-ud-q4_k_m-mtp_rtx-5090_2026-10-16.json) | decode_b8 (greedy, v1-qwen3_5.jsonl) | llamaserver@mtp3 | 12.44 | 1.35 | [release-qwen3.8-27b-ud-q4_k_m](evidence/bench_release-qwen3.8-27b-ud-q4_k_m_rtx-5090_2026-10-15.json), unpaired | 0.712 | — | — | 0.115 |
| [release-qwen3.8-27b-ud-q4_k_m-mtp](evidence/bench_release-qwen3.8-27b-ud-q4_k_m-mtp_rtx-5090_2026-10-16.json) | agent (greedy, v1-qwen3_5.jsonl) | holon@mtp | 7.73 | 1.84 | [release-qwen3.8-27b-ud-q4_k_m](evidence/bench_release-qwen3.8-27b-ud-q4_k_m_rtx-5090_2026-10-15.json), unpaired | 0.736 | 3.01 | 0.995 | 0.083 |
| [release-qwen3.8-27b-ud-q4_k_m-mtp](evidence/bench_release-qwen3.8-27b-ud-q4_k_m-mtp_rtx-5090_2026-10-16.json) | agent (greedy, v1-qwen3_5.jsonl) | holon_http@mtp | 7.46 | 1.88 | [release-qwen3.8-27b-ud-q4_k_m](evidence/bench_release-qwen3.8-27b-ud-q4_k_m_rtx-5090_2026-10-15.json), unpaired | 0.744 | 3.04 | 0.993 | 0.100 |
| [release-qwen3.8-27b-ud-q4_k_m-mtp](evidence/bench_release-qwen3.8-27b-ud-q4_k_m-mtp_rtx-5090_2026-10-16.json) | agent (greedy, v1-qwen3_5.jsonl) | llamaserver@mtp3 | 12.15 | 1.70 | [release-qwen3.8-27b-ud-q4_k_m](evidence/bench_release-qwen3.8-27b-ud-q4_k_m_rtx-5090_2026-10-15.json), unpaired | 0.712 | — | — | 0.145 |
| [release-qwen3.8-27b-ud-q4_k_m-mtp](evidence/bench_release-qwen3.8-27b-ud-q4_k_m-mtp_rtx-5090_2026-10-16.json) | long_prompt (greedy, v1-qwen3_5.jsonl) | holon@mtp | 15.28 | 1.46 | [release-qwen3.8-27b-ud-q4_k_m](evidence/bench_release-qwen3.8-27b-ud-q4_k_m_rtx-5090_2026-10-15.json), unpaired | 0.839 | 2.73 | 1.000 | 0.032 |
| [release-qwen3.8-27b-ud-q4_k_m-mtp](evidence/bench_release-qwen3.8-27b-ud-q4_k_m-mtp_rtx-5090_2026-10-16.json) | long_prompt (greedy, v1-qwen3_5.jsonl) | holon_http@mtp | 15.08 | 1.49 (inflated by repetition) | [release-qwen3.8-27b-ud-q4_k_m](evidence/bench_release-qwen3.8-27b-ud-q4_k_m_rtx-5090_2026-10-15.json), unpaired | 0.920 | 2.86 | 0.999 | 0.347 |
| [release-qwen3.8-27b-ud-q4_k_m-mtp](evidence/bench_release-qwen3.8-27b-ud-q4_k_m-mtp_rtx-5090_2026-10-16.json) | long_prompt (greedy, v1-qwen3_5.jsonl) | llamaserver@mtp3 | 17.65 | 1.44 (inflated by repetition) | [release-qwen3.8-27b-ud-q4_k_m](evidence/bench_release-qwen3.8-27b-ud-q4_k_m_rtx-5090_2026-10-15.json), unpaired | 0.709 | — | — | 0.266 |

## Contributing

Rules and routing by task are in [AGENTS.md](AGENTS.md); adding a model is in
[docs/adding-a-model.md](docs/adding-a-model.md).

## License

Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)); third-party code is listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
