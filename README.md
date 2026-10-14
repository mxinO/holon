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
also serves Qwen3.8), from HF safetensors directories or GGUF files. The recommended file per card and its lock:

| model file (hub repository) | RTX 5080 | RTX 4090 | RTX 5090 | RTX 3090 |
|---|---|---|---|---|
| Qwen3-8B bf16 (`Qwen/Qwen3-8B`) | does not fit | lock | lock † | lock † |
| Qwen3-8B Q4_K_M (`unsloth/Qwen3-8B-GGUF`) | lock | lock † | lock † | lock † |
| Qwen3-8B UD-IQ3_XXS (`unsloth/Qwen3-8B-GGUF`) | lock | borrowed, unverified | borrowed, unverified | borrowed, unverified |
| Qwen3-30B-A3B UD-IQ3_XXS (`unsloth/Qwen3-30B-A3B-GGUF`) | lock | borrowed, unverified | borrowed, unverified | borrowed, unverified |
| Qwen3-30B-A3B Q4_K_M (`unsloth/Qwen3-30B-A3B-GGUF`) | does not fit (17.3 GiB) | lock | lock † | lock † |
| Qwen3.8-27B UD-IQ3_S (`unsloth/Qwen3.8-27B-GGUF`) | lock | borrowed, unverified | borrowed, unverified | borrowed, unverified |
| Qwen3.8-27B UD-Q4_K_M (`unsloth/Qwen3.8-27B-GGUF`) | does not fit (15.3 GiB) | lock | lock † | lock † |

† TBD (release step 7): written in the release environment at release step 7, where every other lock is also
re-propagated and the 27B locks' speculation (MTP) is decided.

TBD (M0.6 merge): the low-precision files (Qwen3-8B NVFP4 and FP8, Qwen3-30B-A3B NVFP4 and MXFP4_MOE) are not in this
tree yet. Their locks are planned on the RTX 5080 and 4090; on the 5090 and 3090 they would borrow, unverified (FP8 not
on the 3090).

**Borrowed, unverified.** Locks are found by the checkpoint's build hash, so a file downloaded anywhere, under any
name, finds its lock. Without a lock for this card, `holon serve` borrows the nearest class's: the same GPU on another
toolchain, then the same SM, then the same toolchain. A borrowed lock's settings apply as they are and its kernel pins
become preferences (a provider that cannot run here falls back to the default choice, logged). It was not verified on
this card. If it does not start with a plugin or memory error, the server falls back to defaults at once.

**Served without a lock** (defaults, logged): every other checkpoint of the three architectures, for example
Qwen3.5 4B / 9B and the bf16 Qwen3.8-27B and Qwen3-30B-A3B (large-memory paths). Defaults serve at most 8 sequences
(`holon serve <file> --max-seqs N` changes it) and a context of min(the model's trained context, 32768) tokens,
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
```

As another user, give the container a writable home, so the cache of weight-file hashes (`~/.cache/holon/weights`)
persists across starts instead of each start re-reading the file:

```bash
docker run --gpus all --user "$(id -u):$(id -g)" -e HOME=/cache -v <cache dir>:/cache \
  -p 8000:8000 -v <model dir>:/models holon:v0.1.0 serve /models/<file> --host 0.0.0.0
```

## Quickstart

```bash
uv run hf download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-IQ3_S.gguf --local-dir models   # RTX 5080; else UD-Q4_K_M
uv run holon serve models/Qwen3.8-27B-UD-IQ3_S.gguf   # starts from this card's lock; the log names it
```

`holon serve` listens on `127.0.0.1:8000` (`holon serve <file> --host H --port P`) and checks no API key: do not
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

The served model id is in `/v1/models` (`curl http://127.0.0.1:8000/v1/models`): the file name without `.gguf` for a
file outside the HF cache. Clients (`pip install openai anthropic` in your own environment):

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

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:8000 ANTHROPIC_AUTH_TOKEN=holon ANTHROPIC_MODEL=<id from /v1/models> \
ANTHROPIC_DEFAULT_HAIKU_MODEL=<same> CLAUDE_CODE_SUBAGENT_MODEL=<same> claude
```

The server drops Claude Code's per-request billing-header system block, which would otherwise defeat prefix caching
(`holon serve <file> --plugin-options http.drop_billing_header=false` keeps it; `/metrics` counts the drops).
Claude Code's auto-compaction assumes a far larger window than the 32K tokens served at most by default.

## Options

Flags override the lock, which overrides the defaults; `holon serve --help` lists every flag (one per engine
setting).

- **Turn speculation off:** `holon serve <file> --max-draft-tokens 0`.
- **Ignore the lock:** `holon serve <file> --no-lock` serves defaults and your flags. Use it if a start hangs: the
  server retries a start that fails other than with a plugin or memory error 4 times, each up to 600 s, before falling
  back to defaults.
- **Change or delete one key of the lock:** dict flags merge per key, and `key=` (empty value) deletes the lock's
  setting: `holon serve <file> --plugin-options kv.format=` serves the default KV format, and
  `holon serve <file> --slots logits_processor=` deselects the grammar processor.
- **Mixed-GPU hosts:** Holon serves on one GPU, the first visible one, and picks the lock of its class. Pin the card,
  for the prebuild too: `CUDA_VISIBLE_DEVICES=1 uv run holon serve <file>`.
- **Other locks:** `holon serve <file> --plans DIR` reads lock histories from `DIR` (default: `plans/` of this tree).

### Structured output

The release locks select the grammar processor `grammar.xgrammar`. With `--no-lock` or without a lock,
`holon serve <file> --slots logits_processor=grammar.xgrammar` selects it; without it, constrained requests are
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
- The context is at most 32768 tokens by default (`holon serve <file> --max-model-len N`), and less when it does not
  fit.
- Under a grammar, an output can run on in whitespace, digits or a string the schema admits until `max_tokens`: on the
  27B with MTP, 2.66% of constrained outputs were truncated against 0.33% unconstrained, over a 2-point gate that was
  accepted as a recorded shortfall. Guards against such runaways are deferred; set `max_tokens`.
- DFlash2 speculation is an opt-in, in no lock ([docs/m04-spec.md](docs/m04-spec.md)).
- Stop the server before a `git pull`: the install is editable, so a supervisor restart would import the new code.

## Performance

TBD (release step 8): the benchmark tables, rendered from the evidence records by `bench/report.py`, each linking its
record; losses included.

## Contributing

Rules and routing by task are in [AGENTS.md](AGENTS.md); adding a model is in
[docs/adding-a-model.md](docs/adding-a-model.md).

## License

Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)); third-party code is listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
