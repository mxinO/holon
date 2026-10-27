# Holon

Holon is an LLM inference engine for one consumer NVIDIA GPU, licensed under Apache-2.0. Its core fixes only the
timing of a step, memory accounting, the plugin resolver and a set of typed contracts, in at most 6000 lines; models,
layer types, quantization formats, kernels, speculative decoding, grammars and frontends are all plugins.

What this buys is isolation. Adding a model touches no shared code: a model is its own plugin package
([docs/adding-a-model.md](docs/adding-a-model.md)). Each model has its own lock per GPU class: the engine settings and
kernel bindings tuned for that card, with the code they were verified against, and `holon serve` starts from it. A
lock changes only through the judge, which checks the model's outputs against fp32 goldens or reproduces them byte for
byte; a change to code a model runs leaves that model's lock unused until the judge re-verifies it. Every published
performance number comes from an evidence record.

Holon serves the OpenAI and Anthropic APIs, with tool calls, reasoning and grammar-constrained structured output, so
an OpenAI client, an Anthropic client or Claude Code can use it directly; the design is in
[docs/design.md](docs/design.md).

A Mixture-of-Experts model larger than the card serves with its experts in pinned host RAM and the most used of them
cached in VRAM ([Offload](#offload)), its output byte-identical to all-resident serving; prefix KV and recurrent-state
checkpoints can move to host RAM instead of being dropped ([KV tiers](#kv-tiers)).

## Supported GPUs and models

| GPU | memory | SM | lock class |
|---|---|---|---|
| RTX 5080 | 16 GB | 120 | `rtx-5080` |
| RTX 4090 | 24 GB | 89 | `rtx-4090` |
| RTX 5090 | 32 GB | 120 | `rtx-5090` |
| RTX 3090 | 24 GB | 86 | `rtx-3090` |

Other GPUs: GGUF files refuse at startup (the GGUF kernels are built for SM 86, 89 and 120 only); bf16 checkpoints
are untested.

Architectures: Qwen3 (`models.qwen3`), Qwen3-MoE (`models.qwen3_moe`), the Qwen3.5 hybrid (`models.qwen3_5`, which
also serves Qwen3.8) and Qwen3.8-Flash-Next (`models.qwen4_exp`, GGUF, with its MTP head), from HF safetensors
directories or GGUF files. The recommended file per card and its lock
(`holon models` lists them for the card it runs on):

| model file (hub repository) | RTX 5080 | RTX 4090 | RTX 5090 | RTX 3090 |
|---|---|---|---|---|
| Qwen3-8B bf16 (`Qwen/Qwen3-8B`) | does not fit | this card's lock | this card's lock | this card's lock |
| Qwen3-8B Q4_K_M (`unsloth/Qwen3-8B-GGUF`) | this card's lock | this card's lock | this card's lock | this card's lock |
| Qwen3-8B UD-IQ3_XXS (`unsloth/Qwen3-8B-GGUF`) | this card's lock | borrowed, unverified | borrowed, unverified | borrowed, unverified |
| Qwen3-8B NVFP4 (`nvidia/Qwen3-8B-NVFP4`) | this card's lock | this card's lock | borrowed, unverified | borrowed, unverified |
| Qwen3-8B FP8 (`Qwen/Qwen3-8B-FP8`) | this card's lock | this card's lock | borrowed, unverified | not served (FP8 needs SM 89 or newer) |
| Qwen3-30B-A3B UD-IQ3_XXS (`unsloth/Qwen3-30B-A3B-GGUF`) | this card's lock | borrowed, unverified | borrowed, unverified | borrowed, unverified |
| Qwen3-30B-A3B Q4_K_M (`unsloth/Qwen3-30B-A3B-GGUF`) | this card's lock, offload | this card's lock | this card's lock | this card's lock |
| Qwen3-30B-A3B NVFP4 (`nvidia/Qwen3-30B-A3B-NVFP4`) | does not fit (16.9 GiB) | this card's lock | borrowed, unverified | borrowed, unverified |
| Qwen3-30B-A3B-Instruct-2507 MXFP4_MOE (`Face314/Qwen3-30B-A3B-Instruct-2507-MXFP4_MOE`) | does not fit (15.9 GiB) | this card's lock | borrowed, unverified | borrowed, unverified |
| Qwen3.8-27B UD-IQ3_S (`unsloth/Qwen3.8-27B-GGUF`) | this card's lock, KV tier | borrowed, unverified | borrowed, unverified | borrowed, unverified |
| Qwen3.8-27B UD-Q4_K_M (`unsloth/Qwen3.8-27B-GGUF`) | does not fit (15.3 GiB) | this card's lock | this card's lock, KV tier | this card's lock |
| Qwen3.8-Flash-Next GSQ-RCO IQ2_XS (`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`) | this card's lock, offload, MTP | borrowed, unverified | borrowed, unverified | borrowed, unverified |
| Qwen3.8-Flash-Next GSQ-RCO IQ3_S (`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`) | borrowed, unverified | borrowed, unverified | this card's lock, offload, MTP | borrowed, unverified |

**This card's lock**: written and verified on this card (L2 against the judge's floor, the snapshot trace identical
over three runs), in the release environment (`docs/dev/release.md` §16, §19) or at a later milestone's close
(`docs/dev/v1-offload.md` "Step 8, part 2", `docs/dev/v1-kv-tiers.md` "Close (2026-10-20)",
`docs/dev/v1-flash-next.md` §22.7). Every lock but Qwen3.8-Flash-Next's selects the grammar processor. The Qwen3.8-27B
locks serve a context of 49,152 tokens (40,960 on the RTX 5080) without speculation (Known limitations).

**Launch flags.** Host memory is a deployment's, never a lock's: a lock that keeps pools in pinned host RAM starts only
with `--host-memory BYTES` (Options). These locks need it, and Qwen3.8-Flash-Next needs files of its own:

| lock | in pinned host RAM | launch with |
|---|---|---|
| Qwen3.8-27B, RTX 5080 and RTX 5090 | a 16 GiB KV tier ([KV tiers](#kv-tiers)) | `--host-memory 17179869184` |
| Qwen3-30B-A3B Q4_K_M, RTX 5080 | the experts ([Offload](#offload)) | `--host-memory 21474836480` and its routing profile |
| Qwen3.8-Flash-Next IQ2_XS, RTX 5080 | the experts, the PLE table, the token embeddings | `--host-memory 70000000000 --memory-fraction 0.97`, the MTP pack and the routing profile ([Qwen3.8-Flash-Next](#qwen38-flash-next)) |
| Qwen3.8-Flash-Next IQ3_S, RTX 5090 | the same | `--host-memory 86000000000 --memory-fraction 0.985`, the same |

The tier needs at least its 16 GiB; the tier locks and the Qwen3-30B-A3B offload lock were verified at
21474836480 (`docs/dev/v1-kv-tiers.md` "Close (2026-10-20)"). Qwen3.8-Flash-Next's values are those its locks were
measured at (`docs/dev/v1-flash-next.md` §22.7). `--memory-fraction` (default 0.9: the share of the memory left after
weights that activations and state may take, here mostly the expert cache) is not in a lock either.

**Borrowed, unverified.** Locks are found by the checkpoint's build hash, so a file downloaded anywhere, under any
name, finds its lock. Without a lock for this card, `holon serve` borrows the nearest class's: the same GPU on another
toolchain, then the same SM, then the same toolchain. A borrowed lock's settings apply as they are and its kernel pins
become preferences (a provider that cannot run here falls back to the default choice, logged). It was not verified on
this card. If it does not start with a plugin or memory error, the server falls back to defaults at once. A borrowed
lock needs the launch flags and files of the lock it borrows (Launch flags above): on an RTX 4090, for example, the
Qwen3.8-27B UD-IQ3_S borrows the RTX 5080's lock and its KV tier.

**Served without a lock** (defaults, logged): every other checkpoint of these architectures, for example
Qwen3.5 4B / 9B and the bf16 Qwen3.8-27B and Qwen3-30B-A3B (large-memory paths). Defaults serve at most 8 sequences
(`holon serve <model> --max-seqs N` changes it) and a context of min(the model's trained context, 32768) tokens,
clamped to what fits.

## Install

This README describes `main`. The last release, v0.1.1
([docs/releases/RELEASE_NOTES_v0.1.1.md](docs/releases/RELEASE_NOTES_v0.1.1.md)), predates expert offload, the KV tiers
and Qwen3.8-Flash-Next: their locks pin `main`'s code, so install `main` for them, or the tag (`--branch v0.1.1`, whose
README describes it) for the release. Release steps 7–10's locks and records name commits of the release line as
first built; the same commits are on `main` under new SHAs, with identical code and a privacy-scrubbed spike README
(`evidence/v0_release_line_sha_map_2026-10-16.txt` maps them; `evidence/history_scrub_commit_map_2026-10-16.txt` maps
older SHAs). v0.1.1's locks were carried to its code with byte-identical snapshot traces.

Host requirements: Linux x86-64, an NVIDIA driver supporting CUDA 13, the CUDA 13 toolkit (`nvcc` on `PATH` or under
`CUDA_HOME`), gcc / g++, git and [uv](https://docs.astral.sh/uv/). uv brings Python 3.12, torch 2.11 for CUDA 13.0 and
every pinned dependency from `uv.lock`.

```bash
git clone https://github.com/mxinO/holon.git holon && cd holon      # main; --branch v0.1.1 for the release
uv sync --frozen                     # Python 3.12, torch 2.11 + CUDA 13 and every pin from uv.lock
uv run holon prebuild                # GGUF kernels and FlashInfer modules for the visible GPUs, once
```

`holon prebuild` compiles the GGUF kernels and FlashInfer's modules for the SM of every visible GPU, once per install;
`holon prebuild --sm 86 89 120` names the targets instead, and `MAX_JOBS=N` caps the parallel compiles.
`uv sync --frozen --group dev` also installs the lock tools (`python -m holon_tools`), which make the offload locks'
routing profiles and Qwen3.8-Flash-Next's MTP pack, and the test dependencies.

### Docker

The `Dockerfile` installs the release's environment for hosts without a CUDA toolkit and prebuilds for SM 86, 89 and
120. Build it from the release tag, never from a working directory; the build takes ≈ 10–15 min on a many-core host,
and the image is ≈ 14.8 GB. It has not been checked against `main`:

```bash
git archive --format=tar v0.1.1 | docker build -t holon:v0.1.1 -           # --build-arg MAX_JOBS=N caps compiles
docker run --gpus all -p 8000:8000 -v <model dir>:/models holon:v0.1.1 serve /models/<file> --host 0.0.0.0
docker run --gpus all -p 8000:8000 -e HF_HOME=/hf -v <hf cache dir>:/hf holon:v0.1.1 \
  serve unsloth/Qwen3.8-27B-GGUF --host 0.0.0.0                            # a hub model, kept in <hf cache dir>
```

As another user, give the container a writable home, so the cache of weight-file hashes (`~/.cache/holon/weights`)
persists across starts instead of each start re-reading the file:

```bash
docker run --gpus all --user "$(id -u):$(id -g)" -e HOME=/cache -v <cache dir>:/cache \
  -p 8000:8000 -v <model dir>:/models holon:v0.1.1 serve /models/<file> --host 0.0.0.0
```

## Quickstart

```bash
uv run holon models                                   # the models locked for this card, by name
uv run holon serve unsloth/Qwen3.8-27B-GGUF --host-memory 17179869184
```

The second line serves this card's locked file, downloaded into the HF cache; the host memory is for the KV tier of
the RTX 5080's and RTX 5090's locks, and unused on the other cards.

`holon serve` takes a local file or directory, or a Hugging Face Hub reference, downloaded into the HF cache once:

| reference | serves |
|---|---|
| `unsloth/Qwen3.8-27B-GGUF:UD-IQ3_S` | the repository's GGUF file whose name ends in `-UD-IQ3_S` (any case; `:Q4_K_M` takes `…-Q4_K_M.gguf` over `…-UD-Q4_K_M.gguf`) |
| `unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_S` | that file, as `holon models` names it |
| `unsloth/Qwen3.8-27B-GGUF` | the file this card has a verified lock for (`holon models`: "this card's lock"); else the repository's only GGUF file, else an error listing them |
| `Qwen/Qwen3-8B` | an HF safetensors checkpoint |

The log names the file chosen and its size; a file larger than the GPU's memory is refused before downloading, unless
`--host-memory` is given (offload) or `--no-size-check`. The hub's own settings apply: `hf auth login` for gated
repositories, `HF_HOME` for the cache, and `HF_HUB_OFFLINE=1` to serve cached models without the network (a tag then
sees only the cached files).
Offline, a hub name is found through the cache's `refs/main`, which a download writes; in a cache filled another way
(by revision, or copied without `refs/`) serve the file's path instead,
`<HF cache>/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/<revision>/<file>`: it finds the same lock.
A split GGUF file (`-00001-of-0000N.gguf`) is served by its first shard's path or its stem; a hub reference
fetches all its shards.

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
model = "Qwen3.8-27B-UD-IQ3_S"             # the served id (/v1/models); the server ignores the model a request names
msg = client.messages.create(model=model, max_tokens=512,
                             messages=[{"role": "user", "content": "Name three prime numbers."}])
print("".join(b.text for b in msg.content if b.type == "text"))   # thinking blocks come first
```

The server also answers `/v1/completions`, `/v1/responses`, `/v1/messages/count_tokens` and `/health`.

- **System prompts:** OpenAI's `developer` messages are system messages; a leading run of system and developer
  messages holding one becomes one system prompt.
- **Reasoning effort:** chat's `reasoning_effort`, the Responses API's `reasoning.effort` and Anthropic's
  `output_config.effort` set the chat template's level where the template has levels (the Qwen3.8 templates: `low`,
  `medium`, `xhigh`, and `high` in the Unsloth GGUF files). Another level of the clients' scale (`minimal` … `max`)
  takes the nearest level the template has (`minimal` the lowest, the higher of two as near); `none` turns thinking
  off; any other value is a 400 naming the levels. Templates without levels (Qwen3, Qwen3.5) ignore the scale's
  values. `chat_template_kwargs.reasoning_effort` still sets the template's variable directly.
- **Cached tokens:** usage reports the prompt tokens the prefix cache served, per request:
  `usage.prompt_tokens_details.cached_tokens` (chat and text completions), `usage.input_tokens_details.cached_tokens`
  (Responses), and Anthropic's `cache_read_input_tokens`, with `input_tokens` the rest of the prompt (Anthropic's
  convention: a client adds the two).

### Claude Code

Once it listens, the server prints the line that starts Claude Code against it, with its model id:

```text
holon: Claude Code:
  ANTHROPIC_BASE_URL=http://127.0.0.1:8000 ANTHROPIC_AUTH_TOKEN=holon ANTHROPIC_MODEL=Qwen3.8-27B-UD-IQ3_S ANTHROPIC_DEFAULT_HAIKU_MODEL=Qwen3.8-27B-UD-IQ3_S CLAUDE_CODE_SUBAGENT_MODEL=Qwen3.8-27B-UD-IQ3_S claude
```

The server drops Claude Code's per-request billing-header system block, which would otherwise defeat prefix caching
(`holon serve <model> --plugin-options http.drop_billing_header=false` keeps it; `/metrics` counts the drops).
Serve the Qwen3.8-27B for Claude Code: its locks serve 49,152 tokens of context (40,960 on the RTX 5080). Claude Code
2.1.300 sends, besides its main loop, side requests of ≈ 31K input tokens (`docs/dev/release.md` §18.2), which a 32K
context would fit with about 1K tokens to spare. Its auto-compaction assumes a far larger window than the served
context, so a long session reaches the served limit first. On an RTX 5080 that also drives a display, see Known
limitations. The side requests need the context to stay above ≈ 32K.

The release's agent demo (`bench/demo/`) is a small fix in a three-module Python project — a seeded bug and a missing
function — chosen to fit a 32K context. It shows the served model completing such a task through Claude Code; it
is not a claim of general agent capability. **Result** (pre-registered, Claude Code 2.1.300, each card's 27B lock,
scored by the harness's own copy of the tests): **3 of 3 runs passed on the RTX 5080 (UD-IQ3_S) and 3 of 3 on the
RTX 4090 (UD-Q4_K_M)**, in 13–16 turns and 2–5 minutes each (`evidence/agent_demo_rtx-5080_2026-10-16.json`,
`evidence/agent_demo_rtx-4090_2026-10-15.json`; `docs/dev/release.md` §18). Claude Code sends no temperature, so the
server's default (1.0) applies; its `output_config.effort` (`high`) selects the Qwen3.8 template's highest level, which
was the template's default before v0.1.1. Since v0.1.1 its usage shows the prefix cache's reads as
`cache_read_input_tokens`, apart from `input_tokens`. The demo ran on the locks as they stood at release step 9, at
the 27B's earlier 32K context; the shipped locks differ in that context, in the FlashInfer plugin's `src` and in
v0.1.1's core, with byte-identical snapshot traces (`docs/dev/release.md` §19.1–§19.2,
[docs/releases/RELEASE_NOTES_v0.1.1.md](docs/releases/RELEASE_NOTES_v0.1.1.md)).

## Options

Flags override the lock, which overrides the defaults; `holon serve --help` lists every flag (one per engine
setting), every plugin option the installed plugins read, and the plugins `--slots` selects among. An option no plugin
declares is warned about; a flag's value outside an option's choices is refused before anything is downloaded, and a
lock's is warned about and that lock not used.

- **Speculation (MTP) on the Qwen3.8-27B**, off in its locks (Known limitations): `holon serve <model>
  --max-draft-tokens 3 --slots drafter=drafters.mtp draft_policy=std.spec.fixed_k verifier=std.spec.greedy
  --plugin-options spec.k=1:3,3:2`; `--max-draft-tokens 0` turns speculation off.
- **Mixed steps:** the plain Qwen3.8-27B locks set `--chunk-with-decode 512`: a step that decodes may also carry up to
  512 prompt tokens, so a long prompt does not stall the streams already decoding. Turn speculation on over such a
  lock (a drafter and `--max-draft-tokens N`) and the same setting instead alternates a prefill step of at most 512
  tokens with a verify step. `--chunk-with-decode 0` turns both off.
- **Pinned host memory** is a host's, never a lock's: a lock whose plugins keep host pools (a state tier's host level,
  an offloaded MoE's experts) starts only with `--host-memory BYTES`. Without it the start is refused with the bytes
  the pools need; a state tier's refusal names the flag and value, e.g. for `state_tier.host.bytes=16GiB`:
  `holon serve <model> --host-memory 17179869184`.
- **Ignore the lock:** `holon serve <model> --no-lock` serves defaults (`plans/defaults.toml`: the grammar processor)
  and your flags. Use it if a start hangs: the server tries a start that fails other than with a plugin or memory
  error 4 times, each up to 600 s, before falling back to defaults.
- **Change or delete one key of the lock:** dict flags merge per key, and `key=` (empty value) deletes the lock's
  setting: `holon serve <model> --plugin-options kv.format=` serves the default KV format, and
  `holon serve <model> --slots logits_processor=` deselects the grammar processor.
- **Mixed-GPU hosts:** Holon serves on one GPU, the first visible one, and picks the lock of its class. Pin the card,
  for the prebuild too: `CUDA_VISIBLE_DEVICES=1 uv run holon serve <model>`.
- **Other locks:** `holon serve <model> --plans DIR` reads lock histories from `DIR` (default: `plans/` of this tree).

### Structured output

Every lock but Qwen3.8-Flash-Next's selects the grammar processor `grammar.xgrammar`. With `--no-lock`, without a
lock or on Qwen3.8-Flash-Next, `holon serve <model> --slots logits_processor=grammar.xgrammar` selects it (unverified
on Qwen3.8-Flash-Next); without it, constrained requests are refused. Constrained:
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
effort (the reject list: [docs/dev/m05-grammar.md](docs/dev/m05-grammar.md), D9). The processor runs two vetting workers,
which take 1.6–2.2 GB of host RAM and 3–4 s to start.

## Offload

An offload lock serves a Mixture-of-Experts model whose experts do not fit the card (`std.moe.offload@1`,
[docs/dev/v1-offload.md](docs/dev/v1-offload.md)): the RTX 5080's Qwen3-30B-A3B Q4_K_M lock and both
Qwen3.8-Flash-Next locks.

- **How:** every expert lives in pinned host RAM, and a VRAM cache holds the most used ones (`placement.lfu@1`,
  least-frequently-used, seeded at start from a routing profile). A decode step fetches the experts it misses over
  PCIe; a long prompt step streams the missing ones through cache slots it lends for the step.
- **Exact:** placement never changes a result. The output is byte-identical to the same model served all-resident,
  checked on an RTX A5000 (Qwen3-30B-A3B, every placement and policy) and an RTX PRO 6000 (Qwen3.8-Flash-Next, which
  fits there), and identical across cache sizes and placements over three runs on the RTX 5080 and 5090
  (`docs/dev/v1-offload.md` "Step 8, part 2", acceptance 2; `docs/dev/v1-flash-next.md` §22.3). Like every trace, it is
  run-identical on one machine, not across machines (Known limitations).
- **Host memory:** `--host-memory BYTES` sizes the pinned pools (values per lock under
  [Supported GPUs and models](#supported-gpus-and-models)); a start whose pools do not fit is refused with a report per
  pool.
- **One sequence:** the offload locks serve `max_seqs` 1, so concurrent requests queue. More sequences take VRAM from
  the expert cache: on the RTX 5080's Qwen3.8-Flash-Next lock, 26.2% of the experts are cached at 1 sequence and 21.9%
  at 4 (`docs/dev/v1-flash-next.md` §22.7).
- **Routing profile:** each lock names one (`moe.placement.profile`): a JSON file of routing counts that seeds the
  cache. It must lie in the directory the model file is read from, its links followed: for a file in the HF cache,
  the repository's `blobs/` directory, where the snapshot's links point. `python -m holon_tools profile <model file>`
  runs greedy generations over `judge/prompts/v1.jsonl` (`--prompts`), writes the profile there under the name
  `<file stem>.profile.json` (`--name` sets another) and prints its path; it is refused on another file. The profile
  only decides what the cache holds first, so outputs do not depend on it. Without the file the lock fails to start
  (Known limitations); `holon serve <model> --plugin-options moe.placement.profile=` serves without one, from a cold
  cache.

The RTX 5080's Qwen3-30B-A3B Q4_K_M lock names `profile_x4_gate.json` (made from held-out chat prompts,
`spikes/x4_gate/README.md`; any profile serves):

```bash
hf download unsloth/Qwen3-30B-A3B-GGUF Qwen3-30B-A3B-Q4_K_M.gguf          # prints the file's path in the HF cache
uv run python -m holon_tools profile <that path> --name profile_x4_gate.json --max-seqs 8 --max-model-len 4096 \
  --host-memory 21474836480
uv run holon serve unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M --host-memory 21474836480
```

## Qwen3.8-Flash-Next

ISTA-DASLab's GSQ-RCO quantizations of Qwen3.8-Flash-Next (`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, each
split in two files): IQ2_XS on the RTX 5080 (16 GB), IQ3_S on the RTX 5090 (32 GB). Their locks offload the experts,
keep the PLE n-gram table (26.8 GiB) and the token embeddings in pinned host RAM, serve 33,792 tokens of context to
one sequence, and draft three tokens a step with the model's MTP head (`docs/dev/v1-flash-next.md` §20.1, §22.7).
The GGUF files carry no MTP head: the drafter reads it from a pack of its own (`mtp.path`), which Strata's public
tools build from the BF16 checkpoint `Qwen/Qwen3.8-Flash-Next` (only the head's tensors are downloaded) and Holon's
rewrites. From the install above (the RTX 5080's file; on the RTX 5090, `IQ3_S`, `86000000000` and `0.985`):

```bash
uv sync --frozen --group dev                         # the lock tools
hf download ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF --include "IQ2_XS/*"

git clone https://github.com/Niko1221/Strata.git strata && git -C strata checkout 296b59d
uv run python strata/tools/mtp_fetch.py fetch --out mtp
STRATA_GGUF_PY=$(uv run python -c "import gguf, pathlib; print(pathlib.Path(gguf.__file__).parents[1])") \
  uv run python strata/tools/mtp_pack.py --src mtp --experts q2_0 --out mtp/mtp-q2_0.gguf
uv run python -m holon_tools mtp-pack mtp/mtp-q2_0.gguf mtp/mtp-q2_0-q8.gguf

uv run python -m holon_tools profile <first file> --max-seqs 1 --max-model-len 4096 --memory-fraction 0.97 \
  --host-memory 70000000000 --plugin-options rows.place=ple:host,token_embd:host ggml_cuda.pass_rows=4096 \
  moe.ring_lend=tail
uv run holon serve ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF --host-memory 70000000000 --memory-fraction 0.97 \
  --plugin-options mtp.path=$PWD/mtp/mtp-q2_0-q8.gguf
```

- **The MTP pack.** Strata's `mtp_fetch.py` reads the head's 31 tensors from the checkpoint's pinned revision and
  checks each one's sha256; its `mtp_pack.py`, run on Holon's environment's gguf-py, writes the pack whose sha256 is
  `a22206a4d74eafbf31063a84610292535fb3b49af9a339963817fd04aed81e4d` (`docs/dev/v1-flash-next.md` §13).
  `holon_tools mtp-pack` rewrites the head's ten large dense projections from BF16 to Q8_0, as Strata's own draft head
  runs them (§21.11), and prints the result's sha256: from that pack it starts with `a55280a0417d4b19`, the pack the
  locks name (`mtp-q2_0-q8@a55280a0417d4b19`). Keep the name `mtp-q2_0-q8.gguf`: under another, the start logs
  `differs from the lock: source:mtp.path`. The pack changes only the drafts; the verify decides the output.
- **The routing profile.** `<first file>` is the `-00001-of-00002.gguf` file `hf download` placed under its snapshot
  directory, `<HF cache>/hub/models--ISTA-DASLab--Qwen3.8-Flash-Next-GSQ-RCO-GGUF/snapshots/<revision>/IQ2_XS/`; the
  profile's default name is the one the lock names. The locks were measured with a profile of 60 task prompts, 128
  tokens each (§20.1); any profile serves the same output.
- **Without the pack** the drafter finds no head in the model files, the lock's start fails with a plugin error, and
  the server falls back to defaults, which cannot hold the model: it exits. `--max-draft-tokens 0` serves without
  drafting, and without the pack.
- **A host short of RAM** can read the PLE table from the model file on the SSD instead:
  `--plugin-options rows.place=ple:ssd,token_embd:host cpu_task.io_threads=4`, with the table's 26.8 GiB less of
  `--host-memory`. The output is byte-identical to the RAM tier's; on the RTX 5090 it cost 16% / 26% of prefill at 1K /
  4K and 6–7% of decode with MTP (`docs/dev/v1-flash-next.md` §22.1). No lock sets it.
- **Accuracy** (`docs/dev/v1-flash-next.md` §22.4, §22.10): the locks pass L2 against the judge's golden and floors;
  GPU self-consistency (decode against prefill, chunked against one-shot, MTP against plain) passes on both cards; on
  72 hard task items (greedy, thinking on) Holon answered 70 correctly on the RTX 5080 and 72 on the RTX 5090, Strata
  70 on each.

## KV tiers

The Qwen3.8-27B locks of the RTX 5080 and RTX 5090 keep a host level under the prefix cache (`std.state_tier`,
[docs/dev/v1-kv-tiers.md](docs/dev/v1-kv-tiers.md)): the KV pages and recurrent-state checkpoints that VRAM evicts move
to 16 GiB of pinned host RAM, and a request whose prefix is there loads them back instead of recomputing that part of
its prompt. The locks also checkpoint the hybrid's recurrent state at branch points and at the end of each answer
(`recurrent.policy` `prompt_end,branch,answer:1`). Launch with `--host-memory 17179869184` or more; without it the
lock's start is refused and the server falls back to defaults.

On an agent trace whose sessions are parked and resumed (`agent_park`; the RTX 5080's Qwen3.8-27B UD-IQ3_S), time to
first token p50 fell to 0.047× the VRAM-only engine's and throughput rose 2.26×; against `llama-server --cache-ram`
sized to serve the trace, TTFT p50 was 0.122× and throughput 1.79× (`docs/dev/v1-kv-tiers.md` "Step 4",
`evidence/k4_agent_park_rtx-5080_2026-10-19.json`; on the tier lock as first written, at 49,152 tokens of context
where the shipped lock serves 40,960). The tier moves state without changing it (tier invariance and run-identity,
through host and SSD levels); the `branch` and `answer` checkpoints change where a resumed prompt is recomputed from,
so they were checked within bounds (KL), not bitwise. An SSD level below the host level (`state_tier.levels=ssd`,
`state_tier.ssd.dir`) is an option no lock sets.

## First start

After `holon prebuild`, a first start compiles only Triton kernels (12–40 per model, cached on disk for later starts)
and FlashInfer's CuTe DSL norms, which compile in memory at every start, about 0.3 s per norm shape and about 1 s per
start. A file outside the HF cache is hashed once to find its lock; the hash is remembered in
`~/.cache/holon/weights`.

## Known limitations

- One GPU per server. Offload covers a MoE model's experts (and Qwen3.8-Flash-Next's PLE table and token
  embeddings); the rest of the model and its KV cache must fit the card.
- **Host RAM:** the offload and KV-tier locks start only with `--host-memory` (Supported GPUs and models), pinned for
  the server's lifetime: 70 GB for Qwen3.8-Flash-Next on the RTX 5080 and 86 GB on the RTX 5090, besides what the
  rest of the host needs.
- **Offload locks serve one sequence** (`max_seqs` 1): concurrent requests wait their turn. `--max-seqs N` serves
  more at a smaller expert cache, its speed unmeasured; the RTX 5080's Qwen3.8-Flash-Next deployment ran 2 sequences at
  `--memory-fraction 0.9` in its exactness checks and did not start at 64 (`docs/dev/v1-flash-next.md` §22.3).
- **Context.** The Qwen3.8-27B locks serve 49,152 tokens (40,960 on the RTX 5080); Qwen3.8-Flash-Next's and the RTX
  5080's Qwen3-30B-A3B offload lock 33,792; the other 8B and 30B locks 8,192 or 16,384 (`holon models` and the lock
  files show each); without a lock, min(the model's trained context, 32768). `holon serve <model> --max-model-len N`
  changes it, and a context that does not fit the card's memory is clamped to what does (logged). Qwen3.8-Flash-Next's
  KV stays in VRAM: streaming a sparse-attention layer's KV through host RAM, as Strata does from 64K contexts, is not
  built (`docs/plan.md` §10, "KV streaming for sparse-attention mixers").
- **A file a lock names but the host lacks fails the start slowly.** Without its routing profile an offload lock's
  start fails with an error the server treats as transient: it tries 4 times (each up to 600 s), then falls back to
  defaults, which cannot hold the model either, and exits. A KV-tier lock without `--host-memory` is retried the same
  way before the defaults serve. Make the profile and pass the flags first ([Offload](#offload),
  [Qwen3.8-Flash-Next](#qwen38-flash-next)).
- **Speculation is off in every lock but Qwen3.8-Flash-Next's** (which drafts with MTP: Performance). On the
  Qwen3.8-27B, MTP gave 11–45% lower TPOT on a sampled Claude-Code-shaped trace, but a speculating stream's longest
  pauses (p95 per-request maximum gap) grew 1.1–3.1×, past the ≤ 1.10× rule fixed before measuring
  (`docs/dev/release.md` §16.6). Turn it on with the flags under Options.
- **Structured output on Qwen3.8-Flash-Next is unverified:** its locks select no grammar processor (Structured
  output).
- **A long prompt stalls the streams already decoding** on the locks that set `chunk_with_decode` 0: every 8B and 30B
  lock. Behind a 30K-token prompt (with the context raised to fit it) the stall was 3.2–11.8 s; `holon serve <model>
  --chunk-with-decode 512` mixes decoding into those prefill steps and cut it 23–37×, for 11–22% more time to that
  prompt's first token (48–54% on two RTX 4090 cells) and 2–15% of its throughput (`docs/dev/release.md` §16.5). The
  Qwen3.8-27B locks set 512.
- **On an RTX 5080 that also drives a display**, the Qwen3.8-27B's 40,960-token context leaves ≈ 260 MiB of headroom
  (its state tier's staging units are kept first) and may be clamped to what fits (logged); Claude Code's ≈ 31K side
  requests need it above ≈ 32K.
- **Traces are run-identical on one machine, not across machines:** two RTX 3090 desktops recorded different
  Qwen3.8-27B snapshot traces at the same lock, both within the judge's L2 budget (`docs/dev/release.md` §19.1).
- **No authentication:** the server checks no API key; do not expose it beyond a trusted network.
- **Every start compiles FlashInfer's CuTe DSL norms** in memory (about 1 s; First start).
- **M0.6's low-precision files on the RTX 5090 and RTX 3090** (8B NVFP4 and FP8, 30B NVFP4, 30B MXFP4_MOE) run with
  borrowed locks, unverified on those cards.
- Under a grammar, an output can run on in whitespace, digits or a string the schema admits until `max_tokens`: on the
  27B with MTP, 2.66% of constrained outputs were truncated against 0.33% unconstrained, over a 2-point gate that was
  accepted as a recorded shortfall. Guards against such runaways are deferred; set `max_tokens`.
- DFlash2 speculation is an opt-in, in no lock ([docs/dev/m04-spec.md](docs/dev/m04-spec.md)).
- Stop the server before a `git pull`: the install is editable, so a supervisor restart would import the new code.

## Performance

### Resident models (the release)

Measured at release step 8 on one desktop per card, each clock-screened before and after, with Holon serving from that
card's lock as it stood then. The shipped locks differ from those in the Qwen3.8-27B's context and in the FlashInfer
plugin's `src`, with byte-identical snapshot traces (`docs/dev/release.md` §19.1–§19.2). **Ratios are the claim**: absolute tokens per second belong to one board. Each ratio below is Holon's
advantage (above 1, Holon is faster) as a range over the four cards; the tables per card, file and trace, rendered by
`bench/report.py` from the evidence records they link, are in
[docs/dev/release.md §17](docs/dev/release.md#17-as-built-step-8), with the method and the absolute numbers per board.

| comparison | Holon's advantage |
|---|---|
| **The release gate:** in-process decode (tg128) against `llama-bench` on the same GGUF file and card, at llama.cpp `cbe28bb3` (the commit Holon's GGUF kernels vendor), depth 0 and 8192, 3 ABBA rounds | 1.04–1.11× on every cell |
| the same, prefill (pp512) | 1.00–1.15× at depth 0, 0.89–1.07× at depth 8192 |
| `holon serve` against `llama-server` (the same pin, matched to the lock's batch, step budget, KV format and KV size), Qwen3-8B Q4_K_M and the card's Qwen3.8-27B, TPOT p50 | 1.07–1.15× at batch 1, 1.20–1.66× at batch 8, 1.33–1.77× on the agent trace |
| the same, output throughput | 1.03–1.90× |
| the same, TTFT of the 30K-token prompt, Qwen3.8-27B | 0.98–1.07× (both engines decode the other streams meanwhile) |
| the same, TTFT of the 30K-token prompt, Qwen3-8B Q4_K_M | 1.03–1.17×, where Holon's lock stalls those streams instead (Where Holon loses) |
| `holon serve` against SGLang 0.5.21 and FreeToken `74a38777`, Qwen3-8B bf16 | TPOT p50 0.99–1.02× (0.82× on the RTX 4090's long prompt), throughput 1.00–1.03× |
| the Qwen3.8-27B with MTP (off in the locks) against plain decoding, `holon serve`, TPOT p50, greedy | 1.51–1.60× at batch 8, 1.27–1.88× on the agent trace |

The gate is ≥ 0.8× the pin on every metric of every GGUF cell, and every cell passes it. Serving runs greedy on
natural-text prompts in 4 ABBA rounds (decode-only files: 3): `decode_b1` / `decode_b8` are 1 or 8 requests of 256 +
256 tokens, `agent` 24 multi-turn tool sessions, `long_prompt` a 30K-token prompt arriving behind three streams. A
second Holon arm against the first moved TPOT and throughput by at most 0.3%, and TTFT p50 by at most 3.5% outside
the long prompt.

Caveats:
- **TTFT against `llama-server` is not a clean comparison.** Before it starts a request, `llama-server` saves its
  previous slot to its RAM prompt cache (`--cache-ram`, on by default); in the one log examined the three streams ahead
  of the 30K prompt waited ≈ 2.65 s for it. Their TTFT ratios (4.7–43×) are mostly that wait, so the table gives the
  30K request's own. The TTFT ratios on the other traces (1.6–9.3× on the agent trace) were not checked for the same
  effect and carry the same caveat.
- **MTP's speedups are unpaired** (against the plain run of the same desktop), greedy, and the RTX 5080's batch 8 is
  inflated by repetitive output. Its single-stream figure is not quoted: a looping prompt inflates it on three cards.
- The longest pause's p99 rests on one request; on the short traces differences under about 30% are not claims.

Where Holon loses:
- **The longest pause on `decode_b8`:** 0.15–0.63× llama.cpp's, 0.06–0.27× SGLang's. Of 8 requests sent together
  over HTTP, `holon serve` starts the first 1–3 at once and pauses them for 57–320 ms while the rest prefill, on the
  8B / 30B files; on the 27B (`max_seqs` 4 on every engine) four requests wait for the first four. The baselines start
  all eight together.
- **The streams behind the 30K prompt** on the locks with `chunk_with_decode` 0 (Known limitations): the 8B Q4_K_M's
  longest pause is 0.09–0.14× llama.cpp's, and on the RTX 4090's bf16 8B 0.01–0.02× SGLang's and FreeToken's, with
  TPOT 0.82×. The 27B locks mix, and there Holon's pause is 3.6–4.0× shorter on three cards; on the RTX
  4090 the ratio is within its rounds' spread (≈).
- **First tokens on the NVFP4 files against SGLang:** 0.22–0.69× on the RTX 5080's 8B and the RTX 4090's 30B; the
  5080's 8B NVFP4 also decodes at 0.90× SGLang's speed at batch 8 (throughput 0.82×), and the 4090's 30B NVFP4
  throughput at batch 8 is 0.98×.
- **Smaller losses on FP8:** `decode_b8` TTFT 0.89–0.97× SGLang's and 0.94–0.96× FreeToken's, the RTX 5080's
  `decode_b1` TTFT 0.97× FreeToken's, the RTX 4090's `decode_b8` longest pause 0.94× FreeToken's; on the bf16 8B,
  the RTX 3090's `decode_b8` TTFT is 0.98× SGLang's.
- **Prefill at depth 8192** is under the pin on every 8B / 30B cell: 0.89–0.91× on the RTX 5080, 0.94–0.96× on the
  RTX 4090 and 5090, 0.98–0.99× on the RTX 3090 (the gate is 0.8×).

FreeToken failed to load the NVFP4 files on every card that ran them (`—` in the tables).

### Qwen3.8-Flash-Next against Strata

Measured at Qwen3.8-Flash-Next's close on one RTX 5080 (IQ2_XS) and one RTX 5090 (IQ3_S), at the code its locks pin,
with `bench/offload_ctx.py`: batch 1, a prompt of natural text of 1K, 4K or 32K tokens, then 256 greedy tokens (EOS
ignored); 8 prompts per context in 2 rounds, every arm a fresh process, the arms interleaved and reversed on odd
rounds. Holon runs each card's lock deployment in process (prefix cache off, the PLE table in host RAM);
Strata `296b59d` runs without its approximations, with its own calibration, drafting off or with its MTP head.
Holon's advantage in tokens per second, above 1 when Holon is faster; the bars, set before measuring, are 1.0 for
decode and 0.9 for prefill, and every row meets its bar ([docs/dev/v1-flash-next.md](docs/dev/v1-flash-next.md)
§22.10, `evidence/fn_close_final_2026-10-24.json`, key `d11`):

| row | RTX 5080, 1K / 4K / 32K | RTX 5090, 1K / 4K / 32K |
|---|---|---|
| decode, drafting off | 1.086 / 1.090 / 1.071 | 1.167 / 1.143 / 1.111 |
| decode, MTP | 1.087 / 1.141 / 1.116 | 1.067 / 1.001 / 1.000 |
| prefill, drafting off | 1.090 / 1.222 / 1.022 | 0.995 / 1.225 / 0.994 |
| prefill, MTP | 1.089 / 1.219 / 1.021 | 1.003 / 1.175 / 0.976 |

Holon decodes with MTP at 129–132 tok/s on the RTX 5080 and 175–194 tok/s on the RTX 5090.

Caveats:
- **The RTX 5090's MTP decode at 4K and 32K sits on its bar** (one of the two rounds 0.4–0.5% under it), with fewer
  tokens per verify pass than Strata's (2.78 against 2.93 at 4K, 3.21 against 3.28 at 32K).
- **Exact against exact.** Strata's defaults add approximations (a routed-expert tail skip, int8 KV, a suffix
  drafter), with which it decoded faster than its exact arms when first measured (`docs/dev/v1-flash-next.md` §13);
  the close did not measure Holon against them.

### Qwen3-30B-A3B offload on the RTX 5080

The RTX 5080's Qwen3-30B-A3B Q4_K_M lock (about half the experts cached: 52.7% at the offload milestone's close)
against FreeToken: decode 1.06 / 1.04 / 1.02× at 1K / 4K / 32K (`docs/dev/v1-offload.md` "Step 8, part 2",
`evidence/x8_close_rtx-5080_2026-10-18.json`); prefill 1.850 / 1.136 / 1.005× against FreeToken run at the lock's step
budget, and 0.959× against FreeToken's default at 32K, after the MoE prefill work (`docs/dev/v1-moe-prefill.md`,
`evidence/mp5_close_2026-10-20.json`), which moved the lock's decode by 0.994× (pooled over 18 prompt and context
pairs).

## Contributing

Rules and routing by task are in [AGENTS.md](AGENTS.md); adding a model is in
[docs/adding-a-model.md](docs/adding-a-model.md).

## License

Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)); third-party code is listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
