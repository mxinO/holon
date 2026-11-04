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
also serves Qwen3.8), Qwen3.8-Flash-Next (`models.qwen4_exp`, GGUF, with its MTP head) and GLM-5.3-Flash
(`models.glm5_next`, GGUF, text only, its MTP head unused), from HF safetensors directories or GGUF files. The recommended file per card and its lock
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
| GLM-5.3-Flash Maya-S-v2 IQ2_XXS (`peasantsmith/GLM-5.3-Flash-Maya-GGUF`) | this card's lock, offload | borrowed, unverified | this card's lock, offload | borrowed, unverified |

**This card's lock**: written and verified on this card (L2 against the judge's floor, the snapshot trace identical
over three runs), in the release environment (`docs/dev/release.md` §16, §19) or at a later milestone's close
(`docs/dev/v1-offload.md` "Step 8, part 2", `docs/dev/v1-kv-tiers.md` "Close (2026-10-20)",
`docs/dev/v1-flash-next.md` §22.7, `docs/dev/v1-glm.md` §13). GLM-5.3-Flash's locks are judged by L2 in its
reference form (bf16 references recorded on an RTX PRO 6000, with budgets fixed before their runs; `docs/dev/v1-glm.md`
D15a). Every lock but Qwen3.8-Flash-Next's selects the grammar processor. The Qwen3.8-27B
locks serve a context of 49,152 tokens (40,960 on the RTX 5080) without speculation (Known limitations).

**Host RAM.** Start every lock with `holon serve <model>` alone: what a lock keeps in pinned host RAM is pinned from
what the host has, never from a number you set ([docs/dev/v1-auto-config.md](docs/dev/v1-auto-config.md)). At its
start the server reads the RAM available once (`MemAvailable`, and a cgroup v2 container's limit less what it holds),
keeps max(6 GiB, 5% of the RAM) free of pinned memory for the rest of the host, and logs the budget; pinned pages
cannot be swapped, so the rest of the host would be. These locks pin host RAM (the figures are estimates until
measured; "other use" counts Holon's own processes, ≈ 3 GiB, est.):

| lock | pinned host RAM | RAM for full speed | with less RAM |
|---|---|---|---|
| Qwen3.8-27B, RTX 5080 and RTX 5090 | a 16 GiB KV tier ([KV tiers](#kv-tiers)) | 32 GB, ≤ 6 GiB other use | the tier takes what is left, down to its floor; below it, the defaults serve without a tier |
| Qwen3-30B-A3B Q4_K_M, RTX 5080 | the experts, ≈ 16.4 GiB ([Offload](#offload)) | 32 GB, ≤ ≈ 8 GiB other use | std's generic spill: the experts the RAM holds (`moe.ram=fit`), the rest read from the model file on the SSD; below the working floor refused, naming each term |
| Qwen3.8-Flash-Next IQ2_XS, RTX 5080 | the experts, the PLE table, the token embeddings, ≈ 61 GiB | ≈ 80 GB | the PLE table read from the model file on the SSD (byte-identical, slower prefill) and the experts by `moe.ram=fit` (all in RAM where they fit); where they do not, without the MTP head (decode 24–37% slower) |
| Qwen3.8-Flash-Next IQ3_S, RTX 5090 | the same, ≈ 74.5 GiB | 96 GB, ≤ ≈ 10 GiB other use | the same |
| GLM-5.3-Flash, RTX 5090 | the experts and the token embeddings, ≈ 81 GiB | 88 GiB (measured: the lock starts under an 88 GiB cap) | mode (ii): the share of the experts the host fits in RAM, the rest read from the model file on the SSD ([GLM-5.3-Flash](#glm-53-flash)); refused below 60 GiB of RAM |
| GLM-5.3-Flash, RTX 5080 | the same | ≈ 88 GiB (est., as the RTX 5090) | std's generic spill, as the Qwen3-30B-A3B row; refused below 60 GiB of RAM |

On every offload lock the RAM-short retry is std's generic spill (`moe.ram=fit`: the RAM tier sized from the host
budget, `docs/dev/v1-glm-ram.md`); a model plugin refines it where it measured its own (GLM on the RTX 5090). Its speed
on Qwen3-30B-A3B and Qwen3.8-Flash-Next is not measured yet.

`--host-memory BYTES` sets the budget instead (warned about where it cuts into what is kept free). The GPU's share
works the same way: `holon serve` sizes an offload lock's expert cache from the startup profile's measured peak and a
reserve (1 GiB on a card that drives a display or that other processes use, 256 MiB otherwise) and keeps today's 10%
margin elsewhere; `--memory-fraction F` sets a fixed share.

**Borrowed, unverified.** Locks are found by the checkpoint's build hash, so a file downloaded anywhere, under any
name, finds its lock. Without a lock for this card, `holon serve` borrows the nearest class's: the same GPU on another
toolchain, then the same SM, then the same toolchain. A borrowed lock's settings apply as they are and its kernel pins
become preferences (a provider that cannot run here falls back to the default choice, logged). It was not verified on
this card. If it does not start with a plugin or memory error, the server falls back to defaults at once. A borrowed
lock pins the host RAM of the lock it borrows (Host RAM above): on an RTX 4090, for example, the Qwen3.8-27B UD-IQ3_S
borrows the RTX 5080's lock and its KV tier.

**Served without a lock** (defaults, logged): every other checkpoint of these architectures, for example
Qwen3.5 4B / 9B and the bf16 Qwen3.8-27B and Qwen3-30B-A3B (large-memory paths). Defaults serve at most 8 sequences
(`holon serve <model> --max-seqs N` changes it) and a context of min(the model's trained context, 32768) tokens,
clamped to what fits.

## Install

Holon installs from source today; the pip install below comes with its first PyPI release.

This README describes `main`. The last release, v0.2.0
([docs/releases/RELEASE_NOTES_v0.2.0.md](docs/releases/RELEASE_NOTES_v0.2.0.md)), serves everything below: install
the tag (`--branch v0.2.0`) for it, or `main`. Its locks were propagated to its code with byte-identical snapshot traces.
Release steps 7–10's locks and records (v0.1.0) name commits of the release line as first built; the same commits are
on `main` under new SHAs (`docs/dev/evidence/v0_release_line_sha_map_2026-10-16.txt`;
`docs/dev/evidence/history_scrub_commit_map_2026-10-16.txt` maps older SHAs).

Host requirements: Linux x86-64, an NVIDIA driver supporting CUDA 13, the CUDA 13 toolkit (`nvcc` on `PATH` or under
`CUDA_HOME`), gcc / g++, git and [uv](https://docs.astral.sh/uv/). uv brings Python 3.12, torch 2.11 for CUDA 13.0 and
every pinned dependency from `uv.lock`.

```bash
git clone https://github.com/mxinO/holon.git holon && cd holon      # main; --branch v0.2.0 for the release
uv sync --frozen                     # Python 3.12, torch 2.11 + CUDA 13 and every pin from uv.lock
uv run holon prebuild                # GGUF kernels and FlashInfer modules for the visible GPUs, once
```

`holon prebuild` compiles the GGUF kernels and FlashInfer's modules for the SM of every visible GPU, once per install;
`holon prebuild --sm 86 89 120` names the targets instead, and `MAX_JOBS=N` caps the parallel compiles.
`uv sync --frozen --group dev` also installs the lock tools (`python -m holon_tools`), which make the offload locks'
routing profiles, and the test dependencies.
The locks were verified with modules built by Ubuntu 24.04's gcc 13, which defines `_FORTIFY_SOURCE=3` by default;
nvcc preprocesses device code with the host compiler's headers, so on another distribution build with
`NVCC_APPEND_FLAGS=-D_FORTIFY_SOURCE=3` (on SM 86 one FlashInfer module's code differs without it:
[docs/dev/v1-pip.md](docs/dev/v1-pip.md) §16).

### pip (from the first PyPI release)

**Not available yet: `holonlm` is not on PyPI.** This is how Holon installs from its first PyPI release: with its
native modules prebuilt for SM 86, 89 and 120 (RTX 30, 40 and 50 series, and the other cards of those architectures:
[Supported GPUs and models](#supported-gpus-and-models)), so no CUDA toolkit is needed. A GPU of another SM needs the
source install above.

Requirements: Linux x86-64, an NVIDIA driver for CUDA 13.0 (R580 or newer), CPython 3.12 with its headers and a C
compiler (Triton builds a small launcher per kernel at first use). The install downloads about 3.2 GB (torch and its
CUDA 13 libraries) and takes about 6.1 GB on disk.

On Ubuntu 24.04:

```bash
sudo apt install python3.12-venv python3.12-dev build-essential
python3.12 -m venv ~/holon && . ~/holon/bin/activate
pip install holonlm
holon serve unsloth/Qwen3-8B-GGUF
```

On any Linux with a C compiler, a uv-managed Python brings its own headers (and serves hosts whose Python is not 3.12):

```bash
uv tool install --python 3.12 holonlm
holon serve unsloth/Qwen3-8B-GGUF
```

Until that release, the seven wheels a release build makes (`scripts/release/build_wheels.py`, or
`.github/workflows/wheels.yml`'s artifact) install the same way by path, `pip install <dir>/*.whl`, which pins and
fetches the same third-party packages from PyPI.

Use a venv without extra package indexes: `holonlm` pins torch 2.11.0 for CUDA 13.0 (PyPI's) and every other package
to the versions the locks were verified with. Measured on the four tested desktops
(`docs/dev/evidence/pip_acceptance_2026-11-04.txt`): `pip install` took 82–143 s (595 s on one slower link) and
`uv pip install` 14–28 s; the first answer came 2–3 minutes after the install started (11 on the slower link), against
10–19 minutes for the source install, most of it `holon prebuild`. Served from the wheels, all 27 locks' snapshot
traces are byte-identical to the source install's, and decode speed is the same (median ratios 0.9958–1.0001).

What fails, and how:

- **No C compiler or Python headers:** the first start is refused at once, with the apt line above.
- **Another torch build** (a `+cu128` or `+cpu` torch from another index): refused, naming the command that replaces
  it: `pip install --force-reinstall --no-deps "torch==2.11.0+cu130" --index-url https://download.pytorch.org/whl/cu130`.
- **A GPU of another SM:** refused at start, naming the source install.
- **Python other than 3.12, aarch64, macOS, Windows:** pip finds no matching distribution.

### Docker

The `Dockerfile` installs Holon from its wheels on Ubuntu 24.04, for hosts without a CUDA toolkit: no toolkit and no
prebuild in the image (≈ 7.0 GB), whose native modules come prebuilt ([docs/dev/v1-pip.md](docs/dev/v1-pip.md)).
Build it from a directory holding a release's seven wheels (`scripts/release/build_wheels.py` makes them), or, once
Holon is on PyPI, with no context at all, which installs this tree's version from PyPI:

```bash
docker build -f Dockerfile -t holon <directory of the wheels>
docker build -t holon - < Dockerfile                                      # from PyPI, once published
docker run --gpus all -p 8000:8000 -v <model dir>:/models holon serve /models/<file> --host 0.0.0.0
docker run --gpus all -p 8000:8000 -e HF_HOME=/hf -v <hf cache dir>:/hf holon \
  serve unsloth/Qwen3.8-27B-GGUF --host 0.0.0.0                            # a hub model, kept in <hf cache dir>
```

The v0.2.0 tag's `Dockerfile` is the source install's image instead (its environment, and a prebuild for SM 86, 89
and 120; ≈ 14.8 GB): `git archive --format=tar v0.2.0 | docker build -t holon:v0.2.0 -` builds it.

The image's `HOME` is `/opt/holon/user`, writable by any user, for every tool in it (the `Dockerfile` after v0.1.1:
v0.1.1's image has none). Mount a directory at `/opt/holon/user/.cache/holon` to keep Holon's own cache across
containers: the hashes of weight files (each start would otherwise re-read the file) and the files locks name.
Containers without a memory limit do not see each other's starts: give each a limit (`--memory`), which the server
reads.

```bash
docker run --gpus all --user "$(id -u):$(id -g)" -v <cache dir>:/opt/holon/user/.cache/holon \
  -p 8000:8000 -v <model dir>:/models holon:<tag> serve /models/<file> --host 0.0.0.0
```

## Quickstart

```bash
uv run holon models                                   # the models locked for this card, by name
uv run holon serve unsloth/Qwen3.8-27B-GGUF
```

The second line serves this card's locked file, downloaded into the HF cache; on the RTX 5080 and RTX 5090 its lock
keeps a KV tier in host RAM (Host RAM).

`holon serve` takes a local file or directory, or a Hugging Face Hub reference, downloaded into the HF cache once:

| reference | serves |
|---|---|
| `unsloth/Qwen3.8-27B-GGUF:UD-IQ3_S` | the repository's GGUF file whose name ends in `-UD-IQ3_S` (any case; `:Q4_K_M` takes `…-Q4_K_M.gguf` over `…-UD-Q4_K_M.gguf`) |
| `unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_S` | that file, as `holon models` names it |
| `unsloth/Qwen3.8-27B-GGUF` | the file this card has a verified lock for (`holon models`: "this card's lock"); else the repository's only GGUF file, else an error listing them |
| `Qwen/Qwen3-8B` | an HF safetensors checkpoint |

The log names the file chosen and its size; a file larger than the GPU's memory is refused before downloading, unless
this card has a lock for it (the lock decides), `--host-memory` is given (offload) or `--no-size-check`. The hub's own
settings apply: `hf auth login` for gated repositories, `HF_HOME` for the cache, and `HF_HUB_OFFLINE=1` to serve cached models without the network (a tag then
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
RTX 4090 (UD-Q4_K_M)**, in 13–16 turns and 2–5 minutes each (`docs/dev/evidence/agent_demo_rtx-5080_2026-10-16.json`,
`docs/dev/evidence/agent_demo_rtx-4090_2026-10-15.json`; `docs/dev/release.md` §18). Claude Code sends no temperature, so the
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
- **Pinned host memory** is a host's, never a lock's: `holon serve` budgets it from the RAM available (Host RAM);
  `--host-memory BYTES` sets the budget. A start whose pools do not fit is refused once with the bytes they need and
  what to do. One `holon serve` of a user at a time reads the RAM and pins (`~/.cache/holon/pin.lock`): a second start
  waits for the first to be ready; two users' starts at the same moment are not serialized.
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
[docs/dev/v1-offload.md](docs/dev/v1-offload.md)): the RTX 5080's Qwen3-30B-A3B Q4_K_M lock, both
Qwen3.8-Flash-Next locks and both GLM-5.3-Flash locks.

- **How:** every expert lives in pinned host RAM, and a VRAM cache holds the most used ones (`placement.lfu@1`,
  least-frequently-used, seeded at start from a routing profile). A decode step fetches the experts it misses over
  PCIe; a long prompt step streams the missing ones through cache slots it lends for the step.
- **Exact:** placement never changes a result. The output is byte-identical to the same model served all-resident,
  checked on an RTX A5000 (Qwen3-30B-A3B, every placement and policy) and an RTX PRO 6000 (Qwen3.8-Flash-Next and
  GLM-5.3-Flash, which fit there), and identical across cache sizes and placements over three runs on the RTX 5080 and 5090
  (`docs/dev/v1-offload.md` "Step 8, part 2", acceptance 2; `docs/dev/v1-flash-next.md` §22.3). Like every trace, it is
  run-identical on one machine, not across machines (Known limitations).
- **Host memory:** pinned from the RAM the host has (Host RAM under
  [Supported GPUs and models](#supported-gpus-and-models)); a start whose pools do not fit is refused with a report per
  pool and what to do.
- **One sequence:** the offload locks serve `max_seqs` 1, so concurrent requests queue. More sequences take VRAM from
  the expert cache: on the RTX 5080's Qwen3.8-Flash-Next lock, 26.2% of the experts are cached at 1 sequence and 21.9%
  at 4 (`docs/dev/v1-flash-next.md` §22.7).
- **Routing profile:** a JSON file of routing counts that seeds the cache; it only decides what the cache holds
  first, so outputs do not depend on it. Holon ships the profiles its locks were measured with
  (`std/holon_std/tiered/profiles/`, matched to the model file); a lock's named profile beside the model file
  (`moe.placement.profile`) is read first, and without either the cache starts cold, logged
  (`--plugin-options moe.placement.profile=none` starts it cold on purpose). `python -m holon_tools
  profile <model file>` makes one from greedy generations over `judge/prompts/v1.jsonl` (`--prompts`), starting from
  the model's lock for this card, and writes it beside the file as `<file stem>.profile.json` (`--name` sets another);
  `--ship` imports one into the shipped profiles.

```bash
uv run holon serve unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M       # the RTX 5080's offload lock, its profile shipped
```

## Qwen3.8-Flash-Next

ISTA-DASLab's GSQ-RCO quantizations of Qwen3.8-Flash-Next (`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, each
split in two files): IQ2_XS on the RTX 5080 (16 GB), IQ3_S on the RTX 5090 (32 GB). Their locks offload the experts,
keep the PLE n-gram table (26.8 GiB) and the token embeddings in pinned host RAM, serve 33,792 tokens of context to
one sequence, and draft three tokens a step with the model's MTP head (`docs/dev/v1-flash-next.md` §20.1, §22.7).
The GGUF files carry no MTP head: the drafter loads an already-packed head as it is; Holon does not pack or quantize
it. Strata's public tools make one from the BF16 checkpoint `Qwen/Qwen3.8-Flash-Next` (only the head's tensors are
downloaded). `holon serve` finds it by the identity the locks name, in
`~/.cache/holon/sources/mtp-q2_0@a22206a4d74eafbf/` (or wherever `--plugin-options mtp.path=FILE` says); make it once
into that directory (the RTX 5080's file; on the RTX 5090, `IQ3_S`):

```bash
uv sync --frozen --group dev                         # the lock tools
git clone https://github.com/Niko1221/Strata.git strata && git -C strata checkout 296b59d
uv run python strata/tools/mtp_fetch.py fetch --out mtp
mkdir -p ~/.cache/holon/sources/mtp-q2_0@a22206a4d74eafbf
STRATA_GGUF_PY=$(uv run python -c "import gguf, pathlib; print(pathlib.Path(gguf.__file__).parents[1])") \
  uv run python strata/tools/mtp_pack.py --src mtp --experts q2_0 \
  --out ~/.cache/holon/sources/mtp-q2_0@a22206a4d74eafbf/mtp-q2_0.gguf

uv run holon serve ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF
```

- **The MTP pack.** Strata's `mtp_fetch.py` reads the head's 31 tensors from the checkpoint's pinned revision and
  checks each one's sha256; its `mtp_pack.py`, run on Holon's environment's gguf-py, writes the experts in Q2_0 and
  the dense projections in BF16, sha256 `a22206a4d74eafbf31063a84610292535fb3b49af9a339963817fd04aed81e4d`
  (`docs/dev/v1-flash-next.md` §13): the pack the locks name (`mtp-q2_0@a22206a4d74eafbf`). Keep the name
  `mtp-q2_0.gguf`: under another, the start logs `differs from the lock: source:mtp.path`. The pack changes only the
  drafts; the verify decides the output.
- **The routing profile.** Holon ships the profiles the locks were measured with (counted with this pack); any profile
  serves the same output.
- **Without the pack** the server serves the lock without drafting, warned about once with the cost (decode 24–37%
  slower on an RTX 5080 and 5090, 1K–32K) and these commands. `--max-draft-tokens 0` serves without drafting and
  never looks for the pack.
- **A host short of RAM** reads the PLE table from the model file instead of 26.8 GiB of pinned RAM: `holon serve`
  retries the lock with `rows.place=ple:ssd,token_embd:host` after a start refused for host memory, unless the file
  is on a rotating or network disk. The output is byte-identical to the RAM tier's; it costs 66% / 80% / 81% of prefill
  at 1K / 4K / 32K and 2–3% of decode with MTP on the RTX 5090, and 60% / 71% / 71% of prefill and 2% of decode on the
  RTX 5080 (`docs/dev/v1-auto-config.md` §9).
- **Accuracy** (`docs/dev/v1-flash-next.md` §22.4, §22.10): the locks pass L2 against the judge's golden and floors;
  GPU self-consistency (decode against prefill, chunked against one-shot, MTP against plain) passes on both cards; on
  72 hard task items (greedy, thinking on) Holon answered 70 correctly on the RTX 5080 and 72 on the RTX 5090, Strata
  70 on each.

## GLM-5.3-Flash

The Maya-S v2 IQ2_XXS quantization of GLM-5.3-Flash (`peasantsmith/GLM-5.3-Flash-Maya-GGUF`, `Maya-S-v2-IQ2_XXS/`,
three files, 96.5 GB) on the RTX 5090 (32K-token prompt steps) and the RTX 5080 (8K steps), one sequence of 33,792
tokens of context, no speculation (`docs/dev/v1-glm.md`). Their locks offload the experts: every expert in pinned host
RAM, the most used cached in VRAM.

```bash
uv run holon serve peasantsmith/GLM-5.3-Flash-Maya-GGUF:Maya-S-v2-IQ2_XXS
```

- **A host short of the lock's RAM (mode (ii)).** On the RTX 5090, a start refused for host memory retries the lock
  with RAM holding the share of the experts the host budget fits (`moe.ram=fit`, `docs/dev/v1-glm-ram.md`) and the
  rest read from the model file on the SSD as they are needed (exclusive tiers, `docs/dev/v1-glm.md` D10;
  `plugins/holon_plugins/models/glm5_next/deploy.py`), logged with its share and cost. The output is the same as the
  lock's (placement only: checked byte-identical on the card). It reads the SSD at every prompt and at some decode
  tokens, so keep the files on a local NVMe drive. **Below 60 GiB of RAM it refuses to start** (the host's RAM:
  `MemTotal`, or a lower cgroup limit; a 64 GB machine reports about 62–63 GiB), naming its minimum and the host's RAM.
  `--plugin-options moe.ram_min=0` serves below it anyway, with std's generic spill instead of GLM's measured set
  (slower: the table's 48 and 56 GiB rows were GLM's set).
  Speed by the host's RAM, measured on one RTX 5090 with the job's RAM capped (medians: 3 rounds at 56–72 GB, 1
  round at 48, 80 and 88; the lock from about 88 GiB, where the host budget holds every expert; tok/s, 1K / 32K;
  `docs/dev/v1-glm-ram.md` §10):

  | RAM | RAM tier (share of all experts) | prefill | decode |
  |---|---|---|---|
  | 48 GiB (refused at start now) | 43% | 109 / 2,195 | 11.7 / 9.3 |
  | 56 GiB (refused at start now) | 53% | 161 / 2,660 | 17.0 / 13.4 |
  | 64 GiB | 63% | 292 / 3,079 | 24.9 / 19.5 |
  | 72 GiB | 73% | 606 / 3,196 | 33.1 / 24.2 |
  | 80 GiB | 83% | 647 / 3,268 | 35.2 / 30.2 |
  | 88 GiB and more (the lock) | all | 710 / 3,454 | 41.7 / 36.8 |

  1K prefill doubles from 64 to 72 GiB and 1K decode rises fastest below 72 GiB; 32K decode keeps rising to about 80
  GiB. From about 80 GiB to the lock speed is flat, and the lock is about 15–20% faster at decode. The start warns
  below 72 GiB (the RAM sizes here are the job's cgroup limits in GiB). The 48 and 56 GiB rows were measured before
  the minimum: the start refuses them now.
- **First start:** about a minute to ready on an RTX 5090 at mode (ii), reading about 79 GiB from the drive (the
  checkpoint about once; `docs/dev/v1-glm.md` §13 step 7). A restart rereads the experts but not the file's hash.
- **Accuracy** (`docs/dev/v1-glm.md` §13): the locks pass L2 in its reference form on both prompt sets (the RTX 5090:
  top-1 94.82% / 91.81%, the RTX 5080: 95.15% / 91.91%, against budgets of 91.90% / 86.78%); GPU self-consistency
  (decode against prefill, chunked against one-shot) and the cover of DSA, the router, KDA and mHC on the model's own
  inputs pass on both cards; on 72 hard task items (greedy, thinking on) Holon answered 69 correctly on the RTX 5090,
  Project Maya 69.
- **Structured output:** the locks select the grammar processor. On a 500-schema sample of JSONSchemaBench it accepts
  307 schemas, and every constrained output that ran to its end conformed (283; unconstrained, 45.5% of 411 did);
  7.8% of constrained outputs were truncated at `max_tokens` 2048, against 17.0% unconstrained
  (`docs/dev/evidence/conformance_glm-5.3-flash-maya-s-v2-iq2_xxs-500_rtx-5090_2026-10-31.json`).

## KV tiers

The Qwen3.8-27B locks of the RTX 5080 and RTX 5090 keep a host level under the prefix cache (`std.state_tier`,
[docs/dev/v1-kv-tiers.md](docs/dev/v1-kv-tiers.md)): the KV pages and recurrent-state checkpoints that VRAM evicts move
to 16 GiB of pinned host RAM, and a request whose prefix is there loads them back instead of recomputing that part of
its prompt. The locks also checkpoint the hybrid's recurrent state at branch points and at the end of each answer
(`recurrent.policy` `prompt_end,branch,answer:1`). On a host with less RAM free the level takes what is left (logged),
down to its floor; below it the server falls back to defaults, without a tier.

On an agent trace whose sessions are parked and resumed (`agent_park`; the RTX 5080's Qwen3.8-27B UD-IQ3_S), time to
first token p50 fell to 0.047× the VRAM-only engine's and throughput rose 2.26×; against `llama-server --cache-ram`
sized to serve the trace, TTFT p50 was 0.122× and throughput 1.79× over 3 rounds (`docs/dev/v1-kv-tiers.md` "Step 4",
row 3d; the VRAM-only figures in `docs/dev/evidence/k4_agent_park_rtx-5080_2026-10-19.json`; on the tier lock as first
written, at 49,152 tokens of context
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
- **Host RAM:** the offload and KV-tier locks pin host RAM for the server's lifetime (Host RAM under Supported GPUs
  and models): ≈ 61 GiB for Qwen3.8-Flash-Next on the RTX 5080 and ≈ 74.5 GiB on the RTX 5090, ≈ 81 GiB for GLM-5.3-Flash (≈ 50 GiB
  in its mode (ii)) (est.). The budget is
  read from `MemAvailable` and a cgroup v2 limit: under cgroup v1, or with ZFS (whose ARC `MemAvailable` does not
  count), pass `--host-memory`. Starts on one host queue behind each other while one reads the RAM and pins. A
  desktop's systemd-oomd may kill a server whose host runs short regardless.
- **Offload locks serve one sequence** (`max_seqs` 1): concurrent requests wait their turn. `--max-seqs N` serves
  more at a smaller expert cache, its speed unmeasured; the RTX 5080's Qwen3.8-Flash-Next deployment ran 2 sequences at
  `--memory-fraction 0.9` in its exactness checks and did not start at 64 (`docs/dev/v1-flash-next.md` §22.3).
- **Context.** The Qwen3.8-27B locks serve 49,152 tokens (40,960 on the RTX 5080); Qwen3.8-Flash-Next's and the RTX
  5080's Qwen3-30B-A3B offload lock and GLM-5.3-Flash's 33,792; the other 8B and 30B locks 8,192 or 16,384 (`holon models` and the lock
  files show each); without a lock, min(the model's trained context, 32768). `holon serve <model> --max-model-len N`
  changes it, and a context that does not fit the card's memory is clamped to what does (logged). Qwen3.8-Flash-Next's
  KV stays in VRAM: streaming a sparse-attention layer's KV through host RAM, as Strata does from 64K contexts, is not
  built (`docs/plan.md` §10, "KV streaming for sparse-attention mixers").
- **Speculation is off in every lock but Qwen3.8-Flash-Next's** (which drafts with MTP: Performance); GLM-5.3-Flash's
  MTP head is not used. On the
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
  accepted as a recorded shortfall; GLM-5.3-Flash truncated 7.8% of constrained outputs at `max_tokens` 2048
  (17.0% unconstrained, its free outputs running past the JSON). Guards against such runaways are deferred; set
  `max_tokens`.
- **GLM-5.3-Flash's mode (ii) is not a lock:** its share of RAM is sized at start, measured on one RTX 5090 from 48 to
  88 GiB (`docs/dev/v1-glm-ram.md` §10). At 64 GiB its decode is below Project Maya's at 4K and 32K (Performance).
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
§22.10, `docs/dev/evidence/fn_close_final_2026-10-24.json`, key `d11`):

| row | RTX 5080, 1K / 4K / 32K | RTX 5090, 1K / 4K / 32K |
|---|---|---|
| decode, drafting off | 1.086 / 1.090 / 1.071 | 1.167 / 1.143 / 1.111 |
| decode, MTP | 1.087 / 1.141 / 1.116 | 1.067 / 1.001 / 1.000 |
| prefill, drafting off | 1.090 / 1.222 / 1.022 | 0.995 / 1.225 / 0.994 |
| prefill, MTP | 1.089 / 1.219 / 1.021 | 1.003 / 1.175 / 0.976 |

Holon decodes with MTP at 129–132 tok/s on the RTX 5080 and 175–194 tok/s on the RTX 5090.

**Since then, the MTP head as packed.** The MTP rows above ran the head's dense projections in Q8_0, a step Holon no
longer takes: it loads Strata's pack as it is (BF16 projections). Re-measured against that Q8_0 head on the same card
and build (`docs/dev/evidence/fn_mtp_strata_pack_ab_2026-10-30.json`), MTP decode is 0.991 / 0.976 / 0.988 on the RTX
5080 and 0.990 / 0.982 / 0.989 on the RTX 5090, with the
same tokens per verify pass and prefill unchanged; scaled, the RTX 5090's MTP decode at 4K and 32K is about 0.98 / 0.99
against Strata, under its 1.0 bar, until a quantization step returns.

Caveats:
- **The RTX 5090's MTP decode at 4K and 32K sits on its bar** (one of the two rounds 0.4–0.5% under it), with fewer
  tokens per verify pass than Strata's (2.78 against 2.93 at 4K, 3.21 against 3.28 at 32K).
- **Exact against exact.** Strata's defaults add approximations (a routed-expert tail skip, int8 KV, a suffix
  drafter), with which it decoded faster than its exact arms when first measured (`docs/dev/v1-flash-next.md` §13);
  the close did not measure Holon against them.

### GLM-5.3-Flash against Project Maya

Measured at GLM-5.3-Flash's close on one RTX 5090 with the Maya-S v2 IQ2_XXS file on the node's NVMe drive, with
`bench/offload_ctx.py`: batch 1, a prompt of natural text of 1K, 4K or 32K tokens, then 256 greedy tokens; 8 prompts
per context in each of 5 rounds after a warm-up, every arm a fresh process, the arms interleaved and reversed on odd
rounds, the page cache evicted before each. Project Maya v1.1.27 runs its default settings. Two hosts: **mode (i)**,
RAM holding every expert (the RTX 5090 lock, `holon serve` with auto-config's values), and **mode (ii)**, a 64 GB host
(the job's RAM capped at 64 GiB for both engines; Holon at the spill's options ([GLM-5.3-Flash](#glm-53-flash)) entered
by hand with a fixed `--memory-fraction 0.99` and `--host-memory 59000000000`, where the spill takes auto-config's
values: about the same VRAM on this headless card, about 100 fewer cached experts on a card that drives a display;
Maya with its RAM tier at 50 GB). The mode (i) rows ran the RTX 5090 lock one revision before the shipped one, whose
snapshot trace is byte-identical (a startup-only change between them). A cell is the median over rounds of each round's ratio of medians;
where the ratio of means differs by more than 2 points it is in parentheses (slow requests are real latency). Holon's
advantage in tokens per second, above 1 when Holon is faster; the bars, set before measuring, are 1.0 for decode and
0.9 for prefill ([docs/dev/v1-glm.md](docs/dev/v1-glm.md), `docs/dev/evidence/glm_close_2026-10-31.json`):

| row | 1K / 4K / 32K |
|---|---|
| decode, mode (i) | 1.178 / 1.106 / 1.052 |
| prefill, mode (i) | 1.077 / 1.017 / 1.074 |
| decode, mode (ii) (64 GB host) | 1.040 (0.985) / 0.982 / 0.923 (0.897) |
| prefill, mode (ii) (64 GB host) | 1.096 / 1.025 / 1.248 |

Holon decodes at 36–41 tok/s in mode (i) and 19–24 tok/s in mode (ii).

Caveats:
- **Mode (ii)'s decode misses its bar at 4K and 32K** (a recorded shortfall): the first 32 tokens after a long prompt
  read from the SSD the experts the prompt displaced from RAM, where Maya's prompt borrows in one pass and leaves part
  of what it borrows empty, so it displaces fewer (`docs/dev/v1-glm.md` §13 step 7).
- **Maya's conversation checkpoints** (on by default) cost its 32K prompts: with them off (`MAYA_GLM_CKPT_EVERY=0`) it
  prefilled 32K at 3,458 tok/s in mode (i) and 3,238 in mode (ii) in earlier sessions of the milestone (not the same
  session as the table)
  (`docs/dev/evidence/glm_n1_maya_rtx-5090_2026-10-25.json`, `glm_step7_mode2_d8h_2026-10-28.json`), against which
  Holon's 32K prefill here is about 0.99 and 0.94.
- **Maya's CPU lane is not its lead:** with it off (`MAYA_GLM_CPU_LANE=0`) Maya decodes at 0.99 / 1.01 / 1.00 of its
  default in mode (ii) (`docs/dev/evidence/glm_step7_mode2_d8k_2026-10-28.json`); Holon computes every expert on the GPU.
- The RTX 5080 (mode (i), 8K steps) was not compared.

### Qwen3-30B-A3B offload on the RTX 5080

The RTX 5080's Qwen3-30B-A3B Q4_K_M lock (about half the experts cached: 52.7% at the offload milestone's close)
against FreeToken: decode 1.06 / 1.04 / 1.02× at 1K / 4K / 32K (`docs/dev/v1-offload.md` "Step 8, part 2",
`docs/dev/evidence/x8_close_rtx-5080_2026-10-18.json`); prefill 1.850 / 1.136 / 1.005× against FreeToken run at the lock's step
budget, and 0.959× against FreeToken's default at 32K, after the MoE prefill work (`docs/dev/v1-moe-prefill.md`,
`docs/dev/evidence/mp5_close_2026-10-20.json`), which moved the lock's decode by 0.994× (pooled over 18 prompt and context
pairs).

## Contributing

Rules and routing by task are in [AGENTS.md](AGENTS.md); adding a model is in
[docs/adding-a-model.md](docs/adding-a-model.md).

## License

Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)); third-party code is listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
