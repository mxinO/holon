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
over three runs), in the release environment (`docs/release.md` §16, §19). Every lock selects the grammar processor.
The Qwen3.8-27B locks serve a context of 49,152 tokens without speculation (Known limitations).

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

Install the release tag. Steps 7–10's locks and records name commits of the release line as first built; the same commits are on `main` under new SHAs, with identical code and a privacy-scrubbed spike README (`evidence/v0_release_line_sha_map_2026-10-16.txt` maps them; `evidence/history_scrub_commit_map_2026-10-16.txt` maps
older SHAs).

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
Build it from the release tag, never from a working directory; the build takes ≈ 10–15 min on a many-core host, and
the image is ≈ 14.8 GB:

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
Offline, a hub name is found through the cache's `refs/main`, which a download writes; in a cache filled another way
(by revision, or copied without `refs/`) serve the file's path instead,
`<HF cache>/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/<revision>/<file>`: it finds the same lock.
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
model = "Qwen3.8-27B-UD-IQ3_S"             # the served id (/v1/models); the server ignores the model a request names
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
Serve the Qwen3.8-27B for Claude Code: its locks serve 49,152 tokens of context. Claude Code 2.1.300 sends, besides
its main loop, side requests of ≈ 31K input tokens (`docs/release.md` §18.2), which a 32K context would fit with about
1K tokens to spare. Its auto-compaction assumes a far larger window than 49,152 tokens, so a long session reaches the
served limit first. On an RTX 5080 that also drives a display, the 49,152-token context leaves ≈ 200 MiB of headroom,
so it may be clamped to what fits; the log shows it (`differs from the lock: engine:max_model_len: …`). The side
requests need it to stay above ≈ 32K.

The release's agent demo (`bench/demo/`) is a small fix in a three-module Python project — a seeded bug and a missing
function — chosen to fit a 32K context. It shows the served model completing such a task through Claude Code; it
is not a claim of general agent capability. **Result** (pre-registered, Claude Code 2.1.300, each card's 27B lock,
scored by the harness's own copy of the tests): **3 of 3 runs passed on the RTX 5080 (UD-IQ3_S) and 3 of 3 on the
RTX 4090 (UD-Q4_K_M)**, in 13–16 turns and 2–5 minutes each (`evidence/agent_demo_rtx-5080_2026-10-16.json`,
`evidence/agent_demo_rtx-4090_2026-10-15.json`; `docs/release.md` §18). Claude Code sends no temperature, so the
server's default (1.0) applies. The demo ran on the locks as they stood at release step 9, at the 27B's earlier 32K
context; the shipped locks differ in that context and in the FlashInfer plugin's `src`, with byte-identical snapshot
traces (`docs/release.md` §19.1–§19.2).

## Options

Flags override the lock, which overrides the defaults; `holon serve --help` lists every flag (one per engine
setting), every plugin option the installed plugins read, and the plugins `--slots` selects among. An option no plugin
declares is warned about; a flag's value outside an option's choices is refused before anything is downloaded, and a
lock's is warned about and that lock not used.

- **Speculation (MTP) on the Qwen3.8-27B**, off in every lock (Known limitations): `holon serve <model>
  --max-draft-tokens 3 --slots drafter=drafters.mtp draft_policy=std.spec.fixed_k verifier=std.spec.greedy
  --plugin-options spec.k=1:3,3:2`; `--max-draft-tokens 0` turns speculation off.
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
- **Context.** The Qwen3.8-27B locks serve 49,152 tokens; the 8B and 30B locks 8,192 or 16,384 (`holon models` and
  the lock files show each); without a lock, min(the model's trained context, 32768). `holon serve <model>
  --max-model-len N` changes it, and a context that does not fit the card's memory is clamped to what does (logged).
- **Speculation is off in every lock.** On the Qwen3.8-27B, MTP gave 11–45% lower TPOT on a sampled
  Claude-Code-shaped trace, but a speculating stream's longest pauses (p95 per-request maximum gap) grew 1.1–3.1×, past
  the ≤ 1.10× rule fixed before measuring (`docs/release.md` §16.6). Turn it on with the flags under Options.
- **A long prompt stalls the streams already decoding** on the locks that set `chunk_with_decode` 0: every 8B and 30B
  lock. Behind a 30K-token prompt (with the context raised to fit it) the stall was 3.2–11.8 s; `holon serve <model>
  --chunk-with-decode 512` mixes decoding into those prefill steps and cut it 23–37×, for 11–22% more time to that
  prompt's first token (48–54% on two RTX 4090 cells) and 2–15% of its throughput (`docs/release.md` §16.5). The
  Qwen3.8-27B locks set 512.
- **On an RTX 5080 that also drives a display**, the Qwen3.8-27B's 49,152-token context leaves ≈ 200 MiB of headroom
  and may be clamped to what fits (logged); Claude Code's ≈ 31K side requests need it above ≈ 32K.
- **Traces are run-identical on one machine, not across machines:** two RTX 3090 desktops recorded different
  Qwen3.8-27B snapshot traces at the same lock, both within the judge's L2 budget (`docs/release.md` §19.1).
- **No authentication:** the server checks no API key; do not expose it beyond a trusted network.
- **Every start compiles FlashInfer's CuTe DSL norms** in memory (about 1 s; First start).
- **M0.6's low-precision files on the RTX 5090 and RTX 3090** (8B NVFP4 and FP8, 30B NVFP4, 30B MXFP4_MOE) run with
  borrowed locks, unverified on those cards.
- Under a grammar, an output can run on in whitespace, digits or a string the schema admits until `max_tokens`: on the
  27B with MTP, 2.66% of constrained outputs were truncated against 0.33% unconstrained, over a 2-point gate that was
  accepted as a recorded shortfall. Guards against such runaways are deferred; set `max_tokens`.
- DFlash2 speculation is an opt-in, in no lock ([docs/m04-spec.md](docs/m04-spec.md)).
- Stop the server before a `git pull`: the install is editable, so a supervisor restart would import the new code.

## Performance

Measured at release step 8 on one desktop per card, each clock-screened before and after, with Holon serving from that
card's lock as it stood then. The shipped locks differ from those in the Qwen3.8-27B's context and in the FlashInfer
plugin's `src`, with byte-identical snapshot traces (`docs/release.md` §19.1–§19.2). **Ratios are the claim**: absolute tokens per second belong to one board. Each ratio below is Holon's
advantage (above 1, Holon is faster) as a range over the four cards; the tables per card, file and trace, rendered by
`bench/report.py` from the evidence records they link, are in
[docs/release.md §17](docs/release.md#17-as-built-step-8), with the method and the absolute numbers per board.

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

## Contributing

Rules and routing by task are in [AGENTS.md](AGENTS.md); adding a model is in
[docs/adding-a-model.md](docs/adding-a-model.md).

## License

Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)); third-party code is listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
