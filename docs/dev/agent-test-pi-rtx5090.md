# Real-agent test: the pi coding agent on an RTX 5090 (2026-10-16)

Report-only, not a release gate. The pi coding agent drove `holon serve` (Qwen3.8-27B UD-Q4_K_M) through
Holon's OpenAI-compatible endpoint. The test had three goals: measure decode and prefill speed under a real agent's
traffic, check the endpoint (streaming, tool calls, reasoning, multi-turn sessions), and report anything broken.
Record: [`docs/dev/evidence/agent_test_pi_rtx-5090_2026-10-16.json`](evidence/agent_test_pi_rtx-5090_2026-10-16.json)
(smoke, probes, counted series and their summary). Harness: `bench/demo/pi_run.py` and `bench/capture_proxy.py`
(bench code only; nothing in the engine changed).

## Results in brief

- **All 7 sessions passed** (2 uncounted smoke runs and 5 counted runs). D7's demo task passed 3 of 3, in 7–8 turns
  and 46–74 s. The longer `workloads` task passed 2 of 2, in 9–10 turns and 99–100 s.
- **The endpoint held up.** All 57 requests got 200 with a well-formed stream that ended with `[DONE]`. All 91 tool
  calls parsed, including up to 5 parallel calls in one answer and pi's `edit` tool with its nested-array argument.
  No call failed, no tool-call or `<think>` markup leaked into the content, and reasoning arrived as
  `reasoning_content` every time.
- **One frontend bug, latent on this model file:** `/v1/chat/completions` passes a `developer` message to the chat
  template as it is (finding 1). pi sends its system prompt that way by default. This model file's Unsloth template
  accepts it. With the Hugging Face (HF) templates of Qwen3.8 and Qwen3.5 the request would fail with a 400. With
  those of Qwen3-8B and Qwen3-30B-A3B (the HF files and Holon's GGUF files alike), the template silently drops pi's
  system prompt.
- **Two fields pi relies on are not honoured.** The top-level `reasoning_effort` is silently ignored (finding 2).
  The usage block reports no cached tokens (finding 3).
- **Decode:** TPOT p50 11.93 ms (83.8 tok/s) and p90 12.25 ms (81.6 tok/s), against tg128's 87.3 tok/s.
- **Prefill** of uncached input, (input − cached) / TTFT: p50 4,227 tok/s over the 32 requests with at least 512
  uncached tokens, against pp512's 4,763 and pp512 at depth 8192's 4,103.
- **Prefix cache:** 81.2% hit rate.
- **Time split:** decode was 89–93% of each session's wall time.

## Setup

- **GPU.** One RTX 5090 desktop, clock-screened with `bench/clock_check.py` (the 27B at its lock, 60 s, batch 4):

  | screen | SM MHz p10 / p50 | thermal slowdown | power cap | steady |
  |---|---|---|---|---|
  | before | 2445 / 2505 | 0 | 1.00 | yes |
  | after | 2032 / 2430 | 0 | 1.00 | no |

  The after-screen's p10 dipped under the power cap. During the runs, which were single-stream and drew less power,
  TPOT followed context depth, not time. Every session's first request decoded at 11.58–11.68 ms.
- **Server.** The commit under test is on a branch cut from `main` at b51af16; it changes only `bench/` and its
  test. The run used a CUDA 13.0 container with `uv sync --group dev` and `holon prebuild --sm 120`.
  `holon serve unsloth/Qwen3.8-27B-GGUF` picked the right file and lock. It then refused to start offline, because
  the shared HF cache has no `refs/main`, as in release step 9 (`docs/dev/release.md` §18.1). So the file's path in the
  cache was served. It started from `plans/…/Qwen3.8-27B-UD-Q4_K_M@322e194ff79741c7/rtx-5090/cu130-torch2.11/3.lock`:
  - context 49,152, `max_seqs` 4, `chunk_with_decode` 512, prefix cache on, the grammar processor, no speculation;
  - plan `sha256:d1daeb23…`, the lock's own;
  - `/metrics` showed no diff.
- **Client.** pi 0.73.1 (`@mariozechner/pi-coding-agent`) on Node 24.18.0, installed into its own npm prefix. The
  record holds a digest of the installed packages.
  - It ran headless: `pi --mode json --provider holon --model <served id> --session-dir <tmp> <prompt>`, one
    session at a time.
  - Its environment held only `PATH`, `HOME` and `PI_CODING_AGENT_DIR` (both fresh and empty), `LANG`, `TERM`,
    `NO_PROXY`, `VIRTUAL_ENV`, `PI_OFFLINE=1`, `PI_TELEMETRY=0` and `PI_SKIP_VERSION_CHECK=1`.
  - It ran with no credentials, settings, skills or context files, and its default tools (read, bash, edit,
    write).
  - Offline mode turns off pi's update check, telemetry and `fd`/`rg` downloads; the default tools need none of
    them. Every request the proxy saw went to the local server.
- **Provider** (`models.json`). The counted runs used:

  ```json
  {"providers": {"holon": {"baseUrl": "http://127.0.0.1:<proxy>/v1", "api": "openai-completions", "apiKey": "holon",
    "models": [{"id": "Qwen3.8-27B-UD-Q4_K_M", "reasoning": true, "contextWindow": 49152}],
    "compat": {"supportsDeveloperRole": false, "thinkingFormat": "qwen-chat-template"}}}}
  ```

  The `compat` block is what pi's docs give for local OpenAI-compatible Qwen servers. One smoke run used the docs'
  minimal provider instead, the same without `compat`; it sends `developer` and `reasoning_effort`. Both configs
  passed.
- **Measurement.** `bench/capture_proxy.py`, extended for OpenAI chat streams, sat between pi and the server.
  - Per request it records input and output tokens, TTFT (time to the first stream chunk, which the server sends
    only after the first token), TPOT ((total − TTFT) / (output − 1)), the finish reason, the tool calls and whether
    their arguments parse as JSON objects, any markup left in the content or reasoning, and whether the stream was
    well formed.
  - `--metrics` adds each request's prefix tokens saved: the server's `/metrics` counter read before and after the
    request. One session ran at a time, so no requests overlapped.
  - No message text is recorded.
- **Tasks.**
  - `demo`: D7's project and prompt, scored as D7. The harness runs its own copy of the 40 tests against the agent's
    `shop/`, in a clean directory, under its own interpreter.
  - `workloads`: a copy of Holon's own stdlib-only `bench/workloads.py` and `bench/driver.py`, three traces and a
    seeded 9-test suite (`bench/demo/workloads_task/`). The prompt asks for `trace_stats(sessions)` with five
    specified keys, a `--stats PATH` option, tests for both, and a passing suite without changing the existing
    tests.
  - A `workloads` run passes only if both checks pass. First, the agent's modules pass the restored seeded tests and
    the harness's 5 hidden tests. Second, the agent's own suite passes with more than the seeded tests and names
    `trace_stats`.
  - Order: one smoke run per provider config, then the counted demo ×3 and workloads ×2, all on one server.

## 1. Endpoint correctness

The checks below cover 57 requests: 15 smoke and 42 counted.

| check | result |
|---|---|
| non-200 responses | none |
| malformed streams (a data line that is not JSON, or no `[DONE]`) | none |
| finish reasons | `tool_calls` 50, `stop` 7 |
| tool calls | 91, every one's arguments a JSON object; pi ran all 91, none failed. Per answer: 1 call 30 times, 2–5 calls 20 times. Text before a call (allowed by the template) 21 times, never markup |
| tool calls left as text in `content` (`<tool_call>`, `<function=`) | none |
| reasoning leaking into `content` (`<think>`, `</think>`) | none; reasoning streamed as `reasoning_content` in every answer |
| multi-turn replay | pi's replay shape works: assistant `content: null` with `reasoning_content` and `tool_calls`, then `tool` messages. A probe sent the same replay twice; the second hit the prefix cache (448 of 501 tokens) |

pi's request shape in the counted runs:

- keys: `model`, `messages`, `stream`, `stream_options: {include_usage: true}`, `store: false`,
  `max_completion_tokens: 16384`, `tools` (4 function tools, `strict: false`), and
  `chat_template_kwargs: {enable_thinking: true, preserve_thinking: true}`;
- roles: `system`, `user` (content as text parts), `assistant` (fields `reasoning_content`, `tool_calls`), `tool`
  (`tool_call_id`);
- no `temperature`, `top_p`, `top_k` or `tool_choice`.

In the minimal-provider smoke run, `developer` replaces `system` and `reasoning_effort: "medium"` replaces
`chat_template_kwargs`.

### Finding 1 (frontend bug, latent here; fixed in v0.1.1): `developer` messages reach the chat template as they are

**Request shape.** pi's default for a model marked `reasoning: true`:

```text
POST /v1/chat/completions
{"model", "messages": [{"role": "developer", "content": <string>},
                       {"role": "user", "content": [{"type": "text", "text": <string>}]}, …],
 "stream": true, "stream_options": {"include_usage": true}, "store": false,
 "max_completion_tokens": 16384, "reasoning_effort": "medium", "tools": [<4 function tools, strict: false>]}
```

**What the frontend does.** The chat route (`plugins/holon_plugins/frontends/http/openai.py`, `ChatRequest` →
`Chat`) passes the messages unchanged to the bundle's `render_chat`. The Responses route already maps `developer` to
`system` (`responses.py`); the chat route does not.

**Effect by chat template:**

| chat template | result |
|---|---|
| this GGUF file (Unsloth's Qwen3.8 template) | accepts `developer` as `system`. Measured at temperature 0: the same 67 prompt tokens and an answer of the same lengths as with `system`, and a prefix-cache hit between the two. The minimal-provider smoke run passed |
| the HF checkpoints `Qwen/Qwen3.8-27B` and `Qwen/Qwen3.5-9B` | raise `Unexpected message role.`, which the frontend returns as a 400: `chat template: Unexpected message role.` |
| `Qwen/Qwen3-8B` and `Qwen/Qwen3-30B-A3B`, and the GGUF files `unsloth/Qwen3-8B-GGUF` and `unsloth/Qwen3-30B-A3B-GGUF` (Q4_K_M) | render without the message: pi's whole system prompt, its tool instructions included, is dropped with no error |

The template rows were found by rendering each cached template on the CPU with a `developer` message. They were not
run against a live server on those models.

**Who meets it.** Any OpenAI client that sends `developer`; pi does so unless `compat.supportsDeveloperRole` is set
to `false`. Mapping `developer` to `system` in the chat route, as the Responses route does, would cover every
template. Not fixed here, as instructed.

### Finding 2 (field ignored; fixed in v0.1.1): top-level `reasoning_effort`

pi's default provider sends its thinking level as a top-level `reasoning_effort` (default `"medium"`). The chat
route ignores unknown fields (`schemas.py`), so the model always thinks at the template's default, `xhigh`. Measured
with the same system and user messages, at temperature 0:

| request | prompt tokens | answer |
|---|---|---|
| no `reasoning_effort` | 67 | (reference) |
| top-level `reasoning_effort` `"low"` | 67 | same lengths (35 tokens; same reasoning and content lengths) |
| top-level `reasoning_effort` `"medium"` | 67 | same lengths |
| `chat_template_kwargs: {reasoning_effort: "low"}` | 55 | different reasoning length (the template wrote its low-effort instruction instead of the `xhigh` one) |

So pi's thinking-level setting has no effect, and the client gets no sign of it. Setting
`chat_template_kwargs.enable_thinking: false` does turn thinking off (0 reasoning characters).

One caution for a fix: this template accepts `low`, `medium`, `high` (read as `xhigh`) and `xhigh`, and raises on
anything else. pi's levels also include `minimal`, so forwarding the field unchecked would turn a silent no-op into a
400.

### Finding 3 (not reported; fixed in v0.1.1): cached tokens in `usage`

The final usage chunk has `prompt_tokens`, `completion_tokens` and `total_tokens`, with no
`prompt_tokens_details.cached_tokens`. pi reads that field, so its usage showed 0 cache reads on all 57 answers (in
pi's own session files, kept locally with the transcripts), while the server saved 81% of the input tokens (below). This is not a correctness problem, but a client cannot see the prefix
cache.

### Other observations

- **Honoured:** `max_completion_tokens` (8 gave finish reason `length` after 8 tokens),
  `stream_options.include_usage`, `chat_template_kwargs`, user content as text parts (flattened), `tools: []` with
  a tool history, and `strict: false` tools (unconstrained, as documented).
- **Ignored, harmlessly:** `store: false`.
- **Sampling.** pi sends no sampling parameters, so every request sampled at the server's defaults (temperature 1.0,
  top-p 1.0, top-k off), as with Claude Code (`docs/dev/release.md` §18.3).
- **Not exercised:** the context limit. pi compacts at `contextWindow` − 16,384 = 32,768 tokens, and the peak
  context was 16,290. No compaction or retry happened.

## 2. Speed under agent load

These figures cover the 42 counted requests (5 sessions). Contexts ran from 1.6K to 16.3K tokens, and output from
22 to 5,020 tokens per request (median 218, mostly reasoning).

| metric | value | published single-stream (`docs/dev/release.md` §17, this card's 27B) |
|---|---|---|
| decode, 1000 / TPOT, p50 | **83.8 tok/s** (TPOT 11.93 ms) | tg128 87.3 tok/s; `holon serve` `decode_b1` TPOT 11.43 ms |
| decode at TPOT p90 | 81.6 tok/s (TPOT 12.25 ms) | |
| prefill of uncached input, (input − cached) / TTFT, p50, all 42 requests | 3,951 tok/s | |
| the same, requests with ≥ 512 uncached tokens (32), p50 (p10–p90) | **4,227 tok/s** (3,427–4,681) | pp512 4,763 tok/s; pp512 at depth 8192, 4,103 tok/s |
| prefix-cache hit rate (prefix tokens saved / input tokens) | **81.2%** (270,592 / 333,274) | |
| TTFT p50 / p90 | 327 / 683 ms | |
| request time (first byte to last) p50 / p90 | 2.9 / 25.9 s | |

**Decode by context depth** (by the request's input tokens):

| input tokens | requests | TPOT p50 (min–max) | tok/s at p50 |
|---|---|---|---|
| < 4,096 | 12 | 11.64 ms (11.60–11.77) | 85.9 |
| 4,096–8,191 | 10 | 11.85 ms (11.79–12.30) | 84.4 |
| 8,192–12,287 | 11 | 12.05 ms (11.89–12.64) | 83.0 |
| 12,288–16,383 | 9 | 12.23 ms (12.15–12.65) | 81.8 |

Below 4K tokens decode runs at 85.9 tok/s, 2% under tg128 (measured at depth 0). No 8–16K decode figure was
published for this card to compare against.

**Prefill by context size:**

| input tokens | requests | uncached tokens p50 | TTFT p50 | prefill p50 |
|---|---|---|---|---|
| < 8,192 | 22 | 1,628 | 355 ms | 4,437 tok/s |
| 8,192–16,384 | 20 | 708 | 199 ms | 3,531 tok/s |

TTFT here includes things pp512 does not: the HTTP round trip, template rendering and tokenization, scheduling, and
the first decode step. That overhead weighs most on short uncached spans: 13 of the 8–16K bin's 20 requests had
fewer than 1,000 uncached tokens (249–5,113, median 708). The 9 requests with at least 2,048 uncached tokens reached 3,824–4,593 tok/s.

**What the uncached tokens are.** Each session's first request (pi's system prompt, 4 tools and the task: 1.6–1.8K
tokens) was uncached, with TTFT 344–417 ms. After that, 44% of all uncached input tokens (27,646 of 62,682) was the
previous turn's own output (its reasoning and tool call), prefilled again on the next turn. This is by design:
- The Qwen3.8 hybrid's recurrent state is checkpointed once per prompt, at its end (`recurrent.policy` `prompt_end`,
  `std/holon_std/state/recurrent/kind.py`). So a hit can reach the previous prompt but not the tokens generated
  after it.
- pi replays the reasoning (`reasoning_content`, rendered by the template's `preserve_thinking`), so those tokens
  are part of every later prompt.
- At the measured prefill rates this cost about 6.6 s of the 373 s the five sessions took (≈ 2%).

**Where the time went:**

| run | wall s | turns | wall per turn s | decode s | TTFT total s | tools and client s |
|---|---|---|---|---|---|---|
| demo 1 | 46.3 | 8 | 5.8 | 41.5 | 2.2 | 2.6 |
| demo 2 | 53.3 | 7 | 7.6 | 47.8 | 2.2 | 3.4 |
| demo 3 | 74.5 | 8 | 9.3 | 68.9 | 2.7 | 2.8 |
| workloads 1 | 99.3 | 9 | 11.0 | 92.3 | 3.9 | 3.2 |
| workloads 2 | 99.6 | 10 | 10.0 | 91.5 | 4.0 | 4.1 |

Decode, mostly of reasoning, is 89–93% of every session. Prefill (all TTFT) is 3–5%, and pi plus the tools are
3–7%. A turn's latency is therefore set by how long the model thinks: the longest single answer was 5,020 tokens,
about 61 s.

## 3. Task outcomes

The token columns read input / cached / output. TTFT and TPOT are the run's medians (interpolated, as `--report`
computes them), TTFT with its maximum in brackets. Peak context is input plus output of one request. "done" is
whether the final reply was exactly the requested word (not scored).

| series | run | scored | turns | wall s | tokens in / cached / out | TTFT p50 (max) ms | TPOT p50 ms | peak context | tool calls / failed | "done" |
|---|---|---|---|---|---|---|---|---|---|---|
| smoke, minimal provider | 1 | pass, 40 / 0 | 7 | 42 | 34,866 / 26,112 / 3,151 | 344 (588) | 11.72 | 8,632 | 13 / 0 | yes |
| smoke, documented provider | 1 | pass, 40 / 0 | 8 | 57 | 53,627 / 42,752 / 4,409 | 310 (631) | 11.90 | 10,923 | 15 / 0 | no |
| demo | 1 | pass, 40 / 0 | 8 | 46 | 46,588 / 37,248 / 3,521 | 254 (604) | 11.82 | 9,283 | 14 / 0 | no |
| demo | 2 | pass, 40 / 0 | 7 | 53 | 38,048 / 28,608 / 4,031 | 344 (636) | 11.84 | 9,312 | 13 / 0 | yes |
| demo | 3 | pass, 40 / 0 | 8 | 74 | 48,864 / 37,440 / 5,760 | 351 (817) | 11.88 | 11,282 | 14 / 0 | no |
| workloads | 1 | pass: harness 14 / 0, own suite 14 / 0 | 9 | 99 | 92,566 / 76,352 / 7,545 | 312 (1,222) | 12.19 | 16,120 | 10 / 0 | yes |
| workloads | 2 | pass: harness 14 / 0, own suite 13 / 0 | 10 | 100 | 107,208 / 90,944 / 7,438 | 330 (717) | 12.22 | 16,290 | 12 / 0 | no |

- **Demo runs.** All took the same path: `find`, a `read` of every package and test file
  and `pyproject.toml` (often 5 reads in one answer), one `edit` each to `money.py` and `discounts.py`, and pytest
  through `bash`. The record names a bash call by its first word, which is `cd` for pi's `cd <dir> && …`
  commands; the local transcripts show those calls ran pytest (and, in the workloads runs, the new CLI). Every diff zero-pads the cents and adds `buy_n_get_one`.
- **Workloads runs.** These listed the directory (`find` or `ls`), read `workloads.py`, `driver.py` and the seeded tests, edited `workloads.py` once or
  twice, wrote `tests/test_stats.py` (5 and 4 new tests), and ran the suite and the CLI.
- **Next to Claude Code's D7 runs** (`docs/dev/release.md` §18.3: 13–16 turns, 131–292 s on an RTX 4090 or 5080): pi
  takes about half the turns. Its system prompt and tools are ≈ 1.6K tokens against Claude Code's ≈ 17K, and it
  sends no side requests. Different cards, so the times are not a comparison.

## Fixed in v0.1.1

The three findings are fixed in v0.1.1 (`docs/releases/RELEASE_NOTES_v0.1.1.md`):

1. **`developer` messages** are system messages on every route; a leading run of system and developer messages that
   holds one becomes one system prompt. pi's request shape now renders with its system prompt kept and no 400 on all
   seven shipped template files (Qwen3-8B and Qwen3-30B-A3B as HF and GGUF, Qwen3.5-9B HF, Qwen3.8-27B as HF and
   GGUF; a CPU test renders each).
2. **`reasoning_effort`** (and the Responses API's `reasoning.effort`, Anthropic's `output_config.effort`) sets the
   template's level: a level the template has as it is, another level of the clients' scale as the nearest one it
   has (`minimal` the lowest), `none` turns thinking off, anything else is a 400. Templates without levels ignore
   the scale.
3. **Cached tokens** are reported per request, counted by the engine at the request's first admission:
   `usage.prompt_tokens_details.cached_tokens` here (and the other routes' equivalents). This needed a core field
   (`TokenDelta.cached_tokens`, plugin ABI 1.1) and a propagation of every lock, with byte-identical traces.

**Re-run** ([`docs/dev/evidence/agent_test_pi_v011_rtx-5090_2026-10-16.json`](evidence/agent_test_pi_v011_rtx-5090_2026-10-16.json)):
the same RTX 5090 desktop, clock-screened (before p10 / p50 2550 / 2640 MHz, steady; after 2047 / 2505, not steady
under the power cap, no thermal slowdown), serving v0.1.1's propagated lock 4, and pi 0.73.1 with the docs' minimal
provider (no `compat`: it sends `developer` and `reasoning_effort`). One demo run per thinking level:

| pi thinking level | sent `reasoning_effort` | template level | first prompt tokens | scored | turns | wall s | output tokens | cache reads pi reported (sum) |
|---|---|---|---|---|---|---|---|---|
| medium (pi's default) | medium | medium | 1,589 | pass, 40 / 0 | 7 | 23 | 1,611 | 23,680 |
| low | low | low | 1,616 | pass, 40 / 0 | 9 | 39 | 2,953 | 41,920 |
| high | high | high (the template reads it as xhigh) | 1,628 | pass, 40 / 0 | 10 | 39 | 2,928 | 51,520 |

- **The thinking level takes effect.** Each level renders a different prompt: the template writes no instruction at
  `medium`, a short one at `low`, the long `xhigh` one at `high`. v0.1.0 ignored the field and always wrote the
  `xhigh` instruction (1,627 tokens on this prompt then). With one run per level the output lengths do not show a
  trend.
- **pi shows cache reads.** Every request after a session's first reported a non-zero `cached_tokens`, which pi
  read as `cacheRead` (up to 9,152 tokens per request); on all 26 requests the server's count equalled the
  proxy's `/metrics` measurement. Hit rate 82.3%, decode TPOT p50 11.79 ms (84.8 tok/s).
- All 26 requests were 200 with clean streams, every tool call parsed, and no markup was left in the content.

## Not covered

- Contexts above 16.3K tokens, pi's compaction and the 49,152-token limit: no session got there.
- Concurrency: one session at a time.
- Speculation (off in the lock).
- Other models and cards. Finding 1's effect on the other templates was found by rendering them, not by serving
  them.

## Reproduce

Serve the card's 27B lock, then run each task. The second task appends its series to the same record; `--report`
writes the summary.

```bash
holon serve unsloth/Qwen3.8-27B-GGUF          # or the file's path in the HF cache, offline
python bench/demo/pi_run.py --server http://127.0.0.1:8000 --gpu-class rtx-5090 \
    --pi <prefix>/node_modules/.bin/pi --task demo --runs 3 --out docs/dev/evidence/agent_test_pi_rtx-5090_<date>.json
python bench/demo/pi_run.py --server http://127.0.0.1:8000 --gpu-class rtx-5090 \
    --pi <prefix>/node_modules/.bin/pi --task workloads --runs 2 --out docs/dev/evidence/agent_test_pi_rtx-5090_<date>.json
python bench/demo/pi_run.py --report docs/dev/evidence/agent_test_pi_rtx-5090_<date>.json
```

The smoke series used `--runs 1 --series smoke_compat_none --compat none` and `--series smoke_compat_documented`.
The record's `probes` were sent straight to the server before the smoke runs. They are listed there with their
request shapes; no text is kept.
