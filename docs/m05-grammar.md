# M0.5 — structured output and tool calls: design and implementation plan

Status: **revision 5, reviewed clean** (2026-10-13). Revs 1–5 were reviewed by a panel of three; every finding is
resolved in §9. **Step 1 built** (2026-10-13): the core seams of §3 with the toy processor. **Step 2 built**
(2026-10-13): the `grammar.xgrammar` plugin, pinned at xgrammar 0.2.8. **Step 3 built** (2026-10-13):
`frontends/http`, the format parsers and constraint builders, the three dialects. **Step 4 measured** (2026-10-14):
D2's decision is A′, and the forward/sample split is deferred to M0.7. **Step 5 evidence** (2026-10-14): §11. Their
records, deviations and §6's measurements are in §11.

Scope: plan §8, amended (D1; plan §2 / §8 updated with this revision):
- the core gets general logits-processor seams (§3); "core diff target zero" becomes "only the seams of §3, each
  with a toy proof". None of the plan ladder's three named M0.5 seams is built as named: the forward/sample split is
  conditional (D2), `host_input` becomes the processor's `fill`, and the "Rewindable grammar" is a host rollback
  inside the plugin (the matcher advances only at commit; the device `rewind` protocol is not involved);
- one frontend package serves the three API dialects (D7), not three frontend plugins;
- `tool_choice: "auto"` stays unconstrained unless a tool is `strict` (D8);
- "100% schema conformance" is defined for strict requests against a published reject list (D9).

Facts about xgrammar 0.2.3, vLLM, SGLang, llama.cpp and the Qwen templates are cited in the local research note
`docs/brainstorm/m05-grammar-research.md` ("R §n"). Builds on M0.4 Gate A (verify rows, `std.spec.greedy` and
`std.spec.rejection`, verify graphs, overlap depth 1).

## 1. Goal and acceptance

Constrained decoding (JSON schema, JSON object, regex, EBNF, structural tags for tool calls) and tool calls through
the OpenAI chat, OpenAI Responses and Anthropic Messages APIs, on every model and with speculation. Acceptance:

1. **Conformance** (D9). On the JSONSchemaBench runs of D9, for strict requests:
   - every accepted, non-truncated output validates (Draft per schema, formats checked): **valid / (accepted −
     truncated) = 100%**, where a request-time `error` finish (a matcher refusal, a failed fill) counts as invalid;
     compile errors and timeouts are rejections (declared coverage), reported with each schema's compile time;
   - truncated outputs are reported, and their share is within 2 points of the same requests unconstrained;
   - declared coverage is reported with rejections by reason;
   - the model-free oracle (JSON-Schema-Test-Suite, every draft) finds no accepted schema that admits an invalid
     instance.
2. **Tool calls.**
   - OpenAI chat: `tools`, `tool_choice` (`auto`, `none`, `required`, named), `parallel_tool_calls`,
     `response_format` (`json_schema`, `json_object`), streaming `tool_calls` deltas; round trips on Qwen3-8B
     (hermes JSON) and the Qwen3.8-27B GGUF (XML `<function=…>`).
   - Anthropic `/v1/messages` (text, `tool_use`, `tool_result`, `thinking` blocks, `tool_choice`, streaming SSE) and
     `/v1/messages/count_tokens`: a non-interactive Claude Code session (`claude -p` with `ANTHROPIC_BASE_URL` at
     Holon serving the 27B on the 5080) completes a scripted multi-tool task.
   - OpenAI Responses: the minimal subset of R §5.2 (text and function-call items, streaming), one round trip.
   - Parsers: for every recorded output, the streaming parse equals the non-streaming parse.
3. **Grammar with speculation.**
   - Greedy: grammar + speculation emits exactly what grammar without speculation emits, with the same finish
     reasons — CPU, tiny models with the oracle drafter (drafts that cross JSON structure), exact; GPU, the
     constrained bounded check of §5 on the 27B + MTP.
   - Sampled: the masked L0-dist test of §5 (masks from the real probe).
4. **Graphs and overlap.** Every path that samples applies the mask (prefill, decode, verify; graphed and eager).
   A constrained request never samples from a mask built without its last token. Measured: the share of graphed
   constrained steps, the host time per constrained step, and D2's penalties.
5. **Isolation.** A failing constraint ends only its request, never the engine and never other requests: an
   unsupported strict schema, an invalid constraint or a compile error is a 400 (an error of kind `client`); a
   compile timeout is a 503 (kind `busy`); a matcher refusal or failed fill during generation is `finish_reason =
   "error"` for every sequence of the request (kind `engine`). Free streams' p99 TPOT during a cold compile is
   measured. A constraint sent to a deployment with no processor selected is a 400. A request restarted by the
   supervisor resumes its matcher.
6. **Invariants.** Processors are opt-in (`--slots logits_processor=…`), and locks record the choice. With none
   selected, nothing changes: the same graphs, every lock's snapshot trace byte-identical. With one selected and no
   constrained request, outputs are byte-identical at a pinned `state_memory` (the extra graphs shrink the KV pool
   otherwise); with constrained requests, snapshot traces are run-identical.
7. **Core:** only the seams of §3, each proven by a toy plugin in `core/tests`; core ≤ 6000 code lines (3650 now,
   ≈ +260 budgeted); the core denylist gains `grammar`, `xgrammar`, `tool_call`, `structural`.

Non-goals: grammar-aware drafting (neither vLLM nor SGLang does it, R §2.3); device-side automata; non-strict
conformance; multimodal content blocks; more than one active logits processor (§3.1).

## 2. Starting point (R §7)

- **Frontend:** OpenAI chat/completions with streaming; `tools` reach the chat template, but no tool parsing;
  `tool_choice` other than `auto` and `response_format` other than text are rejected (`schemas.py:192-193`).
  `FrontendBundle.reasoning_parser()` / `tool_parser()` exist and return `None`. `holon serve` runs exactly one
  frontend (`cli.py:66-70`).
- **Engine:** `LogitsProcessor` (`api/decoding.py:48-54`) and its `ordered_multi` slot exist; nothing loads or calls
  a processor, and ordered-multi ordering is not implemented (`plugins.py:177-193`).
- **Overlap:** `Engine.step` launches step N whole before it commits N − 1 (`engine.py:303-308`); design §7.3's
  table claims grammar works at depth 1 through a forward/sample split that was never built (R §3.1).
- **Speculation:** only `n_draft` returns to the host.
- **Errors:** an exception inside `Engine.step()` ends the engine (`ipc.py:90-97`); only `add()` errors are per
  request, and the client counts one `error` delta as the whole request finished (`ipc.py:146,150`).
- **Restart:** the supervisor resubmits `prompt + streamed` with `max_tokens − len(out)` (`supervisor.py:80-94`).

## 3. Core seams (today → change), each general

1. **One logits processor, opt-in, with a request lifecycle.**
   - Today: an unused protocol.
   - Change (`api/decoding.py`): the deployment selects at most one `logits_processor` plugin
     (`EngineConfig.slots`; locks record it in `[engine]`, like the drafter). Its plugin builds the processor with
     `build(ctx, vocab: VocabView)` (§3.7). The processor:
     - `mask_width: int` — int32 words per mask row (the core allocates the buffer, §3.6). `Engine.add` rejects a
       request carrying any `ext` key that no active plugin's manifest `provides` (a 400, kind `client`), so a
       constraint is never silently ignored (nothing reads `ext` today; no runtime attribute repeats the manifest);
     - `open(rid, params, prompt) -> bool` at `Engine.add`, before anything is registered: whether it constrains
       this request. Raising `RequestError` rejects the request. Slow work (compiling) happens off the engine thread
       (D5);
     - `ready(rid) -> bool`: false while the request's constraint is not built — the request is held as
       `Status.BLOCKED` and skipped by the scheduler (never blocking others: `_prefill` continues past it); `step()`
       returns no deltas when only blocked work remains (instead of raising "no progress"); `has_work()` stays true
       while anything is blocked, and the serve loop's poll timeout comes from `engine.idle_for()`: 0 when work is
       schedulable, 5 ms when only blocked work remains (each iteration promotes requests whose `ready` flipped, and
       surfaces a `ready` that raises), its normal interval when idle. Toy test: one constrained request on an idle
       engine, for a compile that succeeds and for one that times out. `ready` may raise `RequestError` (a compile
       error: kind `client`; a timeout: kind `busy`), which aborts the request;
     - `commit(seq, out: list[int])` after each step's results, **before** the engine acts on stop or length and
       before any delta is built, with the tokens the step appends (truncated at a stop, skipped for finished rows);
       raising `RequestError` aborts the request with kind `engine` and drops that step's tokens (§3.4);
     - `close(rid)` at finish or abort of the request (every sequence, never-started siblings included; drops a
       pending compile); `shutdown()` (optional), called by `Engine.close`, stops the processor's helpers;
     - `fill(rows: list[MaskRow], buf)` at launch: writes the mask rows of the step's constrained logits rows into
       the core's staging buffer (`MaskRow`: the sequence, its row range in the compact block, and for verify rows the
       drafts it will check). It never raises: a row it cannot build gets all tokens allowed and the failure surfaces
       at the next `commit` (which runs before that row's tokens are acted on);
     - `apply(logits, mask, rows)` — device, in place, graph-capturable: applies the compact mask block to the
       logits rows `rows` (the kernel and the mask format are the plugin's, D4).
   - Siblings of an `n > 1` request share one `open`; `fill` and `commit` are per sequence (`seq.index`).
   - Toy proof (`core/tests`): a toy processor — after a marker token, only even tokens; its `apply` is a dense
     `masked_fill_` — on the tiny Qwen3 on the CPU, across chunked prefill, prefix hits, preemption, depth 0 / 1,
     `n > 1`, verify rows with the oracle drafter (right / wrong / mixed), greedy and sampled: every emitted token
     obeys the rule, and the output equals a reference that masks a plain engine's full-vocabulary logits.
2. **Host-token gate (D2).**
   - Today: depth 1 for every decode row; `host_lens_only` gates rows for host-length kernels (`sched.py:111`).
   - Change: a constrained sequence is not scheduled while it has `pending` rows (pending covers inexact verify
     rows) or while `BLOCKED`. When a gated sequence is ready but pending, `step()` commits the in-flight step before
     scheduling (A′, D2) — or, under the alternative A, schedules without it. One option, chosen by D2's
     measurement.
   - Toy proof: constrained and free requests mixed at depth 1; `fill` asserts that every constrained row's host
     tokens are complete; the free rows overlap.
3. **Draft tokens to the host.**
   - Today: `Launched.drafts` carries `n_draft`.
   - Change: when a processor is selected, every step with a drafter also copies `SeqTable.draft_tokens[slots,
     :k_max]` from the same snapshot as `n_draft`, outside the graph, double-buffered like `_out_drafts`;
     `MaskRow.drafts` = the first `min(n_draft, k)` of them, and `fill` also fills the column after the last real
     draft.
   - Toy proof: the toy's probe sees exactly the oracle drafter's drafts.
4. **Per-request errors from the processor.**
   - Change: `RequestError(kind, message)` (in `holon.api`, a `ValueError`; kinds `client`, `busy`, `engine`) from
     `open`, `ready` or `commit` aborts the whole request (`Engine.abort(rid, "error", message)`: one delta per
     sequence, rows in flight discarded); `TokenDelta` carries the kind, and the frontend maps `client` before the
     first token to 400, `busy` to 503, anything else to `finish_reason: "error"`. After a supervisor restart (one
     engine request per sequence) the supervisor aborts the request's other engine rids on an `error` delta. A plain
     `ValueError` from `add` (prompt too long, `n < 1`, …) keeps today's meaning: kind `client`, a 400. Other
     exceptions still end the engine (bugs).
   - Toy proof: a toy processor raises for one request of an `n = 2` pair mid-generation; both its sequences end
     with `error`, the others finish unchanged.
5. **Resume after a restart.**
   - Change: `EngineRequest` and `EngineClient.generate` gain `resume_output: int`, the number of trailing prompt
     tokens that are output; the engine sets `Seq.prompt_len = len(tokens) − resume_output`; the supervisor sends the
     full `max_tokens` (it no longer subtracts). The core calls `commit(seq, tokens[prompt_len:])` once, after
     `ready` and before the sequence is first scheduled; processors do nothing else for resume.
   - Toy proof: a restart in the middle of a toy-constrained greedy generation ends with the same tokens and finish
     reason as an uninterrupted run.
6. **Masks on every sampling path.**
   - Change: the step's constrained logits rows ride the `Uploader` (`inputs.py`: double-buffered host staging
     guarded by events) as two more fields — a compact mask block `[n_constrained_rows, mask_width]` and its logits
     row indices (padded to the graph's row count with an out-of-range index the kernel skips). Free and padding
     rows cost nothing and the core never interprets mask bits. One helper between the head and the sampler or
     verifier calls `apply` on every path (prefill and decode, graphed and eager). The graph key gains one core
     boolean, "this step has constrained rows"; the startup capture set crosses it with every (bucket, width,
     sampler variant) when a processor is selected (capture stages a valid index list of out-of-range rows); the
     memory profile includes `apply`.
   - **Logprobs are of the masked distribution** (vLLM's default raw mode also applies the grammar mask before the
     sampler), clamped by the frontends at −9999 (JSON has no −∞); `api/frontend.py:26-28` is updated. No
     raw-logits copy.
   - Toy proof: graphed vs eager on the same padded plans with constrained rows (the M0.4 eager-runner test extended,
     GPU, Qwen3-0.6B); a plant `mask_rows_not_reset` (a free request takes a finished constrained request's row)
     fails it.
7. **`VocabView` for processors.**
   - Change: the engine passes `build(ctx, vocab)` a `VocabView`: the weight source's tokenizer object
     (`source.tokenizer()`, a fast tokenizer for HF and GGUF alike — decoded token strings lose byte-fragment tokens,
     which breaks multi-byte enums), the padded vocabulary size and the engine's stop-token set. Toy proof: the toy
     processor reads the tiny model's vocabulary; a non-ASCII enum fixture in step 2's oracle.

Line budget: lifecycle, selection and the resource check 90, gate, `BLOCKED` and A′ 25, drafts 15, errors and kinds
35, resume 10, Uploader fields, helper, key and capture 60, `VocabView` 20, API types (§3.8) 20; ≈ +275. The manifest's
unread `max_depth` is removed (the gate comes from `open`).

8. **API for frontends** (`holon.api.frontend`, no core runtime code): `FrontendBundle` gains `tool_format` and
   `reasoning_format` (names with versions, e.g. `"hermes@1"`, `"qwen_xml@1"`, `"think@1"`) and `constraint(…)`
   returning a `grammar.constraint@1` dict (D6); `Parser.feed` yields typed events (`Text`, `Reasoning`,
   `ToolCallStart`, `ToolArgsDelta`, `ToolCallEnd`) and the tool parser takes the request's tools (schema-typed
   values).

## 4. Decisions

- **D1. The general seams of §3 instead of "core diff zero".** Grammar needs a request lifecycle, a depth rule,
  drafts on the host, per-request errors and the vocabulary; each is general (no grammar or format name in core) and
  proven by a toy. Recorded in plan §2 and §8.
- **D2. Depth 0 for constrained sequences first; the forward/sample split only if measured necessary.**
  - Why: a host mask for step N needs N − 1's token; the whole step is launched before N − 1 commits. The gate is
    small and correct with speculation (the drafts are on the host before the launch). The split (vLLM, SGLang; R
    §3.3) keeps depth 1 for constrained rows, at two graphs per constrained step and an engine reorder.
  - Two gate variants: **A** — constrained sequences skip a step while pending (they decode every other step beside
    free rows, ≈ 0.5× their free rate); **A′** — when a ready constrained sequence is pending, commit before
    scheduling (constrained rows decode every step; every row of those steps loses overlap).
  - **Measured on identical tokens**: the same greedy request with a toy all-allowed processor that opens it, vs
    without (R §8.1's isolation of the stall from acceptance changes), host time per step split by phase
    (`phase_us`). Configurations: Qwen3-8B Q4_K_M plain and the 27B UD-IQ3_S + MTP on the RTX 5080; single stream,
    and 1 constrained + 3 free under A and A′.
  - **Rule (fixed now):** A′ by default (constrained rows — agents' tool calls — decode every step), unless under A′
    the free rows of the mixed batch slow by more than 1.15×, then A. Then, adding the measured mean `fill` and
    `apply` time per constrained step (the toy processor has neither): if the single-stream penalty exceeds
    **1.10×** on either model, build the split in M0.5 (step 6), else defer it to M0.7 with the numbers recorded.
  - **Predictions:** host time per constrained step 1–2 ms (commit, schedule, fill, launch); single stream 8B (6.5 ms
    step) **1.15–1.30×** — likely to trigger the split; 27B + MTP (a verify step ≈ 27 ms for 2.56 tokens) 1.04–1.07×.
    Mixed: A gives constrained rows ≈ 0.5× of free; A′ costs free rows the same 1.04–1.30× as single stream.
  - **Decision (step 4, measured on the RTX 5080, §11): A′, the split deferred to M0.7.** Free rows under A′ slow by
    1.050× (8B) and 1.016× (27B + MTP), under 1.15×; the single-stream penalty with grammar.xgrammar's fill and apply
    added is 1.061× (8B) and 1.026× (27B + MTP), under 1.10× on both. Host time per constrained step was 0.4–0.8 ms,
    not 1–2 ms, so both penalties came in under the predictions (8B 1.15–1.30×, 27B 1.04–1.07×).
  - The split's seam, sketched so step 6 is not designed while built: `launch_forward(N)` (prepare → trunk → head
    into a static logits buffer, its own graph) → commit N − 1 → `fill` → `launch_sample(N)` (apply → sampler or
    verifier → store → rewind → drafter, its own graph); only steps with mask rows split, `fill`'s contract is
    unchanged. Reviewed with the panel before step 6 starts.
- **D3. Speculation: host probe per verify column; verifiers unchanged.** Column j of a constrained verify row is
  masked with the matcher's state after the committed tokens and drafts d0…d(j−1): `accept` d0, fill, … through the
  column after the last real draft, stopping at the first refused draft (later columns all-allowed: never
  accepted), then `rollback`. A stop-token draft terminates the matcher: the next column gets the stop set, and the
  rollback restores it. The matcher advances only at `commit`. Exact-match and rejection verifiers are lossless for
  the masked target (a refused draft has p = 0; R §7.3). Drafters are not masked; refused drafts are counted
  (`spec_drafts_refused` in `Engine.stats()`, from `fill`'s count).
- **D4. Mask buffer and kernel.** The core owns the `[max_seqs·(1 + k_max), mask_width]` int32 device buffer and its
  Uploader staging; the processor owns `apply`. `grammar/xgrammar` applies with a Triton kernel of its own (one
  program per row block, no `[R, V, 32]` intermediate), checked against xgrammar's CPU `apply_token_bitmask_inplace`
  as the reference (L0-mask). Steps without mask rows keep today's graphs.
- **D5. `grammar/xgrammar`, a plugin; constraints vetted in killable, memory-capped workers, compiled in helper
  threads.**
  - It accepts `ext["grammar.constraint@1"]` (the key carries the resource's major version and matches the
    manifest's `provides`).
  - **Vetting** (an xgrammar compile cannot be cancelled in a thread, and compiled grammars can be huge: up to
    ≈ 2 GB in memory for two Github_hard schemas, 300 MB for the largest Kubernetes ones, with deserialization slower
    than compiling — reviewers' measurements): two worker processes, started with `spawn` (the engine process has
    CUDA and threads), kept lean (`OMP_NUM_THREADS=1`, `TOKENIZERS_PARALLELISM=false`, `MALLOC_ARENA_MAX=2`; the
    `TokenizerInfo` sent from the parent as JSON, asserted equal in padded size and stop set) and set to die with the
    parent (`PR_SET_PDEATHSIG`, which follows the creating thread: workers are started and respawned only by the
    watchdog's long-lived thread, and the child exits if `os.getppid()` is no longer that parent right after
    `prctl`), each with `GrammarCompiler(max_threads=2)`. A worker compiles the constraint and
    returns only `ok` / the error, the compile time, `memory_size_bytes` and its peak RSS — no grammar crosses the
    pipe.
  - **Memory:** a worker's baseline is ≈ 1.1 GB resident (≈ 6.7 GB virtual after importing xgrammar and torch, so
    `RLIMIT_DATA` cannot cap it — reviewer measurement). The parent's watchdog reads each busy worker's resident
    set (`/proc/<pid>/statm`) every 50 ms and kills it when the compile grows it **1 GB past its size at the job's
    start** (a per-compile delta; portable, no cgroup needed); workers also get `oom_score_adj` 1000 so the kernel
    prefers them to the engine.
  - **Outcomes:** past the **10 s timeout** the worker is killed and respawned and the request is rejected `busy` (a
    respawn re-imports torch: likely 2–5 s, measured at step 2 and the queue bound sized to it); a compiled size over
    **64 MB** or a watchdog kill is rejected `client` ("schema too complex"); any other worker death or exit without a
    report (a kernel OOM kill, a crash) is `busy`.
    A negative cache rejects repeats of compile errors and `client` rejections for the process's life, and of `busy`
    outcomes for 10 minutes. Tested with a lowered watchdog budget on a mid-size fixture, with a small schema as the
    positive control that still compiles under it.
  - **Compiling for use:** a vetted constraint is compiled again in the engine process by two helper threads with
    `max_threads=2` each (xgrammar releases the GIL while compiling, R §8), never on the engine thread; each compile's
    time is bounded by the worker's measured time; the helper backlog is bounded by the sum of its vetted compile
    times (10 s), beyond which a request is rejected `busy`; `close` drops queued compiles; `ready(rid)` flips when
    it is done. Vetting then compiling doubles a cold constraint's compile time (+13–62 ms for tool schemas and
    tags, R §1.2): cold-constraint TTFT is recorded beside the free streams' p99. Compiled grammars are cached by
    constraint (`cache_limit_bytes` = 512 MB), so repeats cost microseconds. Both compile times and the compiled size
    are recorded per schema in the D9 runs.
  - One `GrammarMatcher` per sequence; stop tokens = the engine's stop set plus the request's `stop_token_ids`;
    `ignore_eos` with a constraint is rejected at `open`.
  - JSON: `any_whitespace = True` and **no whitespace cap** (decision S2-1, after step 2; it replaces the cap of 64
    of revs 2–5). A cap of 64 made every compiled JSON grammar ≈ 13.4 MB (152k vocabulary) / 26.3 MB (248k) instead
    of 0.2–0.3 MB, its compile 44 instead of 15 ms, its median mask fill 33 instead of 4.2 µs, held the 512 MB cache
    to ≈ 19 grammars at 248k and made the 64 MB size cap depend on the vocabulary (measured at step 2, §11). A
    whitespace runaway is bounded by `max_tokens` and caught by the truncation gate; step 5 counts truncations that
    end in a whitespace run, and a cap (16–32) returns only if they appear. `json_object` is an object of any JSON
    (xgrammar's `{"type": "object"}`, not strict: OpenAI's promise; xgrammar's builtin JSON also admits a top-level
    array) — fixtures: every JSON-Schema-Test-Suite object dumped compact and at indents 2 and 4 to depth 8 is
    accepted, malformed JSON (escapes, `\uXXXX`, numbers, non-ASCII) refused, as by the builtin; a top-level array or
    scalar refused.
  - `strict_mode` per D9. Schema `pattern`s are ECMA 262 regular expressions (JSON Schema's dialect): a strict
    pattern that xgrammar reads more loosely (`.`, `\S`, `\s` in a negated class) is refused. xgrammar is a pinned
    pip dependency (Apache-2.0): **0.2.8** (§11), the reject-list oracle and the tag fixtures run at it.
- **D6. `grammar.constraint@1`, a plain-dict resource.** The frontend package builds constraints as plain dicts
  (`json_schema` with `strict`, `json_object`, `regex`, `ebnf`, and the tag forms `tag(begin, content, end)`,
  `triggered_tags(triggers, tags, at_least_one, stop_after_first)`, `any_text(excludes)`, `sequence`, `or`, and the
  parameter-value styles `json` and `qwen_xml`); the grammar plugin declares the resource in its manifest and
  translates it to its backend; bundles and frontends never import xgrammar. Tool-call constraints are built by the
  frontend from the bundle's `tool_format` (hermes JSON for Qwen3, `<tool_call><function=…><parameter=…>` for
  Qwen3.5 / 3.8, R §4.1), not by xgrammar's builtin `qwen_3` / `qwen_3_5` tags, which refuse a `<tool_call>` that
  ends reasoning without `</think>` and do not limit `parallel_tool_calls: false` to one call (reviewer finding,
  measured). Reasoning is free text until `</think>` or a `<tool_call>` that ends it. The fields (fixed at step 2,
  `plugins/holon_plugins/grammar/xgrammar/constraint.py`; both sides cite this table):

  | `type` | fields (optional ones with their default) |
  |---|---|
  | `json_schema` | `schema` (object or boolean), `strict` = false, `style` = `"json"` \| `"qwen_xml"` |
  | `json_object` | — (an object of any JSON) |
  | `regex` | `pattern` (xgrammar's regex) |
  | `ebnf` | `grammar` (its rule `root` starts it) |
  | `tag` | `begin`, `content` (a constraint), `end` (a string or a list of them) |
  | `triggered_tags` | `triggers` (strings), `tags` (`tag`s), `at_least_one` = false (true also forbids text before the first tag), `stop_after_first` = false |
  | `any_text` | `excludes` = [] |
  | `sequence`, `or` | `elements` (constraints) |
- **D7. One frontend package, three dialects.** `frontends/openai` becomes `frontends/http`: one FastAPI app with
  routers for OpenAI chat/completions, `/v1/messages` (+ `count_tokens`) and `/v1/responses`, over one pipeline
  (normalize → render → generate → reasoning/tool parse → typed events). The format parsers and constraint builders
  live here once, selected by the bundle's `tool_format` / `reasoning_format` (one hermes parser for every Qwen3
  bundle). Template-specific argument handling (JSON-string arguments parsed for templates that need objects, the
  Qwen3.8 GGUF one, R §4.1) stays in the bundle's `render_chat`. The Anthropic router drops a system block that is
  only an `x-anthropic-billing-header:` line (Claude Code, R §5.4: it changes per request and defeats prefix
  caching), behind an option (default on), counted; `CLAUDE_CODE_ATTRIBUTION_HEADER=0` is the client-side
  alternative. An engine-side failure after admission is a 5xx / `finish_reason: "error"`, never a 400.
- **D8. `tool_choice: "auto"` unconstrained unless a tool is `strict`.** As vLLM and SGLang (R §4.3): constraining
  `auto` would make every agent request a constrained one (D2). `required`, named and `parallel_tool_calls: false`
  constrain the call envelope; a `strict: true` tool also constrains its arguments (and `auto` gets a triggered
  tag).
- **D9. Strict conformance against a reject list.**
  - Reject (400 naming the feature) only for strict requests (`json_schema.strict: true`, a `strict: true` tool);
    non-strict schemas compile best-effort and the response reports nothing new (as vLLM / SGLang).
  - The strict reject list (R §6.2 and the reviewers' probes): `allOf` with ≥ 2 branches, `oneOf`, `not`,
    `if`/`then`/`else`, `multipleOf`, `uniqueItems`, `dependencies` / `dependentRequired` / `dependentSchemas`,
    `contains` / `minContains`, non-annotation siblings of `$ref` / `enum` / `const` / combinators, untyped
    `minLength` / `maxLength` / `minimum` / `maximum` / … (per keyword: untyped `properties` / `items` are enforced),
    required keys not in `properties`, remote and `$anchor` refs, `pattern` beside `format` or a length bound, and
    every `format` outside an allowlist the oracle verifies (xgrammar's date / date-time / ipv4 / hostname checks
    are syntax-only: `2024-02-30` passes; exact regular translations of these four are deferred, §10).
  - The oracle (CI, model-free): over all JSON-Schema-Test-Suite drafts, every schema the server accepts admits no
    invalid instance through the matcher; plus the typed probe table of R §6.2.
  - The runs: JSONSchemaBench test splits minus GitHub-trivial / -ultra and the 13 unsatisfiable Glaive schemas,
    `strict = true`, thinking off, no stop strings, greedy, `max_tokens` 2048; the **full** filtered split on Qwen3-8B
    Q4_K_M (5080) plus one sampled pass (T = 1, two seeds), and a 500-schema sample on the 27B + MTP (4090);
    validation with `validator_for(schema)` (the schema's own draft) and the format extras installed, the harness
    self-testing that each enforced format rejects an invalid string; the split is precompiled once before the
    timed run and every schema's compile time recorded.
- **D10. Stop strings.** A frontend stop string inside constrained output ends the request as `stop` (the user
  asked); conformance runs send none.

## 5. Numerics, judge and predictions (fixed before measuring)

- **L0-mask:** the plugin's `apply` vs xgrammar's CPU `apply_token_bitmask_inplace`: bit order, ids in
  `[len(tokenizer), V)` masked, all-allowed and padding rows untouched; sm_86 / sm_89 / sm_120a; memcheck clean.
- **Masked L0-dist** (M0.4 §5's method: 10⁵ draws at fixed seeds, χ² at p = 1e-3): a prefix-dependent toy grammar
  over a 16-token vocabulary (`TokenizerInfo(encoded_vocab=…)`) whose masks differ per column, the toy Markov target
  and a drafter proposing (prev + 1) mod 16, masks produced by the real `fill` probe. The exact law of the first three
  tokens is enumerated: P(x₁…x₃) = ∏ₜ F(softmax((z(xₜ₋₁) + log 1[M(x<t)]) / T)), with filters F applied after the
  mask (the sampler sees masked logits). Cases: greedy, T = 1, top-p 0.9; a stop-token draft. The harness replays
  every emitted sequence through a fresh matcher and fails on any refusal (part of the plants' signal).
- **Spec identity:** CPU exact (oracle drafter); GPU — the constrained bounded check: 64 JSON-schema prompts on the
  27B + MTP (A5000), no `ignore_eos`; finish reasons and lengths compared; `error` count 0; agreement scored against
  a teacher-forced reference whose logits are masked by a fresh matcher replay (masked argmax); a first divergence
  must be a near-tie of the masked logits — top-2 gap below `CONFIDENT_GAP` (`compare.py:34`), calibrated on the
  first divergences of M0.4's unconstrained 27B spec-vs-plain runs; lengths and finish reasons compared up to the
  first divergence; the judge's KL made −∞-safe (masked top-k logprobs hold −∞: `p · (log p − log q)` with p = 0
  is 0).
- **Conformance** (D9): accepted-but-invalid 0 with errors counted invalid; truncated share within 2 points of
  unconstrained.
- **Parsers:** streaming = non-streaming on every recorded output; fixtures from R §4.2 (hermes, XML with typed
  values, `<tool_call>` ending reasoning, Qwen3.5 `| string` vs Qwen3.8 `tojson` value serialization).
- **Planted bugs** (`planted.py`), each must fail the listed check:

| Plant | Must fail |
|---|---|
| `mask_column_shift` (column j masked with the state after j + 1 drafts) | spec identity (CPU, oracle drafter; GPU bounded); masked L0-dist |
| `no_rollback_after_probe` | spec identity; masked L0-dist; conformance with speculation |
| `stale_mask_depth1` (constrained rows allowed at depth 1) | conformance (errors counted) |
| `mask_bit_order` | L0-mask; conformance |
| `mask_rows_not_reset` | the graphs-vs-eager toy proof |
| `stop_tokens_masked` (a completed matcher's mask omits the stop tokens) | the finish-reason assertion of the isolation test |
| `closing_tokens_masked` (`}` / `]` never allowed) | conformance's truncation gate |

  Power (reviewer simulation, R §6 schemas): a one-step-stale or one-ahead mask excludes the true next token in
  108–109 of 109 random schema walks, so every plant fires within the first tokens.
- **Predictions:**
  - Conformance 100% of accepted; declared coverage of the filtered split under D9's predicate **64–67%**
    (reviewer measurement of the predicate), minus size-cap, memory-cap and timeout rejections — counted by step 2's
    precompile, run under the worker memory cap with two workers.
  - D2: as in D2.
  - Refused drafts inside JSON (27B + MTP, `any_whitespace` on): ≤ 15% of drafts; measured directly by
    `spec_drafts_refused / spec_drafted`, not by comparing acceptance across different texts.
  - Host cost: 2–75 µs per mask fill (R §1.2), (1 + k) fills per verify row; ≈ 1 ms for 64 rows with xgrammar's
    batch matcher at 8 threads.

## 6. Memory and capture

- Mask block `[max_seqs·(1 + k_max), ⌈V/32⌉]` int32 at most: 0.5 MiB for the 27B at max_seqs 4, k 3; 5.8 MiB for
  the 8B at max_seqs 64, k 4 — but only constrained rows are staged and copied. The Triton `apply` has no
  intermediate.
- Graphs double when a processor is selected (the 27B config: 27 → 54: buckets {1, 2, 4} × widths {1, 3, 4} × 3
  sampler variants × mask on/off). The graph-pool and capture-time deltas and the KV tokens lost are measured at step
  1 and recorded; the profile includes `apply`. With no processor selected, nothing changes (acceptance 6).
- **Measured (step 1,** `evidence/processor_overhead_qwen3-0.6b_rtx-a5000_2026-10-13.json`, Qwen3-0.6B bf16,
  rtx-a5000, default settings: max_seqs 64, three interleaved rounds; the toy processor, and the toy with a no-op
  `apply` for the core's own cost): graphs double (27 → 54 plain; 54 → 108 with n-gram k = 3); capture time doubles
  (1.07–1.17 → 2.11–2.20 s plain; 2.53–2.59 → 4.87–5.17 s with k = 3); the graph pool does not grow (526 and
  1568 MiB with and without: the masked graphs reuse the blocks the unmasked ones reserved); KV tokens lost are
  within the run-to-run noise of the pool sizing (plain 166.6k–172.8k without, 167.9k–172.0k with; k = 3 149.3k
  vs 148.0k in the last two rounds, ≤ 1%). The toy's dense `apply` costs nothing measurable over the no-op.
- Host RAM when the grammar processor is selected: two vetting workers at ≈ 1.1 GB resident each (≈ 2.2 GB, up to
  ≈ 4.4 GB at the watchdog bound) plus the 512 MB compiled-grammar cache — measured at step 2; if it matters on
  consumer hosts, one worker (more queueing) or a worker that never imports torch.
- **Measured (step 2,** `evidence/grammar_vetting_rtx-a5000_2026-10-13.json`, xgrammar 0.2.8, CPU, Qwen3's 152k
  and Qwen3.5's 248k vocabularies, no whitespace cap): a vetting worker is ready 3.2–3.7 s after its spawn (the
  respawn time) at 768 / 815 MB resident — two take ≈ 1.6 GB, plus ≤ 256 MB each while compiling (the budget, §11);
  a compiled JSON grammar takes 0.2–0.4 MB at the median (≤ 24 MB on the mid-size sample), so the 512 MB cache holds
  hundreds.

## 7. Steps

| # | Work | Checks | Days |
|---|---|---|---|
| 1 | Core §3.1–3.7 with the toy processor; plan §2 / §8 and design §7.3 / §7.6 / §7.7 / §7.8 / §9 corrected; denylist | core tests; spec matrix with the toy; graphs vs eager (GPU); full suite; capture and memory deltas | 3.5 |
| 2 | `grammar/xgrammar`: `grammar.constraint@1`, compile pool and timeout, matchers, probe, stop set, Triton `apply`, strict reject list with the oracle in CI; L0-mask, masked L0-dist, plants | L0 ×3 classes; oracle; plants | 3 |
| 3 | `frontends/http`: the pipeline, hermes / qwen_xml / think parsers (streaming), constraint builders, the three routers | parser fixtures; API tests; round trips | 4 |
| 4 | D2's measurement (A vs A′, single stream and mixed, 8B and 27B) | records; the rule's decision recorded | 1 |
| 5 | Evidence: conformance runs (D9), constrained bounded check, refused-draft share, Claude Code session | records | 2.5 |
| 6 | (only if D2's rule fires) the forward/sample split, panel-reviewed first | graphs; depth-1 share; D2 re-measured | 3 |
| 7 | Close: panel code review, plan §8.1 | review clean | 1 |

## 8. Risks

- **Template drift.** Qwen3.5 and 3.8 serialize parameter values differently (`| string` vs `tojson`, R §4.1); the
  parser follows the bundle's format version.
- **Host time at large batches:** 64 constrained rows × (1 + k) fills ≈ 1–5 ms per step; the batch matcher with 8
  threads (R §1.2) is the first remedy, the split (D2) the second.
- **Slow compiles** (R §6.2: Snowplow p50 ≈ 2 s on a loaded host, a few schemas over 60 s): the pool and the 10 s
  timeout bound their effect to the request itself.
- **Claude Code's request shape** is only partly confirmed by code (R §5.4): capture a real session's requests
  (headers redacted) before freezing the Anthropic subset.

## 9. Review findings, resolved

**Rev 1** (A: correctness, B: architecture, C: numerics / performance / tests).

| Finding | Resolution |
|---|---|
| B-B1: three frontend plugins cannot share a pipeline; `serve` runs one frontend | one package `frontends/http` with three routers (D7) |
| B-B2 / A-m3: the processor has no tokenizer | `BuildCtx.vocab()` → `VocabView` (§3.7) |
| B-B3 / A-M1 / C-m12: a single pinned buffer races at depth 1; partial writes leave stale rows | mask rows ride the Uploader; the whole block written each constrained step; plant `mask_rows_not_reset` (§3.6) |
| B-B4: `masks()` errors unattributable mid-launch | `fill` never raises; failures surface at `commit` (§3.1) |
| B-B5 / A-m1: resume double-counts `max_tokens` | `EngineRequest.resume_output`, `prompt_len` moved, supervisor stops subtracting (§3.5) |
| B-B6 / A-m7: installed = active changes every lock | opt-in via `--slots`, one processor, locks record it (§3.1, acceptance 6) |
| B-M1 / A-m5: protocol too wide; `variant` uncapturable | `variant` dropped; the core keys on a mask-rows boolean; `apply` stays the plugin's (D4: the kernel is format-specific) |
| B-M2: xgrammar's JSON as the contract | `grammar.constraint@1` plain dicts (D6) |
| B-M3: parsers duplicated per bundle | formats implemented once in the frontend package; bundles name them (D7, §3.8) |
| B-M4: frontend API changes unlisted | §3.8 |
| B-M5 / C-M1: the split undesigned; the rule measured on the wrong model and confounded | split sketched (D2); measured on identical tokens on the 8B and the 27B; predictions re-derived per step |
| B-M6: rejecting non-strict tool schemas breaks agents | strict-only rejection (D9) |
| B-m1, m2: two depth mechanisms; `admit` name | gate from the processor's `open`; renamed `open` / `close` |
| B-m4: frontend options unreachable | per-request `strict` (D8) |
| B-m6 / D7 scoping | argument parsing in the bundle's `render_chat`; billing-header drop scoped and optional; 5xx after admission |
| B-m7: plan / design drift; denylist; LOC | step 1 updates plan §2 / §8 and design §7.3 / 7.6 / 7.7 / 7.8 / 9; denylist terms; 3650; +260 |
| A-M2: the prefill path is unmasked | one helper on every path (§3.6) |
| A-M3 / C-m10: logprobs −∞ break JSON | masked logprobs, clamped at −9999 in the frontends (§3.6) |
| A-M4: one sequence's error leaves its siblings running | `RequestError` aborts the request (§3.4); `open` before registration |
| A-M5 / C-M5: synchronous compile can hang the engine | compile pool, 10 s timeout, `ready` gate (D5) |
| A-m2: builtin Qwen tags refuse `<tool_call>` ending reasoning; `required` allows two calls | our own tags from `tool_format` (D6); fixtures |
| A-m3: `ignore_eos` with a grammar | rejected at `open` (D5) |
| A-m4: `eos_not_allowed` cannot fail | replaced by `stop_tokens_masked` with a finish-reason check (§5) |
| A-m6 / C-m11: memory claim wrong; unpack transient; capture doubling | Triton `apply` without intermediate; buffer sizes and the doubling stated, deltas measured (§6) |
| A-m8: D2 compares different texts | identical tokens via the all-allowed processor (D2) |
| A-m9: draft copy-back details | same snapshot, double-buffered, `min(n_draft, k)`, the column after the last draft; counter in stats (§3.3, D3) |
| A-m10: lifecycle hooks on the right paths | `close` on finish / abort incl. never-started siblings; `commit` gets the appended tokens, skipped for finished rows (§3.1) |
| C-M2: mixed batches unmeasured | A vs A′ measured and chosen by the mixed results (D2) |
| C-M3: plants hide behind error finishes; no spec in conformance | errors count invalid; a conformance pass with speculation; CPU identity with the oracle drafter (§5) |
| C-M4: the M0.4 bounded judge cannot judge constrained runs | a constrained bounded check (§5) |
| C-M6: reject-list holes; draft mismatch; silent format checks; ill-posed CI rule | list extended (`$anchor`, `dependencies`, pattern + format, format allowlist); `validator_for`; format extras with a self-test; the oracle rule restated (D9) |
| C-M7: truncation ungated | truncated share gated vs unconstrained; thinking off; plant `closing_tokens_masked` |
| C-M8: masked L0-dist does not exercise the probe | toy grammar, real `fill`, enumerated law with filters after the mask (§5) |
| C-M9: `any_whitespace` undecided | on, with a run cap of 16; refused drafts measured directly (D5, §5) |
| C-m13: coverage predicate, sample size | 68% from the real predicate; full 8B split + a sampled pass; 500 on the 27B (D9) |
| C-m14: L0 reference | xgrammar's CPU apply (§5) |

**Rev 2.**

| Finding | Resolution |
|---|---|
| B-M1: a constraint silently ignored with no processor selected | processors declare their `resource`; `Engine.add` rejects unserved `ext` keys (§3.1) |
| B-M2 / A-m3 / C-R2-1: async compile errors: wrong status, no channel; timeouts counted invalid | `ready` may raise; `RequestError` kinds `client` / `busy` / `engine` on `TokenDelta` → 400 / 503 / `error`; compile errors and timeouts are rejections in conformance (§3.1, §3.4, acceptance 1, 5) |
| A-M1: an idle engine with only a compiling request raises "no progress" | not-ready requests `BLOCKED`, skipped; `step()` returns empty; the loop polls (§3.1) |
| A-M2: decoded token strings break multi-byte constraints | `VocabView` carries the tokenizer object; non-ASCII enum fixture (§3.7) |
| B-M3 / C-R2-2 / A-m4: a timed-out compile holds its thread | killable worker processes, negative cache, bounded queue (D5); free-stream p99 during a cold compile measured |
| B-m1 / C-R2-6 (block cost): the core writes all-allowed rows | compact constrained rows + row indices; `apply(logits, mask, rows)`; the format stays the plugin's (§3.6) |
| A-m5: capture reads an all-zero mask | capture stages an out-of-range index list (§3.6) |
| B-m2: `VocabView` path contradictory | `build(ctx, vocab)` argument form (§3.1, §3.7) |
| B-m3 / A-m2: resume replay contradictory | the core calls `commit(seq, replayed)` once after `ready`; `EngineClient.generate` gains `resume_output` (§3.5) |
| B-m4 / A-m1: streaming vs `commit` order; commit after finish | `commit` before stop / length handling and before deltas; a `RequestError` drops the step's tokens (§3.1) |
| B-m5: request-wide error after a restart | the supervisor aborts sibling rids on an `error` delta (§3.4) |
| B-m6: manifest `max_depth` dead | removed in step 1 |
| A-m6: acceptance 6 promised too much | byte-identical at a pinned `state_memory` |
| A nit: "processed mode" | wording fixed (§3.6) |
| C-R2-3: whitespace cap 16 refuses pretty JSON | 64 (D5) |
| C-R2-4: constrained bounded check undefined | `CONFIDENT_GAP` calibrated on M0.4 runs, lengths to the first divergence, −∞-safe KL (§5) |
| C-R2-5: coverage prediction high; date-time 400s | 64–67%; exact format regexes deferred (§10) |
| C-R2-6: D2 toy misses fill / apply; "combined" undefined | measured fill and apply added; A′ unless free rows slow > 1.15× (D2) |
| C-R2-7: L0-dist harness must fail on refusals | replay through a fresh matcher (§5) |

**Rev 3.**

| Finding | Resolution |
|---|---|
| C-R3-1 / A (rev 3 major): compiled grammars cost more to ship and deserialize than to compile (up to ≈ 2 GB, 46 s) | workers vet only (timeout, kill, a 64 MB size cap) and return no grammar; a helper thread compiles for use; sizes and times recorded (D5) |
| C-R3-2: worker start method, threads, respawn, tokenizer match | `spawn`, `max_threads=2`, bounded queue across respawns, `TokenizerInfo` equality asserted (D5) |
| B-m1 / A nit: the unserved-`ext` rule ambiguous, duplicating the manifest | any `ext` key no active plugin's manifest provides is rejected; no runtime attribute (§3.1) |
| B-m2: worker lifecycle | `spawn`, exit on parent death, `shutdown()` from `Engine.close` (§3.1, D5) |
| B-m3: blocked-only work spins the loop | `has_work()` excludes it (§3.1) |
| B-m4: existing admission errors need a kind | plain `ValueError` from `add` is kind `client` (§3.4) |
| A (rev 3 minor): the whitespace cap misses `json_object` and untyped subschemas | `json_object` from our capped EBNF; untyped subschemas stated, covered by `max_tokens` and the truncation gate (D5) |
| B nit: §10 before §9 | moved |

**Rev 4.**

| Finding | Resolution |
|---|---|
| A / B (rev 4 major): a blocked request on an idle engine is never stepped | `has_work()` stays true while blocked; the loop polls every 5 ms via `engine.idle_for()`, promoting ready requests; toy tests for success and timeout on an idle engine (§3.1) |
| C-R4-1: a worker can exhaust host memory before the size check | 2 GB `RLIMIT_DATA` per worker, `oom_score_adj` 1000, a dead worker = "schema too complex" (D5) |
| A-m1: a timeout cached forever | timeouts expire from the negative cache after 10 minutes (D5) |
| A-m2 / B-m1 / C-R4-3: one helper thread queues every compile; unbounded backlog; cold TTFT | two helper threads, a backlog bounded by vetted time, `close` drops queued compiles; cold-constraint TTFT recorded (D5) |
| B nit: the `ext` key carries no version | `ext["grammar.constraint@1"]` (D5) |
| A nit / C-R4-4: the `json_object` EBNF untested | derived from the builtin with only the whitespace rule replaced; oracle fixtures (D5) |
| C-R4-2: coverage ignores the size cap | restated net of size, memory and timeout rejections (§5) |

**Rev 5.**

| Finding | Resolution |
|---|---|
| C-R5-1: an OOM-killed worker rejects a valid schema forever | failures under the worker's own limit are `client` and permanent; kernel OOM kills are `busy`, expiring like timeouts (D5) |
| C nit: the worker cap far above the size cap | 1 GB, recalibrated at step 2 from the measured peak-to-final ratio (D5) |
| B note: confirm the engine-side peak | workers report each compile's peak RSS, recorded in the D9 runs (D5) |

**Rev 5, round 2.**

| Finding | Resolution |
|---|---|
| A (rev 5 major): `RLIMIT_DATA` caps virtual memory, ≈ 6.7 GB after the imports; a tight limit hangs or crashes the worker | a parent watchdog on resident memory (baseline + 1 GB) kills the worker → `client`; lean worker environment, `TokenizerInfo` as JSON, `PR_SET_PDEATHSIG`; any unreported death `busy` with the TTL; a positive control in the test (D5) |
| A-m1: `oom_score_adj` makes workers the host's first victim | kept, but an OOM kill is `busy` and expires, never a permanent rejection (D5) |

**Rev 5, round 3.**

| Finding | Resolution |
|---|---|
| B-m1: `PR_SET_PDEATHSIG` follows the creating thread | workers started and respawned only by the watchdog thread; the child checks `getppid()` after `prctl` (D5) |
| B-m2: the workers' host RAM unrecorded | §6, measured at step 2 |
| C nit: respawn time | likely 2–5 s, measured; the queue bound sized to it (D5) |

## 10. Deferred, deliberately

- Exact regular translations of the date / date-time / ipv4 / hostname formats (v1; rejected under strict now).
- More than one logits processor (ordering, chained masks).
- Constraining `tool_choice: "auto"` by default (D8).

## 11. Build record

### Step 1 (2026-10-13)

Built: §3.1–3.8 (core +184 code lines, 3650 → 3834; budget ≈ +275), the denylist terms, the plant
`mask-rows-not-reset`, design §4.1 / §7.3 / §7.4 / §7.6 / §7.7 / §7.8 / §9 corrected. The toy processor
`testing.toy_mask` (an entry-point test plugin, so engine processes can select it) masks odd tokens after a marker.

Proofs:
- `plugins/tests/test_logits_processor.py`, tiny Qwen3 on the CPU: one matrix (depth 0, depth 1 with A′ and A; no
  drafter and the oracle drafter right / wrong / mixed; roomy and preempting pools; two passes for prefix hits;
  chunked prefill; greedy constrained, free, and an n = 2 sampled constrained request): greedy outputs equal HF's
  logits masked by the rule, sampled ones the depth-0 run, the processor's `commit` saw exactly the emitted tokens and
  its `fill` exactly the drafts the drafter wrote, every constrained row launched with its tokens committed, free rows
  overlapped under A; one engine for the lifecycle: blocked on an idle engine until `ready`, a `busy` timeout, a
  `client` rejection at `open`, an unserved `ext` key (also with no processor selected), an n = 2 request failing
  mid-generation beside unchanged requests, a resumed request.
- `plugins/tests/test_ngram.py` (GPU, Qwen3-0.6B): graphs equal eager on the same padded plans with three constrained
  requests, two finishing early; under `mask-rows-not-reset` the graphed run differs.
- `test_supervisor.py`: a kill in the middle of a constrained stream (marker = its first token) resumes it exactly; a
  failed sequence aborts its siblings' engine requests. `test_ipc.py`: a request blocked on an idle engine process
  completes, the loop polling every 5 ms rather than spinning; rejections carry kind `client`.
- `test_frontend_openai.py`: kinds map to 400 / 503 / 500; masked logprobs clamp to −9999.

Measured (§6): graphs and capture time double with a processor selected; graph pool unchanged; KV tokens lost
≤ 1%, within the pool sizing's run-to-run noise.

Deviations (simplest option consistent with the decisions):
- The toy proofs live in `plugins/tests`, the toy plugin in `testing/` (acceptance 7 and §3 say `core/tests`): every
  proof needs a built model (the tiny Qwen3, the oracle drafter), which `core/tests` never builds; M0.4 placed its
  toy drafters the same way.
- `FrontendBundle.constraint(…)` is an optional hook documented in the protocol, not a required method: D6 / D7 build
  constraints in the frontend from `tool_format`; a bundle implements it only for a format the frontend lacks.
  `tool_parser(tools)` takes the request's tools.
- `VocabView.tokenizer` is a callable (`WeightSource.tokenizer`, loaded when called): test checkpoints carry no
  tokenizer, and the toy needs none.
- A `RequestError` ends the request with the kind its raiser chose (`commit` failures are `engine` by contract).
- `Seq` gains `constrained` and `drafts` (the host copy of a constrained sequence's drafts).
- The masks' row indices are padded on every masked step, eager ones too (one code path; the graph reads them all).
- The existing OpenAI frontend maps kinds now (`client` 400, `busy` 503, otherwise 500 — an engine failure without a
  kind was a 400 before; streamed errors carry the same code). **Deferred to step 3** (`frontends/http`): a failure
  after the first token as `finish_reason: "error"` on the choice (today an SSE error event, or a 5xx when not
  streamed).
- `logits_processor` keeps its `ordered_multi` cardinality (§10 defers several processors); the engine selects at
  most one, by `EngineConfig.slots`.
- `Engine.add` also rejects a `resume_output` at or past `max_tokens` (a finished sequence is never resubmitted).
- A resumed request's `prompt_len` excludes its output, so a recurrent state's prompt-end checkpoint
  (`std.state.recurrent`) is published at the original prompt's end, not after the resumed output.
- `Supervisor.generate` refuses `resume_output` (it resumes requests itself).


### Step 2 (2026-10-13)

Built: `plugins/holon_plugins/grammar/xgrammar` — `constraint.py` (the `grammar.constraint@1` forms and their
translation, the strict reject list), `vetting.py` and `worker.py` (D5's workers and watchdog), `plugin.py` (the
processor: negative cache, helper threads, compiled-grammar cache, matchers, the probe, the stop set), `kernel.py` (the
Triton `apply`); the plants `mask-column-shift`, `no-rollback-after-probe`, `mask-bit-order`, `stop-tokens-masked`,
`closing-tokens-masked` (`judge/holon_judge/planted.py`); the JSON-Schema-Test-Suite vendored unmodified at a pinned
commit (`plugins/tests/fixtures/jsts.sh`, checked by `scripts/ci/run_all.sh`); `bench/grammar_vetting.py`. No core
change. The `grammar.constraint@1` fields are D6's table.

Proofs (`plugins/tests/test_grammar.py`, 29 tests, ≈ 45 s of test time; with `test_logits_processor.py` 54 tests in
65–92 s on a shared host):
- The oracle (CPU, model-free, in CI): every draft of the suite — 454 groups accepted under strict (drafts 4 to
  2020-12; every draft-3 schema refused), their 978 invalid instances all refused, compact and with `\u` escapes —
  plus a 30-row typed probe table (R §6.2's enforced features, a non-ASCII enum and a draft-07 recursive tree among
  them), each accepted with its valid instance admitted, and 16 schemas xgrammar would read differently from JSON
  Schema (found by the review and the panel; refused).
- `json_object`: every 2020-12 object instance to depth 8 (537), compact and at indents 2 and 4, a 300-space run,
  non-ASCII text and escaped surrogate pairs are admitted and 22 malformed texts refused, as by xgrammar's builtin JSON
  grammar; unlike it, a top-level array or scalar is refused.
- Every constraint form compiles and admits / refuses samples; malformed dicts and schemas, a nested strict violation
  and a regex compile error are refused; a compile error reports `client`, running out of memory `busy`.
- L0-mask (sm_86, rtx-a5000): `apply` equals xgrammar's CPU `apply_token_bitmask_inplace` bit for bit on 152k, 248k
  and 1000-token vocabularies, with all-allowed, all-refused and padding rows, eager and replayed in a CUDA graph with
  new contents; memcheck 0 errors (`plugins/tests/memcheck_grammar_apply.py`); `mask-bit-order` fails it. sm_89 and
  sm_120a: pending (to run on the consumer cluster: the `-k apply` tests and the memcheck script).
- Masked L0-dist (10⁵ sequences, χ² at p = 1e-3, the real `fill` probe over a prefix-dependent toy grammar of 16
  tokens with a multi-character token and stops after two or three tokens): greedy exact; T = 1 p = 0.48; top-p 0.9
  p = 0.57; stop-token drafts (the drafter proposes the stop token first) p = 0.09; no refusals in any replay. The
  plants, run once (not kept: the end-to-end checks below keep them failing): `mask-column-shift` — 553 refused
  sequences and p = 0 at T = 1, p = 0 with stop drafts; `no-rollback-after-probe` — 678 refused, p = 0 at T = 1, 207
  refused with stop drafts.
- End to end (CPU, a tiny Qwen3 over Qwen3's 152k vocabulary and tokenizer): a regex with n = 2, a strict schema (the
  random model fills it with whitespace runs and is cut at `max_tokens`), a structural tag around a schema and
  `json_object` beside a free request give exactly HF's greedy tokens masked by a fresh matcher, with the same finish
  reasons — plain, and with the oracle drafter's right, wrong and mixed drafts (refused drafts counted). The plants:
  `mask-column-shift`, `no-rollback-after-probe` (speculating), `mask-bit-order` and `stop-tokens-masked` end the
  constrained requests with `error` (a commit the matcher refuses) instead of `stop`; `closing-tokens-masked` cuts
  `json_object` at `length` instead of `stop` (§5's truncation signal).
- Token-level masks with the real tokenizer: a non-ASCII enum (byte-fragment tokens) is admitted token by token,
  ids past the tokenizer (151669 … 151935) are never allowed, a stop token only once the constraint is complete.
- Isolation on one engine, beside requests that finish unchanged: a regex compile error ends its request `client`,
  a compile past a lowered timeout (0.3 s) `busy` — the same constraint again is refused at once from the negative
  cache — and a 53 MB compile under a lowered watchdog budget (16 MB) `client`; the respawned worker then vets a small
  schema (2 MB) under that budget (positive control); `ignore_eos` with a constraint, malformed and too deeply nested
  constraints, stop tokens past the vocabulary, a constraint over 1 MiB and a constraint on a deployment without the
  processor are 400s (kind `client`), never exceptions that end the engine. A checker whose workers cannot start fails
  its waiting constraint `busy` and refuses new ones instead of leaving them blocked forever.
- Step-1 review items: `test_logits_processor.py` records the running constrained sequences each decode step leaves
  out — none under A′, some under A (a mutation making A′ behave as A now fails all 8 depth1-commit cases); the
  preempting cases assert preemptions > 0. Design §4.2 notes one logits processor.

**Decision S2-1 (panel, after step 2): no whitespace cap.** Measured (this evidence, before and after; 152k / 248k):

| | cap 64 (at 69357e5) | no cap (at b328173) |
|---|---|---|
| tool schema: compiled, compile | 13.4 / 26.3 MB, 43 / 83 ms | 0.19 / 0.27 MB, 15 / 28 ms |
| `json_object`: compiled, compile | 13.5 / 26.4 MB, 54 / 114 ms | 0.14 / 0.21 MB, 14 / 30 ms |
| sample schemas: compiled p50, compile p50 | 13.4 / 26.3 MB, 41–45 / 80–86 ms | 0.16–0.32 / 0.23–0.38 MB, 15–17 / 28–33 ms |
| mask fill, median over a tool-call walk | ≈ 33 µs (the panel's measurement) | 3.8 / 4.1 µs (`json_object`'s pretty walk 20 / 15 µs) |
| the 512 MB cache holds | ≈ 38 / 19 grammars | hundreds |

The cap is dropped; a whitespace runaway ends at `max_tokens` (step 5 counts truncations ending in a whitespace run;
a cap of 16–32 returns only if they appear).

Measured (D5, `evidence/grammar_vetting_rtx-a5000_2026-10-13.json` at b328173: no whitespace cap; mid-size subsets
only, 40 schemas each of Glaiveai2K, Github_easy, Github_medium and 29 of WashingtonPost; 152k / 248k vocabularies):
- worker start and respawn 3.2–3.7 s; resident 768 / 815 MB when ready;
- compile (worker ≈ engine thread): tool schema 15 / 28 ms, `json_object` 14 / 30 ms, a 10-tool structural tag 18 /
  42 ms, schemas p50 15–17 / 28–33 ms, p90 up to 2.9 s / 1.1 s, max 8.7 s / 9.7 s (one Github_easy and two
  Github_medium schemas time out at 10 s with the larger vocabulary); compiled p50 0.16–0.38 MB, ≤ 24 MB;
- peak resident growth while compiling ≤ 14 MB;
- mask fill: median 1.3–4.1 µs over the tool-call, tag and regex walks, 15–20 µs over the pretty-printed
  `json_object` walk, ≤ 98 µs; at a schema's start p50 14–26 µs, ≤ 273 µs;
- the strict reject list accepts 96 of the 149 sampled schemas (64%; §5's predicted coverage 64–67%).

Review: one reviewer, four rounds (4504ed3 → 92db331 → 69357e5 → d2a14a5 → 0ec738a; clean), each fix with a test that
failed before where a test could show it; then the panel's step-2 items (b328173). Found and fixed: malformed
constraints and out-of-vocabulary stop ids escaping `open` as exceptions that end the engine process; `$ref` targets
in unwalked locations, `properties` names also matched by `patternProperties`, `propertyNames` beside `properties`,
older drafts' `prefixItems` / `exclusiveMinimum` / `id`, and escaped `$ref`s bypassing the reject list; workers dying
before they are ready leaving requests blocked forever; watchdog-thread exceptions disabling vetting; abandoned
compiles held outside the cache; unbounded negative and positive caches; super-linear checks on the engine thread (ref
re-walks, Python regex over client patterns, enum × type lists); a class-end misread reopening `[^\s]`; the panel's
`.` (U+2028 / U+2029), a top-level array in `json_object`, and out-of-memory compiles refused permanently.

Deviations (simplest option consistent with the decisions):
- **xgrammar 0.2.8, not 0.2.3.** 0.2.3 expands bounded repetition (a 64-character whitespace cap compiled a
  3-property schema to 330 MB and `json_object` to 763 MB) and 0.2.8 is vLLM's pin. 0.2.8 needs apache-tvm-ffi
  ≥ 0.1.11 (the shared stack had 0.1.9; FlashInfer 0.6.12 accepts it and its kernel tests pass at 0.1.14). The oracle,
  fixtures and tag forms run at 0.2.8.
- **No whitespace cap** (decision S2-1, above; D5 updated).
- **`json_object` is an object**, not xgrammar's builtin JSON (which admits a top-level array).
- **Workers are fresh interpreters started with `subprocess`** (`python -P -m …worker`) rather than
  `multiprocessing`'s `spawn`: the same isolation with an explicit environment (also `CUDA_VISIBLE_DEVICES=""`). The
  `TokenizerInfo` JSON (5–8 MB) goes through an inherited, already unlinked file read with `pread`: written to a pipe
  it would block the watchdog thread until the child read it, and a named temporary file could be removed under a
  long-running server.
- **Watchdog budget 256 MB** (D5: 1 GB, recalibrated at step 2 from the peak-to-final ratio): a compile measured
  ≤ 1.4× its compiled size at its peak, so one that ends under the 64 MB size cap peaks under ≈ 90 MB.
- **The vetting queue holds 8 waiting constraints** (more are `busy`): with two workers, a 10 s timeout and a 3.5 s
  respawn, the last waits at most ≈ 54 s. Constraints waiting, running or reported are not queued twice. With no
  worker ready and three starts failed in a row, the checker is down: waiting and new constraints are `busy` (not
  remembered) until a worker starts. A compile that runs out of memory is `busy` (remembered 10 minutes).
- **A constraint's JSON is at most 1 MiB** (a 400 beyond), bounding the checks `open` runs on the engine thread.
- **The reject list grew where the oracle, the review and the panel found holes:** `time`, `ipv6` and `email` leave
  the format allowlist (leap seconds at a wrong hour; a leading zero in an embedded IPv4 octet; a `\u00a9` escape in a
  quoted local part), leaving `uri`, `uuid`, `duration`; `contentMediaType` / `contentEncoding` / `contentSchema`
  (draft 7 asserts them); patterns (ECMA 262, JSON Schema's dialect) with an unescaped `.` outside a class (xgrammar's
  admits U+2028 / U+2029), `\S` or `\s` in a negated class (ECMA's whitespace is Unicode's); `patternProperties` or
  `propertyNames` beside `properties`; `$id`, `$schema` or a string `id` below the root; `$schema` other than drafts
  4, 6, 7, 2019-09, 2020-12; `prefixItems` before 2020-12, draft 4's `exclusiveMinimum` / `exclusiveMaximum`, `items`
  as an array and `additionalItems`; `$dynamicRef` / `$recursiveRef` and their anchors; `$ref`s with `%` or `~`
  escapes; every `$ref` target is checked like the rest. `type` beside `enum` / `const` is allowed when every value
  has that type. A `propertyNames` schema needs `type: string` for its typed keywords (like any other schema).
- **`spec_drafts_refused` is not in `Engine.stats()` yet:** the processor counts refused drafts
  (`XGrammar.drafts_refused`), but surfacing them in the engine's stats needs a general core hook (an optional
  `stats()` of the processor merged into `Engine.stats`), deferred to step 5, where the share is measured.
- xgrammar's `strict_mode` (no undeclared properties) is on for every schema, strict or not (D9's "stricter" rows).
- The masked L0-dist's plant checks ran once and are recorded above rather than kept as tests (≈ 8 s); the
  end-to-end plant checks (< 1 s each) stay.

Step-5 obligations (panel S2-2, S2-3):
- declared coverage over the **full** filtered split, per tokenizer (152k and 248k), with compile timeouts counted per
  tokenizer (the 10 s timeout refuses more at 248k: 1 of 40 Github_easy and 2 of 40 Github_medium schemas here);
- cold-constraint vs warm (cached) TTFT;
- the precompile pass of D9 on an idle host (the compile times above came from a shared one);
- the truncations that end in a whitespace run (S2-1's condition for a cap).

### Step 3 (2026-10-13)

Built: `plugins/holon_plugins/frontends/http` (was `frontends/openai`; entry point `frontends.http`, extra `http`;
provides `frontend.openai@1`, `frontend.anthropic@1`, `frontend.openai_responses@1`) — one app (`app.py`) with a
router per dialect: `openai.py` (chat and text completions), `anthropic.py` (`/v1/messages`, `count_tokens`),
`responses.py` (`/v1/responses`); `pipeline.py` (a router's `Conversation` → `Chat`: render, constraint, a parser
per choice; `Blocks`, the block shape Anthropic and Responses share); `formats.py` (think@1, hermes@1, qwen_xml@1 /
@2: each format's streaming parser and the constraint of its output, plain `grammar.constraint@1` dicts); `web.py`
(errors by kind, event streams with a keepalive). The bundles name their formats: `reasoning_format` think@1;
`tool_format` hermes@1 for Qwen3 and Qwen3 MoE (declared; their template writes arguments sent as JSON text as they
are), and for Qwen3.5 / 3.8 read off the template by rendering a call with a boolean argument — qwen_xml@1 (Qwen3.5,
`| string`: Python's `True`) or qwen_xml@2 (Qwen3.6+ / 3.8, `tojson`); the Qwen3.5 bundle's `render_chat` parses
tool-call arguments sent as JSON text into objects (the XML templates iterate them; Qwen3.8's raises on text). Each
tool format exposes its call opener (`opener`, beside `call`). No core change.

Proofs (`plugins/tests/test_frontend_http.py`, 78 tests of which 50 new, ≈ 27 s with the GPU one; the package is
≈ 1.2k lines, plan §2 budgets frontends ≈ 1.5k):
- Parsers, R §4.2's fixtures (12): hermes with reasoning the model opens, text and two calls, both markers inside an
  argument string, arguments written as a JSON string, a body that is no call, a call cut by the end of the text;
  XML with `<tool_call>` ending reasoning without `</think>`, closed reasoning then text then a call, typed values
  under @1 and @2 (`True` / `None` are a boolean / null under @1 and strings under @2; `007` for a string parameter
  and `5` for an `anyOf` [string, null] one stay strings; multi-line strings), a call cut inside a parameter,
  markers that start nothing. For each, the parse of every split of the text (character by character, and random cuts)
  equals the whole text's.
- Constraints (7 forms + 5 choices): every builder's output passes `constraint.check` and compiles; samples admitted
  and refused (a second call under `parallel_tool_calls: false` or a named choice, text before a required call, a
  wrong argument type under a strict tool, JSON before open reasoning closes); every admitted sample parses into the
  expected number of calls. `auto` without a strict tool is free (D8); `none` neither renders nor parses tools; a
  response format with a constrained tool choice, `required` without tools and a named tool not among them are 400s.
- Dialects (fake engine): OpenAI tool calls streamed as `tool_calls` deltas reassemble to the non-streamed message;
  each tool choice's constraint reaches the engine; Anthropic's request becomes the expected conversation (billing
  header dropped and counted, tool results ahead of the text, thinking as reasoning, `any` + no parallel → exactly
  one call, thinking disabled → `enable_thinking: false`), `count_tokens`, its error shape, and its streamed blocks
  reassemble to the non-streamed content; Responses' input items (a call joins the assistant turn, its output a tool
  message, reasoning the next assistant's) and its streamed items equal the non-streamed output, sequence numbers in
  order; a failure after the first token is `finish_reason: "error"` (chat, streamed and not), `event: error` /
  `response.failed` streamed, a 500 otherwise — never a 400 (D7; deferred from step 1); a stream quiet for 15 s
  sends a keepalive (Anthropic's `ping` event, an SSE comment otherwise) and ends normally.
- Bundles with the real templates (Qwen3-0.6B, Qwen3.5-9B, Qwen3.8-27B): the format named, a call sent with JSON-text
  arguments renders, and the frontend's parser reads the rendered call back.
- Round trips: a tiny random Qwen3 over Qwen3's tokenizer and template (CPU, grammar.xgrammar; whitespace-only tokens
  given zero logits so its greedy JSON ends) — a forced call and the conversation with its result, through each
  dialect; Qwen3-0.6B (GPU): a required call (`{"city": "Paris"}`), the streamed answer from its result, reasoning the
  model opens (Anthropic, a thinking block then the text) and a strict JSON schema (Responses).

Deviations (simplest option consistent with the decisions):
- **The parsers take `final`** (`feed(text, final=False)`): a streaming parser holds back what may still be a marker
  and needs the end of the text to release it; §3.8's `Parser.feed(text)` has no end. The frontend uses its own
  parsers, chosen by format name; `FrontendBundle.reasoning_parser()` / `tool_parser()` (every bundle returns None)
  are not consulted; step 7 deletes them and the `Parser` protocol (panel, below).
- **Format versions:** qwen_xml@1 is Qwen3.5's template (`| string`), @2 Qwen3.6+ / 3.8's (`tojson`). The Qwen3.5
  bundle reads its format off its template (the plugin serves both), rather than declaring it.
- **hermes emits a call when it closes** (its arguments as one delta, checked JSON; a body that is not a call stays
  text): its body is read as one JSON value (`raw_decode`) followed by whitespace and `</tool_call>` — a
  `</tool_call>` inside one of its strings does not close it; qwen_xml streams a parameter at a time (`{`, each
  `"key": value`, `}`), a cut parameter dropped at the end.
- **Text and reasoning runs are stripped** at both ends (the formats put newlines around markers).
- **The reasoning state is read off the prompt's last 16 tokens:** opened (`<think>`: Qwen3.5 / 3.8 thinking on),
  closed (`</think>`: thinking off), or left to the model (Qwen3: a leading `<think>` opens it). The constraint
  follows it: reasoning is free text holding neither `</think>` nor `<tool_call>`, ended by `</think>` and
  whitespace or — before calls — by the first call; optional when left to the model, absent when closed.
- **Constrained envelopes:** a non-strict tool's arguments follow its schema best-effort (`strict: false`, as vLLM);
  `required` admits text after the calls (xgrammar's triggered tags), named and `parallel_tool_calls: false` end at
  the call; a response format with a constrained tool choice is a 400; `tool_choice: "none"` neither renders nor
  parses the tools; `finish_reason` is `tool_calls` whenever a call was parsed (named too; vLLM says `stop` there).
- **Failures after the first token:** OpenAI `finish_reason: "error"` on every running choice (a 200, or a stream
  chunk); Anthropic `event: error` / 500 (`api_error`); Responses `response.failed` / 500. Before the first token,
  the kind's status (400 / 503 / 500), in each dialect's error shape.
- **Anthropic:** text blocks joined with newlines (as vLLM joins text parts); `redacted_thinking` dropped; image
  blocks and server tools (a `type` other than `custom`) are 400s; `thinking: disabled` → `enable_thinking: false`;
  `stop_sequence` is never reported (`end_turn`, as llama.cpp and SGLang); thinking signatures are empty.
- **The billing-header drop** is `build_app(drop_billing_header=True)`; `holon serve` has no flag for it (the core
  passes frontends no options); dropped blocks are counted in `/metrics` (`billing_headers_dropped`).
- **Responses:** an absent `strict` is false (OpenAI's Responses default is true, which under D8 would constrain
  every request with tools); `store`, `previous_response_id` and `background` are 400s; `reasoning.effort` ignored.

Step-5 commands (the real round trips and the Claude Code session; `--slots logits_processor=grammar.xgrammar` serves
constrained tool choices and response formats):

    holon serve <Qwen3-8B Q4_K_M .gguf> --slots logits_processor=grammar.xgrammar --port 8000
    holon serve <Qwen3.8-27B UD-IQ3_S .gguf> --port 8000 --max-seqs 4 --max-model-len 32768 \
      --max-step-tokens 2048 --memory-fraction 0.9 --max-draft-tokens 3 --slots logits_processor=grammar.xgrammar \
      drafter=drafters.mtp draft_policy=std.spec.fixed_k verifier=std.spec.greedy \
      --plugin-options recurrent.checkpoints=4 kv.format=q8_0 spec.k=1:3,3:2   # M0.4's MTP flags, a longer context
    # OpenAI chat / Anthropic / Responses round trips: a tool call (auto, required, named), its result, the answer
    ANTHROPIC_BASE_URL=http://127.0.0.1:8000 ANTHROPIC_AUTH_TOKEN=holon ANTHROPIC_MODEL=<served model id> \
    ANTHROPIC_DEFAULT_HAIKU_MODEL=<served model id> CLAUDE_CODE_SUBAGENT_MODEL=<served model id> \
    claude -p "<the scripted multi-tool task>" --allowedTools "Read,Write,Edit,Bash,Glob,Grep" --output-format json

  The served model id is `/v1/models`'s; capture the session's requests (headers redacted) to check R §5.4's
  unverified items, and count the dropped billing headers in `/metrics`.

**Step 3, panel review** (clean for step 4; fixed before step 5):
- *Major:* the hermes parser closed a call at the first `</tool_call>`, so a grammar-admitted call whose argument
  string held that text parsed into text. Fixed as above; fixture "markers inside an argument string" (fails before).
- *F3-1:* fixtures for hermes arguments written as a JSON string (the re-parse) and an `anyOf`-typed XML parameter.
- *F3-2:* parsing a long call cost quadratic time (each piece re-searched and re-copied the held text: a 400K-character
  qwen_xml call took 18.4 s, hermes 6.4 s, a stop-string choice 9.1 s for 1.6M characters). Held text is now kept as
  pieces and searched only where a marker may newly start (`formats._Until`), a block's text as pieces joined once,
  and a choice keeps only its unsent text: 0.13 s / 0.13 s / 1.3 s (`/tmp` bench, CPU). Event streams send a keepalive
  after 15 s of silence (`web.KEEPALIVE`).
- *F3-3:* the shared `plugins/tests/conftest.py` collects garbage only after tests that left GPU memory reserved
  (≈ 0.24 s per test before); the GPU suites still release their pools (`test_frontend_http.py` then `test_ngram.py`
  in one run: 83 passed).
- *m2 / m3:* Qwen3 and Qwen3 MoE declare hermes@1 and pass arguments through; the template probe and the argument
  parsing stay in the Qwen3.5 bundle; the pipeline reads each format's call opener instead of naming `<tool_call>`.

For step 7 (deferred, both core):
- delete `FrontendBundle.reasoning_parser()` / `tool_parser()` and the `Parser` protocol from `holon.api` — dead API:
  the frontend owns the formats;
- pass `EngineConfig.plugin_options` to `FrontendPlugin.serve`, so frontends read namespaced options (e.g.
  `http.drop_billing_header`; today a `build_app` argument only).

For step 5 (measure and record): the largest gap between events per stream; frontend CPU per token at the agent
concurrency; TTFT split per dialect into render + constraint build, cold vetting + compile, and warm; and echo the
effective `strict: false` in the Responses API's `tools`.

### Step 4 (2026-10-14): D2's measurement

`bench/gate_cost.py` on the RTX 5080 (`evidence/gate_cost_qwen3-8b-q4km_rtx-5080_2026-10-13.json`,
`evidence/gate_cost_qwen3.8-27b-iq3s-mtp_rtx-5080_2026-10-13.json`; commit 8c59010). Same greedy requests (natural
text cut from `judge/prompts/v1.jsonl` / `v1-qwen3_5.jsonl`, 256 tokens in, 256 out, `ignore_eos`) on fresh engines
in interleaved rounds (none / A / A′; four rounds, the first discarded): `none` selects no processor; A and A′
select the toy (`testing.toy_mask`) with a no-op `apply`, opening the constrained requests with a marker that never
occurs (all-allowed masks). Every engine gets the same pinned `state_memory`. Engine-thread CPU time per phase under
blocking-sync CUDA waits. Workloads: single stream (four requests, one at a time) and mixed (one constrained and three
free, at once; two batches). Then grammar.xgrammar's `fill` per constrained step, on the same engine flags over a
JSON-schema request (a person with an address: 4 requests), and its Triton `apply` per call (CUDA-graph replay) are
measured and added.

| | Qwen3-8B Q4_K_M | Qwen3.8-27B UD-IQ3_S + MTP (k 3 at batch 1, 2 above) |
|---|---|---|
| single stream TPOT, none / A / A′ (ms; spread over rounds) | 6.511 / 6.874 / 6.875 (≤ 0.03) | 9.486 / 9.717 / 9.712 (≤ 0.006) |
| single-stream penalty A′, toy (per round) | 1.056 (1.048–1.056) | 1.024 (1.023–1.026) |
| xgrammar fill per constrained step: mean / p50 / max (µs) | 36.6 / 23.3 / 1881 (1 column) | 73.8 / 68.2 / 896 (3.92 columns) |
| Triton apply per call (µs) | 1.43 (1 row) | 1.74 (1 row), 3.14 (4 rows) |
| **single-stream penalty A′ with fill and apply** (rule: split if > 1.10×) | **1.061** | **1.026** |
| mixed, free rows' TPOT none / A / A′ (ms) | 7.333 / 7.281 / 7.696 | 14.239 / 14.233 / 14.466 |
| **free rows' slowdown A / A′** (rule: A if A′ > 1.15×) | 0.993 / **1.050** | 1.000 / **1.016** |
| constrained row vs free rows, while the free rows run, A / A′ | 2.00× / 1.00× | 1.86× / 0.91× |
| host CPU per decode step, none → A′: schedule, fill, launch, commit; whole step (µs) | 32, 0, 279, 28; 371 → 34, 4, 307, 29; 407 | 40, 0, 603, 35; 717 → 43, 8, 650, 49; 790 |
| wall per step, none → A′ (µs) | 6506 → 6871 | 27329 → 27980 (2.83 tokens per verify step) |
| verify rows by k (per round) | — | k 3: 389, k 2: 754 |
| verify steps overlapped, none / A′ | — | 571 / 32 of 571 |

Decision (D2): **A′** (free rows 1.050× / 1.016× < 1.15×), and **the split is deferred to M0.7** (1.061× / 1.026×
≤ 1.10× on both models). `Engine.commit_for_constrained` stays True. Against the predictions: host time per
constrained step is 0.4 ms (8B) and 0.8 ms (27B), not 1–2 ms, so the 8B's penalty (1.061×) is under its predicted
1.15–1.30× and the 27B's (1.026×) under 1.04–1.07×. Under A a constrained row decodes at half the free rows' rate,
as predicted.

Preconditions: under A′ every sequence's tokens and the verify rows by (k, accepted) equal the run without a
processor, in every round. Under A they do not, deterministically: one sequence of the 8B's second mixed batch (the
constrained one) and one free sequence of the 27B's first mixed batch differ, and the 27B's verify rows change (632
vs 571 verify steps). A changes which rows share a step, and the GGUF kernels are not batch-invariant: on the 5080,
the same 8B prompt gives three different greedy continuations alone, in a batch of two and in a batch of four (digests
f70006ec…, c6920275…, ff3a4ca8…; one-off check through `gate_cost`'s helpers). A's timings are therefore compared on
equal lengths (`ignore_eos`) rather than identical tokens; A′'s, which decide, are on identical tokens.

What selecting a processor costs a deployment (§6; `evidence/gate_sizing_qwen3.8-27b-iq3s-mtp_rtx-5080_2026-10-13.json`,
three interleaved rounds, each engine sized in a fresh process): the 27B + MTP at `--max-model-len 32768` has 37,808
KV tokens without a processor and **35,616 with one (−5.8%)**: its state budget is 81 MB smaller (2.784 → 2.703 GB),
and KV is the small remainder after the recurrent state; graphs 27 → 54, capture 2.9 → 5.7 s. The Qwen3-8B Q4_K_M
(`evidence/gate_sizing_qwen3-8b-q4km_rtx-5080_2026-10-13.json`, `--max-seqs 16 --max-model-len 32768`) loses 0.13%
(61,168 → 61,088 KV tokens); graphs 15 → 30, capture 0.6 → 1.0 s. Step 1 measured ≤ 1% on Qwen3-0.6B; the 27B's
5.8% is a shortfall against that expectation. Traced (one engine per process, the sizing's components): the worst
step's peak (505 MiB) and the graph pool's growth (206 MiB) are the same with and without the processor, and so is
what torch allocates; the free memory the sizing reads is 86 MiB lower with it, outside torch's allocator — most
likely the extra graphs' instantiated executables (the profile captures the largest bucket's graphs, twice as many
with masked variants; a 27B graph has thousands of kernel nodes; not confirmed with a driver-level count). Capturing every graph then
takes a further 518 vs 364 MiB outside the pools, which the 10% headroom absorbs (576 MiB stay free). Fewer masked
graphs (e.g. only the widths and buckets constrained steps use) would win it back; not done (a capture-set change
for the panel).

Found on the way: an engine built after another in the same process sizes its pools from less free memory — a
closed engine's reference cycles hold its weights until the collector runs (≈ 2 GB less free for the 27B + MTP's
second engine without a collection), and with the collector run after each close, sizing the 27B in-process still
came up short (a 2.26 GiB budget where one 32k-token request needs 2.43; not traced further). The first `gate_cost` record's `processor_selected` (27B:
39,584 vs 6,432 KV tokens) is an artefact of that, superseded by the sizing record above, where each engine has its
own process. D2's rounds were unaffected (every engine got the same pinned pool,
and the workloads need ≤ 2k tokens).

### Step 5 (2026-10-14): evidence

**Core change: slot objects' own counters** (the architecture reviewer's seam, step-2 deferral): `Engine.stats()` adds
an optional `stats() -> dict[str, int]` of every slot object — sampler, verifier, drafter, draft policy, logits
processor — under `<plugin name>.<key>` (`engine.py` +6 lines, `api/decoding.py` documents it). Toy proof: the toy
processor reports `rows_filled`, and `test_logits_processor.py`'s matrix asserts `testing.toy_mask.rows_filled` equals
the rows it filled (fails without the seam). grammar.xgrammar reports `drafts_refused`, read by
`test_grammar.py`'s speculation cases through the engine's stats. **Deviation:** the counter is
`grammar.xgrammar.drafts_refused`, not D3's `spec_drafts_refused` (namespaced by plugin, as the seam reports every
slot object); the refused-draft share is `grammar.xgrammar.drafts_refused / spec_drafted`. Core: 3868 code lines.

**L0-mask on sm_89 and sm_120** (RTX 4090, RTX 5080, at ec9e90d): `pytest plugins/tests/test_grammar.py -k apply`
3 passed on each; `compute-sanitizer --tool memcheck python -m tests.memcheck_grammar_apply` (152k, 248k and 1000-token
vocabularies, padding rows): 0 errors on each. With step 2's sm_86 run, L0-mask holds on the three classes.

**Conformance (D9), the precompile on an idle host** (`bench/conformance.py precompile`, RTX 4090 node's CPU,
`evidence/conformance_precompile_rtx-4090_2026-10-14.json`): the filtered test split is 2,668 schemas (the eight
subsets minus Github_trivial / ultra; 3 of the 13 unsatisfiable Glaive schemas are in the test split). Each schema
goes through the server's own checks — D9's reject list, then the vetting worker with the plugin's defaults (two
workers, 10 s, 256 MB budget, 64 MB size cap) — with its compile time recorded:

| | 152k (Qwen3-8B) | 248k (Qwen3.8-27B) |
|---|---|---|
| accepted under strict (D9's predicate; predicted 64–67%) | 1634 (61.2%) | 1634 (61.2%) |
| refused by vetting: compile timeouts (10 s) / xgrammar errors | 11 / 6 | 50 / 6 |
| **declared coverage** (compiled / split) | **1617 (60.6%)** | **1578 (59.1%)** |
| compile time of the compiled: p50 / p90 / p99 / max (ms) | 18 / 69 / 7325 / 8615 | 35 / 66 / 6158 / 9470 |
| compiled size p50 / p99 / max (MB); peak growth max (MB) | 0.2 / 17.6 / 27.0; 19.4 | 0.2 / 13.7 / 21.5; 10.9 |

Per subset (152k): Github_easy 406/577, Github_medium 305/586, Github_hard 121/368, Glaiveai2K 451/510,
JsonSchemaStore 34/148, Kubernetes 184/320, Snowplow 98/121, WashingtonPost 18/38. Most rejections: `oneOf` 243,
typed keywords without `type` 131 (+97 for `format`), embedded schema resources 79, loose patterns 67, `date-time` 62.
**Shortfall:** strict acceptance is 61.2%, under the predicted 64–67% (the reject list grew in step 2: `time`,
`email`, `ipv6`, the ECMA pattern rules, embedded resources), and the 248k tokenizer's timeouts cost 1.5 points more.

**The 27B + MTP, 500 sampled schemas** (RTX 4090, `holon serve` with grammar.xgrammar and MTP as M0.4, `--max-seqs 4
--max-model-len 32768`; `evidence/conformance_qwen3.8-27b-iq3s-mtp-500_rtx-4090_2026-10-14.json`; core and plugins
as 8c59010). Passes: cold (the server's first sight of each schema, `max_tokens` 1, concurrency 2), warm (the same
again), greedy (`max_tokens` 2048, concurrency 4), free (no `response_format`):
- accepted 301, rejected 192 (400: the reject list) and 7 (503: compile timeouts, still refused after two retries);
- **greedy: valid 293 of 293 judged (100%), 0 request-time errors**, 8 truncated;
- **refused-draft share 1.11%** (436 of 39,328 drafts; predicted ≤ 15%); every verify step graphed, none overlapped
  (A′);
- **truncated share 2.66% vs 0.33% unconstrained on the same 301 schemas: +2.33 points — over the 2-point gate
  (shortfall).** The 8 truncations: 2 whitespace runaways right after `{` (sp_218, sp_70 — the frontend strips a
  text run's trailing whitespace, so they show as 2048 tokens with one character; the harness now counts them from
  that), 3 number runaways (`"width": 0.0000…` for 2048 tokens: a JSON number has no length bound), 1 repetition
  inside a string, 2 long but legitimate objects (also long unconstrained). Without the whitespace runaways the gap
  is 1.66 points. S2-1's condition for a whitespace cap ("returns only if they appear") is met; the cap's cost,
  measured (`evidence/whitespace_cap_cost_rtx-a5000_2026-10-14.json`): at 16, compiled grammars 3.4 / 6.7 MB (152k /
  248k) instead of 0.2, mask fill p50 41–63 µs instead of 4–16, compile +3–9 ms; at 32, 6.8 / 13.2 MB. Not
  applied: a decision for the panel (it trades every constrained step's fill time and the cache's capacity for two
  runaways in 301), and number runaways need a bound of their own.
- TTFT, cold / warm: p50 240 / 198 ms, p99 13,973 / 1,960 ms (the cold p99 is vetting and compiling: up to the 10 s
  timeout); the largest gap between a greedy stream's events: p50 126 ms, p99 3.5 s, max 34 s — while other requests'
  long prompts (up to 30k tokens) prefill in 2048-token chunks, a running stream gets no tokens (FCFS, no mixed
  prefill/decode steps); keepalives are SSE comments, not counted;
- frontend CPU per token at concurrency 4: 109 µs (greedy, chat with `response_format`), 74 µs (free).

**The constrained bounded check (§5) on the 27B + MTP** (RTX A5000; `python -m holon_judge constrained`, new:
`judge/holon_judge/constrained.py`; `evidence/constrained_bounded_qwen3.8-27b-iq3s-mtp_rtx-a5000_2026-10-14.json`).
64 JSON-schema prompts (strict-accepted Glaiveai2K and Github_easy schemas under 4000 characters, alternating, chat
rendered with thinking off), greedy, no `ignore_eos`, `max_tokens` 512, on a plain and a speculating engine (both
with grammar.xgrammar), each free and constrained, on the production path and again with logprobs; a reference (the
plain engine's config eager, depth 0, no prefix cache, one sequence at a time) re-scores every output teacher-forced,
masked by a fresh matcher replay. Results:
- constrained: 59 of 64 identical (58 of 64 in the logprob runs), **0 errors**, finish reasons equal wherever the
  tokens are; the 5–6 first divergences fall at masked reference gaps of **0.006, 0.042, 0.093, 0.100, 0.111, 0.124
  nats**, and at each both engines' own masked logprobs of the two tokens are within 0.15 nats of each other (a
  near-tie on both sides, recorded per divergence);
- masked agreement with the reference: plain 99.712% ± 0.118%, spec 99.692% ± 0.086%; at confident positions 100%
  and 100% (3234 / 3313 positions);
- KL(plain ‖ spec) over the positions before divergence: constrained p50 6.4e-7, p99 2.8e-3; free p50 3.6e-7, p99
  2.6e-3 (budget 2×): within;
- refused drafts 207 of 12,161 (1.7%).
- **Verdict PASS, borderline:** the threshold is `CONFIDENT_GAP` (0.1) raised to the largest unconstrained
  first-divergence gap — the free pair's (≤ 0.106) and 64 natural-text calibration prompts continued 128 tokens
  unconstrained (§5: M0.4's spec-vs-plain runs) — and the calibration's largest gaps (7.93 and 1.13 nats) are both
  in post-`<|im_end|>` text (`ignore_eos`): there the decode path (plain) and the prefill path (the reference's
  teacher forcing, the verifier's rows) disagree, every plain configuration generating the same tokens (prefix
  cache on / off, depth 0 / 1, graphs / eager, processor selected or not: checked one-off). Without those two, the
  unconstrained gaps reach 0.229 and the constrained ones (≤ 0.124) stay under them; against the uncalibrated 0.1,
  two constrained divergences (0.111, 0.124) would fail. Flagged for the panel (the threshold rule's calibration
  should stop at the stop token).
- The plants: `mask-column-shift` and `no-rollback-after-probe` FAIL it (63 of 64 constrained speculating requests
  end with `error`, first divergences at masked gaps up to 15 nats, KL p99 10.4 / 0.16), as §5 requires
  (`evidence/constrained_bounded_plant-*_rtx-a5000_2026-10-14.json`).
- Found on the way: the first run (a reference that scored all sequences in one batch) failed with gaps 0.14–0.39 —
  the GGUF kernels are not batch-invariant, so the reference's rows depended on which other sequences a run asked it
  to score (the same divergence scored 0.045 in one run and 0.175 in another); the reference now scores one
  sequence at a time. And the KL needed `p = 0` terms dropped (`compare.kl_topk`: a masked distribution's −∞
  logprobs gave NaN; the bucketed-KL test gained a masked case that fails without it).

**Round trips per dialect** (`bench/round_trips.py`, new; RTX 5080; `evidence/round_trips_qwen3-8b-q4km_rtx-5080_…`,
`…_qwen3.8-27b-iq3s-mtp_rtx-5080_…`, at 425e167): OpenAI chat, Anthropic Messages and OpenAI Responses × tool choice
named / required / auto (a strict `get_weather` beside a second tool), the call, its result, the answer; each
request sent cold (streamed), then streamed and whole on the cached prefix. On **both models all 9 calls are
`get_weather` with schema-valid arguments, all 9 answers use the result, and every streamed answer reassembles to the
whole one** (the parse does not depend on how the text arrives). Cold requests generated the same as warm ones except
two 27B answers (Anthropic and Responses after a named call: the first request without a cached prefix chose other
words — not a parse difference). TTFT, cold (a new constraint: vetting and compiling) / warm: 8B 82–169 / 11–16 ms,
27B 146–414 / 52–80 ms; the frontend's render and constraint build (`pipeline.Chat`, timed in-process): 0.39 ms (8B)
and 0.62 ms (27B) per request, the same in each dialect. The 27B's counters: 0 refused drafts of 3015 (`auto` with a
strict tool is constrained; its JSON was drafted right).

**The Claude Code session** (`evidence/claude_code_session_rtx-5080_2026-10-14.json`): `holon serve` the 27B
(UD-IQ3_S + MTP + grammar.xgrammar, §11's flags, `--host 0.0.0.0`) on an RTX 5080 node of the consumer cluster;
Claude Code 2.1.296 (`claude -p … --allowedTools Read,Write,Edit,Bash,Glob,Grep --output-format json`) on the
development host, reaching it through `bench/capture_proxy.py` (new: records each request's shape, never its text,
header values redacted but the API version and beta flags). Task, in a scratch directory: read `inventory.csv`,
compute the total value with a bash command, write `total.txt`, append a line to `report.md` with the Edit tool.
**Completed:** 6 turns in 21 s, `total.txt` = `6.00`, `report.md` ends with `Total value: 6.00`, result `done`.
Four `/v1/messages?beta=true` requests (streamed, `max_tokens` 32000, `thinking: {type: adaptive, display:
omitted}`, `output_config: {effort: high}`, `context_management`, `metadata.user_id` a JSON string), 22 tools each
with `$schema` and `additionalProperties: false` and no `cache_control`; three system blocks (the billing header,
then two with `cache_control`); parallel `tool_use` in one turn; no `count_tokens` and no side calls to a small model
(with nonessential traffic off); **`billing_headers_dropped` 4**, prefix tokens saved 52,544. R §5.4's unverified
items are settled by this capture.
- **Found and fixed (frontends.http):** the first attempt failed at once with a 400 — Claude Code sends `role:
  "system"` messages inside `messages` (its per-turn reminders, beta `mid-conversation-system`), which the step-3
  router refused. Now a system message before the first turn joins the system prompt and a later one becomes a user
  turn (the Qwen templates take one leading system message; folding later ones into it would change every turn's
  prefix — no prefix-cache hits, vLLM's documented trade-off); `test_anthropic_request_as_a_conversation` covers both
  (fails before). Recorded as a deviation from step 3's Anthropic subset.

**The 8B, the full filtered split** (RTX 5080, `holon serve` with grammar.xgrammar, `--max-seqs 16 --max-model-len
32768`; `evidence/conformance_qwen3-8b-q4km-full_rtx-5080_2026-10-14.json`; core and plugins as 8c59010). Passes:
cold and warm (`max_tokens` 1, concurrency 2), greedy, free, and sampled at T = 1 with seeds 0 and 1 (`max_tokens`
2048, concurrency 16):
- cold: accepted 1625 of 2668 (60.9%), rejected 1040 (400: the reject list) and 3 (503: compile timeouts); at
  concurrency 16, 9–10 more are 503 "compiles are queued" (the helper backlog bound: the compiled-grammar cache does
  not hold the whole split, so warm schemas recompile) — a client retrying later gets them;
- **greedy: 1582 valid of 1583 judged, 1 invalid, 0 request-time errors.** The invalid one is a **reject-list hole,
  found and fixed:** JsonSchemaStore's Ansible `vars` schema has `patternProperties` keyed by a pattern with a
  negative lookahead, which xgrammar ignores (it warns), so a key the lookahead forbids (`hosts`) was admitted. Strict
  requests now refuse lookarounds and backreferences in patterns (b77d45c; the oracle's hole list gained three cases
  that fail before; of the split, only `vars` is newly refused). Under the fixed list: **1582 of 1582 (100%)**;
- **sampled (T = 1): 1587 of 1587 (seed 0) and 1584 of 1584 (seed 1) valid, 0 errors**;
- truncated: greedy 1.98% vs 3.72% unconstrained on the same 1615 schemas (−1.73 points; sampled −1.98 / −1.86):
  **within the gate**. Of the 33 greedy truncations: 2 whitespace runaways (o78734, o82694), about 20 digit or
  base64-like runaways inside unbounded numbers and strings (`0.0000…`, `1234567890…`), and long objects that the
  unconstrained run also truncates;
- TTFT cold / warm p50 119 / 76 ms, p99 9.2 / 4.7 s; largest gap between a greedy stream's events p50 90 ms, p99 3.7
  s, max 36 s (a 30k-token prompt prefilling: FCFS, no mixed steps); frontend CPU 101 µs per token (greedy, 16
  streams), 66 µs (free).

**Frontend measurements at the agent concurrency** (`evidence/bench_m05-8b-agent-frontend_rtx-5080_2026-10-13.json`:
`bench/run.py` on the agent trace, 4 streams, two rounds; `holon_http_chat`, new — the chat endpoint with six tools
offered, so the reasoning and hermes parsers read every stream — beside `holon_http`, completions): the frontend
process's CPU is **231 µs per generated token through chat, 209 µs through completions** (TPOT p50 7.0 / 7.4 ms; the
frontend runs in its own process); the largest gap between a chat stream's events: p50 48 ms, p99 1.04 s, max 1.35
s. The TTFT split per dialect: render and constraint build 0.4 ms (8B) / 0.6 ms (27B), the same in every dialect;
cold constraint (vetting and compiling) 82–414 ms; warm 11–80 ms (round trips above); over the conformance split, cold
vs warm p50 119 vs 76 ms (8B) and 240 vs 198 ms (27B).

**Step-7 items done here** (cheap, both core): the `Parser` protocol and `FrontendBundle.reasoning_parser()` /
`tool_parser()` are deleted (dead API: the frontend owns the formats; the three Qwen bundles and the test fake lose
their stubs; design §7.8 updated); `FrontendPlugin.serve(…, options)` gets the deployment's
`EngineConfig.plugin_options`, and `frontends.http` reads `http.drop_billing_header` (default true; `false`, `0`,
`no`, `off` keep the block). No test covers the option's parsing (a one-line read; `build_app`'s flag is tested).

New tooling (bench and judge, no core): `bench/gate_cost.py` (D2), `bench/conformance.py` (D9: precompile, passes,
report), `bench/round_trips.py`, `bench/capture_proxy.py`, `bench/backends/holon_http_chat.py` (with the HTTP
backends' per-stream event gaps and the frontend process's CPU time), `holon_judge constrained`. Full suite at
425e167 on the A5000: 1021 passed, 55 skipped; `scripts/ci/run_all.sh`: clean (core 3868 code lines).

Deviations (step 5):
- `spec_drafts_refused` is `grammar.xgrammar.drafts_refused` (the general seam namespaces by plugin).
- The constrained bounded check calibrates with natural-text runs as well as the free pair, and its reference scores
  one sequence at a time; it ran on 64 Glaiveai2K / Github_easy schemas under 4000 characters (dev-host rule: no
  heavy JSONSchemaBench compiles there).
- The conformance runs' cold pass is the server-side precompile (each schema's first request, `max_tokens` 1,
  concurrency 2) and the precompile pass proper ran offline through the same checks and workers; the runs were made
  at fbced8f (core and plugins as 8c59010), before the lookaround fix, whose effect (one schema) is applied in the
  text above, not re-run.
- The truncation counts: whitespace runaways are counted from tokens without text, since the frontend strips a text
  run's trailing whitespace (the first harness counted trailing whitespace and missed them).
- The Claude Code session ran with the server on a consumer-cluster RTX 5080 node and the client on the development
  host, over the network.

For the panel (and the owner):
1. **Truncation gate failed on the 27B + MTP** (+2.33 points; the 8B passes at −1.73). S2-1's condition for a
   whitespace cap is met (2 runaways on the 27B's 301, 2 on the 8B's 1616); a cap of 16 costs 17–30× the compiled size
   and 3–10× the mask fill (measured above). Number and string runaways (no length bound in JSON) are the larger
   share and no cap helps them. Decide: the cap (16 or 32), a bound for numbers, or accept the shortfall.
2. **The constrained bounded check passed only on its calibrated threshold**: its constrained divergences (≤ 0.124
   nats) are near-ties next to the unconstrained ones (≤ 0.229), but two exceed the uncalibrated 0.1, and the
   calibration's maximum (7.93) comes from post-EOS text where decode and prefill disagree. The calibration should
   stop at the stop token (and the decode-vs-prefill gap after `<|im_end|>` belongs to M0.4's numerics).
3. **The 27B's KV cost of selecting a processor is 5.8%**, from masked graphs' executables (not the pool); fewer
   masked graphs would recover it.
4. **Strict acceptance 61.2% vs the predicted 64–67%**; the reject list's growth since rev 5 is why; with the
   lookaround fix the 8B's greedy conformance is 100% (1 invalid before it).
5. Engines built one after another in one process size their pools from less free memory (≈ 2 GB less for the
   27B + MTP's second engine): judges and benches that build several engines per process inherit it (they pin or
   tolerate it today); `gate_cost` now sizes in fresh processes. Worth tracing in core (what `Engine.close` leaves
   behind), outside this step's scope.
