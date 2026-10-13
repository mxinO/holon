# M0.5 — structured output and tool calls: design and implementation plan

Status: **revision 5, reviewed clean** (2026-10-13). Revs 1–5 were reviewed by a panel of three; every finding is
resolved in §9. **Step 1 built** (2026-10-13): the core seams of §3 with the toy processor; its record, deviations
and §6's measurements are in §11.

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
  - JSON: `any_whitespace = True` with a whitespace run cap of 64 (the cap counts the newline and the indentation;
    64 admits nesting depth 8 at 4-space indentation, so pretty-printed JSON and MTP's drafts of it stay allowed).
    xgrammar applies the cap only to typed schema rules: `json_object` is compiled from xgrammar's builtin JSON EBNF
    with only its whitespace rule replaced by the capped one (fixtures: every JSON-Schema-Test-Suite instance dumped
    compact and at indents 2 and 4 to depth 8 is accepted, malformed JSON — escapes, `\uXXXX` surrogates, numbers,
    non-ASCII — refused, and it agrees with the builtin except for whitespace past the cap); untyped subschemas
    (`{}`, `{"type": "object"}` without properties) stay uncapped — stated, and covered by `max_tokens` and the
    truncation gate.
  - `strict_mode` per D9. xgrammar is a pinned pip dependency (Apache-2.0); the pin is chosen at step 2 (0.2.3 now;
    vLLM 0.2.8, SGLang 0.2.7) and the reject-list oracle and the tag fixtures are re-run at it.
- **D6. `grammar.constraint@1`, a plain-dict resource.** The frontend package builds constraints as plain dicts
  (`json_schema` with `strict`, `json_object`, `regex`, `ebnf`, and the tag forms `tag(begin, content, end)`,
  `triggered_tags(triggers, tags, at_least_one, stop_after_first)`, `any_text(excludes)`, `sequence`, `or`, and the
  parameter-value styles `json` and `qwen_xml`); the grammar plugin declares the resource in its manifest and
  translates it to its backend; bundles and frontends never import xgrammar. Tool-call constraints are built by the
  frontend from the bundle's `tool_format` (hermes JSON for Qwen3, `<tool_call><function=…><parameter=…>` for
  Qwen3.5 / 3.8, R §4.1), not by xgrammar's builtin `qwen_3` / `qwen_3_5` tags, which refuse a `<tool_call>` that
  ends reasoning without `</think>` and do not limit `parallel_tool_calls: false` to one call (reviewer finding,
  measured). Reasoning is free text until `</think>` or a `<tool_call>` that ends it.
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
- Host RAM when the grammar processor is selected: two vetting workers at ≈ 1.1 GB resident each (≈ 2.2 GB, up to
  ≈ 4.4 GB at the watchdog bound) plus the 512 MB compiled-grammar cache — measured at step 2; if it matters on
  consumer hosts, one worker (more queueing) or a worker that never imports torch.

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

## 11. Step 1 record (2026-10-13)

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

