# M0.5 — structured output and tool calls: design and implementation plan

Status: **revision 1, for review** (2026-10-13).

Scope: plan §8, amended:
- the core gets the logits-processor seams the plan's ladder already lists for M0.5 (plan §2: "forward/sample split,
  host_input, Rewindable grammar"); "core diff target zero" is replaced by "only the general seams of §3, each with a
  toy proof" (D1);
- `tool_choice: "auto"` stays unconstrained by default, as in vLLM and SGLang (D8);
- "100% schema conformance" is defined against a published reject list (D9).

Builds on M0.4 Gate A (verify rows, `std.spec.greedy` exact-match and `std.spec.rejection`, verify graphs, overlap
depth 1). Facts about xgrammar, vLLM, SGLang, llama.cpp and the Qwen templates are cited in the local research note
`docs/brainstorm/m05-grammar-research.md` ("R §n" below); CPU measurements there were taken with xgrammar 0.2.3.

## 1. Goal and acceptance

Constrained decoding (JSON schema, JSON object, regex, EBNF, tool-call structural tags) and tool calls through the
OpenAI chat, OpenAI Responses and Anthropic Messages APIs, on every model and with speculation. Acceptance:

1. **Conformance.** On a pinned JSONSchemaBench test sample (D9): declared coverage reported with rejections by
   reason; truncated outputs (`finish_reason = length`) counted separately; **valid / (accepted − truncated) = 100%**
   under jsonschema Draft 2020-12 with format checking. The model-free matcher oracle (JSON-Schema-Test-Suite) shows
   no accepted-invalid instance for any schema the server accepts.
2. **Tool calls.**
   - OpenAI chat: `tools`, `tool_choice` (`auto`, `none`, `required`, named), `parallel_tool_calls`,
     `response_format` (`json_schema`, `json_object`), streaming `tool_calls` deltas; round trips on Qwen3-8B
     (hermes JSON) and the Qwen3.8-27B GGUF (XML `<function=…>`).
   - Anthropic `/v1/messages` (text, `tool_use`, `tool_result`, `thinking` blocks, `tool_choice`, streaming SSE)
     and `/v1/messages/count_tokens`: a non-interactive Claude Code session (`claude -p`, `ANTHROPIC_BASE_URL` at
     Holon serving the 27B on the 5080) completes a scripted multi-tool task.
   - OpenAI Responses: the minimal subset of R §5.2 (text and function-call items, streaming), one round trip.
   - Parsers: reasoning (`<think>`) and tool parsers per bundle, streaming, checked against the non-streaming parse
     on every recorded output.
3. **Grammar with speculation.** Greedy: grammar + MTP emits exactly what grammar without speculation emits (CPU
   tiny models exact; GPU bounded on the 27B + MTP). Sampled: L0-dist of `std.spec.rejection` under a column mask
   (the emitted law is the masked target's).
4. **Graphs and overlap.** Steps with constrained rows replay captured graphs (the mask applied inside); a
   constrained request never samples from a mask built without its last token. Measured: the share of graphed
   constrained steps; the decode-rate cost of constrained requests (D2's rule decides the forward/sample split).
5. **Isolation.** A failing grammar (compile error, unsupported schema, a token the matcher refuses, a crash
   restart) ends only its request (`finish_reason = "error"` or a 400), never the engine; a restarted constrained
   request resumes its matcher.
6. **Invariants.** With no constrained request, nothing changes: same graphs, every lock's snapshot trace
   byte-identical. With constraints, snapshot traces are run-identical.
7. **Core:** only the seams of §3, each proven by a toy plugin in `core/tests`; core ≤ 6000 code lines (3643 now).

Non-goals: grammar-aware drafting (the drafter is not masked; R §2.3, neither vLLM nor SGLang does it); device-side
grammar automata; JSON-schema features xgrammar does not enforce (rejected, D9); multimodal content blocks.

## 2. Starting point (R §7)

- **Frontend:** OpenAI chat/completions with streaming; `tools` reach the chat template, but no tool parsing;
  `tool_choice` other than `auto` and `response_format` other than text are rejected (`schemas.py:192-193`).
  `FrontendBundle.reasoning_parser()` / `tool_parser()` exist and return `None` in every bundle.
- **Engine:** `LogitsProcessor` (`api/decoding.py:48-54`: `argmax_invariant`, `host_input(plan)`, `apply`) and its
  `ordered_multi` slot exist, but nothing loads or calls a processor.
- **Overlap:** `Engine.step` launches step N whole (forward, sample, store, rewind, drafter in one graph) before it
  commits N − 1 (`engine.py:303-308`); design §7.3's table claims grammar `host_input` works at depth 1 through a
  forward/sample split that was never built (R §3.1). Per-Seq `max_depth` is declared in manifests, never read.
- **Speculation:** only `n_draft` returns to the host; draft tokens stay in `SeqTable.draft_tokens`.
- **Errors:** an exception inside `Engine.step()` ends the engine (`ipc.py:90-97`); only `add()` errors are
  per request. A supervisor restart resubmits `prompt + streamed tokens` with no marker of where output begins.

## 3. Core seams (today → change), each general

1. **Logits processors with a request lifecycle.**
   - Today: `host_input(plan)`, `apply(logits, host_input, ctx)`, unused.
   - Change (`api/decoding.py`): the `logits_processor` slot's plugins build a processor (`build(ctx)`, like
     `DraftPolicy`) with
     - `admit(seq) -> Admission | None`: per request at `Engine.add`, after the tokenizer; reads its own
       `SamplingParams.ext` keys; `None` = this processor does not touch the request; an `Admission` says whether
       its masks need the request's latest token on the host (`host_tokens`). Raising `RequestError` rejects the
       request (a per-request error delta, as `add()` errors are today).
     - `commit(seq, emitted: list[int])`: after each commit, with the request's newly emitted tokens (accepted
       drafts and the bonus token, or the decode token; output replayed after a restart included, §3.5). Raising
       `RequestError` finishes the request with `finish_reason = "error"` (§3.4).
     - `release(seq)`: at finish or abort.
     - `masks(rows: list[MaskRow]) -> None`: at launch, fills the static mask rows of the step's constrained logits
       rows (`MaskRow`: the request, its draft tokens for verify rows, the destination row range), on the host,
       into a pinned buffer the core uploads with the step's inputs.
     - `variant(slots) -> Hashable`: whether the step has rows this processor masks (part of the graph key).
     - `apply(logits, ctx)`: device, graph-capturable, called by the core between the head and the sampler or
       verifier; reads the static mask buffer the core owns.
   - Several processors chain in manifest order; the mask buffer is one per processor that declares it
     (`mask_width(vocab)`), allocated once, ledger-counted.
   - Toy proof: a toy processor that forbids odd tokens after a marker token, on the tiny Qwen3 (CPU), across
     chunked prefill, prefix hits, preemption, depth 0 / 1, `n > 1`, verify rows with the oracle drafter
     (right / wrong / mixed drafts), sampled and greedy: every emitted token obeys the rule, and the output equals
     a reference run that masks the full-vocabulary logits of a plain engine.
2. **Host-token gate (the depth rule, D2).**
   - Today: depth 1 for every decode row; `host_lens_only` gates rows for host-length kernels (`sched.py:111`).
   - Change: a request admitted with `host_tokens` is not scheduled while it has `pending` or `inexact` rows, so its
     masks are built from committed tokens; `step()` already commits first when nothing else can run
     (`engine.py:296-298`). Others keep depth 1 beside it. This is the per-Seq `max_depth` of design §7.3 made real.
   - Toy proof: the toy processor with `host_tokens`, a mix of constrained and free requests at depth 1: the
     constrained ones never take a step while a token is pending (asserted inside `masks`), the free ones overlap.
3. **Draft tokens to the host.**
   - Today: `Launched.drafts` carries `n_draft`.
   - Change: verify-capable steps also copy `SeqTable.draft_tokens[slots, :k_max]` into the pinned results
     (≤ 16 × 4 int64 per step); `MaskRow` gets the drafts the next verify step will check.
   - Toy proof: the toy processor's probe sees exactly the oracle drafter's drafts.
4. **Per-request errors from plugins.**
   - Today: none at step time.
   - Change: `RequestError` (in `holon.api`) raised by a processor's `commit` or `masks` finishes that request with
     `finish_reason = "error"` and a message delta; the engine keeps stepping. Other exceptions still end the engine
     (they are bugs).
   - Toy proof: a toy processor that raises for one request mid-generation; the others finish unchanged.
5. **Output prefix on resubmission.**
   - Today: the supervisor resubmits `prompt + streamed` (`supervisor.py:80-94`).
   - Change: `Seq.replayed` (core-set, from a new `SamplingParams.resume_output: int` the supervisor fills) is the
     number of trailing prompt tokens that are output; processors receive them through `commit(seq, …)` right
     after `admit`, and stop / max-token accounting counts them.
   - Toy proof: a restart in the middle of a toy-constrained generation resumes with the same tokens as an
     uninterrupted greedy run.
6. **Masks inside graphs.**
   - Change (`Executor._decode`): `logits = p.apply(logits, ctx)` for each processor whose variant is active;
     the graph key gains every processor's `variant(slots)`; the startup capture set gains the constrained variant
     for each (bucket, width) only when a processor is installed; the memory profile includes `apply`.
   - Logprobs (steps that report them are eager): computed from the raw logits, as `api/frontend.py:26-28` states;
     the eager path keeps the unmasked logits for them.
   - Toy proof: graphed vs eager on the same padded plans, constrained rows included, give the same tokens (the
     M0.4 eager-runner test extended with the toy processor; GPU, Qwen3-0.6B).

Line budget: lifecycle and calls 70, gate 15, drafts 15, errors 30, resume 25, graphs and keys 30; ≈ +185.

## 4. Decisions

- **D1. The core seams of §3 instead of "core diff zero".** Grammar needs the request lifecycle, a depth rule,
  drafts on the host and per-request errors; all are general (no grammar or format name in core), each with a toy.
  The plan's own ladder (plan §2) lists these seams for M0.5.
- **D2. Depth 0 per constrained request first; the forward/sample split only if measured necessary.**
  - Why: a host mask for step N needs step N − 1's token; today the whole step is launched before N − 1 commits.
    The per-request gate is small and correct with speculation at once (the drafts are on the host before the
    launch). The split (vLLM, SGLang; design §7.3) keeps depth 1 for constrained rows but needs two graphs per
    constrained step, a static logits buffer and an engine reorder (R §3.3, §8.1).
  - Rule fixed now: measure a constrained single stream (JSON schema, greedy) on the 27B UD-IQ3_S with MTP on the
    RTX 5080 against the same request unconstrained. If constrained TPOT is more than **1.10×** unconstrained,
    build the split in M0.5 (step 6); otherwise defer it to M0.7 with the number recorded.
  - Prediction: per constrained step the GPU idles for commit + schedule + mask + launch ≈ 1.0–2.0 ms against a
    10.6 ms MTP step: **1.08–1.18×** (it may trigger the split). Mixed batches: constrained rows run every other
    step beside free ones (their decode rate halves at depth 1 traffic; predicted ≤ 0.6× of free rows' rate).
- **D3. Speculation: host probe per verify column; verifiers unchanged.**
  - Column j of a constrained verify row is masked with the matcher's state after the committed tokens plus drafts
    d0…d(j−1): `accept` d0, fill, … stop at the first draft the matcher refuses (all-allowed after it, those
    columns are never accepted), then `rollback`. The matcher itself advances only at `commit`.
  - Exact-match and rejection verifiers stay as they are: a refused draft has p = 0 under its column's mask, so
    acceptance stops there; rejection stays lossless with respect to the masked target (R §7.3).
  - Drafters are not masked; grammar-refused drafts are counted (`spec_drafts_refused`).
- **D4. Static mask buffer and our own apply.** One int32 bitmask buffer `[max_seqs·(1 + k_max), ⌈V/32⌉]`
  (≈ 0.5 MiB for the 27B at max_seqs 4, k 3), reset to all-allowed; the host writes only constrained rows. `apply`
  unpacks and `masked_fill_`s in torch (capturable, no JIT; xgrammar's CUDA backend compiles with nvcc at import,
  R §8.3). Steps without constrained rows keep today's graphs (the variant is off).
- **D5. `grammar/xgrammar`, a plugin.** Compiles the request's constraint (`ext["grammar"]`: `json_schema`,
  `json_object`, `regex`, `ebnf` or `structural_tag`) once per request at `admit`, with a compile cache keyed by
  the constraint and the tokenizer; one `GrammarMatcher` per sequence; stop tokens = the request's EOS set (xgrammar
  allows one EOS by default, R §1.1); a terminated matcher masks to the stop tokens. Compile is synchronous at
  `add` (1–62 ms cold, 11–17 µs cached, R §1.2); a thread pool with an admission gate comes only if traces show
  stalls. xgrammar is a pip dependency (Apache-2.0) of the plugin, pinned.
- **D6. Constraint construction belongs to the model bundle; parsing to the frontend.** The bundle maps (tools,
  tool_choice, parallel_tool_calls, response_format, thinking) to a constraint in xgrammar's structural-tag JSON
  (Qwen3: hermes JSON in `<tool_call>`; Qwen3.5 / 3.8: `<tool_call><function=…><parameter=…>`, R §4.1), with the
  reasoning span free until `</think>` (or a `<tool_call>` that ends it, R §4.2). Reasoning and tool parsers live
  in the bundle (`Parser.feed(text)`, `channel()`), streaming, with schema-typed parameter values. Core sees only
  `ext` keys it does not read.
- **D7. One chat pipeline, three dialects.** A shared pipeline (normalize → render → generate → reasoning/tool
  parse → typed events) under three thin frontends: OpenAI chat (existing, extended), `frontends/anthropic`
  (`/v1/messages`, `/v1/messages/count_tokens`), `frontends/openai_responses`. Tool-call arguments given as JSON
  strings are parsed to objects before rendering (the Qwen3.8 GGUF template raises on strings, R §4.1). A system
  block that only carries a per-request billing header (Claude Code, R §5.4) is dropped from the prompt so prefix
  caching works; it is never sent to the model either way.
- **D8. `tool_choice: "auto"` unconstrained by default.** vLLM and SGLang do the same (R §4.3); constraining it
  would make every agent request a depth-0 request (D2). `required`, named and `parallel_tool_calls: false` are
  constrained; `--plugin-options grammar.tools=strict` constrains `auto` too.
- **D9. Conformance against a reject list.** Schemas using a feature xgrammar does not enforce are rejected with a
  400 naming the feature (R §6.2: `allOf` with ≥ 2 branches, `oneOf`, `not`, `if`/`then`/`else`, `multipleOf`,
  `uniqueItems`, `dependent*`, `contains`, siblings of `$ref`/`enum`/`const`/combinators, typed keywords without
  `type`, required keys not in `properties`, remote refs, unknown formats, length with `pattern`/`format`). The list
  is checked by the JSON-Schema-Test-Suite matcher oracle in CI: every keyword it shows under-constrained must be on
  it. The conformance run: JSONSchemaBench test splits except GitHub-trivial and -ultra, the 13 unsatisfiable
  Glaive schemas dropped, a pinned sample of 500, greedy, `max_tokens` 2048, `any_whitespace=False`, Qwen3-8B
  Q4_K_M on the 5080 and the 27B on the 4090.
- **D10. Stop strings and constraints.** A frontend stop string inside constrained output ends the request as
  `stop` (the user asked); conformance runs send none.

## 5. Numerics, judge and predictions (fixed before measuring)

- **L0-mask:** `apply` vs a reference unpack (bit order, vocab padding past V, all-allowed rows) on sm_86, sm_89,
  sm_120a; memcheck clean.
- **L0-dist (masked):** `std.sampling` and `std.spec.rejection` under a column mask: 10⁵ draws at fixed seeds, χ²
  at p = 1e-3 (M0.4 §5's method) — the first-token law and the joint law of three tokens on the toy Markov target
  equal the masked, renormalized target; exact zero counts outside the mask.
- **Spec identity:** greedy + MTP with a JSON-schema grammar emits exactly what greedy without speculation emits
  (CPU tiny qwen3_5 exact; GPU: the 27B UD-IQ3_S on the A5000, 64 schema prompts, bounded with the M0.4 judge).
- **Conformance:** D9's run; accepted-but-invalid must be 0 (100%).
- **Parsers:** every recorded output's streaming parse equals its non-streaming parse; hermes and XML parsers
  against the vLLM/SGLang reference behaviour on a fixture set (R §4.2), schema-typed values.
- **Planted bugs** (`planted.py`), each must fail the listed check:

| Plant | Must fail |
|---|---|
| `mask_column_shift` (column j masked with the state after j + 1 drafts) | spec identity |
| `no_rollback_after_probe` | spec identity; conformance |
| `stale_mask_depth1` (constrained rows allowed at depth 1, mask from the previous token) | conformance |
| `mask_bit_order` (bits reversed within a word) | L0-mask; conformance |
| `eos_not_allowed` (terminated matcher masks everything) | the toy isolation test (error, not a hang) |

- **Predictions:**
  - Conformance 100% of accepted; declared coverage ≈ 60% of the sample (R §6.2: 62% carry no ignored keyword).
  - D2's constrained single stream: 1.08–1.18× TPOT on the 5080 (27B + MTP).
  - MTP acceptance inside JSON strings drops ≤ 15% relative to unconstrained (drafts of free text in strings
    stay valid; structural tokens are near-deterministic).
  - Host cost per constrained row: 2–75 µs per mask fill (R §1.2), (1 + k) fills for verify rows.

## 6. Memory fit

The mask buffer is ≈ 0.5 MiB at max_seqs 4 (27B, k 3, V 248 320); pinned host mirror the same; the eager logprob
path's raw-logits copy exists only on logprob steps (eager, measured by the profile). The 5080's fit (39.6K KV
tokens with MTP, M0.4 §10) moves by < 1 MiB.

## 7. Steps

| # | Work | Checks | Days |
|---|---|---|---|
| 1 | Core §3.1–3.6 with the toy processor; design §7.3 / §7.6 corrected | core tests; spec matrix with the toy; graphs vs eager (GPU); full suite | 3 |
| 2 | `grammar/xgrammar`: compile cache, matchers, probe, EOS set, reject list (D9) with the matcher oracle in CI; L0-mask, masked L0-dist, plants | L0 ×3 classes; oracle; plants | 2.5 |
| 3 | Bundles: Qwen3 / Qwen3.5 constraints (structural tags), reasoning and tool parsers (streaming) | parser fixtures; streaming = non-streaming | 2.5 |
| 4 | Frontends: OpenAI chat (tools, tool_choice, response_format, streaming tool_calls), Anthropic, Responses; the shared pipeline | API tests; Claude Code session; round trips | 3 |
| 5 | Evidence: conformance (D9), spec identity on the 27B, D2's measurement, MTP acceptance in JSON | records | 2 |
| 6 | (only if D2's rule fires) forward/sample split for constrained steps | graphs; depth-1 share; D2 re-measured | 3 |
| 7 | Close: panel code review, plan §8.1, design docs | review clean | 1 |

## 8. Risks

- **Template drift.** Qwen3.5 and 3.8 templates differ in value serialization (`| string` vs `tojson`, R §4.1); the
  parser must follow the template the request was rendered with.
- **Reasoning ended by `<tool_call>`** without `</think>` (R §4.2, SGLang comment): the constraint must accept it;
  a fixture covers it.
- **Host time at large batches:** 64 constrained rows × (1 + k) fills ≈ 1–5 ms per step; batch fill in a thread pool
  (`BatchGrammarMatcher`, 8 threads ≈ 1 ms for 64 rows, R §1.2) if it shows.
- **xgrammar version.** vLLM pins 0.2.8, SGLang 0.2.7, we have 0.2.3 (R §4.3); pick one, re-run the reject-list
  oracle at the pin.

## 9. Review findings, resolved

(none yet)
