# M0.4 — speculative decoding: MTP, DFlash2, n-gram: design and implementation plan

Status: **revision 3, reviewed clean** (2026-10-13). Revisions 1–3 were reviewed by a panel of three; every finding is
resolved in §9.

Scope: plan §7, amended:
- DFlash/DFlash2 is added. It was previously a v2 draft-model drafter.
- MTP loads from the target checkpoint itself, not from a separate GGUF.
- `checkpoint_slots` is dropped (D6).

Builds on:
- the M0 spec seams (design §7.5–7.6): the SeqTable, verify rows, `std.spec.greedy`, and the `drafter` /
  `draft_policy` / `verifier` slots;
- M0.3: GDN state, in-step snapshots, the 27B GGUF locks.

Drafter facts come from source, with citations, in the local research note `docs/brainstorm/m04-drafters-spec.md`.
It covers SGLang `dflash_worker_v2.py` / `eagle_worker_v2.py` / `qwen3_5_mtp.py`, llama.cpp `dflash.cpp` /
`qwen35.cpp` / `speculative.cpp`, and transformers `candidate_generator.py`.

## 1. Goal and acceptance

Decode faster with speculation on consumer cards, without changing the output distribution. Acceptance:

1. **Drafters.**
   - **The plugins:**
     - `drafters/mtp`: the built-in MTP layer of Qwen3.5 / Qwen3.8, through `std.spec.mtp_head@1`;
     - `drafters/dflash`: z-lab DFlash2 (grouped conv, candidate selector; plain DFlash is both off);
     - `drafters/ngram`: GPU prompt lookup.
   - **L0-drafter** (§5) per drafter. Each draft head runs against a pure-torch reference of the published forward,
     on recorded target hidden states, with the GGUF weights dequantized and q8_1 emulated per path. The head is
     judged like L2: against the floor of a bf16 reference vs an fp32 reference, with top-1 agreement at confident
     positions. MTP chain steps 2 and later are teacher-forced.
2. **Lossless.**
   - **Greedy:** speculative output is **identical** to non-speculative greedy.
     - Every accepted token equals the verify step's own argmax (self-consistency, exact).
     - **CPU:** identical tokens on tiny models.
     - **GPU:** bounded (L1.5, with the KL arm, which works because verifiers now report logprobs, §3.5) on Qwen3-8B
       with n-gram, on Qwen3.5-9B bf16 with MTP, and on the 27B GGUF with MTP and with DFlash2.
   - **Sampled:** speculative decoding (`std.spec.rejection`) passes L0-dist (§5). The checks are the marginal χ²,
     the joint distribution of the first three emitted tokens on a toy Markov target, and exact zero counts outside
     the filtered support.
   - Both hold across the `test_spec.py` matrix (right / wrong / mixed / alternate drafts × depth 0 / 1 × roomy /
     preempting), extended to a tiny qwen3_5 with the oracle drafter and a tiny MTP.
3. **Recurrent rollback** (GDN replay, D6).
   - **Bitwise, at the GDN unit's level:** for every n_accept < k as well as n_accept = k, the unit's `rewind`
     leaves the SSM state and the conv window equal to 1 + n single-token decode calls. Checked on every kernel
     class and on the CPU reference, which computes its gating per token. Rows that are not rewound stay
     bitwise untouched (R ≥ 2 rows of q ≥ 2).
   - **Fallback, if a kernel class is not bitwise against the decode calls** (two Triton specializations,
     store-in-loop vs store-after): check bitwise against a call of the same `out_slots` kernel over the 1 + n
     accepted tokens alone, and hold the decode comparison to the drift and L0 bounds.
   - **Drift:** 200 verify/replay steps from a non-zero S0 stay within the GDN L0 bound of an f64 recurrence.
   - **No checkpoint lands in a verify row.** No checkpoint is registered at the end of a row with `n_logits > 1`,
     or for a request with an uncommitted verify row.
4. **Graphs and overlap.**
   - Verify steps replay captured graphs for the deployment's q (one per drafter policy, D7), drafter work and
     replay included.
   - Graph and eager runs are bitwise equal per step when eager runs the graph's exact padded plan, sub-steps
     included. Padded vs unpadded runs stay a bounded check.
   - Speculation runs at overlap depth 1.
   - Measured: the share of spec steps that are graphed and that run at depth 1.
5. **Speed** (evidence records; predictions in §5):
   - acceptance length (mean and per position) and decode tok/s, single stream and on the agent trace;
   - MTP on the RTX 5080 (UD-IQ3_S) and the RTX 4090 (UD-Q4_K_M);
   - DFlash2 on the RTX 4090 (q8_0 KV);
   - against `llama-server` with the same drafts.
6. **Invariants.**
   - With speculation off, nothing changes: the MTP block is not loaded, the pool layout is unchanged, and L2 and
     every lock's snapshot trace stay identical.
   - With speculation on, snapshot traces (tokens and per-step `n_accept`) are run-identical for identical
     schedules. A change in `n_accept` needs approval, like a token change: it is the acceptance-length regression
     gate.
7. **Core:** only the seams of §3, each proven by a toy plugin in `core/tests`. Core stays ≤ 6000 lines (4614 now,
   about 450 added).

Non-goals:
- tree verification and EAGLE-3;
- grammar under speculation (M0.5 builds on §3.5);
- the CPU lookup drafter;
- the Strata-style cost-model policy (§7).

## 2. Starting point

**Already built in M0:**
- Verify rows of 1 + k tokens, all producing logits, with drafts read on the device from `SeqTable.draft_tokens`.
- `std.spec.greedy` returns tokens and `n_accept`; `_store` advances `num_computed` by 1 + n_accept.
- `propose(ctx)` runs after **sampled** steps only.
- The scheduler takes the smallest k any row allows (`sched.py:98`) and reserves 1 + k (`sched.py:104`).
- Spec steps are eager.
- Only greedy, logprob-free requests are drafted (`engine.py:118-124`).
- `attention.paged_verify@1` exists and reads lengths on the device.

**Declared but unbuilt:**
- `ModelSpec.taps`, `ModelSpec.draft_heads` / `DraftHeadDecl` (`api/model.py:33-40, 73-74`).
- `Rewindable.commit`, `host_commit` / `CommitView` and `verify_mode`. The core never calls any of them.
- **So a GDN model under speculation is wrong today:** a verify row advances the state over rejected drafts.

**The GDN kernels** (`kernels/fla`):
- Any row longer than one token takes the chunked path. It gathers the row's state, runs ≈ 8 kernels and scatters
  the state back (≈ 0.9 GiB moved per row per call across 48 layers), and rounds S to bf16 in one product.
- The vendored `fused_sigmoid_gating.py` is a recurrent kernel built for spec decode: multi-token rows, fp32 S
  throughout. It serves decode today (one token per row).
- **That kernel has a bug on rows longer than one token** (`:166-176`). Inside its token loop it stores S at every
  token, to `indices + i_n·stride_seq + i_t`; `stride_indices_tok` is computed but unused. With our 1-D `slots`,
  token t of row n writes `slots[n + t]`: a neighbour's live state. D6 patches it.

**MTP:**
- **Where it is stored:**
  - HF `mtp.*` (15 tensors in Qwen3.5-4B, Qwen3.5-9B and Qwen3.8-27B).
  - GGUF `blk.64.*` in the unsloth 27B files: 334.7 MiB (Q6_K, with q8_0 k/v). The recipe skips both forms today.
  - The output head `output.weight` is Q5_K, 834 MiB, in UD-IQ3_S, and Q6_K, 995 MiB, in UD-Q4_K_M.
- **The layer:** the target's gated full-attention layer plus a SwiGLU MLP.
  - In front of it: `fc` [H, 2H] over `concat(enorm(embed(t)), hnorm(h))`, embedding first.
  - After it: `mtp.norm`, then the target's `lm_head`. All norms are zero-centred.
- **`h`:** the target's **final-norm** hidden state, i.e. the `lm_head` input.

**DFlash2** (z-lab Qwen3.8-27B-DFlash2):
- **Architecture:** 5 Qwen3 layers (H 5120, 32/8 heads, head_dim 128, plain RMSNorm), sliding window 2048,
  non-causal; block 8.
- **Context:** `RMSNorm(fc(concat of the residual outputs of target layers 5, 19, 33, 47, 61)))`. Layers 5, 33
  and 61 are GDN; 19 and 47 are full attention. The context enters every draft layer as K/V only.
- **Conv and selector:** the conv is grouped and dynamic (taps 2, group 16), confined to the block. The selector
  takes the top 16 candidates per slot from the target `lm_head`, scores them with bilinear codebooks of rank 256,
  and walks them greedily or ancestrally. It emits q over the 16 candidates.
- **Sharing and size:** the target's `embed` and `lm_head` are shared. Weights are 1.92 B parameters (Q4_K_M GGUF
  1.06 GiB). The draft KV is 20 480 B per token.

**Measured on the 5080:**
- UD-IQ3_S with q8_0 KV at `max_seqs` 4 and 4 checkpoints: 1940 MiB of KV, which is 3654 units, which is
  **58 464 tokens** (`evidence/fit_qwen3.8-27b-gguf_2026-10-09.json`).
- A batch-1 decode step takes **15.6 ms** (tg128 63.7 tok/s); a batch-4 step takes 21.2 ms.
- GDN state is 147 MiB per sequence.

## 3. Core seams (today → change), each general

1. **Drafter construction and auxiliary models, namespaced, built only on demand.**
   - Today: `DraftHeadDecl` is declared and nothing is built. Drafters are only `p.load`ed (`engine.py:98`).
   - Change: a drafter plugin builds through `DrafterPlugin.build(ctx: BuildCtx, target) -> Drafter`. The core
     calls it only when a drafter is bound.
     - `target.shared(name)` returns the target's built modules (`embed`, `norm`, `lm_head`). They are shared, not
       loaded twice.
     - `target.draft_head(name)` asks the model plugin (the drafter cannot import it). The model plugin answers
       `ModelPlugin.draft_head(config, source, name) -> AuxSpec | None`, with weight rules over the target's own
       source. The target keeps its skip.
   - Weights:
     - `target.tensors()` gives the target's own source.
     - `ctx.weight_source(path)` opens a drafter's own checkpoint, resolved through `detect_source` (weight sources
       are plugins).
     - `WeightMap` gains an `only=` prefix filter, so an aux map over the target's source names only its
       tensors.
   - `target` is a `holon.api` Protocol, `TargetView`.
   - `ctx.aux_model(namespace, AuxSpec(unit, layers, weights), tensors)` builds into a **child context**.
     - `unit` is a Ref that builds its own internals. `layers` only drives state binding.
     - The child maps state names, planner groups and `BatchPlan.publishes` to `<namespace>.<name>`. A GQA unit's
       `ctx.state("kv", i)` and `ctx.meta("kv")` therefore resolve to `mtp.kv`, with the units unchanged.
     - It loads the weights under `weights`.
   - With speculation off nothing is built, and the recipe hash, the pools and the locks are unchanged.
     `canonical_hash` skips default-valued fields (`terms.py:63-79`), so deleting the fields below does not move
     any hash.
   - Paged aux state is its own spec in the prefix index (`assembly.py:93-100`), so drafter KV is prefix-shared and
     truncated by length. The ledger fills paged pools to equal token counts (`memory.py:173-178`).
   - `DraftHeadDecl`, `ModelSpec.draft_heads` and `ModelSpec.taps` are deleted.
   - Toy proof:
     - a toy drafter with its own paged spec and an `in_step` recurrent spec;
     - it is prefix-hit and rolled back, never sees the target's publishes, and is absent when unbound.
2. **Sub-steps: a drafter's own forward rows.**
   - Today: every planner is planned once per step, for the step's batch.
   - Change: `StepCtx.substep(q, offset)` returns a child step whose rows are the step's sampled rows. Each gets q
     tokens at device positions `num_computed[slot] + offset + j`, read after `_store` has advanced
     `num_computed`.
     - Each sub-step owns its static buffers: positions, lengths, write indices, input ids. The input ids are
       filled by the drafter.
     - It is planned only for the child namespace's groups, through the verify-class kernels (device lengths,
       host upper bounds), with no publishes. Its metas are prepared immediately before its forward. Nothing it
       touches is shared with the parent step or another sub-step: one kernel instance per sub-step.
   - Sub-steps are planned at launch and captured inside the step's graph. A drafter declares its sub-step shapes
     (`Drafter.substeps(k) -> list[int]`), so the capture set stays finite.
   - **Reservation:** `Drafter.lookahead(k)` is the number of positions past the row it writes (MTP: k; DFlash:
     block). The scheduler reserves:
     - `1 + k + lookahead` for a verify row;
     - `n + lookahead` for a prefill chunk that ends its prompt;
     - always clamped to `max_model_len − num_computed`, and included in the scheduler's fast-path check
       (`sched.py:91`).
   - **The length limit:**
     - Columns at or past `max_model_len`, whether padded verify columns or drafter positions, get their
       **position** clamped to `max_model_len − 1` and their write sent to the null unit. Nothing then indexes past
       the block table or the rope cache: FlashInfer's cos/sin cache has `max_model_len` rows and no bounds check.
       Their outputs are never accepted.
     - The drafter caps `n_draft` at `max_model_len − 1 − start`.
   - Toy proofs:
     - a toy drafter writes a paged spec k positions ahead at depth 1, with graphs. Its pages must be reserved and
       never another request's, from the first draft after prefill on.
     - verify rows one token below the limit.
3. **Drafter lifecycle.**
   - Today: `propose(ctx)` after sampled steps.
   - Change:
     - `Drafter.admit(slot, tokens)` (host) receives the request's full token history, prefix-hit tokens included,
       and resets per-slot carries. The drafter copies the history itself with a pinned asynchronous copy, as the
       GDN unit does (`gdn/units.py:68-70`). The step uploader is sized per step and cannot hold a 32K history.
     - `Drafter.run(ctx, h)` runs after **every** step's sample stage, prefill chunks included. It gets the
       trunk's output `h`, the device `n_accept`, and `ctx.substep`. It writes the drafter's state for the step's
       tokens, then drafts for the rows that sampled.
   - Toy proof: a toy drafter whose per-slot hash of everything admitted and absorbed equals the committed
     history, across chunked prefill, rejection, preemption and prefix hits.
4. **Taps.**
   - Today: none.
   - Change: `Model.trunk` calls consumers a drafter registered at build (`ctx.tap(i, fn)`) with the residual
     stream after layer i. What it keeps is the consumer's choice: a copy, or an fp32 accumulation.
   - Toy proof: a toy consumer sees the tensors of a model-only reference run.
5. **Verification: rewind, draft lengths, sampling, logprobs.**
   - **Rewind.**
     - Today: nothing is rolled back.
     - Change: after the verifier, inside the sample stage and before `Drafter.run`, the core calls `rewind(ctx)`
       on every built module (aux models' included) and every loaded slot object that defines it. They are found
       by duck typing, like `finalize_weights`.
     - `Rewindable` becomes `rewind(ctx, path=None)`.
     - Deleted: `verify_mode`, the kinds' `commit` and `host_commit`, and `CommitView`. None of them has a caller.
     - Paged state rewinds by length.
   - **Draft lengths.**
     - Today: one k per step, the minimum over its rows.
     - Change: the policy sets the step's k (D7), so the width is q = 1 + k.
     - The drafter writes `SeqTable.n_draft [S]` on the device, bounded by the deployment's **maximum** k, not the
       step's k. At depth 1 the drafts for step N + 1 are written before the host picks that step's k.
     - Short rows pad with token 0, and every verifier caps acceptance at `min(n_draft, k)`.
     - Drafters that write no state (n-gram) draft at the maximum k on every step, k = 0 steps included. `observe`
       then reports the uncapped would-have-drafted `n_draft`, which `Launched` brings back to the host with the
       tokens.
     - Drafters that write KV (MTP, DFlash) draft at the step's k, which their reservation covers
       (`lookahead(k)`). They never use the k = 0 fallback, and the next step's verifier caps acceptance at
       `min(n_draft, k)` either way.
     - A prefill step has no width of its own. Its k is `policy.k(seqs)` evaluated when it is scheduled, and that
       one value sets both the drafting of its prompt-ending chunks and their `n + lookahead(k)` reservation.
       A toy test shows the n-gram policy re-enabling after a hit.
     - Every running row joins a verify step: rows without drafts carry `n_draft = 0`. Prefill and decode never
       share a step (`sched.py:59`).
   - **Sampling.**
     - `Sampler` gains:
       - `distribution(logits, ctx, rows)`: the filtered probabilities `sample` draws from;
       - `uniform(ctx, positions, stream)`: counter-based, where stream 0 is bit-identical to today's draw;
       - `logprobs(logits, tokens, ctx)`.
     - Verifiers reach the sampler through `ctx.sampler`.
     - Verifiers return logprobs for emitted tokens, so logprob and sampled requests speculate too. The
       temperature and logprob gate in `_draft_k` is deleted.
     - `std.spec.greedy` samples rows with temperature > 0 through `ctx.sampler` from their first column, with
       `n_accept = 0`. It is never wrong for mixed traffic, only slower than `std.spec.rejection`.
     - **Gate B adds** the sparse draft probabilities `draft_q_ids` / `draft_q` [S, k_max, Kq]: no Gate A drafter
       needs them (Kq = 1). **Contract:** a drafter samples exactly from the q it publishes. Rows past `n_draft`
       are never read.
   - **Determinism.**
     - Greedy speculative output is deterministic.
     - Sampled speculative output is lossless in distribution, but not batch-invariant: drafts depend on batch
       shape and prefix history.
   - Toy proof: the oracle drafter, and the L0-dist tests of §5.
6. **Verify graphs.**
   - Today: decode only (`executor.py:150`).
   - Change: verify steps are captured per (bucket, q, sampler variant, planner keys), with q-wide padding rows.
     The capture set is every bucket × the policy's declared q set (MTP {3, 4}; n-gram {5} plus plain decode),
     because the policy picks k from running requests but the bucket comes from the scheduled rows. That is
     ≈ 18 graphs for MTP; their memory is measured.
   - The startup memory profile measures:
     - the largest verify graph with its sub-steps and the replay;
     - graph instantiation memory, by `cudaMemGetInfo` across captures.
7. **Publishing rule.**
   - Today: a step's row end is published whenever a kind asks.
   - Change: never at the end of a row with `n_logits > 1`, or for a request with an uncommitted verify row
     (`seq.inexact`). Its host start is then only an upper bound.
   - Consequence: at depth 1, a speculating request always has an uncommitted verify row, so `every:n` takes no
     checkpoints while generating. `prompt_end` (the default) is unaffected.
   - Toy proof: a checkpoint request after a partially accepted verify is never registered at the wrong position.

No model, mixer or drafter names enter the core; the CI denylist enforces this.

## 4. Decisions

- **D1. One milestone, two gates.**
  - **Gate A:** all of §3 except taps; GDN replay; MTP; n-gram; both verifiers; verify graphs; `fixed_k`; the 5080
    and 4090 MTP evidence.
  - **Gate B:** taps, the `causal` / `window` attention facts, sparse `draft_q`, `drafters/dflash`, and the 4090
    evidence.
- **D2. MTP from the target checkpoint, through a std head.**
  - `std.spec.mtp_head@1`, with knobs `order` ("embed_first") and `norm` (a Ref), plus a `reference.py` oracle.
    This is the shared shape of the Qwen3.5, DeepSeek-V3, GLM-4.5 and MiMo heads.
  - The qwen3_5 plugin's `draft_head("mtp")` returns an `AuxSpec`:
    - `unit`: the head;
    - `layers`: one `LayerDecl(attn, mlp)` with the trunk's own Refs;
    - `weights`: rules for `mtp.*` and `blk.{n}.*`. The target's own map keeps skipping them.
  - The head uses the target's `embed` and `lm_head` through `target.shared`. The memory sum in §5 counts only the
    335 MiB MTP block; the 834 MiB head is the target's.
  - The MTP KV is the aux spec `mtp.kv` (one segment), 1/16 more KV per token.
  - The generic `drafters/mtp` runs any head declared this way.
  - The "separate MTP GGUF" in the plan is superseded.
- **D3. MTP pairing (llama.cpp's): the MTP row at position p pairs (token(p), h(p − 1)).**
  - Every MTP KV entry then depends only on tokens ≤ p. SGLang's pairing (h(p), token(p+1)) would put a token from
    past a cached page into it.
  - `run` covers every input row of the step. It needs h(start − 1): it is carried per slot, tagged with its
    position, and replaced by zero unless the tag equals start − 1 (after a prefix hit, a preemption or a slot
    reuse).
  - `run` then drafts with **k one-row sub-steps** at offsets 0 … k − 1 (`substeps(k) = [1] × k`).
    - The first takes `last_token` (the bonus, or the token sampled after prefill) and h from the row of the
      emitted token, `query_start[r] + n_query[r] − n_logits[r] + n_accept[slot]`.
    - `_store` writes `n_accept = 0` for every sampled non-verify row, so a prefill's last chunk never reads a
      previous owner's value. Toy proof: a 1-token last chunk on a slot whose previous owner accepted 3.
    - Each later one takes the previous draft and its own output hidden.
    - This holds on every kind of step: full acceptance, partial acceptance, and a prefill's last chunk alike.
    - Rev 2's "substitute the bonus in the absorb pass" is dropped. It has no row when every draft is accepted,
      nor after a prefill.
  - The first sub-step rewrites the entry the next step's absorb pass would write, with the same pair.
  - Cost per step: one MTP layer pass over the step's rows (no head), plus k one-row MTP forwards with the head.
- **D4. DFlash2: a plugin with its own checkpoint and its own attention unit.**
  - `drafters/dflash` declares taps 5, 19, 33, 47, 61 (from `target_layer_ids`). Its attention unit composes std
    `linear`, `rmsnorm`, `rope` and `kv_write@1` for context K/V.
  - Its block attention is `attention.paged_verify@1` with two new **general** facts: `causal` (False:
    bidirectional within the row, causal to context) and `window` (2048). Sliding-window models need these too.
  - The grouped conv and the selector are units in the plugin.
  - The block is always 8 rows; DFlash declares k = 7 only.
  - Mask row: a shipped `mask_embedding.pt` stays the drafter's **own** vector and never overwrites the target's
    `embed[248070]` (`<|audio_start|>`). Without one, the target row is used, as llama.cpp does.
- **D5. Sampled drafting.**
  - MTP and n-gram draft deterministically (Kq = 1). Rejection with a deterministic proposal accepts d with
    probability p(d), and otherwise samples from p with d removed. That is lossless and needs no q.
  - DFlash2 walks ancestrally at the request's temperature with its own uniform stream (Kq = 16).
  - One verifier, `std.spec.rejection`, handles both. Greedy rows reduce to greedy-exact.
- **D6. GDN rollback by masked replay, on the recurrent kernel. No per-token state copies.**
  - **Why:** `checkpoint_slots` would cost (1 + k) × 147 MiB per sequence.
  - **Kernel routing:** verify rows and replays route to the recurrent kernel (`fused_sigmoid_gating.py`); the
    chunked kernel stays for prefill.
  - **Contract amendment:** `gated_delta_rule@1` gains optional `out_slots` [R], the units final states go to. It is
    declared as the kernel fact `out_slots=True`, so a provider without it can never be selected.
  - **The vendor.patch on the recurrent kernel:**
    - adds a constexpr path that stores S **once, after the token loop**, to `out_slots[n]`;
    - leaves the T = 1 decode specialization's arithmetic unchanged, so spec-off traces stay identical;
    - this fixes the per-token store into neighbours' units (§2).
    - L0 runs R ≥ 2 rows of q ≥ 2 and asserts every other unit is bitwise untouched.
  - **During verify:**
    - The verify forward writes final states to the padding slot's unit. That unit is a discard sink: only graph
      padding rows use it, always with `has_init` False.
    - The unit saves into a static replay buffer, ledger-counted:
      - the conv window [C, K−1]: 2.8 MiB per row for the 27B;
      - the row's conv input x and gates a, b: 0.95 MiB per token, for `max_seqs × (1 + k_max)` tokens.
  - **`rewind`:**
    - restores the conv window;
    - re-runs the conv over x, then the recurrent kernel from S0 over the 1 + k rows, with a = b = −1e4 at
      positions ≥ 1 + n_accept;
    - in fp32, g = −exp(A)·softplus(−1e4) = −0 and β = 0, so those tokens leave S bit-identical (checked against
      the kernel's thresholded softplus and `tl.sigmoid`);
    - writes the conv window at the accepted end.
  - No new kernel and no host length; captured in the graph.
  - **Cost:** one recurrent launch per GDN layer with a single final store, ≈ 0.29 GiB moved per row, ≈ 0.5 ms
    (≈ 3% of a step), paid twice (verify, replay).
  - **The CPU reference and provider** compute g and β per token inside the loop. Computed vectorised over T, CPU
    transcendentals differ with tensor length, and 79 of 150 cases were then not bitwise.
- **D7. Policy per step: `DraftPolicy.k(seqs) -> int`, plus `observe(seq, k, n_draft, n_accept)`.**
  - k is chosen per step, because q is per step. `seqs` are the running requests before reservation.
  - `observe` gets the lagged host results, one step late at depth 1.
  - **`draft_policy/fixed_k`:** a k per running-request bucket.
    - MTP: k = 3 at 1–2 requests, k = 2 at 3 or more.
    - DFlash2: k = 7.
    - n-gram: k = 4, but **k = 0 (a plain decode step) when no running request hit in its last 4 steps**. A miss
      at full width costs ≈ 3.3 ms single stream and may be a net loss at 4 requests (§5).
  - **The capture set** is every bucket × the declared q set (§3.6).
  - The Strata-style cost model is §7.
- **D8. Turning it on.**
  - Speculation is a slot binding (drafter, policy, verifier); k is the policy's tunable (design §5.7, level 1).
  - It stays off by default until Gate A's evidence lands, then is turned on for Qwen3.5 / 3.8 + MTP where the
    agent trace measures faster. Locks gain the binding through `propagate`, with approval.

## 5. Numerics, judge and predictions (fixed before measuring)

**L0-drafter**
- The reference is a pure-torch implementation of the head:
  - MTP from the HF module tree;
  - DFlash2 from research note §1.5 and §2.
- For GGUF heads the reference uses dequantized weights and emulates q8_1 activations per path (`q8_1.py`), as Tier
  B does.
- The floor is the bf16 reference vs the fp32 reference.
- The verdict follows L2:
  - top-1 agreement ≥ the floor's − 0.25 points, at confident positions (margin > the floor's 99th-percentile
    |Δlogit|);
  - mean KL ≤ 2 × the floor's.
- MTP steps 2 and later are teacher-forced.

**L0-dist** (`std.spec.rejection`)
- 10⁵ draws per case at fixed seeds; the χ² critical value is at p = 1e-3, with cells of expected count < 5 pooled.
- Cases:
  - deterministic d;
  - sparse q with Kq = 4;
  - d outside the top-p support (an exact zero count is required outside the support);
  - q ≈ p (the residual-mass guard);
  - a joint test of the first three emitted tokens on a toy Markov target (vocab 16, k = 3).
- The simulated power against `rejection_keeps_draft` is λ ≈ 5K–25K vs a critical value of 103.

**Planted bugs** (`planted.py`). Each one must fail the listed checks; every other check stays clean.

| Plant | Must fail |
|---|---|
| `mtp_concat_order` | L0-drafter; the acceptance-length gate |
| `dflash_tap_offset` | L0-drafter |
| `replay_off_by_one` | §1.3 bitwise; L1.5 |
| `replay_conv_row_end` (window taken at the row end) | §1.3; L1.5 |
| `verify_writes_state` | §1.3; L1.5 |
| `rejection_keeps_draft` | L0-dist (marginal) |
| `rejection_reused_uniform` (one coin across the chain) | L0-dist (joint) |
| `unmasked_padding` (pad drafts accepted past `n_draft`) | the exact assert `n_accept ≤ n_draft` on a toy target with p(0) large; Gate B: L0-dist with sparse q and `n_draft < k` (stale q rows). Lossless for Kq = 1, so no L1.5. |
| `verify_row_publish` (a checkpoint at a verify row's end) | the §3.7 toy proof |

**Acceptance-length predictions** (agent trace, greedy). Compared with llama.cpp at the same settings; a gap over
15% is investigated before any speed claim.
- MTP k = 3: 2.3–2.8 tokens per step.
- DFlash2: 3.5–5.
- n-gram: 1.3–1.8.

**Speed predictions** (decode; cost model: verify GEMMs ≈ +0.7 ms per extra column, plus drafts and replay)
- **5080 + MTP, single stream:** 1.5–1.8× vs 15.6 ms.
  - Each draft reads the MTP layer (335 MiB) plus the Q5_K head (834 MiB): ≈ 1.6 ms. The head is 71% of draft
    bytes.
  - The optimum k is flat between 2 and 4.
- **5080 + MTP at `max_seqs` 4:** 1.15–1.3× TPOT. With k = 2, 12-token verify steps go to mmq.
- **n-gram (Qwen3-8B and the 27B):** speed is recorded against the break-even acceptance per bucket. The predicted
  break-even single stream on the 5080 is ≈ 1.2 tokens per step, at a step of 15.6 + 4 × 0.7 + 0.5 ms.
- **4090 + DFlash2, single stream:** 2.0–3.0× vs 21.4 ms.
- **Pivot:** below 1.2× single stream for MTP on the 5080, after a perf pass → profile the step breakdown (verify
  GEMMs, GDN verify and replay, drafts) and fix the largest item. The candidates are a vocabulary-restricted head
  (v1) and an mmvq path for up to 16 rows.

**Memory, measured basis (5080, q8_0 KV, `max_seqs` 4, 4 checkpoints)**
- Start from 1940 MiB of KV.
- Subtract:
  - 0.9 × 335 MiB of MTP weights;
  - the replay buffer: 16 rows × 0.95 MiB, plus 4 × 2.8 MiB of conv windows;
  - the verify-graph pool, ≈ 100 MiB, measured at startup.
- That leaves ≈ 1520 MiB, at 0.564 MiB per unit (17/16 of today's): **≈ 42K tokens** (58K today).
- The 32K agent context still fits.

**4090 + DFlash2**
- With bf16 KV: ≈ 31–33K tokens, which misses 32K.
- With q8_0 KV: ≈ 54K tokens.
- **Gate B runs on q8_0 KV,** and the fit is recorded.

## 6. Steps

| # | Work | Checks | Days |
|---|---|---|---|
| 1 | Core: §3.1–3.3 and §3.5–3.7 with toy proofs. Line budgets: construction and namespacing 90, sub-steps and device prep 110, verify graphs and profile 60, admit/run/rewind 50, `n_draft`, padding and limit 50, sampler and verifier plumbing 40, policy, lookahead and publishing 40; deletions −40 | core tests; full suite; core ≤ 5100 lines | 4.5 |
| 2 | GDN: the recurrent kernel's single final store to `out_slots` (vendor.patch), the CPU provider and the reference (gating per token); routing; the replay buffer; `rewind` | unit-level `rewind` bitwise vs 1 + n decode calls (n < k and n = k; neighbours untouched) on sm_86/89/120; drift; the extended `test_spec` matrix on tiny qwen3_5 | 2 |
| 3 | `std.spec.mtp_head@1`, qwen3_5 `draft_head`, `drafters/mtp` | L0-drafter; CPU identity; Qwen3.5-9B bf16 bounded (A5000) | 3 |
| 4 | `drafters/ngram` (device history from `admit` + `run`), `fixed_k` | Qwen3-8B bounded; acceptance recorded | 1 |
| 5 | §3.5 sampler seam, `std.spec.rejection`, verifier logprobs | L0-dist; plants | 2.5 |
| 6 | §3.6 verify graphs; the 27B GGUF end to end | graph vs eager on the same padded plan, bitwise per step; depth-1 share; memory fit | 2 |
| 7 | **Gate A:** 5080 / 4090 MTP evidence vs llama-server; panel code review; merge | records; review clean | 2 |
| 8 | Taps, attention `causal` / `window` facts, `drafters/dflash` (context KV, block attention, conv, selector); header dump of the checkpoint first | L0-drafter; bounded 27B (4090) | 4 |
| 9 | **Gate B:** DFlash2 vs llama-server on the 4090 (q8_0 KV); panel code review; plan §7.1 status; merge | records; review clean | 2 |

## 7. Deferred, deliberately

- **The Strata-style cost-model policy.** It needs the per-position acceptance telemetry from step 7.
- **DFlash KV window reclaim** (freeing pages older than 2048 positions). This is a paged-kind policy, for v1 and
  sliding-window models. It reclaims 37% of per-token KV under q8_0.
- **The MTP hidden at a prefix hit's first position** (zero in D3). Carry it as a prefix-index payload if the
  acceptance cost measures above 2%.
- `checkpoint_slots` (dropped, D6), tree verification and EAGLE-3 (v2), and the CPU lookup drafter.

## 8. Risks

- **Sub-step planning cost.** Each verify step plans the target groups, the aux groups for the absorb pass, and
  the aux groups for k sub-steps on the host. If host time per step grows by more than 1 ms, cache plans per
  (bucket, q): the shapes are fixed per graph.
- **Weight names and `mask_embedding.pt`** are unverified until the DFlash2 checkpoint's headers are dumped
  (step 8).
- **Sliding-window off-by-one.** HF admits ≤ 2048; SGLang and llama.cpp admit ≤ 2047. We follow llama.cpp, whose
  GGUF we load. L0-drafter at contexts over 2048 settles it.
- **cuBLAS and mmq at 16-row verify steps** may be slower than the cost model assumes. The pivot covers it.

## 9. Review findings, resolved

**Rev 1** (reviewers: A correctness, B architecture, C numerics/performance/tests).

| Finding | Resolution |
|---|---|
| A-B1 / B-M5: `-1` padding crashes the embedding; padded rows are greedily decoded or crash on logprobs; reservation is per row | `n_draft` on the device, pad token 0, verifiers cap at `n_draft` (§3.5); every row verifies, sampled and logprob rows included; reserve 1 + k + lookahead (§3.2) |
| A-B2 / B-B1 / A-M1: drafter forwards have no positions, metas or pages; MTP writes past the reservation | §3.2 sub-steps and `lookahead` |
| A-M2 / B-m7: wrong checkpoint after a partial verify (`every:n`) | §3.7 publishing rule plus a toy proof |
| A-M3 / C-M7: verify graphs and replay not capturable or unaccounted | recurrent kernel with a handle cache per (rows, q) (D6); q-wide padding; memory profile incl. instantiation (§3.6); q set = the deployment's (D7) |
| A-M4 / B-M1 / C-m10: MTP loaded with speculation off; two build paths | one path, `aux_model`, built only for a bound drafter (§3.1) |
| A-M5 / B-M6: n-gram cannot see prefix-hit tokens; k decided on the wrong side | `Drafter.admit(slot, tokens)`; `n_draft` on the device |
| A-M6 / B-M2: aux names collide with the target's `kv` | namespaced child contexts (§3.1) |
| B-M3: DFlash attention as GQA knobs | own unit in the plugin; general `causal` / `window` facts (D4) |
| B-M4: MTP head belongs in std | `std.spec.mtp_head@1` with a reference (D2) |
| B-M7: Rewindable shape; discovery; `verify_mode` | `rewind(ctx, path)` by duck typing; `verify_mode` deleted (§3.5) |
| C-M1: chunked kernel for verify and replay is slow and inexact | recurrent kernel route; bitwise masked replay; drift test (D6, §1.3) |
| C-M2: memory on stale numbers | §5 redone on the measured q8_0 fit: ≈ 42K tokens |
| C-M3: speed basis | 15.6 ms basis, head bytes, max_seqs-4 prediction, k per bucket (D7), pivot on the step breakdown |
| C-M4: L1.5 KL arm blind to speculation | verifiers return logprobs (§3.5) |
| C-M5: L0-drafter bars unsound for GGUF | dequant + q8_1 emulation reference; floor-relative verdict; teacher forcing (§5) |
| C-M6: L0-dist marginal only | joint, zero-count, pooling, residual and padding cases; two plants (§5) |
| C-M8: DFlash2 misses 32K on 4090 bf16 KV | Gate B on q8_0 KV (§5) |
| A-m1: stale carried hidden | position-tagged carry (D3) |
| A-m2 / C-m5: CPU bitwise impossible | CPU: identical tokens; rollback bitwise at op level (§1.2–1.3) |
| A-m3: replay buffer size | 0.95 MiB/token with the conv recomputed from x; ledger-counted (D6) |
| A-m4: tap layers not all GDN | §2 corrected |
| A-m5: FlashInfer host lengths exclude inexact rows | verify steps read device lengths; noted for the scheduler change in step 1 |
| A-m6: uniform streams | stream 0 bit-identical; DFlash has its own stream (§3.5, D5) |
| A-m7: GGUF rules hit `blk.64` | the head has its own aux weight map; the target keeps its skip (D2) |
| B-m1: `out_slots` as a fact; drop the conv amendment | kernel fact; conv window saved by the unit (D6) |
| B-m2 / C-m8: dead tap fields; fp32 accumulation | fields deleted; the consumer chooses copy or fp32 (§3.4) |
| B-m3: one drafter hook | `run(ctx, h)` (§3.3) |
| B-m4: Gate B is not "only a plugin" | D1 restated |
| B-m5: §3.3 toy proof | admitted/absorbed-history hash test (§3.3) |
| B-m6: `[spec]` lock section | slot bindings + policy tunable (D8) |
| B-m8 / C-M7: q set | per deployment (D7) |
| B-m9: verifier access to the sampler | `ctx.sampler` (§3.5) |
| B-m10: which norm | `h` is the target's final-norm output; `mtp.norm` is the head's (§2, D2) |
| B: determinism claim; sparse-q contract | weakened, added (§3.5) |
| C-m1: acceptance-length regression gate | `n_accept` in snapshot traces (§1.6) |
| C-m2: GDN speculation with prefix hits and preemption | extended `test_spec` matrix (§1.2) |
| C-m3: self-consistency | restored (§1.2) |
| C-m4: more plants | `replay_conv_row_end`, `verify_row_publish`, `rejection_reused_uniform`, `unmasked_padding` |
| C-m6 / C-m7: rollback and graph checks | bitwise masked replay + drift; graph vs eager bitwise per step |
| C-m9: fold MTP's first draft into the absorb pass | bonus substitution (D3) |

**Rev 2.**

| Finding | Resolution |
|---|---|
| A-M1 / C-N1: the recurrent kernel stores S per token into neighbours' units | vendor.patch: one store after the loop to `out_slots[n]`; T = 1 path unchanged; neighbour-untouched L0 (D6, §2) |
| A-M2 / B-N2 / C-m4: bonus substitution has no row at full acceptance or after prefill | dropped; k one-row sub-steps from offset 0 on every step (D3) |
| A-M3: verify rows near `max_model_len` index past the table | positions at or past the limit go to the null unit; `n_draft` capped; toy at limit − 1 (§3.2) |
| A-M4: prefill's last chunk reserves no lookahead | `n + lookahead` for a prompt-ending chunk (§3.2) |
| A-M5 / B-n1: sub-steps share buffers and kernel instances; plan every group; inherit publishes | per-sub-step static buffers and instances; child groups only; no publishes; prepared right before the forward (§3.2) |
| B-N1: drafter construction and sharing undefined | `DrafterPlugin.build(ctx, target)`; `target.shared` / `target.draft_head`; `AuxSpec(unit, layers, weights)` (§3.1, D2) |
| B-n2: policy per request vs per step | `DraftPolicy.k(seqs)` per step; `observe` lagged (D7) |
| B-n3: sparse q over-built for Gate A | moved to Gate B (§3.5, D1) |
| B-n4: admit staging | the drafter's own pinned asynchronous copy (§3.3) |
| B-n5: publishes leak through namespacing | the child maps or filters publishes; aux `in_step` toy (§3.1) |
| B-n6: `host_commit` / `CommitView` have no caller | deleted (§3.5) |
| B-n7: line budget | per-seam budgets in step 1; core ≤ 5100 |
| C-N2: `unmasked_padding` cannot fail L1.5 | must-fail checks corrected (§5) |
| C-N3: n-gram misses pay full width | `fixed_k` falls back to k = 0 without recent hits; break-even recorded (D7, §5) |
| C-m1: graph vs eager bitwise | on the same padded plan; padded vs unpadded stays bounded (§1.4) |
| C-m2: §1.3 must be unit-level | unit `rewind` with the conv window, n < k cases, neighbours (§1.3, step 2) |
| C-m3: `verify_row_publish` prediction | the toy proof only (§5) |
| C-m5: 16 vs 12 tokens; capture set | corrected (§5, §3.6, D7) |
| C-m6: L0-drafter number | floor − 0.25 points top-1; KL ≤ 2 × floor (§5) |
| A-m1: CPU reference not bitwise | gating per token in the reference and the CPU provider (D6) |
| A-m2: greedy verifier with sampled rows | samples them from the first column, `n_accept = 0` (§3.5) |
| A-m3: conv window not counted | 2.8 MiB per row, ledger-counted (D6, §5) |
| A-m4: no `every:n` checkpoints while speculating | stated (§3.7) |

**Rev 3.**

| Finding | Resolution |
|---|---|
| B-R1: n-gram fallback switches itself off (`n_draft ≤` step k) | `n_draft` bounded by the deployment's maximum k; verifiers cap at `min(n_draft, k)`; drafting and `observe` continue on k = 0 steps (§3.5) |
| B-r1: aux tensors unspecified | `target.tensors()`, `ctx.weight_source(path)`, `WeightMap(only=…)`, `TargetView` (§3.1) |
| C-n1: fallback latches off | covered by B-R1; `observe` gets the uncapped length; re-enable toy (§3.5) |
| C-n2: bitwise across two kernel specializations | fallback named (§1.3) |
| A-M1: h from row `n_accept` is stale after prefill (another request's hidden, or out of bounds) | h from the emitted token's row; `_store` zeroes `n_accept` for non-verify rows; toy proof (D3) |
| A-M2: positions past the limit overrun the rope cache | positions clamped as well as pages (§3.2) |
| A-M3: n-gram fallback never resumes | drafting at the maximum k on every step; `n_draft` back to the host (§3.5) |
| A-m1: reservation clamp dropped | restored, with the fast-path check (§3.2) |
| A-m2: capture set misses shapes | every bucket × the declared q set (§3.6, D7) |
| A-M1 (rev 3 recheck): drafting at the maximum k overruns MTP's reservation | only stateless drafters draft at the maximum k; MTP and DFlash draft at the step's k (§3.5) |
