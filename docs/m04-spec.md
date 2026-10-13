# M0.4 — speculative decoding: MTP, DFlash2, n-gram: design and implementation plan

Status: **revision 2, under review** (2026-10-13). Rev 1 was reviewed by a panel of three; every finding is resolved
in §9.

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
   - **Bitwise:** the masked replay of a verify row equals a call over its 1 + n accepted tokens alone. Checked on
     every kernel class and on the CPU reference.
   - **Drift:** 200 verify/replay steps from a non-zero S0 stay within the GDN L0 bound of an f64 recurrence.
   - **No checkpoint lands in a verify row.** No checkpoint is registered at the end of a row with `n_logits > 1`,
     or for a request with an uncommitted verify row.
4. **Graphs and overlap.**
   - Verify steps replay captured graphs for the deployment's q (one per drafter policy, D7), drafter work and
     replay included.
   - Graph and eager runs are bitwise equal per step.
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
- `Rewindable.commit` and `verify_mode`. The core never calls `commit` or reads `verify_mode`.
- **So a GDN model under speculation is wrong today:** a verify row advances the state over rejected drafts.

**The GDN kernels** (`kernels/fla`):
- Any row longer than one token takes the chunked path. It gathers the row's state, runs ≈ 8 kernels and scatters
  the state back (≈ 0.9 GiB moved per row per call across 48 layers), and rounds S to bf16 in one product.
- The vendored `fused_sigmoid_gating.py` is a recurrent kernel built for spec decode: multi-token rows, fp32 S
  throughout, per-token state indices. It is unused.

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

1. **Auxiliary models, namespaced, built only on demand.**
   - Today: `DraftHeadDecl` is declared and nothing is built.
   - Change: `BuildCtx.aux_model(namespace, spec, tensors)` builds a ModelSpec-shaped set of units into a **child
     context**.
     - The child maps state names and planner groups to `<namespace>.<name>`. A GQA unit's `ctx.state("kv", i)` and
       `ctx.meta("kv")` therefore resolve to `mtp.kv`, with the units unchanged.
     - It binds the state into its own groups and planners, and loads the weights under the spec's map.
   - Where specs come from:
     - **MTP:** the model plugin answers `draft_head(config, source, name) -> ModelSpec | None`, with weight rules
       over the target's own source. The target keeps its skip.
     - **DFlash:** the drafter plugin brings its own checkpoint.
   - The core calls `aux_model` only for a drafter that is bound. With speculation off nothing is built, and the
     recipe hash, the pools and the locks are unchanged.
   - Paged aux state is a separate spec in the prefix index (`assembly.py:93-100` already indexes every paged
     spec), so drafter KV is prefix-shared and truncated by length.
   - `DraftHeadDecl`, `ModelSpec.draft_heads` and `ModelSpec.taps` are deleted.
   - Toy proof: a toy drafter with its own paged spec, prefix-hit, rolled back, and absent when unbound.
2. **Sub-steps: a drafter's own forward rows.**
   - Today: every planner is planned once per step, for the step's batch.
   - Change: `StepCtx.substep(q, offset)` returns a child step whose rows are the step's sampled rows. Each gets q
     tokens at device positions `num_computed[slot] + offset + j`.
     - Its `input_ids` are a static buffer the drafter fills.
     - Its page write indices are derived on the device.
     - Its metas come from every planner (target and aux groups) through the verify-class kernels, which read
       device lengths and are planned on the host from upper bounds.
   - Sub-steps are planned at launch with the step and captured inside its graph. A drafter declares its sub-step
     shapes up front (`Drafter.substeps(k) -> list[int]`), so the capture set stays finite.
   - **Reservation:** `Drafter.lookahead(k)` is the number of positions past 1 + k it writes (MTP: k; DFlash: block).
     The scheduler reserves `1 + k + lookahead` per row, clamped at `max_model_len`. Drafters clamp their positions
     to the same limit.
   - Toy proof: a toy drafter writing a paged spec k positions ahead at depth 1, with graphs. Its pages must be
     reserved, never another request's.
3. **Drafter lifecycle.**
   - Today: `propose(ctx)` after sampled steps.
   - Change:
     - `Drafter.admit(slot, tokens)` (host; staged through the Uploader like `Sampler.admit`) receives the
       request's full token history, prefix-hit tokens included. It also resets per-slot carries.
     - `Drafter.run(ctx, h)` runs after **every** step's sample stage, prefill chunks included. It gets the
       trunk's output `h`, the device `n_accept`, and `ctx.substep`.
     - It writes the drafter's state for the step's tokens, then drafts for the rows that sampled.
   - Toy proof: a toy drafter whose per-slot hash of everything admitted and absorbed equals the committed
     history, across chunked prefill, rejection, preemption and prefix hits.
4. **Taps.**
   - Today: none.
   - Change: `Model.trunk` calls consumers a drafter registered at build (`ctx.tap(i, fn)`) with the residual
     stream after layer i. What a consumer keeps is its own choice: a copy, or an fp32 accumulation.
   - Toy proof: a toy consumer sees the tensors of a model-only reference run.
5. **Verification: rewind, draft lengths, sampling, logprobs.**
   - **Rewind.**
     - Today: nothing is rolled back.
     - Change: after the verifier, inside the sample stage and before `Drafter.run`, the core calls `rewind(ctx)`
       on every built module and every loaded slot object that defines it, found by duck typing like
       `finalize_weights`.
     - `Rewindable` becomes `rewind(ctx, path=None)`. `verify_mode` and the kinds' device `commit` are deleted;
       paged state rewinds by length. `host_commit` stays.
   - **Draft lengths.**
     - Today: one k per step, the minimum over its rows.
     - Change: the drafter writes `SeqTable.n_draft [S]` on the device. The host's k is an upper bound that sets
       the step's width q = 1 + k from the policy (D7).
     - Rows with fewer drafts pad with token 0, and every verifier caps acceptance at `n_draft`.
     - Every running row joins a verify step: rows without drafts carry `n_draft = 0`.
   - **Sampling.**
     - `Sampler` gains:
       - `distribution(logits, ctx, rows)`: the filtered probabilities `sample` draws from;
       - `uniform(ctx, positions, stream)`: counter-based, where stream 0 is bit-identical to today's draw;
       - `logprobs(logits, tokens, ctx)`.
     - Verifiers reach the sampler through `ctx.sampler`.
     - The SeqTable gains sparse draft probabilities `draft_q_ids` / `draft_q` [S, k_max, Kq], where Kq is the
       drafter's declared candidate count (1 = deterministic). **Contract:** a drafter samples exactly from the q it
       publishes.
     - Verifiers return logprobs for emitted tokens, so logprob and sampled requests speculate too. The temperature
       and logprob gate in `_draft_k` is deleted.
   - **Determinism.**
     - Greedy speculative output is deterministic.
     - Sampled speculative output is lossless in distribution, but not batch-invariant: drafts depend on batch
       shape and prefix history.
   - Toy proof: the oracle drafter with known q, and the L0-dist tests of §5.
6. **Verify graphs.**
   - Today: decode only (`executor.py:150`).
   - Change: verify steps are captured per (bucket, q, sampler variant, planner keys), with q-wide padding rows.
   - The startup memory profile measures:
     - the largest verify graph with its drafter sub-steps and the replay;
     - graph instantiation memory, by `cudaMemGetInfo` across captures.
7. **Publishing rule.**
   - Today: a step's row end is published whenever a kind asks.
   - Change: never at the end of a row with `n_logits > 1`, or for a request with an uncommitted verify row
     (`seq.inexact`). Its host start is then only an upper bound.
   - Toy proof: a checkpoint request after a partially accepted verify is never registered at the wrong position.

No model, mixer or drafter names enter the core; the CI denylist enforces this.

## 4. Decisions

- **D1. One milestone, two gates.**
  - **Gate A:** all of §3 except taps; GDN replay; MTP; n-gram; both verifiers; verify graphs; `fixed_k`; the 5080
    and 4090 MTP evidence.
  - **Gate B:** taps, the `causal` / `window` attention facts, `drafters/dflash`, and the 4090 evidence.
- **D2. MTP from the target checkpoint, through a std head.**
  - `std.spec.mtp_head@1`, with knobs `order` ("embed_first") and `norm` (a Ref), plus a `reference.py` oracle.
    This is the shared shape of the Qwen3.5, DeepSeek-V3, GLM-4.5 and MiMo heads.
  - The qwen3_5 plugin's `draft_head("mtp")` returns:
    - the head;
    - one `LayerDecl(attn, mlp)` with the trunk's own Refs;
    - weight rules for `mtp.*` and `blk.{n}.*`, which are then removed from the target's skip only inside this aux
      map.
  - The MTP KV is the aux spec `mtp.kv` (one segment), 1/16 more KV per token.
  - The generic `drafters/mtp` runs any head declared this way.
  - The "separate MTP GGUF" in the plan is superseded.
- **D3. MTP pairing (llama.cpp's): the MTP row at position p pairs (token(p), h(p − 1)).**
  - Every MTP KV entry then depends only on tokens ≤ p. SGLang's pairing (h(p), token(p+1)) would put a token from
    past a cached page into it.
  - `run` covers every input row of the step. It needs h(start − 1): it is carried per slot, tagged with its
    position, and replaced by zero unless the tag equals start − 1 (after a prefix hit, a preemption or a slot
    reuse).
  - **On verify steps, the token at row n_accept + 1 is replaced by the bonus token** before the MTP forward. That
    row's output is then the first draft, so no extra forward is needed for it.
  - Then k − 1 one-row sub-steps follow. Cost: one (1 + k)-row MTP layer and k heads.
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
  - **Contract amendment:** `gated_delta_rule@1` gains optional `out_slots` [R], the units final states go to,
    declared as the kernel fact `out_slots=True`, so a provider without it can never be selected. It comes with a
    vendor.patch that splits the recurrent kernel's read and write indices.
  - **During verify:**
    - The verify forward writes final states to the padding slot's unit. That unit is a discard sink: only graph
      padding rows use it, always with `has_init` False.
    - The unit saves the conv window [C, K−1] and the row's conv input x and gates a, b into a static replay buffer
      (0.95 MiB per token for the 27B; `max_seqs × (1 + k_max)` tokens, ledger-counted).
  - **`rewind`:**
    - restores the conv window;
    - re-runs the conv over x, then the recurrent kernel from S0 over the 1 + k rows, with a = b = −1e4 at
      positions ≥ 1 + n_accept;
    - in fp32, g = −exp(A)·softplus(−1e4) = −0 and β = 0, so those tokens leave S bit-identical;
    - writes the conv window at the accepted end.
  - No new kernel and no host length; captured in the graph.
  - **Cost:** one recurrent launch per GDN layer, ≈ 0.29 GiB moved per row, ≈ 0.5 ms (≈ 3% of a step).
- **D7. `draft_policy/fixed_k`, per drafter and per batch-size bucket.**
  - `DraftPolicy.choose(seq, rows)` gains the step's row count.
  - Defaults: MTP k = 3 at 1–2 rows and k = 2 at 3 or more; DFlash2 k = 7; n-gram k = 4 (`n_draft` is 0 on a miss).
  - **The graph q set is the deployment's** (e.g. {3, 4} for MTP), not a union over drafters.
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
- The verdict follows L2: top-1 agreement at confident positions (margin > the floor's 99th-percentile |Δlogit|),
  plus a KL bound.
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
| `unmasked_padding` (pad drafts accepted past `n_draft`) | the exact assert `n_accept ≤ n_draft`; L1.5 |
| `verify_row_publish` (a checkpoint at a verify row's end) | §1.3 toy; prefix-hit bounded |

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
- **5080 + MTP at `max_seqs` 4:** 1.15–1.3× TPOT. Verify rows of 16 tokens go to mmq.
- **4090 + DFlash2, single stream:** 2.0–3.0× vs 21.4 ms.
- **Pivot:** below 1.2× single stream for MTP on the 5080, after a perf pass → profile the step breakdown (verify
  GEMMs, GDN verify and replay, drafts) and fix the largest item. The candidates are a vocabulary-restricted head
  (v1) and an mmvq path for up to 16 rows.

**Memory, measured basis (5080, q8_0 KV, `max_seqs` 4, 4 checkpoints)**
- Start from 1940 MiB of KV.
- Subtract:
  - 0.9 × 335 MiB of MTP weights;
  - the replay buffer, 16 rows × 0.95 MiB;
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
| 1 | Core: §3.1–3.3 and §3.5–3.7 with toy proofs (aux namespacing, sub-steps and lookahead, admit/run, rewind, `n_draft`, publishing rule) | core tests; full suite | 4 |
| 2 | GDN: `out_slots` in the recurrent kernel (vendor.patch), the CPU provider and the reference; routing; the replay buffer; `rewind` | L0 bitwise + drift on sm_86/89/120; the extended `test_spec` matrix on tiny qwen3_5 | 2 |
| 3 | `std.spec.mtp_head@1`, qwen3_5 `draft_head`, `drafters/mtp` | L0-drafter; CPU identity; Qwen3.5-9B bf16 bounded (A5000) | 3 |
| 4 | `drafters/ngram` (device history from `admit` + `run`), `fixed_k` | Qwen3-8B bounded; acceptance recorded | 1 |
| 5 | §3.5 sampler seam, `std.spec.rejection`, verifier logprobs | L0-dist; plants | 2.5 |
| 6 | §3.6 verify graphs; the 27B GGUF end to end | graph vs eager bitwise per step; depth-1 share; memory fit | 2 |
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

- **Sub-step planning cost.** Each verify step plans the target groups, plus the aux groups, plus k − 1 MTP
  sub-steps on the host. If host time per step grows by more than 1 ms, cache plans per (bucket, q): the shapes are
  fixed per graph.
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
