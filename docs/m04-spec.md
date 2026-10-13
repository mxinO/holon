# M0.4 — speculative decoding: MTP, DFlash2, n-gram: design and implementation plan

Status: **revision 1, under review** (2026-10-13). Scope: plan §7, amended. DFlash/DFlash2 is added: it was a v2
draft-model drafter. MTP now loads from the target checkpoint itself, not from a separate GGUF. `checkpoint_slots` is
dropped (D6). Builds on the M0 spec seams (design §7.5–7.6: the SeqTable, verify rows, `std.spec.greedy`, the
`drafter` / `draft_policy` / `verifier` slots) and on M0.3 (GDN state, in-step snapshots, the 27B GGUF locks).
Drafter facts come from source, with citations, in the local research note `docs/brainstorm/m04-drafters-spec.md`.
The references read were SGLang `dflash_worker_v2.py` / `eagle_worker_v2.py` / `qwen3_5_mtp.py`, llama.cpp
`dflash.cpp` / `qwen35.cpp` / `speculative.cpp`, and transformers `candidate_generator.py`.

## 1. Goal and acceptance

Decode faster with speculation on consumer cards, without changing the output distribution. Acceptance:

1. **Drafters.**
   - `drafters/mtp`: the built-in MTP layer of Qwen3.5 / Qwen3.8, HF `mtp.*` or GGUF `blk.<n>.nextn.*`.
   - `drafters/dflash`: z-lab DFlash2, i.e. DFlash with the grouped conv and the candidate selector. Plain DFlash is
     the same code with both turned off.
   - `drafters/ngram`: GPU prompt lookup.
   - Each drafter's draft logits match a pure-torch reference of the published forward on recorded target hidden
     states: top-1 agreement ≥ 99% and logits within the GEMM-reorder bound (L0-drafter, §5).
2. **Lossless.**
   - Greedy speculative output is **identical** to non-speculative greedy:
     - bitwise on CPU tiny models;
     - bounded on GPU (the L1.5 bound used today for overlap vs sync): Qwen3-8B with n-gram, Qwen3.5-9B bf16 with
       MTP, and the 27B GGUF with MTP and DFlash2.
   - Sampled speculative decoding (`std.spec.rejection`) passes L0-dist: the emitted token distribution matches the
     target's under χ² at fixed seeds, for deterministic and for sparse-q drafts.
3. **Recurrent rollback** (GDN replay, D6). Verify plus commit of n accepted drafts leaves the SSM and conv state
   equal to n + 1 plain decode steps:
   - within the GDN L0 bound per layer (CPU reference and every kernel class);
   - bounded end to end (L1.5).
   - Prefix checkpoints never fall inside or at the end of a verify row.
4. **Graphs and overlap.** Verify steps (decode rows that carry drafts) replay captured graphs for every `spec.q` the
   policy declares, drafter work included. Speculation runs at overlap depth 1. Measured: the share of spec steps
   graphed and the share run at depth 1.
5. **Speed** (evidence records; the predictions are in §5):
   - acceptance length and decode tok/s, single stream and on the agent trace;
   - MTP on the RTX 5080 (UD-IQ3_S) and the RTX 4090 (UD-Q4_K_M);
   - DFlash2 on the RTX 4090;
   - against `llama-server` with the same drafts at the same commit as the M0.3 records.
6. **Judge invariants hold.** L2 and the locks' snapshot traces are unchanged with speculation off. Snapshot traces
   with speculation on are run-identical.
7. **Core:** only the seams of §3, each proven by a toy plugin in `core/tests`. Core stays ≤ 6000 lines (4614 now).

Non-goals:
- tree verification (`path` stays None; v2) and EAGLE-3;
- grammar under speculation (M0.5, which builds on §3.4);
- logprobs on speculative rows (they stay non-speculative);
- the CPU lookup drafter (`lookup_cpu`, deferred: the GPU n-gram covers prompt lookup without gating depth);
- the Strata-style cost-model policy (§7).

## 2. Starting point

- **Already in place (M0):**
  - verify rows of 1 + k tokens, all producing logits, with drafts read on the device from `SeqTable.draft_tokens`;
  - `std.spec.greedy` returns tokens and `n_accept`; `_store` advances `num_computed` by 1 + n_accept;
  - the drafter's `propose(ctx)` runs after **sampled** steps only;
  - the scheduler takes the smallest k any row allows;
  - spec steps are eager;
  - only greedy, logprob-free requests are drafted (`Engine._draft_k`);
  - `attention.paged_verify@1` exists (lengths read on the device).
- **Declared but unbuilt:**
  - `ModelSpec.taps` and `ModelSpec.draft_heads` / `DraftHeadDecl`;
  - `Rewindable.commit`, which the core never calls;
  - the recurrent kind's `verify_mode = "trivial"`. **Today a GDN model under speculation is wrong:** a verify row
    advances the state over rejected drafts.
- **The MTP layer:**
  - HF `mtp.*` has 15 tensors in Qwen3.5-4B, Qwen3.5-9B and Qwen3.8-27B (local snapshots). GGUF `blk.64.nextn.*`
    is in the unsloth 27B files (`qwen35.nextn_predict_layers = 1`); the recipe skips both today.
  - The layer is the target's gated full-attention layer plus a SwiGLU MLP, i.e. the **same Refs as the trunk's
    attention layers**. In front of it:
    - `fc` [H, 2H] over `concat(enorm(embed(t)), hnorm(h))` (embedding first);
    - after it, `norm` and the target's `lm_head`.
  - All norms are zero-centred (GGUF stores w + 1).
  - `h` is the target's **final-norm** hidden state, the `lm_head` input. SGLang and llama.cpp code agree on this;
    llama.cpp's comments disagree.
  - Pairing: the MTP row at position p takes (h(p), token(p+1)) and predicts token(p+2).
- **DFlash2 (z-lab Qwen3.8-27B-DFlash2):**
  - **Shape:** 5 Qwen3 layers (H 5120, 32/8 heads, head_dim 128, plain RMSNorm), sliding window 2048, non-causal;
    block 8 (anchor + 7 masks, mask id 248070).
  - **Context:** `RMSNorm(fc(concat of the residual outputs of target layers 5, 19, 33, 47, 61)))`; all five are
    GDN layers. It is injected into every draft layer as K/V only: k_proj → k_norm → RoPE, and v_proj, with no
    input norm.
  - **DFlash2 additions:**
    - a grouped dynamic conv (taps 2, group 16) around the attention and the MLP, confined to the block;
    - a selector: top-16 candidates per slot from the target `lm_head`, bilinear predecessor/successor codebooks of
      rank 256, a greedy or ancestral walk, and q over 16 candidates.
  - **Sharing:** the target's `embed` and `lm_head` are reused.
  - **Weights:** 1.92 B params (3.58 GiB bf16; Q4_K_M GGUF 1.06 GiB).
  - **Draft KV:** 20 480 B per token; only the last 2047 positions are ever read.
- **Memory on the 5080** (M0.3 §4.8): UD-IQ3_S at `max_seqs` 4 with 4 checkpoints leaves ≈ 0.8 GiB of KV
  (≈ 13K tokens).
  - MTP adds its ≈ 0.3 GiB of weights, already in the file, plus 1/17 more KV per token.
  - DFlash2 does not fit beside the 27B on 16 GB, so it targets the 4090 / 5090.
- **GDN state:** 147 MiB per sequence per copy for the 27B (SSM 144 MiB fp32, conv 2.8 MiB).

## 3. Core seams (today → change), each general

1. **Auxiliary layers.**
   - Today: `DraftHeadDecl` exists, but nothing is built.
   - Change: the core builds and state-binds a draft head's `layers` with the model, at indices after the trunk,
     and builds its `unit` (a new `DraftHeadDecl.unit: Ref` holding the head's own parameters).
   - Same mixer Ref, same state: so the MTP layer's attention joins the target's `kv` spec as one more pool segment.
     Its KV pages are therefore allocated, prefix-shared and truncated with the target's, at no extra cost.
   - Drafters that ship their own checkpoint (DFlash) get the same service through a `BuildCtx` call:
     `ctx.aux_model(spec: ModelSpec-like, tensors)` builds units, binds their state specs into their own groups (with
     planners), and loads the weights under the source's map.
   - Toy proof: a toy drafter with one paged spec of its own, prefix-hit and rolled back.
2. **Taps.**
   - Today: `ModelSpec.taps` is never read.
   - Change: `Model.trunk` calls registered tap consumers, `on_tap(name, h, ctx)`, at the declared points
     `layer:<i>` (the residual stream after layer i). The trunk's output (pre-norm) is passed to the drafter
     anyway, so there is no `final` tap.
   - Why callbacks rather than stored copies: DFlash's `fc` over five concatenated taps is a sum of five partial
     matmuls. It accumulates `Σ W_i h_i` in one [T, H] buffer instead of holding 5 × [T, H]. That is 105 MB for a
     2048-token step at H = 5120.
   - Toy proof: a toy consumer sees the same tensors as a model-only reference run.
3. **Drafter lifecycle.**
   - Today: `propose(ctx)` runs after sampled steps.
   - Change: `Drafter.absorb(ctx, h)` runs after **every** step's sample stage, prefill chunks without sampled rows
     included. It receives the trunk output `h` [T, H] and writes the drafter's own state for the step's computed
     tokens. Then `propose(ctx)` runs for the rows that sampled, as today.
   - Both run inside the decode/verify graph and eagerly otherwise.
   - For verify rows, `n_accept` is already in the SeqTable when `absorb` runs. State written for rejected rows lies
     past the new length and is overwritten later, as in SGLang and llama.cpp.
4. **Rewindables.**
   - Today: `Rewindable.commit` is never called.
   - Change: units register Rewindables at build (`ctx.rewindable(obj)`). After the verifier, inside the sample
     stage, the core calls `commit(slots, n_accept)` on each, in registration order, before `absorb`.
   - State kinds stay Rewindables too. Paged state is "trivial" (truncation by length, no device work).
   - Toy proof: a toy recurrent unit whose replay restores the decode-path state.
5. **Sampled verification.**
   - Today: greedy only (`_draft_k` requires temperature 0).
   - Change:
     - The `Sampler` protocol gains `distribution(logits, ctx, width)`, returning the filtered target
       probabilities per row (temperature → top-k → top-p → min-p, exactly as `sample` applies them), and
       `uniform(ctx, stream)`, the counter-based uniforms hashed from (seed, position, stream). Verification is
       then as batch-invariant as sampling.
     - The SeqTable gains sparse draft probabilities `draft_q_ids` / `draft_q` [S, k_max, Kq], where Kq is the
       drafter's declared candidate count (1 = deterministic draft).
     - A verifier declares `samples = True` to lift the temperature gate.
   - Toy proof: an oracle drafter with known q; the χ² test of §5.
6. **Verify graphs.**
   - Today: decode steps only (`n_query == 1`).
   - Change: steps whose rows are all verify rows of one width q = 1 + k are captured per (bucket, q, sampler
     variant, planner keys). The q set is the union of the policy's declared k values (finite, design §7.6), and
     the capture budget is measured at startup as today.
   - Scheduler: rows are padded to the step's q.
     - Today the step takes the smallest k any row allows.
     - Change: the policy's k is honoured per row, and short rows are padded with dummy drafts that are never
       accepted (`draft_tokens = -1`, and the verifier masks positions past the row's k).
     - This keeps one graph per q, not per mix.
   - Toy proof: the oracle drafter at depth 1 with graphs on (CPU: eager equivalence; GPU: graphs vs eager bounded).

The core gets no model, mixer or drafter names; the CI denylist applies. Estimated core growth: ≈ 350 lines.

## 4. Decisions

- **D1. One milestone, two gates.**
  - Gate A: the §3 seams, GDN replay, MTP, n-gram, both verifiers, graphs, the fixed-k policy, and the 5080 / 4090
    MTP evidence.
  - Gate B: DFlash2.
  - Reason: Gate A already exercises every seam DFlash needs except the auxiliary checkpoint and taps. Gate B then
    only adds a plugin.
- **D2. MTP from the target checkpoint.** It loads through the target's weight map: new rules for `mtp.*` and
  `blk.{n}.nextn.*`, and the skip is removed. The plan's "separate MTP GGUF" is superseded, because the unsloth files
  already carry `blk.64`.
  - The MTP head is a `DraftHeadDecl("mtp", unit=Ref("qwen3_5.mtp_head@1"), layers=(LayerDecl(attn, mlp),),
    shares=("embed", "lm_head", "norm"))` in the qwen3_5 recipe.
  - The head unit (fc, enorm, hnorm, final norm) lives in the **model plugin**, because the concat order and the
    norms are this family's.
  - The generic `drafters/mtp` plugin runs any declared head through a small contract: `head(token_ids, h_prev,
    ctx) → h`, then the target `lm_head`.
- **D3. MTP pairing.** We follow llama.cpp: the MTP row at **position p** pairs (token(p), h(p − 1)).
  - Why not SGLang's (h(p), token(p+1)) at p: that row depends on a token **past p**. A prefix-cached page would
    then hold MTP KV computed with the original request's next token, so a different continuation reuses a wrong
    entry. With llama.cpp's pairing every MTP KV entry depends only on tokens ≤ p, like the target's, so pages share
    safely.
  - Per step, `absorb` runs the MTP layer over every input token of the step. It needs h(start − 1), the previous
    step's last hidden, carried per slot ([S, H]).
    - At a prefix hit's first position that hidden was never computed here. That one row uses a zero hidden: it
      costs acceptance, not correctness. The cost is measured, and §7 has the payload fix if it matters.
  - Verify rows: `absorb` covers the 1 + k rows. `propose` then chains k single-row MTP forwards, starting from
    (sampled token, h at the accepted row).
  - The first chained forward rewrites the entry the next step's `absorb` would write. The pair is identical, so the
    result is the same.
  - Cost: one (1 + k)-row MTP layer forward (no `lm_head`) plus k one-row forwards, each with a full `lm_head`
    (248 320 × H). For k > 3 the head reads dominate, so the default is k = 3 (§5).
- **D4. DFlash2 as a plugin with its own checkpoint.**
  - `drafters/dflash` declares the taps `layer:5,19,33,47,61` (read from `target_layer_ids`) and builds through
    `ctx.aux_model` (§3.1).
  - Its attention is the GQA unit with two new knobs:
    - `context_kv` (writes K/V from a given hidden state without attention), new;
    - `block_causal=False` (bidirectional within the 1 + k block, causal to context; `attention.paged_verify@1`
      gains a `causal` fact).
  - Its state is a `paged@1` spec in its own group; the 2048-token window caps what attention reads, not what is
    stored (§7).
  - The conv and the selector are units in the plugin.
  - The mask row:
    - If the checkpoint ships `mask_embedding.pt`, the drafter keeps it as its **own** vector. It never overwrites
      the target's `embed[248070]` (SGLang does; that token is `<|audio_start|>`, live in the target's vocabulary).
    - Otherwise the drafter uses the target's row, as llama.cpp does.
- **D5. Sampled drafting.**
  - MTP and n-gram draft deterministically (argmax / lookup, Kq = 1). Rejection sampling with a deterministic
    proposal accepts d with probability p(d), and on rejection samples from p with d removed, renormalized. This is
    lossless and needs no q.
  - DFlash2 drafts by an ancestral walk at the request's temperature (Kq = 16, its q rows), as SGLang does.
  - One verifier, `std.spec.rejection`, handles both. Greedy rows inside it reduce to greedy-exact.
- **D6. GDN rollback by masked replay. No per-token state copies.**
  - Why: per-token checkpoints (`checkpoint_slots`) would cost (1 + k) × 147 MiB per running sequence. On a 16 GB
    card that is the whole KV budget, so the option is dropped.
  - Mechanism:
    - The verify forward writes the row's final state to a discard unit (the padding slot's), so the live state
      stays at the pre-verify value S0. This needs a contract amendment on `gated_delta_rule@1` and
      `causal_conv1d@1`: optional `out_slots` [R], the units final states are written to (default `slots`).
    - The forward also keeps the row's GDN inputs (q, k, v, a, b and the conv input x; ≈ 1 MB per token across the
      27B's 48 layers) in a static per-step buffer.
    - The GDN unit's Rewindable `commit` then re-runs the **same** `gated_delta_rule` call from S0 over the row's
      1 + k tokens, with `a = b = −1e4` at positions ≥ 1 + n_accept. This gives g = −exp(A)·softplus(−1e4) = 0 and
      β = sigmoid(−1e4) = 0, so S ← S for those tokens. The final state written is exactly the state after the
      accepted prefix.
    - The conv state is gathered from (saved window ‖ x) at the accepted end.
  - No new kernel, no host length, graph-capturable.
  - Cost: one GDN core over ≤ 1 + k tokens per row per layer.
  - The masking identity is exact in the recurrence. Kernels compute it in chunked form, so equality with decode is
    bounded, not bitwise, on GPU (L0 and §1.3).
- **D7. Policy: `draft_policy/fixed_k`** per drafter (MTP k = 3; DFlash2 k = 7 = block − 1; n-gram k = 4 on a hit, 0
  otherwise). It declares its k set, which becomes the graph q set {2, 4, 5, 8}. The Strata-style cost model is §7.
- **D8. Where speculation is on.**
  - Per deployment: a `[spec]` section in the lock or engine flags (`--drafter`, `--draft-k`).
  - Off by default until Gate A's evidence lands. Then it is on for Qwen3.5 / 3.8 with MTP wherever it measures
    faster on the agent trace.
  - Locks gain the setting through `propagate`, with approval.

## 5. Numerics, judge and predictions (fixed before measuring)

- **L0-drafter.**
  - Draft heads run on hidden states recorded from the target. MTP is compared against a pure-torch reference
    written from the HF module tree; DFlash2 against a pure-torch reference of §1.5 / §2 of the research note.
  - Bars:
    - top-1 agreement ≥ 99% over 2048 positions;
    - max |Δlogit| ≤ the GEMM-reorder bound used for the trunk.
- **Planted bugs** (`planted.py`). Each lists the check it must fail; every other check stays clean.

  | Plant | Description | Must fail |
  |---|---|---|
  | `mtp_concat_order` | hidden first | L0-drafter; acceptance collapses (< 1.2), while lossless greedy stays identical |
  | `dflash_tap_offset` | taps layer id + 1 | L0-drafter |
  | `replay_off_by_one` | replays n_accept, not 1 + n_accept | §1.3 rollback; L1.5 spec vs non-spec |
  | `rejection_keeps_draft` | residual not excluding d | L0-dist χ² |
  | `verify_writes_state` | `out_slots` ignored | §1.3; L1.5 |

  Lossless checks cannot see drafter bugs. Only L0-drafter and acceptance length can. That is why both exist.
- **L0-dist.**
  - 10⁵ samples per case over a vocabulary of 64 at fixed seeds; χ² p > 1e-3 against the target distribution.
  - Cases: deterministic d, sparse q with Kq = 4, and d outside the target's top-p.
- **Acceptance-length predictions** (agent trace, greedy). Each is checked against llama.cpp at the same draft
  settings; a gap > 15% is investigated before any speed claim.
  - MTP k = 3: 2.3–2.8 tokens/step.
  - DFlash2: 3.5–5 (z-lab reports ≈ 5–6 on chat; agent tool text is less predictable).
  - n-gram on the agent trace: 1.3–1.8 (resent histories make lookup hit often).
- **Speed predictions** (single stream, decode).
  - 5080, IQ3_S + MTP: 1.5–1.9× (a ≈ 12 ms target step against ≈ 3 ms of MTP and verify overhead).
  - 4090, Q4_K_M + DFlash2: 1.9–2.6×.
  - At the agent trace's concurrency (`max_seqs` 4) the gain shrinks. Prediction: 1.2–1.5× TPOT.
  - **Pivot:** below 1.2× single-stream for MTP on the 5080 after a perf pass → profile the MTP `lm_head`. The
    fallback is a vocabulary-restricted draft head, which is v1.
- **Memory (5080, max_seqs 4, 4 checkpoints).**
  - MTP weights 0.3 GiB plus the replay buffer: 32 rows × 1 MB = 32 MB.
  - The KV budget falls from ≈ 13K to ≈ 8K tokens.
  - If that fails the 8K agent turn, run `max_seqs` 3. The fit is recorded either way.

## 6. Steps

| # | Work | Checks | Days |
|---|---|---|---|
| 1 | Core seams §3.1–3.4 with toy plugins; drafter `absorb`; Rewindable fan-out | core tests; full suite | 3 |
| 2 | GDN replay: `out_slots` in the fla patch, the ggml-free CPU provider and the reference; the unit's Rewindable | L0 rollback (sm_86/89/120), CPU tiny bitwise | 2 |
| 3 | MTP head (qwen3_5 recipe + head unit, HF and GGUF rules) + `drafters/mtp` | L0-drafter; CPU tiny greedy identity; Qwen3.5-9B bf16 bounded (A5000) | 3 |
| 4 | `drafters/ngram`, `draft_policy/fixed_k` | Qwen3-8B bounded; acceptance recorded | 1 |
| 5 | §3.5 sampler seam + `std.spec.rejection` | L0-dist; plants | 2 |
| 6 | §3.6 verify graphs + per-row k padding; the 27B GGUF end to end | graphs vs eager bounded; depth-1 share | 2 |
| 7 | **Gate A evidence:** 5080 / 4090 MTP single stream + agent trace vs llama-server; panel code review | records; review clean | 2 |
| 8 | `ctx.aux_model` + taps (§3.1–3.2 rest), `drafters/dflash` (context KV, block attention, conv, selector), GGUF/HF loading | L0-drafter; bounded 27B (4090) | 4 |
| 9 | **Gate B evidence:** DFlash2 vs llama-server on the 4090; panel code review; plan §7.1 status; merge | records; review clean | 2 |

## 7. Deferred, deliberately

- The Strata-style cost-model policy (choose / observe). It needs acceptance telemetry from step 7 first.
- DFlash KV window reclaim: freeing pages older than 2048 positions. A paged-kind policy; v1, with sliding-window
  models.
- `checkpoint_slots`: dropped (D6).
- Tree verification and EAGLE-3: v2.
- The CPU lookup drafter.
- The MTP hidden at a prefix hit's first position (D3): carry it as a prefix-index payload (H × 2 B per block) if
  the acceptance cost of the zero-hidden row measures above 2%.
- Vocabulary-restricted draft heads: v1, if the §5 pivot fires.

## 8. Risks

- **Replay cost on the 27B.** 48 extra GDN core calls per verify step. If it costs > 15% of a verify step, fuse
  the replay into the next step's GDN call (prepend the cached tokens with outputs discarded) as a follow-up.
- **DFlash conv block-size coupling.** Position-in-block is row mod B. Our padding of short rows (§3.6) must keep
  DFlash rows at B = 8 always. DFlash declares k = 7 only.
- **`mask_embedding.pt` provenance and weight names.** These are unverified until the checkpoint is downloaded
  (step 8 begins with a header dump).
- **Sliding-window off-by-one.** HF allows a distance of ≤ 2048, SGLang and llama.cpp ≤ 2047. Follow llama.cpp,
  since its GGUF is what we load; L0-drafter at contexts > 2048 settles it.

## 9. Review findings, resolved

(none yet)
