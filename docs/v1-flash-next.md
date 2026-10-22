# v1 — FN: Qwen3.8-Flash-Next on Holon, then its expert offload compared with Strata

Status: **rev 4 (2026-10-18), the third panel round applied: minors only, no majors.**
- **Round 1 (rev 1 → rev 2):**
  - Panel A (architecture): 1 blocker, 7 majors, 9 minors.
  - Panel B (correctness): 2 blockers, 8 majors, 12 minors.
  - Panel C (performance and fairness): 1 blocker, 8 majors, 10 minors.
- **Round 2 (rev 2 → rev 3):**
  - Panel A: 1 major, 3 minors.
  - Panel B: 2 majors, 7 minors.
  - Panel C: 1 major, 7 minors.
- **Round 3 (rev 3 → rev 4):** A 3, B 3, C 1 minor + 2 nits.

Every finding was checked against the code; the disposition of each is in §12. The lead's decisions for both rounds
are applied; §11 records them.

Parameters marked "est." freeze when step 0's evidence lands (§8).

**Scope.** Plan §10 ("Expert caching/offload", milestone FN) and `docs/v1-offload.md`:
- D2: the target;
- §1 acceptance 4: the program bars;
- §4.7: FN's core needs;
- plan §10's two FN follow-ups: aux-model (MTP) experts through the offload store, and tune's interleaved F5b
  benchmark.

**Builds on:**
- milestone X as merged (`b5b1e7f`: offload store, ABI 1.3);
- M0.3 (Qwen3.8's GDN, gated attention, q8_0 KV);
- M0.4 (MTP as an aux model).

**Depends on the MoE prefill track** (`docs/v1-moe-prefill.md`):
- That track owns `LinearExperts`' prefill dispatch and `kernels.ggml_cuda`'s `vendor.patch`.
- It lands first, with its own lock propagation, before FN's step 4.
- FN rebases onto it and uses its D1–D5, including D4, which fuses SwiGLU into gate/up's mmq store.
- At FN's top-10 shape the track's `pass_rows` is **3,072**: ⌊32,768 ÷ 10⌋ rounded down to whole 512-row `CHUNK`s
  (track doc §5's FN rows).
- **What FN needs from it** (the lead has sent these to that track; FN refers to them by these names):
  - **(T1)** a workspace / output argument on `linear_experts@2` and `linear_experts_arena@1`;
  - **(T2)** contiguous-tail lending from one class arena, with the evicted slots returning empty;
  - **(T3)** the `pass_rows` seam;
  - **(T4)** the IQ1_M mmq hunk (D1′), with its `MMQ_Q8_1_DS_LAYOUT_D4` case and both vec-dot instances, MMA and
    dp4a;
  - **(T5)** Q2_0 in `kernels.ggml_cuda`'s `TYPES`;
  - FN's types and the E = 512, top-10 shape in its L0.

**References** (local clones; FN cites `file:line` from them):

| Abbreviation | Reference | Version |
|---|---|---|
| **St** | Strata | `296b59d` (2026-10-14), MIT |
| **FT** | FreeToken, `python/freetoken/` | `21d9578` (2026-10-16), Apache-2.0 |
| **LC** | llama.cpp, `src/` | `27c7a76` (2026-10-17) |
| **HFm** / **HFx** | HF transformers 5.19.0, `models/qwen4_exp/modeling_qwen4_exp.py` / `modular_qwen4_exp.py` | — |
| **SG** | sglang 0.5.21, `srt/models/qwen4_exp_mtp.py` | — |

"R §3.1" is X's study of Strata's tiers, `docs/brainstorm/v1-offload-research.md` §3.1.

## 0. The model

From `config.json` (`Qwen4ExpForConditionalGeneration`, text model `qwen4_exp_text`) and the GGUF headers (§0.1).

| Part | Fact | Source |
|---|---|---|
| Trunk | 48 layers, hidden H = 2560, vocab 248,320, untied head, **no input / post / final norms** | HFm:1276-1301, 1485; LC qwen4exp.cpp:511-521 |
| Mixers | layers 3, 7, …, 47 (12) are **QSA**; the other 36 are **GDN** | HFx:182-192; St `geometry.hpp:26, 75-77` |
| QSA attention | 24 q / 2 kv heads, head_dim 256; `q_proj` per head = `[q(256) ‖ gate(256)]`; sigmoid output gate; zero-centred q/k norms; NeoX RoPE on the first 64 dims, θ = 1e7 (the interleaved mRoPE [11,11,10] is plain 1-D RoPE for text) | HFm:811-893, 77-142 |
| QSA indexer | `index_qk_proj` → q: 4 × 128, zero-centred `q_layernorm`, RoPE at the query position; raw k: 1 × 128, cached unnormed and unrotated | HFm:665-771 |
| QSA selection | Blocks are complete 4-token groups of the visible tokens. `kbar_b = rope(k_layernorm(bf16(mean_fp32(k[4b..4b+3]))), pos 4b)`; `s_b = Σ_h relu(⟨q_h, kbar_b⟩) / √128`. Per query: top min(512, n_blocks) blocks ∪ the open tail block (`scores.topk`, tie order unspecified, HFm:749). Exactly dense while p + 1 ≤ 2,051. | HFm:720-760; FT attention.py:173-174 |
| GDN | Qwen3.8's (16 k / 48 v heads × 128, conv 4, fp32 state). The gated norm uses **`sigmoid(z)`** where Qwen3.5 / 3.8 use `silu(z)`; this is the only difference | HFx:313-327; LC qwen4exp.cpp:639-643; St `layer.hpp:8-17` |
| Hyper-connections | residual R = [T, 4 · 2560], initialised as the embedding repeated 4×. Two `GatedResidual`s per layer (attention, FFN) and an output mixer (see below). No stream-to-stream matrix. | HFm:995-1023, 1472 |
| MoE | router [512, 2560], fp32 softmax over 512, top-10, renormalised; SwiGLU experts I = 640. Shared expert: SwiGLU I = 640 × `sigmoid(x · w_shexp_gate)`, one scalar per token. `out = routed + gated shared`. Input is the HC-mixed x, with no pre-norm. | HFm:952-992; St `shared_expert.hpp:3-8` |
| PLE n-gram | Once, at 0-based layer 1 (`ple_layer_ids` [2] is 1-indexed, HFm:1260; GGUF `ple.layers` [1]). Table, hash and block: below. | HFm:1104-1247; HFx:576-632; LC qwen4exp.cpp:1216-1459; St `ple.hpp:3-13`, `ple.cu:123-163` |
| MTP | 1 layer, outside HF (HFm:1314, 1590 ignore `mtp.*`). See below. | LC qwen4exp.cpp:289-301, 501, 524-616; SG qwen4_exp_mtp.py:72-114; St `generate.cpp:683-685` |

**Hyper-connections** (HFm:995-1023):
- `Rn = grouped zero-centred RMSNorm(R)`, statistics per 2560-wide stream.
- **mix:** `x = mean_s(sigmoid(W_up · silu(W_down · Rn / 4))_s ⊙ Rn_s)`.
- **inject:** `w = 2 · sigmoid(W_inj · Rn / 4)`.
- **combine:** `R_s += w_s · y`. This adds to the un-normed R.
- Weights: `W_down` [320, 10240], `W_up` [10240, 320], `W_inj` [4, 10240].
- The output mixer is the mix without the inject.

**PLE** (HFm:1104-1247):
- **Table:** 16 hash heads (8 bigram, 8 trigram) × 160, giving 320,001,536 × 160 rows. As IQ4_NL that is 90 B per row, 28.80 GB in total.
- **Hash:** `(t0·m0) ^ (t1·m1) [^ (t2·m2)] mod vocab_h + offset_h`.
  - The window is cut at EOS 248044.
  - A fresh sequence's previous ids are EOS (HFm:1130).
- **Block:**
  - `K = norm_key(W_k E)`, `V = W_v E`, `Q = norm_query(R)`.
  - `u_s = ⟨K_s, Q_s⟩ / √2560`.
  - `g_s = sigmoid(sign(u_s) · √max(|u_s|, 1e-6))` (HFm:1239; St `ple.cu:123-126`).
  - `G_s = g_s · V`.
  - `R += G + silu(conv(norm_conv(G)))`.
  - The conv is depthwise with kernel 4 and **dilation 3**: taps t−9, t−6, t−3, t.

**MTP:**
- `R_s = fc_e(enorm(e)) + fc_h(hnorm(h))_s`, where h is the trunk's final R before the output mixer (LC
  qwen4exp.cpp:501).
- Then one QSA + MoE (its own 512 experts) + shared-expert HC layer, its output mixer, and the shared `lm_head`. No
  PLE.
- **hnorm has two published forms:**
  - LC: per stream (qwen4exp.cpp:296-297, 581).
  - SG and Strata's default: **pooled**, one RMS over all 10240 features (SG qwen4_exp_mtp.py:76-83; St
    `generate.cpp:683-685`, `--mtp-hnorm stream` opt-in).
  - FN pins pooled (D9).

### 0.1 Files

GGUF headers were parsed by HTTP range; the parsed offsets sum to each shard's size exactly.

| File (2 shards) | Total | Routed experts | PLE table | Rest | Per expert per layer | Expert types (layers) |
|---|---|---|---|---|---|---|
| ISTA-DASLab GSQ-RCO **IQ2_XS** | 68.03 GB | 35.45 | 28.80 | 3.76 | 1.443 MB | gate = up: IQ2_S 34, IQ2_XXS 11, **IQ1_M 3** (layers 8, 13, 37); down: **Q2_0** 48 |
| ISTA-DASLab GSQ-RCO **IQ3_S** | 83.61 GB | 50.29 | 28.80 | 4.51 | 2.047 MB | gate = up: IQ2_S 20, IQ3_XXS 17, IQ3_S 10, IQ4_XS 1; down: IQ4_NL 39, **Q2_0** 9 |

- **MTP is not in these files** (1,224 tensors, no `nextn`). Strata's MTP pack is a GGUF of its own:
  - routed experts in Q2_0, with a per-block scale that minimises squared error (not ggml's reference rounding);
  - dense parts in **BF16**;
  - HF names, fused `gate_up_proj`.
  - 0.89 GB in total (St `tools/mtp_pack.py:9-16`; `setup.py:6537-6545`).
- **Q2_0 is upstream ggml type 42** (64 values in 18 B, value `(c − 1)·d`):
  - The vendored `cbe28bb3` has it in `ggml.h:432`, mmvq (`mmvq.cu:43`), mmq (`mmq.cuh:569-572`), dequantize
    (`dequantize.cuh:26-42`) and ggml-cpu's `quants.c`.
  - It is missing from Holon's `TYPES` tables (`kernels/ggml_cuda/plugin.py:83-85`, `kernels/ggml_cpu/plugin.py:22-23`)
    and from PyPI gguf-py 0.19.0, whose enum stops at Q1_0 = 41. So `weights/gguf/reader.py:82` and the judge's
    `GGUFReader` fail on these files.
  - LC `27c7a76`'s own gguf-py has it (`constants.py:6202, 6419`; `quants.py:679`).
- **IQ1_M** has an mmvq kernel (`mmvq.cu:61`) but no mmq loader (no template instance, no case in `mmq.cuh`). Strata
  falls back to FP16 for it, 16 experts per launch (St `prefill.cpp:997-1000`). For FN's choice see D1′.
- Both files are **split**, and Holon refuses split files (`std/holon_std/weights/gguf/hub.py:44`). Shard 2 is
  exactly the PLE table.
- The tokenizer's pre-tokenizer is `qwen35`, which Holon already has (`weights/gguf/tokenizer.py:21`).

## 1. Goal and acceptance

**Goal.** Serve Qwen3.8-Flash-Next from its GGUF files, text-only:
- exact against an independent fp32 oracle;
- all-resident on a 96 GB card;
- through X's offload store on 16 GB and 32 GB cards;
- then meet the program bars against Strata.

1. **Model:** `models.qwen4_exp`, `derived_from` `models.qwen3_5`, serving `arch.gguf.qwen4exp@1` and
   `arch.Qwen4ExpForConditionalGeneration@1`.
   - **(a) Tiny model on the CPU.** A random-weight reduced config (§6.1), with every norm, conv, `A_log` and
     `dt_bias` randomised, written by LC's converter, matches HF.
     - Logits within 1e-4.
     - Identical QSA token sets per (layer, position), with lower-index ties.
     - A prefix hit is bitwise equal to recomputing.
     - Holon's MTP head, loaded from a Strata-format MTP pack the tool writes, matches the SG pooled reference within
       1e-4.
   - **(b) Full model** (owner decision 2026-10-21, §18: full-model L2 Tier B against fp32 goldens and bf16 floors is
     dropped for FN; step 5's L2 results stay as history, §17). All-resident on the RTX PRO 6000, IQ2_XS and IQ3_S:
     - **Self-consistency, no golden:** Holon's paths against each other on long prompts — the decode path (1-token
       steps) vs prefill, chunked vs one-shot, MTP vs plain — per-token KL and NLL differences, with a bar near the
       bf16 noise between correct paths (§18).
     - **Task accuracy:** a fixed set of 60–100 public or synthetic items with checkable answers (multi-step math,
       code run against tests, reasoning and multiple choice, needle / recall at 8K and 32K, multi-turn), Holon and
       Strata greedy on the same file: Holon within a tolerance stated before the run, and no gibberish answer; the
       answers saved.
     - **A sample dump:** Holon's answers to a handful of hard questions, for a human look.
2. **Bounded and run-identical** (full model, RTX PRO 6000):
   - chunked vs one-shot prefill (chunk 250: straddles 4-token blocks and the PLE window, and is not a multiple of 4);
   - prefix-hit resume vs recompute, with a 4K system prompt from `v1-long` (sparse decode after a restore);
   - graphs vs eager, overlap 1 vs 0;
   - snapshot run-identical ×3, with a ≥ 8K-token request and MTP on;
   - the MTP verify path bounded (L2);
   - accepted tokens per verify pass ≥ 2.2 at 3 drafts on the chat and agent prompts. The count is Strata's: accepted
     drafts + 1.
   - **Absolute decode check:** bounded's `divergence` (decode rescored by a prefill reference) is measured on the
     long trace twice:
     - **sparse:** the model as built;
     - **dense control:** the same trace under a test-only forced-dense QSA mode (identity selection in prefill and
       decode alike).
     - Gate: KL p99 sparse ≤ max(2 × dense, dense + 2.5 SE).
     - **An MTP-on sparse arm** runs the same gate against the same dense control, so the sparse verify kernel gets an
       absolute check too.
     - **Pre-registered on the RTX PRO 6000 before any lock:** the unplanted build passes, and
       `qsa_decode_select_shift` fails, in both arms. If the plant passes, the statistic is tightened before any
       lock: first p95, then a per-position paired KL difference. That difference pairs positions only when both
       modes are scored teacher-forced on the same tokens (one trace's ids, rescored by each mode), not on two free
       decodes, whose tokens diverge.
   - **GPU unit-level bitwise check of `qwen4_exp.ple_block@1`:** chunked vs one-shot, and replay-rewind vs k decode
     steps. The fp32 conv is a per-channel 4-tap sum with no cross-row reduction, so both pairs must match bitwise.
3. **Kernels.** L0 for every new kernel against its `reference.py` on sm_86, sm_89 and sm_120, memcheck clean.
   - Decode-step target on the RTX 5080: `qsa_select` ≤ 20 µs per layer at 32K context (8K blocks) for 4 query rows of
     one sequence; 8 rows are reported, not gated (panel C, step 4).
   - ≤ 768 kernel launches per decode token, all graphed (48 layers × 16: a per-token average, so layers may differ),
     is a step-7 gate (§14's launch table).
4. **Offload exactness.**
   - On the RTX PRO 6000, the offload store at a capped fit, with `lfu` and the MTP experts in the store, is
     snapshot byte-identical to all-resident.
   - On the 5080 and 5090: cross-placement identity ×3.
   - The PLE RAM and SSD tiers are byte-identical.
   - F5b, where it is used, is bounded against the path-aware floor.
5. **Program bars** (`docs/v1-offload.md` §1.4; protocol D11; statistics D11):

   | Against | Card, file | Decode | Prefill |
   |---|---|---|---|
   | Strata, no approximation | RTX 5080 IQ2_XS; RTX 5090 IQ3_S | ≥ 1.0× | ≥ 0.9× |
   | Strata defaults vs Holon + A1 (**reported, not gated**: owner decision 2026-10-20) | same | 1.0× | 0.9× |

   - Each row is gated twice: drafting off, and MTP matched. Contexts are 1K / 4K / 32K.
   - Holon runs **its lock's configuration**. F5b rows that differ from the lock are reported separately.
   - Holon exact vs Strata's defaults is reported, not gated.
   - **The gated rows are the no-approx and MTP-matched ones** (owner decision, 2026-10-20). The Strata-defaults rows
     are reported against their 1.0× / 0.9× marks, not gated.
   - FreeToken (`models/qwen4_exp`) and llama.cpp (`qwen4exp`) are reported only: one arm each, 3 rounds.
   - Plan §9's kill line (< 0.8×) applies to the like-for-like rows.
6. **Memory** (§7):
   - **Fits:** RTX 5080 IQ2_XS at 32K with MTP; RTX 5090 IQ3_S at 32K with MTP.
   - The step size is a lock setting **per card**: one deployment at `--ctx 33792` serves the 1K, 4K and 32K rows.
   - The measured fit is reported against §7.
   - **Pinned host bytes:** every one is accounted against `host_memory`.
7. **Locks:** `rtx-5080` (IQ2_XS, offload, Strata's MTP pack) and `rtx-5090` (IQ3_S, the same), written by
   `tune --new-model`.
   - Existing locks are propagated once, by whichever of FN and K closes second (D16).
8. **Core:** C1–C3 (§4) only, additive.
   - ABI **1.4 or 1.5 by merge order with K**: whichever lands its seams first takes 1.4.
   - ≤ +47 code lines relative to FN's step-1 base, toy-proven.

**Non-goals:**
- the vision tower;
- KV streaming for QSA (K4: owned by FN as its first follow-up after the close, in plan §10);
- Strata's suffix drafter and adaptive draft depth;
- NVFP4 packs, Q8_0 / BF16 PLE tables, the Coder variant;
- multi-GPU;
- Qwen3.6-35B-A3B: optional later evidence for `std.ffn.sum` and `gated_swiglu` (D13);
- the MoE prefill speed work itself (the MoE prefill track, D1′).

## 2. Decisions

**D1 — Files: ISTA GSQ-RCO IQ2_XS on the RTX 5080 and IQ3_S on the RTX 5090.**
- These are the bars' files and Strata's recommended family (St `setup.py:243-309, 383-389`), so both engines read
  identical bytes.
- **Q2_0:**
  - add `q2_0` to `kernels.ggml_cuda`'s `TYPES` (the MoE prefill track, T5) and to `kernels.ggml_cpu`'s (FN);
  - add `q2_0` to the judge's `Q8_0_DOT`: ggml-cpu dots Q2_0 against Q8_0 activations, so the CPU reference and F5b's
    floor need it (`std/holon_std/moe/reference.py:266`);
  - move the workspace's gguf-py to LC `27c7a76`'s tree. This is one source for std, the judge and the golden. Its
    relock is covered by FN's single propagation (D16).
  - That tree also calls itself 0.19.0, so `file_meta`'s `version("gguf")` cannot tell it from PyPI's
    (`judge/holon_judge/gguf.py:375-388`). `file_meta` therefore records the llama.cpp commit and the sha256 of
    gguf-py's `quants.py`.
  - For split files, `file_meta` hashes every shard.

**D1′ — IQ1_M is a native mmq tile loader, delivered in the MoE prefill track's `vendor.patch`, so it stays exact.**
- **The encoding** (checked against the vendored code):
  - IQ1_M stores grid values in {−1, 0, 1}, a delta of ±1/8 per group of 8 (bit `0x08` of `qh`), and 3-bit scales
    per 16 values (`vecdotq.cuh:1286-1326`).
  - Scaled by 8, a weight is `8·g ± 1` ∈ {−9, −7, −1, 1, 7, 9}: int8.
  - Its scale per 16 values is `d · (2s + 1) / 8`.
- **The loader:** that is the layout of the `q8_0_16` mmq path that IQ2_XS and IQ2_S already use, with fp32 scales
  per 16 values (`mmq-load-tiles.cuh:1214-1218`). An `iq1_m` loader writes int8 `8·(g + δ)` and per-16 float scales,
  and reuses that path's vec dots: dp4a, as in `mmq.cuh:653`, and MMA, as in `:786`.
  - The hunk also needs an `IQ1_M → MMQ_Q8_1_DS_LAYOUT_D4` case in `mmq_get_q8_1_ds_layout`, whose default case
    aborts (`mmq.cuh:60-100`). With it, the judge's default D4 activation blocking matches.
  - The weights are dequantized exactly, and activations stay q8_1, so Tier B's floor needs no special case.
- **Ownership:** the MoE prefill track owns and lands this hunk (T4) before FN's step 4. FN supplies its L0 cases.
- **Fallback, only if the loader fails L0:** dequantize to bf16 into grouped scratch (16 experts per launch, as Strata
  does), behind the track's `pass_rows` seam (T3). Its floor casts those rows' weights to bf16, and its scratch is
  counted in §7.
- **FN's requirements on the MoE prefill track's L0 and step 0:**
  - FN's types: IQ2_S, IQ2_XXS, IQ1_M, Q2_0, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS;
  - FN's shape: E = 512, top-10, H = 2560, I = 640.
  - FN's "seed of the speed item" claim (rev 1) is struck.

**D2 — The oracle: HF transformers' `qwen4_exp` at 5.19.0, in a golden-only venv. The full model streams one layer
at a time through the judge's existing `_streamed`.**
- **The venv** (`local/envs/golden-qwen4`):
  - transformers 5.19.0 and tokenizers ≥ 0.23.1;
  - `-e judge -e std` with `--no-deps`, or with uv overrides of std's `transformers==5.6.2` pin
    (`std/pyproject.toml:8`; `golden.py:19` imports `holon_std`);
  - `--link-mode copy`.
  - The workspace keeps 5.6.2.
- **Documented deviation from HF:** the fixture tool and the golden driver wrap the indexer's `topk` with a stable
  descending sort, so lower blocks win ties.
  - The wrapper is scoped to the indexer module's call (a patched method on its class instance), not to a global
    `torch.topk` / `Tensor.topk`, which the router and `topk_rows` also use.
  - Why: ReLU sums make exact 0.0 ties common. Panel B found 5–11 boundary ties per 100-token tiny prompt, and HF's
    `topk` broke 4–11 of them differently.
  - The rule is recorded in every golden's metadata as a deviation from HF.
- **The streamed driver already exists:** `_streamed` (`judge/holon_judge/golden.py:229-266`). FN's judge edits:
  - detect MoE by the presence of a `*SparseMoeBlock`, not by `model_type.endswith("_moe")` (`golden.py:293-294`);
  - add the shared expert and its gate to the floor's mirror, q8_1-emulated (today `moe.py:34-46` drops them);
  - a final-norm hook, because Qwen4Exp's head is `hyper_connection_mixer`, not `model.norm` (`golden.py:264`);
  - the head computed in row chunks (a 32K × 248,320 fp32 logits tensor is 32.5 GB);
  - a lazy PLE row gather in place of `ngram_embedding`. It dequantizes only the hashed rows from the GGUF (the full
    table in fp32 is 205 GB), and asserts that the GGUF's hash constants equal HF's recomputed buffers;
  - split-file reading;
  - the tie rule above.
- **The floor:** the same driver in bf16 on the RTX PRO 6000 with `--q8-1 mmq|mmvq`, one layer resident at a time.
- **HF's own attention at 32K:** N4 runs one 32K prompt under `MemoryMax`, recording time and peak RSS.
  - If the attention materialises [24, T, T], the driver calls the layer per query block, through HF's cache (a
    `DynamicCache` per prompt per layer), so GDN, PLE and indexer state carry across blocks.
  - That mode is not bitwise equal to one-shot (≈ 5e-8 measured on the tiny model), so the golden's metadata records
    the mode.
- **Tiny fixtures:**
  - `testing/holon_testing/qwen4exp_files.py`, run in the golden venv, builds the random model.
  - It randomises every `Qwen4ExpTextRMSNorm`, the PLE conv, `A_log` and `dt_bias`, as `qwen3_5_files.py:36-37`
    does. HF's init zeroes the PLE conv (HFm:1336, 1352), and the tool asserts these weights are nonzero.
  - The vocab is 248,320, as the converter requires.
  - LC's converter at `27c7a76` writes the GGUF, an independent writer for the conventions: norm + 1, `ssm_a`, the
    conv layout, the HC reshapes.
  - At vocab 248,320, full logits for one prompt would be ≈ 123 MB and the f32 GGUF ≈ 130 MB. So:
    - **Committed:** HF's output-mixer x [T, 64], the selected token sets and the PLE row ids. The test recomputes
      the logits with the GGUF's `lm_head`.
    - **Not committed:** the tiny GGUF. A pinned script rebuilds it (`python -m holon_testing.qwen4exp_files` in the
      golden venv with LC `27c7a76`) and checks its sha256. CI skips only the GGUF-dependent checks when LC's
      converter is absent, as `qwen3_5_files.py:86-87` does. **Deviation from rev 3** ("CI fetches the artifact";
      lead decision, step 0): no host for the artifact exists.
  - **The MTP pack:** the tool also writes a Strata-format MTP pack for the tiny model.
    - First it writes the tiny model's `mtp.*` tensors in `tools/mtp_fetch.py`'s layout: one raw bf16 file per tensor
      plus `mtp-manifest.json` (St `tools/mtp_fetch.py:1-12`).
    - Then St `tools/mtp_pack.py` packs them unchanged: HF names, fused `gate_up_proj`, BF16 dense parts, F32 norms,
      MSE-scaled Q2_0.
    - Holon's MTP rules load this file on the real model, so the pack format gets its own oracle (§6.1).
  - **Tiny MoE sizes:** `moe_intermediate_size` and the shared expert's are 64, a multiple of Q2_0's 64-value block,
    so the pack's Q2_0 experts are well formed.
- **MTP oracle:**
  - The pooled form: a reference transcribed from SG's module. It reads HF names and applies 1 + w itself, run on the
    pack's own dequantized tensors.
  - The stream form: LC's MTP graph on the tiny GGUF.
  - On the real model, MTP is judged by the verify path's L2 and by acceptance.

**D3 — Hyper-connections: the trunk carries R flattened to [T, n·H] in fp32. A wrapper embed expands it; the model's
`norm` is the output mixer. C1 states R's width and dtype for drafters and the profile.**
- **2-D:** every core site indexes rows on dim 0 (`executor.py:232-233, 426-427, 615`).
- **A core fact:** MTP needs R, and the profile hands drafters `zeros(T, spec.hidden, cfg.dtype)` (`executor.py:266`).
  The profile now also asserts that the trunk's real output matches the declared fact, so a mis-declared recipe fails
  at startup.
- **fp32:** R accumulates 96 scaled updates. Strata and llama.cpp keep it fp32. It is 40 KB per token.
- **Rejected:**
  - expand in block 0 and collapse in the last block: MTP needs R;
  - a 3-D h: no gain.

**D4 — PLE lives in the gated-residual block of layer 1, with recurrent per-request state declared through the block
(C2, C3).**
- **State per request:**
  - the conv history: 9 = (4 − 1)·3 rows × 10240 in **fp32**, 368 KB. The rows are an fp32 norm output, so storing
    them in bf16 would make chunked vs one-shot, hit vs recompute, and verify vs decode differ;
  - the last two token ids (EOS for a fresh slot).
- **The spec:** a `Slot` spec of kind `recurrent@1`.
  - `alignment` and `in_step` are Ref parameters the recipe sets to GDN's 64. A std unit cannot know GDN's chunk.
  - The kind checkpoints at `lcm(alignment, page_size)` (`std/holon_std/state/recurrent/kind.py:26, 54`), and a
    prefix hit is the fixed point over every kind (`core/holon/core/state.py:190-201`). With matching alignments the
    PLE payloads sit on GDN's blocks, so hits are not silently zero.
  - `qwen4_exp.ple_block@1` writes its in-step payloads (the 9 conv rows before each listed position) the way GDN's conv does.
- **Verify:** GDN's way — `draft_slots` 0, then replay.
  - The unit saves the window and the step's conv inputs.
  - Its `rewind` writes the window and the last two ids at the accepted end. The core's `rewind_modules` visits every
    submodule (`core/holon/api/state.py:135-140`).
- **Rejected:**
  - a GDN wrapper mixer: it cannot see R, and two mixers in group `recurrent` are refused (`assembly.py:155-157`);
  - recomputing the conv window from tokens: it needs past R.

**D5 — The PLE table is a std row table, `std.tiered.rows@1`. RAM is the default; SSD is the fallback.**
- **RAM tier:**
  - a mapped host pool (S2) loaded from shard 2's extents (S4) via `tiered/files.py`;
  - the GPU gathers 16 rows × 90 B per token zero-copy and dequantizes them in-kernel.
- **SSD tier:**
  - Prefill rows are read ahead by a shared object's `before_launch` (S1) with O_DIRECT 4 KiB page reads, deduplicated
    and sorted, double-buffered. `before_launch` sees only the plan (`hooks.py:172`, `BatchPlan.host_tokens`), so
    after a prefix hit the first two positions of a chunk cannot be hashed on the host. Those rows take the in-graph
    read path.
  - Decode rows are read in-graph through X's `cpu_task@1` copy contract (`std/holon_std/moe/kernels.py:150-155`).
  - No row cache in rev 2; step 6 measures whether one pays.
- **Pairing in D11:** the SSD tier runs against Strata's `--ple-io direct` when a desktop cannot hold the RAM tier.
- **Reuse:** S2, `files.read` and `cpu_task`. No new core seam.

**D6 — QSA is `std.mixers.qsa@1` (`derived_from` gqa), with two paged specs in separate groups.**
- **The specs:**
  - `"kv"` is gqa's layout and its formats.
  - `"idx"` holds, per page, the raw index keys (page × 128 bf16) and the pooled keys (page / 4 × 128 fp32).
  - `page_size % 4 == 0` is required.
  - Two groups, because the core refuses a page group shared by paged specs (`state.py:149-150`). This corrects
    `docs/v1-offload.md` §4.7.
- **Pooling:**
  - A block is pooled when its 4th token is appended, reading the **stored** raw keys, never in-register values.
  - A block completed by rejected drafts is rewritten when its positions are appended again.
- **Selection:**
  - fp32 scores with IEEE products, no TF32 (step 4: summed within a thread, as `tl.dot(ieee)` was slower); an exact
    top-k, ties going to the lower block (D2's rule); shape-invariant.
  - The score workspace is sub-batched: 512 queries per pass at most (St `sel_batch`, `prefill.cpp:993`).
- **Dense and sparse rule:** a step runs the dense gqa kernels only when every one of its rows has a host-known
  length ≤ 2,051. Otherwise it runs the sparse kernels, and each query whose own length is ≤ 2,051 selects all its
  blocks. Verify rows count by their upper bound.
  - **Consequence:** a short request's bits depend on whether a long request shares its step. Holon does not promise
    batch invariance. The snapshot fixes the schedule, so run-identity holds.
  - **Rejected:** choosing per row, with dense and sparse launches over row subsets. The subsets vary per step, which
    breaks fixed-shape graph capture.
- **Raw keys for every token cost 96 MB at 32K.** They are kept on purpose: rewind and prefix hits stay stateless.
  Strata keeps only the open block's tail (B-m11, §12).

**D7 — The shared expert composes beside `std.moe@1`: `std.ffn.sum@1(parts)` and `std.ffn.gated_swiglu@1`, routed
MoE first.**
- The router-prepared q8_1 is consumed inside `std.moe@1`'s own call (`moe/units.py:247-254, 273-289`). An L0 test
  asserts that FN's layer never re-quantizes (the generation check, `kernels/ggml_cuda/plugin.py:434-450`).
- `std.moe@1` gains one additive option, `fused_gate_up` (default off; step 3 review): the MTP pack stores each
  expert's gate and up rows as one tensor, which the core's weight rules cannot split. Nothing else in it changes.

**D8 — GDN gains `out_gate` (`silu` | `sigmoid`).**
- It is passed to the unit and the kernel facts only when not the default, so the Qwen3.8 locks' recorded facts do
  not move.
- A provider that does not know the fact fails loudly (`gdn_triton/plugin.py:20`).

**D9 — MTP is an aux model: `std.spec.mtp_hc_head@1`, with Strata's own MTP pack, pooled hnorm, full-vocab drafts.**
- **The head:**
  - `fc_e(enorm(e)) + fc_h(hnorm_pooled(R))_s` per stream;
  - one gated-residual layer (QSA in the aux namespace, `std.ffn.sum@1`);
  - `head.out(x)`, its output mixer, before the target's `lm_head`.
- **The drafter:**
  - `drafters/mtp` reads `head.carries == "residual"` and carries `spec.residual`'s width and dtype instead of
    `norm(h)`.
  - It calls `lm_head(head.out(x))`. Today it calls `lm_head(x)` (`drafters/mtp/plugin.py:70-74`).
- **Weights:** Strata's `mtp/mtp-q2_0.gguf` for both engines (HF names, fused `gate_up_proj`, BF16 dense parts), via a
  name and layout map in the model plugin's MTP rules.
  - Both locks name this file.
  - Its sha256 and producing commands are recorded in the bundle and the evidence: the Strata commit, and
    `tools/mtp_fetch.py` / `tools/mtp_pack.py` with their arguments and source checkpoint revision. A lock can then
    be reproduced from Strata's MIT tools alone, without installing Strata.
  - Porting the MSE-scale quantizer into Holon is deferred.
  - Unsloth's Q8_0 MTP (2.78 GB) would be a different deployment.
- **hnorm is pinned to pooled:** Strata's default and SG's. The `stream` parameter is kept only for LC's cross-check.
- **Draft vocabulary:** full on Holon. Strata's matched rows run `STRATA_MTP_FULL_HEAD=1` (St `generate.cpp:3502`);
  its subset head stays on in its defaults row as a speed feature, since drafts never change the output.
- **MTP experts:** through the offload store (D12a), or resident if tune measures that faster.

**D10 — The kernels are Triton, in new kernel plugins (`kernels.qsa_triton`, `kernels.hc_triton`, `kernels.rows_io`).**
- They are adapted from FreeToken (Apache-2.0, with notices); Strata's CUDA (MIT) is the performance reference.
- **Precision.**
  - Selection scores are `input_precision="ieee"`, always.
  - The HC projections default to `ieee`. In prefill that runs fp32 on CUDA cores: ≈ 5.2 TFLOP per 4K chunk, ≈ 5%
    of Strata's 5080 prefill time (est.).
  - `tf32x3` is allowed for the HC projections only if their L0 stays within the fp32 budget and L2 holds. The choice
    is a kernel fact, so the lock records it.
- The PLE kernels belong to the model plugin (D13′).
- A kernel moves to CUDA in step 7 (perf closure) if it is above 5% of a step and a prototype gains ≥ 10%.

**D11 — The comparison protocol.**
- **The harness:** `bench/offload_ctx.py` gains a Strata backend and a Holon backend.
  - Strata's server is chat-only (`serve/server.py:6`), so the Strata backend speaks the engine's `--serve` protocol
    with raw token ids, as `serve/server.py` does after tokenizing. One-shot `--tokens-file` runs
    (`generate.cpp:704-705`) are a cross-check.
  - Every engine is timed from the client with offload_ctx's formulas (C / prompt time; (n − 1) / (last − first)),
    in fresh processes, one engine resident at a time.
  - Strata runs with `--max-context 33792`.
- **EOS:**
  - In serve mode Strata forces `stop_eos` but keeps a configurable list (`generate.cpp:2525-2531`), and `--eos-ids`
    replaces that list (`:2187-2190`).
  - N0 tries `--eos-ids` with an id that never occurs (one outside the vocab, if the parser accepts it; else an unused
    added-token id) as Strata's ignore-EOS. Holon runs `ignore_eos`.
  - If that fails, the spans are pre-screened instead: a span where either engine emits EOS within 256 tokens is
    dropped for every engine before the gated rounds.
- **Prompt corpus:**
  - The text of `v1-long.jsonl`, `v1-qwen3_5.jsonl` and `v1.jsonl`, concatenated in that order and tokenized with the
    model's tokenizer. Est. ≈ 200K tokens; `v1.jsonl`'s ids are Qwen3's, so its text is re-tokenized.
  - **Context-specific offsets.** For context C ∈ {1K, 4K, 32K}, with index c = 0, 1, 2, the candidate starts are
    o_{C,k} = (k × ⌊N / 6⌋ + c × ⌊N / 18⌋) mod N, for k = 0…5 (N is the corpus length).
    - No two contexts share a start.
    - At 32K the candidates overlap by less than half when N ≥ 200K.
  - **The warm-up span** is the corpus's last 2,048 tokens. Starts are shifted forward so that none lies inside it.
  - **Exactly 3 spans per context** are used: the first 3 candidates, or the first 3 to survive EOS screening if
    `--eos-ids` fails.
  - **No prefix reuse.** Both engines run with prefix reuse as deployed. Every timed request must log 0 reused
    tokens (Holon's `cached_tokens`; Strata's prompt-cache log, `--prompt-cache`, St `generate.cpp:784`). A request
    that reuses any token is invalid, and the cell's span is replaced by the next candidate.
    - If either engine's logs cannot show reuse, prefix reuse is turned off in both: Holon's prefix cache off, Strata
      `--prompt-cache 0`.
  - The ids, offsets and the corpus's sha256 go in the evidence.
- **Strata, no approximation:**
  - `STRATA_ROUTE_TAIL_SKIP=0 --kv fp16 --ple-io ram`, after `--calibrate` on that card and file (`docs/DETAILS.md:489-503`; its settings are recorded);
  - KV streaming is off at these contexts (`--kv-resident` only at ≥ 64K, St `setup.py:6604-6620`), and this is
    recorded;
  - `STRATA_IQ_MT_MIN` is left at its default. It is a reproducibility switch, not an approximation, and costs Strata
    1–3% (`docs/DETAILS.md:97-104`);
  - the prefill CPU share is off: `STRATA_PREFILL_CPU_SHARE=0` (lead decision, step 0). Rev 4 assumed it never ran on
    the 1,024-token prompts, but Strata reads them as 1,023-token chunks, below the share's 1,024-token limit
    (`prefill.cpp:178-181`), and the share is not exact (its log: mean KL ≈ 0.004). Off, it measured no slower (§13).
- **Strata defaults:**
  - setup's command for the card: `--expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5`, the suffix
    drafter, int8 KV above 8K, PLE O_DIRECT, calibrated (St `setup.py:6570-6576`);
  - against Holon with A1, q8_0 KV above 8K and MTP.
- **Drafting:**
  - **Off:** Strata without `--mtp` and with `--suffix-draft 0` (St `generate.cpp:2134`); Holon with no drafter.
  - **Matched:** both run Strata's MTP pack at 3 drafts. Strata uses `--spec 4 --spec-min-p 0 --suffix-draft 0`,
    `STRATA_MTP_FULL_HEAD=1`, and pooled hnorm (its default).
  - Accepted tokens per pass are recorded for both engines in every MTP row.
- **RAM-short pairing:** Holon's SSD tier against Strata's `--ple-io direct`.
- **Statistics** (fixed now):
  - ≥ 5 interleaved rounds (reversed on odd rounds), each with exactly 3 distinct prompt spans per context;
  - **paired**: each (round, span) pair gives one Holon / Strata ratio, measured in the same round on the same span;
  - the cell statistic is the median of those ratios, with a 95% bootstrap CI that resamples the pairs;
  - **pass** if the CI's lower bound ≥ bar − 0.02;
  - **fail** if the CI's upper bound < bar;
  - otherwise **escalate** to the owner.
  - There are 48 gated cells: 2 cards × 2 baselines × 2 drafting modes × 3 contexts × 2 metrics.
  - Each cell reports accepted tokens per pass and the expert hit rate, and the hit rate over the first 32 decode
    tokens. Prefill lending returns evicted slots empty, in both engines, so decode starts with a dip.
- **Reported rows:** FreeToken and llama.cpp, one arm each, 3 rounds.
- **Cost:**
  - ≈ 2.5–3 h per card per full set (5 rounds × 3 spans × 8 gated arms, plus the reported arms; more spans would
    double it);
  - calibration, ≈ 30 min per card and file;
  - at least two full sets (N1 and step 9), plus step 7's iterations.
  - Disk for files, packs and MTP sources is checked before scheduling.

**D12 — Offload: X's defaults (`lfu`, F5a, F7), plus the following.**
- **(a) Aux-model experts std minor** (plan §10):
  - The MTP layer becomes the store's next layer group. Today it is refused (`std/holon_std/moe/offload.py:63-65`).
  - Its resolve takes row kinds for the drafter's sub-steps. The store's own `before_launch` derives them
    (`std/holon_std/tiered/store.py:457-465`), from `step.draft_k` and the sampled rows the host knows.
  - Drafters have no `before_launch` (`core/holon/api/hooks.py:103-119`), so no hook and no core seam is added.
  - Counts are zeroed at the target's group 0, and kernels bind after the aux model is built.
  - L0 checks the store's per-group count deltas for sub-steps.
- **(b) Tune's F5b benchmark is interleaved** before FN uses its ≥ 5% rule (`tools/holon_tools/tune.py:25, 70-104`).
- **(c) Prefill scratch is lent from the cache: the MoE prefill track's T1 and T2.**
  - **Why it is not plugin-internal (round 2):**
    - The MoE scratch is allocated with `torch.empty` inside `kernels.ggml_cuda` (`plugin.py:464, 481`).
    - The `linear_experts@2` and `linear_experts_arena@1` contracts take `(x, parts, ids) → y`, with no workspace
      or output argument (`std/holon_std/moe/kernels.py:17, 40`).
    - X's `lend(n)` hands out the coldest slots, which are scattered.
  - **What the track lands (lead decision):**
    - **T1:** the workspace / output argument on both contracts.
    - **T2:** contiguous-tail lending from one class arena, the largest (the gate/up class of the most common type),
      regardless of temperature, as FreeToken and Strata do (St `generate.cpp:536-538, 3731-3741`, `pf_borrow`).
      The evicted slots return empty.
  - **What is lent:** D10's `out` and `scratch` for one pass, at the pass's live peak, which spans two calls. With
    D4, gate/up's `out` is the SwiGLU output (25.6 KB per token); it stays live as the down call's input, beside the
    down call's fp32 `out` (102.4 KB) and the q8_1 rows and id maps (≈ 12 KB): **≈ 140 KB per token**.
  - A pass is the track's `pass_rows`, 3,072 at top-10. So **≈ 0.43 GB** is lent during any step of ≥ 3,072 tokens,
    and ≈ 0.29 GB during a 2K step.
  - **The arena-choice rule stays FN's stated one:** the largest class arena. The lead forwards the rule to the
    track.
  - FN's only use: the offload deployment turns lending on, through the lock (d).
- **(d) The step size is a lock setting per card** (`max_step_tokens`, with lending on or off).
  - One deployment at `--ctx 33792` serves the 1K, 4K and 32K rows: the 1K and 4K prompts are single steps, and the
    8K-step peak is reserved in every row.
  - **Who writes it.** tune benchmarks decode only, so it does not choose prefill settings (track D0). FN's two locks
    carry three hand-set values, chosen from step 7's prefill and decode measurements and changed only through
    `propagate`:
    - `max_step_tokens`: **4,096 on the RTX 5080** (lead decision after step 0: its 32K prefill is compute-bound at
      ≈ 2.85K tok/s, under the 4K step's streaming cap of ≈ 6.9K, and 4K steps free ≈ 0.9 GB of cache, §13); 8,192
      on the RTX 5090;
    - `ggml_cuda.pass_rows` = 4,096, the cap, which yields the proven 3,072 at top-10;
    - `moe.ring_lend` = `tail`.
- **(e) F5b rows** are reported separately. Gated rows use the lock's configuration, and X's F5b-lock escalation still
  stands.
- **(f) The profile** comes from `route_dump` on the RTX PRO 6000 (held-out prompts) and ships in the bundle.

**D13 — Qwen3.6-35B-A3B is off the critical path.** It is optional later evidence for `std.ffn.sum` and
`gated_swiglu` on an existing trunk.

**D13′ — Std naming.**
- The hyper-connection units are named for FN's gated-residual form: `std.hyper.gr_block@1`, `std.hyper.gr_mix@1`
  (the output mixer), and the kernels `hc_gr_mix@1` / `hc_gr_combine@1`. GLM's mHC, which has a stream-mixing matrix,
  will add siblings beside them.
- `std.hyper.expand@1` (repeating the embedding) is form-free.
- The PLE unit (`qwen4_exp.ple@1`) and its kernel contracts stay in `models.qwen4_exp` until a second user appears.
  `std.tiered.rows@1` is in std.
- The PLE contracts are prefixed like the unit: `qwen4_exp.ple_rows@1` and `qwen4_exp.ple_block@1`.
  - `plugins/contracts.txt` lists contracts globally, and model plugins already prefix theirs
    (`transform.qwen3_5.*`).
  - Moving an unprefixed name to std later would count as a removal, which needs a major.

**D14 — Split GGUFs are one checkpoint** in `std.weights.gguf`, with extents per shard. The judge gains the same.

**D15 — Text-only.**

**D16 — Locks and propagation.**
- **New locks:** `rtx-5080` (IQ2_XS) and `rtx-5090` (IQ3_S), both offload with Strata's MTP pack. No RTX PRO 6000
  lock.
- **FN changes `src` in:**
  - the core (every lock);
  - `std.mixers.gdn`, `std.moe.offload` (aux experts), `std.weights.gguf`;
  - `kernels.ggml_cpu` (Q2_0 type). `kernels.ggml_cuda`'s T1–T5 are the MoE prefill track's, and propagate with
    it;
  - **FN's lock settings, hand-set and changed only through `propagate`** (D12d): `ggml_cuda.pass_rows` = 4,096
    (yielding 3,072 at top-10), `max_step_tokens` = 4,096 (RTX 5080) / 8,192 (RTX 5090), `moe.ring_lend` = `tail`;
  - `drafters/mtp` and `drafters/dflash` (no lock uses either; checked `plans/`);
  - the workspace gguf-py relock.
- **Order:** the MoE prefill track propagates its own changes first. Of FN and K, whichever closes second propagates
  every lock once (K's D14).

## 3. What Holon has today (relevant pieces)

| Piece | State |
|---|---|
| Model trunk | `embed → blocks → norm → lm_head`, h `[T, H]` (`core/holon/core/build.py:265-295`). The core passes the model's `norm` Ref to every block (`build.py:273`). |
| Unit classes | `BuildContext.unit_factory` exists only in the core (`build.py:59`). The ABI's `BuildCtx` has only `unit(ref)` (`api/model.py:87`). |
| State | Only mixers declare state (`assembly.py:139-161`). A Slot-only group without a planner gets the default planner, whose meta has no `graph_key`, so CUDA graphs are silently disabled (`executor.py:285-287, 313-315`). Shared paged page groups are refused (`state.py:149-150`). |
| GDN | Two Slot specs, `draft_slots` 0, `alignment = in_step = 64`, replay rewind (`gdn/units.py:7-13, 117-124`) |
| Judge | `_streamed` per-layer driver (`golden.py:229-266`); MoE detection by `_moe` suffix (`:293-294`); mirror without a shared expert (`moe.py:34-46`); bounded trace ≤ 1,408 tokens (`bounded.py:53-68`); snapshot long request 2,048 + 8 (`snapshot.py:129-147`) |
| MoE, offload, MTP, GGUF | as rev 1. Resolve fits E = 512 (`kernels/expert_io/plugin.py:25-26, 144-147`). `topk-moe.cu` instantiates 512 (`:341`). |
| Core | 4,603 / 6,000 code lines, ABI 1.3 |

## 4. Core seams (ABI 1.4 or 1.5 by merge order; ≤ +47 lines over FN's step-1 base)

| # | Seam | Today → change | Toy proof | Lines |
|---|---|---|---|---|
| C1 | **Residual width** | h is `[T, hidden]` → `ModelSpec.residual: Residual \| None = None`, with `Residual(width, dtype)`. Trunk, taps and `Drafter.run` carry `[T, width]` of `dtype`. The profile builds the drafter's input from it and asserts that the trunk's real output matches. | A toy model with an expand embed `[T, 2H]` fp32 and a collapsing norm. Trunk, `forward`, prompt logprobs, `_worst_step`, a tap and a toy drafter are checked in the profile, eager and captured. A mis-declared width fails at startup. | ≈ 15 |
| C2 | **State declared by the block** | Only `layer.mixer` → also `ModelSpec.block`, over all layers, with the same `state_specs` / `planner` hooks. **On this new block-declared path only**, a non-paged group without a planner is refused at bind. Mixers keep today's behaviour, plus one stderr line when a planner-less group disables graphs (as `_choose` warns, `build.py:96-98`). This keeps the minor purely additive (`design.md:351`). | In `test_payloads.py`: a toy block declares, **through a nested Ref** (C3), a Slot running sum at one layer with alignment 64 / in_step 64. Beside it, a toy mixer kind with alignment 16. Asserts a prefix-hit length > 0, cut and in-step publish, adopt and replay rewind, and graphs captured. | ≈ 24 |
| C3 | **`BuildCtx.unit_class(ref)`** | A plugin cannot reach a nested Ref's class → the registered class serving `ref` (`unit_factory` exposed), so a composite unit forwards `state_specs` / `planner` / alignment from the unit it wraps without importing it | Covered by C2's nested Ref | ≈ 5 |

- **Dropped from rev 1:** C2's FFN branch. FFN `forward(x)` has no StepCtx (`blocks/units.py:11-25`), and
  `sub_planners` refuses recurrent aux state (`assembly.py:125-127`).
- **Generality:**
  - `check_core_denylist.py` gains `qsa|ngram|indexer`.
  - C1 serves GLM's mHC.
  - C2 and C3 serve any composite stateful block.
- **ABI:** all three are additive (a defaulted field, optional hooks on the block, a member of a core-implemented
  protocol). Regenerate `core/abi.txt`, and check `check_abi --base` and the line count against FN's step-1 base.
  Whichever of K and FN merges its seams second rebases and takes the next minor.

## 5. Units, contracts, model plugin

| Unit | Role |
|---|---|
| `std.hyper.expand@1(inner, n)` | the inner embed repeated to `[T, n·H]`, fp32 |
| `std.hyper.gr_block@1(n, rank, pre=Ref?, pre_at=(1,))` | `pre` → mix → mixer → combine → mix → ffn → combine. Ignores the `norm` site parameter (the core passes the model's norm, here the output mixer). Forwards `pre`'s state via `unit_class` (C2, C3). |
| `std.hyper.gr_mix@1(n, rank)` | the output mixer, used as `ModelSpec.norm` |
| `qwen4_exp.ple@1` (model plugin) | ids, the row gather, K/V, norms, the signed-√ gate, the dilated conv with fp32 state and replay rewind |
| `std.tiered.rows@1` | the row table, RAM or SSD |
| `std.mixers.qsa@1` | D6 |
| `std.ffn.sum@1`, `std.ffn.gated_swiglu@1` | D7 |
| `std.spec.mtp_hc_head@1` | D9 |
| edits | `std.mixers.gdn` (`out_gate`), `std.weights.gguf` (split files), `std.moe.offload` (aux experts; lending is the MoE prefill track's T1–T2), `drafters/mtp` (residual carry, `head.out`), `drafters/dflash` (refuses `residual.width ≠ hidden`) |

**Contracts** (each with `reference.py` and a CPU provider):
- `hc_gr_mix@1`, `hc_gr_combine@1` (fp32, ieee);
- `rows_gather@1` (bitwise dequant: IQ4_NL / Q4_0 / Q5_0 / Q8_0 / BF16);
- `qwen4_exp.ple_rows@1` (int64, bitwise; fresh-slot EOS);
- `qwen4_exp.ple_block@1` (fp32; in-step payloads; replay rewind). Its contract states the bitwise test's
  preconditions:
  - chunked and one-shot, or rewind and decode, get identical K / V projection inputs: the test feeds both the same
    projected rows, since `qlinear`'s bits may depend on rows per call (the track's D5);
  - every reduction (the grouped norms, the √2560 dot) is per row, in an order independent of the row count and the
    launch shape;
- `qsa_index@1` (pooling from stored raw keys);
- `qsa_select@1` (fp32 ieee; lower-index ties; sub-batched workspace);
- `qsa_attend@1` (prefill, decode, verify; gqa's KV formats). **It reads q8_0 KV in-kernel in all three** (panel C,
  step 0): the defaults rows' q8_0 must gain what Strata's int8 gains at 32K (+15–20% prefill), not dequantize first.

**`models.qwen4_exp`:**
- The recipe sets `residual=Residual(10240, "float32")`, `block=Ref("std.hyper.gr_block@1", n=4, rank=320,
  pre=Ref("qwen4_exp.ple@1", alignment=64, in_step=64, …), pre_at=(1,))`. `alignment` and `in_step` are GDN's chunk,
  asserted against `std.mixers.gdn`'s constant in the plugin's test.
- Layers are `qsa` / `gdn(out_gate="sigmoid")`; every FFN is `std.ffn.sum@1(moe, gated_swiglu)`.
- The GGUF rules use LC's tensor names and transforms.
- The MTP rules map Strata's `mtp-q2_0.gguf`.

## 6. Numerics and judge (thresholds and predicted verdicts fixed now)

### 6.1 CPU: the tiny model vs the HF fixture

**Config:** 8 layers (QSA at 3 and 7, PLE at 1), H = 64, n = 4, rank 16, vocab 248,320.

| Group | Values |
|---|---|
| Attention | 4 / 2 heads × 64 |
| Indexer | 2 × 32, ratio 4, budget 16 |
| GDN | 2 / 6 × 32 |
| MoE | 8 experts top-2, I = 64; shared expert I = 64 (multiples of Q2_0's 64-value block) |
| PLE | 4 heads × 16, `ngram_vocab_size_base` 1000 |
| MTP | 1 layer |

Everything is randomised (D2). The 100-token prompts contain an EOS; they are chunked at 7 and 13, then decoded for
24 steps.

| Check | Threshold | Predicted |
|---|---|---|
| output-mixer x and logits (recomputed with the GGUF's `lm_head`), chunked and one-shot, vs the fixture | max \|Δ\| ≤ 1e-4, top-1 identical | pass |
| QSA token sets per (layer, position) | identical under the lower-index rule; the test asserts that boundary ties occur | pass |
| PLE row ids | bitwise | pass |
| prefix hit vs recompute | bitwise | pass |
| MTP head, pooled vs SG's transcription; stream vs LC's graph | ≤ 1e-4 | pass |
| MTP head loaded from the tiny Strata-format pack vs SG's pooled reference on the pack's own dequantized tensors (names, the 1 + w convention, the `fc_embedding` / `fc_hidden` split, gate / up order) | ≤ 1e-4 | pass |
| plants `qsa_dense_limit_off_by_one`, `ple_conv_dilation_1`, `ple_eos_cut_off` | the tiny test fails | fail (the conv is nonzero) |

### 6.2 Full model

**Setup:**
- Goldens: fp32, streamed, for IQ2_XS and IQ3_S, on `v1-qwen3_5` and `v1-long`.
- Floors: bf16, with `--q8-1` per path and the shared expert mirrored. IQ1_M needs no special case under D1′; under
  its fallback, those rows' weights are cast to bf16. F5b's CPU set uses Q8_K blocking with `q2_0` in `Q8_0_DOT`.
- Pass rules: `docs/design.md:1395`.

| Check | Predicted |
|---|---|
| L2, both files, both sets | pass; KL p99 ≤ 1.3× floor; selection flips < 1% of blocks on `v1-long` (reported) |
| bounded: chunk 250, prefix hit (4K system prompt), graphs vs eager, overlap | pass |
| snapshot ×3 incl. an 8K request with MTP | run-identical |
| MTP verify L2; accepted per pass | pass; ≥ 2.2 |
| N4's zero-tie rate on layer 3 | reported (it decides how often the tie rule matters) |
| decode divergence, long trace vs its forced-dense control, MTP off and MTP on (acceptance 2) | pass: KL p99 sparse ≤ max(2 × dense, dense + 2.5 SE); pre-registered with `qsa_decode_select_shift` failing in both arms |
| GPU unit test, `qwen4_exp.ple_block@1`: chunked vs one-shot, replay-rewind vs k decode steps | bitwise |
| offload vs all-resident (capped, `lfu`, MTP experts in the store) | byte-identical |
| PLE RAM vs SSD | byte-identical |

### 6.3 Plants

Each plant changes only its own path.

| Plant | Breaks | Predicted |
|---|---|---|
| `qsa_select_shift` | prefill selection shifted one block past the budget | L2 fails on `v1-long`, passes on `v1-qwen3_5` |
| `qsa_decode_select_shift` | the same, in decode and verify only | L2 clean (teacher-forced) and the bounded checks clean (both sides decode with the plant); **the decode-divergence gate fails in both arms** (sparse KL p99 rises, the forced-dense control does not); pre-registered on the RTX PRO 6000 before any lock |
| `qsa_pool_rope_last` | pooled key rotated at 4b + 3 | L2 fails on `v1-long`; if not, the long set gains recall prompts before any lock |
| `hc_inject_unscaled` | no `/n` in the inject | L2 fails |
| `ple_gate_no_sqrt` | the gate without the signed √ | L2 fails |
| `ple_state_not_carried` | the conv window is zeroed at a chunk start | bounded chunk-250 check fails; one-shot L2 clean |
| `ple_state_bf16` | the conv history stored in bf16 | **the GPU unit bitwise test fails** (chunked ≠ one-shot); the model-level checks stay clean (below bf16 noise) |
| `shexp_gate_off` | the shared expert ungated | L2 fails (the floor mirrors the gate) |
| `mtp_stream0` | MTP reads stream 0 for every stream | acceptance drops ≥ 0.4 per pass; outputs unchanged |
| `aux_row_padding` | one real MTP row marked PADDING (reads the null slot) | snapshot with MTP experts in the store ≠ all-resident |
| `rows_off_by_one` | the SSD tier reads the neighbouring row | RAM-vs-SSD identity fails; L2 fails |

## 7. Memory fit (est.; step 0 and tune measure)

**One rule for both cards:** budget = 0.9 × (visible − context − weights) − non-lendable peak − staging. The expert
cache is the budget less KV and per-request state.
- **Visible VRAM:**
  - RTX 5080: X measured 15,841 MiB = 16.61 GB (`evidence/x4_gate_rtx-5080_2026-10-17.json`).
  - RTX 5090: the nominal 32 GiB less the same ≈ 460 MiB, so 33.88 GB (est.).
- **Context:** ≈ 0.9 GB, derived from the same evidence. X's 30B pool budget was 11.747 GiB = 12.61 GB with 1.05 GB
  of weights and its ≈ 0.6 GB step peak, so 0.9 × (16.61 − c − 1.05) − 0.6 = 12.61 gives c ≈ 0.88.
- **Weights:** IQ2_XS 3.76 GB / IQ3_S 4.51 GB, plus Strata's MTP pack 0.89 GB, resident.
- **Staging and null slots:** ≈ 40 per class.
  - IQ2_XS has 4 expert classes (gate / up IQ2_S, IQ2_XXS, IQ1_M; down Q2_0): ≈ 0.2 GB.
  - IQ3_S has 6 (gate / up IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS; down IQ4_NL, Q2_0): ≈ 0.34 GB.

**Per-request state:**
- **KV at 32K:** 13 × 2,432 B per token ≈ 1.04 GB bf16, 0.63 GB as q8_0.
- **GDN:** (`max_seqs` + 1) × (1 + `draft_slots` 0) = 2 slots × 36 × 3.06 MiB ≈ 0.23 GB, plus one checkpoint
  ≈ 0.12 GB. Rev 1's 8 slots were wrong (A-M2).
- **PLE:** 368 KB per slot.
- **KV and state together:** 1.39 GB.

**The step peak:**
- **Not lendable** (R, HC, GDN with FLA's chunk state — `h` alone is ≈ 24.6 KB per token, `chunk_delta_h.py:378` —
  QSA, selection sub-batches): **≈ 200 KB per token**. Step 0 measures it.
- **Lendable** (D10's `out` + `scratch` for one pass, at its live peak across gate/up's `out` and the down call,
  with D4, D12c): ≈ 140 KB per token × the pass. A pass is min(step, `pass_rows`), and `pass_rows` is 3,072 at
  top-10.

**RTX 5080, IQ2_XS:** 0.9 × (16.61 − 0.9 − 4.65) = 9.95 GB, less KV and state 1.39 and staging 0.2, leaves 8.36 GB
for the cache and the non-lendable peak.

| `max_step_tokens` (one per card; hand-set, D12d) | 2K | 4K | 8K |
|---|---|---|---|
| Non-lendable peak | 0.41 GB | 0.82 | 1.64 |
| Lent during prefill (one pass's buffers, gate/up's `out` with the down call's; the ring's ≈ 1.1 GB comes on top, as in X) | 0.29 | 0.43 | 0.43 |
| **Decode expert cache, with lending** | **7.9 GB (22%)** | **7.5 (21%)** | **6.7 (19%)** |
| Decode expert cache, without lending | 7.7 (22%) | 7.1 (20%) | 6.3 (18%) |
| Prefill cap from streaming non-resident and lent experts at ≈ 47.5 GB/s | ≈ 27.7 GB per step → ≈ 3.5K tok/s | ≈ 28.1 → ≈ 6.9K | ≈ 28.8 → ≈ 13.5K |

Notes on the 5080:
- **One deployment per card:** `--ctx 33792` serves every row, so the 8K step's peak is reserved in the 1K and 4K
  rows too. The lock steps **4K** with lending (lead decision after step 0, D12d), reserving the 4K step's peak.
- **Lent slots return empty** (T2), so a 32K decode starts with a dipped hit rate. Each cell reports the hit rate over
  the first 32 decode tokens (D11). Strata's borrowing does the same.
- **MTP experts in the store** instead of resident: ≈ +0.7 GB of cache.

**RTX 5090, IQ3_S:** 0.9 × (33.88 − 0.9 − 5.40) = 24.82 GB, less 1.39, 0.34 (6 classes) and 1.64 (8K steps), gives
≈ 21.45 GB: 43% of 50.29 GB. Its prefill cap is similar, ≈ 13.5K tok/s at 8K steps.

**RTX PRO 6000, all-resident:** 40.1 / 55.7 GB of weights (IQ2_XS / IQ3_S, with MTP), plus the PLE table resident
on the card (28.8 GB; step 5 reads it through `std.ops.embedding@1`, the RAM tier being step 6's): 68.9 / 84.5 GB.
KV up to 128K is 4.1 GB.
- **The fit at 32K (step 3 review, C):** IQ3_S needs 84.5 + 1.39 (KV and state) + 1.64 + 0.43 (the 8K step's peaks)
  + 0.9 (context) ≈ 88.9 GB; IQ2_XS ≈ 73.3 GB. Visible is ≈ 102.6 GB (96 GiB less ≈ 460 MiB, est.), so the rule
  gives 0.9 × (102.6 − 0.9 − 84.5) = 15.5 GB for the 3.5 GB of KV, state and peaks: both fit, IQ3_S with ≈ 12 GB to
  spare. At 128K, IQ3_S's KV rises by ≈ 3.1 GB and still fits.

**Host pinned memory:**
- IQ2_XS: experts 35.5 + PLE 28.8 ≈ 64.3 GB.
- IQ3_S: 79.1 GB.
- Add 0.7 GB with MTP experts in the store.
- With the PLE SSD tier: only the experts.
- Host RAM per class is checked before scheduling (not recorded here).

**Decode, proxy anchors** (C-m1):
- Hit rate 75–90%: Strata reports that band on single 12–16 GB cards (`docs/DETAILS.md:1694`), and X's 30B hit
  ≈ 80% at 25% cached. So misses are 69–173 MB per token on IQ2_XS.
- Dense weights read per token are 3.42 GB, including 1.26 GB of BF16 HC: ≈ 3.8–4.2 ms on the 5080.
- F5b's gain is marginal at ≈ 2 misses per layer.
- These replace rev 1's "≈ 50% hits". This section supersedes `docs/v1-offload.md` §7's FN rows.

## 8. Step 0 — measure before freezing

On the RTX 5080 and 5090 desktops, one heavy job at a time under `MemoryMax`.

| # | Measure | Decides |
|---|---|---|
| N0 | The harness: offload_ctx's Strata backend (the `--serve` protocol, raw ids) and Holon backend; Strata built at `296b59d`; its packs for both files; its MTP pack, with its sha256 and producing commands recorded; `--calibrate` per card and file; `--eos-ids` as Strata's ignore-EOS, else EOS pre-screening of the spans; the corpus and its span offsets | D11 runnable |
| N1 | Strata no-approx and defaults, drafting off and matched, 1K / 4K / 32K, both cards (D11's statistics); FreeToken and llama.cpp once | the denominators; replaces rev 1's 2.3K tok/s community reference (PCIe-bound at 1K, not our file) |
| N2 | The zero-copy row gather (1–4,096 tokens) and O_DIRECT 4 KiB reads at queue depth 16–256 | D5's SSD read-ahead depth |
| N3 | Proxy roofline: routing traces from llama.cpp's `qwen4exp` (eval-callback top-k dump) or Strata's routing dump; X's M1 / M3 rates; Strata's phase timers (St `prefill.cpp:2811, 2915`), hit rate and chosen chunk from its logs; dense bytes and launch counts; FN's non-lendable step peak per token (§7's 200 KB) from a trunk-only allocation profile | every acceptance-5 row predicted **at step 0**; a row predicted < 0.9× goes to the owner before step 1 ends |
| N4 | (step 2) HF 5.19.0 tiny model; `_streamed` on layers 1 (PLE) and 3 (QSA), and one 32K `v1-long` prompt with time and peak RSS; the zero-tie rate | D2's driver form (query-chunked or not); its cost |
| N5 | (step 5) Holon's own routing traces on the RTX PRO 6000; the M6 replay at 20–50% cached | re-checks N3's roofline |

## 9. Steps

| Step | Work | Checks | Days |
|---|---|---|---|
| 0 | N0–N3 (GPU desktops) | evidence; predicted rows | 4 |
| 1 | C1–C3, the planner-required bind check on the block path (with the mixer warning), toy proofs, denylist | core tests; ABI against the step-1 base; LOC | 3 |
| 2 | The oracle: golden venv; the judge edits of D2 (scoped tie wrapper, query-block mode); gguf-py at LC's tree and `file_meta`'s provenance; `q2_0` in `Q8_0_DOT`; split files in the judge; `v1-long` with recall prompts (+ sha256); the tiny tool with LC's converter, the Strata-format MTP pack, the small fixtures and the pinned GGUF artifact; N4 | the driver equals HF's whole forward bitwise on the tiny model; N4 | 5 |
| 3 | std units and the model plugin with references and CPU providers; split GGUF; `out_gate`; Q2_0 types | §6.1; Qwen3.8's snapshot unchanged | 5 |
| 4 | Triton kernels with L0 on 3 SMs and memcheck; first tuning against the launch and `qsa_select` targets; the GPU unit bitwise test of `qwen4_exp.ple_block@1`; rebase onto the MoE prefill track (T1–T5); IQ1_M loader L0 cases (or its fallback) | acceptance 3; the PLE unit bitwise test | 7 |
| 5 | The full model on the RTX PRO 6000: goldens and floors, L2, bounded, the decode-divergence gate, snapshot, MTP; **kernel-only all-resident comparison with Strata** (both fit, as `x4_gap` did); N5 | acceptance 1b, 2; plants | 5 |
| 6 | Offload: aux experts (a, the store's `before_launch`), F5b interleave (b), lending on through the lock (c, using T1–T2), PLE SSD tier, profile, exactness | acceptance 4; plants | 5 |
| 7 | **Perf closure** on the bar cards: nsys per layer, fusions, CUDA rewrites by D10's rule; measures `max_step_tokens` × lending for D12d's hand-set lock values | every row within its roofline band, or the gap explained; ≤ 768 launches per decode token (48 × 16; gate) | 4 |
| 8 | Locks (`tune --new-model`) on the RTX 5080 and RTX 5090 | L2, snapshot, fit vs §7 | 2 |
| 9 | The comparison (D11) | acceptance 5 | 3 |
| 10 | Close: suite, panel code review, propagation if FN closes second, plan §10 | acceptance 6–8 | 2 |

**Schedule.** The work sums to **45 days** (rev 2: 44; step 2 grows by a day for the MTP-pack oracle and the fixture
artifact).

**Critical path:** 0 ∥ 1 (4) → 2 ∥ 3 (6: step 3's CPU check waits for step 2's fixtures, which now land a day later)
→ 4 (5; its QSA / HC work starts during step 3) → 5 (5) → 6 (5) → 7 (4) → 8 (2) → 9 (3) → 10 (2) = **36 days**
(rev 2: 35).

That figure assumes:
- two agents in steps 0–4;
- the MoE prefill track landing T1–T5 before step 4 ends. Its own estimate was 9–14 days before T1–T5 were added;
  the track re-estimates. If it slips:
  - step 5 starts with IQ3_S, which has no IQ1_M;
  - step 6's lending (T1–T2) waits for it.

Step 9's two full measurement sets take ≈ 2.5–3 h per card each (D11), which fits in its 3 days.

Not counted: waiting on K's core round if K holds the core when FN's step 1 is ready.

With one agent the steps run in series: 45 days.

## 10. Risks

- **R1: decode at the 5080's fit.** At ≈ 19% cached (8K steps; 22% at 2K) the decode is miss- and dense-bound. N3's roofline is
  pre-registered; a row predicted below the bar goes to the owner at step 0. F5b is the reported fallback.
- **R2: prefill.** It depends on the MoE prefill track (D2, compact launches) and on D12c's lending. N3 predicts it at
  step 0. A shortfall after the track lands is escalated, not absorbed.
- **R3: the oracle.**
  - The streamed golden at 32K: N4 measures it.
  - The tie rule: documented, and its rate measured.
  - Two judge environments: the golden venv is used only for recording.
- **R4: LC vs HF GGUF conventions**, for example the PLE table's FP8 `weight_scale`. Golden and candidate share the
  file's table; the error against the checkpoint is reported.
- **R5: QSA near-ties** between fp32 and bf16 inputs: bounded; the flip rate is reported.
- **R6: Strata moves fast.** It is pinned and re-checked at the close.
- **R7: pinned host memory** (64–79 GB). The SSD tier is paired with Strata's direct I/O.
- **R8: dependency order.** The MoE prefill track and K both touch shared code.
  - The order is fixed: the MoE prefill track first, then FN and K by merge order. One propagation, by the second to
    close.
  - The track's scope grew by T1–T5 for FN: the contract argument, tail lending, `pass_rows`, IQ1_M, Q2_0. Its
    landing date is FN's step-4 dependency.

**Deferred:** K4 (FN's follow-up in plan §10), a PLE row cache, Strata's suffix drafter, `token_embd` on the host
(0.34 GB, a later tune option), and Qwen3.6-35B-A3B.

## 11. Lead decisions applied (2026-10-18; rev 1's open questions closed)

1. Qwen3.6-35B-A3B is off the critical path (D13).
2. A golden-only venv, with the tie rule recorded (D2).
3. Strata `--kv fp16 --ple-io ram`; `STRATA_IQ_MT_MIN` dropped; `--calibrate`; the SSD pairing (D11).
4. Drafting-off and MTP-matched rows both gated, with the statistics; Strata's MTP pack, pooled hnorm, full-vocab
   drafts (D9, D11).
5. Gate on the lock's configuration; F5b reported separately (D12e).
6. IQ2_XS stays the 5080's file; IQ1_M becomes a native loader, with grouped scratch as the fallback (D1′).
7. The MoE prefill track owns the prefill dispatch and `vendor.patch` and lands first; the proxy roofline moves to
   step 0 (D1′, N3).
8. FreeToken and llama.cpp are reported only (D11).
9. Core order with K, the budget relative to the step-1 base, C3, no FFN branch, stronger toys (§4).
10. Prefill scratch is lent; the step size is a lock setting (D12c–d, §7).
11. Perf closure (step 7) and the kernel-only comparison (step 5).
12. K4 goes into plan §10, owned by FN.
13. HC units are named for their form; PLE stays in the model plugin (D13′).

**Round 2 (2026-10-18):**
- **A-N1:** planner-less state is refused only on the new block path; mixers get a one-line stderr warning when
  graphs are disabled (§4 C2).
- **A-N2 and C's major:** the MoE prefill track owns and lands T1–T5 before FN's step 4; D12c refers to them.
- **A-N3:** the store's own `before_launch` derives the sub-step row kinds (D12a).
- **B-R-M1:** an absolute decode-divergence gate against a dense control, and a GPU unit bitwise test of
  `qwen4_exp.ple_block@1` (acceptance 2).
- **B-R-M2:** the tiny tool writes a Strata-format MTP pack, checked against SG's pooled reference (D2, §6.1).
- **B-m7:** the HC projections default to `ieee`; `tf32x3` only within the L0 budget and with L2 holding, recorded
  (D10).
- **C minors:** 200 KB per token non-lendable (step 0 measures it); staging subtracted; the pass size stated; the step
  is per card; one budget formula; `--eos-ids`; paired statistics with a stated corpus; the early-decode hit rate;
  the MTP pack's provenance; 2.5–3 h per card per full set (§7, D9, D11).

**Round 3 (2026-10-18):**
- **A-F1:** `pass_rows` is 3,072 at top-10. The lent bytes and §7 are re-derived. The lock values are set by hand
  through `propagate`, not by tune (D12d, D16).
- **A-F2:** FN depends on the track's D1–D5; D4 shrinks the lent buffers to ≈ 140 KB per token.
- **A-F3:** FN keeps "the largest class arena"; the rule is forwarded to the track.
- **B:** a forced-dense control on the same trace, with an MTP-on arm, pre-registered with the plant; the PLE test's
  preconditions in the contract; tiny `moe_intermediate_size` 64 and `mtp_fetch.py`'s layout.
- **C:** context-specific offsets, the warm-up span off every start, and 0 reused tokens per timed request; IQ3_S's 6
  classes (5090 cache ≈ 21.45 GB); exactly 3 spans.

No open questions remain for the owner. Any acceptance-5 cell that D11's rule escalates goes to the owner when it is
measured.

## 12. Review log

**Rev 1** (2026-10-18): drafted from cited source studies. Found on the way:
- Q2_0 is upstream;
- shared page groups are refused by the core;
- split GGUFs are refused.

**Rev 1 → rev 2: dispositions.** Every finding was checked against the cited code before its disposition.

**Panel A (architecture):**

| # | Finding | Disposition |
|---|---|---|
| A-B1 | No public way to reach a nested Ref's class | **Accepted.** Verified: `unit_factory` exists only in the core (`build.py:59`); `BuildCtx` has only `unit` (`api/model.py:87`). Added C3 `unit_class`; C2's toy declares its state through a nested Ref. |
| A-M1 | PLE checkpoints must align with GDN's | **Accepted.** Verified at `kind.py:26, 54-70` and `state.py:190-201`. The `alignment` / `in_step` Ref parameters are set by the recipe; `ple_block` writes in-step; the toy asserts a hit > 0 across alignments. |
| A-M2 | GDN verify is replay, not draft slots | **Accepted.** Verified: `gdn/units.py:7-13, 121-124` (`draft_slots` 0). PLE uses replay through `rewind_modules` (`api/state.py:135-140`); §7 has 2 slots. |
| A-M3 | PLE state dtype | **Accepted:** fp32, 368 KB (D4). Same as B-M6. |
| A-M4 | A Slot group without a planner disables graphs silently | **Accepted.** Verified: `_PagesPlanner.plan_bounds` returns None for non-paged state (`assembly.py:82-83`), giving no `graph_key` and `graphs = None` (`executor.py:285-287, 313-315`). C2 refuses it at bind; the toy captures graphs. |
| A-M5 | ABI 1.4 collision with K; K's line budget | **Accepted.** Merge order decides 1.4 / 1.5. The budget is relative to the step-1 base (≤ +45; ≤ +47 in rev 3). The second to close propagates (K's D14). |
| A-M6 | K4 is owned by no one | **Accepted.** Plan §10 makes FN the owner, as its first follow-up. D11 records Strata's `--kv-resident` as off at ≤ 32K (`setup.py:6604-6620`). |
| A-M7 | Std contracts encode FN's form | **Accepted:** the `gr_*` names (D13′); PLE moves to the model plugin. |
| A-m1 | An int-keyed Ref dict fails | **Accepted.** Verified at `terms.py:81-83`. `pre=Ref`, `pre_at=(1,)`. |
| A-m2 | The profile should assert the actual width; the dflash claim is wrong | **Accepted:** C1 asserts it; a dflash refusal is added (no lock uses dflash). |
| A-m3 | The FFN branch is unusable | **Accepted:** dropped. |
| A-m4 | D16's list is incomplete | **Accepted:** D16 completed. |
| A-m5 | Pass `out_gate` only when not the default | **Accepted:** D8. |
| A-m6 | Denylist additions | **Accepted:** `qsa\|ngram\|indexer`. |
| A-m7 | `before_launch` cannot hash after a hit | **Accepted:** those rows take the in-graph path (D5). |
| A-m8 | The MTP head / drafter interface | **Accepted:** `head.carries`, `head.out` (D9). |
| A-m9 | The block receives the model's norm | **Accepted:** documented as ignored (§5). |

**Panel B (correctness):**

| # | Finding | Disposition |
|---|---|---|
| B-B1 | `topk` tie order vs the lower-index rule | **Accepted:** the stable rule in the tool and the driver, recorded as a deviation from HF; the tiny test asserts that ties occur; N4 measures the rate. |
| B-B2 | The judge's MoE floor is wrong for `qwen4_exp` | **Accepted.** Verified at `golden.py:293-294, 264` and `moe.py:34-46`. The edits are listed in D2 and step 2; `_streamed` is reused. |
| B-M1 | The PLE table in the golden | **Accepted:** a lazy row gather and a hash-constant assert. |
| B-M2 | Q2_0 outside std | **Accepted, modified:** gguf-py at LC's tree (it has Q2_0, `quants.py:679`) instead of a shim; `q2_0` in `Q8_0_DOT` (verified missing at `reference.py:266`); split files in the judge. |
| B-M3 | Sparse decode is untested beyond L0 | **Accepted.** Verified at `bounded.py:53-68` and `snapshot.py:129-147`. A 4K system prompt, an 8K snapshot request with MTP (per-model parameters, so no other model's snapshot is re-recorded), and plant `qsa_decode_select_shift`. |
| B-M4 | Zeroed tiny weights; an independent GGUF writer | **Accepted:** randomised (verified HF zero init per the finding; the tool asserts nonzero); LC's converter; vocab 248,320. |
| B-M5 | The PLE gate's signed √ was missing | **Accepted:** §0, `ple_block`, plant `ple_gate_no_sqrt`. |
| B-M6 | Conv history dtype; pooling source | **Accepted:** fp32 state; pooling from stored keys; plant `ple_state_bf16`. |
| B-M7 | `aux_rowkind_target` cannot fail | **Accepted.** Verified: row kinds only mask and count (`store.py:461-467`). Replaced by `aux_row_padding` plus an L0 count-delta check. |
| B-M8 | The golden venv does not install | **Accepted:** `--no-deps` or overrides; tokenizers ≥ 0.23.1. |
| B-m1 | The dense / sparse rule | **Accepted:** D6. |
| B-m2 | "Chunked bitwise on the CPU" is unproven | **Accepted:** chunked is checked vs HF at 1e-4; bitwise only hit vs recompute. |
| B-m3 | Plant chunk size | **Accepted:** 250. |
| B-m4 | The IQ1_M bf16 path is not exact | **Superseded by D1′:** the native loader is exact; the fallback's floor casts. |
| B-m5 | Triton TF32 | **Accepted:** `ieee`. |
| B-m6 | Golden head memory | **Accepted:** chunked. |
| B-m7 | N3's layers | **Accepted:** layers 1 and 3, plus a 32K prompt (N4). |
| B-m8 | hnorm and the MTP oracle | **Accepted:** pooled pinned (St `generate.cpp:683-685`); Strata's per-pass definition; SG for pooled, LC for stream. |
| B-m9 | A fresh slot's ids | **Accepted:** EOS. |
| B-m10 | `page_size % 4` | **Accepted.** |
| B-m11 | Keep only the open block's raw tail | **Rejected.** It saves 96 MB at 32K (≈ 1.3% of the 5080's cache). A tail ring would be per-request state that verify and prefix hits must carry; per-token raw keys keep both stateless. Revisit with K4. |
| B-m12 | `v1-long` needs recall prompts | **Accepted:** 4 of 8. |

**Panel C (performance and fairness):**

| # | Finding | Disposition |
|---|---|---|
| C-B1 | The 32K prefill bar vs the fit | **Accepted.** Prefill scratch is lent (D12c); the step size is a lock setting (D12d); §7 shows it per step. Verified Strata's `--prefill auto` and borrowing at `generate.cpp:536-538, 3738-3741`. |
| C-M1 | The roofline comes too late | **Accepted:** proxy roofline at step 0 (N3); N5 re-checks. |
| C-M2 | The MoE prefill track | **Accepted:** dependency, types and shape added to its L0; D1's dequant path scoped to a fallback; "seed" claim struck. The track's D7 rejection of dequant + GEMM is noted. |
| C-M3 | The Strata harness | **Accepted:** offload_ctx backends; raw ids through `--serve`; client timing; `--max-context 33792`. Rev 2 said Strata had no ignore-EOS switch; round 2 found `--eos-ids` (see C2-m4). |
| C-M4 | MTP bytes differ | **Accepted:** Strata's own pack for both; full-vocab drafts in matched rows (`STRATA_MTP_FULL_HEAD=1`, `generate.cpp:3502`); both locks name the file. |
| C-M5 | Variance and multiple testing | **Accepted:** D11's statistics. |
| C-M6 | Strata's calibration | **Accepted:** `--calibrate` per card and file, recorded. |
| C-M7 | Decode loss points | **Accepted:** launch and `qsa_select` targets (acceptance 3); kernel-only comparison at step 5; profiling on the bar cards; perf-closure step 7. |
| C-M8 | Golden at 32K | **Accepted:** N4 measures it; a query-chunked driver if needed. |
| C-m1 | Hit-rate anchor | **Accepted:** 75–90% (§7). |
| C-m2 | IQ1_M scratch | **Accepted, for the fallback only:** 16 experts per launch. |
| C-m3 | Select workspace | **Accepted:** sub-batched, counted. |
| C-m4 | `IQ_MT_MIN`; CPU share | **Accepted:** dropped; recorded. |
| C-m5 | RAM tier pairing | **Accepted:** SSD vs `--ple-io direct`. |
| C-m6 | GDN draft copies | **Count rejected, rest accepted.** GDN declares `draft_slots` 0 (verified, `gdn/units.py:121-124`), so 2 slots, 0.23 GB, accounted. `token_embd` on the host deferred. |
| C-m7 | The 2.3K tok/s reference | **Accepted:** N1 replaces it. |
| C-m8 | Schedule | **Accepted:** re-estimated (§9). |
| C-m9 | F5b rows | **Accepted:** D12e. |
| C-m10 | Citations | **Accepted:** the suffix flag is at `generate.cpp:2134`; MTP dense parts are BF16; X §7's FN rows are superseded. **Rejected:** quoting the community report's directory path verbatim, because its name contains a CPU model string that `check_private.py` refuses; it is cited by date and card. |

**Rev 2 → rev 3: dispositions** (round 2). Every finding was checked against the cited code.

**Panel A:**

| # | Finding | Disposition |
|---|---|---|
| A-N1 (major) | Refusing planner-less state for mixers breaks the minor-version promise | **Accepted** (lead decision). Verified: `design.md:351` (minors only add); planner-less Slot state runs eagerly today (`assembly.py:77-86`, `executor.py:285-287`). The refusal now applies only to the new block-declared path; mixers get a stderr line when graphs go off, as `build.py:96-98` warns. C2 is +2 lines. |
| A-N2 | D1′ depends on a seam the MoE prefill track does not have | **Accepted.** The track owns and lands T1–T5 (lead decision; sent to the track). FN names them in its header and in D1′ / D12c. |
| A-N3 | Drafters have no `before_launch` | **Accepted.** Verified: drafter hooks are `hooks.py:103-119` only. The store's own `before_launch` (`store.py:457-465`) derives the sub-step rows (D12a). |
| A-N4 | K's doc describes FN's old seams | **Accepted, forwarded.** K's doc is outside this worktree. The coordinator updates `docs/v1-kv-tiers.md`'s FN line to "C1–C3, ≤ +47 lines, no FFN branch". |
| A note | PLE contract names | **Accepted:** `qwen4_exp.ple_rows@1`, `qwen4_exp.ple_block@1` (D13′). |

**Panel B:**

| # | Finding | Disposition |
|---|---|---|
| B-R-M1 (major) | The decode plant and the bf16-state plant cannot fail their assigned checks | **Accepted.** Verified: bounded scores the baseline by the same reference, so the floor carries the plant (`bounded.py:205-236`). Added a decode-divergence gate against a dense control and a GPU unit bitwise test of `qwen4_exp.ple_block@1`. The plants are reassigned to these checks (§6.3). |
| B-R-M2 (major) | The deployed MTP file has no oracle path | **Accepted.** The tiny tool writes a Strata-format pack with St `mtp_pack.py`; Holon's head loaded from it is checked against SG's pooled reference on its dequantized tensors within 1e-4 (§6.1). |
| B2-m1 | Vocab-248K fixtures are large; CI would skip | **Accepted.** Verified at `qwen3_5_files.py:86-87`. Commit the x, the selected sets and the row ids; the GGUF is a sha256-pinned artifact that CI fetches. |
| B2-m2 | The IQ1_M hunk needs the D4 layout case and the MMA vec-dot | **Accepted.** Verified: the default case of `mmq_get_q8_1_ds_layout` aborts (`mmq.cuh:60-100`); the MMA `q8_0_16` instance exists (`:786`). Both are in T4. |
| B2-m3 | gguf-py provenance; split-file hashing | **Accepted.** Verified at `judge/holon_judge/gguf.py:375-388`. `file_meta` records the llama.cpp commit, the `quants.py` sha256, and every shard's sha256. |
| B2-m4 | The query-block fallback must use HF's cache; record the mode | **Accepted** (D2). |
| B2-m5 | Scope the tie wrapper | **Accepted:** the indexer module only (D2). |
| B2-m6 | The per-step dense / sparse rule makes bits depend on batching | **Accepted, stated** (D6). A per-row choice is rejected: varying row subsets break fixed-shape graphs. |
| B2-m7 | `ieee` HC projections cost prefill time | **Accepted** (lead decision): `ieee` by default; `tf32x3` only with L0 within budget and L2 holding, recorded as a kernel fact (D10). |

**Panel C:**

| # | Finding | Disposition |
|---|---|---|
| C2-M (major) | Lending is not plugin-internal and cannot form dense buffers | **Accepted.** Verified: `torch.empty` scratch inside `kernels.ggml_cuda` (`plugin.py:464, 481`); no workspace argument in the contracts (`std/holon_std/moe/kernels.py:17, 40`). T1 (workspace argument) and T2 (contiguous-tail lending, slots returning empty) are owned by the MoE prefill track (lead decision); D12c refers to them. |
| C2-m1 | The peak and cache are optimistic | **Accepted.** Verified: FLA's chunk state `h` is allocated per chunk (`chunk_delta_h.py:378`). Now 200 KB per token non-lendable (step 0 measures it), staging subtracted, pass = min(step, 4,096). The 8K cell is 6.7 GB (19%). |
| C2-m2 | "Per card and context" is wrong | **Accepted:** one deployment per card at `--ctx 33792` (acceptance 6, D12d, §7). |
| C2-m3 | The 5090 formula mixes conventions | **Accepted:** one formula for both cards (§7). 5090: 21.6 GB, 43%. |
| C2-m4 | Strata can ignore EOS | **Accepted.** Verified: `--eos-ids` replaces the list (`generate.cpp:2187-2190`), and serve keeps it (`:2525-2531`). N0 tries it; pre-screening is the fallback. |
| C2-m5 | Statistics pairing; the span supply | **Accepted:** paired by (round, span); the corpus is three prompt sets' text re-tokenized; offsets k × ⌊N/6⌋ (D11). Verified: `v1.jsonl` is 318 KB. |
| C2-m6 | The prefill cap ignores what lending evicts; the early-decode hit rate | **Accepted:** the cap includes the lent bytes (≈ 13K tok/s at 8K steps); the first-32-token hit rate is reported per cell. |
| C2-m7 | Locks name a file nobody else can make | **Accepted:** the pack's sha256 and producing commands are recorded; porting the MSE quantizer is deferred (D9). |
| cost note | 2.5–3 h per card per full set | **Accepted** (D11; it fits in step 9). |

**Rev 3 → rev 4: dispositions** (round 3, minors only). Every finding was checked against the cited code or
document.

| # | Finding | Disposition |
|---|---|---|
| A-F1 | `pass_rows` is 3,072 at top-10, not 4,096; who writes the value | **Accepted.** Verified in the track's doc (§5's FN rows: ⌊32,768 ÷ 10⌋ in whole `CHUNK`s). Its D0 says `ggml_cuda.pass_rows` is a hand-set lock option written through `propagate`, and that tune benchmarks decode only. Lent bytes and §7 re-derived; D12d and D16 state FN's lock values (`pass_rows` 4,096 → 3,072 effective, `max_step_tokens` 8,192, `moe.ring_lend` `tail`), all hand-set. |
| A-F2 | Depend on D1–D5, including D4; re-derive the lent bytes | **Accepted.** With D4 the gate/up output is gone; the live peak is SwiGLU (25.6 KB) + down (102.4 KB) + workspaces (≈ 12 KB) ≈ 140 KB per token. The lent amount is 0.43 GB per 3,072-row pass. |
| A-F3 | The arena-choice rule | **Accepted as the lead decided:** FN keeps "the largest class arena"; the rule is forwarded to the track. |
| B3-1 | The dense control; pre-registration; MTP arm | **Accepted:** a test-only forced-dense mode on the same trace; pre-registered on the RTX PRO 6000 before any lock (unplanted passes, plant fails, else the statistic is tightened); an MTP-on sparse arm (acceptance 2, §6.2). |
| B3-2 | The PLE bitwise test's preconditions | **Accepted:** stated in `qwen4_exp.ple_block@1`'s contract (§5). |
| B3-3 | Tiny `moe_intermediate_size`; the MTP source layout | **Accepted:** I = 64 for routed and shared experts; the tool writes `mtp_fetch.py`'s raw-bf16 + manifest layout (verified: St `tools/mtp_fetch.py:1-12`) before `mtp_pack.py`. |
| C3-m | Span offsets, warm-up span, prefix reuse | **Accepted:** context-specific offsets; the warm-up span (the corpus's last 2,048 tokens) is off every start; 0 reused tokens required per timed request, or reuse off in both (Strata `--prompt-cache`, St `generate.cpp:784`). |
| C3-n1 | IQ3_S has 6 expert classes | **Accepted.** Verified from the parsed header's types: gate / up IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS; down IQ4_NL, Q2_0. Staging ≈ 0.34 GB; 5090 cache ≈ 21.45 GB. |
| C3-n2 | The span count vs the cost | **Accepted:** exactly 3 spans per context; the 2.5–3 h estimate stands. |

**Budget and schedule after rev 4:**
- **Core:** unchanged, ≤ +47 lines.
- **5080 expert cache:** with lending unchanged (7.9 / 7.5 / 6.7 GB at 2K / 4K / 8K steps). Without lending it rises
  to 7.7 / 7.1 / 6.3 GB. The 8K prefill cap is ≈ 13.5K tok/s.
- **5090 expert cache:** 21.6 → 21.45 GB, still 43%.
- **Schedule:** unchanged, 45 days of work and 36 on the critical path. The forced-dense mode and the MTP arm fit in
  step 5; the MTP source layout fits in step 2.

## 13. Step 0 record (2026-10-18)

**Status: step 0 measured; N3's decision rule tripped** (rows predicted below 0.9×, §8). Escalated to the owner before
step 1. Step 1 waits for K's core round anyway. Step 2 is not started beyond what the oracle check needed (the judge's
`qwen4exp` reader and the tiny model, below). Predictions were committed before any run
(`evidence/fn0_predictions_2026-10-18.json`).

**N0, the harness** (`bench/offload_ctx.py`, `spikes/fn0/arms_*.json`, `bench/setup_baselines.sh fn`):
- **Strata.** Built at `296b59d` with its pinned ggml (llama.cpp `3cf03257`), for sm_120, with `-march=native` on the
  bar cards' host class. Its packs come from `tools/iq_pack.py`, 1.5 GB each; the experts are read from the GGUF.
- **MTP pack** `mtp-q2_0.gguf`: sha256 `a22206a4d74eafbf31063a84610292535fb3b49af9a339963817fd04aed81e4d`. Produced by
  `tools/mtp_fetch.py fetch` (checkpoint `de4b8e4d`, all 31 tensors sha256-verified) and then
  `tools/mtp_pack.py --experts q2_0`.
- **Raw ids and EOS.** The client drives `--serve` with raw ids (`GEN 256 <ids>`). `--eos-ids 999999` works as
  ignore-EOS: every timed request ended at its length, so no span needed pre-screening.
- **No reuse.** With `--prompt-cache 0`, every timed request logged 0 reused tokens. `--max-context 33792` is set.
- **Drafting off.** A native pack refuses `--spec` < 2, so drafting off is `--spec 2` without `--mtp`, with
  `--suffix-draft 0`. With no drafts, `--spec 2` and `--spec 4` decode alike (89.1 vs 90.8 tok/s).
- **Calibration** (`tools/calibrate.py`, setup's command, recorded in `evidence/fn0_calibrate_<card>`):
  - RTX 5080: `--pcie-frac 0.55 --spec-min-p 0.70 --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97`;
  - RTX 5090: `--adapt-every 1 --adapt-swaps 80 --adapt-decay 0.97`.
  - Every Strata arm takes them. The MTP-matched arms keep `--spec-min-p 0`.
- **Corpus.** N = 189,707 tokens, sha256 `de504e58…`. The 32K candidates overlap by 1,151 tokens.

**N1** (5 rounds × 3 spans, the median of the 15 values; tok/s; prefill / decode at 1K · 4K · 32K):

| Card, file | Row | Prefill | Decode | Accepted per pass | Strata's cache |
|---|---|---|---|---|---|
| RTX 5080 IQ2_XS | no-approx, off | 1,622 · 2,727 · 3,086 | 89.9 · 89.5 · 83.9 | — | 7,658 slots, 10.26 GiB (31%) |
| | no-approx, MTP | 1,570 · 2,703 · 3,062 | 131.1 · 108.8 · 118.5 | 2.84 · 2.46 · 3.01 | |
| | defaults, off | 1,548 · 2,675 · 3,701 | 99.3 · 96.4 · 92.0 | — | |
| | defaults, MTP | 1,522 · 2,656 · 3,674 | 154.6 · 129.2 · 142.6 | 2.37 · 1.79 · 2.46 | |
| | llama.cpp 27c7a76 (`-fit`, 3 rounds) | 401 · 418 · 403 | 36.5 · 36.1 · 34.3 | — | |
| RTX 5090 IQ3_S | no-approx, off | 2,124 · 4,607 · 5,650 | 113.3 · 110.5 · 105.8 | — | 13,183 slots, 25.03 GiB (54%) |
| | no-approx, MTP | 2,086 · 4,572 · 5,622 | 189.9 · 147.9 · 177.0 | 3.01 · 2.49 · 3.05 | |
| | defaults, off | 1,955 · 4,001 · 6,500 | 121.2 · 118.0 · 114.8 | — | |
| | defaults, MTP | 1,909 · 3,996 · 6,464 | 206.7 · 177.0 · 197.3 | 2.64 · 2.13 · 2.94 | |
| | llama.cpp 27c7a76 (`-fit`, 3 rounds) | 479 · 507 · 477 | 47.1 · 47.0 · 46.6 | — | |

- **Against the predictions** (`evidence/fn0_predictions_2026-10-18.json`):
  - Strata's decode came in above them: no-approx off 90 vs 70 (RTX 5080) and 113 vs 95 (RTX 5090); MTP-matched
    131 / 109 / 119 vs 110 / 105 / 95 on the 5080.
  - Its prefill was within the band: 1.62K / 2.73K / 3.09K vs 1.6K / 3.0K / 3.0K (5080), 2.12K / 4.61K / 5.65K vs
    2.0K / 5.0K / 5.0K (5090).
  - llama.cpp's decode was above its prediction (36 vs 25–30; 47 vs 35–45) and its prefill below (≈ 400 vs 500–700).
- **FreeToken `21d9578` cannot serve these files:**
  - IQ2_XS fails with "no ggml MoE kernel for iq1_m with T=2048";
  - IQ3_S drops every prompt over 8,192 tokens despite `--max-seq-len-override 33792`. At 1K / 4K only (3 rounds), it
    runs at 907 / 3,202 prefill and 112.5 / 113.1 decode tok/s on the RTX 5090
    (`evidence/fn0_n1_freetoken_rtx-5090_2026-10-18.json`).
  - Its row stays reported-only (D11).

**Found on the way:**
- **Strata's VRAM fit is much larger than §7's Holon budget.** It reserves 700 MiB, uses no 0.9 factor, and borrows
  4.18 GiB of its cache for the prompt path, which gives 31% / 54% cached against §7's 19% / 43%.
  - Its decode hit rate is 86–89% on the 5080 and 92–93% on the 5090.
  - Holon's `lfu`, replayed on Strata's own decode routing at Strata's cache size, gives 87.8% / 95.3%, so the replay
    matches. At §7's cache it gives 78.2% / 92.7%.
- **D11's 1K premise is wrong but harmless.** Strata reads a 1,024-token prompt as a 1,023-token chunk, so its CPU
  share (on by default below 1,024; its log says mean KL ≈ 0.004) is armed. Measured with
  `STRATA_PREFILL_CPU_SHARE=0` it makes no difference: 1,623 vs 1,623 and 2,120 vs 2,124 tok/s (3 rounds,
  `evidence/fn0_n1_cpu_share_<card>`).
- **The MTP bytes differ.** Strata's runtime draft layer (`tools/mtp_rt.py`) re-quantizes the pack's BF16 dense parts
  to Q8_0. "Identical bytes" (D9) holds for the pack, not for Strata's runtime: Holon's BF16 reads are larger per
  draft.
- **What Strata's defaults gain over no-approx, beyond its tail skip:**
  - int8 KV adds +20% / +15% to 32K prefill;
  - the suffix drafter with the calibrated draft floor adds +18% / +20% to 4K MTP decode.
- **The non-lendable step peak** (`spikes/fn0/n3_peak.py`; one GDN and one QSA layer with the shipped FLA kernel, HC
  unfused) is **222 KB per token** at 2K–8K steps (predicted 200). §7's 8K-step cache on the 5080 drops from 6.7 to
  6.5 GB.

**N2** (`spikes/fn0/n2_rows.py`, `evidence/fn0_n2_rows_<card>`):
- **Zero-copy row gather.** 6.7 / 7.4 µs for 1 token, then 0.50 / 0.43 µs per token from 1K tokens on (RTX 5080 /
  5090). That is 3.4× the predicted 0.15 µs, but 4 ms per 8K step is still negligible.
- **O_DIRECT 4 KiB reads.** The SSD saturates at about 380K IOPS (1.56 GB/s) from queue depth 64 with 2 submitting
  threads; p50 is 150 µs at queue depth 64. Predicted: 450K at 64 and 600K at 256.
  - At 16 rows per token without deduplication, that serves about 24K prefill tok/s.
  - **D5's read-ahead:** queue depth 64 over 2 threads.

**N3, the proxy roofline** (`spikes/fn0/n3_roofline.py` rev 2, `evidence/fn0_n3_roofline_<card>`; method in its
docstring). Rev 2 takes panel C's review of the step-0 evidence (below):
- **Caches per row.** §7's budget at the lock's step, with the measured 222 KB/token peak.
  - RTX 5080 at 4K steps: 7.41 GB with the MTP pack for the MTP rows; 8.30 GB without it for the drafting-off rows.
  - RTX 5090 at 8K steps: 21.27 / 22.16 GB.
- **`lfu` replays on Strata's decode routing:**
  - single tokens for drafting off: 82.8% / 93.2% (5080 / 5090);
  - 4-token window unions for MTP verify passes (26.9 distinct experts per layer per pass): 71.6% / 89.1%.
- **MTP rows.** A pass reads the dense bytes once and the union's experts once, plus 3 draft steps of the MTP head
  (its 0.18 GB dense parts, the full-vocabulary head, 10 Q2_0 experts). It yields Strata's recorded accepted tokens
  per pass from the matched rows.
- **Prefill factors:** X's F5a fetch rate (40.3 GB/s) and per-step overhead (2.64 ms at 16 launches per layer).
  Strata's phase shares are scaled by HC ×1.5, GDN ×1.1, QSA ×1.2 and mmq ×1.0 (assuming the MoE prefill track's
  D1–D5).
- **Strata's "dequant" phase is relabelled** as the per-expert gather and its issue gaps; it is present on IQ3_S too.
  Rev 1 read it as IQ1_M's dequantization at ×0.5.
  - **Rev 3 factor: ×0.3 [0.1, 0.5].** Holon's arena mmq reads the cache slots in place, so there is no gather; only
    the issue gaps remain (panel C).
- **Other bounds and gains:**
  - the PCIe streaming bound of a step's routed non-resident experts;
  - q8_0 KV gaining Strata's own int8 gain in the defaults rows (`qsa_attend@1` reads q8_0 in-kernel, §5);
  - **rev 3 adds the MTP head's own cost** over each pass's verify rows: 0.275 ms (panel C: 0.25–0.3).
  - **rev 4 (step 4, panel C) replaces the QSA guess** (0.24 ms per pass beyond 2,051 tokens) with the measured
    selection + sparse attention of the 12 QSA layers: per layer 41.7 / 44.0 µs for a decode row at 4K / 32K and
    56.7 / 65.9 µs for a 4-row verify pass (RTX A5000, real indexer inputs, cold caches; §14), scaled to the
    50-series by ×0.67 [0.55, 0.85] (their clocks and bandwidth): ≈ 0.34 / 0.35 ms per decode token, 0.46 / 0.53 ms
    per pass.
  - **rev 4 carries a launch band.** The overhead is scaled to 16 launches per layer on average (step 7's gate, ≤ 768
    per decode token); each decode cell is also given at ≈ 22, today's layer-weighted projection, and the band's upper
    end is 24 (every layer at the QSA layers' count; §14's launch table). The fusions that close the gap have owners
    (§14).
- **Bars: decode 1.0×, prefill 0.9×** (acceptance 5). Rev 2's table measured every cell against 0.9×; that is
  corrected here.
- **Holon / Strata, central [band] → at ≈ 22 launches per layer (rev 4; supersedes revs 1–3).** The central and band
  assume step 7's 16 launches per layer on average; the arrow gives the central at today's layer-weighted projection
  (≈ 22). At the band's upper end (24) the 5080's drafting-off cells are 0.92 · 0.90 · 0.96.

| Gated row (bar) | RTX 5080: 1K · 4K · 32K | RTX 5090: 1K · 4K · 32K |
|---|---|---|
| no-approx, drafting off, decode (1.0×) | 1.032 · 1.005 · 1.071 [0.93–1.16] → **0.95 · 0.92 · 0.98** | 1.13 · 1.12 · 1.17 [1.01–1.30] → 1.00 · **0.995** · 1.04 |
| no-approx, MTP-matched, decode (1.0×) | **0.89 · 0.90** · 1.06 [0.86–1.11] → **0.85 · 0.86** · 1.02 | **0.97** · 1.02 · 1.03 [0.92–1.09] → **0.91 · 0.96 · 0.97** |
| no-approx, prefill (0.9×) | 1.19 · 0.99 · 0.93 [0.73–1.36] | 1.05 · 1.03 · 0.97 [0.76–1.18] |
| *RTX 5080 at step 7's memory target:* drafting off · MTP-matched decode | 1.06 · 1.03 · 1.10 · **0.93 · 0.95** · 1.12 → 0.97 · 0.94 · 1.01 · 0.89 · 0.91 · 1.07 | — |

| Report-only row (lead's recommendation, owner item) | RTX 5080 | RTX 5090 |
|---|---|---|
| defaults vs Holon + A1, drafting off, decode | as no-approx | as no-approx |
| defaults vs Holon + A1, MTP, decode | 0.84 · 0.83 · 0.92 | 0.95 · 0.91 · 1.02 |
| defaults vs Holon + A1, prefill | 1.25 · 1.01 · 0.93 | 1.14 · 1.18 · 0.97 |

- **Gated cells predicted to miss their bar (central, 16 launches):**
  - **RTX 5080 MTP-matched decode at 1K and 4K** (0.89, 0.90). They still miss at step 7's memory target (0.93,
    0.95).
  - **RTX 5090 MTP-matched decode at 1K** (0.97).
  - The 5080's 4K drafting-off decode is 1.005. Rev 4's first cut put it at 0.99: an artifact of carrying the
    A5000's QSA times to the 5080 unscaled.
  - Per D11, these go to the owner when they are measured.
  - Every prefill cell clears 0.9× at the center.
- **At ≈ 22 launches per layer** (today's layer-weighted projection), every drafting-off decode cell of the RTX 5080
  misses (0.95 / 0.92 / 0.98) and the 5090's 4K does (0.995); its MTP cells miss at every length. Step 7's ≤ 768 per
  token gate and the named fusions (§14) are what keep these cells at their central values.
- **Model error.** Cells within a few percent of their bar (rev 2's 0.896 / 0.899; here the 5080's 4K drafting-off
  decode at 1.005 and 32K prefill at 0.93, and the 5090's 1K MTP at 0.97) sit inside the model's error band. They are
  re-predicted after step 7 from measured costs; the QSA term and its scale are measured on the 50-series in the
  cluster session.
- **Levers on the MTP decode cells beyond memory** (panel C; candidates, not commitments):
  - **F5b's CPU split in the RTX 5080 lock.** Lead decision 5 allows it once the judge's path-aware floor passes; it
    is a step 6–7 candidate.
  - **MTP experts in the store** instead of resident: ≈ +0.7 GB of cache (§7).
  - **A multi-token F5b CPU path** that reads each weight once per verify window, as Strata's pool does.
    - Panel C's estimate: at equal cache, Holon's MTP gain over drafting off is 1.40 against Strata's 1.47, so closing
      that gap is worth up to ≈ 5% on these cells.
    - A candidate for step 7, not a commitment.
- **Causes:**
  - The MTP cells: §7's smaller cache (the 4-token verify unions miss more) and Strata's multi-token CPU pass.
  - The defaults MTP rows: also Strata's suffix drafter and calibrated draft floor, which Holon does not have (a
    non-goal). Rev 1's "int8 KV" cause for the defaults prefill was mostly a modelling gap.
- `stable_topk` moves to `std.mixers.qsa`'s reference once step 3 writes it (panel A, optional).

**The oracle check (N4's 32K run)** (`spikes/fn0/n4_oracle.py`):
- **The judge reads FN's GGUFs** (`judge/holon_judge/gguf.py`, `golden.py`, `moe.py`): the `qwen4exp` name map, split
  files, the indexer's fused q/k, lazy PLE rows with the hash constants checked, the scoped tie rule, the mixer as the
  final norm, and the head in row blocks.
- **On the tiny model the streamed golden equals HF's whole forward bit for bit** (`judge/tests/test_golden_qwen4exp.py`,
  in the golden venv).
- **The 32K `recall-32k` prompt (31,878 tokens) on IQ2_XS, CPU, 32 threads, under a 180 GB cap** (predicted: 2 h
  [1–4], 50 GB [30–80], query blocks needed):
  - It finished in 5,504 s (1.5 h) at a peak RSS of 47.1 GB. A GDN layer takes a median 87 s and a QSA layer 172 s;
    the dequantization of each layer's experts dominates.
  - HF's attention did not materialise [24, T, T] (torch's CPU SDPA tiles a boolean mask), so **D2's query-block mode
    is not needed**, and the golden's metadata records the one-shot mode.
  - **Tie rate on the real model:** 57 boundary ties in 357,924 sparse selections (0.016%) over the 12 QSA layers. The
    tie rule rarely changes the full model's selections, unlike the tiny model's ReLU sums.
  - Record: `evidence/fn0_n4_oracle_cpu_2026-10-18.json`.

**Lead decisions on the stop (2026-10-18):**
1. **Decided by the owner (2026-10-20): the Strata defaults rows are reported, not gated.** The gated rows are the
   no-approx and MTP-matched ones, which compare engine with engine; the defaults rows measure features outside FN's
   scope (the suffix drafter is a non-goal, and int8 KV is a format choice). (Was: the lead's recommendation, pending
   the owner.)
2. **The memory fit becomes a perf-closure target (step 7)**, restated after panel C (M1):
   - **The target:** ≥ 23% (≈ 8.2 GB) of the experts cached on the RTX 5080 with the MTP pack resident. It is set
     against Strata's MTP arm (7,016 slots, 28.5%), not its drafting-off arm's 31%.
   - **Three levers:**
     - **4K steps** on the 5080 lock (lead decision; D12d, D16): +0.9 GB.
     - **A fixed reserve in place of the 0.9 fraction:** +≈ 0.8 GB at 0.97.
       - The core's existing `memory_fraction` and `state_memory` settings already reach it (`EngineConfig`,
         `engine.py:75-76`), so no core change is needed.
       - A reserve stated in MiB would be a new core option. **K/FN core-round item.**
     - **`token_embd` on the host:** +0.34 GB.
   - **Row at risk:** RTX 5080, no-approx decode, 0.95–1.01× in rev 1. In rev 2 the risk shifts to the 5080's MTP and
     32K prefill rows (above).
3. **`STRATA_PREFILL_CPU_SHARE=0` in the no-approx rows** (D11 corrected; `spikes/fn0/arms_strata_<card>.json`).
4. **FreeToken is N/A per file** (reported only anyway):
   - IQ2_XS: no ggml MoE kernel for IQ1_M;
   - IQ3_S: prompts over 8,192 tokens are dropped, so its 32K cell is N/A; 1K and 4K are recorded.
5. **The tiny GGUF is rebuilt, not fetched.** The small fixtures are committed; a pinned script rebuilds the GGUF and
   checks its sha256; CI skips only the GGUF-dependent part (D2's deviation).
6. **Step 2's remainder continues; steps 1 and 3 wait** for milestone K's core seams.

**Step 2, done** (2026-10-18; every part that needs no core seam):
- **The golden venv** `local/envs/golden-qwen4`: transformers 5.19.0, tokenizers 0.23.3, and `-e` installs of judge,
  std and testing with `--no-deps`.
- **The judge edits of D2:**
  - the `qwen4exp` name map and value rules;
  - split files, with every shard hashed in `file_meta`, and gguf-py's version plus its `quants.py` sha256;
  - lazy PLE rows, with the hash constants checked against HF's;
  - the tie rule, scoped to the indexer's call;
  - the hyper-connection mixer as the final norm, and the head in row blocks;
  - MoE found by its block, and the shared expert in the floor's mirror;
  - floors that pass unquantized tensors the map lacks.
  - Query blocks are not needed (N4).
- **gguf-py at LC `27c7a76`** in the workspace lock. Its `quants.py` only adds Q1_0 / Q2_0, so existing types
  dequantize as before. The locks' relock is FN's single propagation (D16).
- **Q2_0 on the CPU path:** `q2_0` is in `Q8_0_DOT` and in `kernels.ggml_cpu`'s `cpu_experts@1`, using ggml-cpu's own
  Q2_0 · Q8_0 dot; L0 covers a synthetic class. The judge's mmq / mmvq blockings already match Q2_0: D4, no stored
  sums.
- **`v1-long`** (sha256 `d8858951…`).
- **The tiny model:**
  - its GGUF and MTP pack are rebuilt byte for byte to pinned sha256s;
  - the fixtures are committed in `testing/holon_testing/qwen4exp_fixtures.safetensors`: two prompts with 31 boundary
    ties;
  - in the golden venv, the streamed golden equals HF's forward bit for bit.
- **N4** is done.

**The pooled MTP reference is done too** (panel B F6 corrects this record: SGLang 0.5.21's `qwen4_exp_mtp.py` was in the
local uv cache all along):
- `judge/holon_judge/qwen4exp.py` (`mtp_pooled`, `pack_state`) transcribes it onto transformers' modules.
- It reads the tiny Strata-format pack as the checkpoint's tensors within 1e-4. The pack's Q2_0 experts keep the
  source layout (gate rows before up rows, expert order).

Steps 1 and 3 wait for milestone K's core seams.

**Step 2's code review (panel A: architecture; panel B: correctness), dispositions:**

| # | Finding | Disposition |
|---|---|---|
| B-F1 (major) | `_LazyRows` ignored the golden's dtype; the bf16 floor and the shared-expert mirror were untested | **Accepted.** `LazyRows` returns the golden's dtype. New tests (golden venv): a CPU bf16 floor of the tiny model runs to completion (top-1 > 90% of the fp32 golden's), and `moe.mirror` on an fp32 tiny block matches HF's `SparseMoeBlock` within 1e-6, with the gated shared term nonzero. |
| B-F2 | The Q2_0 L0 never exercised code 3 (+2): ggml's quantizer (d = amax) does not write it | **Accepted.** The synthetic Q2_0 class is built from random code bytes and fp16 scales, and the test asserts that code 3 occurs. |
| B-F3 = A-m2 | The tie rule was recorded only on the GGUF path | **Accepted.** `archs.prepare` runs wherever the golden builds a model (GGUF, streamed HF, whole HF), and records the deviation whenever an indexer is wrapped. |
| B-F4 | The fixtures read the indexer's mask as a bool | **Accepted.** `tiny_hf` pins `attn_implementation="sdpa"`, and the fixtures also accept eager's float mask (`mask == 0`). The fixtures are unchanged. |
| B-F5 | A quantized indexer projection failed with a bare KeyError; some readers saw only shard 1 | **Accepted.** `_linear` refuses a quantized `indexer.q_proj` / `k_proj` with a clear message. `emulate_q8_1`, `q8_1_experts`, `hf_types`, `engine_shapes` and `mmvq_step` read every shard (`_tensors`). |
| B-F6 | SG's module was available; do the pooled MTP reference now | **Accepted.** `qwen4exp.mtp_pooled` and `pack_state`; the tiny pack checked within 1e-4 (above). |
| A-m1 | Qwen4Exp specifics in the generic `golden.py` | **Accepted.** `judge/holon_judge/qwen4exp.py` holds them (the tie rule, lazy PLE rows with the hash check, the mixer as the final norm, the MTP reference); `golden.py` calls `archs.prepare` / `archs.final_norm`. |
| A-m3 | Two definitions of the tie rule; a global `Tensor.topk` patch in the fixtures; the test wrapped a shared model | **Accepted.** One definition (`qwen4exp.wrap_indexers`, scoped to each indexer's call, with an optional tie counter). The fixtures import it (`holonlm-testing` now depends on `holonlm-judge`), and the tests wrap models of their own. |
| A-m4 | std's `gguf>=0.17` floor vs Q2_0 | **Accepted: release item.** A user's PyPI gguf lacks Q2_0, so either std gets its own size entry for type ids gguf-py lacks, or the floor rises once a PyPI gguf with Q2_0 exists. No code change now. |

**Step 0's evidence review (panel C), dispositions** (the `-R2` rows are the panel's second round, on the step-2 fixes and roofline rev 2):

| # | Finding | Disposition |
|---|---|---|
| C-M1 | Step 7's memory target measured against the wrong Strata arm | **Accepted.** ≥ 23% (≈ 8.2 GB) with the pack, against Strata's MTP arm's 28.5%. Levers: 4K steps (decided; D12d, D16), a fixed reserve through the existing `memory_fraction` / `state_memory` (a MiB-reserve option is a K/FN core-round item), and `token_embd` on the host. The roofline's drafting-off rows are sized without the MTP pack. |
| C-M2 | q8_0's gain left out of the model | **Accepted.** `qsa_attend@1` reads q8_0 in-kernel in prefill, decode and verify (§5). The defaults rows model q8_0's gain, and the escalation text now names the suffix drafter as the real cause. |
| C-m | Strata's "dequant" phase mislabelled | **Accepted.** Relabelled as the per-expert gather and its issue gaps, at ×1.0 [0.5, 1.5]. |
| C-m | The MTP rows were not modelled | **Accepted.** `lfu` is replayed on 4-token window unions, with the recorded acceptance and the per-draft head and dense bytes. |
| C-R2-M (major) | The decode bar is 1.0×, not 0.9× | **Accepted** (rev 3). The gated table is restated against 1.0× for decode and 0.9× for prefill. The 1K / 4K MTP cells are predicted to miss, even at step 7's memory target; levers beyond memory are named, and the cells go to the owner when measured (D11). |
| C-R2-m | The gather factor | **Accepted:** ×0.3 [0.1, 0.5]. Holon's arena mmq reads slots in place. |
| C-R2-m | The MTP head's cost over the verify rows | **Accepted:** 0.275 ms per pass. |
| C-R2-m | The 0.896 / 0.899 cells | **Accepted:** they sit inside the model's error band and are re-predicted after step 7 from measured costs. |
| A-R2-m | The Q2_0 release item; the judge's std dependency; `stable_topk`'s home | **Accepted.** Plan §10 has the release item. `judge/pyproject.toml` declares `holonlm-std` (pre-existing: the judge imports `holon_std` at 40 sites; relocked). `stable_topk` moves to `std.mixers.qsa`'s reference in step 3. |
| B-R2-n | The mirror test should show the shared term is nonzero | **Accepted:** it asserts `shared.abs().max() > 1e-4`. |
| C-m | N1's no-approx arms ran with the CPU share on | **Accepted, noted.** N1 ran with Strata's CPU share on (the as-run arms, with their environments, are inside each `evidence/fn0_n1_*` record's `arms`). It measured as immaterial (`evidence/fn0_n1_cpu_share_<card>`). The committed `spikes/fn0/arms_*` files now set it off. |

## 14. Step 4 record: the kernel batch before the core seams (2026-10-19)

**Status: done on sm_86, the panel's review applied (below). Deferred:** the sm_89 / sm_120 L0 runs and the
`qsa_select` target on the RTX 5080 (one cluster session); the MoE pieces (IQ1_M, Q2_0 on the GPU, lending); and
anything that needs C1–C3 or the unit (PLE's in-step payloads and replay rewind, the whole layer's launch count). No
core edits.

**What landed:**
- Op contracts with their oracles (`reference.py`) and CPU providers (`testing.ref_kernels`):
  - `std.hyper`: `hc_gr_mix@1` (a block's attention-half combine rides in its FFN-half mix), `hc_gr_combine@1`.
    R is complete at every block boundary (the contract): the trunk, taps, C1's consumers, drafters and the output
    mixer never see a pending combine. They are named for the gated-residual form, so GLM's mHC adds siblings.
  - `std.mixers.qsa`: `qsa_index@1`, `qsa_select@1` (shape-invariant), `qsa_attend@1` (every gqa layout read in the
    kernel, q8_0 included; the output gate applied at the store). Precondition: each sequence's tokens are
    consecutive rows in position order.
  - `models.qwen4_exp`: `qwen4_exp.ple_rows@1`, `qwen4_exp.ple_block@1` (contracts and oracles only; numerics
    undeclared, as the other model plugins).
  - GDN: `gated_rmsnorm`'s `out_gate` fact.
- Triton kernel plugins: `kernels.hc_triton`, `kernels.qsa_triton`, `kernels.ple_triton` (each `bounded`, cuda).
  `qsa_attend` and its split merge are adapted from FreeToken 21d9578 (Apache-2.0), with provenance and notices. The
  rest is ours, written after reading FreeToken's and Strata's kernels.

**Checks, all passing:**
- The oracles against HF's modules, in float32 on the tiny shapes, in the golden venv
  (`plugins/tests/test_fn_oracles.py`): the gated residual (both mixers); PLE's rows bitwise (EOS cases, chunks); PLE's
  block (dilation 3); the indexer's mask under D2's tie rule, with ties occurring; the attention under that mask.
- L0 on the RTX A5000 (`test_qsa_kernels.py`, `test_hc_ple_kernels.py`):
  - HC mix within `l0` (bf16 / f32 over n·H terms) at 0–77 tokens, with and without inject (f32 beside bf16 `down`)
    and a pending combine; the residual it leaves bitwise what `hc_gr_combine` leaves; a whole block as the unit
    runs it (R complete on return) within `l0` of the oracle's block;
  - PLE rows bitwise; the PLE block within `l0(f32, H)`, plus acceptance 2's GPU unit test (chunked and 1-token steps
    bitwise one shot; replay rewind waits for C2);
  - `qsa_index` in one launch (raw keys bitwise, pooled within `l0`, blocks completed across and within calls; k in
    bf16 and in f32, pooled as raw stores it);
  - `qsa_select` exactly equal to the oracle on tie-heavy keys (a boundary tie at ReLU's 0.0 asserted), a valid top-k
    on continuous keys, 8,192 and 8,195 blocks; shape-invariant (a query's scores and selection bitwise alone and in a
    128-row call); sub-batching bit-neutral;
  - `qsa_attend` within the attention budget for bf16 / fp16 / q8_0 (fp8 from sm_89), gated and not (NaN gates on
    padding rows still give 0), at 32, 8, 4 and 1 splits; the plant "the open block reads p + 1" fails.
- Memcheck is clean (`tests.memcheck_fn_kernels`).
- GDN's silu path is bitwise main's: a Qwen3.8-27B UD-IQ3_S snapshot (the v1-qwen3_5 trace, 354 steps, fixed pools)
  is byte-identical to main's GDN code, and the kernel's silu outputs equal main's on 25M elements.

**Choices made while tuning** (measured on the RTX A5000):
- **Scores:** float32 products summed within a thread, one program per (query, 64 blocks), one warp. `tl.dot` is out:
  `ieee` runs 2× slower at 1 row, and Triton rewrites a broadcast product-sum into a TF32 dot (relative error 1e-3:
  not IEEE). Scoring several rows per key tile (M2) was measured three ways (the rows one after another over an
  L1-resident tile, an `ieee` dot over 4–16 rows, an outer product per dim) and none beat one program per row from 2
  to 512 rows: the scores are bound by issue, not by key reads, and rows of a tile run together and share it through
  L2. So `_score` keeps one program per row; the tile is fixed (shape invariance).
- **Top-k:** the k-th key exactly, a bit per round with an early exit, from registers up to 8,192 blocks. Measured and
  dropped (m2): 2 bits a round (3 thresholds packed in one reduction; 16.8 against 14.0 µs on real scores), skipping
  the leading bits a lower and an upper bound share (no gain on real scores, slower on synthetic ones), Triton's
  `tl.histogram` and `tl.topk`. A CUDA radix select stays a step-7 contingency.
- **Attend:** split by shape alone (32 ways for decode, none for prefill). The variant measured for M1 — the last
  split to finish merging every split of its kv head's 12 query heads itself, serially — saved a launch but cost 24 µs
  at decode: that variant is rejected (a merge spread over more programs was not tried). The output gate rides in the
  store instead, which saves gqa's two gate launches.

**Launches per layer (decode).** Step 4's kernels, counted by the profiler at a decode shape. The rest of the layer
(projections, q / k norms and rope with the KV write, o_proj, the MoE's router, top-k, experts and shared expert, and
GDN's conv / rule / norm on the GDN layers) is today's units, counted at step 5 when the unit runs. Full layers project
to ≈ 20–24 launches, ≈ 22 layer-weighted (QSA layers ≈ 24, panel C); N3 carries the 16–24 band with 22 as its
projection point (§13, rev 4). The step-7 gate is a per-token average: ≤ 768 launches per decode token (48 × 16,
acceptance 3).

| Piece | Before (2a761fb) | Now |
|---|---|---|
| HC: two mixes, two combines | 10 (rstd, cat, down, up ×2; combine ×2) | 5 (down, up ×2; the attention half's combine rides in the FFN half's mix; the block's last combine before it returns) |
| QSA: index · select · attend · output gate | 2 · 3 (with a fill) · 2 · 2 (gqa's sigmoid, mul) | 1 · 2 · 2 (1 for prefill) · 0 |
| PLE (layer 1 only): rows · block | 2 · 3 | 2 · 3 |
| Step 4's share of a QSA layer | 19 | 10 |

The fusions that close 20–24 → 16, with owners:
- **FN, step 7:** GDN's fused input projection; QSA's q, kv and indexer projections in one call, so the QSA layers
  are not left near 19; the block's last HC combine folded into the MoE combine's epilogue, within the same block only
  (R complete on return: the trunk contract); router and top-k in one launch.
- **FN, step 7, after the MoE prefill track has landed:** the shared expert run as an extra expert slot in the routed
  calls. It edits `kernels.ggml_cuda`, so it goes through the MoE prefill track's ownership rules and FN's own
  propagation.

**QSA latency** (`evidence/fn4_qsa_rtx-a5000_2026-10-19.json`, `..._4k_...`; `bench/fn_kernels.py`): CUDA-graph replay
on N4's real indexer inputs (layer 3 of `recall-32k`, 31,878 tokens, 7,969 blocks; layer 7 within 1.5 µs), every
per-layer buffer in 24 copies (1.6 GB) so no call finds its keys in L2; `qsa_attend` gated on bf16 KV.

| Query rows | `qsa_select`, one sequence | one per sequence | `qsa_attend`, one sequence |
|---|---|---|---|
| 1 | 25.3 µs | — | 18.6 µs |
| 2 | 27.3 µs | 45.1 µs | 22.5 µs |
| **4 (the target's shape)** | **28.8 µs** | 57.5 µs | 37.1 µs |
| 8 (reported) | 32.5 µs | 95.6 µs | 48.5 µs |
| 1 / 4 at 4K (1,024 blocks) | 23.4 / 22.2 µs | — | 18.2 / 34.5 µs |
| 512 (a prefill sub-batch) | 1,041 µs | — | 2,610 µs |

- **Target: ≤ 20 µs at 4 rows of one sequence on the RTX 5080** (restated, acceptance 3); not yet measured there. On
  the A5000 it is 28.8 µs: the top-k's ≈ 14–17 µs (31 dependent block reductions at ≈ 0.45 µs) and ≈ 10 µs of scores.
  N3 scales the QSA term to the 5080 by ×0.67 [0.55, 0.85] (its clocks and bandwidth), which puts this near 19–20 µs:
  tight. To measure in the cluster session: a fixed 32-block score tile (more programs per row; still one tile for
  every shape, so still shape-invariant). The contingency is a CUDA radix top-k, then the score epilogue narrowing the
  top bits.
- **N3** now uses these numbers for its QSA term (§13, rev 4).

**The panel's review of 5b2576f and 2a761fb, dispositions:**

| # | Finding | Disposition |
|---|---|---|
| B (major) | GDN's silu path changed its expression (`z/(1+e)` → `z·(1/(1+e))`) | **Fixed.** silu keeps the original expression; the new form is SIGMOID's only. Checked against main: a Qwen3.8-27B snapshot byte-identical, and the kernel's outputs equal on 25M elements. (5b2576f's form gave the same bits in both checks too: the drift was latent here, not observed.) |
| B-1 | `qsa_select` shape invariance | **Fixed.** One tile, chunk and warp count for every shape; stated in the contract; an L0 checks a query's scores and selection alone against a 128-row call, bitwise. |
| B-2 | A 0.0 tie; an attend mutation | **Done.** `test_select_ties` asserts a boundary tie at 0.0 (a zero key for 85% of blocks); the plant "open block reads p + 1" fails `test_attend`. |
| B-3 | inject / down dtypes | **Fixed.** Each weight is read in float32 (no cast of inject to down's dtype); the L0 uses an f32 inject beside a bf16 down. |
| B-4 | The fp16 comment | **Reworded** as triton_attn's. |
| C-M1 | Launches | **Done in part.** `_rstd` folds into `_down`, the attention half's combine into the FFN half's mix (5 HC launches per layer, after A's major), `_write_raw` + `_pool` into `_index`. The serial last-split merge variant was measured (+24 µs at decode) and rejected; the output gate folds into the attend's store instead (same count). Table above; the step-7 gate (now ≤ 768 launches per decode token). |
| C-M2 | Key reads per tile | **Measured, not adopted:** three row-grouped forms, none faster from 2 to 512 rows on sm_86 (above). Shape invariance holds. |
| C-m1 | Measurement | **Done:** 24 copies of every per-layer buffer (1.6 GB) and real indexer q / k from N4's oracle run (`spikes/fn0/n4_oracle.py --capture`). |
| C-m2 | Top-k variants | **Measured, dropped** (above). The CUDA radix top-k stays a step-7 contingency. |
| C-m3 | N3's QSA term | **Done:** rev 4 of §13. One more gated cell is predicted to miss: the RTX 5080's 4K drafting-off decode (0.99; 1.01 at step 7's memory target). |
| C | Acceptance 3's target | **Restated:** ≤ 20 µs at 4 rows of one sequence; 8 rows reported, not gated. |
| A (major, round 2) | A layer's last combine carried into the next layer's mix (and the output mixer) left R incomplete at block boundaries: taps, C1, the MTP head and `_logits`' copy-or-not would see or double-apply it | **Fixed** (lead decision): fusion within a block only. The contract states R is complete at every block boundary; the attention half's combine rides in the FFN half's mix and the block applies its last combine before returning (5 HC launches per layer). A test runs a whole block that way against the oracle's. |
| A-minor (round 2) | `w` both an input and an output | **Renamed** `y_prev`, `w_prev`; the contract states `w_prev` is only read and the returned `w` is always a new tensor, so they never alias. |
| B-1 (round 2) | Fused `qsa_index`: in-call k in k's dtype; an unstated precondition | **Fixed:** in-call k reads are cast to raw's dtype before the mean (tested with f32 k and bf16 raw); the contract states that each sequence's tokens are consecutive rows in position order. |
| B-2 (round 2) | `_merge`: a NaN gate on a padding row gives NaN | **Fixed:** the gate multiply is masked by `total > 0`; the test puts NaN in padding rows' gates and expects 0. |
| C (major, round 2) | Launch budget: full layers ≈ 20–24, not 16 | **Done:** N3 carries a 16–24 band (§13 rev 4: the 24-launch centrals), and the closing fusions are named with owners (above). |
| C (round 2) | The QSA term on the 5080 | **Done:** ×0.67 [0.55, 0.85]; rev 4's first-cut 0.99 for the 5080's 4K drafting-off decode was an artifact of the unscaled A5000 input (now 1.005). |
| C (round 2) | A fixed 32-block score tile; 512 rows; the merge wording | **Done:** the tile is a cluster-session measurement; select at 512 rows recorded (1.04 ms); "rejected" scoped to the serial last-split variant. |
| C (follow-up) | The band's projection point; the gate's form | **Done:** N3 adds ≈ 22 (layer-weighted) as the projection point, 24 the band's upper end; the gate is ≤ 768 launches per decode token (48 × 16); QSA's projections in one call named as an FN step-7 fusion. |
| A-1 | PLE's kernels | **Moved** to `kernels.ple_triton` (cuda, `bounded`); the model plugin keeps the contracts and oracles, numerics undeclared. |
| A-2 | Model constants in std | **Removed** (`RATIO`, `BUDGET`, `SUB_BATCH`). |

## 15. Step 1 record: the core seams (2026-10-19)

**Status: built, for the panel's code review.**
- **Base:** milestone K's head (0aad1ab, ABI 1.4), with FN's committed work merged on top (561e3e7). The branch is
  `fn-core`; nothing is merged to main.
- **ABI 1.5** (additive over K's 1.4): `Residual` and `ModelSpec.residual`; `BuildCtx.unit_class`. `core/abi.txt`
  is regenerated.
- **Core code lines:** 4,967 against the base's 4,940, so **+27** of the ≤ +47 budget (+23 before the panel's
  round). If the owner drops K's `SeqView.matched` (+6 at K's count), FN adds it in this round: about +33, still
  inside the budget.

**The seams:**
- **C1, residual width:** `ModelSpec.residual: Residual(width, dtype) | None`. The trunk, taps and `Drafter.run`
  carry `[T, width]` of that dtype; the final norm returns `[T, hidden]`. The startup profile checks the trunk's real
  output against the declaration (a mismatch fails there) and feeds the drafter's profile pass the declared shape.
- **C2, state declared by the block:** `bind_state` asks `ModelSpec.block`'s class too, over all layers, through the
  same `state_specs` / `planner` hooks as a mixer's.
  - On this new path, a group of Slot state without a planner is refused at bind.
  - A mixer's planner-less group keeps today's default planner, plus one stderr line when CUDA graphs are on: its
    steps run without them.
  - A group declared twice (two mixers, or a mixer and the block) is refused.
- **C3:** `BuildCtx.unit_class(ref)` returns the registered class serving a Ref. The core's internal `unit_factory`
  is renamed to it, so it adds no lines.
- **Denylist:** `check_core_denylist.py` gains `qsa|ngram|indexer`.

**Toy proofs** (`core/tests/test_block_seams.py`, toys in `holon_testing.toy_streams`, on a tiny Qwen3):
- **C1:** two float32 streams `[T, 2H]` (a wide embed, a wide block, a collapsing norm). On the CPU, eager:
  greedy output equals a model-only build's; prompt logprobs come back; the drafter and a layer-0 tap see
  `[T, 128]` float32. On the GPU: the same from CUDA graphs, decode and verify steps graphed, the startup profile
  passing. A model declaring 3H while its trunk returns 2H fails at startup with the profile's message.
- **C2 + C3:** the block forwards a nested Ref's hooks (`ctx.unit_class(pre)`): a Slot running sum of the token ids
  at layer 1 (`recurrent` kind, alignment 64, `in_step` 64), which adds (sum mod 97) / 97 to the residual, so a wrong
  state changes the logits. Beside it, layer 0's mixer is a second running sum in its own group.
  - A 200-token prompt in steps of at most 100 tokens is cut at 64 and 128 (the budget rounded to the grid) and
    checkpointed at 192 in-step.
  - A second request with that prefix hits at 192 and adopts the payload.
  - Speculative verify steps (prompt lookup) accept drafts, and `rewind` restores the sum after the accepted tokens.
  - On the CPU, greedy output equals the model-only build's. Three plants fail it: no rewind, no in-step publish,
    and `has_init` ignored.
  - On the GPU, the decode and verify graphs are captured.
- **The planner rule:** the plain wide block, which forwards specs but no planner, is refused at bind. A
  planner-less mixer group gets the default planner and the stderr line (a direct `_bind_layers` check).
- **Deviation from §4's toy text:** the toy mixer beside the block has alignment 64, not 16 (panel A, below: one
  alignment for every recurrent spec).

**C2's alignment rule.** Every non-paged (recurrent) spec of a model must share one alignment. The prefix index files
payloads per block of lcm(page, every alignment), and a kind checkpoints on its own lcm(page, alignment) grid, so a
spec of a smaller alignment publishes off the block grid and no hit can use it. The core says so at startup
(`prefix_index`, one stderr line); a mixer at alignment 16 beside the block's 64 shows it in the toy test.

**The panel's review of step 1, dispositions:**

| # | Finding | Disposition |
|---|---|---|
| A-1 | A mixer-declared recurrent spec beside the block's | **Done:** layer 0's mixer is a second running sum (`toy.sum.mixer`, its own group, alignment 64, `in_step` 64, in place of that layer's attention). Both sums publish in-step at 192, and b's hit is still 192. |
| A-2 | Recurrent specs must share one alignment | **Recorded** (above); `prefix_index` warns when non-paged specs' alignments differ (+2 lines); tested at 16 beside 64. |
| A-3 | Nits | **Done:** the planner-less warning is not repeated by `sub_planners` (`plan_only`: tested); `ModelSpec.residual.dtype` must name a torch dtype (a `PluginError` at build); the toy planner-less mixer uses `Paged`; `Model.trunk`'s docstring reflowed. |
| B-1 | The startup assert on real engines | **Done:** before merge, D16's propagation starts every lock family's bf16 GPU engine, which runs the startup profile and its check; that is the startup-assert check of record. Run now on an RTX A5000: Qwen3.8-27B UD-IQ3_S (residual (5120, bf16), 9 graphs) and Qwen3-30B-A3B Q4_K_M ((2048, bf16), 9 graphs) start and decode. |
| B-2 | Which state is bound | **Stated** in `AuxSpec`'s and `DeclaresState`'s docstrings: the target's mixers and block, and auxiliary models' mixers; an aux unit declares none. |
| B-3 | The block's state through a tier | **Done:** with the `tier` slot (`testing.toy_tier`), a filler request pushes a's blocks off a tight device pool; b hits at 192, its blocks loaded from the tier and both sums' checkpoints adopted from the tier's copies; greedy output equals the model-only build's. |

## 16. Step 3 record: std units and the model plugin (2026-10-19)

**Status: built, for the panel's code review.** No core edits in this step: the core stays at +27 code lines over the
base, ABI 1.5.

**What step 3 adds:**
- **std units:** `std.hyper.expand@1`, `gr_mix@1` (the model's norm) and `gr_block@1` (R complete at every block
  boundary; `pre`'s state hooks forwarded by `ctx.unit_class`); `std.ffn.sum@1` and `std.ffn.gated_swiglu@1` (D7);
  `std.mixers.qsa@1` (D6; `stable_topk` now lives in its reference, the judge imports it); `std.spec.mtp_hc_head@1`
  (D9).
- **`models.qwen4_exp`:** the GGUF recipe (`residual = Residual(n·H, "float32")`, the converter's w + 1 norms and
  `ssm_a = −exp(A_log)` undone by value transforms, the PLE table padded to a multiple of 128 rows as the converter
  keeps it), `qwen4_exp.ple@1` (D4), and `draft_head("mtp")` over Strata's pack.
- **Edits:** `drafters.mtp` (a head whose `carries` is `"residual"` takes and carries R, `lm_head(head.out(x))`; the
  plugin option `mtp.path` reads the head's weights from their own file); `drafters.dflash` refuses a target whose
  residual is not hidden-size; `std.weights.gguf` reads split files (D14); `out_gate` and Q2_0 on the CPU were in
  the step-4 batch.

**PLE (`qwen4_exp.ple@1`):**
- **State:** two Slot specs of kind `recurrent@1` in group `ple`: `ple.history` (the last two ids, int64) and
  `ple.conv` (the last (K − 1)·dilation = 9 rows of C, f32), alignment and `in_step` 64. The recipe's 64 is asserted
  against `std.mixers.gdn`'s `CHUNK` in the CPU test.
- **`qwen4_exp.ple_block@1` returns C [T, n·H] f32** (the approved contract change): the oracle, the CPU provider, the
  Triton kernel and its L0 return it. In-step checkpoints are filled from the state before the step and the step's
  own ids and C.
- **Verify rows** run on a scratch copy of their rows' state, so the real state stays as before the step; `rewind`
  writes the window and the two ids at 1 + n_accept by gathers from the saved state and the step's ids and C.

**Decisions taken while building (for the panel):**
- **D5 moves to step 6.** The PLE table is read through `std.ops.embedding@1` (resident: dense, or quantized through
  `qembedding@1`), a `table` Ref of the unit. `std.tiered.rows@1` exists only to serve the table from RAM or SSD, and
  `rows_gather@1`'s dequantization is `qembedding@1`'s; both land with the tiers in step 6, where the Ref changes and
  the unit does not.
- **Fused stacks are unit parameters.** The core maps one checkpoint tensor to one parameter and refuses transforms on
  quantized tensors, so the pack's fused tensors are read whole: `std.moe@1(fused_gate_up=True)` holds one gate / up
  stack (per expert gate's rows, then up's: the merged stack's layout), served by the resident store's new `fused`;
  `std.mixers.qsa@1(fused_index_qk=True)` holds one `index_qk_proj` and splits its output. Both default off, so no
  lock's facts move. The offload store gains `fused` with the aux experts in step 6 (D12a); until then a fused stack
  with `moe.store` set fails at load with a clear message.
- **The pack is a file of its own** (`mtp.path`, as `dflash.path`); `GGUFSource` reads a file without a tokenizer
  (an empty vocabulary). The MTP rules map its HF names under the root `mtp.`, with no transforms (its norms are w).
- **The head's e** is the first H columns of the target's embed (`std.hyper.expand@1` repeats it), so the drafter
  keeps using the target's shared embed. On the GPU its pooled hnorm reads R in the model dtype; on the CPU all is
  f32.
- **QSA's planner** picks the index planner by its spec's layout (`qsa.idx`), not by name: an aux model's spec names
  are namespaced (`mtp.idx`).
- **The stream form of hnorm** and its check against LC's MTP graph are not built: FN pins pooled (D9), and the check
  needs a C++ harness over LC 27c7a76's `graph_mtp`. The `stream` parameter is added only with that check (deferred;
  plan §10).

**§6.1 on the CPU** (`plugins/tests/test_qwen4_exp_cpu.py`, the tiny GGUF and pack at their pinned sha256):

| Check | Result |
|---|---|
| prompt logprobs vs the fixture's, one-shot and chunks of 7, 10, 13 | max \|Δ\| 9.5e-7 on both prompts at every chunking; top-1 identical |
| greedy tokens (24 decode steps) | HF's |
| QSA token sets at layers 3 and 7 | identical; boundary ties occur (18 and 13 per prompt) |
| PLE row ids | bitwise |
| prefix hit (at 64) vs recompute | bitwise tokens and logprobs |
| MTP head from the pack vs the judge's pooled reference on the pack's tensors (`mtp_x`, new in the fixtures) | max \|Δ\| 2.4e-7 one-shot, 3.6e-7 in chunks of 13 (the carried R) |
| greedy speculation with MTP, the drafts made right on alternate steps | HF's tokens; verify rows accept 0 and 3 drafts (GDN, PLE, QSA rewound) |
| plants `qsa_dense_limit_off_by_one`, `ple_conv_dilation_1`, `ple_eos_cut_off` | each fails the test |
| `drafters.dflash` on this target | refused |

**On the GPU (RTX A5000):** `qwen4_exp.ple_block@1`'s C bitwise chunked vs one-shot (`test_hc_ple_kernels`); the PLE
unit (`test_qwen4_exp_units`, Qwen3.8-Flash-Next's sizes): a verify row of 1 + 3 tokens leaves the real state
unchanged, and rewound to n accepted drafts gives bitwise the state and the residual updates of 1 + n one-token steps,
n = 0 … 3; an in-step checkpoint is bitwise the state a step ending there leaves. Both feed the K / V projections one
row at a time, the contract's precondition.

**Split GGUF (D14):** a file opened by its first shard (`<name>-00001-of-0000N.gguf`) is one checkpoint: the whole
file's name, configuration and tensors, each extent in its own shard, `identity()` the sha256 of the shards' digests;
a missing shard, or a later shard named, is refused. A hub reference names the set and fetches every shard. Tested
with a tiny Qwen3 written in 5-tensor shards: it serves what the whole file serves.

**Qwen3.8's snapshot is unchanged:** Qwen3.8-27B UD-IQ3_S, bf16, the `v1-qwen3_5` trace (354 steps), `--max-seqs 8`
and a 2 GiB state budget on an RTX A5000: this step's tree and the base (0aad1ab) write byte-identical snapshot files
(sha256 `dd32244b…`), and the steps' outputs equal main's recorded snapshot of step 4's GDN check.

**The panel's review of step 3, dispositions:**

| # | Finding | Disposition |
|---|---|---|
| C (blocker for step 5) | The PLE table has no GPU kernel: `kernels.ggml_cuda`'s `qembedding` refuses rows that are not a multiple of 256 wide (PLE's are 160, IQ4_NL) | **Fixed** in its own commit (`6b05ed0`, "Holon:" comments): the check is per type's block (QK_K = 256 unless listed). The legacy types' `get_rows` reads 32-value blocks as is; IQ4_NL's reads 256-value groups, so a narrower IQ4_NL row is gathered as bytes and converted (2 launches), bitwise gguf-py's. L0 at width 160 for IQ4_NL, Q4_0, Q5_0, Q8_0, both output dtypes. **To carry onto the MoE prefill track's branch when the tracks meet** (it edits that track's plugin). §7's RTX PRO 6000 row gains the 28.8 GB table: IQ3_S ≈ 88.9 GB at 32K, fits. |
| C (major) | Launches ≈ 25–31 per layer | **Measured, and the cheap ones done.** On a 4-layer slice of the IQ2_XS file (`local` tooling: layers 0–3, every expert, the PLE hash tables cut to 1,000 rows, Q2_0 requantized to Q4_0 since the CUDA Q2_0 path is the track's), eager, sparse QSA, one decode token: **GDN 32–34, QSA 35–37, PLE's layer +11, embed / output mixer / head 7 per token**, so ≈ 34 per layer layer-weighted (1,640 per token on IQ2_XS, 1,654 on IQ3_S; the ± 2 is whether a layer's shared gate and up share a type and merge). A 4-row verify step adds the replay saves: GDN 39–41, PLE's layer 59, QSA 37. Done now: the indexer's rope rotates q only (`rope@1` with no key heads; the k clone is gone), the shared expert's output is added inside the routed combine (`std.ffn.sum@1` hands it to `std.moe@1.add_residual`: one launch less, bitwise the same sum), and `expand` casts and repeats in one copy. **Step-7 fusions instead** (each needs a kernel or contract change): the shared gate as an extra output row of the shared gate / up call (the gate is BF16 while gate / up are quantized, often of two types: ≈ 4 launches with its sigmoid and product), a cat-free SwiGLU for unmerged gate / up, the PLE key / value projections in one GEMM, the indexer q / k in one call with `q_norm` + rope fused; owner FN, with those already listed (§14). N3 now carries the band 16–37 with the measured 34 as its projection point (`launches_34`, `launches_37`; regenerated records): **at 34 launches per layer every decode cell is below its bar** (0.73–0.98 on the RTX 5080, 0.76–0.86 on the 5090), so step 7's ≤ 768 per token (16 per layer) is what the decode rows rest on. The MTP head has its own launch budget in N3: 46 per draft sub-step (the head 39, its output mixer 2, the lm_head 2, the inner embedding 2, the argmax 1), at the band's overhead per launch; it moves the RTX 5090's 32K defaults MTP row to 0.999 central. |
| C-minor 3 | `fused_index_qk`'s strided views | **Already off by default** (only the MTP pack sets it). The kernels now read the strided k in place (`qsa_index@1` takes strided rows, stated in its contract); q still becomes dense rows at `q_norm` (one copy, in the MTP head's one layer). |
| A (major) | QSA depends on gqa's private methods | **Done:** `std.mixers.gqa@1` 1.1.0 (additive) documents its composing surface — `heads`, `kv_heads`, `head_dim`, `o_proj`, `kv_layer()`, `project()`, `project_and_write()`, `write_kv()` — and QSA calls only those (the judge's plant and a KV test follow the renames). |
| A-minor | D7 says `std.moe@1` is not edited | **Updated:** it gains the additive `fused_gate_up` option. |
| A-minor | HF `arch.Qwen4ExpForConditionalGeneration@1` | **Deferred, recorded:** D1 serves the GGUF files only; the HF checkpoint is the oracle's (D2), so the rules wait for a deployment that serves it. |
| A-minor | QSA's `ratio` and `budget` defaults | **Done:** both required; the recipe passes them. |
| A-minor | The MTP head's embedding assumption | **Done:** `drafters.mtp` passes a residual-carrying head the target embed's `inner` (expand's embedding before the repeat), and the head refuses an embedding that is not hidden-size wide. |
| A-nit | A file without a tokenizer | **Done:** `GGUFSource.tokenizer()` says the file has none and is no model to serve. |
| B-minor | The speculation test | **Done:** the first j drafts made right and the next wrong, j = 0, 1, 2, 3 in turn, so rows accept 0, 1, 2 and 3 drafts; logprobs within 1e-5 of a run without a drafter (and the tokens HF's). |
| B-minor | Split shards out of order | **Done:** each shard's `split.no` and `split.count` must match its place in the set (and the first's `split.no` 0); tested by swapping two shards' files. |
| B-minor | Step 6 and the arena store | **Plan note:** step 6 gives the arena store `fused` for the MTP experts. |

**Found on the real file** (the 4-layer slice, while measuring):
- **The deployed files store the shared expert's gate as [H]** (the tiny file's converter wrote [1, H]): a `qwen4_exp.row` value transform makes it a row either way. The slice's every tensor maps and loads.
- **Blocker for step 5, the MoE prefill track's:** `kernels.ggml_cuda` serves matmuls only at `in` a multiple of 256 (`_supports_matmul`, still so on the track's branch), and FN's routed down and the shared expert's down have `in` = 640. The counts above ran with that check relaxed to 128 in a scratch copy; the track's T5 (Q2_0) needs the `in` = 640 case for both `linear_experts@2` and `qlinear@2`, with L0.
- **Propagation (D16) grows:** step 3 also changes `src` in `std.mixers.gqa`, `std.ffn`, `std.ops` (the `rope@1` text), `std.moe`, `std.spec.mtp_head` and `std.weights.gguf`, which the close's single propagation covers.

## 17. Step 5 record: the full model on the RTX PRO 6000 (2026-10-20)

**Status: stopped at a failed gate (L2, IQ3_S `v1-long` decode path, below), for the panel's review.** FN merged
the MoE prefill track (a8c8a97: T6's width-640 kernels, the
qembedding fix, IQ1_M, Q2_0, lending, `pass_rows`) at b6a79f2, then ran step 5 on one RTX PRO 6000 (96 GB, sm_120),
all-resident: experts, the 28.8 GB PLE table and the MTP pack in VRAM. The fp32 goldens streamed one layer at a time
on the CPU (32 threads; 1.5–3.6 h and 40–72 GB peak RSS per file and set). Every result below is on the final code
(30c7fd7 and the judge's 9bf717c), re-run after the two fixes. The numbers are in
`evidence/fn5_step5_rtx-pro-6000_2026-10-20.json`; the drivers are `spikes/fn5/` and the judge's commands.

**Two bugs found and fixed during the step:**
- **QSA's index kernels addressed units as if dense** (ba55212). The unit views each idx unit's layer segment as raw keys
  then pooled keys, so units are strided; `qsa_index@1` and `qsa_select@1`'s Triton kernels ignored `stride(0)`, wrote
  pooled keys over other units' bytes and selected from wrong ones. The CPU oracle (strided views) and step 4's L0
  (dense test pools) could not see it. **Found by the decode-divergence gate:** its first run failed every unplanted arm
  (sparse KL p99 9.3 nats against dense 0.19); a scoring engine then gave different logprobs for the same prompt after
  another one, from position 2,051 (the dense limit) on. The kernels now take the unit strides, the contract states
  the layout, and L0 runs on the unit's own strided views (failing before the fix).
- **Decode spent 78% of its step in one HC kernel** (30c7fd7). The tiled `_down` gave a token's 10,240-long reduction to
  six programs: 442 µs per call, 97 calls per token. Decode and verify steps (≤ 16 tokens) now take GEMV-shaped kernels
  (`_down_gemv` over (stream, 512-wide K chunk), `_down_sum` in a fixed order, `_up_mix_gemv`): the IQ2_XS decode step
  goes from 54.7 to 8.2 ms. L0 unchanged.
- Also: the judge's GGUF floor skipped PLE's table (45cf5fb), and the decode-path L2 runs in 1-token steps (at 8-token
  steps some matmuls took mmq, which the mmvq floor does not emulate).

**The decode-divergence gate (acceptance 2; IQ2_XS; `spikes/fn5/gates.py divergence`).** Each `v1-long` prompt (4K–32K)
is a session decoding 128 tokens; a reference of the same QSA mode re-scores them teacher-forced (KL at 1,024
positions, session-bootstrap SEs). Bound: max(2 × dense, dense + 2.5 SE).

| Arm | KL p95 | KL p99 | gate (p99) |
|---|---|---|---|
| dense control | 0.0163 | 0.128 | bound 0.334 |
| sparse | 0.0300 | 0.235 | pass |
| MTP (sparse, 3 drafts) | 0.0080 | 0.090 | pass |
| sparse + `qsa_decode_select_shift` | 0.0993 | 0.435 | **fails** |
| MTP + `qsa_decode_select_shift` | 0.0785 | 0.542 | **fails** |

- As pre-registered: the unplanted arms pass, both planted arms fail. **Noise:** 8 sessions make the p99 noisy. The run
  just before the HC fix had the dense p99 at 0.19 (bound 0.61), where both plants passed p99; the pre-registered
  tightening to p95 then separated them (bound 0.020: sparse 0.019 passed, the plants 0.094 / 0.078 failed). Both
  records are kept; the step-7 re-run should use more sessions.

**MTP acceptance** (3 drafts, `v1-qwen3_5`'s chat and agent prompts, 256 tokens each): **2.71 tokens per pass (IQ2_XS),
2.74 (IQ3_S)** against the ≥ 2.2 gate.

**Bounded and run-identical** (IQ2_XS; `v1-long`, a 4,096-token system prompt so decode runs sparse after a restore;
16 sessions × 4 turns × 128 tokens): prefix on / off × overlap 1 / 0 and chunked prefill at 250 tokens pass (231,744
prompt tokens restored from the prefix cache); graphs vs eager pass; MTP speculation vs plain decoding pass (prefix on,
depth 1, and depth 0). Snapshot ×3 with an 8,192-token request (a prefix hit of 8,192), MTP on: **run-identical**.

**Plants (§6.3; L2 on IQ2_XS):** `qsa_select_shift` fails on `v1-qwen3_5` too (§6.3 predicted a pass there: its three
prompts past 2,051 tokens suffice), `hc_inject_unscaled` and `ple_gate_no_sqrt` fail; the `v1-long` ones are below.

**Launches per token** (eager steps counted with the profiler, IQ2_XS): **1,678 per decode token (≈ 35 per layer)**, the
GEMV mix adding one launch per mix; a verify pass of 1 + 3 rows with its three draft sub-steps: 2,615. Step 7's gate is
768.

**Kernel-only comparison with Strata, all-resident** (`bench/offload_ctx.py --corpus fn`, 3 rounds, medians over the
three spans; Strata no-approx with `--expert-cache 24576`, its PLE in RAM as it requires, Holon's in VRAM; Holon at
8,192-token steps). Decode tok/s, Holon / Strata:

| File, arm | 1K | 4K | 32K |
|---|---|---|---|
| IQ2_XS, drafting off | 123.5 / 133.4 (0.93) | 125.5 / 134.5 (0.93) | 121.1 / 137.2 (0.88) |
| IQ2_XS, MTP | 171.2 / 221.4 (0.77) | 143.4 / 189.9 (0.76) | 186.0 / 230.8 (0.81) |
| IQ3_S, drafting off | 116.5 / 123.8 (0.94) | 120.6 / 126.6 (0.95) | 114.1 / 126.0 (0.91) |
| IQ3_S, MTP | 176.1 / 176.7 (1.00) | 138.3 / 167.8 (0.82) | 165.8 / 228.1 (0.73) |

Prefill tok/s, Holon / Strata (drafting off): IQ2_XS 3,085 / 2,719 (1.13), 3,867 / 4,392 (0.88), 3,960 / 5,379 (0.74);
IQ3_S 3,108 / 2,745 (1.13), 3,794 / 4,476 (0.85), 3,845 / 5,480 (0.70). Strata's accepted tokens per pass in its MTP
arms: 2.2–3.4. The gaps sit where §14's launch table and step 7 put them: ≈ 35 launches per decode layer, the MTP head's
own budget, and the prefill's fp32 HC projections and GDN at long contexts.

**N5:** Holon's own decode routing (IQ2_XS, all-resident, 1,530 decode tokens over D11's 1K and 4K spans) replayed in
N3's roofline in place of Strata's trace: the lfu hit rates rise 0.4–1.4 points and the decode ratios 1–3% (e.g. the
RTX 5080's no-approx decode 1.04 / 1.01 / 1.08 at 1K / 4K / 32K central); of N3's rows below their bar only the RTX
5090's 32K defaults MTP row (report-only) leaves the list.

**L2 Tier B** (bf16 engine vs fp32 golden, against the bf16 floor emulating q8_1 on the path; prefill path at 8,192-token
steps, decode path at 1-token steps):

| File, set, path | top-1 | KL p50 | KL p99 (floor; budget) | ΔPPL vs floor (budget) | Verdict |
|---|---|---|---|---|---|
| IQ2_XS, `v1-qwen3_5`, prefill | 97.26% | 4.4e-4 | 0.301 (0.316; 0.631) | −0.24% (3.13%) | pass |
| IQ2_XS, `v1-qwen3_5`, decode | 97.16% | 5.2e-4 | 0.301 (0.301; 0.601) | +1.27% (1.57%) | pass |
| IQ2_XS, `v1-long`, prefill | 94.67% | 1.8e-3 | 0.634 (0.752; 1.50) | +2.29% (3.82%) | pass |
| IQ2_XS, `v1-long`, decode | 94.62% | 1.9e-3 | 0.661 (0.729; 1.46) | −0.84% (4.92%) | pass |
| IQ3_S, `v1-qwen3_5`, prefill | 96.51% | 6.7e-4 | 0.467 (0.486; 0.97) | −1.28% (3.43%) | pass |
| IQ3_S, `v1-qwen3_5`, decode | 97.13% | 6.0e-4 | 0.355 (0.382; 0.76) | −0.12% (3.08%) | pass |
| IQ3_S, `v1-long`, prefill | 93.55% | 2.7e-3 | 1.00 (1.08; 2.15) | −2.78% (7.13%) | pass |
| **IQ3_S, `v1-long`, decode** | 93.51% | 2.6e-3 | 0.915 (1.08; 2.16) | **−6.56% (6.36%)** | **FAIL** |

**The gate that failed, and where step 5 stops.** L2 on IQ3_S's `v1-long` decode path fails on one statistic: the
engine's perplexity sits 4.53% below the golden's and the floor's 2.18% above, a difference of 6.56% ± 2.45% against a
6.36% budget. Its top-1, KL p50 and KL p99 are inside budget (the KL p99 below the floor's). Per prompt, the shift comes
from `recall-4k` (the engine's NLL 0.37 nats per token below the golden's, the floor's 0.12 above), a prompt of
high-entropy filler (the golden's NLL 4.9 nats per token), and it spans the dense positions (0–2,050) as much as the
sparse ones, so it is not QSA's selection. The same prompt's decode and prefill paths differ by 0.1–0.26 nats per token
in both files (IQ2_XS: decode −0.11 against prefill on positions 0–2,050), which the KL statistics barely see (those
are tail tokens, outside the top-20). Not yet diagnosed: which path-dependent op (the mmvq trunk, the head's output
dtype on the decode path, GDN's recurrent update, the GEMV HC mix) moves tail log-probabilities this much. Step 5
stops here, as instructed.

**`qsa_pool_rope_last` passes L2 on `v1-long`** (KL p99 0.652 against 0.634 unplanted; it fired on every step). §6.3's
pre-registered rule applies: the long set gains recall prompts before any lock.

**Deviations and notes:**
- The `v1-qwen3_5` floors ran on an RTX A5000 (one layer resident fits 24 GB); the `v1-long` floors need the RTX PRO
  6000 (an A5000 runs out of memory in HF's GDN at 32K).
- The first node's card reported uncorrectable ECC errors (an engine failed at its first allocation); a second node
  served the step.

### 17.1 The failed gate, diagnosed (2026-10-21)

**The culprit: the GEMV HC mix of 30c7fd7 (FN's own kernel), on its pending-combine path.** `_up_mix_gemv` added the
block's pending combine to R inside the kernel, stored it and used it; with more than one warp, Triton re-ran the R
loads where it converted layouts, so some columns of the mixer's output x read R with the combine applied twice. R
itself stayed right. Every decode and verify step (≤ 16 tokens) took this path at every FFN half, so the decode path's
FFN inputs were corrupted; prefill (8,192-token steps) never took it. The L0 caught it on an RTX A5000 after the merge
with main (`mix x, 1 tokens, pending`: usage 4.7); it had passed on the RTX PRO 6000 at L0's shapes.
- **The fix** (1f88194): the GEMV mix applies a pending combine with `_combine` first (the same fp32 expression), then
  mixes R as left; one launch more per pending mix. L0 passes on sm_86 at 1, 7 and 77 tokens.
- **The evidence** (IQ3_S, the full model on the RTX PRO 6000, `recall-4k`'s first 2,048 tokens, the decode path in
  1-token steps against one prefill step; target log-probability decode − prefill, nats per token, by position band
  0–511 / 512–1,023 / 1,024–2,047):

| HC mix | mean | mean \|·\| | top-1 log-prob, decode (prefill −0.258) |
|---|---|---|---|
| 30c7fd7 (the step-5 runs) | +0.193 / +0.512 / +0.754 | 0.26 / 0.57 / 0.85 | −0.312 (flatter) |
| fixed (1f88194) | −0.035 / −0.160 / +0.084 | 0.21 / 0.29 / 0.33 | −0.255 |

  The old kernel raised the targets' probability systematically and flattened the predictions, as panel B's sign
  argument said. The fixed paths still differ per token (mean \|·\| 0.2–0.3 on this high-entropy filler) with mixed signs.
- **Correction (after the re-runs, 17.2): the full model does not separate the race from path noise; L0 found it.**
  The racy − fixed difference on `recall-4k` is consistent across setups (+0.23 / +0.79 / +0.79 between the L2
  recordings, +0.23 / +0.67 / +0.67 in the probe above), but the fixed decode path's own offset from prefill moves by
  up to 0.5 nats per token with benign setup changes (the L2 recording, 4,006 tokens and 8 sequences: −0.21 / −0.49 /
  −0.43; the probe, 2,048 tokens and one sequence: −0.04 / −0.16 / +0.08). Over all of `v1-long`, decode vs prefill
  gives the same statistics with either kernel (IQ2_XS: KL mean 0.043 fixed, 0.045 racy; KL p50 0.0023 both; the worst
  1K band's \|ΔNLL\| 0.50 and 0.63; IQ3_S: KL mean 0.063 and 0.057), the per-layer residual taps are equal (layer 0's
  rel diff 6.95e-3 in both), and the top-1 sharpness shift has no consistent sign. So step 5's L2 miss (−6.56% against
  6.36%) is not shown to be the race's doing: it sat at the margin on a prompt whose decode-path NLL swings by that
  much with any path change. The race is a real bug (L0 fails on sm_86) and is fixed; full-model self-consistency
  catches errors larger than this noise, not errors of the race's size (§18).
- **Ruled out first** (B's probes): causality is exact (changing the tokens from position 1,000 on leaves every row
  before it bitwise, row 999's distribution included, on both paths); the output scale is the same (post-mixer ‖x‖
  147.1 against 147.3); on a 4-layer slice no single swap of a decode-path op for its prefill form (the tiled HC at
  T = 1, GDN's chunked kernel at T = 1, mmq at T = 1) removes the layer-0 difference, which is bf16-rounding sized
  (0.5–0.8% relative at the GDN mixer's output) and grows to ≈ 8% relative R by layer 47 on the full model, the same
  with either HC kernel.
- **Every decode-path result of §17 ran with the racy kernel**: the decode-path L2 candidates, the decode-divergence
  gate, MTP acceptance, the bounded checks, the snapshot and the kernel-only comparison. They are re-run with the fix
  (17.2); the goldens, floors, prefill-path candidates and plants are unaffected.
- **The failed gate passes with the fix:** L2 on IQ3_S's `v1-long` decode path, ΔPPL vs floor **+1.01% (budget
  5.99%)** where it was −6.56% (6.36%); top-1 93.68% (floor 93.42%), KL p50 2.4e-3 (2.5e-3), KL p99 0.949 (1.08).
  `recall-4k` now sits 0.31 nats per token above the golden, with sharper top-1s — the side of the mmvq floor (+0.12)
  and of the decode run before 30c7fd7 (the tiled HC mix, 8-token steps: +0.30). This high-entropy filler prompt moves
  as a whole with the logits' scale, so its NLL swings with any path's numerics; the racy kernel sat on the other side
  (−0.37, flatter).

**Panel B's minor findings:** the decode-divergence statistic is p95 (or the paired per-position KL) from now on, and
step 7 re-runs it with more sessions; `qsa_index` / `qsa_select` refuse unit views whose rows are not dense (8a6071c);
`qsa_attend`'s L0 already reads the gqa unit-segment KV views (`empty_cache`: q8_0's values and scales in one segment
per unit, strided); `qsa_pool_rope_last` is recorded as caught by the tiny-model fixtures (`test_plants_fail`), like
`qsa_dense_limit_off_by_one`, and `v1-long` does not grow for it.

### 17.2 Re-runs with the fixed mix (2026-10-21)

Every decode-path result of §17 re-ran on the RTX PRO 6000 with 1f88194. The records are in
`evidence/fn5_step5_rerun_rtx-pro-6000_2026-10-21.json`, the racy runs beside them as history.

| Check | With the fixed mix | Step 5 (racy mix) |
|---|---|---|
| L2 decode path: IQ2_XS `v1-qwen3_5` / `v1-long` | 1.95% vs 1.57%: fails (below) / −0.75%: passes | 1.27% / −0.84%: pass |
| L2 decode path: IQ3_S `v1-qwen3_5` / `v1-long` | −2.40%: passes / +1.01% vs 5.99%: passes | −0.12% / −6.56% vs 6.36%: fails |
| Decode divergence (8 sessions), p95: sparse / MTP / dense; plants | 0.013 / 0.008 / 0.016; 0.077 / 0.079 fail (bound 0.041) | 0.019 / – / 0.016; 0.094 / 0.078 fail |
| MTP accepted tokens per pass (≥ 2.2) | 2.73 (IQ2_XS), 2.76 (IQ3_S) | 2.71, 2.74 |
| Bounded: prefix, overlap, chunk 250, graphs, MTP vs plain | all pass | all pass |
| Snapshot ×3, 8K request, MTP on | run-identical | run-identical |
| Launches per decode token / verify pass | 1,726 / 2,667 (one `_combine` more per pending mix) | 1,678 / 2,615 |

At p99 the 8-session divergence gate is too noisy to separate the plants (dense p99 0.20 ± 0.16, bound 0.60, both
plants under it); the pre-registered tightening to p95 separates them, as in step 5. Step 7 re-runs it with 29
sessions (each prompt cut at four lengths; `gates.py --cuts 4`).

**Kernel-only comparison with Strata, all-resident** (as §17; decode tok/s, Holon / Strata):

| File, arm | 1K | 4K | 32K |
|---|---|---|---|
| IQ2_XS, drafting off | 126.7 / 135.8 (0.93) | 124.1 / 131.1 (0.95) | 122.2 / 137.5 (0.89) |
| IQ2_XS, MTP | 166.2 / 224.5 (0.74) | 145.1 / 190.1 (0.76) | 203.1 / 231.2 (0.88) |
| IQ3_S, drafting off | 119.3 / 124.9 (0.96) | 121.8 / 124.5 (0.98) | 115.1 / 126.1 (0.91) |
| IQ3_S, MTP | 161.9 / 174.4 (0.93) | 148.6 / 168.2 (0.88) | 156.5 / 228.7 (0.68) |

Prefill tok/s, drafting off: IQ2_XS 3,082 / 2,644 (1.17), 3,881 / 4,407 (0.88), 3,968 / 5,401 (0.73); IQ3_S 3,111 /
2,735 (1.14), 3,801 / 4,482 (0.85), 3,851 / 5,497 (0.70). The decode ratios are within a few points of step 5's.

**IQ2_XS `v1-qwen3_5`'s decode-path ΔPPL miss** (1.95% > 1.57%; its top-1 and KL inside budget and the best of any
path). A bisect on the full model (the same set in 1-token steps, one decode-path op at a time forced to its prefill
form) finds no decode-path error:

| IQ2_XS `v1-qwen3_5`, 1-token steps | ΔPPL vs golden | top-1 | KL mean | KL p99 | ΔPPL vs mmvq floor (budget 1.57%) |
|---|---|---|---|---|---|
| decode path (1f88194) | +1.99% | 97.41% | 1.54e-2 | 0.296 | 1.95%: fails |
| the same, CUDA graphs off | bitwise equal | | | | |
| tiled HC mix at T = 1 | +1.45% | 97.38% | 1.48e-2 | 0.274 | 1.41%: passes |
| every op in its prefill form (tiled HC, GDN chunked, mmq; graphs off) | −0.49% | 97.28% | 1.63e-2 | 0.345 | (mmq floor: −1.52%, budget 3.23%: passes) |
| prefill path (one step per prompt) | +0.80% | 97.26% | 1.60e-2 | 0.301 | (mmq floor: −0.24%, 3.13%: passes) |
| floors: mmvq / mmq | +0.03% / +1.05% | 97.27% / 97.18% | 1.69e-2 / 1.76e-2 | 0.301 / 0.316 | |

- Paired per position, KL(golden ‖ ·) of the decode path differs from every other engine path by less than 0.5
  standard errors (prompts as clusters), and sits 1.6 SE below its floor's. The NLL differences between these
  equally-correct paths span 2.5 points of ΔPPL (clustered SEs 0.5–1.2%): two runs with the same op forms (the
  all-swapped decode and the prefill) differ by 1.3%, 2.5 SE. ΔPPL on 24 sequences is the noise the judge's docstring
  describes ("two correct bf16 paths differ by a paired SE of ≈ 0.9%").
- The budget is narrow here because it caps the paired SE at √2 × the floor's own, and the mmvq floor's is the
  smallest of the eight (0.37%; the mmq floor's 0.84% gives 3.13%). Uncapped, the budget would be 1.99%.
- On the combined set (`v1-qwen3_5` + `v1-long`, the existing recordings concatenated) it passes: ΔPPL vs floor
  +0.93% (budget 2.22%), top-1 95.24% (floor 94.83%), KL p99 0.574 (0.632). The judge's rule was not changed.
- **Closed as moot** by the owner's change of FN's accuracy method (§18).

## 18. Owner decision: FN's accuracy method (2026-10-21)

Full-model L2 Tier B against layer-streamed fp32 goldens and bf16 floors is **dropped for FN**: too slow (1.5–3.6 h
per golden on the CPU), and not what FN needs. No more goldens or floors are recorded; §17's L2 results stay as
history. Acceptance 1(b) is restated (§1). FN's accuracy gates are now all fast:

- **(a) L0 kernels against their oracles** (in place).
- **(b) The tiny model on the CPU, exact against HF** (in place; acceptance 1(a)).
- **(c) Self-consistency on the real model, no golden.** Holon's paths against each other on a few long prompts:
  the decode path (1-token steps) vs prefill, chunked vs one-shot, MTP vs plain; per-token KL and NLL differences,
  with the bar near the bf16 noise between correct paths (§17.1–17.2: paths with the same ops differ by mean |Δ
  target log-prob| 0.2–0.3 nats on high-entropy filler, and by up to 1.3% in perplexity on `v1-qwen3_5`). Kept as
  the check that showed the HC race's shift on `recall-4k`; the re-runs show that shift sits inside the paths' noise
  (§17.1's correction, below).
  **Judged by KL (owner, 2026-10-21):** the KL statistics against the bars as set decide (c) — the tool's set bars
  are the KL mean (≤ 0.13) and p50 (≤ 0.007); the p99 is reported beside them, as no bar was set for it —; the
  per-band NLL is reported, not gated (the float64 evidence of §19.5: an SGEMM path as accurate as the tiles it replaced moved
  `summary-6k`'s bands past 0.75). Under this rule C's and B's trees pass (19.5).
- **(d) Task accuracy on hard questions.** A fixed set of about 60–100 public or synthetic items with checkable
  answers: multi-step math (exact numbers), code run against tests, reasoning and multiple choice, needle / recall at
  8K and 32K, a couple of multi-turn. Holon and Strata greedy on the same file, both scored. Bar: Holon within a
  tolerance stated before the run, and no gibberish answer; the answers saved for the owner to read. Licensing
  checked and provenance recorded in the repository.
  - **The set** (`spikes/fn7/tasks/fn_tasks.jsonl`, 72 items, written by `spikes/fn7/tasks/build.py` with seed 7):
    all synthetic, so it carries this repository's license and no third-party text — math 24 (eight templates with
    exact answers computed by the generator), code 14 (specifications and asserts written here), reasoning 16
    (seating orders and knights-and-knaves generated with one solution, weekday arithmetic; A–E), recall 12 (needle,
    two-hop and sum needles in generated ledger filler of ≈ 8K and ≈ 31K tokens), multi-turn 6. Reference answers
    (local) score 72 of 72 through the scorer, so the tests and answers agree.
  - **The run** (`spikes/fn7/tasks/run.py`): IQ2_XS on the RTX PRO 6000; the chat template with thinking on; greedy,
    8,192 new tokens at most; Holon plain decoding, Strata's no-approximation arm of §17 (drafting off).
  - **The bar, stated before the first run:** Holon within 5 items (6.9 points) of Strata's score on the 72, and no
    Holon answer degenerate (the token limit hit inside the reasoning, nothing after it, or a 40-character span
    repeated 6 times) where Strata's is not.
  - **The sample dump** (e): eight items' answers, written by the scorer (`--samples`).
- **(e) A sample dump:** a short readable file of Holon's answers to a handful of hard questions.

**Precision rule (owner, 2026-10-21):** no fp32 runs, and no CPU runs on big models — the CPU only for a super-tiny
model, and only when really needed; bf16 on the GPU is the highest-precision reference worth running, since serving
is always at lower bits. For FN: no new goldens or floors, nothing full-model on the CPU, the tiny-model HF match
stays (seconds), and every self-consistency comparison runs in bf16 on the GPU. The step-5 re-runs that needed a
golden or floor are dropped; MTP, bounded, snapshot, the Strata comparison, launches and step 7 stay.

**What (c) can see** (measured on the step-5 recordings, §17.1's correction): decode vs prefill on `v1-long` gives KL
mean 0.04–0.06, KL p50 0.002–0.003 and band \|ΔNLL\| up to 0.5 nats between correct paths, and the HC race did not
move those statistics. The bars (`spikes/fn7/consistency.py`: KL mean ≤ 0.13, KL p50 ≤ 0.007, every 1K band's
\|ΔNLL\| ≤ 0.75, MTP vs plain KL mean ≤ 0.13 up to the first differing token) are set at about twice that noise, and
the decode-only plant `qsa_decode_select_shift` must fail them. Kernel races and other errors below the noise are
L0's to find, on all three SMs.

## 19. Step 7 record: perf closure (started 2026-10-21, on recommended defaults; the owner's answers pending)

Scope until the owner answers: item 0 (where the time goes) and item 1 (launch fusions inside FN's own units and
kernels). Out of scope: the `kernels.ggml_cuda` fusions (the shared expert, router + top-k) and the tensor-core HC
prefill (a change to D10). Development rules (owner, 2026-10-21): affected tests only while developing, the full
suite sharded at hand-off, no lock propagation before FN's close, plants and bounded checks once at the step's close,
3-round A/Bs.

### 19.1 Item 0: the breakdown

`spikes/fn7/breakdown.py` profiles eager steps under nsys (graphs off, so every kernel sits under the host's NVTX
ranges) and attributes each kernel's GPU time to its component. Records: `evidence/fn7_breakdown_2026-10-21.json`.
The full model (IQ2_XS, all-resident) on the RTX PRO 6000, before step 7's changes (7d94ed5); 4K / 32K context
(31,744 tokens, the longest prompt's chat part); ms of GPU time per step, kernels in brackets:

| Component | Decode 4K | Decode 32K | Verify 4K (1 + 3 rows, 3 draft sub-steps) | Prefill 4K (one step) | Prefill 32K (per 8,192-token step) |
|---|---|---|---|---|---|
| HC mix + combine | 2.24 (387) | 2.24 (387) | 4.46 (387) | 666.5 (243) | 1,147.3 (241) |
| GDN (36 layers) | 1.92 (396) | 1.92 (396) | 3.76 (576) | 101.3 (1,404) | 194.6 (1,593) |
| MoE routed | 1.99 (240) | 1.99 (240) | 3.27 (240) | 286.0 (1,824) | 471.1 (3,816) |
| QSA (12 layers) | 1.06 (181) | 1.10 (181) | 1.35 (193) | 86.5 (387) | 215.6 (654) |
| Shared expert | 1.01 (482) | 0.92 (482) | 1.02 (434) | 13.4 (580) | 24.4 (943) |
| PLE (layer 1) | 0.11 (9) | 0.11 (9) | 0.13 (13) | 1.8 (9) | 3.3 (9) |
| lm_head | 0.23 (2) | 0.23 (2) | 0.28 (2) | 0.3 (2) | 0.1 (0) |
| MTP head (all of its sub-steps) | – | – | 2.22 (311) | – | – |
| Other (sampling, planning copies, embedding) | 0.06 (25) | 0.06 (25) | 1.07 (333) | 0.5 (29) | 0.4 (24) |
| **Step** | **8.62 (1,722)** | **8.57 (1,722)** | **17.56 (2,489)** | **1,156 (4,478)** | **2,057 (7,280)** |

- **Decode is GPU-bound at ≈ 8.6 ms** (graph replay: 126.7 tok/s at 1K in §17.2, 7.9 ms), not launch-bound once
  graphed; eager it takes 45 ms. Its four big parts: the HC mixes 26% (97 mixes, 22 µs each, against ≈ 7 µs to read
  each mix's 13 MB of bf16 weights at the card's bandwidth: the largest FN-owned gap), the routed MoE 23%, GDN 22%,
  QSA 12%; the shared expert 11% in 10 launches per layer.
- **Prefill is the HC mixes' fp32 projections: 55–57%** of a step (D10's IEEE dot products; the tensor-core change is
  out of scope), then the routed experts 23–25%, QSA 8–11%, GDN 9–10%.
- **A verify pass costs 2.0× a decode step**; the MTP head is 2.2 ms of its 17.6 (its lm_head 0.68, its QSA 0.57).
- **Launches per decode token: 1,722** — per GDN layer ≈ 34 (HC 8, GDN 11, MoE 5, shared expert 10), per QSA layer
  ≈ 38 (QSA 15), PLE 9, embedding, output mixer and head ≈ 30.
- **On the RTX 5080** the full model does not fit before step 6's PLE host table (28.8 GB in VRAM today). A 4-layer
  IQ2_XS slice (layers 0–3, every expert, the PLE table cut to 1,000 rows per head) gives the same per-layer launch
  counts and, per layer, decode 1.37 ms for 4 layers (lm_head 27%, HC 16%, MoE 15%, GDN 14%), prefill 4K 170 ms (HC
  mixes 55%), the same shares as the full model's.

### 19.2 Item 1 so far: launch fusions in FN's units and kernels

- **HC, one-token steps in two launches** (ccb776d, ae70929): `_down_gemv` writes a pending combine's R + w · y to a
  scratch buffer and every `_mix_gemv` program sums the partials itself, so no `_combine` and no `_down_sum` launch
  (no kernel stores into a buffer it reads). Bitwise the same R, x and combine weights as before (64 cases on sm_86
  and sm_120). At 2–16 tokens the repeated sums cost more than the launches saved, so verify keeps `_combine` and
  `_down_sum`. Per mix on an RTX A5000: 26.3 against 29.7 µs (1 token), unchanged at 2–8.
- **QSA's indexer q and k in one matmul** (a952ba9): split linears that can share one are merged once loaded into
  `index_qk_proj` (the layout transformers stores); the tiny model still matches HF.
- **Measured on the RTX PRO 6000** (4K, eager under nsys, 37d8797): decode 1,722 → 1,565 kernels per token, GPU
  8.62 → 7.82 ms per step (−9%; HC mixes 2.18 → 1.67 ms); a verify pass 2,489 → 2,477 kernels, 17.56 → 17.17 ms.
- **Against the ≤ 768 gate:** the in-scope candidates left are small — the indexer's q norm and rope in one kernel
  (12), GDN's a / b projections in one GEMM (36; their recurrent kernel reads them as contiguous [T, HV], so a vendored
  `fla` patch or a copy), the PLE glue (≈ 4). That leaves ≈ 1,500. The remainder sits in the out-of-scope items: the
  shared expert (10 launches per layer: two activation quantizations, gate / up / down, the BF16 shared gate's dot
  product, sigmoid and product, SwiGLU, a cat) and router + top-k + reduction (5 per layer), and in GDN's 11 per layer
  (three quantized matmuls with their two activation quantizations, conv, its state, the recurrent kernel, the norm).
  ≤ 768 needs those; the owner's answer decides it.

### 19.3 Accuracy gate (c), first run

`spikes/fn7/consistency.py` on IQ2_XS, RTX PRO 6000, bf16 (`evidence/fn7_consistency_IQ2_XS_2026-10-21.json`;
a952ba9, the HC and indexer changes in): **passes**.

| Pair (4 prompts, 49,596 positions) | KL mean (≤ 0.13) | KL p50 (≤ 0.007) | KL p99 | mean ΔNLL | worst 1K band (≤ 0.75) |
|---|---|---|---|---|---|
| decode (1-token steps) vs prefill | 0.024 | 0.0018 | 0.33 | −0.041 | −0.27 (`summary-6k`) |
| chunked (250) vs one-shot | 0.023 | 0.0016 | 0.35 | +0.019 | +0.15 (`recall-8k`) |
| MTP vs plain, 256 tokens (231 positions up to each first difference) | 0.0018 | – | 0.015 | – | first differences at 9, 35, 39, 144 |

The pre-registered plant (`qsa_decode_select_shift`, which must fail) runs once at the step's close.

### 19.4 Accuracy gate (d) and the sample dump (e)

The 72-item set on IQ2_XS, RTX PRO 6000, greedy with thinking on, 8,192 new tokens at most
(`evidence/fn7_tasks_IQ2_XS_2026-10-21/`: both engines' answers, the scores, `samples.md`). **Passes** the bar
stated before the run (within 5 items, no Holon-only degenerate answer):

| | Correct | Math 24 | Code 14 | Reasoning 16 | Recall 12 | Multi-turn 6 | Degenerate |
|---|---|---|---|---|---|---|---|
| Holon | **70** | 23 | 13 | 16 | 12 | 6 | 2 (`code-08`, `code-12`) |
| Strata (no-approx) | **69** | 24 | 12 | 15 | 12 | 6 | 3 (`code-08`, `code-12`, `reasoning-04`) |

- The engines agree on 69 items' correctness and give the same final value on 70. Holon misses `math-06` (1,532
  against 1,531); Strata misses `reasoning-04` (its reasoning loops to the limit) and the two code items both miss.
- **The degenerate answers are the model's, in both engines:** on `code-08` and `code-12` each engine's greedy
  reasoning falls into a repeated line until the token limit (Holon: "Potential hidden issue: …", Strata: "Need maybe
  support …"). No Holon-only degenerate answer, and no gibberish: the answers read as the model's (medians 443 and
  453 tokens).
- **The sample dump** (`samples.md`): eight items — two math, two code (one of them `code-08`'s loop, shown as it
  is), two reasoning, the 32K recall and one multi-turn — with the question, the end of the reasoning and the
  answer.

### 19.5 Levers C and B (the perf reviewer's re-plan: GPU time against each component's floor)

- **C — the prefill HC projections as cuBLAS SGEMMs** (bbdfc01; TF32 refused, so float32 products with IEEE sums as
  D10 requires). Triton's IEEE float32 tiles reached 9% of float32 peak (bigger tiles: 13%); cuBLAS's SGEMM 50% at
  these shapes. The mix is now `_prep` → batched SGEMM → `_act` → SGEMM → `_mix_out`, in passes of 4,096 rows.
  - RTX PRO 6000, full model (nsys): the HC mixes per 8,192-token step **1,127 → 332 ms** at 32K, 657 → 173 ms in a
    4K step; a 32K prefill step 2,057 → 1,244 ms. The target was ≤ 270 ms: missed by 23% (the float32 scratch's
    reads and writes, ≈ 2.3 GB per mix, are about a third of it).
  - A/B, 3 rounds against 3f5e7d6: **prefill 3,902 → 6,420 tok/s at 4K, 3,958 → 6,388 at 32K** (Strata 4,407 and
    5,401 in §17.2); decode unchanged (137.9 / 136.6, 132.1 / 132.1).
  - L0 (with the stress case) passes on sm_86; decode and verify are bitwise unchanged.
- **B — the GEMV mix at 2-16 rows** (e6dcdcb): each up tile held while the rows loop, a pending combine through the
  `rn` scratch (no `_combine`), `_down_gemv` retuned for rows. Per mix on an RTX A5000: 69 → 49 µs at 4 rows, 117 →
  79 at 8 — 1.7× the one-row time, not the 1.5× target: the per-row reductions stay serial (holding the rows in one
  tile through Triton's IEEE dot or a deferred 3-D sum was slower). One row bitwise unchanged; L0 passes.
- **B's A/B** (3 rounds against bbdfc01, RTX PRO 6000): **MTP decode 214.1 → 239.4 tok/s at 4K (+12%), 159.2 →
  168.3 at 32K (+6%)**; plain decode and prefill unchanged.
- **The HC L0 on sm_120** (with the stress case) passes on both trees.
- **Self-consistency (c) on C's tree fails one bar on one prompt:** chunked vs one-shot on `summary-6k` has a worst
  1K band of +0.89 nats (bar 0.75; its mean +0.47, KL mean 0.10 on that prompt), decode vs prefill −0.52; the other
  prompts are as before (bands ≤ 0.15). It is not an accuracy loss: against a float64 oracle the SGEMM path and the
  Triton tiles it replaced have the same error at 250, 1,412, 4,096 and 5,508 rows (x relative 1.13e-7, R 7.6e-6
  max, both). The SGEMMs only round differently, by row count (cuBLAS picks its kernel per shape), and `summary-6k`
  moves as a whole with any such change (as `recall-4k` did in §17.1). The bar, set at 1.5× the worst band seen
  before, does not hold this prompt's spread. **Owner, 2026-10-21: (c) is judged by KL** (the bars as set: mean
  and p50); the bands are reported. So C's tree passes (KL mean 0.025 / 0.033, p50 0.0018 / 0.0020; p99 0.36 /
  0.52).
- **On B's tree** decode vs prefill and chunked vs one-shot give C's numbers exactly (B changes neither one-row decode
  nor prefill), so the same one-prompt miss; MTP vs plain passes (KL mean 0.0017; first differences at 9, 35, 139
  and 132 tokens).

**Owner decision (2026-10-21) on the RTX 5080's MTP rows:** decided once measured on the card in step 6 — the F5b
CPU split in the lock, or accepting about 0.95–1.0. Both arms are measured on the 5080: F5b off, and F5b on at tune's
thread count; the F5b-on arm runs under the path-aware floor check, now GPU self-consistency plus the task gate, no
fp32 (§20).

### 19.6 The owner's new opening: llama.cpp's FN kernels and Strata, per component (2026-10-21)

**Item 1 — the vendored ggml already carries Flash-Next's kernels.** `kernels.ggml_cuda` is pinned at llama.cpp
cbe28bb3 (2026-10-13), and 3cf03257 (2026-09-20, "CUDA: enable sparse fa for qwen4", the commit Strata's `native_*`
kernels cite) is its ancestor: the pin's tree has `gated_delta_net.cu`, `ssm-conv.cu`, `dsv4-hc.cu` (HC pre / post),
`lightning-indexer.cu` and the sparse flash attention. No pin move and no `vendor.patch` rebase is needed; only the
matmul and MoE files are vendored, and item 2 decides whether any of these joins them. **No lock class moves** from
item 1. From step 7 so far only FN-only units changed (`kernels.hc_triton`, the indexer merge in `std.mixers.qsa`), so
at FN's close only FN's own locks are new; no other model's lock moves.

**Items 2 and 3 — where the GPU time goes, per component, in all three engines** (RTX PRO 6000, IQ2_XS; ms of GPU
time; `evidence/fn7_component_comparison_2026-10-21.json`). Holon: nsys breakdowns (§19.1, with levers C and B).
llama.cpp: its own `qwen4exp` graph and CUDA kernels at 27c7a76 (newer than the pin), `llama-bench` under nsys, each
graphed decode kernel attributed by its place in the half-layer (`dsv4_hc_post` delimits). Strata: its own stage
timers (`STRATA_VERIFY_PROFILE`, `STRATA_PREFILL_TIMING`), whose GPU stamps add about a fifth to its decode (9.0 ms
per token stamped against ≈ 7.4 ms in its decode tok/s).

| Decode, 4K context, per token | Holon | llama.cpp | Strata (stamped) |
|---|---|---|---|
| HC mixes and combines | **1.73** | 4.04 | 2.64 |
| GDN mixer (36 layers) | 1.87 | 2.65 | **1.70** |
| QSA mixer (12 layers) | **1.02** | 2.01 | 1.70 |
| Router (+ top-k) | **0.47** | 0.52 | 0.53 |
| Routed experts | 1.43 | 1.61 | **1.30** |
| Shared expert | 0.92 | 1.27 | **0.87** |
| PLE, lm_head, other | 0.39 | – | 0.29 |
| **Step** | **7.82** | 12.12 | 9.01 (≈ 7.4 unstamped) |

- **Item 2's verdict: no llama.cpp component beats ours**, so none is wired in. Their HC is generic ops around
  `dsv4_hc_pre` / `dsv4_hc_post` (≈ 11 launches a mix, bf16 `mul_mat_vec_f` GEMVs: 4.0 ms against 1.7); their GDN runs
  a fast recurrence (`gated_delta_net_cuda`, 3.0 µs a layer against our fused 6.5 µs) but pays 8 µs of separate
  norms and gates around it (2.65 ms against 1.87); their QSA (lightning indexer, sparse flash attention) is 2.0 ms
  against 1.0. At 32K: HC 3.98, GDN 2.65, QSA 2.35; Holon's QSA 1.10. Throughput, `llama-bench` under nsys: decode 54
  (4K) and 72 (32K) tok/s, prefill 3,766 (4K) and 3,455 (an 8K chunk at 24K) tok/s.
- **Where Holon still trails Strata at decode:** the routed experts (1.43 against 1.30; `kernels.ggml_cuda`'s mmvq,
  held as D), the shared expert (0.92 against 0.87; held as D) and GDN (1.87 against 1.70: our conv and its state
  write take 0.20 ms against Strata's 0.08, our a / b projections 0.25 against 0.18; the recurrence and the
  quantized projections are level). Strata's stage times include its stamps, so these gaps are upper bounds.
- **Verify (1 + 3 rows; Strata's window of ≈ 4 rows):** Strata 12.85 ms GPU a window plus 1.8–2.3 ms of drafting;
  Holon 17.6 ms before B (HC 4.46, GDN 3.76, routed experts 2.79, the MTP head 2.22, other 1.07), B taking ≈ 1.4 ms
  off the HC. Strata's HC at 4 rows is 4.09 and its GDN 2.30: our GDN at 4 rows (3.76) and the verify "other"
  (sampling, rewind; 1.07) are the gaps.
- **Prefill, a 32K prompt:** Strata 6.18 s of GPU (HC read 0.46, GDN 1.35, QSA 1.72, routed MoE ≈ 2.49, router and
  shared 0.10); Holon ≈ 4.98 s (4 steps: HC 1.41, GDN 0.77, QSA 0.85, routed MoE 1.73, router 0.12, shared 0.10).
  **Holon trails only in the HC** (1.41 s against 0.46): float32 SGEMMs (D10) against Strata's projections.
- **Item 4's candidates** (own kernels where both trail Strata), for the perf plan's budgets: the HC prefill (D10
  decides how far float32 can go), GDN's conv and gates at decode and GDN at 4 rows, the verify "other"; the routed
  and shared experts wait for D.


## 20. Step 6 record: Flash-Next end to end on limited VRAM (2026-10-21/22; the owner's priority)

Scope (owner, 2026-10-21): the PLE table in pinned host RAM; expert offload for FN (the MTP experts in the store,
lfu with F5a, lent prefill scratch with tail lending, the 5080's memory levers); first end-to-end runs on an RTX 5080
(IQ2_XS) and an RTX 5090 (IQ3_S) — a chat, gates (c) and (d), decode / MTP decode / prefill against Strata on the
same card; the 5080 decode step per component. Not yet: the PLE SSD tier (later, per the owner), tune's interleaved
F5b benchmark (step 6 (b)), the plants and exactness checks of step 6's close. Records:
`evidence/fn6_e2e_2026-10-22.json`, the task answers in `evidence/fn6_tasks_2026-10-22/`; the measuring scripts in
`spikes/fn6/` (`ring_diag.py`, `ring_exact.py`, `cached.py`, `hitrate.py`, `f5b_split.py`).

### 20.1 What was built

- **`std.tiered.rows@1`** (eb917f0, D5): a row table the deployment places on the device or in pinned host RAM
  (`rows.place`); a quantized table is read from its file extent into a mapped host pool and its `qembedding@1` kernel
  gathers the step's rows across the bus. FN's PLE table (IQ4_NL, 26.8 GiB) and `token_embd` (IQ4_XS, 322 MiB) use
  it; both on the host in every step-6 deployment. Bitwise the same logits host or device (an IQ2_XS slice). **Gather
  cost** on the RTX 5080, full model (nsys): the PLE rows (an `index_select` of mapped host memory, then IQ4_NL
  dequantization) 17 µs and `token_embd`'s `get_rows` 9 µs per decode token — 0.2% of the step; 2.0 and 1.0 ms per
  4K prefill step (0.1%). On an RTX A5000 slice against the device table: +8 µs per decode token, +1.4 ms per 4K step.
- **The MTP head's experts in the offload store** (6f90a1e, D12a): the auxiliary model's layer is the store's next
  group, its fused gate/up stack an arena piece; its passes read the step's row kinds, then decode kinds in the
  drafter's sub-steps; the lent slots' return leaves a marker record (`RETURN`) so the host mirror follows sub-step
  admissions after it. A placement profile counted on the target alone serves the deployment with the MTP file
  (21bb40f).
- **F5b for FN** (31e5e0b): `cpu_experts@1` takes reductions of whole blocks of each matrix's type (32 for Q2_0 …),
  so FN's inter 640 with Q2_0 down is served; L0 gains FN IQ2_XS's three expert classes.
- **Lending, made to happen on the bar cards** (a319e0f, 57e7bb4). The first runs on both cards lent nothing: a layout
  lends only when every class's cache can lend a prefill pass's staging (512 slots), and FN's small classes — the
  IQ2_XS file's IQ1_M gate/up of 3 layers, the IQ3_S file's IQ4_XS gate/up of 1 layer — were seeded fewer; every
  class then kept 512 staging slots, and every prefill pass staged its misses through zero-copy fetches (on the 5080
  a 4K step was 1.06 s of `k_fetch` in 2.42 s). Two fixes in `std.tiered`: a ring step streams per class (a319e0f; a
  class whose lent slots cannot hold both halves stages into them while the others stream, `ring_ok` 2), and a class
  the seed leaves short is floored — its hottest items seeded first, up to the pass's staging — so the layout lends
  (57e7bb4; a layout that lent unfloored is unchanged). On the 5080 an 8K prompt then takes 3.1–3.2 s against 5.3–5.5
  s, a 4K step 1.49 s of wall against 2.43, and the staging slots (≈ 1.5 GB) became cache. A ring step scores bitwise
  as a staged one on the card (`ring_exact.py`: target and top-k logprobs equal).
- **The deployment** (both cards; one `holon serve`-style flag set, `spikes/fn7` takes it): `std.moe.offload@1` with
  lfu (device-owned) and F5a, `moe.ring_lend=tail`, `ggml_cuda.pass_rows=4096`, `rows.place=ple:host,token_embd:host`,
  a placement profile from 60 task prompts (`holon_tools profile`, 128 tokens each), `--host-memory` 70 GB (5080) /
  86 GB (5090), `--max-step-tokens` 4,096 (5080) / 8,192 (5090), `--ctx 33792`. MTP: `--max-draft-tokens 3` with
  the Strata-format pack (`mtp-q2_0.gguf`), `std.spec.fixed_k`, greedy verifier.

### 20.2 The bar tables

`bench/offload_ctx.py`, batch 1, `--corpus fn`, contexts 1K / 4K / 32K, 3 rounds, medians; Holon / Strata tok/s
(ratio). Strata runs its step-0 calibration (`spikes/fn0/arms_strata_<card>.json`: the no-approximation arms,
`--pcie-frac 0.55`, its adaptive cache; drafting off, or its MTP head at `--spec 4`). Bars: decode ≥ 1.0, prefill ≥
0.9. Strata's rows are from the same allocation as Holon's (the 5080's from the first bench, the 5090's re-run with
Holon's after 57e7bb4).

**RTX 5080, IQ2_XS** (57e7bb4; the memory levers: 4K steps, `token_embd` on the host, a reserve — `memory_fraction`
0.97 in place of 0.9):

| Row | 1K | 4K | 32K |
|---|---|---|---|
| Decode, drafting off | 80.3 / 87.9 (0.91) | 79.5 / 87.3 (0.91) | 74.1 / 80.5 (0.92) |
| Prefill, drafting off | 1,165 / 1,540 (0.76) | 2,746 / 2,733 (**1.00**) | 2,633 / 3,121 (0.84) |
| Decode, MTP | 98.4 / 128.4 (0.77) | 88.5 / 102.5 (0.86) | 92.5 / 115.0 (0.80) |
| Prefill, MTP | 1,088 / 1,496 (0.73) | 2,617 / 2,680 (**0.98**) | 2,516 / 3,093 (0.81) |

The same at `memory_fraction` 0.9: decode 0.88 / 0.88 / 0.90, MTP decode 0.73 / 0.83 / 0.77, prefill 0.74 / 1.00 /
0.84, MTP prefill 0.70 / 0.86 / 0.72. Before the lending fixes (037232e, staging every pass): decode 0.84 / 0.83 /
0.86, prefill 0.79 / 0.62 / 0.52, MTP decode 0.64 / 0.71 / 0.68, MTP prefill 0.77 / 0.60 / 0.50.

**RTX 5090, IQ3_S** (57e7bb4; 8K steps, `token_embd` on the host, `memory_fraction` 0.9):

| Row | 1K | 4K | 32K |
|---|---|---|---|
| Decode, drafting off | 106.3 / 109.7 (0.97) | 108.8 / 106.0 (**1.03**) | 98.5 / 106.5 (0.93) |
| Prefill, drafting off | 1,248 / 2,208 (0.57) | 4,244 / 4,779 (0.89) | 5,069 / 5,917 (0.86) |
| Decode, MTP | 144.7 / 185.5 (0.78) | 113.5 / 149.6 (0.76) | 128.1 / 183.9 (0.70) |
| Prefill, MTP | 1,171 / 2,174 (0.54) | 3,943 / 4,743 (0.83) | 4,831 / 5,883 (0.82) |

Before the lending fixes: decode 0.92 / 0.98 / 0.88, prefill 0.67 / 0.46 / 0.41, MTP decode 0.66 / 0.70 / 0.63.

- **Experts cached on the 5080** (`moe.cache` fill; C-M1's target ≥ 23% with the MTP pack): drafting off 23.2% (7.65
  GiB) at 0.9, 25.5% (8.41 GiB) at 0.97; with the MTP pack 20.5% (6.91 GiB) and **22.7% (7.66 GiB)** — the pack's 512
  gate/up experts all resident, floored (one layer's class). Strata: 31% drafting off, 28.5% with its MTP head (§13).
- **Strata's cache hit rate** at decode: 0.85–0.91 (5080), 0.92–0.96 (5090); its accepted tokens per pass 2.2–3.4.
  Holon's MTP accepts ≈ 3.0 tokens per pass (the 5080 smoke: 200 tokens in 66 passes).
- **1K prefill regressed with lending** (5080 1,220 → 1,137 at 0.9; 5090 1,467 → 1,248): a 1K prompt is a ring
  step (`moe.ring_tokens` 512), whose lend rounds and tail cost more than staging saves at that size. `ring_tokens`
  is one of D12d's hand-set values that step 7 measures per card.

### 20.3 Gates (c) and (d), and the chat

- **Chat** through `holon serve` (OpenAI frontend, MTP on): a haiku and its explanation, 406 tokens in 5.8 s on the
  5080, 333 in 3.8 s on the 5090 (wall, from the request; reasoning included).
- **(c) GPU self-consistency** (`spikes/fn7/consistency.py`, bf16, 4 prompts, 49,596 positions; the deployment's
  flags, its own step as the reference). Ran on 21bb40f, before the lending fixes: a ring step is bitwise a staged one
  on the card, and placement changes no bits (the offload store is bitwise the resident store). **Both pass**:

| Pair | 5080 KL mean / p50 / p99 | 5090 KL mean / p50 / p99 | worst 1K band (reported) |
|---|---|---|---|
| decode (1-token steps) vs prefill | 0.025 / 0.0018 / 0.35 | 0.032 / 0.0031 / 0.46 | −0.41 / +0.32 |
| chunked (250) vs one-shot | 0.024 / 0.0019 / 0.36 | 0.033 / 0.0028 / 0.47 | +0.37 / +0.58 |
| MTP vs plain (to each first difference) | 0.0009 / – / 0.0075 | 0.011 / – / 0.15 | – |

- **(d) The task set** (72 items, greedy, thinking on, 8,192 new tokens; Holon on 21bb40f, Strata's no-approx arm
  drafting off, same card):

| Card | Holon | Strata | Degenerate: Holon | Degenerate: Strata |
|---|---|---|---|---|
| RTX 5080 (IQ2_XS) | **71** / 72 | 70 / 72 | `code-08` (loop), `code-12` (limit inside the reasoning) | none |
| RTX 5090 (IQ3_S) | **71** / 72 | 70 / 72 | `code-08` | `code-08`, `code-12` |

  The score clause passes on both cards. **The degenerate clause fails on the 5080**: `code-08` and `code-12` are
  Holon-only there. They are the items where greedy reasoning at these bits loops or runs out: on the PRO 6000 both
  engines' answers were degenerate on both (§19.4), on the 5090 Strata's on both. Every engine-and-card pair's answer
  to them differs from the others' within the first 120–960 characters, so which run escapes is a matter of path;
  the 5080's Strata run escaped on both. Reported to the owner as measured; the bar is unchanged.
  **Lead's recommendation, pending the owner (2026-10-22):** judge the clause on degenerate rates pooled across the
  cards, with `code-08` / `code-12` treated as path chance. Pooled over the three cards' runs, Holon 5 / 216 (2.3%;
  7 / 288 with the 5080's F5b-on run) and Strata 5 / 216 (2.3%): equal. The bar text stays until the owner answers.

### 20.4 The RTX 5080 decode step per component

`spikes/fn7/breakdown.py` (nsys, eager steps, GPU ms per step; 57e7bb4, the deployment at `memory_fraction` 0.9;
graphed, a decode token takes ≈ 13.0 ms, Strata's 11.4):

| Component | Decode 4K | Decode 32K | Verify 4K (1 + 3 rows) |
|---|---|---|---|
| **Miss transfers** (`k_fetch`: F5a's zero-copy gathers, 2 per layer) | **4.79** | **5.85** | **16.22** |
| Routed experts (mmvq, its activation quantization, the weighted sum) | 1.74 | 1.72 | 3.66 |
| Router + top-k (`holon_moe_router`) | 0.53 | 0.53 | 0.53 |
| The store's resolve (`k_resolve`, 1 per layer) | 0.50 | 0.52 | 0.59 |
| GDN (36 layers) | 2.22 | 2.22 | 4.07 |
| HC mixes and combines | 2.08 | 2.09 | 3.23 |
| Shared expert | 1.10 | 1.11 | 0.92 |
| QSA (12 layers) | 1.04 | 1.12 | 1.29 |
| lm_head, PLE (host rows), other | 0.60 | 0.60 | 1.65 |
| MTP head (its sub-steps; its experts' fetches 0.91) | – | – | 3.77 |
| **Step** | **14.62** | **15.78** | **35.93** |

- **The miss transfers are a third of a decode step and 45% of a verify pass.** The cache hits 0.81 of decode
  lookups (0.77 with MTP; `spikes/fn6/hitrate.py` on `v1-long` prompts the placement profile never saw), against
  Strata's 0.85–0.91 on this card from a larger cache (31% of the experts against Holon's 23–25%). A decode token
  misses 94 of its 480 routed experts (10 per layer), 128 MiB of gate/up and down: ≈ 28 GB/s against the breakdown's
  4.79 ms. A verify pass routes 4 rows: its fetches take 3.4× a decode step's.
- **The levers, ranked by the step's time they can reach:**
  1. Fewer miss bytes — a larger cache (the reserve gave +2.3 points of cached experts and +2–4% decode, measured in
     §20.2; C-M1's ≥ 23% with the pack is 22.7% at 0.97) and a hit rate nearer Strata's (lfu's admission against
     Strata's adaptive swaps); verify gains most.
  2. Routed experts, router and resolve: 2.8 ms (`kernels.ggml_cuda`'s mmvq and router fusions, held as D / E).
  3. GDN 2.2 ms (its conv and gates, §19.6's item-4 candidates); HC 2.1 ms.
  4. The shared expert 1.1 ms (held as D), QSA 1.0 ms.
  5. F5b's CPU split: measured below (§20.5), it loses on this card.
- The store's own work at decode (resolve 0.50, the PLE and `token_embd` host gathers 0.03) is ≈ 4% of the step.

### 20.5 The 5080's MTP rows: F5b off and on (owner, 2026-10-21)

- **Tune's split** (`tune.split` at batch 1 with MTP, the deployment's flags; the card's 8-core CPU): F5a 95.1 tok/s;
  the CPU's share at 0.5 with 2 / 4 / 6 threads 26.5 / 46.5 / 63.7; at 6 threads, shares 0.25 / 0.4 / 0.5 / 0.6 / 0.75
  / 1.0 give 87.7 / 76.1 / 63.7 / 56.4 / 43.7 / 20.4. Best: 0.25 at 6 threads, −7.7%: **not taken** by tune's ≥ 5%
  rule. (Before the lending fixes: F5a 87.5, best 77.2, −11.7%.)
- **The F5b-on arm** at that share and thread count, against F5b off (both `memory_fraction` 0.9, same allocation):
  decode 70.3 / 72.7 / 67.9 against 77.6 / 77.0 / 72.4; MTP decode 81.1 / 74.5 / 85.0 against 93.5 / 84.7 / 88.0;
  prefill unchanged. F5b costs 3–13% on every decode row.
- **Its checks** (no fp32, §18): gate (c) passes (decode vs prefill KL mean 0.025, p50 0.0018, p99 0.37; chunked as
  F5b off — a 250-token pass takes no CPU share; MTP vs plain KL mean 0.0026, p99 0.062); gate (d): 70 / 72 against
  Strata's 70, the same two items degenerate as F5b off (`code-08`, `code-12`).
- **So neither of the owner's two options holds as framed:** F5b does not raise the MTP rows on this card, and the
  MTP rows are 0.77 / 0.86 / 0.80 with the reserve, not ≈ 0.95–1.0. The gap is §20.4's: verify passes spend 45% of
  their GPU time fetching misses. The decision goes back to the owner with these numbers.

### 20.6 Where step 6 stands

- **Met:** FN serves end to end on both bar cards (chat, gates (c) on both, (d)'s score on both); the PLE table and
  `token_embd` live in host RAM at ≈ 0.2% of a decode step; the MTP experts go through the store; lending and the
  ring work on both cards; the 5080's three memory levers are applied. Rows at or above the bar: the 5080's 4K
  prefill (1.00; 0.98 with MTP), the 5090's 4K decode (1.03).
- **Below the bar:** decode 0.91–0.92 (5080) and 0.93–0.97 (5090) off the 4K row; MTP decode 0.77–0.86 and 0.70–0.78;
  prefill at 32K 0.84 / 0.86, at 1K 0.76 / 0.57 (`ring_tokens`, §20.2).
- **For the owner:** (d)'s degenerate clause on the 5080 (§20.3), and the 5080 MTP rows (§20.5).
- **Not yet done in step 6:** the PLE SSD tier, tune's interleaved F5b benchmark (b), the plants and exactness checks,
  the full suite.

## 21. Step 7 on limited VRAM (2026-10-22; the lead's order after the Strata source study)

The owner delegated the step's decisions to the lead (2026-10-22), who re-ordered the work after a study of Strata's
source: its 0.85–0.91 hit rate leaves out the misses its GPU reads over PCIe (≈ 0.83 counted Holon's way); its real
difference is miss serving — ⌊n · pcie_frac⌋ of a layer's misses over PCIe, the rest on its CPU pool beside the GPU's
hits —, a CPU path ≈ 5× ours per expert, its prompt buffers borrowed from the cache, and THP-backed host arenas.
The items, in order: 0 the bench's hit rates both ways; 1 the ring's copies and 1K prompts; 2 `k_fetch` at link rate;
3 the shared expert on a forked stream; 4 F5b's CPU speed, then a Strata-style split; 5 memory reclaim; 6 MTP
(catch-up, the head's experts pinned); 7 the HC prefill as bf16x3; 8 D/E. F5b as it measured in §20.5 is dropped as
a lever. Each item: L0, the GPU self-consistency gate, a 3-round A/B. Development rules as §19's. Records:
`evidence/fn7_step7_2026-10-22.json`; the scripts in `spikes/fn6/` and `spikes/fn7/`.

**Gate (d)'s degenerate clause** (§20.3): the lead's recommendation is recorded there, pending the owner.

### 21.1 Items 0–3

- **0 — hit rates both ways** (16440d1): `bench/offload_ctx.py` reads Strata's PCIe-miss field (`DONE`'s offloaded
  count, St #588) and reports `hit_rate` (Strata's own) and `hit_rate_all` (hits over every routed expert, as Holon
  counts), with Strata's PCIe share; Holon's arms report `hit_rate_all` from the store's taps and the MTP tokens per
  pass.
- **1 — the ring's copies and 1K prompts** (16440d1, 7159bcb): FN's deployment takes `moe.ring_copy=dma` and
  `moe.ring_tokens=1025` (a 1K prompt stages; a ring step's lend rounds and tail cost more than staging saves at that
  size). A ring step now lends the tail its own passes take (sized by the step's tokens), not the largest pass's.
  The first card run of that change found a bug: the rest of the tail's range still held residents the rounds could
  pick, counted as held but outside the pool, so a ring list fell short (DMA: a copy past dst; SM: a repeated
  staging slot, mmq's column error). Fixed in 7159bcb: a tail step's lend end is its own tail's start.
- **2 — `k_fetch` toward link rate** (313c85f): `spikes/fn7/fetch_bw.py` on the RTX 5080 (8 GiB mapped pool, units
  of an IQ2_XS expert's 1.5 MB): the store's 8-CTA zero-copy gather alone reaches 43–45 GB/s over 4 KiB pages and
  45–48 GB/s over THP (`madvise(MADV_HUGEPAGE)`, page-locking 0.017 s/GiB there), Triton at 8–128 CTAs 44–51,
  copy-engine DMA 46–54. So the in-engine ≈ 27 GB/s is not the gather kernel's bandwidth: it is two small launches per
  layer (2–4 MB). `expert_fetch@1` gains an optional `gather_roles` — a group's every role in one launch, chunks of
  both roles dealt round-robin over `expert_io.fetch_ctas` CTAs per role — which the store's resolver uses
  (`expert_io.fetch_roles=off` withholds it for A/Bs). THP host pools would add ≈ 5% more and need a core platform
  option: not built yet.
- **3 — the shared expert on a forked stream** (74bf2d3): `std.ffn.sum@1` runs its later parts on a side stream,
  joined just before `std.moe@1`'s combine reads their sum (`add_residual`'s `join`); bitwise the same on IQ2_XS with
  the offload store in prefill and graphed 1-token steps (`spikes/fn7/bitwise.py`), and the 5080's deployment with
  items 1 and 3 bitwise its base at a 4K prompt, prefill and decode. `ffn.fork=off` keeps one stream.
