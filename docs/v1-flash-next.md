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
   - **(b) Full model.** L2 Tier B, all-resident on the RTX PRO 6000, for ISTA IQ2_XS and IQ3_S, against
     layer-streamed fp32 goldens (D2). Prompt sets:
     - `v1-qwen3_5.jsonl`. Only 3 of its 24 prompts pass the 2,051-token dense limit.
     - A new `v1-long.jsonl`: 8 prompts of 4K–32K tokens, 4 of them recall prompts.
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
   - ≤ 16 kernel launches per layer, all graphed, is a step-7 gate (§14's launch table).
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
   | Strata defaults vs Holon + A1 | same | ≥ 1.0× | ≥ 0.9× |

   - Each row is gated twice: drafting off, and MTP matched. Contexts are 1K / 4K / 32K.
   - Holon runs **its lock's configuration**. F5b rows that differ from the lock are reported separately.
   - Holon exact vs Strata's defaults is reported, not gated.
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
- `std.moe@1` is not edited.

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

**RTX PRO 6000, all-resident:** 40.1 / 55.7 GB of weights (IQ2_XS / IQ3_S, with MTP). KV up to 128K is 4.1 GB.

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
| 7 | **Perf closure** on the bar cards: nsys per layer, fusions, CUDA rewrites by D10's rule; measures `max_step_tokens` × lending for D12d's hand-set lock values | every row within its roofline band, or the gap explained; ≤ 16 launches per decode layer (gate) | 4 |
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
  - **rev 4 carries a launch band.** The overhead is scaled to 16 launches per layer (step 7's gate); each decode cell
    is also given at 24, today's projection for a full layer (§14's launch table). The fusions that close the gap
    have owners (§14).
- **Bars: decode 1.0×, prefill 0.9×** (acceptance 5). Rev 2's table measured every cell against 0.9×; that is
  corrected here.
- **Holon / Strata, central [band] → at 24 launches per layer (rev 4; supersedes revs 1–3).** The central and band
  assume step 7's 16 launches; the arrow gives the central at 24.

| Gated row (bar) | RTX 5080: 1K · 4K · 32K | RTX 5090: 1K · 4K · 32K |
|---|---|---|
| no-approx, drafting off, decode (1.0×) | 1.03 · 1.01 · 1.07 [0.93–1.16] → **0.92 · 0.90 · 0.96** | 1.13 · 1.12 · 1.17 [1.01–1.30] → **0.96 · 0.96** · 1.00 |
| no-approx, MTP-matched, decode (1.0×) | **0.89 · 0.90** · 1.06 [0.86–1.11] → **0.84 · 0.85** · 1.01 | **0.97** · 1.02 · 1.03 [0.92–1.09] → **0.89 · 0.94 · 0.95** |
| no-approx, prefill (0.9×) | 1.19 · 0.99 · 0.93 [0.73–1.36] | 1.05 · 1.03 · 0.97 [0.76–1.18] |
| *RTX 5080 at step 7's memory target:* drafting off · MTP-matched decode | 1.06 · 1.03 · 1.10 · **0.93 · 0.95** · 1.12 → 0.94 · 0.92 · 0.98 · 0.88 · 0.89 · 1.06 | — |

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
- **At 24 launches per layer** (today's projection), every drafting-off decode cell of the RTX 5080 misses (0.92 /
  0.90 / 0.96) and the 5090's 1K / 4K do (0.96 / 0.96); its MTP cells miss at every length. Step 7's ≤ 16 gate and
  the named fusions (§14) are what keep these cells at their central values.
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
1. **Owner item; the lead recommends report-only for the Strata defaults rows.** Gate the no-approx and MTP-matched
   rows, which compare engine with engine. Report the defaults rows without gating them: they measure features outside
   FN's scope (the suffix drafter is a non-goal, and int8 KV is a format choice). Pending the owner's answer; work does
   not block on it.
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
to ≈ 20–24 launches (panel C); N3 carries a 16–24 band (§13, rev 4), and ≤ 16 is a step-7 gate (acceptance 3).

| Piece | Before (2a761fb) | Now |
|---|---|---|
| HC: two mixes, two combines | 10 (rstd, cat, down, up ×2; combine ×2) | 5 (down, up ×2; the attention half's combine rides in the FFN half's mix; the block's last combine before it returns) |
| QSA: index · select · attend · output gate | 2 · 3 (with a fill) · 2 · 2 (gqa's sigmoid, mul) | 1 · 2 · 2 (1 for prefill) · 0 |
| PLE (layer 1 only): rows · block | 2 · 3 | 2 · 3 |
| Step 4's share of a QSA layer | 19 | 10 |

The fusions that close 20–24 → 16, with owners:
- **FN, step 7:** GDN's fused input projection; the block's last HC combine folded into the MoE combine's epilogue,
  within the same block only (R complete on return: the trunk contract); router and top-k in one launch.
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
| C-M1 | Launches | **Done in part.** `_rstd` folds into `_down`, the attention half's combine into the FFN half's mix (5 HC launches per layer, after A's major), `_write_raw` + `_pool` into `_index`. The serial last-split merge variant was measured (+24 µs at decode) and rejected; the output gate folds into the attend's store instead (same count). Table above; ≤ 16 is a step-7 gate. |
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
| A-1 | PLE's kernels | **Moved** to `kernels.ple_triton` (cuda, `bounded`); the model plugin keeps the contracts and oracles, numerics undeclared. |
| A-2 | Model constants in std | **Removed** (`RATIO`, `BUDGET`, `SUB_BATCH`). |
