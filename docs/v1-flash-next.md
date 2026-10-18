# v1 — FN: Qwen3.8-Flash-Next on Holon, then its expert offload compared with Strata

Status: **rev 3 (2026-10-18), the second panel round applied.**
- **Round 1 (rev 1 → rev 2):**
  - Panel A (architecture): 1 blocker, 7 majors, 9 minors.
  - Panel B (correctness): 2 blockers, 8 majors, 12 minors.
  - Panel C (performance and fairness): 1 blocker, 8 majors, 10 minors.
- **Round 2 (rev 2 → rev 3):**
  - Panel A: 1 major, 3 minors.
  - Panel B: 2 majors, 7 minors.
  - Panel C: 1 major, 7 minors.

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
- FN rebases onto it and uses its D1–D3 and D5.
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
   - **Absolute decode check:** bounded's `divergence` (decode rescored by a prefill reference) on the long trace is
     gated against the same measure on a dense control trace (every decode position ≤ 2,051): KL p99 sparse ≤
     max(2 × dense, dense + 2.5 SE).
   - **GPU unit-level bitwise check of `qwen4_exp.ple_block@1`:** chunked vs one-shot, and replay-rewind vs k decode
     steps. The fp32 conv is a per-channel 4-tap sum with no cross-row reduction, so both pairs must match bitwise.
3. **Kernels.** L0 for every new kernel against its `reference.py` on sm_86, sm_89 and sm_120, memcheck clean.
   - Decode-step targets on the RTX 5080: ≤ 16 kernel launches per layer, all graphed; `qsa_select` ≤ 20 µs per
     layer at 32K context (8K blocks, ≤ 8 query rows).
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
    - **Not committed:** the tiny GGUF. It is a sha256-pinned artifact that CI fetches, so §6.1 runs in CI instead of
      skipping as `qwen3_5_files.py:86-87` does when the converter is absent.
  - **The MTP pack:** the tool also writes a Strata-format MTP pack for the tiny model with St `tools/mtp_pack.py`
    (HF names, fused `gate_up_proj`, BF16 dense parts, F32 norms, MSE-scaled Q2_0). Holon's MTP rules load this file
    on the real model, so the pack format gets its own oracle (§6.1).
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
  - fp32 scores (Triton `input_precision="ieee"`); an exact top-k, ties going to the lower block (D2's rule).
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
  - Spans per context start at fixed offsets, wrapping: o_k = k × ⌊N / 6⌋ for k = 0…5, where N is the corpus
    length. That gives 6 candidates per context; the first 3 that survive (all 6 if `--eos-ids` works) are used.
  - At 32K the 6 candidates overlap by less than half when N ≥ 200K.
  - The ids, offsets and the corpus's sha256 go in the evidence.
- **Strata, no approximation:**
  - `STRATA_ROUTE_TAIL_SKIP=0 --kv fp16 --ple-io ram`, after `--calibrate` on that card and file (`docs/DETAILS.md:489-503`; its settings are recorded);
  - KV streaming is off at these contexts (`--kv-resident` only at ≥ 64K, St `setup.py:6604-6620`), and this is
    recorded;
  - `STRATA_IQ_MT_MIN` is left at its default. It is a reproducibility switch, not an approximation, and costs Strata
    1–3% (`docs/DETAILS.md:97-104`);
  - the prefill CPU share stays on (below 1,024-token chunks only, `prefill.cpp:178-181`). The 1K prompts are exactly
    1,024 tokens.
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
  - ≥ 5 interleaved rounds (reversed on odd rounds), each with ≥ 3 distinct prompt spans per context;
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
  - ≈ 2.5–3 h per card per full set (≥ 5 rounds × ≥ 3 spans × 8 gated arms, plus the reported arms);
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
  - **What is lent:** the MoE's transient buffers for one pass — gate/up output, SwiGLU output, fp32 down output,
    q8_1 rows, ≈ 215 KB per token. A pass is 4,096 tokens (the track's D1 cap; its D6 8K passes are conditional),
    so ≈ 0.88 GB is lent during an 8K step.
  - FN's only use: the offload deployment turns lending on, through the lock (d).
- **(d) The step size is a lock setting per card** (`max_step_tokens`, with lending on or off).
  - One deployment at `--ctx 33792` serves the 1K, 4K and 32K rows: the 1K and 4K prompts are single steps, and the
    8K-step peak is reserved in every row.
  - Tune chooses it by weighing 32K prefill against decode.
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
- `qwen4_exp.ple_block@1` (fp32; in-step payloads; replay rewind);
- `qsa_index@1` (pooling from stored raw keys);
- `qsa_select@1` (fp32 ieee; lower-index ties; sub-batched workspace);
- `qsa_attend@1` (prefill, decode, verify; gqa's KV formats).

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
| MoE | 8 experts top-2 + shared |
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
| decode divergence, long trace vs dense control (acceptance 2) | pass: KL p99 sparse ≤ max(2 × dense, dense + 2.5 SE) |
| GPU unit test, `qwen4_exp.ple_block@1`: chunked vs one-shot, replay-rewind vs k decode steps | bitwise |
| offload vs all-resident (capped, `lfu`, MTP experts in the store) | byte-identical |
| PLE RAM vs SSD | byte-identical |

### 6.3 Plants

Each plant changes only its own path.

| Plant | Breaks | Predicted |
|---|---|---|
| `qsa_select_shift` | prefill selection shifted one block past the budget | L2 fails on `v1-long`, passes on `v1-qwen3_5` |
| `qsa_decode_select_shift` | the same, in decode and verify only | L2 clean (teacher-forced) and the bounded checks clean (both sides decode with the plant); **the decode-divergence gate fails** (sparse KL p99 rises, the dense control does not) |
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
- **Staging and null slots:** ≈ 40 per class × 4 classes ≈ 0.2 GB.

**Per-request state:**
- **KV at 32K:** 13 × 2,432 B per token ≈ 1.04 GB bf16, 0.63 GB as q8_0.
- **GDN:** (`max_seqs` + 1) × (1 + `draft_slots` 0) = 2 slots × 36 × 3.06 MiB ≈ 0.23 GB, plus one checkpoint
  ≈ 0.12 GB. Rev 1's 8 slots were wrong (A-M2).
- **PLE:** 368 KB per slot.
- **KV and state together:** 1.39 GB.

**The step peak:**
- **Not lendable** (R, HC, GDN with FLA's chunk state — `h` alone is ≈ 24.6 KB per token, `chunk_delta_h.py:378` —
  QSA, selection sub-batches): **≈ 200 KB per token**. Step 0 measures it.
- **Lendable** (the MoE scratch for one pass, D12c): ≈ 215 KB per token × the pass. A pass is min(step, 4,096)
  tokens, the MoE prefill track's D1 cap.

**RTX 5080, IQ2_XS:** 0.9 × (16.61 − 0.9 − 4.65) = 9.95 GB, less KV and state 1.39 and staging 0.2, leaves 8.36 GB
for the cache and the non-lendable peak.

| `max_step_tokens` (one per card; tune's axis) | 2K | 4K | 8K |
|---|---|---|---|
| Non-lendable peak | 0.41 GB | 0.82 | 1.64 |
| Lent during prefill (scratch for one pass; the ring's ≈ 1.1 GB comes on top, as in X) | 0.44 | 0.88 | 0.88 |
| **Decode expert cache, with lending** | **7.9 GB (22%)** | **7.5 (21%)** | **6.7 (19%)** |
| Decode expert cache, without lending | 7.5 (21%) | 6.7 (19%) | 5.8 (16%) |
| Prefill cap from streaming non-resident and lent experts at ≈ 47.5 GB/s | ≈ 27.8 GB per step → ≈ 3.5K tok/s | ≈ 28.6 → ≈ 6.8K | ≈ 29.3 → ≈ 13K |

Notes on the 5080:
- **One deployment per card:** `--ctx 33792` serves every row, so the 8K step's peak is reserved in the 1K and 4K
  rows too. The proposed lock is 8K steps with lending; tune decides.
- **Lent slots return empty** (T2), so a 32K decode starts with a dipped hit rate. Each cell reports the hit rate over
  the first 32 decode tokens (D11). Strata's borrowing does the same.
- **MTP experts in the store** instead of resident: ≈ +0.7 GB of cache.

**RTX 5090, IQ3_S:** 0.9 × (33.88 − 0.9 − 5.40) = 24.82 GB, less 1.39, 0.2 and 1.64 (8K steps), gives ≈ 21.6 GB:
43% of 50.29 GB. Its prefill cap is similar, ≈ 13K tok/s at 8K steps.

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
| 7 | **Perf closure** on the bar cards: nsys per layer, fusions, CUDA rewrites by D10's rule, step-size and lending tune axes | every row within its roofline band, or the gap explained | 4 |
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

