# v1 — FN: Qwen3.8-Flash-Next on Holon, then its expert offload compared with Strata

Status: **rev 1, draft for the challenge panel** (2026-10-18). No panel round yet (§12). Parameters marked "est."
freeze when step 0's evidence lands (§8).

**Scope.** Plan §10 ("Expert caching/offload", milestone FN) and `docs/v1-offload.md`:
- D2 sets the target;
- §1 acceptance 4 sets the program bars FN answers to;
- §4.7 lists FN's core needs;
- plan §10 has FN's two follow-ups: aux-model (MTP) experts through the offload store, and tune's interleaved F5b
  benchmark.

FN builds on milestone X as merged (`b5b1e7f`: offload store, ABI 1.3), on M0.3 (Qwen3.8's GDN, gated attention,
q8_0 KV) and on M0.4 (MTP as an aux model).

**References** (local clones; FN cites `file:line` from them):

| Ref | Version | Abbreviation |
|---|---|---|
| Strata | `296b59d` (2026-10-14), MIT | **St** |
| FreeToken | `21d9578` (2026-10-16), Apache-2.0 | **FT**: `python/freetoken/` |
| llama.cpp | `27c7a76` (2026-10-17) | **LC**: `src/` |
| HF transformers 5.19.0 | `transformers/models/qwen4_exp/modeling_qwen4_exp.py` / `modular_qwen4_exp.py` | **HFm** / **HFx** |
| sglang 0.5.21 | `srt/models/qwen4_exp_mtp.py` | **SG** |

`docs/brainstorm/v1-offload-research.md` §3.1 ("R §3.1") is X's study of Strata's tiers.

## 0. The model in one table

From `config.json` (`Qwen4ExpForConditionalGeneration`, text `qwen4_exp_text`) and the GGUF headers (§0.1).

| Part | Fact | Source |
|---|---|---|
| Trunk | 48 layers, hidden H = 2560, vocab 248,320, untied head, **no input / post / final norms** | HFm:1276-1301, 1485; LC qwen4exp.cpp:511-521 |
| Mixers | layers 3, 7, …, 47 (12) are **QSA**; the other 36 are **GDN** | HFx:182-192; St `geometry.hpp:26, 75-77` |
| QSA attention | 24 q / 2 kv heads, head_dim 256, `q_proj` = per head `[q(256) ‖ gate(256)]`, sigmoid output gate, zero-centred q/k norms, NeoX RoPE on the first 64 dims (θ = 1e7; the interleaved mRoPE [11,11,10] is plain 1-D RoPE for text) | HFm:811-893, 77-142 |
| QSA indexer | `index_qk_proj` → q 4 × 128 (zero-centred `q_layernorm`, RoPE at the query position), raw k 1 × 128 (cached unnormed and unrotated) | HFm:665-771 |
| QSA selection | complete 4-token blocks of the visible tokens; `kbar_b = rope(k_layernorm(bf16(mean_fp32(k[4b..4b+3]))), pos 4b)`; `s_b = Σ_h relu(⟨q_h, kbar_b⟩) / √128`; per query, top min(512, n_blocks) blocks ∪ the open tail block; dense exactly while p + 1 ≤ 2,051 | HFm:720-760; FT attention.py:173-174 |
| GDN | Qwen3.8's (16 k / 48 v heads × 128, conv 4, fp32 state); the gated norm uses **`sigmoid(z)`** where Qwen3.5/3.8 use `silu(z)` | HFx:313-327; LC qwen4exp.cpp:639-643; St `layer.hpp:8-17` |
| Hyper-connections | residual R = [T, 4 · 2560], initialised as the embedding repeated 4×. Two `GatedResidual`s per layer (attention, FFN) plus an output mixer, specified below. No stream-to-stream matrix. | HFm:995-1023, 1472 |
| MoE | router [512, 2560], fp32 softmax over 512 experts, top-10, renormalised; SwiGLU experts I = 640; shared expert SwiGLU I = 640 × `sigmoid(x · w_shexp_gate)` (one scalar per token); `out = routed + gated shared`; input is the HC-mixed x, with no pre-norm | HFm:952-992; St `shared_expert.hpp:3-8` |
| PLE n-gram | Applied once, at 0-based layer 1: `ple_layer_ids` [2] is 1-indexed (HFm:1260); GGUF `ple.layers` [1]; St `ple.hpp:44`. **Table:** 16 hash heads (8 bigram, 8 trigram) × 160 dims, 320,001,536 rows × 160 (IQ4_NL: 90 B per row, 28.80 GB). **Hash:** `(t0·m0) ^ (t1·m1) [^ (t2·m2)] mod vocab_h + offset_h`, with a window cut at EOS 248044. **Block:** `R += G + silu(conv_dil3(norm(G)))` with `G_s = gate_s(⟨norm(K E)_s, norm(R)_s⟩/√2560) · V E`; depthwise conv with kernel 4 and **dilation 3** (taps t−9, t−6, t−3, t) | HFm:1104-1247; HFx:576-632; LC qwen4exp.cpp:1216-1459; St `ple.hpp:3-13`, `ple.cu:137-163` |
| MTP | 1 layer, outside HF (HFm:1314, 1590 ignore `mtp.*`). Per stream, `R_s = fc_e(enorm(e)) + fc_h(hnorm(h)_s)`, where h is the trunk's final R before the output mixer. Then a full QSA + MoE (own 512 experts) + shared HC layer, its output mixer, and the shared `lm_head`. No PLE. | LC qwen4exp.cpp:289-301, 524-616; SG qwen4_exp_mtp.py:72-114 |

**Hyper-connections in detail** (HFm:995-1023):
- `Rn = grouped zero-centred RMSNorm(R)`, with statistics per 2560-wide stream.
- **mix:** `x = mean_s(sigmoid(W_up · silu(W_down · Rn / 4))_s ⊙ Rn_s)`, giving [T, 2560].
- **inject:** `w = 2 · sigmoid(W_inj · Rn / 4)`, giving [T, 4].
- **combine:** `R_s += w_s · y`.
- Weight shapes: `W_down` [320, 10240], `W_up` [10240, 320], `W_inj` [4, 10240].

**Open discrepancy (MTP):** LC normalises `hnorm` per stream with a [2560, 4] weight (qwen4exp.cpp:296-297, 581); SG
normalises over all 10240 features (SG qwen4_exp_mtp.py:76-83). D9 settles it.

### 0.1 Files (GGUF headers parsed by HTTP range; offsets sum to the shard sizes exactly)

| File (2 shards) | Total | Routed experts | PLE table | Rest (resident) | Per expert per layer | Expert types (layers) |
|---|---|---|---|---|---|---|
| ISTA-DASLab GSQ-RCO **IQ2_XS** | 68.03 GB | 35.45 | 28.80 | 3.76 | 1.44 MB | gate = up: IQ2_S 34, IQ2_XXS 11, **IQ1_M 3** (8, 13, 37); down: **Q2_0** 48 |
| ISTA-DASLab GSQ-RCO **IQ3_S** | 83.61 GB | 50.29 | 28.80 | 4.51 | 2.05 MB | gate = up: IQ2_S 20, IQ3_XXS 17, IQ3_S 10, IQ4_XS 1; down: IQ4_NL 39, **Q2_0** 9 |
| ISTA-DASLab GSQ-RCO IQ3_XXS | 75.84 GB | 42.91 | 28.80 | 4.12 | 1.75 MB | down: Q2_0 30, IQ4_NL 18 |
| unsloth UD-Q4_K_XL | 111.33 GB | 77.02 | 28.80 | 5.51 | 3.13 MB | gate / up Q4_K; down mostly Q5_1 |

Notes on the files:
- **Contents.**
  - Each file holds 1,224 tensors and **no MTP tensors**. MTP ships separately: unsloth and bartowski `mtp-*.gguf` (BF16 7.77 GB, Q8_0 4.14 GB, the MTP layer alone ≈ 2.78 GB at Q8_0), SC117 `mtp-q2_0` (0.89 GB).
  - Strata packs its own MTP from the BF16 checkpoint: Q2_0 experts, 708 MB resident (St `mtp.hpp:15-19`; `setup.py:6537-6545`).
- **Q2_0 is upstream ggml, not custom.**
  - It is type 42: 64 values in 18 B, code `(c − 1)·d`. It is in LC `ggml.h` and in Holon's vendored commit `cbe28bb3`: `ggml_cuda/vendor/.../ggml.h:432`, with mmq / mmvq / get_rows / convert cases (`mmvq.cu:43`, `mmq.cuh:569-572`, `dequantize.cuh:26-42`) and in ggml-cpu's `quants.c`.
  - Holon's `TYPES` tables omit it (`kernels/ggml_cuda/plugin.py:83-85`, `kernels/ggml_cpu/plugin.py:22-23`).
  - gguf-py 0.19.0's enum stops at Q1_0 = 41, so `weights/gguf/reader.py:82` fails on any ISTA file today.
  - It is the same format as Strata's (St `dequant_s2.hpp:3-6`).
- **IQ1_M** has mmvq in the vendored ggml (`mmvq.cu:61`) but no mmq (no template instance; `mmq.cuh` has no case). So the IQ2_XS file's three IQ1_M layers need a prefill path (D1).
- **Shards.** Holon's GGUF source refuses split files (`std/holon_std/weights/gguf/hub.py:44`), and every Flash-Next file is split. Shard 2 is exactly the PLE table, 28,800,138,432 B (St `ngram.hpp:193-197`).
- **Other fields.** Tokenizer pre `qwen35` (already in `weights/gguf/tokenizer.py:21`). HC weights are BF16 in ISTA and Q8_0 in unsloth. The router is BF16.

## 1. Goal and acceptance

**Goal.** Serve Qwen3.8-Flash-Next from its GGUF files, text-only:
- exactly against an independent fp32 oracle;
- all-resident on a 96 GB card;
- through X's offload store on 16 GB and 32 GB cards;
- then meet the program bars against Strata, the strongest engine for this model.

1. **Model** (`models.qwen4_exp`, `derived_from` `models.qwen3_5`; `arch.gguf.qwen4exp@1`,
   `arch.Qwen4ExpForConditionalGeneration@1`):
   - **(a) Tiny model.** A random-weight reduced config (§6.1) matches the HF reference on the CPU. Logits are
     within fp32 tolerance, and every QSA (layer, position) selects the same token set. A prefix hit is bitwise equal
     to recomputing.
   - **(b) Full model, L2 Tier B.** All-resident on the RTX PRO 6000, for ISTA IQ2_XS and IQ3_S, against layer-streamed
     fp32 goldens (D2). It runs on two prompt sets:
     - `v1-qwen3_5.jsonl`;
     - a new long set, `v1-long.jsonl` (8 prompts, 4K–32K tokens), where the indexer actually selects. The short set
       crosses the 2,051-token dense limit in only 3 of 24 prompts (its longest is 2,146 tokens).
2. **Bounded and run-identical** (full model, RTX PRO 6000):
   - chunked vs one-shot prefill, with chunk edges that straddle 4-token blocks and the PLE conv window;
   - prefix-hit resume vs recompute;
   - graphs vs eager;
   - overlap 1 vs 0;
   - the snapshot trace run-identical over 3 fresh engines;
   - MTP on: the verify path bounded (L2) and the snapshot run-identical. Mean accepted tokens per pass at 3 drafts
     ≥ 2.2 on the chat and agent prompts (Strata: 2.4–3.2, St `docs/HOW_IT_WORKS.md:43-46`).
3. **Kernels:** L0 for every new kernel against its `reference.py` on sm_86, sm_89 and sm_120, with
   `compute-sanitizer` memcheck clean.
4. **Offload exactness** (X's acceptance 2 carried to FN):
   - The offload store at a capped fit, with `lfu` and the MTP layer's experts through the store, is snapshot
     byte-identical to all-resident on the RTX PRO 6000.
   - On the RTX 5080 and 5090, cross-placement identity holds over 3 runs.
   - The PLE table's RAM and SSD tiers are byte-identical.
   - F5b, where tune picks it, is bounded (L2 against the path-aware floor, run-identical over 3 runs).
5. **Program bars** (`docs/v1-offload.md` §1.4, proposed there; owner to confirm). The protocol is D11; rows are
   medians of interleaved rounds at 1K / 4K / 32K:

   | Against | Card, file | Decode | Prefill |
   |---|---|---|---|
   | Strata, no approximation | RTX 5080 IQ2_XS; RTX 5090 IQ3_S | ≥ 1.0× | ≥ 0.9× |
   | Strata defaults vs Holon with A1 | same | ≥ 1.0× | ≥ 0.9× |

   - Each row is measured twice: with drafting off, and with MTP matched (D11).
   - Holon exact against Strata's defaults is reported, not gated.
   - FreeToken (`models/qwen4_exp`) and llama.cpp (`qwen4exp`, static split and `--moe-cache-mib`) are reported rows
     wherever they run the file (open question 8).
   - Plan §9's kill line (< 0.8×) applies to the like-for-like rows.
6. **Memory** (§7):
   - **RTX 5080:** IQ2_XS at 32K with MTP.
   - **RTX 5090:** IQ3_S at 32K with MTP.
   - Each fit is reported against §7's prediction.
   - Every pinned byte (experts, the PLE table) is accounted against `host_memory`.
7. **Locks:**
   - `rtx-5080` (IQ2_XS, offload) and `rtx-5090` (IQ3_S, offload), written by `tune --new-model`, approved on
     passing verdicts.
   - The existing locks are propagated once, at the close (D16).
8. **Core:** only C1 and C2 (§4), each an additive ABI minor (→ ABI 1.4), toy-proven, ≤ 60 code lines in total.

**Non-goals:**
- the vision tower (mmproj);
- KV streaming for QSA (Strata C4, milestone K);
- Strata's suffix drafter and adaptive draft depth;
- NVFP4 packs, Q8_0 / BF16 PLE tables, the Coder variant (256 experts; expected to work, not gated);
- multi-GPU;
- Qwen3.6-35B-A3B (D13, owner to confirm);
- the plan's MoE prefill-speed item, unless step 0's roofline needs it (R2, escalated).

## 2. Decisions

**D1 — Files and quant types: ISTA GSQ-RCO IQ2_XS (RTX 5080) and IQ3_S (RTX 5090), the bars' files.** These are
Strata's recommended family (St `setup.py:243-309, 383-389`), so the comparison runs on identical bytes.
- **Q2_0:**
  - add it to `kernels.ggml_cuda`'s and `kernels.ggml_cpu`'s `TYPES` (the vendored kernels exist, so no
    `vendor.patch` change);
  - add a Holon-side type entry in `std/holon_std/weights/gguf/reader.py` for ids gguf-py lacks: `{42: (64, 18)}`,
    named `q2_0`.
- **IQ1_M** (IQ2_XS gate/up at layers 8, 13, 37):
  - decode runs mmvq;
  - prefill rows above the mmvq limit run a new `linear_experts` path in `kernels.ggml_cuda`: dequantize the
    called slots to bf16 (the vendored `convert.cu` IQ1_M case) into a workspace, then a grouped bf16 GEMM.
  - It is deterministic and slot-invariant. The cost is 3 of 48 layers' gate/up.
  - The same path later serves plan §10's MoE prefill-speed item.
  - Its floor is path-aware: no q8_1 emulation on those layers' prefill rows (§6).
- **Rejected:** switching the RTX 5080 bar to IQ3_XXS (no IQ1_M, 42.9 GB of experts) or Q2_0. That would change the
  bar's file, which is the owner's call (open question 6).

**D2 — The oracle is HF transformers' `qwen4_exp`, at a pinned version, in a golden-only environment; the full
model's golden and floor are layer-streamed.**
- The workspace pins transformers 5.6.2 (`std/pyproject.toml:8`), which has no `qwen4_exp`. Bumping it would re-lock
  the runtime closure and propagate every existing lock for nothing.
- `holon_judge golden` therefore runs from a separate venv, `local/envs/golden-qwen4`: transformers 5.19.0 plus
  `-e judge`, installed with `--link-mode copy`. The golden records that version in its metadata.
- **Tiny-model fixtures:** CI cannot import the class. A tool in `testing/holon_testing/qwen4exp_files.py` (run in
  the golden venv) writes the random tiny model, its GGUF and HF's fp32 outputs as committed fixtures (§6.1). CI's
  CPU test compares Holon against the fixture.
- **Full model:**
  - 177B parameters are 708 GB in fp32 and 354 GB in bf16, so neither the CPU golden nor the bf16 floor fits whole.
  - `golden --stream-layers` builds the HF model on the meta device. It runs every prompt through one decoder layer
    at a time: the layer's dequantized fp32 weights (≈ 10 GB of experts), the residual for all prompts' tokens in RAM
    (≈ 7 GB for ≈ 160K tokens), the QSA and PLE state per prompt. It then frees the layer.
  - The floor is the same driver in bf16 on the RTX PRO 6000 with `--q8-1`, one layer resident at a time.
  - The driver calls HF's own decoder layer, rotary, mixer and head modules. It does not re-implement the model.
  - It is generic over HF decoder-layer models: a judge feature, not a model special case.
- **MTP:** HF has no MTP, so MTP is judged by its verify path's L2 and by acceptance (acceptance 2).
- **Cost:** HF's indexer loops per query in Python (HFm:720-760). A 32K prompt costs minutes per QSA layer (est.),
  acceptable once per golden.

**D3 — Hyper-connections: the trunk carries R flattened as [T, n · H] in fp32; the embed unit expands; the model's
`norm` is the HC output mixer; one core fact gives drafters and the profile R's width and dtype (C1).**
- **Why 2-D:** every core site that touches h indexes rows on dim 0 — `_logits`, prompt logprobs, `_worst_step`,
  `head` (`core/holon/core/executor.py:232-233, 426-427, 615`). A flattened `[T, 10240]` keeps them unchanged; the
  `[T, n, H]` view is the units' business.
- **Why a core fact:** the MTP head needs R itself, not the mixed x (§0). The executor's startup profile hands the
  drafter `zeros(T, spec.hidden, cfg.dtype)` (`executor.py:266`), which would mis-shape it.
- **Why fp32:**
  - R accumulates 96 sublayer updates, each scaled by `w ∈ (0, 2)`. Strata (St `gr.hpp:16-19`) and llama.cpp keep
    it fp32; HF and FT keep it bf16.
  - Cost: 40 KB per token, and about 2% of a 4K prefill chunk's time in extra R traffic (est.: 96 read-writes of
    168 MB at ≈ 900 GB/s ≈ 35 ms more than bf16).
  - The golden is fp32 and the floor bf16, so fp32 R moves the candidate toward the golden.
- **Rejected:**
  - (a) block 0 expands and the last block collapses, with no core change. MTP needs R, and taps would see a shape the
    contract does not state.
  - (b) a 3-D h in the core: more contract text, no gain.

**D4 — PLE lives in the HC block of layer 1 and owns per-request recurrent state; blocks may declare state (C2).**
- PLE reads and writes R before layer 1's attention mix (HFm:1276-1301). It therefore sits in the block, not in the
  mixer (which sees x) or the embed (which runs before layer 0).
- **State per request** (HFm:1128-1145, 1193, 1213):
  - the conv history, 9 = (4 − 1)·3 rows × 10240 in the model dtype (184 KB);
  - the last two token ids;
  - kept as a `Slot` spec of kind `recurrent@1` with `draft_slots` (verify and rewind as GDN's conv state does,
    `std/holon_std/mixers/gdn/units.py:117-124`) and GDN's checkpoint alignment, so a prefix hit restores it with
    GDN's state.
- **Today:** only mixers declare state (`core/holon/core/assembly.py:139-161`).
- **Rejected:**
  - a GDN mixer wrapper that declares PLE's spec: it cannot see R;
  - recomputing the conv window from the previous 9 tokens' rows: those need R at earlier positions, so the window
    is not recomputable from tokens alone.

**D5 — The PLE table is a row table in X's host tiers: RAM by default, SSD when RAM is short.**
- The table is a new std unit, `std.tiered.rows@1`: a general large row table read a few rows per token.
- **RAM tier (default):**
  - The table (28.80 GB) is loaded once into a mapped host pool (S2; `mmap` + `cudaHostRegister`, exact bytes).
  - Loading reads shard 2 from its extents (S4) through `tiered/files.py` (O_DIRECT, 8 threads).
  - A GPU kernel hashes ids, gathers the 16 rows of 90 B per token zero-copy and dequantizes in-kernel: 1.4 KB per
    token, 5.9 MB per 4K chunk.
- **SSD tier (RAM short; Strata's default `--ple-io direct`, St `generate.cpp:399-403`):**
  - **Prefill:** a shared object's `before_launch` (S1) knows the chunk's ids on the host. Its `cpu_task` I/O workers
    `pread` 4 KiB pages with O_DIRECT, deduplicated and sorted, one chunk ahead, into a pinned staging pool
    (double-buffered as St `prefill.cpp:2160-2172` does).
  - **Decode:** the ids exist only on the device (overlap), so the gather kernel posts its rows and the graph waits,
    through X's in-graph `cpu_task@1` copy contract (`std/holon_std/moe/kernels.py:150-155`): 16 reads per token, 64
    per verify window of 4.
  - **Row cache:** none in rev 1. Strata keeps 1M rows (St `ple_reader.hpp:14-16`). Step 7 measures whether one
    pays.
- **Not reused:** `TieredStore` itself. Its per-piece Python list and `[P, X, 4]` descriptors are sized for
  thousands of experts, not 320M rows (`std/holon_std/tiered/store.py:65, 172, 348`).
- **Reused:** the S2 pools, `files.read` and the `cpu_task` copy contract. No new seam, as §4.7 expected.

**D6 — QSA is a new std mixer, `std.mixers.qsa@1` (`derived_from` `std.mixers.gqa`), with two paged specs in
separate groups.**
- **The specs:**
  - `"kv"` keeps gqa's layout and every `kv.format` (bf16, q8_0, fp8: `std/holon_std/mixers/gqa/kernels.py:58`).
  - `"idx"` holds, per page of `page_size` tokens: raw index keys (page × 128 bf16) and pooled keys
    (page / 4 × 128 fp32, as St `qsa.hpp:79-85`).
  - Per token per layer: 2,048 B (bf16 KV) + 256 + 128 B.
- **Why two specs:** both get their own `BlockTable` and planner group. Pages advance in lockstep, because both count
  the same tokens. The dense kernels keep reading `"kv"` unchanged.
- **Correction to `docs/v1-offload.md` §4.7:** the core refuses one page group shared by two paged specs
  (`core/holon/core/state.py:149-150`). FN does not need it. GLM's DSA decides for itself.
- **Pooling and selection:**
  - **Pooling:** a block's pooled key is written when its 4th token is appended. A block completed by rejected drafts
    is rewritten when its positions are appended again, before any query can see it complete. So verify and rewind
    need no extra state.
  - **Selection:** scores in fp32; an exact top-k over blocks, with ties going to the lower block (St
    `qsa_select.hpp:12-15`). Block granularity is exact because scores are constant within a block.
  - **The dense limit:** while a query's n ≤ 2,051, selection is the identity, so the mixer runs gqa's dense kernels.
    One kernel switch per sequence; each run is deterministic.
  - **Prefill:** every query row selects for itself, causally (HFm:720-760; St `prefill.cpp:2740-2935`).

**D7 — The shared expert composes beside `std.moe@1`, at block level (§4.7). The order is routed MoE first.**
- **New units:**
  - `std.ffn.sum@1(parts)` runs its FFN parts in order and adds their outputs.
  - `std.ffn.gated_swiglu@1` is SwiGLU × `sigmoid(x · w_gate)`.
- **Why MoE first:** `std.moe@1` consumes the router-prepared q8_1 inside its own call
  (`std/holon_std/moe/units.py:247-254, 273-289`). A shared expert after it cannot stale the handle (§4.7). An L0
  test on FN's layer asserts the experts never re-quantize (the handle's generation check,
  `kernels/ggml_cuda/plugin.py:434-450`).
- **Unchanged:** `std.moe@1` is untouched, so X's offload store applies through its `moe.store` option as before. The
  shared expert stays resident.

**D8 — GDN gets an output-gate fact; no new mixer.**
- `std.mixers.gdn@1` gains `out_gate: "silu" | "sigmoid"` (default `silu`, so every Qwen3.8 trace is unchanged).
- `gated_rmsnorm@1` gains the matching fact; `kernels.gdn_triton` serves both.
- The edit changes the unit's `src`, so Qwen3.8-27B's locks propagate once at the close (D16).

**D9 — MTP is an aux model: a new head, the existing drafter carrying R. Its hnorm form is decided by acceptance.**
- **New head:** `std.spec.mtp_hc_head@1` (`derived_from` `std.spec.mtp_head`):
  - per-stream `fc_e(enorm(e)) + fc_h(hnorm(R)_s)`;
  - one HC layer (QSA in the aux namespace, `std.ffn.sum@1` of MoE + shared expert);
  - the output mixer;
  - the target's `lm_head`.
- **Drafter:** `drafters/mtp` carries `spec.residual`'s width and dtype, not `norm(h)`, when its head asks for the
  residual. Today it carries `(max_seqs + 1, hidden)` of `norm(h)`
  (`plugins/holon_plugins/drafters/mtp/plugin.py:33-35, 52-63`).
- **Weights:** from a separate MTP GGUF (`ctx.weight_source(path)`):
  - matched to Strata for the bars: Q2_0 experts and Q8_0 projections, quantized from unsloth's BF16 MTP file by the
    pinned llama.cpp's `llama-quantize`, or SC117's `mtp-q2_0` if its types match;
  - Holon's own default: unsloth's `mtp-Q8_0`.
- **hnorm:** both forms are implemented as a head parameter. Step 5 runs both on the RTX PRO 6000 and keeps the one
  with higher acceptance; the other is dropped. The GGUF's `nextn.hnorm` is {2560, 4}, which suggests LC's per-stream
  form.
- **Experts:** through the offload store (D12's std minor), or resident when tune measures that faster.

**D10 — The new kernels are Triton, in new kernel plugins; Strata's CUDA is the performance target.**
- **The plugins:**
  - `kernels.qsa_triton`: index append + pool, block scores + top-k, sparse decode attention (split-K over the
    selected blocks + merge), sparse prefill attention;
  - `kernels.hc_triton`: `hc_mix@1` (norm + down + silu + up + sigmoid + mean + inject, with the previous half's
    combine folded in as St `fused_gr.hpp:1-17`), `hc_combine@1`, the PLE block (grouped norms, gate, dilated conv
    with state);
  - `kernels.rows_io`: the n-gram hash and the row gather from a mapped pool or a staging pool.
- **Where they come from:**
  - FreeToken's Triton kernels (FT `kernel/triton/{hc,ple}.py`, `kernel/triton/qsa/*.py`, Apache-2.0) are adapted
    with notices and `derived_from` records.
  - Strata's (MIT) are references for fusion and launch shape.
- **Rewrites:** a kernel moves to CUDA only if step 6's profile shows it above 5% of a step and a CUDA prototype gains
  ≥ 10%.
- **Why new plugins:** existing plugins keep their `src`.

**D11 — The comparison protocol** (acceptance 5).
- **Same everything:** same files and bytes, card, context, prompts (X's natural prompts, greedy, 256 output tokens)
  and host, interleaved rounds. Strata is pinned at `296b59d` and re-checked against its newest release at the close
  (`docs/v1-offload.md` §10).
- **Strata, no approximation:**
  - `STRATA_ROUTE_TAIL_SKIP=0 STRATA_IQ_MT_MIN=1` (§1.4's settings; St `expert_source.cpp:3663-3678`,
    `native_expert.cpp:238`);
  - plus `--kv fp16`, so both engines keep 16-bit KV. Strata's setup picks int8 above 8K (St `setup.py:6286-6294`).
    This refines §1.4; open question 3.
  - plus `--ple-io ram` against Holon's RAM tier, so the row is engine against engine, not disk against RAM.
  - Its speed features stay on: adaptive placement, the PCIe split, the prefill CPU share, tensor-core prompt kernels.
    They round differently but approximate nothing; Holon's counterpart is F5b where tune picks it.
- **Strata defaults:** its setup's command for the card (St `setup.py:6570-6576`):
  - `--expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5`, with the suffix drafter;
  - int8 KV above 8K;
  - PLE O_DIRECT.
  - Against it, Holon runs A1 + q8_0 KV above 8K + MTP, with its default PLE tier.
- **Drafting:**
  - **Off:** Strata without `--mtp` and with `--suffix-draft 0` (St `generate.cpp:7238-7243`); Holon without a
    drafter.
  - **Matched:** both MTP at 3 drafts, the same MTP weights' types (D9), Strata `--spec 4 --spec-min-p 0
    --suffix-draft 0`.
  - Both off and matched rows are gated (open question 4).
- **Measures:** Holon `bench/holon_tg.py` and `llama_bench.py`-style prefill; Strata `serve/server.py` with
  `tools/strata_bench.py`'s timings (St `tools/strata_bench.py:1-26, 131-136`). Evidence goes in
  `evidence/fn_strata_<gpu-class>_<date>.json`.

**D12 — Offload deployment: X's defaults, plus two follow-ups X left to FN.**
- **Defaults:** `lfu` (F11) with F5a; F7's prefill ring; a Flash-Next routing profile from `route_dump` on the RTX
  PRO 6000 (natural prompts, held-out split), shipped in the bundle.
- **Follow-up (a), the aux-model experts std minor** (plan §10):
  - Today `std.moe.offload@1` refuses an aux model's experts (`std/holon_std/moe/offload.py:63-65`).
  - FN adds the MTP layer as the store's next layer group. Its resolve calls take row kinds from the drafter's
    sub-steps: one decode row per sequence per sub-step, written by the drafter's `before_launch` with the target's.
  - Counts are zeroed at the target's group 0 only, and the store binds kernels after the aux model is built.
  - **Exactness:** the snapshot with MTP experts through the store is byte-identical to MTP all-resident.
- **Follow-up (b), tune's F5b benchmark** (plan §10): interleave the arms over several natural prompts (as
  `spikes/x6_cpu/decode.py:2-4`) before FN's locks use its ≥ 5% rule (`tools/holon_tools/tune.py:25, 70-104`).
- **F5b locks:** X escalated that no tool writes an F5b lock and that `check_locks.py` refuses tier-carrying locks.
  Until the lead decides, FN's F5b rows are benchmark deployments and its locks are F5a (open question 5).

**D13 — Qwen3.6-35B-A3B leaves FN's critical path** (proposed; plan §10 names it FN's first step; owner to confirm).
- FN's own oracle (D2) covers the shared expert, the sigmoid gates and GDN. FN's MTP covers aux-model experts.
- The 35B adds a model plugin variant (`qwen35moe`) and its own locks, about 3 days, without retiring any FN risk.
- It becomes a short follow-up on FN's units: `models.qwen3_5` with `std.ffn.sum@1`.

**D14 — Split GGUF files are one checkpoint.**
- `std.weights.gguf`'s hub and source read a `-0000i-of-0000n` set as one source: tensor tables merged, each
  tensor's extent naming its shard (S4 already carries a path per part).
- The refusal at `hub.py:44` goes.
- Both FN's files and unsloth's need this.

**D15 — Text-only.** The GGUFs carry no vision tensors; `image_token_id` positions hash as that id in PLE (LC
qwen4exp.cpp:1229-1234), which text never produces.

**D16 — Locks and propagation.**
- New locks:
  - `rtx-5080`: IQ2_XS, offload, MTP;
  - `rtx-5090`: IQ3_S, offload, MTP.
- No RTX PRO 6000 lock (a correctness card).
- FN edits existing `src` in four places: `std.mixers.gdn`, `drafters/mtp`, `std.weights.gguf`, and the
  `ggml_cuda` / `ggml_cpu` type tables. The existing locks therefore propagate once at the close: bitwise for the
  dense ones, and L2 where a decode-only kernel changed (plan §10's decode-run L2 item applies).

## 3. What Holon has today (relevant pieces)

| Piece | State |
|---|---|
| Model trunk | `embed → blocks → norm → lm_head`; h is `[T, H]` by contract (`core/holon/core/build.py:265-295`; `api/model.py:142-147`) |
| State | only mixers declare it (`assembly.py:139-161`); `paged@1`, `recurrent@1` with checkpoints and draft slots; a page group shared by two paged specs is refused (`state.py:149-150`) |
| Attention | `std.mixers.gqa@1` with the sigmoid output gate (`gqa/units.py:166-172`), KV formats bf16 / q8_0 / fp8, fused `qk_rope_kv` |
| GDN | `std.mixers.gdn@1` (Qwen3.8's 16/48 heads), silu-gated norm, chunk 64 checkpoints |
| MoE | `std.moe@1`: softmax top-k (`topk-moe.cu` instantiates up to 512 experts, `:341`), router-prepared q8_1; no shared expert anywhere (`docs/m02-moe.md:381-382`) |
| Offload | X's store: `lfu` / `static` / `profile_decay` / `lru`, F5a / F5b, F7, F9, page-cache tier, aux models refused; the resolve fits E = 512 in shared memory ((roles + 11) · 512 · 4 B + 36 B ≤ 48 KiB, `kernels/expert_io/plugin.py:25-26, 144-147`) |
| MTP | `std.spec.mtp_head@1` + `drafters/mtp` over the target's `norm(h)` |
| GGUF | single files; types without Q2_0; tokenizer pre `qwen35` |
| Core | 4,603 / 6,000 code lines (`scripts/ci/check_core_loc.py`), ABI 1.3 |

## 4. Core seams (additive ABI minors → ABI 1.4)

| # | Seam | Today → change | Toy proof | Lines |
|---|---|---|---|---|
| C1 | **Residual width** | Today h is `[T, hidden]` everywhere, and the profile hands drafters `zeros(T, hidden, cfg.dtype)`. → `ModelSpec.residual: Residual \| None = None`, where `Residual(width: int, dtype: str)` and None means `(hidden, model dtype)`. The trunk's h, taps and `Drafter.run`'s h are `[T, residual.width]` of `residual.dtype`. The executor's profile uses it. `TargetView.spec` already exposes the spec to drafters. The embed, block and norm units are unchanged in the core: they read n from their Ref params. | A toy model whose embed expands to `[T, 2H]` fp32 and whose norm collapses it. Trunk, `forward`, prompt logprobs, `_worst_step`, a tap and a toy drafter in the profile all see the stated shape, in graphs and eager. Plugins refuse widths they cannot take: dflash refuses `width ≠ hidden` (`dflash/plugin.py:136`). | ≈ 15 |
| C2 | **State declared by blocks and FFN units** | Today `_bind_layers` binds only `layer.mixer`'s `state_specs` / `planner`. → It binds every Ref a layer is built from: `ModelSpec.block` over all layers, each `LayerDecl.ffn` over its layers, and the mixers as today. The same hooks (`state_specs(params, layers, ctx)`, `planner`) apply, and "a group declared twice" stays an error. Aux models get the same through `AuxSpec`'s layer declarations. | `test_payloads.py` gets a toy block holding a `Slot` running sum at one listed layer. It covers cut, publish, prefix hit, adopt, and draft-slot rewind beside a toy mixer's state. A toy FFN with paged state is bound and sized. | ≈ 20 |

- **Not core:** HC units and kernels, QSA, PLE and its row table, the shared expert, MTP's head, split GGUF, Q2_0 /
  IQ1_M, the aux-model store minor, tune.
- **Generality:**
  - "hyper", "residual" and "row" are allowed by `check_core_denylist.py`; no model, mixer or format name enters.
  - C1 serves GLM's mHC as is (`docs/v1-offload.md` §4.7).
  - C2 serves any block-level stateful unit, for example a later Engram table (FT
    `models/deepseek_v41/engram_table.py`).
- **ABI:** both are additive: a field with a default, and optional hooks on existing protocols. Regenerate
  `core/abi.txt`, bump `ABI` to 1.4 (`core/holon/api/manifest.py:19`) and pass `check_abi --base`.

## 5. std units, kernel contracts, model plugin

**std units** (new unless marked):

| Unit | Role |
|---|---|
| `std.hyper.embed@1(inner, n)` | the inner embed's `[T, H]` repeated to `[T, n·H]` in the residual dtype |
| `std.hyper.block@1(n, rank, pre={layer: Ref})` | per layer: `pre` (PLE at layer 1) → `hc_mix` → mixer → `hc_combine` → `hc_mix` → ffn → `hc_combine`; the norm site param is ignored (no pre-norms); declares `pre`'s state (C2) |
| `std.hyper.head_norm@1(n, rank)` | the output mixer (mix without inject), as `ModelSpec.norm` |
| `std.ple.ngram@1(table, …)` | n-gram ids, the row gather (`std.tiered.rows@1`), K/V projections, grouped norms, gate, the dilated conv with its `recurrent@1` state, `R += G + silu(conv)` |
| `std.tiered.rows@1` | a large row table in a mapped host pool (RAM) or a file with O_DIRECT readers (SSD); `gather(row_ids) → rows` on the device |
| `std.mixers.qsa@1` | D6 |
| `std.ffn.sum@1`, `std.ffn.gated_swiglu@1` | D7 |
| `std.mixers.gdn@1` (edit) | `out_gate` (D8) |
| `std.spec.mtp_hc_head@1` | D9 |
| `std.weights.gguf` (edit) | split files (D14), the Q2_0 type entry (D1) |
| `std.moe.offload@1` (edit) | aux-model experts (D12) |

**Kernel contracts** (each with `reference.py` and a CPU provider in `testing/holon_testing/ref_kernels`):

| Contract | Signature (sketch) | Notes |
|---|---|---|
| `hc_mix@1` | `(R [T, nH], norm_w, W_down, W_up, W_inj?) → x [T, H], w [T, n]` | fp32 activations; bf16 or quantized weights through `qlinear@2` for the projections in prefill; one fused kernel for decode rows |
| `hc_combine@1` | `(R, y [T, H], w) → R` in place | may be folded into the next `hc_mix` (a fused variant with its own contract) |
| `ple_rows@1` | `(input_ids, prev-token state, positions, consts) → row_ids [T, 16]` | int64; the EOS cut; bitwise against the oracle |
| `rows_gather@1` | `(row_ids, table view \| staging, type) → E [T, 2560]` | IQ4_NL / Q4_0 / Q5_0 / Q8_0 / BF16 rows dequantized; bitwise |
| `ple_block@1` | `(R, E, K/V outputs, norms, conv_w, conv state) → R, conv state` | fp32; the state of draft slots as GDN's |
| `qsa_index@1` | `(raw k [T,128], positions, idx pages) → pooled keys written` | pools blocks completed in the step; the norm and RoPE at the block's first position |
| `qsa_select@1` | `(q_idx [T,4,128], idx pages, lengths) → block ids [T, ≤512]` | fp32 scores; exact top-k, ties to the lower block; device lengths for verify and sub-steps |
| `qsa_attend@1` (prefill / decode / verify) | `(q, kv pages, block ids, tail) → o` | split-K over selected blocks + merge for decode; KV formats as gqa's |
| `linear_experts@2` IQ1_M path | as @2 | D1: dequantize the called slots to bf16, then a grouped GEMM above the mmvq limit |

**Model plugin `models.qwen4_exp`** (`plugins/holon_plugins/models/qwen4_exp/`, `derived_from = "models.qwen3_5@b5b1e7f"`):
- **Recipe:** `ModelSpec(hidden=2560, residual=Residual(10240, "float32"), block=Ref("std.hyper.block@1", n=4,
  rank=320, pre={1: Ref("std.ple.ngram@1", …)}), embed=Ref("std.hyper.embed@1", …), norm=Ref("std.hyper.head_norm@1",
  …), …)`.
- **Layers:** `qsa` at `full_attention_interval`, else `gdn(out_gate="sigmoid")`; every ffn is
  `std.ffn.sum@1(moe, gated_swiglu)`.
- **GGUF rules:**
  - LC's names: `hc_{attn,ffn}_{norm,down,up,inject}`, `output_hc_*`, `indexer.*`, `ple_*`, `per_layer_token_embd`,
    `ffn_*_shexp`, `ffn_gate_inp_shexp`;
  - norms stored as w + 1, `ssm_a` = −exp(A_log), V heads tiled: FT `gguf.py:1-4, 154-159`; LC
    `conversion/qwen4exp.py:153-169`;
  - hparams from `qwen4exp.*` (`hyper_connection.*`, `attention.indexer.*`, `ple.*`).
- **HF rules:** for the tiny model and the BF16 checkpoint.
- **`draft_head("mtp")`:** `AuxSpec(Ref("std.spec.mtp_hc_head@1"), …)` over the MTP file.

## 6. Numerics and judge (thresholds and predicted verdicts fixed now)

### 6.1 CPU: tiny model vs HF

**Config:** 8 layers (QSA at 3 and 7, PLE at 1), H = 64, n = 4, rank 16, vocab 512 with EOS 511.

| Group | Values |
|---|---|
| Attention | 4 / 2 heads × 64, rotary 16 |
| Indexer | 2 heads × 32, ratio 4, **budget 16** (selection bites past 19 tokens) |
| GDN | 2 / 6 heads × 32 |
| MoE | 8 experts top-2 + shared |
| PLE | 4 heads × 16, `ngram_vocab_size_base` 1000, one split |
| MTP | 1 layer |

Constraints: HFx:196-265, LC qwen4exp.cpp:76-123.

**Sequence:** 100-token prompts with an EOS mid-sequence, chunked at 7 and 13 (straddling 4-blocks and the PLE
window), then 24 decode steps, with random fp32 weights.

| Check | Threshold | Predicted |
|---|---|---|
| logits vs the HF fp32 fixture | max \|Δ\| ≤ 1e-4, top-1 identical | pass |
| selected token sets per QSA (layer, position) | identical | pass (fp32 both, random weights: no near-ties) |
| PLE row ids | bitwise | pass |
| prefix hit vs recompute, chunked vs one-shot | bitwise (CPU, same path) | pass |
| MTP head vs LC's formula (both hnorm forms) | max \|Δ\| ≤ 1e-4 against a small formula reference in the test | pass |

### 6.2 Full model, L2 and bounded

**Setup:**
- Goldens: fp32, layer-streamed (D2), for ISTA IQ2_XS and IQ3_S on `v1-qwen3_5.jsonl` + `v1-long.jsonl`.
- Floors: bf16 HF with `--q8-1 mmq|mmvq` per path. Path-aware: IQ1_M prefill rows unemulated (D1); F5b's CPU set
  under Q8_K blocking, as X's.
- Candidate: Holon Tier B on the RTX PRO 6000, all-resident.
- Pass rules: L2's (`docs/design.md:1395`).

| Check | Predicted | Why |
|---|---|---|
| L2, IQ2_XS and IQ3_S, both sets | pass; KL p99 ≤ 1.3× floor | fp32 R and HC beat the floor's bf16 residual. QSA selection flips between golden and candidate (bf16 inputs) are < 1% of selected blocks on the long set, reported. |
| bounded: chunked, prefix hit, graphs vs eager, overlap 1 vs 0 | pass | — |
| snapshot ×3 | run-identical | — |
| MTP verify path L2 | pass | — |
| MTP acceptance | ≥ 2.2 per pass at 3 drafts | the better hnorm form; the other predicted ≤ 1.6 |
| offload (capped, `lfu`, MTP experts in the store) vs all-resident | byte-identical | — |
| PLE RAM vs SSD tier | byte-identical | — |

### 6.3 Plants

Each plant changes only its own path; the bounded baseline stays clean.

| Plant | Breaks | Predicted |
|---|---|---|
| `qsa_select_shift` | selected blocks shifted by one past the budget | L2 **fails on `v1-long`**, passes on `v1-qwen3_5` (shows why the long set exists) |
| `qsa_pool_rope_last` | pooled key rotated at 4b + 3 instead of 4b | L2 fails on `v1-long` (KL p99). If it passes, the long set lacks recall prompts and grows before any lock. |
| `qsa_dense_limit_off_by_one` | dense up to 2,052 | L2 passes (one position); caught by the tiny CPU test (selected sets differ at 20) — recorded as the CPU test's plant |
| `hc_inject_unscaled` | inject without the `/n` | L2 fails everywhere |
| `ple_conv_dilation_1` | conv taps t−3..t | L2 fails (KL p99) |
| `ple_state_not_carried` | the conv window reads zeros at a chunk start | bounded chunked-vs-one-shot fails; one-shot L2 clean |
| `shexp_gate_off` | shared expert added ungated | L2 fails |
| `mtp_stream0` | MTP reads stream 0 for every stream | acceptance drops ≥ 0.4 tokens per pass; outputs unchanged |
| `aux_rowkind_target` | the MTP layer's resolve uses the target's row kinds | snapshot with MTP experts through the store differs from all-resident |
| `rows_off_by_one` | the SSD tier reads the neighbouring row | RAM-vs-SSD identity fails; L2 fails |

The EOS cut is checked by the tiny CPU test only: EOS 248044 is rare inside prompts.

## 7. Memory fit (estimates; step 0 measures)

Budget = VRAM after the CUDA context × `memory_fraction` (≈ 1.4 GB idle on the RTX 5080 at 0.9), less the worst
step's peak.

**Per-token and per-request costs:**
- **KV per token:** 12 QSA layers × 2,432 B (bf16 KV + index keys), plus the MTP layer's, ≈ 31.6 KB. That is ≈ 1.04
  GB at 32K; with q8_0 KV, ≈ 19 KB (0.63 GB).
- **GDN state:** 36 × ≈ 3.1 MiB per slot. Slots = (`max_seqs` + 1) × (1 + drafts) = 8 at `max_seqs` 1 with 3 drafts
  ≈ 0.92 GB, plus 2 checkpoints ≈ 0.23 GB.
- **PLE state:** 184 KB per slot.

| Card, file, context | Non-expert | MTP | KV | GDN + ckpt | Activations | Expert cache | Host pinned |
|---|---|---|---|---|---|---|---|
| RTX 5080 16 GB, IQ2_XS, 32K, MTP | 3.76 GB | 0.8 GB resident (or in the store) | 1.04 | 1.15 | ≈ 0.8 (2K chunks) | ≈ 7.6 GB ≈ 21% of 35.45 (≈ 8.4 GB ≈ 24% with MTP experts in the store) | experts 35.5 + PLE 28.8 ≈ 64.3 GB (35.5 with PLE on SSD) |
| RTX 5090 32 GB, IQ3_S, 32K, MTP | 4.51 GB | 0.8 | 1.04 | 1.15 | ≈ 1.0 | ≈ 23.9 GB ≈ 47% of 50.29 | 50.3 + 28.8 ≈ 79.1 GB |
| RTX PRO 6000 96 GB, IQ2_XS / IQ3_S, all-resident | 3.76 / 4.51 | 2.8 (Q8_0) | ≤ 4.2 at 128K | 1.15 | ≈ 1.5 | all: 35.45 / 50.29 | PLE 28.8 |

- **Routed bytes per token:** 48 × 10 × 1.44 MB = 691 MB (IQ2_XS); 984 MB (IQ3_S).
- **Decode traffic at the 5080's fit:** under skewed routing `lfu` hits above the cached fraction. If hits are ≈
  50% (est.; N4 replays Flash-Next traces), misses move ≈ 0.35 GB per token over PCIe. That is the decode
  roofline's main term, hence D12's F5b question.
- **Host RAM per desktop class** is checked before scheduling (not recorded here). The 5090's 79 GB pinned is the
  largest ask; a class short of it runs the PLE SSD tier.

## 8. Step 0 — measure before freezing

On the RTX 5080 and 5090 desktops (the bar cards), one heavy job at a time under `MemoryMax`. Results go in `evidence/`.

| # | Measure | Decides |
|---|---|---|
| N1 | Strata at `296b59d`, defaults and no-approx, drafting off and matched, 1K / 4K / 32K, both cards. llama.cpp (`qwen4exp`, at its pinned commit) and FreeToken where they run. | the bars' denominators; whether open question 3's KV refinement matters |
| N2 | The zero-copy row gather: 16 random 90 B rows per token from a 28.8 GB mapped pool, at 1 – 4,096 tokens. O_DIRECT 4 KiB random reads at queue depth 16 – 256 (X's M7 covers expert-sized reads only). | D5's SSD prefetch depth; whether the RAM tier's gather is under 1% of a decode step |
| N3 | The oracle: HF 5.19.0 tiny model in the golden venv; the streamed driver on 2 layers of the real file | D2's feasibility and cost |
| N4 | After step 5 (needs the model): routing traces on the RTX PRO 6000; X's M6 replay at 20–50% cached; per-layer prefill expert time vs chunk | the roofline for every acceptance-5 row, written before step 8 measures them; R2's trigger |

## 9. Steps

| Step | Work | Checks | Days |
|---|---|---|---|
| 0 | N1–N3; host RAM per class | evidence; the panel reviews the parameters they set | 2 |
| 1 | C1 and C2 with toy proofs (ABI 1.4) | core tests; `check_abi --base`; LOC; denylist | 2 |
| 2 | The oracle: golden venv, `golden --stream-layers`, the qwen4exp GGUF → HF name map in `judge/holon_judge/gguf.py`, `v1-long.jsonl` (+ sha256), the tiny-model tool and fixtures | the driver reproduces HF's whole-model forward bitwise on the tiny model; 2 real layers stream | 3 |
| 3 | std units with reference kernels and CPU providers (§5), split GGUF, the Q2_0 type entry, `models.qwen4_exp`, GDN `out_gate` | §6.1 on the CPU; Qwen3.8's snapshot unchanged with `out_gate` default | 4 |
| 4 | CUDA / Triton kernels (D10), Q2_0 and IQ1_M in `ggml_cuda` / `ggml_cpu`, the IQ1_M prefill path, the q8_1 handle L0 (D7) | L0 on sm_86 / 89 / 120 + memcheck | 5 |
| 5 | Full model all-resident on the RTX PRO 6000: goldens and floors, L2, bounded, snapshot; MTP head and drafter carrying R, the hnorm A/B; the routing traces (N4) | acceptance 1b, 2; plants 1–7 | 4 |
| 6 | Offload: the aux-model experts minor, the PLE row table's SSD tier with prefill prefetch, the Flash-Next profile, offload exactness on the RTX PRO 6000, tune's interleaved F5b benchmark; roofline (N4) | acceptance 4; plants 8–9; profiles; tune's F5b on a known case reproduces X's interleaved −6% | 4 |
| 7 | Locks on the RTX 5080 (IQ2_XS) and RTX 5090 (IQ3_S): `tune --new-model`, verdicts, approval | L2, snapshot, memory fit vs §7 | 2 |
| 8 | The comparison (D11): acceptance 5 rows, FreeToken and llama.cpp rows | evidence; every row vs the roofline | 3 |
| 9 | Close: full suite, panel code review, one propagation of the existing locks, plan §10 status | acceptance 6–8 | 2 |

**Order:** 2 and 3 run in parallel after 1; 4 can start with QSA and HC as soon as 3's references exist. FN changes
the core in step 1 only, so another track (K) rebases on it once. Total ≈ 31 days.

## 10. Risks

- **R1: decode at the 5080's fit is miss-bound.** With ≈ 21–24% of experts cached, decode depends on `lfu`'s hit
  rate and PCIe.
  - Strata uses a CPU-heavy split there (R §3.1); Holon's answer is F5b. The F5b-lock question (D12) can block the
    lock, though not the measured row.
  - The roofline (N4) predicts the row before step 8 measures it. A row predicted below 0.9× goes to the owner
    before step 7.
- **R2: prefill vs Strata.**
  - X's prefill already trails FreeToken by 0.88–0.93× on 30B-A3B, from the expert GEMMs (plan §10's MoE
    prefill-speed item).
  - Strata's 5080 prefill is ≈ 2.3K tok/s at 4K (St's 2026-10-05 community RTX 5080 report under `bench/results/`, README:131-136; IQ3_S, not our file).
  - If N4's roofline predicts < 0.9×, pulling that item into FN is a scope change to escalate. D1's dequant-GEMM
    path is its seed.
- **R3: the oracle's cost and version.**
  - The streamed golden is new judge code, and HF's per-query indexer loop is slow.
  - transformers 5.19.0 vs the workspace's 5.6.2 splits the judge's environments. Mitigation: the golden venv is used
    only for recording, and its version sits in every golden's metadata.
- **R4: an LC vs HF disagreement in GGUF conventions** (for example the dropped FP8 `weight_scale` of the PLE table:
  LC `conversion/base.py:442-457`, FT `ple_disk.py:111`).
  - Step 2's driver dequantizes the GGUF, so golden and candidate share the file's table whatever its scale. The
    file-vs-checkpoint error is reported, not gated.
  - The ISTA table is IQ4_NL at 7.6% per-row error vs BF16 (St `docs/DETAILS.md:1914-1919`), which sets
    Flash-Next's quality on these files, not Holon's numerics.
- **R5: QSA near-ties.** fp32-vs-bf16 score differences can flip a selected block, a bounded effect. The long set's
  flip rate is reported; KL p99 is the gate.
- **R6: Strata moves fast** (`docs/v1-offload.md` §10). The pin is `296b59d`; the close re-checks the newest
  release.
- **R7: pinned host memory.** 64–79 GB pinned. The SSD tier for PLE is the fallback; S2 accounts every byte.

**Deferred:**
- KV streaming for QSA (K4);
- a PLE row cache (step 6 decides);
- CUDA rewrites past D10's rule;
- Qwen3.6-35B-A3B (D13);
- Strata's suffix drafter;
- `--ple-io mmap` parity.

## 11. Open questions for the owner

1. **D13:** take Qwen3.6-35B-A3B off FN's critical path (plan §10 names it the first step)?
2. **D2:** a golden-only venv with transformers 5.19.0, rather than bumping the workspace (relock + a propagation of every lock)?
3. **D11:** Strata's no-approximation settings add `--kv fp16` (both engines on 16-bit KV) and `--ple-io ram` beside
   §1.4's two variables.
4. **D11:** gate both the drafting-off rows and the MTP-matched rows, or one of them?
5. **D12:** X's F5b-lock escalation. If tune picks F5b on the 5080 at ≈ 22% fit, FN's lock needs that decision.
6. **D1:** keep IQ2_XS as the 5080's bar file? It needs the IQ1_M prefill path; IQ3_XXS or Q2_0 avoid it.
7. **R2:** if the roofline predicts prefill < 0.9× Strata, pull plan §10's MoE prefill-speed item into FN?
8. **Acceptance 5:** FreeToken and llama.cpp rows reported only, or gated as in X (≥ 1.0× FreeToken)?

## 12. Review log

- **Rev 1** (2026-10-18): drafted from cited source studies of Strata, FreeToken, llama.cpp, HF 5.19.0 and sglang,
  of Holon at `b5b1e7f`, and of nine parsed GGUF headers. Three things were found on the way:
  - Q2_0 is upstream (D1);
  - shared page groups are refused by the core (D6, a correction to `docs/v1-offload.md` §4.7);
  - split GGUFs are refused (D14).

  No panel round yet.
