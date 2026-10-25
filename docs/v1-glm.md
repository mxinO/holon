# v1 — GLM: GLM-5.3-Flash on Holon, then its expert offload compared with Project Maya

Status: **rev 1 (2026-10-25), for the challenge panel.** Nothing is built. Parameters marked "est." freeze when
step 0's evidence lands (§8).

**Scope.** Plan §10 ("Expert caching/offload", milestone GLM) and `docs/v1-offload.md`:
- D2: the target (45 layers: 11 MLA + DSA indexer, 34 KDA; 288 experts top-8 + 1 shared; mHC; sigmoid `noaux_tc`
  routing with a SwiGLU clamp of 10; MTP);
- §1 acceptance 4: the program bar against Project Maya on the RTX 5090 with Maya-S;
- plan §10's std minors that land at GLM's start: **F10 with RAM short** and **lending and the prefill ring under
  exclusive tiers**; K4 (KV streaming, "GLM's DSA adopts it") is examined and deferred (D17).

**Builds on** `main` at `145a80df`: X (the offload store, ABI 1.3), K (state tiers, ABI 1.4), FN (ABI 1.5: C1
`ModelSpec.residual`, C2 block-declared state, C3 `unit_class`, `ActivationLender`; `std.hyper`, `std.mixers.qsa`,
`std.tiered.rows`, MTP experts through the store, `fetch_ctas`, the high-priority split fetch, THP pools,
`plugins/tests/test_reserve.py`), the MoE prefill track (`pass_rows`, tail lending, fused SwiGLU in mmq's store).

**References** (local clones; this doc cites `file:line` from them):

| Abbrev. | Reference | Version |
|---|---|---|
| **My** | Project Maya, `local/refs/project-maya-v1.1.27` | v1.1.27 `4615f34` (2026-10-24), MIT |
| **HFm** | HF transformers 5.19.0, `models/glm5_next/modeling_glm5_next.py` | — |
| **LC** | llama.cpp, `src/models/glm5-next.cpp`, `conversion/glm.py` | `27c7a76` |
| **VL** | vLLM's GLM-5.3-Flash model, `vllm/models/glm5next/` (`git show 6dd813e0:…`) | `6dd813e0`, the FLA pin of `kernels.fla` |
| **FT** | FreeToken, `python/freetoken/models/glm5_next/` | `21d9578`, Apache-2.0 |
| **CFG** | the checkpoint's `config.json`, `tokenizer.json`, `chat_template.jinja` | `zai-org/GLM-5.3-Flash@46a050be` |

"G-A / G-B / G-C §n" is the local research note `docs/brainstorm/glm-maya-study.md` (not published): Part A the
model, Part B Project Maya v1.1.27, Part C what Holon has; its verification log marks what was confirmed from source.
It supersedes `docs/brainstorm/v1-offload-research.md` §3.1b (Maya v1.1.15) where they differ.

## 0. The model

From CFG and HFm, cross-checked against LC, VL and Maya (G-A §0–§8).

| Part | Fact | Source |
|---|---|---|
| Trunk | 45 layers, H = 4096, vocab 154,880, untied head; plain-`w` RMSNorm (eps 1e-5) before each mixer and FFN; final norm | HFm:66-80, 1294-1295, 1510 |
| Layer mix | KDA at i mod 4 ≠ 3 (34), DSA at 3, 7, …, 43 (11); dense FFN at 0–2 (I = 12,288), MoE at 3–44 | CFG; HFm:1290-1292 |
| mHC | 4 streams of H, initialised as the embedding repeated. Per sub-block: `r = rms(flat X)` (unweighted, 16,384 values, eps 1e-5); `m = fn·r` (24 values, fp32); `pre = σ(s0·m[0:4] + b) + 1e-6`; `post = 2σ(s1·m[4:8] + b)`; `C = softmax_dst(s2·m[8:24] + b) + 1e-6`, then 1 + 19 alternating normalizations (src, then dst / src), eps 1e-6; block input `Σ_n pre_n X_n`; write-back `X'_d = post_d · y + Σ_s C[s][d] · X_s`. Collapse: the plain mean of the streams, then the final norm | HFm:286-337, 1306-1349; LC:384-541 |
| KDA | 64 heads × 128; q/k/v projections, each a causal depthwise conv (4 taps, fp32 weights) then SiLU; gate `g = −5 · σ(exp(A_log) · (f_b f_a x + dt_bias))` per channel; `β = σ(b_proj x)`; l2-normed q, k (eps 1e-6), q / √128; delta rule `S ← diag(e^g) S; S += k β(v − Sᵀk)ᵀ; o = Sᵀq` in fp32; output `rms(o) · σ(g_b g_a x)` then `o_proj` | HFm:350-364, 430-514, 665-770; LC:809-887 |
| DSA attention | NoPE MLA, 64 heads: `q = q_b(rms(q_a x))` [64, 256]; latent `c = rms(kv_a x)` [512] is all the cache holds; scores `q_k · c`, scale 1/16; absorbed: `q̃ = W_kᵀ q` (512), `ctx = Σ p c`, `o = W_v ctx` | HFm:1148-1235; LC:1034-1095 |
| DSA indexer | 32 heads × 128 from `q_r`; key `LayerNorm(wk x)` (with bias, eps 1e-6) and gate `γ = W_g x` cached per token; a pool of 4 tokens has key `Σ softmax₄(γ + ape)_c · k_c` per channel; score `Σ_h w_h(x) · relu(q_h · K̄)/√128/√32` with signed `w = weights_proj(x)`; top 512 complete pools ∪ the open pool's 0–3 tokens; **exactly dense while t + 1 ≤ 2,051** (FN's QSA rule); no RoPE anywhere; every layer `full` | HFm:773-1061; G-A §3.2 |
| MoE | router fp32 sigmoid; `p + bias` selects the top 8 (n_group 1); weights from the unbiased p, `/(Σ + 1e-20)`, × 2.5; experts `silu(min(g, 10)) · clamp(u, ±10)`, I = 2,048; one ungated shared expert (same clamp); the dense FFNs take the clamp too | HFm:110-206; LC llama-graph.cpp:2295-2397 |
| MTP | 1 NextN layer (`blk.45`): `x = eh_proj[enorm(E[t+1]) ‖ hnorm(h_t)]`, h the trunk's **post-final-norm** hidden; a plain pre-norm DSA (own latent and indexer cache) + MoE (288 own experts) + shared layer; `shared_head_norm`, the trunk's head; no streams. `index_share_for_mtp_iteration`: draft step 0 selects, later steps reuse | LC:548-675, 789-796; VL `common/mtp.py:95-110, 147-149`; VL `common/model.py:801-802` |
| Per token / sequence | latent 1,024 B × 11 = **11 KiB per token** (12 with MTP); KDA state 34 × (4 MiB + 288 KiB) = **145.6 MiB per sequence** | G-A §7 |

**Not in HF:** MTP (HFm:1379 drops `layers.45`). **Ties:** pools whose heads all ReLU to 0 score exactly 0.0, and
HF's own prefill and decode break them differently (a 0.073 logit jump on a tiny fp32 model, G-A §3.2). Maya breaks
ties to the lower pool (My `ref/glm.py:192-196`).

### 0.1 Files

| File | Size | Expert types | Per expert | Rest |
|---|---|---|---|---|
| **Maya-S v2** (`peasantsmith/GLM-5.3-Flash-Maya-GGUF`, `Maya-S-v2-IQ2_XXS/`, 3 shards) | **96.47 GB** | gate / up IQ2_XXS; down IQ2_S (layers 7–40), IQ3_XXS (3–6, 41–44); MTP Q2_K / Q3_K | 7.01 / 7.54 MB; 9.11 MB (MTP) | routed 86.03 GB; MTP experts 2.62 GB; non-expert 7.65 GB (Q6_K trunk, Q8_0 KDA gates and indexer, F32 router / mHC / norms) |

- Arch `glm5-next` (llama.cpp knows it, `llama-arch.cpp:161`); every type is upstream ggml and already in
  `kernels.ggml_cuda`'s and `kernels.ggml_cpu`'s `TYPES` (G-C §5); the NextN block is in the file; the
  pre-tokenizer is `glm4` (G-A verification log). `docs/v1-offload.md`'s "≈ 93 GB" is 96.47 GB (89.8 GiB).
- Split GGUFs are read as one checkpoint (FN D14).
- **Maya's pack rounds the router, its bias, the mHC weights, the indexer and KDA's small tensors to BF16**
  (`tools/iq_pack.py:62-82, 128-136`, G-B §1); Holon reads the file's F32. The two engines therefore differ in their
  arithmetic even on the same bytes; the comparison is speed at matched settings plus task accuracy (§6), never
  token identity.

## 1. Goal and acceptance

**Goal.** Serve GLM-5.3-Flash from Maya's own GGUF, text-only, exact against an independent oracle on a tiny model,
through X's offload store on a 32 GB card (and a 16 GB one), with exact defaults and reproducible placement; then meet
the program bars against Project Maya on the RTX 5090, with RAM holding every expert and on a 64 GB host.

1. **Model:** `models.glm5_next`, serving `arch.gguf.glm5-next@1` (text only).
   - **(a) Tiny model on the CPU** (§6.1): a random reduced config written by LC's converter matches HF 5.19.0 in
     fp32: logits within 1e-4 and top-1 identical, one-shot and chunked; identical DSA token sets per (layer,
     position) under the lower-pool tie rule (the test asserts that ties occur); a prefix hit bitwise equal to
     recomputing; the MTP head within 1e-4 of a reference transcribed from LC's `graph_mtp`. The GGUF tokenizer gives
     llama.cpp's ids on a multilingual round-trip corpus.
   - **(b) Full model, by the owner's method** (FN §18; no fp32 goldens, no big-model CPU runs):
     - **(c) GPU self-consistency:** decode (1-token steps) vs prefill and chunked (250) vs one-shot on the 4
       `v1-long` prompts retokenized for GLM: KL mean ≤ 0.13 and KL p50 ≤ 0.007 (FN's bars); MTP vs plain (if MTP is
       built, D7): KL mean ≤ 0.02, p99 ≤ 0.3, ≥ 100 agreeing positions over 8 prompts (FN §22.8). Two
       pre-registered must-fail plants (§6.3).
     - **(d) Task accuracy:** FN's 72-item synthetic set (`spikes/fn7/tasks`), the recall filler rebuilt to GLM's
       token lengths; Holon and Maya greedy on the same file and card, the same rendered ids. Bar, stated now:
       Holon within 5 items of Maya's score, and no Holon answer degenerate where Maya's is not (pooled rates, the
       owner's ruling of 2026-10-24).
     - **(e) A sample dump** of 8 answers for a human look.
     - **(f) The cover** (FN §22.6's lesson, built from the start): on the real model's own inputs, DSA's selection
       matches its oracle on 100% of rows, one KDA layer's output and state and one mHC pre/post pair are within their
       L0 budgets.
2. **Exact and run-identical:**
   - **placement:** snapshot traces byte-identical across fits (all-resident `std.moe@1` on the RTX PRO 6000, the
     deployment's cache, a capped cache), across policies (`lfu`, deferred admission, F10 spares), with the ring and
     lending on and off, and **RAM holding every expert vs RAM short (exclusive tiers + SSD)**;
   - run-identical ×3 on the RTX 5090 and the RTX 5080 at their locks, MTP on if a lock takes it.
3. **Kernels:** L0 for every new kernel against its `reference.py` on sm_86, sm_89 and sm_120, memcheck clean. At
   most **720 launches per decode token** (45 layers × 16, all graphed). `plugins/tests/test_reserve.py` gains a
   tiny-GLM case: every step shape fits the profiled reserve.
4. **Program bars** (`docs/v1-offload.md` §1.4; protocol D11), RTX 5090, Maya-S, drafting as each engine deploys it
   (Maya: none on one NVIDIA GPU, G-B §5; Holon: its lock, D7), 1K / 4K / 32K:

   | Host mode | Against | Decode | Prefill |
   |---|---|---|---|
   | **(i) RAM holds every expert** | Maya's defaults (no approximation by default, G-B §8) | ≥ 1.0× | ≥ 0.9× |
   | **(ii) 64 GB host** (both engines under one 64 GiB memory cap, the same pinned expert tier, the GGUF on the node's NVMe, D10) | same | ≥ 1.0× | ≥ 0.9× |

   - 12 gated cells. Plan §9's kill line (< 0.8×) applies to each.
   - **Reported, not gated:** Maya's deterministic recipe (G-B §8); Holon with MTP if no lock takes it; the RTX 5080
     (16 GB) with Maya-S, both engines, host mode (i); Maya's 32K arm with `MAYA_GLM_CKPT_EVERY=0` beside its default
     (D11).
5. **Memory** (§7): the 5090 fits at 33,792 tokens in both host modes; the 5080 at 33,792 tokens in mode (i). Every
   pinned host byte is accounted against `host_memory`; in mode (ii) each engine's whole process stays under the cap.
   The measured fit is reported against §7.
6. **Core:** **no core change is planned (§4).** Any core line needs the lead's sign-off and a toy proof; a hard cap of
   +15 code lines over GLM's base, counted by `scripts/ci/check_core_loc.py` at every step close. ABI stays 1.5; std
   and plugin contracts change only additively.
7. **Locks:** `rtx-5090` (mode (i)) and `rtx-5080`, written by `tune --new-model`; mode (ii) runs the 5090 lock with
   a host override (D15). Every other lock is propagated once, at GLM's close.
8. **Tests:** the suite grows only by GLM's own tests (one shared tiny-GLM engine per module); `--durations` watched.

**Non-goals:** the vision tower; multi-GPU; DFlash2 (CC BY-NC-ND); K4 (D17); Maya-S24 / M / L and Unsloth's files;
NVFP4 / EXL3; Maya's approximate opt-ins (AG1–AG6: Holon's A1 `approx` exists and stays off); the CPU split as a
default (D8f).

## 2. Decisions

**D1 — One file on every card: Maya-S v2.** It is the bar's file (`docs/v1-offload.md` §1.4) and Maya's choice for
the 5090 (`maya.py:1280`); both engines read the same bytes. The 16 GB row uses it too (Maya's setup would pick
Maya-S24 there; Q4 asks). The download is pinned to the repository's revision and its shards' sha256 are checked
against `maya.py:124-160`.

**D2 — The oracle: HF transformers 5.19.0 `glm5_next` on the tiny model only, in a golden-only venv.**
- `local/envs/golden-glm`: transformers 5.19.0, CPU torch, **no `flash-linear-attention`** (HF swaps in FLA's
  `chunk_kda` / `fused_recurrent_kda` when installed, HFm:463, 517, which work in exp2: G-A §2.3); `-e judge -e std
  --no-deps`, `--link-mode copy`. The workspace keeps its transformers pin. (FN's `golden-qwen4` was removed at FN's
  close.)
- fp32 on the CPU: HF's bf16 rounding points (bf16 streams, β's sigmoid, the bf16 MoE combine, G-A §9) are no-ops
  there, so Holon's fp32-residual form (LC's and Maya's) matches.
- **Documented deviations from HF**, recorded in every fixture's metadata: the indexer's `topk` (HFm:889) and the
  router's `topk(sorted=False)` (HFm:177) are wrapped, per module instance, with a stable descending sort (lower
  pool / lower expert wins), as FN's `wrap_indexers`.
- **Random init hides conventions** (HFm:1381-1421): the fixture randomises `A_log`, `dt_bias`, the selection bias,
  hc `base` / `scale`, the pooling gate and APE, every norm weight and LayerNorm bias, and re-randomises the pad row
  (pad = 154820 = EOS). It asserts each is non-default.
- **MTP oracle:** no HF code exists. A reference transcribed from LC's `graph_mtp` (`glm5-next.cpp:548-675`: e then
  h, h post-final-norm, no position-0 zeroing) runs on the fixture's random `blk.45` tensors. VL's position-0 zeroing
  (`fused_eh_norm.py:32`) touches only the head's position-0 entry; Holon follows LC and records it.
- **Tiny fixture tool:** `testing/holon_testing/glm5next_files.py` (FN's `qwen4exp_files.py` pattern): the random
  HF model, LC's converter at `27c7a76` with the real `tokenizer.json` (G-A §10: it accepts HF's saved layout),
  pinned sha256; the GGUF is rebuilt by the tool, not committed; CI skips only the GGUF-dependent checks when LC's
  converter is absent.

**D3 — mHC: siblings in `std.hyper`, fp32 throughout.**
- The residual is `Residual(16384, "float32")` through C1, exactly as FN's (G-C §1); 64 KiB per token.
- Units: `std.hyper.mhc_block@1(n, mixer, ffn, norms)` — `pre` → the block's RMSNorm → mixer → `post`; `pre` →
  RMSNorm → FFN → `post` — and `std.hyper.mean@1(inner)`, the collapse (the stream mean, then `inner`, the model's
  RMSNorm), used as `ModelSpec.norm`. `std.hyper.expand@1` is reused unchanged.
- Contracts: **`hc_mhc_pre@1`** `(R [T, n·H] f32, fn [24, n·H] f32, base [24], scale [3]) → (x [T, H] of dtype,
  post [T, n] f32, comb [T, n, n] f32)`: the unweighted RMS, the 24-wide product summed in a fixed order, the affine
  maps, the Sinkhorn of HFm:316-325 exactly (1 softmax, 20 + 19 normalizations), the weighted collapse rounded once;
  **`hc_mhc_post@1`** `(R, y [T, H], post, comb) → R'` with `R'_d = post_d · y + Σ_s comb[s, d] · R_s` in f32.
  A provider may fuse a sub-block's post with the next one's pre (FN's carried combine, `std/holon_std/hyper/units.py:113`).
- Precision: the 24-wide product `ieee` by default; `bf16x3` / `tf32x3` only within L0's fp32 budget and recorded as
  a kernel fact (FN D10's rule).
- Why not HF's bf16 write-back: Holon's residual is f32 like LC's and Maya's; HF in fp32 (the tiny oracle) is the
  same math.

**D4 — KDA: `std.mixers.kda@1` (`derived_from` gdn), kernels vendored from the FLA pin we already carry.**
- State: GDN's two `Slot` specs (G-C §4) — the recurrent `[64, 128, 128]` f32 and the conv windows of q, k, v
  (3 × 8,192 channels × 3) — `alignment = in_step = 64`, replay rewind, checkpoints through `std.state.recurrent`.
- Op `kda@1` (std minor beside `gated_delta_rule@1`): `(q, k, v, f [T, H·Dk], β [T, H], A_log [H], dt_bias [H·Dk])`
  with the fact `lower_bound` (−5); chunked and recurrent paths, in-step snapshots, per-token states as GDN's.
- Kernels: `kernels.fla`'s `vendor.sh` adds `ops/kda.py` and `fused_recurrent.py` at the same vLLM commit
  `6dd813e0` (`fused_recurrent_kda` :109, `chunk_kda` :1507, `fused_kda_gate_chunk_cumsum` :1301 with
  `lower_bound` :1281); `vendor.patch` drops their vLLM imports (`CustomOp`, `math_utils`) as it does for GDN's. FN's
  `SUB_TOKENS` packing and in-step snapshots in `chunk_delta_h` apply unchanged (it already takes per-channel `gk`).
- **Rewind:** GDN's identity step uses `softplus(−1e4) = 0` (`gdn/kernels.py:44`); the bounded gate has no finite
  input that gives g = 0, so the KDA rewind masks g itself (a `masked` row flag), with a test that a masked row
  leaves the state bitwise unchanged.
- β is the fp32 sigmoid (LC, Maya, FT; HF's bf16 sigmoid is a no-op in the fp32 oracle).

**D5 — DSA: `std.mixers.dsa_mla@1`, design §9's `dsa_mla` mixer: an MLA latent `Paged` spec plus an indexer
`Paged` group.**
- **Specs:** `"kv"` — the latent, `[page, 512]` bf16, **one copy read as both K and V** (a GQA layout with one KV head
  would store it twice, G-C §3); `"idx"` — per page the raw keys (bf16 128), the gates γ (bf16 128) and the pooled
  keys (f32 128 per 4 tokens): 640 B per token per layer. Two groups, as QSA's (the core refuses one page group shared
  by paged specs). `page_size % 4 == 0`.
- **Pooling** (`dsa_index@1`, new): a pool's key is computed when its 4th token is appended, from the **stored** raw
  keys and gates (rewind and prefix hits stay stateless, FN D6); a pool completed by rejected drafts is rewritten.
- **Selection:** `qsa_select@1` gains an optional per-row head-weight input `w [T, Hi]` (fact `head_weights`, default
  off so FN's facts do not move) and `Hi` up to 32 (its Triton provider allows 16 today, G-C §3). fp32 scores, IEEE
  products, the exact top-k with lower-pool ties, shape-invariant, sub-batched workspace — FN D6 unchanged. The
  dense / sparse rule is QSA's (dense while every row's host-known length ≤ 2,051; verify rows by their upper bound).
- **Attention** (`mla_attend@1`, new): absorbed MQA over the latent for prefill, decode and verify, dense and over
  the selected pools + tail; `W_k` / `W_v` come split in the GGUF (`attn_k_b` [H, 512, 256], `attn_v_b`, LC
  `conversion/glm.py:526-536`) and are read as such (no `finalize_weights` absorption needed); scale 1/16 a fact.
  The dense path is the sparse kernel with every pool selected (Maya's `DSA_ALL`) unless step 0's probe (N5) shows a
  dense kernel ≥ 10% faster on the 5090's prefill.
- **Why not FlashInfer:** 0.6.12 has no NoPE or sparse MLA for sm_120 (its trtllm / xqa paths need a 64-wide rope
  part and sm100; fa2's MLA wrapper with `head_dim_kpe = 0` is untested, G-C §3); N5 probes it.
- **The MTP layer's DSA** runs in the aux namespace with a device-length planner (aux paged state is allowed,
  `core/holon/core/assembly.py:108-136`), as FN's head runs QSA; with `index_share_for_mtp_iteration` the head's
  sub-step 0 selects and sub-steps 1+ reuse its pools (VL `common/mtp.py:147-149`).

**D6 — MoE: two generic contract minors, facts passed only when not the default.**
- **Router:** `moe_topk@1` and `moe_router@1` gain the facts `scoring` (`softmax` | `sigmoid`), `bias` (an `[E]` f32
  parameter used for selection only), `scale` (× after renormalising); `std.moe@1` gains the bias parameter. E need not
  be a power of two: the vendored `topk-moe.cu` already has sigmoid, the bias, the scale and `case 288`
  (`:136-141, 170, 259-275, 337`); only Holon's wrapper (`csrc/api.cu:476-488`, softmax only, null bias) and
  `_supports_moe_topk` (`plugin.py:764-766`, powers of two) change, plus `HOLON_ROUTER_CASE(288)` in the fused router.
  **Today E = 288 has no `moe_topk@1` kernel at all** (G-C §5).
- **Clamp:** `linear_experts@2`, `linear_experts_arena@1` and `silu_mul@1` gain `glu_limit` (absent: no clamp):
  `silu(min(g, L)) · clamp(u, −L, L)`. ggml's `GGML_GLU_OP_SWIGLU_CLAMP` is vendored in mmvq (`mmvq.cu:863, 1043`);
  the fused-SwiGLU mmq store and `kernels.ggml_cpu` get the same form; every oracle too.
- The shared expert composes as `std.ffn.sum@1(moe, swiglu)` (FN D7), ungated; the router-prepared q8_1 rule of
  `docs/v1-offload.md` §4.7 holds (routed gate / up before the shared expert).
- Why facts and not a GLM unit: sigmoid + bias + scale is DeepSeek-V3's `noaux_tc` and the clamp is gpt-oss / Step's;
  both are generic.

**D7 — MTP: built and judged; shipped in a lock only where it is measured faster.**
- `std.spec.mtp_head@1` exists for this shape (`order = "embed_first"`, a prenorm block, its norm; G-C §7); GLM's head
  is `blk.45` of the main GGUF, so no MTP pack. Two plugin edits: `drafters/mtp` unwraps an `expand`-wrapped embed
  whenever it has an `inner` (today only when the head carries the residual, `plugin.py:42-46`), and `MTPHead` gains
  `write_kv` so a catch-up skips the MoE (as `MTPHCHead` has).
- Its experts go through the store as `aux.*` (FN D12a), which needs D8(d) when RAM is short.
- **Maya runs no MTP on one NVIDIA GPU** (`glm_mtp.cu:22-24, 78-98`): consecutive tokens share ≈ 1 of 8 experts, so
  a verify window pays nearly every row's misses (My `bench/results/PROGRESS.md:76-78`). The bar rows are therefore
  each engine's deployment; Holon's lock takes MTP only if tune's interleaved bench measures ≥ 5% on the mode (i)
  rows, and Holon-with-MTP is reported either way.
- **Build or defer** (Q3): if step 0's replay (N2) predicts MTP decode ≤ 0.9× plain at the 5090's fit, the head is
  deferred to a follow-up (it saves ≈ 4 days: the aux DSA, `write_kv`, D8(d), the MTP gates).

**D8 — Offload: X's store, with plan §10's std minors landing first (step 1), each tested on the Qwen3-30B-A3B vehicle
with RAM capped before GLM uses it.**
- **(a) F10 with RAM short:** the delta-log record "push b's slots onto the layer's empty stack", replayed by
  `ResidencyMap.commit` outside borrowing steps, so the host mirror stays exact and exclusive tiers accept
  `placement.promote@1` (plan §10; `docs/v1-offload.md` §14 step 8).
- **(b) Lending and the prefill ring under exclusive tiers:** a lent piece is demoted (its tier code to RAM if a free
  RAM unit exists, else SSD-only — the file is always valid) before its slot is overwritten; the ring's stages read
  the next layer's SSD-only routed set ahead into pinned landing units while this layer computes (Maya's disk landing
  ring, G-B §4). This removes today's rule "exclusive ⇒ staging = a prefill pass's, no ring, no lending, 512-token
  passes" (`std/holon_std/tiered/store.py:240-251, 435`).
- **(c) The high-priority split fetch under exclusive tiers** (`resolver.py:81` refuses it today), ordered after the
  SSD wait for SSD misses.
- **(d) Aux experts under exclusive tiers:** the device tables are made at `before_materialize` in every layout
  (today an exclusive store makes them at the target's first `resolver()`, so the MTP head's pieces raise,
  `store.py:200-226`).
- **(e) Staging per class** follows the lending layout: `k × max_seqs × (1 + drafts)` (8 at GLM's lock) instead of a
  prefill pass's 288. **What (b) + (e) buy on the 5090 in mode (ii): 2.94 GB of VRAM staging (288 × 3 classes of 4.3 /
  2.7 / 3.2 MB) returned to the cache, ≈ 400 experts, +3.3 points of cache share, and as much pinned host memory.**
- **(f) The CPU split (F5b)** stays reported only (FN lost 2–14% with it; Maya's lane is worth +4–10% on a 5090,
  `docs/SETTINGS.md:259-261`). It is not a default unless D4 of `docs/v1-offload.md` picks it (≥ 5%, bounded).
- **Placement defaults:** mode (i) `lfu` (F11, X's default where RAM holds every expert); mode (ii) the least
  modelled time among F11 with deferred admission (X's RAM-short default) and F10 spares, by step 0's replay on GLM's
  traces (N2) — X's D10 rule — confirmed by step 6's run.
- **The seed** is a GLM routing profile from `route_dump` on held-out prompts (FN D12f), shipped in the bundle.

**D9 — Prefill: one step size per deployment, the largest whose loan fits.** The step's activations are lent from the
arena's tail (`moe.lend_peak`, FN §21.12) and the MoE pass buffers too (T2). In mode (ii) the SSD-only experts a step
routes (≈ 9 GB est., §7) are read once per step, so larger steps pay off far more than in mode (i) (Maya reads 32K
chunks, lending ≤ 90% of its pool: `glm_prefill.cu:1007-1026, 1123-1138`). Each deployment's `max_step_tokens` is
chosen in step 7 by FN §21.17's objective (the least decode ratio, highest, with every prefill row at its bar × 1.05).

**D10 — The two host modes.**
- **(i) RAM holds every expert:** a bar desktop whose RAM pins all 88.65 GB of experts (MTP's included) plus the
  process; F11 on; no SSD tier.
- **(ii) 64 GB:** each engine's process under `systemd-run --scope -p MemoryMax=64G -p MemorySwapMax=0`; **the same
  pinned expert tier for both: 50 GiB** (est.: Maya's own default on a 64 GB host is MemAvailable − 6 GiB, G-B §7;
  step 0 measures both engines' RSS outside the tier, and the tier is the largest that both start and serve under
  the cap, if that is less); the GGUF on the node's local NVMe, read with `O_DIRECT` by both. Maya's tier is set
  explicitly (`MAYA_GLM_RAM_GB`, GiB) because its free-RAM probe ignores the cgroup.

**D11 — The comparison protocol: FN's D11 as closed (`bench/offload_ctx.py`).**
- **The Maya backend** speaks the engine's stdin protocol with raw ids (`GEN <n> temperature=0 <ids>` → `T <id>` …
  `DONE …`, `generate.cpp:2063, 2213-2222, 7005-7041`), the protocol `./maya.sh --bench` drives; the HTTP server has
  no raw-id or ignore-EOS path (G-B §8). `--glm-pack` from `tools/iq_pack.py --compat-bf16`, as Maya's setup builds it.
- **Maya's settings:** its setup's defaults (`--expert-cache auto --prefill auto --kv-streaming on --expert-profile
  data/expert-profile-glm.bin`), `--max-context 33792 --kv-resident 33792` (a 32K config never streams; without the
  second flag 33,792 would), `--calibrate` run once per card and its values recorded, `MAYA_GLM_SLOTS=0` (it writes the
  previous conversation to the SSD before each prompt, outside `prompt_ms`), loop guards off
  (`MAYA_GLM_LOOP_ANSWER=0 MAYA_GLM_LOOP_THINK=0`), `--eos-ids 2147483647` as ignore-EOS (N0 checks `produced` = n
  in every `DONE`; else FN's EOS pre-screen). Built at `4615f34` on the bench host (`GGML_NATIVE`), sm_120.
- **Maya's checkpoint cut:** by default a 32K raw-id prompt is read as 16,384 + 16,384 (`generate.cpp:1603-1630`).
  The 32K cells are gated against **the faster of** Maya's default and `MAYA_GLM_CKPT_EVERY=0` (Q5).
- **Holon:** its lock's configuration, `--ctx 33792`, `ignore_eos`, prefix reuse as deployed with 0 reused tokens per
  timed request (both engines log reuse: Holon's `cached_tokens`, Maya's `DONE … reused`).
- **Corpus and spans:** FN's (the text of `v1-long`, `v1-qwen3_5`, `v1`), tokenized with GLM's tokenizer; FN's
  context-specific offsets, warm-up span and screening.
- **Statistics:** 8 prompts per context, 3 rounds in ABBA order, each arm a fresh process, a discarded warm-up first
  in each allocation; the cell is the ratio of medians; **pass** ≥ bar; within 3% of the bar → 6 rounds (skill §4a);
  below → the owner. The run, not the prompt, is the unit (FN §22.9, panel C).
- **Per cell:** expert hit rate (all, and the first 32 decode tokens), SSD reads per token (mode (ii)), Maya's
  `glm stat:` lines (pool, RAM tier, CPU plan), peak VRAM and RSS.

**D12 — Kernels: Triton in kernel plugins; CUDA only by FN D10's rule.** `kernels.dsa_triton` (`dsa_index`,
`qsa_select` with weights, `mla_attend`), `kernels.hc_triton` (the mHC pair, decode split-K over 16,384 inputs, the
prefill product in fixed-order passes), `kernels.fla` (KDA), `kernels.ggml_cuda` (router and clamp wiring). Adapted
code from FreeToken (Apache-2.0) is declared per AGENTS.md; Maya (MIT) is the performance reference. A kernel moves to
CUDA in step 7 if it is over 5% of a step and a prototype gains ≥ 10%.

**D13 — Tokenizer and frontend (std / plugin minors).** `std.weights.gguf`'s `_PRE` gains `glm4` (and `glm5`): BPE
with `ignore_merges = True` (llama.cpp's CHATGLM4, `llama-vocab.cpp:2419-2424`; `tokenizers` 0.22 supports it,
checked), no normalizer, GLM's split regex (`\p{N}{1,3}`, `tokenizer.json`); `eom` (`<|observation|>`) read as a stop
token beside `eot`. A `glm_xml@1` tool-call format (`<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>…`,
My `serve/frontend.py:8-10`). The template opens `<think>` itself, so `think@1` is checked on a prompt that ends in it.

**D14 — Text only.**

**D15 — Locks and propagation.**
- **New:** `rtx-5090` (Maya-S, mode (i), `max_seqs` 1 as FN's, MTP per D7) and `rtx-5080` (mode (i)). Mode (ii)
  runs the 5090 lock with `--host-memory` and `moe.ram` overrides recorded in the evidence (K's 27B precedent);
  no lock is keyed on a host's RAM (Q10).
- **GLM changes `src` in:** `std.moe` (router facts, bias), `std.tiered` (D8), `std.hyper`, `std.mixers.qsa`
  (weights), `std.weights.gguf` (tokenizer: every GGUF lock), `kernels.ggml_cuda`, `kernels.ggml_cpu`, `kernels.fla`,
  `drafters/mtp`, `std.spec.mtp_head`. New facts are passed only when not the default, so traces should advance
  byte-identical; one propagation at the close (release §22).

**D16 — Accuracy is the owner's method (FN §18), from the start.** No goldens or floors; L0 on three SMs; the tiny
model's HF match; gates (c), (d), (e), (f); the exactness checks of acceptance 2. The plants of §6.3 are pre-registered.

**D17 — K4 (KV streaming of the latent) is deferred.** At ≤ 32K the latent and the indexer pages are 0.62 GB (§7) and
Maya does not stream a 32K configuration (G-B §6). K4 stays FN's follow-up; GLM's DSA adopts it when a ≥ 64K row is
in scope.

## 3. What Holon has today (G-C)

| Piece | State |
|---|---|
| Core | 5,019 / 6,000 code lines, ABI 1.5; denylist names `mla`, `dsa`, `kda`, `qsa`, `indexer`, `mtp`, `moe`, not `glm`, `mhc`, `sinkhorn` (`scripts/ci/check_core_denylist.py:8-12`) |
| Residual and state seams | C1 carries `[T, 16384]` f32 unchanged; KDA's Slots are GDN's; the latent and `idx` are two paged groups as QSA's; aux paged state is allowed (`assembly.py:108-136`) |
| `std.hyper` | FN's gated-residual form only (no stream mixing); `expand@1` reusable |
| Attention | GQA, QSA (`qsa_index` / `qsa_select` ≤ 16 heads / `qsa_attend` GQA), no MLA anywhere |
| Linear attention | GDN through vendored FLA at vLLM `6dd813e0` (no KDA files); `fused_sigmoid_gating.py` has an `IS_KDA` path never used |
| MoE | softmax routers only, power-of-two E only, no clamp anywhere; every Maya-S type in `TYPES` |
| Store | exclusive RAM tier + F9 SSD + F11 deferred admission work; no lending / ring / split under exclusive tiers; F10 refused under exclusive tiers; aux experts raise under exclusive tiers |
| GGUF | split files; gguf-py at LC `27c7a76` knows `glm5-next`; tokenizer `qwen2` / `qwen35` only |
| Judge and bench | `spikes/fn7/{consistency,cover,exact,ledger,lend_min,admit_sim}.py`, the 72-item task set, `offload_ctx.py` with a Strata backend (Maya's protocol is the same family) |

## 4. Core seams: none planned

| # | Need | Today → change | Why no core line |
|---|---|---|---|
| — | mHC's 4-stream residual | C1 `ModelSpec.residual` → unchanged: `Residual(16384, "float32")` | FN built it for this (`docs/v1-offload.md` §4.7); the profile builds the drafter's input at the residual width and GLM's drafter norms it itself (`executor.py:246-249, 283`) |
| — | KDA state | mixer `DeclaresState` with two `Slot` specs → unchanged | GDN's shape (G-C §4) |
| — | Latent + indexer pages | two `Paged` groups → unchanged | QSA's layout (FN D6) |
| — | MTP head with DSA and 288 experts | aux model, aux paged state, store `aux.*` group → unchanged | FN D9 / D12a; the head's planner is device-length (`assembly.py:129-136`) |
| — | Step-peak loan in mode (ii) | `ActivationLender` → unchanged | the store returns the loan once D8(b) lets an exclusive layout lend |
| — | Memory accounting | THP host pools, `host_memory`, the CLI size check → unchanged | X's S2 and the CLI row (`docs/v1-offload.md` §4) |

- **Generality:** `check_core_denylist.py` gains `glm|mhc|sinkhorn|nextn` (a CI change, not core).
- **If a core need appears** (FN's `ActivationLender` did, mid-build): a toy proof in `core/tests/test_payloads.py`,
  the lead's sign-off per line, the count at each step close, cap +15 (acceptance 6). FN overshot its budget by two
  uncounted lines; GLM counts at every step close, not at the end.

**Std and plugin contract minors** (additive; `std/contracts.txt` / `plugins/contracts.txt` regenerated):

| Contract or unit | Today → change |
|---|---|
| `moe_topk@1`, `moe_router@1` | softmax, power-of-two E → facts `scoring`, `bias`, `scale`; any E the provider instantiates (D6) |
| `linear_experts@2`, `linear_experts_arena@1`, `silu_mul@1` | no clamp → fact `glu_limit` (D6) |
| `qsa_select@1` | `Σ_h relu(q_h·K̄)`, ≤ 16 heads → optional `w [T, Hi]` (fact `head_weights`), Hi ≤ 32 (D5) |
| new `dsa_index@1`, `mla_attend@1`, `kda@1`, `hc_mhc_pre@1`, `hc_mhc_post@1` | — (D3–D5) |
| `moe_tiers@1`, `ResidencyMap`, the store | D8 (a)–(e) |
| `std.weights.gguf` | `glm4` / `glm5` pre-tokenizer, `eom` (D13) |
| `drafters/mtp`, `std.spec.mtp_head@1` | embed unwrap; `MTPHead.write_kv` (D7) |

## 5. Units, contracts, model plugin

| Unit | Role |
|---|---|
| `std.hyper.mhc_block@1(n, …)` | D3's sub-block pair around a mixer and an FFN; forwards the mixer's state via C3 as `gr_block` does |
| `std.hyper.mean@1(inner)` | the collapse, as `ModelSpec.norm` |
| `std.mixers.kda@1` | D4 |
| `std.mixers.dsa_mla@1` | D5 |
| `std.moe@1` (+ bias), `std.ffn.sum@1`, `std.ffn.swiglu@1` (+ limit) | D6 |
| `std.spec.mtp_head@1` | D7 |

**Contracts** (each with `reference.py` and a CPU provider in `testing/holon_testing/ref_kernels`): `hc_mhc_pre@1`,
`hc_mhc_post@1`, `kda@1` (+ its rewind), `dsa_index@1`, `qsa_select@1` (weights), `mla_attend@1` (dense, sparse;
prefill, decode, verify; reads the latent in bf16), the router and clamp facts.

**`models.glm5_next`** (`derived_from` `models.qwen4_exp`, FN's layout): the recipe reads `cfg["gguf"]`
(`glm5-next.*` keys; `head_count_kv` per layer and `key_length` = 512 are read raw, not through the generic mapping,
G-C §8), sets `residual = Residual(16384, "float32")`, `embed = std.hyper.expand@1(inner = std.tiered.rows@1)` (the
token table on the host, as FN's; 0.52 GB of VRAM back), `norm = std.hyper.mean@1(rmsnorm)`, per layer `LayerDecl(kda
| dsa_mla, swiglu(limit) | sum(moe(sigmoid, bias, 2.5, limit), swiglu(limit)))` inside `mhc_block`; GGUF rules per
LC's names (`glm5-next.cpp:61-184`; `ssm_a` = −exp(A_log) from the converter); MTP rules over `blk.45.*`; the bundle
(chat template, `glm_xml@1`, `think@1`).

## 6. Numerics and judge (thresholds and predicted verdicts fixed now)

### 6.1 CPU: the tiny model vs HF (fp32)

**Config:** 8 layers (DSA at 3 and 7; dense FFN at 0; KDA elsewhere), H = 64, n = 4 streams, KDA 2 heads × 32,
MLA 2 heads (q_lora 32, kv_lora 32, nope 32, v 32), indexer 4 heads × 16 with kpool 4 and top-k 8 (sparse from
t + 1 > 11), 8 experts top-2, I = 64 (a multiple of every block size used), shared expert, swiglu_limit 10 with
weights scaled so the clamp binds on ≥ 5% of rows, vocab 154,880 (the real tokenizer), 1 NextN layer. The prompts:
100 tokens containing an EOS, chunked at 7 and 13 (straddling pools and not a multiple of 4), then 24 decode steps.

| Check | Threshold | Predicted |
|---|---|---|
| logits (recomputed with the GGUF's head), one-shot and chunked, vs HF | max \|Δ\| ≤ 1e-4, top-1 identical | pass |
| DSA token sets per (layer, position) | identical under the lower-pool rule; ties asserted to occur | pass |
| prefix hit vs recompute | bitwise | pass |
| MTP head vs the LC-transcribed reference | ≤ 1e-4 | pass |
| clamp bind rate on the fixture | ≥ 5% of expert rows (asserted, so the clamp is exercised) | pass |
| tokenizer vs `llama-tokenize` (LC `27c7a76`) on a multilingual round-trip corpus (code, CJK, digits, emoji) | identical ids | pass |
| plants `mhc_comb_transposed`, `dsa_pool_ape_off`, `router_bias_in_weights`, `kda_gate_softplus`, `glu_clamp_off` | the tiny test fails | fail |

### 6.2 Full model (Maya-S)

| Check | Where | Predicted |
|---|---|---|
| (c) decode vs prefill; chunked (250) vs one-shot: KL mean ≤ 0.13, p50 ≤ 0.007 | 5090 deployment | pass (FN: 0.025–0.031 / 0.002–0.003) |
| (c) MTP vs plain (if built): KL mean ≤ 0.02, p99 ≤ 0.3, ≥ 100 positions / 8 prompts | 5090 | pass |
| (d) 72 items vs Maya, greedy, thinking on, 16,384 new tokens at most, the template's default effort | both engines' 5090 deployments, mode (i) | pass (within 5 items) |
| (f) the cover: DSA selection 100% equal; KDA layer, mHC pair within L0's budgets | RTX PRO 6000 and 5090 | pass |
| exactness (acceptance 2): fits, policies, ring / lending, mode (i) vs (ii), ×3 | RTX PRO 6000; 5090; 5080 | byte-identical |
| the clamp's bind rate on the bench prompts | reported (decides whether `glu_clamp_off` can bite in (c)) | — |
| zero-score ties at the selection boundary, per 1,000 rows | reported | — |

### 6.3 Plants (each changes only its target path)

| Plant | Breaks | Predicted |
|---|---|---|
| `dsa_decode_select_shift` | decode and verify selection one pool past the budget | **(c) decode vs prefill fails** on `v1-long` (pre-registered); tiny decode test fails |
| `kda_gate_softplus` | the recurrent (decode) KDA kernel uses the unbounded softplus gate | **(c) decode vs prefill fails** (pre-registered); tiny fails |
| `kda_state_not_carried` | a chunked prefill step starts from a zero state | (c) chunked vs one-shot fails; tiny chunked fails |
| `mhc_comb_transposed` | `comb` applied as C, not Cᵀ, everywhere | tiny fails; (c) passes (both paths); cover fails |
| `dsa_pool_ape_off` | APE left out of pooling, everywhere | tiny fails; (c) passes; cover fails |
| `router_bias_in_weights` | the bias added to the weights too | tiny fails; (c) passes |
| `glu_clamp_off` | the mmvq (decode) path without the clamp | L0 and tiny fail; (c) fails only if the bind rate (§6.2) is ≥ 1e-3 of rows, else passes (recorded, not a gap: L0 and tiny cover it) |
| `mtp_h_prenorm` | the head reads h before the final norm | the MTP reference fails; tokens per pass drop ≥ 0.3 |
| `exclusive_lent_not_demoted` | a lent piece's tier code left at RAM under an exclusive tier | `exact.py` mode (i) vs (ii) differs |
| `promote_spare_unlogged` | F10's delta record dropped | the host mirror check fails; `exact.py` differs |
| `aux_row_padding`, `lent-slot-still-mapped` (existing) | as FN / X | `exact.py` differs |

## 7. Memory fit (est.; step 0 and tune measure)

**The RTX 5090** (FN §23.1's ledger: 31.88 GiB accounted at 1 sequence), mode (i), 33,792 tokens, 1 sequence:

| Item | GiB | Note |
|---|---|---|
| outside the allocator | 0.95 | FN's 0.92 + the larger host mapping |
| parameters (non-expert, token table on the host) | 6.64 | 7.65 − 0.52 GB; + 0.16 GB with the MTP block |
| state pools | 1.00 | latent 0.38 GB + `idx` 0.24 GB + KDA 2 slots and 1 checkpoint 0.46 GB; + 0.06 GB for the MTP head's DSA |
| graphs' reserve | 0.12 | FN |
| activation reserve (the worst step's live peak) | 1.02 | **est. ≈ 270 KB per token at a 4K step**: HC streams 64 KiB, MLA's q̃ [T, 64, 512] bf16 64 KiB, KDA temporaries, ggml dequant buffers; N4 measures it |
| slack | 0.44 | FN |
| **the arena** | **21.70 (23.3 GB)** | |
| staging and null slots (D8e) | 0.11 | 3 classes × (8 + 1) |
| **experts cached** | **≈ 3,260 = 27%** of 12,096 | Maya: ≈ 3,250 at its 700 MiB reserve (G-B §2), with no activation reserve (its prompt path borrows) |

- **Mode (ii):** the same VRAM; RAM tier 50 GiB ≈ 7,550 experts; **SSD-only ≈ 1,290 experts (9.2 GB)**. Without
  D8(b, e), staging would be 288 per class: 2.94 GB of VRAM (≈ 2,860 experts, 24%) and as much pinned RAM.
- **Activation at larger steps** (D9): ≈ 270 KB per token → 4.4 GB at 16K, lent from the arena during prompt steps.
- **Host, mode (i):** experts 86.03 GB + MTP 2.62 GB pinned (THP), + the token table 0.52 GB.

**The RTX 5080 (16 GB), mode (i):** visible 15.47 GiB; parameters 6.64, state 1.00, outside 0.6, activation 0.75 (a
3K step), graphs 0.1, slack 0.4 → arena ≈ 6.0 GiB → **≈ 890 experts, 7.3%**. It fits; decode is miss-bound (≈ 50% hits
est.), so it is a reported row (Q4). Maya-S24 would add ≈ 1.7 GB of experts (≈ 240) but is another 94.7 GB download.

**The RTX PRO 6000 (96 GB), step 5:** ≈ 102.6 GB visible (FN §7's estimate) less the same items leaves ≈ 91.7 GB for
experts against 88.65 GB (MTP's included): **all-resident fits, ≈ 3 GB to spare (est.)** — acceptance 2's reference
fit, as FN's. If N1-style measurement shows it does not, the store at its largest fit is the reference instead.

**Decode roofline, the 5090, mode (i)** (pre-registered form; N4 fills h, the dense bandwidth and the miss rate):
`t ≈ dense / B + max(336·h·e / B, 336·(1−h)·e / P) + t_fixed`, with dense ≈ 7.1 GB a token, e = 7.11 MB, B ≈ 1.55 TB/s,
P ≈ 50 GB/s (FN §23.6, THP), t_fixed ≈ 1 ms: **h = 0.70 / 0.75 / 0.80 / 0.85 → 50 / 57 / 66 / 78 tok/s.** Maya's
published 5090 rows are 23.0 (Maya-S, a 64 GB host) and 30+ (Maya-M, 96 GB) tok/s (`README.md:123, 135`).

**Prefill caps:** mode (i), every non-resident expert streamed per step (≈ 63 GB at 53 GB/s): 1K 0.86K, 4K 3.5K,
8K 6.9K, 16K 13.8K tok/s. Mode (ii) adds the step's SSD-only reads (9.2 GB): at 3 / 6 GB/s, 4K 1.0 / 1.5K, 16K
3.9 / 6.0K, 32K 7.7 / 12.1K tok/s — **why D9's large steps matter there.**

## 8. Step 0 — measure before freezing

On the RTX 5090 desktop (and the 5080 for N1's reported row), one heavy job at a time, under `MemoryMax`.

| # | Measure | Decides |
|---|---|---|
| N0 | The harness: Maya built at `4615f34` (sm_120, on the bench host) and its pack; `offload_ctx`'s Maya backend; `--eos-ids 2147483647` (`produced` = n); `reused` = 0; the corpus retokenized; `--calibrate` per card, values recorded | D11 runnable |
| N1 | Maya's rows: 5090 modes (i) and (ii), 1K / 4K / 32K, defaults and `CKPT_EVERY=0` at 32K, the deterministic recipe; 5080 mode (i); Maya's ledger (pool experts, RAM tier, SSD-only count, CPU plan, disk reads per token), its RSS outside the tier | the denominators; D10's tier size |
| N2 | Maya's route traces (`STRATA_GLM_ROUTE_LOG`, `glm_fast_path.cu:4148-4151`) on the bench corpus and the 72 prompts' prefixes, both modes → the M6-style replay (`spikes/x0_offload/m6`, `spikes/fn7/admit_sim.py`, `lend_min.py`) at 27% (5090) and 7% (5080): hit rates of `lfu`, LRU, F10 spares, deferred admission with a 50 GiB RAM tier and N3's SSD rate; MTP verify windows (misses per pass vs per token) | D8's mode (ii) default; D7 build or defer; h for N4 |
| N3 | The node's NVMe: `O_DIRECT` expert-sized reads (2.7–4.3 MB) at depth 1–32 over 2–8 I/O threads, and chunked reads of one expert over workers (Maya's `READ_CHUNKS`); its free space for a 97 GB copy | `cpu_task.io_threads`, the SSD read path in D8(b) |
| N4 | The roofline per acceptance-4 row: dense bytes per component (G-A §8), h from N2, P from FN §23.6, SSD from N3; the activation peak per token at 4K / 8K / 16K from a kernel-level probe on an A5000 (vendored `chunk_kda` at 64 × 128 with FN's `SUB_TOKENS`, MLA's q̃, the HC streams, dequant buffers) | every gated row predicted; **a row predicted < 0.9× goes to the owner before step 1 ends** |
| N5 | (½ day) FlashInfer 0.6.12's fa2 MLA with `head_dim_kpe = 0` on sm_120: compiles? its 32K prefill time against a Triton prototype of `mla_attend` | D5's dense path |
| N6 | Maya's answers to the 72 items on the 5090 (its defaults, the same rendered ids), saved as gate (d)'s reference while Holon is built | gate (d) ready at step 5 |

## 9. Steps

| Step | Work | Checks | Days |
|---|---|---|---|
| 0 | N0–N6 | evidence; predicted rows; owner escalation of any < 0.9× | 4 |
| 1 | D8 (a)–(e) in `std.tiered` / `kernels.expert_io`, on the Qwen3-30B-A3B vehicle with RAM capped | `exact.py`-style identity RAM-capped vs RAM-holds-all (A5000); the vehicle's offload lock's trace byte-identical; plants `exclusive_lent_not_demoted`, `promote_spare_unlogged`; core diff empty | 5 |
| 2 | The oracle: `golden-glm` venv; the fixture tool with the tie wrappers, randomised init, the LC-transcribed MTP reference; `tokenizer.json` pinned; the tokenizer round-trip corpus | the HF driver bitwise vs HF's whole forward on the tiny model | 3 |
| 3 | Std units and contracts with references and CPU providers (D3–D6, D13), the model plugin, MTP edits (D7) | §6.1 on the CPU; Qwen3.8 / Flash-Next snapshots unchanged; core diff empty | 6 |
| 4 | Kernels (D12): router and clamp wiring, KDA vendoring + rewind mask, mHC Triton, `dsa_index`, `qsa_select` weights, `mla_attend`; first tuning against the launch target; `test_reserve.py`'s tiny-GLM case | acceptance 3 (L0 ×3 SMs, memcheck) | 7 |
| 5 | The full model on the RTX PRO 6000 (all-resident) then the 5090: gates (c), (d), (e), (f); exactness; the clamp bind rate and tie counts; `route_dump` profile; Holon's own traces re-check N2 | acceptance 1(b), 2; §6.3's plants | 4 |
| 6 | Offload on the bar cards: mode (i) and (ii) deployments, the placement default by N2's rule confirmed, the SSD tier on the node's NVMe, MTP's measured value | acceptance 2 on both cards; fit vs §7 | 4 |
| 7 | Perf closure: per-component breakdown vs Maya (nsys), D9's step sizes, fusions, CUDA rewrites by D12's rule; a per-change perf smoke | every gated row within its roofline band, or the gap explained; ≤ 720 launches per decode token | 6 |
| 8 | Locks (`tune --new-model`): `rtx-5090`, `rtx-5080` | L2-free per release §22 (new locks: snapshot, fit vs §7) | 2 |
| 9 | The comparison (D11) | acceptance 4 | 3 |
| 10 | Close: the GPU-sharded full suite, panel code review, propagation, plan §10 | acceptance 5–8 | 2 |

**Schedule:** 46 days of work. **Critical path** with two to three agents: 0 ∥ 1 ∥ 2 (5) → 3 (6; step 4's kernels
start beside it) → 4 (5) → 5 (4) → 6 (4) → 7 (6) → 8 (2) → 9 (3) → 10 (2) = **37 days**. Step 1 is independent of
the model and lands first (one branch changes shared offload code at a time).

## 10. Risks

- **R1 — Mode (ii) decode is SSD-bound.** ≈ 2–7 SSD reads a token (Maya's 64 GB host: 2.0–2.5, `PROGRESS.md:41-44`)
  at 2.5–6 GB/s cost 3–20 ms against a ≈ 15 ms mode (i) token. Holon's SSD path today has 2 I/O threads and waits
  inline per layer call (G-C §6); Maya reads 8 chunks over 8 workers. N3 sizes the path; D8(b)'s read path takes the
  measured depth. Escalated at step 0 if predicted < 0.9×.
- **R2 — Mode (ii) prefill.** Maya reads up to 32K chunks; at a 4K step Holon pays ≈ 9 GB of SSD reads per 4K tokens
  (§7). D9's large steps need a 4–9 GB loan at 16–32K tokens. If the loan cannot reach 16K, the row is escalated with
  the curve.
- **R3 — Maya's CPU lane** (+4–10% on a 5090, exact only against ggml-cpu, bounded against the GPU). Holon stays
  exact by default. If a mode (i) decode row is predicted < 1.0× after step 7, the lever is an exact fast CPU row
  (Maya's G9: bit-identical to ggml-cpu's dot) under F5b's bounded contract — escalated, not absorbed (Q7).
- **R4 — Kernel scale.** MLA's 64 heads × 512 accumulator in Triton on sm_120's ≈ 99 KB of shared memory; the
  indexer's 32 heads in fp32 (8× QSA's); KDA's 64 × 128 temporaries against the reserve (FN's GDN overrun:
  `test_reserve.py` from step 4).
- **R5 — The hit rate at 27%.** Every decode row rests on h (§7); Maya's replay gave LRU 67%, LFU 77.5% on one GPU
  (`PROGRESS.md:84-85`). N2 measures it on GLM's traces before any build.
- **R6 — Oracle gaps.** No HF MTP (LC-transcribed reference); HF's FLA swap (venv without FLA); exact-zero ties (the
  wrapper, ties counted).
- **R7 — Maya moves fast** (v1.1.15 → v1.1.27 in 8 days; a DeepSeek engine landed beside GLM's). Pinned at `4615f34`,
  re-checked at the close; a newer Maya that changes the GLM one-GPU path re-runs N1 only with the owner's go.
- **R8 — Host RAM for mode (i):** ≈ 89 GB pinned plus the process on the bar desktop; checked at N1 (pinning limits,
  THP). If it does not pin, mode (i) runs with the largest tier that does and is reported as such.
- **R9 — Shared offload code (step 1)** moves the vehicle's and FN's offload paths: their traces must stay
  byte-identical (exclusive-only code paths); `holon_tools impact` at step 1's close.
- **R10 — MTP** likely loses at 27% cached (D7); its cost is bounded by the build-or-defer rule.
- **R11 — Disk:** the scratch filesystem has ≈ 300 GB free; GLM needs ≈ 100 GB there and ≈ 97 GB on the bar node's
  NVMe (§13).

**Deferred:** K4 (D17); MTP if N2 predicts a loss (D7); an exact fast CPU row (R3); Maya-S24 / M / L; DFlash2;
`--parallel` batching rows (D12 of `docs/v1-offload.md` already groups misses; `max_seqs` 1 in the locks); the vision
tower; prefill tail skip (Maya's last-layer skip, exact; ≈ 1/45 of the MoE work: step 7 if cheap).

## 11. Owner questions (each with the recommended default, which the build follows until answered)

| # | Question | Recommended default |
|---|---|---|
| Q1 | Gate both host modes on the 5090 (12 cells), mode (ii) as a 64 GiB cap on a larger desktop? | **Yes** |
| Q2 | Mode (ii)'s tier: the same pinned expert tier for both engines, 50 GiB unless step 0 shows less fits under the cap? | **Yes** |
| Q3 | MTP: build and judge it, ship it only where ≥ 5% faster, defer it if step 0's replay predicts ≤ 0.9×? | **Yes** |
| Q4 | The 16 GB row: RTX 5080, Maya-S, mode (i), reported (not gated), with a lock? | **Yes** (no Maya-S24 download) |
| Q5 | Maya's 32K cells: gate against the faster of its default and `CKPT_EVERY=0`? | **Yes** (conservative) |
| Q6 | The prefill bar stays 0.9× (v1-offload §1.4; FN closed with it), decode 1.0× in tok/s? | **Yes** |
| Q7 | Holon stays exact (GPU-only) by default; a CPU split only by `docs/v1-offload.md` D4's rule, reported? | **Yes** |
| Q8 | Core budget: 0 planned, a hard cap of +15 lines with a sign-off per line? | **Yes** |
| Q9 | Disk: download Maya-S only (96.5 GB); no HF FP8 checkpoint (328 GB); keep FN's GGUFs (≈ 115 GB) for FN's propagations? | **Yes** |
| Q10 | Mode (ii) runs the 5090 lock with host overrides, not a lock keyed on host RAM? | **Yes** |
| Q11 | Maya's arms after its own `--calibrate` on each card (its best settings)? | **Yes** |

## 12. Logistics

**Model files.** Maya-S v2 from `peasantsmith/GLM-5.3-Flash-Maya-GGUF` (`Maya-S-v2-IQ2_XXS/`, 3 shards: 47.84 + 47.80
+ 0.84 GB = **96.47 GB**), into the shared HF cache with `--include "Maya-S-v2-IQ2_XXS/*"` at a pinned revision, sha256
checked against `maya.py:124-160`. `tokenizer.json`, `config.json` and `chat_template.jinja` from
`zai-org/GLM-5.3-Flash@46a050be` (20 MB). Not downloaded: the HF FP8 checkpoint (328 GB; the tiny oracle needs only
HF's code), Maya-S24 / M / L (94.7 / 116.0 / 156.3 GB), Unsloth's files, DFlash2.

**Disk.**

| Item | Where | Size |
|---|---|---|
| Maya-S | the shared HF cache | 96.5 GB |
| Maya v1.1.27 build (+ its fetched llama.cpp) and its pack (`--compat-bf16`: the index and ≈ 1 GB of dense weights) | `bench/.envs/maya`, `bench/.envs/maya-pack` | ≈ 3 GB |
| route traces (N2), tiny fixtures, evidence | `local/tmp/glm`, `evidence/` | < 2 GB |
| **total on the scratch filesystem** | | **≈ 100 GB** of ≈ 300 GB free |
| mode (ii)'s copy of the GGUF | the bar node's local NVMe (per allocation, removed after) | 96.5 GB |

**Cards and hosts.**
- **RTX 5090** desktop: the bars (modes (i) and (ii)), N0–N3, N6, steps 6–9. Host RAM must pin ≈ 89 GB of experts
  plus the process for mode (i) (R8); the node's NVMe needs ≈ 97 GB free for mode (ii).
- **RTX 5080** desktop: the reported 16 GB row, its lock, sm_120 L0.
- **RTX 4090**: sm_89 L0 and memcheck.
- **RTX PRO 6000** (96 GB, a whole node, not a shared one): step 5's full-model gates and acceptance 2's all-resident
  reference.
- **A5000s**: sm_86 L0, the tiny CPU tests, step 1 on the vehicle, N4's probe, the suite.

## 13. Review log

Rev 1: no review yet.
