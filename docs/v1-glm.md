# v1 — GLM: GLM-5.3-Flash on Holon, then its expert offload compared with Project Maya

Status: **rev 2 (2026-10-25), for the panel's second round.** Rev 1 drew A 6 majors + 12 minors, B 9 + 17, C 3
blockers + 10 majors + 13 minors; §13 answers every finding. The lead's calls on the main threads (2026-10-25) are
applied and recorded in §11. Step 0 has started (N0 done, N1 and N2 running; §8). Nothing in the engine is built.
Parameters marked "est." freeze when step 0's evidence lands.

**Scope.** Plan §10 ("Expert caching/offload", milestone GLM) and `docs/v1-offload.md`:
- D2: the target (45 layers: 11 MLA + DSA indexer, 34 KDA; 288 experts top-8 + 1 shared; mHC; sigmoid `noaux_tc`
  routing with a SwiGLU clamp of 10; MTP);
- §1 acceptance 4: the program bar against Project Maya on the RTX 5090 with Maya-S;
- plan §10's std minors that land at GLM's start: **F10 with RAM short** and **lending and the prefill ring under
  exclusive tiers**. K4 (KV streaming, "GLM's DSA adopts it") is proposed for deferral: Q12, a scope question for the
  owner.

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
"N0 / N1 / N2" cite step 0's records under `bench/.out/glm0_*` and `local/tmp/glm/traces` (evidence records at step
0's close).

## 0. The model

From CFG and HFm, cross-checked against LC, VL and Maya (G-A §0–§8; panel A verified every row).

| Part | Fact | Source |
|---|---|---|
| Trunk | 45 layers, H = 4096, vocab 154,880, untied head; plain-`w` RMSNorm (eps 1e-5) before each mixer and FFN; final norm | HFm:66-80, 1294-1295, 1510 |
| Layer mix | KDA at i mod 4 ≠ 3 (34), DSA at 3, 7, …, 43 (11); dense FFN at 0–2 (I = 12,288), MoE at 3–44 | CFG; HFm:1290-1292 |
| mHC | 4 streams of H, initialised as the embedding repeated. Per sub-block: `r = rms(flat X)` (unweighted, 16,384 values, eps 1e-5); `m = fn·r` (24 values, fp32); `pre = σ(s0·m[0:4] + b) + 1e-6`; `post = 2σ(s1·m[4:8] + b)`; `C = softmax_dst(s2·m[8:24] + b) + 1e-6`, then 1 + 19 alternating normalizations (src, then dst / src), eps 1e-6; block input `Σ_n pre_n X_n`; write-back `X'_d = post_d · y + Σ_s C[s][d] · X_s`. Collapse: the plain mean of the streams, then the final norm | HFm:286-337, 1306-1349; LC:384-541 |
| KDA | 64 heads × 128; q/k/v projections, each a causal depthwise conv (4 taps, fp32 weights) then SiLU; gate `g = −5 · σ(exp(A_log) · (f_b f_a x + dt_bias))` per channel; `β = σ(b_proj x)`; l2-normed q, k (eps 1e-6), q / √128; delta rule `S ← diag(e^g) S; S += k β(v − Sᵀk)ᵀ; o = Sᵀq` in fp32; output `rms(o) · σ(g_b g_a x)` with the norm kept f32 through the weight and gate, then `o_proj` | HFm:350-397, 430-514, 665-770; LC:809-887 |
| DSA attention | NoPE MLA, 64 heads: `q = q_b(rms(q_a x))` [64, 256]; latent `c = rms(kv_a x)` [512] is all the cache holds; scores `q_k · c`, scale 1/16; absorbed: `q̃ = W_kᵀ q` (512), `ctx = Σ p c`, `o = W_v ctx` | HFm:1148-1235; LC:1034-1095 |
| DSA indexer | 32 heads × 128 from `q_r`; key `LayerNorm(wk x)` (with bias, eps 1e-6) and gate `γ = W_g x` cached per token; a pool of 4 tokens has key `Σ softmax₄(γ + ape)_c · k_c` per channel; score `Σ_h w_h(x) · relu(q_h · K̄)/√128/√32` with signed `w = weights_proj(x)`; top 512 complete pools ∪ the open pool's 0–3 tokens; **exactly dense while t + 1 ≤ 2,051** (FN's QSA rule); no RoPE anywhere; every layer `full` | HFm:773-1061; G-A §3.2 |
| MoE | router fp32 sigmoid; `p + bias` selects the top 8 (n_group 1); weights from the unbiased p, normalised, × 2.5; experts `silu(min(g, 10)) · clamp(u, ±10)`, I = 2,048; one ungated shared expert (same clamp); the dense FFNs take the clamp too | HFm:110-206; LC llama-graph.cpp:2295-2397 |
| MTP | 1 NextN layer (`blk.45`): `x = eh_proj[enorm(E[t+1]) ‖ hnorm(h_t)]`, h the trunk's **post-final-norm** hidden; a plain pre-norm DSA + MoE (288 own experts) + shared layer; `shared_head_norm`, the trunk's head; no streams. `index_share_for_mtp_iteration` (VL only): draft step 0 selects, later steps reuse — **Holon selects fresh per sub-step** (D7) | LC:548-675, 789-796; VL `common/mtp.py:95-110, 147-149` |
| Per token / sequence | latent 1,024 B + indexer 640 B per DSA layer: **≈ 17.9 KiB per token** over 11 layers; KDA state 34 × (4 MiB f32 + 144 KiB conv in the model dtype) = **140.9 MiB per sequence** | G-A §7; D4 |

**Ties:** pools whose heads all ReLU to 0 score exactly 0.0, and HF's own prefill and decode break them differently
(G-A §3.2). Holon and Maya break them to the lower pool (My `ref/glm.py:192-196`).

### 0.1 Files

| File | Size | Expert types | Per expert | Rest |
|---|---|---|---|---|
| **Maya-S v2** (`peasantsmith/GLM-5.3-Flash-Maya-GGUF@f16ee4ba`, `Maya-S-v2-IQ2_XXS/`, 3 shards; downloaded, sha256 verified) | **96.47 GB** | gate / up IQ2_XXS; down IQ2_S (layers 7–40), IQ3_XXS (3–6, 41–44); MTP Q2_K / Q3_K | 7.01 / 7.54 MB; 9.11 MB (MTP) | routed 86.03 GB; MTP experts 2.62 GB; non-expert 7.65 GB (Q6_K trunk, Q8_0 KDA gates and indexer, F32 router / mHC / norms / conv) |

- Arch `glm5-next`; pre-tokenizer `glm4`; every type upstream ggml and already in `kernels.ggml_cuda`'s and
  `kernels.ggml_cpu`'s `TYPES` (G-C §5). Split GGUFs are one checkpoint (FN D14).
- **Maya's pack rounds the router, its bias, the mHC weights, the indexer and KDA's small tensors to BF16**
  (`tools/iq_pack.py:62-82`); Holon reads the file's F32. The comparison is speed at each engine's exact defaults
  plus task accuracy, never token identity. Maya's pack reads the GGUF from the pack's parent directory (N0).

## 1. Goal and acceptance

**Goal.** Serve GLM-5.3-Flash from Maya's own GGUF, text-only, exact against an independent oracle on a tiny model,
through X's offload store on a 32 GB card (and a 16 GB development card), with exact defaults and reproducible
placement; then meet the program bars against Project Maya on the RTX 5090, with RAM holding every expert and on a
64 GB host.

1. **Model:** `models.glm5_next`, serving `arch.gguf.glm5-next@1` (text only).
   - **(a) Tiny model on the CPU** (§6.1): a random reduced config written by LC's converter matches HF 5.19.0 in
     fp32: logits within 1e-4 and top-1 identical, one-shot and chunked; identical DSA token sets per (layer,
     position) under the lower-pool tie rule (ties asserted to occur); a prefix hit bitwise equal to recomputing. The
     GGUF tokenizer gives llama.cpp's ids on a multilingual round-trip corpus.
   - **(a′) Tiny model on the GPU** (§6.1, C-M7): the same tiny GGUF through the GPU providers against the CPU engine
     at a bf16 bound — the check that runs every GPU-only path on real-shaped data.
   - **(b) Full model, by the owner's method** (FN §18; no fp32 goldens, no big-model CPU runs):
     - **(c) GPU self-consistency:** decode (1-token steps) vs prefill and chunked (250) vs one-shot on the 4
       `v1-long` prompts retokenized for GLM: KL mean ≤ 0.13 and KL p50 ≤ 0.007 (FN's bars). Two pre-registered
       must-fail plants (§6.3).
     - **(d) Task accuracy:** FN's 72-item synthetic set (`spikes/fn7/tasks`), the recall filler rebuilt to GLM's
       token lengths, rendered with GLM's template; Holon and Maya greedy on the same file and card, the same ids.
       Bar, stated now: Holon within 5 items of Maya's score, and no Holon answer degenerate where Maya's is not
       (pooled rates, the owner's ruling of 2026-10-24).
     - **(e) A sample dump** of 8 answers for a human look.
     - **(f) The cover** (FN §22.6, extended, C-M6): on the real model's own inputs, over ≥ 1 prompt of ≥ 8K tokens
       in 2,000-token steps (≥ 4 step boundaries, not multiples of 64) and ≥ 256 decode rows: DSA selection equal to
       its oracle on the engine's own q and pooled keys (a row whose differing pools lie within fp32 error of the
       512th score counts as equal, C-m13); the router's ids and weights; one KDA layer's output and state at each
       step start; one mHC pre/post pair; one `mla_attend` call at a sparse length. Bounds set FN's way (§6.2).
   - **On the shipped tree** (B-M7): gates (c) and (f) and `exact.py` run again at step 10 on the final tree at each
     lock's step size; gate (d) runs again if any decode-path kernel changed after step 5.
2. **Exact and run-identical** (A-M4): at **one fixed `max_step_tokens`** (the 5090 lock's), snapshot traces are
   byte-identical across fits (all-resident `std.moe@1` on the RTX PRO 6000, the deployment's cache, a capped cache),
   across policies (`lfu`, deferred admission, F10 spares), with lending and the ring on and off, and between RAM
   holding every expert and RAM short (exclusive tiers + SSD). A deployment's other step size is a performance
   setting outside this claim. Run-identical ×3 on the RTX 5090 and the RTX 5080 at their locks.
3. **Kernels:** L0 for every new kernel against its `reference.py` on sm_86, sm_89 and sm_120, memcheck clean; every
   decode step graphed. **Launches per decode token are reported, not gated** (C-M3: FN never met its ≤ 768): step 4
   takes a launch inventory on a 4-layer slice (FN's method) and the count × the measured per-launch cost enters the
   roofline. `plugins/tests/test_reserve.py` covers tiny GLM from its first kernel, including a multi-row step past
   `SUB_TOKENS` with KDA and a 16K-equivalent sub-batched step (FN §22.9).
4. **Program bars** (`docs/v1-offload.md` §1.4; protocol and statistics D11), RTX 5090, Maya-S, each engine's
   drafting as deployed (Maya: none on one NVIDIA GPU, G-B §5; Holon: none unless D7 builds and ships MTP), 1K / 4K /
   32K:

   | Host mode | Against | Decode | Prefill |
   |---|---|---|---|
   | **(i) RAM holds every expert** | Maya's calibrated defaults (no approximation by default, G-B §8) | ≥ 1.0× | ≥ 0.9× |
   | **(ii) 64 GB host** (both engines under one 64 GiB step memory limit, the same RAM tier in experts, the GGUF on the node's NVMe, D10) | same | ≥ 1.0× | ≥ 0.9× |

   - 12 gated cells, each predicted in §7.4 now. Plan §9's kill line (< 0.8×) applies to each.
   - **Reported, not gated:** Maya's deterministic recipe; Maya's 32K arm with `MAYA_GLM_CKPT_EVERY=0` (the gate takes
     the faster arm, D11); the RTX 5080 with Maya-S, both engines, mode (i); Holon with MTP if it is built and no lock
     takes it.
5. **Memory** (§7): the 5090 fits at 33,792 tokens in both modes at each mode's step size; the 5080 at 33,792 tokens
   in mode (i). Every pinned host byte is accounted against `host_memory`; in mode (ii) each engine's process stays
   under the step's memory limit, checked against the limit's own accounting (N1).
6. **Core:** **no core change is planned (§4).** Any core line is escalated to the owner with a toy proof; a hard cap
   of **+5** code lines, counted at every step close by `scripts/ci/check_core_loc.py` against `145a80df`. ABI stays
   1.5; std and plugin contracts change only additively.
7. **Locks:** `rtx-5090` (mode (i)) and `rtx-5080` (the development class), by `tune --new-model`. Mode (ii) is one
   named, pre-registered override set on the 5090 lock (D15). Every other lock is propagated once, at GLM's close.
8. **Tests:** the default `pytest` run grows by ≤ 5% (`--durations` top 10 reported at each step close); one shared
   tiny-GLM engine per module.

**Non-goals:** the vision tower; multi-GPU; DFlash2 (CC BY-NC-ND); K4 (Q12); Maya-S24 / M / L and Unsloth's files;
NVFP4 / EXL3; Maya's approximate opt-ins (AG1–AG6; Holon's A1 `approx` stays off); the CPU split as a default (D8f).

## 2. Decisions

**D1 — One file on every card: Maya-S v2** (downloaded at `f16ee4ba`, sha256 verified against `maya.py:124-136`). It
is the bar's file and Maya's choice for the 5090 (`maya.py:1280`); the 16 GB row uses it too (Q4).

**D2 — The oracle: HF transformers 5.19.0 `glm5_next` on the tiny model only, in a golden-only venv.**
- `local/envs/golden-glm`: transformers 5.19.0, CPU torch, **no `flash-linear-attention`** (HF swaps in FLA's
  exp2-domain KDA kernels when installed, HFm:463, 517); `-e judge -e std --no-deps`, `--link-mode copy`.
- fp32 on the CPU: HF's bf16 rounding points (bf16 streams, β's sigmoid, the bf16 MoE combine, G-A §9) are no-ops,
  so Holon's fp32-residual form (LC's, Maya's) matches.
- **Documented deviations from HF**, recorded in every fixture's metadata: the indexer's `Tensor.topk` (HFm:889) and
  the router's `torch.topk` (HFm:168, 177) are wrapped per module instance (the router's inside its forward, A-m2)
  with a stable descending sort (lower pool / lower expert wins).
- **Random init hides conventions** (HFm:1381-1421): the fixture randomises `A_log`, `dt_bias`, the selection bias,
  hc `base` / `scale`, the pooling gate and APE, every norm weight and LayerNorm bias, and the pad row (pad = EOS =
  154820); it asserts each is non-default. **The comb logits are scaled so that ≥ half the rows are peaked
  (|s₂·m| ≥ 8, rows colliding on one destination)** (C-M8), and the expert weights so that the clamp binds on ≥ 5% of
  rows of every path (routed, shared, dense, both sides of `clamp(u)`) (C-m6).
- **Tiny MLA dims are distinct** (kv_lora 48, nope 32, v 24), so a transposed layout fails on shape (C-m7).
- **No MTP oracle is needed while MTP is deferred** (D7). If N2 says build, the reference is transcribed from LC's
  `graph_mtp` (`glm5-next.cpp:548-675`: e then h, h post-final-norm, fresh selection per call) and Holon matches LC.
- **Tiny fixture tool:** `testing/holon_testing/glm5next_files.py` (FN's `qwen4exp_files.py` pattern): the random HF
  model, LC's converter at `27c7a76` with the real `tokenizer.json` (G-A §10), pinned sha256; the GGUF is rebuilt,
  not committed; CI skips only the GGUF-dependent checks when LC's converter is absent.

**D3 — mHC: siblings in `std.hyper`, fp32 throughout, the in-block fusion in the contract.**
- The residual is `Residual(16384, "float32")` through C1, exactly as FN's (G-C §1); 64 KiB per token.
- Units: `std.hyper.mhc_block@1(n, mixer, ffn, norms)` — the attention half's mix → RMSNorm → mixer → write-back;
  the FFN half's mix → RMSNorm → FFN → write-back — **declares no state** (the core binds the mixer's state from
  `LayerDecl.mixer`, `assembly.py:140-178`; A-m4, B-m1); its phases are named `mix` / `writeback`, not `pre`
  (`gr_block`'s PLE parameter). `std.hyper.mean@1(inner)` is the collapse (the stream mean, then `inner`, the model's
  RMSNorm), used as `ModelSpec.norm`. `std.hyper.expand@1` is reused unchanged.
- **Contracts** (B-M4, A-m5):
  - `hc_mhc_mix@1(R [T, n·H] f32, fn [24, n·H] f32, base [24], scale [3], y_prev?, post_prev?, comb_prev?) → (x [T, H]
    of dtype, post [T, n] f32, comb [T, n, n] f32)`. With `y_prev` it first applies the attention half's pending
    write-back to R (in f32), **only within one block**; then the unweighted RMS, the 24-wide product summed in a
    fixed order, the affine maps, the Sinkhorn of HFm:316-325 exactly (1 softmax, 20 + 19 normalizations), the
    weighted collapse rounded once. R is complete at every block boundary (`std/holon_std/hyper/kernels.py:6-10`).
  - `hc_mhc_writeback@1(R, y [T, H], post, comb) → R'` with `R'_d = post_d · y + Σ_s comb[s, d] · R_s` in f32: the
    block's last write-back.
- Precision: the 24-wide product `ieee` by default; `bf16x3` / `tf32x3` only within L0's fp32 budget, recorded as a
  kernel fact (FN D10).
- Plant `mhc_sinkhorn_short` (19 iterations) must fail L0 (peaked inputs), the tiny tests and the cover (C-M8).

**D4 — KDA: `unit.std.mixers.kda@1` and `op.kda@1` inside the `std.mixers.gdn` package** (B-M1: one family, one
package, as `std.spec.mtp_head` ships two heads), sharing GDN's `Planner`, state specs, replay buffers, in-step conv
publishes and rewind flow; kernels vendored into `kernels.fla` from the FLA pin we already carry.
- State: GDN's two `Slot` specs — the recurrent `[64, 128, 128]` f32 and the conv windows of q, k, v (3 × 8,192
  channels × 3) **in the model dtype**, as GDN's (exact: the windows hold bf16 projections; B-m6) —
  `alignment = in_step = 64`, replay rewind, checkpoints through `std.state.recurrent`.
- Conv weights stay F32 as stored (HF keeps them f32, HFm:1378): `causal_conv1d@1`'s contract states that weights
  are used in their stored dtype; L0 gains an f32-weight case (B-m6).
- Op `kda@1`: `(q, k, v, f [T, H·Dk], β [T, H], A_log [H], dt_bias [H·Dk])`, fact `lower_bound` (−5); chunked and
  recurrent paths, in-step snapshots, per-token states as GDN's.
- Kernels: `kernels.fla`'s `vendor.sh` adds `ops/kda.py` and `fused_recurrent.py` at vLLM `6dd813e0`
  (`fused_recurrent_kda` :109, `chunk_kda` :1507, `fused_kda_gate_chunk_cumsum` :1301, `lower_bound` :1281);
  `vendor.patch` drops their vLLM imports. FN's `SUB_TOKENS` packing and in-step snapshots apply unchanged.
- **Rewind** (A-m3, B-m7, C-m5): no new flag. `gdn_rewind@1` is generalised to a per-row gate width (`a` [L, T', W],
  W = Hv for GDN, H·Dk for KDA); a masked KDA row sets `f = −inf` (so `g = −5 · σ(−inf) = −0` exactly, for any
  `A_log`) and `β = 0`. Tests: a masked row leaves the state bitwise unchanged on the chunked path (FLA's 16-row
  sub-chunks) and the recurrent path.
- **o_norm:** `gated_rmsnorm@1` gains a fact `round = "none"` (passed only for KDA): the norm stays f32 through the
  weight and gate (HFm:385-397), where Qwen's convention rounds before the weight (A-m7).
- β is the fp32 sigmoid (LC, Maya, FT).

**D5 — DSA: `unit.std.mixers.dsa_mla@1`, `op.dsa_index@1` and `op.mla_attend@1` inside the `std.mixers.qsa` package**
(B-M1), reusing `IndexPlanner`, `dense_limit` / `width` and the model-only path; one mixer because the indexer's
query reads MLA's `q_r`, so the indexer cannot wrap the attention as QSA's does (B, verified clean).
- **Specs:** `"kv"` — the latent, `[page, 512]` in the model dtype, **one copy read as both K and V**; `"idx"` — per
  page the raw keys and gates γ (model dtype, 128 each) and the pooled keys (f32, 128 per 4 tokens) (A-m1): 640 B per
  token per layer on the GPU. Two groups, as QSA's. `page_size % 4 == 0`.
- **Pooling** (`dsa_index@1`): a pool's key is computed when its 4th token is appended, from the **stored** raw keys
  and gates (rewind and prefix hits stay stateless, FN D6); a pool completed by rejected drafts is rewritten.
- **Selection:** `qsa_select@1` gains an optional per-row head-weight input `w [T, Hi]` (fact `head_weights`, default
  off so FN's facts do not move) and Hi up to 32. fp32 scores, IEEE products, the exact top-k with lower-pool ties,
  shape-invariant, sub-batched workspace (FN D6). In prefill a `bf16x3` tensor-core split is allowed within the fp32
  L0 budget (C-m3; FN's HC precedent).
- **Attention** (`mla_attend@1`): absorbed MQA over the latent for prefill, decode and verify, over the selected pools
  + tail; `W_k` / `W_v` come split in the GGUF (`attn_k_b`, `attn_v_b`; dequantized from Q6_K at load, +0.27 GB as
  BF16, counted in §7). Decode's `q̃` is f32 (64 × 512 a token: free); prefill's `q̃` precision is set by L0 against
  the f32 reference (fp16 with a per-head power-of-two scale, Maya's form, or bf16; budget rel L2 ≤ 2e-4; C-m2).
- **The dense path is the sparse kernel over every pool** (Maya's `DSA_ALL`), so a row's attention bits do not depend
  on co-scheduled rows; the planner's `dense` only skips scoring. The tiny test asserts it (one row alone vs beside a
  long row; B-m13). N5 may replace it only with a kernel ≥ 10% faster on the 5090's prefill (it times the
  non-absorbed form too, C-m2), and then this property is dropped and recorded.
- **Kernels** extend `kernels.qsa_triton` (`heads ≤ 32`, `head_weights`, `dsa_index`, `mla_attend`); no new kernels
  package copies its `_score` / `_topk` (B-m3, lead call 8).

**D6 — MoE: two generic contract minors, facts passed only when not the default; the router is a real kernel change
(A-M1, lead call 5).**
- **Router contract:** `moe_topk@1` and `moe_router@1` gain the facts `scoring` (`softmax` | `sigmoid`), `bias` (an
  `[E]` f32 parameter used for selection only), `scale`; `std.moe@1` gains the bias parameter. The reference
  normalises as ggml computes it, `× 1 / max(Σ, 6.1e-5)` (≤ 1 ulp from HF's `/(Σ + 1e-20)`; A-m8).
- **Router kernels** (`kernels.ggml_cuda`):
  - `holon_topk_select` (inside `holon_moe_router`, `topk-moe.cu:468-535`) gains the sigmoid + bias + scale path,
    and `holon_moe_router` takes the bias pointer and scale; `HOLON_ROUTER_CASE(288)`;
  - **the prompt-chunk router GEMM takes E = 288 and stays row-invariant**: BN = 32 / 96 tiles and `router_chains`
    returning a divisor of hidden / 32 (`plugin.py:686`, `csrc/api.cu:127-129` reject 288 today). No fallback to
    `F.linear` (cuBLAS, not row-invariant), so `pass_rows` keeps its value;
  - `_supports_moe_topk` admits 288 (`plugin.py:764-766`);
  - L0 on both paths (decode `rows`, prompt chunks) with `scoring=sigmoid`, bitwise against the reference.
- **Clamp:** `linear_experts@2`, `linear_experts_arena@1` and `silu_mul@1` gain `glu_limit`: `silu(min(g, L)) ·
  clamp(u, −L, L)`. ggml's `GGML_GLU_OP_SWIGLU_CLAMP` is vendored in mmvq (`mmvq.cu:863, 1043`); the fused-SwiGLU mmq
  store and `kernels.ggml_cpu` take the same form; L0 records that ggml's `fminf` / `fmaxf` drop NaN where HF's clamp
  propagates it (C-m6).
- The shared expert composes as `std.ffn.sum@1(moe, swiglu)` (FN D7), ungated, after the routed gate/up (the
  prepared-q8_1 rule, `docs/v1-offload.md` §4.7).

**D7 — MTP: deferred unless step 0 predicts ≥ 1.0× (lead call 6, B-m11).**
- Maya loads no MTP on one NVIDIA GPU (`glm_mtp.cu:22-24, 78-98`). The analytic estimate at 26% cached: a 4-row verify
  at ≈ 0.75 acceptance ≈ 0.9× plain (C-m10). N2 replays the verify windows on Maya's traces (`replay_glm.py
  --windows`); **build only if it predicts ≥ 1.0× on the mode (i) rows**. The schedule plans on the deferral.
- If built: `std.spec.mtp_head@1` with GLM's `blk.45` (no MTP pack); **each draft sub-step selects its own pools**
  (no `index_share_for_mtp_iteration`: no core carrier exists for a sub-step index, B-M5; drafts are verified, only
  acceptance moves; recorded as a deviation from VL `common/mtp.py:147-149`; A-M5); `drafters/mtp` unwraps the embed
  when `target.spec.residual is not None` (B-m2); `MTPHead.write_kv` with `PreNorm.write_kv` (a `std.blocks` edit)
  and `dsa_mla.write_kv` (A-m6, B-m2); experts as `aux.*` through D8(d).

**D8 — Offload: X's store; plan §10's std minors in step 1, in this order (B-M9, lead call 7), each tested with RAM
capped before GLM uses it.** The store gains **no new options** (B-m8); provider facts live in their kernel plugins.
- **(b)+(e) as one change — lending and the prefill ring under exclusive tiers, and the staging that follows.**
  Deleting `or self.exclusive` (`store.py:240`) and `not self.exclusive` (`:435`) needs: a lent piece is **demoted
  first** — to a free RAM unit, else over RAM's coldest expert when hotter (Maya's `LEND_PARK_LRU`, −40% disk reads
  after a prompt, A-m11), else SSD-only (the file is always valid) — before its slot is overwritten; staging per
  class becomes `k × max_seqs × (1 + drafts)` (8 at GLM's lock), returning **2.94 GB of VRAM and as much pinned host
  memory** in mode (ii). Pipelined parts stay as in the RAM-holds-all layout. Tested on Qwen3-30B-A3B with `moe.ram`
  capped (`exact.py`: capped vs uncapped identical) and the plant `exclusive_lent_not_demoted`, forced to bite by a
  RAM tier small enough that the stale unit is reused (A-m10).
- **(a) F10 with RAM short — the full requirement set** (A-M2, B-M2):
  - a **heat**: `placement.promote@1` exposes its recency (`last`) as an f32 heat, so `moe_tiers@1` can rank;
  - a **victim rank** for the spare refill: the layer's LRU resident (Maya's rule);
  - **refill evictions go through `moe_tiers@1`**: each victim is demoted D2H into a free RAM unit, else over RAM's
    coldest when hotter, else SSD-only, in the graph, before the spare's next promotion;
  - `kernels.promote`'s CUDA resolve **emits the delta record** "push b's slots onto the layer's empty stack", replayed
    by `ResidencyMap.commit` outside borrowing steps;
  - `LOG` (`store.py:56`) sized for the records;
  - plants: `promote_spare_unlogged` and a victim tier-code plant.
- **(c) The high-priority split fetch under exclusive tiers** (C-m1): two fetch launches — the RAM misses at once at
  high priority, the SSD misses after their read completes — so a RAM miss never waits behind an SSD read.
- **(f) The SSD path, rebuilt for decode and prefill** (C-B2, lead call 2) — D8f below.
- **(d) Aux experts under exclusive tiers — only if D7 builds MTP:** the residency, piece and owner tables move to
  `before_materialize`; **the `cpu_task` I/O worker and each group's read binding stay at `resolver()`**, in the
  build, since a shared object made after the seal raises (`core/holon/core/build.py:125-129`; B-M3). Tested on FN's
  tiny MTP fixture with `moe.ram` capped (Qwen3-30B has no aux model).
- **(g) The CPU split (F5b)** stays reported only, decided at step 0's end (R3).
- **Placement defaults:** mode (i) `lfu` (F11); mode (ii) the least modelled time among F11 with deferred admission
  and F10 spares, from N2's replay at each engine's own cache share (C-M1), confirmed by step 6's run.
- **The seed** is a GLM routing profile from `route_dump` on held-out prompts (FN D12f), shipped in the bundle.

**D8f — The SSD read path: io_uring from one submitting thread** (C-B2, lead call 2).
- **Where:** `kernels.cpu_task`'s I/O worker (the provider behind `std.tiered`'s file tier, `std/holon_std/tiered/
  gguf_direct.py`). The existing pread worker threads are replaced by **one submitting thread** that drives an
  io_uring (raw syscalls, no liburing dependency) at **queue depth 16–32**. No thread pool: FN's 8 pread threads cost
  decode −32% by taking the engine's cores (FN §22.1).
- **What it does per layer call:** every SSD miss of the call is submitted at once — each expert's three extents as
  4 KiB-aligned chunks (an aligned superset read into a pinned landing unit, no bounce copy where the extent allows)
  — and completions set mapped flags the fetch launch waits on. RAM misses go first in their own launch (D8c).
- **Target:** ≤ 1.3 ms per 7.1 MB expert (≥ 5.5 GB/s) in the engine on the bar node. **Proven first by a microbench
  on the bar node's NVMe at step 0 (N3)**, with the container's io_uring availability checked; if io_uring is
  unavailable there, N3 measures chunked `pread` from the one I/O thread at the same depth via `preadv2` with
  `RWF_NOWAIT` fallback, and the lead gets the rate before step 1.
- **Prefill read-ahead** (A-M3, M4, lead call 2): the next layer's routed set cannot be known a layer ahead, so rev
  1's "routed read-ahead" is dropped. Maya's rule is adopted **if N2's replay shows it pays**: for steps ≥ 1,024 tokens,
  read the next MoE layer's **whole** SSD-only set ahead, ≥ 2 layers deep (C-m12; ≈ 0.5 GB of pinned landing units,
  counted under mode (ii)'s limit); below 1,024 tokens, each layer reads its own routed SSD-only set once routed.
- **Core:** none. The contract (`cpu_task@1`'s read task) is unchanged; the depth is a `kernels.cpu_task` fact.

**D9 — Prefill: large steps with sub-batched activations** (C-B3, lead calls 3–4).
- **Why:** N0 measured Maya's 32K prompt at 3,195 tok/s, read as 2 × 16K chunks that are compute-bound (≈ 0.29 ms per
  token per chunk; its copy lane is busy 4.3 s of a 4.7 s 16K chunk). A 4K-step Holon pays the ≈ 68 GB stream eight
  times; from 8K steps the stream hides under compute (§7.4).
- **Target: 16K steps in both modes with a reserve near a 4K step's.** Only R (`[T, 16384]` f32, 64 KiB per token) and
  the block's x / y (8 KiB each) span the step; everything else runs in row sub-batches of ≤ 2,048 rows: MLA's `q̃`
  and attention (queries sub-batched over all keys), the indexer's scores (FN's 512-query passes), the dense FFN and
  the shared expert, KDA through FLA's `SUB_TOKENS` packing, the MoE through `pass_rows`. Live set ≈ 80–150 KiB per
  token: **≈ 1.3–2.4 GB at 16K**, against ≈ 1.1 GB at a 4K step today; the excess is lent from the arena's tail
  (`moe.lend_peak`, FN §21.12) and returns after the prompt.
- **FN's lesson (§22.9):** every sub-batched op keeps the reserve true for a step holding several prompts' rows;
  `test_reserve.py`'s tiny-GLM case covers it from step 4 (acceptance 3).
- Sub-batching changes no bits relative to the same step size (row-independent ops in fixed call shapes); identity
  across step sizes is not claimed (acceptance 2).
- **1K prompts: stage the routed experts, not the ring** (C-M4, lead call 4): steps below 2,048 tokens use FN's ring of
  parts (`moe.stream_tokens = 2048`, `moe.pipeline = 8`), which moves only routed misses. N0: a 1K prompt routes 89% of
  all experts (Maya's log: 8,438 staged + 2,338 resident), so the saving is ≈ 10% of the bytes, and more in mode (ii),
  where the SSD-only experts are the coldest.
- **N4 measures** the prefill compute per token (MMQ on IQ2_XXS / IQ2_S / IQ3_XXS / Q6_K at 4K–16K rows, KDA, MLA,
  indexer) and the activation live set at 4K / 8K / 16K / 32K steps, before §7.4's prefill rows freeze.
- **Prefill tail skip** (C-m4): Maya skips the last layer's output projection and FFN for rows nothing reads (2.4% of
  PCIe- or SSD-bound prefill). `StepCtx.logits_index` (`core/holon/api/batch.py:151`) names the rows whose logits
  are computed; the block can skip the rest when the step has no prompt-logprob rows and no drafter. Built in step 4
  if that holds without a core line; else escalated (acceptance 6). Counted as Maya's edge in §7.4 until built.

**D10 — The two host modes** (lead call 10, B-M6, C-M9).
- **(i) RAM holds every expert:** the bar desktop pins all 86.03 GB of routed experts plus the process; F11 on; no SSD
  tier. (N0: Maya pinned a 65.89 GB RAM tier beside its 22.25 GB pool; every expert held.)
- **(ii) 64 GB:** each engine runs in its own job step with a 64 GiB memory limit: the step's memory cgroup is the
  limit (pinned host memory counted against it, checked at N1). **The RAM tier is equal in experts**: N1 reads Maya's
  RAM-tier slot count at `MAYA_GLM_RAM_GB` set to fit under the limit (`glm fast: … RAM tier … slots`); Holon's
  `moe.ram` is set to the same number of experts, per class proportionally. Each engine keeps its own VRAM cache, so
  Holon's smaller cache shows as more SSD-only experts — its own cost.
- **Page-cache hygiene:** after the GGUF is copied to the node's NVMe and before every mode (ii) arm, the file's pages
  are evicted (`posix_fadvise(POSIX_FADV_DONTNEED)` per shard); each arm checks `O_DIRECT` on the engine's shard fds
  (`/proc/<pid>/fdinfo` flags), records the device's read bytes (`/sys/block/<nvme>/stat`) against each engine's own
  read counter, each engine's SSD-only expert count and `AnonHugePages`.
- **The pre-registered mode (ii) override set** on the 5090 lock, one named configuration in the evidence: `moe.tiers`
  with the file tier; `moe.ram` (experts per class, as above); `--host-memory`; the placement policy (D8's rule);
  `max_step_tokens` (D9's measured value); `kernels.cpu_task`'s queue depth (N3). Nothing else differs from the lock.

**D11 — The comparison protocol: FN's D11 as closed, with the run as the unit** (C-M10, lead call 11).
- **The Maya backend** (`bench/offload_ctx.py`, built and checked at N0): the engine's stdin protocol with raw ids
  (`GEN <n> temperature=0 <ids>` → `T <id>` … `DONE …`, `generate.cpp:2063, 2213-2222, 7005-7041`); `--glm-pack` from
  `tools/iq_pack.py --compat-bf16`; `--max-context 33792 --kv-resident 33792` (a 32K configuration never streams);
  `--eos-ids 2147483647` (N0: `produced` = n on every request); `MAYA_GLM_SLOTS=0`; loop guards off; its setup's
  defaults (`--expert-cache auto --prefill auto --kv-streaming on --expert-profile data/expert-profile-glm.bin`) and
  **its own `--calibrate` settings** for the card (Q11, recorded). Built at `4615f34`, sm_120, on the bench host.
- **Maya's checkpoint cut:** a 32K raw-id prompt is read as 16,384 + 16,384 by default; the 32K cells take **the
  faster of** Maya's default and `MAYA_GLM_CKPT_EVERY=0` arms (Q5).
- **Holon:** its lock's configuration (mode (ii): the override set), `--ctx 33792`, `ignore_eos`; 0 reused tokens per
  timed request (both engines report reuse).
- **Corpus and spans:** FN's text corpus tokenized with GLM's tokenizer (184,531 tokens); 8 spans per context (FN's
  six starts and two midpoints), the warm-up span excluded; all 24 requests in one fresh process per run.
- **Statistics** (fixed now):
  - **the run's statistic**, per context and metric: total tokens over total time across its 8 prompts (decode:
    Σ(n − 1) / Σ(t_last − t_first); prefill: Σ C / Σ prompt time), each engine timed from the client;
  - **runs** alternate in ABBA order within an allocation, a discarded warm-up run first;
  - **the cell:** the median over adjacent (Holon, Maya) run pairs of the per-pair ratio;
  - **rounds:** at least 3 (6 pairs); more if N1's measured Maya run-to-run CV needs them for a ratio SE ≤ 1%;
  - **pass** ≥ bar; **a cell within 3% of its bar is re-run on a second allocation** (another node), the two medians
    pooled; below the bar → the owner.
- **Per cell:** expert hit rates (all, and the first 32 decode tokens after a prompt), SSD reads per token and device
  read bytes (mode (ii)), Maya's `glm stat:` and `glm prefill:` lines, peak VRAM and RSS.

**D12 — Kernels: Triton in the existing kernel plugins; CUDA only by FN D10's rule.** `kernels.qsa_triton` (selection
with weights, `dsa_index`, `mla_attend`), `kernels.hc_triton` (the mHC pair: decode split-K over 16,384 inputs; the
prefill product in fixed-order passes), `kernels.fla` (KDA), `kernels.ggml_cuda` (router and clamp),
`kernels.cpu_task` (D8f). Adapted code from FreeToken (Apache-2.0) is declared per AGENTS.md. A kernel moves to CUDA
in step 7 if it is over 5% of a step and a prototype gains ≥ 10%.

**D13 — Tokenizer and frontend.** `std.weights.gguf`'s `_PRE` gains `glm4` only (B-m10): BPE with
`ignore_merges = True` (llama.cpp's CHATGLM4), no normalizer, GLM's split regex (`\p{N}{1,3}`); `eom`
(`<|observation|>`) read as a stop token. `glm_xml@1` (`<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>…`,
My `serve/frontend.py:8-10`) is checked by a real-client smoke with a tool call at every merge (skill §4a). The
template opens `<think>`, so `think@1` is checked on a prompt that ends in it.

**D14 — Text only.**

**D15 — Locks and propagation.**
- **New:** `rtx-5090` (Maya-S, mode (i), `max_seqs` 1) and `rtx-5080` (mode (i); the development lock class, where
  GLM's perf iteration runs; B-m17). Mode (ii) is the 5090 lock plus D10's pre-registered override set, one named
  configuration recorded in the evidence; no lock is keyed on a host's RAM (Q10).
- **Shared code** is listed by `holon_tools impact <base>..HEAD` at every step close (B-m5), not by hand. Expected:
  `std.moe`, `std.ffn`, `std.tiered`, `std.hyper`, `std.mixers.gdn`, `std.mixers.qsa`, `std.weights.gguf` (every
  GGUF lock), `std.blocks` (if MTP), `kernels.ggml_cuda`, `kernels.ggml_cpu`, `kernels.fla`, `kernels.qsa_triton`,
  `kernels.hc_triton`, `kernels.expert_io`, `kernels.cpu_task`, `placement.promote`, `kernels.promote`,
  `frontends.http`. New facts are passed only when not the default, so traces should advance byte-identical; one
  propagation at the close (release §22).
- **Merge order** (B-m14): step 1's `std.tiered` / `std.moe/offload.py` changes merge into the branch first; steps 3–4
  rebase on them, and `std.moe/units.py` and `offload.py` are edited by one step at a time.

**D16 — Accuracy is the owner's method (FN §18), from the start, and again on the shipped tree** (acceptance 1).

**D17 — K4 (KV streaming of the latent): proposed deferral, Q12.** At ≤ 32K the latent and the indexer pages are
0.6 GB (§7) and Maya does not stream a 32K configuration; the bars stop at 32K, as FN's did.

## 3. What Holon has today (G-C)

| Piece | State |
|---|---|
| Core | 5,019 / 6,000 code lines, ABI 1.5; denylist names `mla`, `dsa`, `kda`, `qsa`, `indexer`, `mtp`, `moe`, not `glm`, `mhc`, `sinkhorn`, `nextn` |
| Residual and state seams | C1 carries `[T, 16384]` f32 unchanged; KDA's Slots are GDN's; the latent and `idx` are two paged groups as QSA's; aux paged state is allowed (`assembly.py:108-136`) |
| `std.hyper` | FN's gated-residual form only (no stream mixing); `expand@1` reusable |
| Attention | GQA, QSA (`qsa_select` ≤ 16 heads, no head weights; `qsa_attend` GQA); no MLA anywhere |
| Linear attention | GDN through vendored FLA at vLLM `6dd813e0` (no KDA files) |
| MoE | softmax routers only; the fused router and its prompt GEMM refuse E = 288; no clamp anywhere; every Maya-S type in `TYPES` |
| Store | exclusive RAM tier + F9 SSD (2 pread threads, inline waits) + F11 deferred admission work; no lending / ring / split under exclusive tiers; F10 refused under exclusive tiers (no heat, evictions outside `moe_tiers`); aux experts raise under exclusive tiers |
| GGUF | split files; gguf-py at LC `27c7a76` knows `glm5-next`; tokenizer `qwen2` / `qwen35` only |
| Judge and bench | `spikes/fn7/{consistency,cover,exact,ledger,lend_min,admit_sim}.py`, the 72-item task set; `offload_ctx.py` with a Maya backend and the `glm` corpus (N0); `spikes/glm0` (Maya route traces → M6 replay) |

## 4. Core seams: none planned

| # | Need | Today → change | Why no core line |
|---|---|---|---|
| — | mHC's 4-stream residual | C1 `ModelSpec.residual` → unchanged: `Residual(16384, "float32")` | the profile builds the drafter's input at the residual width (`executor.py:246-249, 283`) |
| — | KDA state | mixer `DeclaresState` with two `Slot` specs → unchanged | GDN's shape |
| — | Latent + indexer pages | two `Paged` groups → unchanged | QSA's layout (FN D6) |
| — | Large prefill steps | `ActivationLender` → unchanged | the store lends once D8(b) lets an exclusive layout lend |
| — | SSD reads | `cpu_task@1` → unchanged | io_uring lives in the kernel plugin (D8f) |
| — | MTP (if built) | aux model, aux paged state, store `aux.*` group → unchanged | fresh selection per sub-step needs no sub-step index (D7) |
| — | Tail skip | `StepCtx.logits_index` → unchanged | to be confirmed at step 4; else escalated (D9) |

- **Generality:** `check_core_denylist.py` gains `glm|mhc|sinkhorn|nextn` (a CI change, not core).
- **Import contracts** (B-m4): the new modules live in existing packages; step 3 checks the import-linter contracts
  (`pyproject.toml:100-115`) with them.

**Std and plugin contract minors** (additive; `std/contracts.txt` / `plugins/contracts.txt` regenerated):

| Contract or unit | Today → change |
|---|---|
| `moe_topk@1`, `moe_router@1` | softmax, power-of-two E → facts `scoring`, `bias`, `scale`; E = 288 in both router paths (D6) |
| `linear_experts@2`, `linear_experts_arena@1`, `silu_mul@1` | no clamp → fact `glu_limit` (D6) |
| `qsa_select@1` | `Σ_h relu(q_h·K̄)`, ≤ 16 heads → optional `w [T, Hi]` (fact `head_weights`), Hi ≤ 32 (D5) |
| `gdn_rewind@1` | gate width Hv → any per-row width (D4) |
| `gated_rmsnorm@1` | rounds before the weight → fact `round = "none"` (D4) |
| `causal_conv1d@1` | weight dtype unstated → weights in their stored dtype (D4) |
| new `kda@1` (gdn), `dsa_index@1`, `mla_attend@1` (qsa), `hc_mhc_mix@1`, `hc_mhc_writeback@1` (hyper) | — |
| `moe_tiers@1`, `ResidencyMap`, the store, `placement.promote`, `kernels.promote` | D8 (a)–(d) |
| `kernels.cpu_task` | io_uring I/O thread, depth fact (D8f) |
| `std.weights.gguf` | `glm4` pre-tokenizer, `eom` (D13) |

## 5. Units, contracts, model plugin

| Unit | Package | Role |
|---|---|---|
| `std.hyper.mhc_block@1`, `std.hyper.mean@1` | `std.hyper` | D3 |
| `std.mixers.kda@1` | `std.mixers.gdn` | D4 |
| `std.mixers.dsa_mla@1` | `std.mixers.qsa` | D5 |
| `std.moe@1` (+ bias), `std.ffn.sum@1`, `std.ffn.swiglu@1` (+ limit) | `std.moe`, `std.ffn` | D6 |
| `std.spec.mtp_head@1` | `std.spec.mtp_head` | D7, if built |

**Contracts** (each with `reference.py` and a CPU provider in `testing/holon_testing/ref_kernels`): `hc_mhc_mix@1`,
`hc_mhc_writeback@1`, `kda@1` (with `gdn_rewind@1` generalised), `dsa_index@1`, `qsa_select@1` (weights),
`mla_attend@1` (prefill, decode, verify), the router and clamp facts.

**`models.glm5_next`**, written fresh (no `derived_from`: it shares no names, HC, PLE or rules with `qwen4_exp`;
B-m9): the recipe reads `cfg["gguf"]` (`glm5-next.*` keys; `head_count_kv` per layer and `key_length` = 512 read raw,
G-C §8), sets `residual = Residual(16384, "float32")`, `embed = std.hyper.expand@1(inner = std.tiered.rows@1)` (the
token table on the host, 0.52 GB of VRAM back), `norm = std.hyper.mean@1(rmsnorm)`, per layer `LayerDecl(kda |
dsa_mla, swiglu(limit) | sum(moe(sigmoid, bias, 2.5, limit), swiglu(limit)))` inside `mhc_block`; GGUF rules per
LC's names (`glm5-next.cpp:61-184`; `ssm_a` = −exp(A_log) from the converter); the bundle (chat template,
`glm_xml@1`, `think@1`).

## 6. Numerics and judge (thresholds and predicted verdicts fixed now)

### 6.1 Tiny model: CPU (fp32 vs HF) and GPU (vs the CPU engine)

**Config:** 8 layers (DSA at 3 and 7; dense FFN at 0; KDA elsewhere), H = 64, n = 4 streams, KDA 2 heads × 32,
MLA 2 heads (q_lora 32, kv_lora 48, nope 32, v 24), indexer 4 heads × 16 with kpool 4 and top-k 8 (sparse from
t + 1 > 11), 8 experts top-2, I = 64, shared expert, swiglu_limit 10 with weights scaled so the clamp binds on ≥ 5% of
every path's rows, peaked comb logits on ≥ half the rows, vocab 154,880 (the real tokenizer). Prompts: 100 tokens
with an EOS, chunked at 7 and 13, then 24 decode steps.

| Check | Where | Threshold | Predicted |
|---|---|---|---|
| logits (the GGUF's head), one-shot and chunked, vs HF | CPU | max \|Δ\| ≤ 1e-4, top-1 identical | pass |
| DSA token sets per (layer, position) | CPU | identical under the lower-pool rule; ties asserted | pass |
| prefix hit vs recompute | CPU | bitwise | pass |
| one row alone vs beside a long row (the dense path is the sparse kernel) | CPU, GPU | bitwise | pass |
| clamp bind rate per path | CPU | ≥ 5% | pass |
| tokenizer vs `llama-tokenize` (LC `27c7a76`) on a multilingual round-trip corpus | CPU | identical ids | pass |
| the tiny GGUF through the GPU providers (graphs and eager) vs the CPU engine: logits, decode and chunked | GPU (A5000) | max \|Δ\| ≤ 3e-2 and top-1 identical on ≥ 99% of positions (a bf16 bound, set from the clean run ×3 before any plant) | pass |

### 6.2 Full model (Maya-S)

| Check | Where | Predicted |
|---|---|---|
| (c) decode vs prefill; chunked (250) vs one-shot: KL mean ≤ 0.13, p50 ≤ 0.007 | 5090 deployment, step 5 and step 10 | pass (FN: 0.025–0.031 / 0.002–0.003) |
| (d) 72 items vs Maya, greedy, thinking on, 16,384 new tokens at most, the template's default effort | both engines' 5090 mode (i) deployments | pass (within 5 items) |
| (f) the cover (acceptance 1f). Bounds FN's way: each set at ≈ 10× the clean run's maximum and ≥ 10× below its plant's effect, both measured before the bound is written | RTX PRO 6000 and 5090 | pass |
| exactness (acceptance 2) at the lock's step | RTX PRO 6000; 5090; 5080 | byte-identical |
| the clamp's bind rate on the bench prompts | reported | — |
| zero-score ties at the selection boundary, per 1,000 rows | reported | — |

### 6.3 Plants (each changes only its target path; "everywhere" means every provider but not `reference.py`)

| Plant | Breaks | Run by | Predicted |
|---|---|---|---|
| `dsa_decode_select_shift` | decode and verify selection shifted one pool (b → b + 1, FN's form) | (c), (a′) | **(c) fails on its p50** (pre-registered statistic: ≈ 0.01–0.015 against 0.007; FN's analogue failed p50 at 0.013 and passed the mean) and (a′) fails; the CPU tiny test cannot run it |
| `kda_gate_softplus` | the recurrent (decode) KDA kernel uses the unbounded softplus gate | (c), L0, (a′) | **(c) fails** (pre-registered); L0 and (a′) fail |
| `kda_state_not_carried` | a chunked prefill step starts from a zero state, everywhere | tiny CPU, (c), cover | tiny chunked fails; (c) chunked vs one-shot fails; cover fails |
| `mhc_comb_transposed` | `comb` applied as C, not Cᵀ, in every provider (decode and prefill) | tiny CPU (through the CPU provider), cover, L0 | all three fail; (c) passes (both paths planted) |
| `mhc_sinkhorn_short` | 19 Sinkhorn iterations, everywhere | L0 (peaked), tiny CPU, cover | all three fail; (c) passes |
| `dsa_pool_ape_off` | APE left out of pooling, everywhere | tiny CPU, cover | both fail; (c) passes |
| `router_bias_in_weights` | the bias added to the weights too, everywhere | tiny CPU, cover (router ids and weights) | both fail; (c) passes |
| `glu_clamp_off` | the mmvq (decode) path without the clamp | L0 (inputs that bind it), (a′) | L0 and (a′) fail; (c) fails only if the bind rate (§6.2) is ≥ 1e-3 of rows, else passes (recorded) |
| `exclusive_lent_not_demoted` | a lent piece's tier code left at RAM | `exact.py` with a small RAM tier that reuses the unit | differs |
| `promote_spare_unlogged` | F10's delta record dropped | the host mirror check, `exact.py` | fails, differs |
| `promote_victim_code` | a refill victim's tier code not moved | `exact.py` RAM-short | differs |
| `aux_row_padding`, `lent-slot-still-mapped` (existing) | as FN / X | `exact.py` | differs |
| `mtp_h_prenorm` (if MTP is built) | the head reads h before the final norm | the LC-transcribed MTP reference | fails; the acceptance drop is reported, not pre-registered (A-m12) |

## 7. Memory fit and predictions

### 7.1 The RTX 5090 (C-M1: the measured total, statics included)

The 5090's measured total is **31.448 GiB** (FN `evidence/fn_close_2026-10-24.json`); rev 1's 31.88 double-counted
FN's activation reserve and left out statics.

| Item | GiB | Note |
|---|---|---|
| outside the allocator | 0.95 | FN's 0.92 + the larger host mapping |
| parameters (non-expert, token table on the host) | 6.89 | 7.13 GB + `attn_k_b` / `attn_v_b` as BF16 (+0.27 GB) |
| statics (FlashInfer workspace, mmq id workspaces) | 0.30 | FN's 5090: 0.40–0.48 GiB with RoPE tables GLM lacks |
| state pools | 0.98 | latent 0.38 GB + `idx` 0.24 GB + KDA 2 slots and 1 checkpoint 0.43 GB |
| graphs' reserve | 0.12 | FN |
| activation reserve | 1.02 | ≈ a 4K step's live set; at 16K steps D9 keeps it ≈ 1.0–1.3 and lends the rest during prompts; N4 measures |
| slack | 0.44 | FN |
| **the arena** | **≈ 20.7 (22.3 GB)** | |
| staging and null slots (D8e) | 0.11 | 3 classes × (8 + 1) |
| **experts cached** | **≈ 3,120–3,150 (25.8–26.0%)** | **Maya measured 3,360 slots / ≈ 3,300 resident (27.3%)** at N0 (`glm fast: … expert pool 22.25 GB … 3360 total`; VRAM 3,270–3,304 in use): Holon caches ≈ 5–7% fewer |

- **Mode (ii):** the same VRAM; RAM tier equal to Maya's in experts (D10), est. ≈ 7,550 at 50 GiB; SSD-only ≈ 12,096 −
  3,120 − 7,550 ≈ 1,430 for Holon against ≈ 12,096 − 3,300 − 7,550 ≈ 1,250 for Maya (+14%).
- **Host, mode (i):** experts 86.03 GB pinned (THP), + the token table 0.52 GB.

### 7.2 The RTX 5080 (16 GB), mode (i)

Visible 15.47 GiB; parameters 6.89, statics 0.2, state 0.98, outside 0.6, activation 0.75 (a 4K step), graphs 0.1,
slack 0.4 → arena ≈ 5.6 GiB → **≈ 840 experts, 6.9%** (C-m11: ≈ 850). Its IQ3_XXS down class (8 layers × ≈ 20 slots)
cannot hold a ring, so the 5080 stages routed misses (the ring of parts) at every step size (C-m11). A reported row
and the development lock (Q4).

### 7.3 The RTX PRO 6000 (96 GB), step 5

≈ 102.6 GB visible less the same items leaves ≈ 91 GB for experts against 86.03 GB: **all-resident fits at
`--memory-fraction` ≥ 0.98** (C-m8), acceptance 2's reference fit. If it does not, the store at its largest fit is the
reference instead.

### 7.4 Predicted Holon / Maya ratios per gated cell (C-B1, lead call 1)

**Inputs.** Maya: N0 (one uncalibrated run, mode (i)); N1's calibrated runs replace these when they land (§8).
Holon: the roofline below with C's corrected inputs (C-M2) and N2's replay. Mode (ii) Maya is not measured yet: its
side is C's model scaled by Maya's mode (i) measured / modelled ratio.

**Reconciling C's 0.87–1.00 with N0.** C's model put Maya's mode (i) token at its first-order roofline with the
lane, ≈ 19–20 ms. **N0 measured 27.5–28.0 ms** (35.9 tok/s at h 0.72 over the bench spans; 30.6–39.4 tok/s by prompt
at h 0.63–0.76, N2). Maya's own counters: 58 RAM fetches and 36 CPU-lane experts per token, the lane busy 11.4 ms
a token; ≈ 7 ms a token is not explained by bytes, PCIe or the lane (its per-route host service handshake and per-layer
lane waits are the likely terms). **The measurement is right for Maya**, so C's Maya side is replaced by it. For
Holon the first-order model stays the prediction: FN's step-0 roofline put the 5090's drafting-off decode at
1.00–1.04× Strata, and FN closed at 1.11–1.17× (FN §13, §22.10). The band's low end allows Holon to pay ≈ 7 ms of
unmodelled cost too.

**The hit rate.** N2's replay of Maya's own bench routing (8,192 decode tokens): Maya's policy (promote-on-fetch,
LRU) at Maya's share (27.8%) gives **0.719 against Maya's live 0.722** — the replay is validated. Holon's `lfu` at
Holon's share (26%) gives **0.720** (`lfu` at 27.8%: 0.737). So Holon's ≈ 6% smaller cache costs ≈ nothing in h
with `lfu` against Maya's LRU. (Indicative: the profile was trained on the test trace; the held-out replay with the
tasks trace replaces it at step 0's end.)

**Holon's mode (i) decode token:** dense 7.13 GB at 1.55 TB/s × 0.8 (Q6_K mmvq) = 5.75 ms; misses 336 × (1 − 0.72) ×
7.11 MB = 0.67 GB at P = 47–49 GB/s = 13.7–14.2 ms, the hits under them; t_fixed 2–3 ms (`k_resolve` 0.32, ≈ 180 mHC
calls 0.6–0.9, 11 selections 0.3, KDA state 0.18, glue 0.6; C-M2): **≈ 21.5–23.0 ms, 43–46.5 tok/s**; + Maya's
unexplained 7 ms at the band's low end: 28.5–30 ms.

| Cell (bar) | Holon | Maya (N0) | Predicted ratio: central [band] | What decides it |
|---|---|---|---|---|
| (i) decode 1K / 4K / 32K (1.0) | 43–46.5 tok/s [33–46.5] | 35.6–35.9 tok/s | **1.22 [0.93–1.30]** | Holon's unmodelled cost; Maya's calibrated lane |
| (i) prefill 1K (0.9) | routed misses ≈ 58 GB at 50–53 GB/s + compute under it: ≈ 1.2–1.35 s | 1.44 s engine, 1.59 s client | **1.2 [0.95–1.35]** | the ring of parts (D9); N4's compute |
| (i) prefill 4K (0.9) | ≈ 68 GB stream 1.3–1.4 s ∥ compute ≈ 1.2 s: 1.45–1.6 s | 1.91 s engine, 2.03 s client | **1.3 [1.0–1.4]** | the same |
| (i) prefill 32K (0.9) at 16K steps | 2 × max(stream ≈ 1.4 s, compute ≈ 4.7 s × c) | 10.25 s (2 × 16K, compute-bound) | **1.0 [0.85–1.2]** | c = Holon's prefill compute per token against Maya's (N4); 4K steps would give 0.85–0.92 |
| (ii) decode (1.0) | the mode (i) token + r SSD reads × 1.3 ms (D8f), r = 2–8 | the mode (i) token + r × 1.2 ms (8 chunk workers) | **1.15 [0.85–1.25]**; 0.83–1.03 if D8f stays at today's 2.8 ms a read | D8f's rate (N3); r (N1, N2) |
| (ii) prefill 1K / 4K (0.9) | + the routed SSD-only set (≈ 1,430 SSD-only, the coldest) at N3's rate | + its ≈ 1,250 at its rate | **1.0 [0.8–1.2]** | Holon's +14% SSD-only experts; both rates; overlap with PCIe |
| (ii) prefill 32K (0.9) at 16K steps | 2 × max(SSD set ≈ 10.2 GB / rate, stream, compute) | its 2 × 16K | **0.95 [0.8–1.15]** | compute-bound if the rate ≥ ≈ 2.5 GB/s; SSD-bound below |

**Cells with a band below their bar go to the lead at step 0's end** (N4's rule), with levers and costs:
- **(ii) decode** — the lever is D8f's rate; if N3's microbench misses 1.3 ms a read: Maya's chunked reads from one
  thread (no pool), or a larger RAM tier for both (not a lever: equal by D10).
- **(ii) prefill 1K / 4K / 32K** — levers: a larger Holon cache (`moe.lend_peak`, the 151 MiB stack trim of FN §23.6,
  `--memory-fraction` 0.99: ≈ +0.6 points) and D8b's demote-into-RAM rule (fewer SSD-only experts after prompts).
- **(i) and (ii) 32K prefill** — levers: 16K steps (D9), prefill compute (step 7), the tail skip (+2.4%).
- **(i) decode at its band's low end** — the lever is R3's exact CPU row, decided at step 0's end, not step 7.

## 8. Step 0 — measure before freezing

On the RTX 5090 desktop (and the 5080 for the reported row), one heavy job at a time.

| # | Measure | Status | Decides |
|---|---|---|---|
| N0 | The harness: Maya built at `4615f34` (sm_120) and its pack; `offload_ctx`'s Maya backend; `produced` = n; `reused` = 0; the corpus retokenized | **done** (one run: decode 35.9 / 35.6 / 35.9, prefill 646 / 2,016 / 3,195 tok/s at 1K / 4K / 32K; pool 3,360; h 0.72–0.74) | D11 runnable |
| N1 | Maya's rows: `--calibrate` (Q11), then mode (i) ≥ 5 rounds × 2 arms (its run-to-run CV), mode (ii) under a 64 GiB job-step memory limit (pinned memory counted; the tier in experts; hygiene), the deterministic recipe; the 5080 mode (i); Maya's ledger lines | **calibrate and mode (i) running** | the denominators; D10's tier; D11's rounds |
| N2 | Maya's route traces: decode (`STRATA_GLM_ROUTE_LOG`) **and prompts (`--dump-routing`)** (C-M5) on the bench plan and the 72 prompts; the replay **of the bench's exact workload** (warm-up, prompts in order, 256-token decodes, each prompt step's lends at the candidate step sizes; FN's `lend_min.py` / `after_prompt.py` rule, FN §21.15) at each engine's own share; F10 vs deferred admission in mode (ii); MTP verify windows; the 1K / 4K routed fraction (ring vs staging) | decode traces done (bench 8,192 tokens; tasks running); prompt traces next | D8's mode (ii) default; D7; D9's 1K rule; h |
| N3 | **The SSD microbench on the bar node's NVMe** (lead call 2): io_uring from one thread at depth 8 / 16 / 32, 7.1 MB experts as 3 extents of aligned chunks, random experts from the GGUF; `pread` from one thread as the fallback; io_uring availability in the container; the drive's free space | to build | D8f's design and rate |
| N4 | The roofline per §7.4 row: dense bytes per component, h from N2, P from FN §23.6, SSD from N3; **the prefill compute probe** (MMQ on IQ2_XXS / IQ2_S / IQ3_XXS / Q6_K at 4K–16K rows, KDA via FLA at 64 × 128, MLA absorbed and not, the indexer) and **the activation live set at 4K / 8K / 16K / 32K** with D9's sub-batching, on the A5000 and the 5090 | to build | every gated row frozen; **a row predicted below its bar goes to the lead before step 1 ends** |
| N5 | (½ day) FlashInfer 0.6.12's fa2 MLA with `head_dim_kpe = 0` on sm_120, and the non-absorbed prefill form: compiles? 32K prefill time against a Triton prototype of `mla_attend` | to build | D5's dense path |
| N6 | Maya's answers to the 72 items on the 5090 (its calibrated defaults, the same rendered ids), saved as gate (d)'s reference | after N1 | gate (d) ready at step 5 |

## 9. Steps

| Step | Work | Checks | Days |
|---|---|---|---|
| 0 | N0–N6 | evidence; §7.4 frozen; escalations | 4 |
| 1 | D8 in order: (b)+(e), then (a), then (c); D8f's io_uring path in `kernels.cpu_task` (after N3) — on the Qwen3-30B-A3B vehicle with RAM capped | `exact.py` capped vs uncapped (A5000, then the 5090 node's NVMe); the vehicle's offload lock trace byte-identical; the D8 plants; core diff empty | 6 |
| 2 | The oracle: `golden-glm`; the fixture tool (tie wrappers, randomised init, peaked combs, distinct MLA dims); the tokenizer round-trip corpus | the HF driver bitwise vs HF's whole forward on the tiny model | 3 |
| 3 | Std units and contracts with references and CPU providers (D3–D6, D13) in their packages; the model plugin; the import-linter contracts | §6.1 CPU rows; Qwen3.8 / Flash-Next snapshots unchanged; core diff empty | 6 |
| 4 | Kernels (D12): router (sigmoid, bias, scale, E = 288 in both paths) and clamp; KDA vendoring + rewind; mHC; `dsa_index`, `qsa_select` weights, `mla_attend`; D9's sub-batching; the tail skip if core-free; the launch inventory; `test_reserve.py`'s tiny-GLM case | acceptance 3; §6.1's GPU row (a′) | 7 |
| 5 | The full model on the RTX PRO 6000 (all-resident) then the 5090: gates (c), (d), (e), (f); exactness; the clamp bind rate and tie counts; `route_dump` profile; Holon's own traces re-check N2 | acceptance 1(b), 2; §6.3's plants | 4 |
| 6 | Offload on the bar cards: mode (i), and mode (ii) with the pre-registered override set; the placement default by N2's rule confirmed; the SSD path on the node's NVMe | acceptance 2 on both cards; fit vs §7 | 4 |
| 7 | Perf closure: per-component breakdown vs Maya (nsys), D9's step sizes, fusions, CUDA rewrites by D12's rule; a per-change perf smoke | every gated row within its band, or the gap explained | 6 |
| 8 | Locks (`tune --new-model`): `rtx-5090`, `rtx-5080` | snapshot; fit vs §7 | 2 |
| 9 | The comparison (D11), a second allocation for any cell within 3% | acceptance 4 | 3 |
| 10 | Close: gates (c), (f) and `exact.py` on the final tree at each lock's step, (d) if a decode kernel changed after step 5; the GPU-sharded suite; panel code review; propagation; plan §10 | acceptance 1 (shipped tree), 5–8 | 3 |

**Schedule:** 48 days of work with MTP deferred (+ ≈ 4 if N2 says build). **Critical path** with two to three agents:
0 ∥ 1 ∥ 2 (6) → 3 (6; step 4's kernels start beside it) → 4 (5) → 5 (4) → 6 (4) → 7 (6) → 8 (2) → 9 (3) → 10 (3) =
**39 days**. Step 1 is independent of the model and merges into the branch first (D15).

## 10. Risks

- **R1 — Mode (ii) decode rests on D8f's rate** (§7.4: 0.83–1.03 at today's 2.8 ms a read, 1.15 at 1.3 ms). N3 proves
  the rate before step 1 builds it; io_uring may be blocked in the container (N3 checks).
- **R2 — 32K prefill** in both modes needs 16K steps (D9) and Holon's prefill compute at Maya's (N4's probe). If the
  loan cannot reach 16K, or compute is > 1.15× Maya's, the rows go to the lead with the curve.
- **R3 — Maya's CPU lane** (36 experts a token at N0). Holon stays exact by default. If (i) decode's predicted band
  after N1 falls below 1.0, the lever — an exact fast CPU row under F5b's bounded contract (Maya's G9) — is decided at
  step 0's end with the lead, not at step 7.
- **R4 — Kernel scale:** MLA's 64 heads × 512 accumulator in Triton on sm_120's ≈ 99 KB of shared memory; the
  indexer's 32 heads in fp32; KDA's 64 × 128 temporaries against the reserve (`test_reserve.py` from step 4).
- **R5 — Holon's smaller cache** (≈ 5–7% fewer experts than Maya's measured pool): neutral in h with `lfu` (N2) in
  mode (i), +14% SSD-only experts in mode (ii).
- **R6 — Oracle gaps:** HF's FLA swap (venv without FLA); exact-zero ties (the wrapper, ties counted).
- **R7 — Maya moves fast** (v1.1.15 → v1.1.27 in 8 days). Pinned at `4615f34`, re-checked at the close; a newer Maya
  that changes the GLM one-GPU path re-runs N1 only with the owner's go.
- **R8 — Host RAM for mode (i):** ≈ 86 GB pinned plus the process (N0: Maya pinned 88 GB of tiers on the bar node).
- **R9 — Shared offload code (step 1)** moves the vehicle's and FN's offload paths: no committed lock uses an exclusive
  tier (`plans/`, panel B), so exclusive-only changes move src pins but no traces; `holon_tools impact` at step 1's
  close.
- **R10 — Variance:** Maya's defaults are timing-dependent (17 of 70 start pairs identical, `SETTINGS.md:244-257`); N1
  measures its CV and D11 sizes the rounds.
- **R11 — Disk:** 275 GB free on scratch after the download; the bar node's NVMe has 2.2 TB free (N0).

**Deferred:** K4 (Q12); MTP unless N2 says build (D7); an exact fast CPU row (R3); Maya-S24 / M / L; DFlash2;
`--parallel` batching rows (`max_seqs` 1 in the locks); the vision tower.

## 11. Lead decisions applied (2026-10-25) and owner questions

**Lead's calls on rev 1's review** (each applied where cited):
1. Per-cell predictions with bands, Maya from measurement (§7.4). 2. The SSD path: io_uring, one thread, depth 16–32,
≤ 1.3 ms a read, proven by N3 first; the routed read-ahead dropped, Maya's whole-set rule if it pays (D8f). 3. 16K
steps with sub-batched activations; N4's compute and activation probes; `test_reserve.py` from day one (D9). 4. 1K
prompts stage routed experts (D9). 5. The router as a real kernel change, row-invariant (D6). 6. MTP selects fresh
per sub-step; deferred unless N2 predicts ≥ 1.0× (D7). 7. F10's full requirement set; step 1 ordered (b)+(e), (a),
(c), with (d) after Q3 (D8). 8. KDA in `std.mixers.gdn`, DSA in `std.mixers.qsa`, `qsa_triton` extended (D4, D5).
9. Gates on the shipped tree; plant verdicts fixed; `mhc_sinkhorn_short`; the cover on decode rows (acceptance 1,
§6.3). 10. Mode (ii): a 64 GiB job-step memory limit, the override set, hygiene, the tier in experts, one fixed step for identity
(D10, acceptance 2). 11. The run as the unit; a second allocation within 3%; Maya's CV at N1; the ratio defined
(D11). 12. The measured ledger; launches reported (§7.1, acceptance 3). 13. Q12.

**Owner questions** (each with the recommended default, which the build follows until answered):

| # | Question | Recommended default |
|---|---|---|
| Q1 | Gate both host modes on the 5090 (12 cells), mode (ii) as a 64 GiB step memory limit on a larger desktop? | **Yes** |
| Q2 | Mode (ii)'s RAM tier equal in experts for both engines, sized to fit both under the limit? | **Yes** |
| Q3 | MTP: deferred unless step 0's replay predicts ≥ 1.0× on the mode (i) rows; if built, shipped only where ≥ 5% faster? | **Yes (defer)** |
| Q4 | The 16 GB row: RTX 5080, Maya-S, mode (i), reported (not gated), with a lock because it is the development lock class? | **Yes** (no Maya-S24 download) |
| Q5 | Maya's 32K cells: gated against the faster of its default and `CKPT_EVERY=0`? | **Yes** |
| Q6 | The prefill bar stays 0.9× (v1-offload §1.4; FN closed with it), decode 1.0× in tok/s? | **Yes** |
| Q7 | Holon stays exact (GPU-only) by default; a CPU split only by `docs/v1-offload.md` D4's rule, reported; R3's lever decided at step 0's end? | **Yes** |
| Q8 | Core: 0 planned, a hard cap of +5 lines, each line escalated to the owner? | **Yes** |
| Q9 | Disk: Maya-S only (done, 96.5 GB); no HF FP8 checkpoint (328 GB); keep FN's GGUFs (≈ 115 GB) for FN's propagations? | **Yes** |
| Q10 | Mode (ii) = the 5090 lock + one pre-registered override set (file tier, `moe.ram` in experts, `--host-memory`, placement, step size, I/O depth; D10), not a lock keyed on host RAM? | **Yes** |
| Q11 | Maya's arms after its own `--calibrate` on each card? | **Yes** (running) |
| Q12 | Defer K4 (KV streaming): the bars stop at 32K, as FN's did; a scope change against plan §10's "GLM's DSA adopts it"? | **Yes (defer)** |

## 12. Logistics

**Model files (done).** Maya-S v2 in the shared HF cache at `f16ee4ba` (47.84 + 47.80 + 0.84 GB = 96.47 GB), sha256 of
all three shards equal to `maya.py:124-136` on scratch and on the node's copy; `config.json`, `tokenizer.json`,
`chat_template.jinja`, `generation_config.json`, `tokenizer_config.json` at `46a050be`. Not downloaded: the HF FP8
checkpoint, Maya-S24 / M / L, Unsloth's files, DFlash2.

**Disk.**

| Item | Where | Size |
|---|---|---|
| Maya-S | the shared HF cache | 96.5 GB |
| Maya v1.1.27 build and its pinned llama.cpp `3cf03257` | `local/build/maya-v1.1.27`, `local/build/maya-llama.cpp` | ≈ 0.5 GB |
| Maya's pack (built per allocation on the node) | the node's NVMe, beside the GGUF copy | 0.9 GB |
| route traces, tiny fixtures, evidence | `local/tmp/glm`, `evidence/` | < 2 GB |
| **total on scratch** | | **≈ 97 GB; 275 GB free after it** |
| the GGUF copy for every run | the bar node's NVMe (per allocation) | 96.5 GB |

**Cards and hosts.**
- **RTX 5090** desktop: the bars (modes (i) and (ii)), N0–N3, N6, steps 6–9; one allocation holds the container,
  the GGUF copy and Maya's pack.
- **RTX 5080** desktop: the 16 GB row, its lock, sm_120 L0.
- **RTX 4090**: sm_89 L0 and memcheck.
- **RTX PRO 6000** (96 GB, a whole node): step 5's full-model gates and acceptance 2's all-resident reference.
- **A5000s**: sm_86 L0, the tiny tests, step 1 on the vehicle, N4's probe, the suite.

## 13. Review log

### Rev 1 → rev 2 (panel: A correctness, B architecture, C numerics / performance)

Rev 1 drew A 6 majors + 12 minors, B 9 majors + 17 minors, C 3 blockers + 10 majors + 13 minors. Every finding was
checked against the code; dispositions:

| Finding | Disposition |
|---|---|
| A-M1 fused router softmax-only; E = 288 has no row-invariant prompt GEMM | **Accepted** (D6): sigmoid + bias + scale in `holon_topk_select` and `holon_moe_router`; the router GEMM takes 288, no `F.linear` fallback; L0 both paths |
| A-M2 F10: no heat; refill victims not demoted | **Accepted** (D8a): heat, victim rank, evictions through `moe_tiers@1`, the record, `LOG`; plants |
| A-M3 the next layer's routed SSD set is unknowable | **Accepted** (D8f): dropped; Maya's whole-set rule for steps ≥ 1,024, if N2 shows it pays |
| A-M4 mode (i)/(ii) identity vs different step sizes | **Accepted** (acceptance 2): identity at one fixed `max_step_tokens` |
| A-M5 MTP index sharing underspecified, untestable by LC | **Accepted** (D7): fresh selection per sub-step; deviation from VL recorded |
| A-M6 "tiny fails" for GPU-only plants | **Accepted** (§6.3): each plant names the test that runs it; the GPU tiny run (a′) added |
| A-m1 cache dtypes vs the fp32 tiny bar | **Accepted** (D5): model dtype; pooled keys f32 |
| A-m2 router tie wrapper must patch `torch.topk` | **Accepted** (D2) |
| A-m3 the masked gate can be exact with f = −inf | **Accepted** (D4): `f = −inf`, `β = 0`, no new flag |
| A-m4 `mhc_block` must not forward the mixer's state | **Accepted** (D3) |
| A-m5 the fused post→pre stays inside one block | **Accepted** (D3, in the contract) |
| A-m6 `MTPHead.write_kv` needs `PreNorm` and `dsa_mla` `write_kv` | **Accepted** (D7), if MTP is built |
| A-m7 `gated_rmsnorm@1` rounding | **Accepted** (D4): fact `round = "none"` |
| A-m8 sigmoid reference follows ggml's normalisation | **Accepted** (D6) |
| A-m9 accounting omissions (indexer bytes, MTP classes) | **Accepted** (§0, §7; MTP classes only if built) |
| A-m10 `exclusive_lent_not_demoted` may not bite | **Accepted** (D8b): a small RAM tier forces reuse |
| A-m11 lending under a full RAM tier sends lent pieces to SSD | **Accepted** (D8b): demote over RAM's coldest when hotter (Maya's `LEND_PARK_LRU`); counted in §7.4's mode (ii) |
| A-m12 `mtp_h_prenorm`'s acceptance drop not safe to pre-register | **Accepted** (§6.3) |
| B-M1 KDA and DSA as separate packages copy machinery | **Accepted** (D4, D5; lead call 8) |
| B-M2 F10 under exclusive tiers needs more than the record | **Accepted** (D8a) |
| B-M3 tables at `before_materialize` break the I/O worker | **Accepted** (D8d): only the tables move; `io` and read binds stay at `resolver()` |
| B-M4 the post→pre fusion must be in the contract | **Accepted** (D3): `hc_mhc_mix@1` takes the pending write-back |
| B-M5 index sharing has no carrier without a core seam | **Accepted** (D7): dropped |
| B-M6 mode (ii) runs an unlocked configuration | **Accepted** (D10, D15, Q10): the full override set pre-registered as one named configuration |
| B-M7 acceptance judged on code not shipped | **Accepted** (acceptance 1, step 10) |
| B-M8 deferring K4 is a scope change | **Accepted** (Q12) |
| B-M9 step 1's order; (d) conditional; (e) is (b) | **Accepted** (D8 order; §9) |
| B-m1 `mhc_block` state; phase names | **Accepted** (D3) |
| B-m2 D7's edits undercounted; the unwrap rule | **Accepted** (D7) |
| B-m3 `dsa_triton` would copy `qsa_triton` | **Accepted** (D5, D12) |
| B-m4 import-linter list | **Accepted** (§4, step 3) |
| B-m5 D15's src list incomplete | **Accepted** (D15): `holon_tools impact` at each step close; list completed |
| B-m6 KDA state size and conv weight dtype | **Accepted** (D4, §0): conv state in the model dtype (140.9 MiB a sequence); weights f32 as stored |
| B-m7 a rewind flag adds an input and a vendor hunk | **Accepted** (D4): `gdn_rewind@1` generalised, f = −inf |
| B-m8 landing units unsized; no new store options | **Accepted** (D8, D8f): ≈ 0.5 GB counted; no new store options |
| B-m9 `derived_from qwen4_exp` overstated | **Accepted** (§5): written fresh |
| B-m10 `glm5` alias speculative; `glm_xml@1` unmeasured | **Accepted** (D13): `glm4` only; real-client smoke with a tool call |
| B-m11 Q3's thresholds misaligned | **Accepted** (D7, Q3): defer unless ≥ 1.0× |
| B-m12 cap +15 too loose | **Accepted** (acceptance 6, Q8): +5, each line to the owner |
| B-m13 the dense path gives row independence | **Accepted** (D5, §6.1) |
| B-m14 a rebase rule for step 1 | **Accepted** (D15) |
| B-m15 acceptance 8 not measurable | **Accepted** (acceptance 8): ≤ +5% |
| B-m16 `mhc_comb_transposed` must target both providers | **Accepted** (§6.3) |
| B-m17 a lock for a reported row | **Accepted** (D15, Q4): the 5080 is the development lock class |
| C-B1 no per-cell predictions | **Accepted** (§7.4): central and band per cell; Maya from N0, replaced by N1; C's Maya side reconciled with the measurement |
| C-B2 the decode SSD path is out of scope | **Accepted** (D8f; lead call 2), proven by N3 |
| C-B3 32K prefill needs ≥ 8–16K steps | **Accepted** (D9): 16K steps with sub-batched activations; N4's compute and activation probes. Note: N0 shows Maya's 32K is compute-bound (≈ 0.29 ms a token), so C's 0.44–0.62 at 4K steps becomes ≈ 0.85–0.92 against the measured Maya |
| C-M1 the ledger overstated the arena | **Accepted** (§7.1): 31.448 GiB, statics; ≈ 3,130 experts against Maya's measured ≈ 3,300 |
| C-M2 the roofline optimistic, no Maya side | **Accepted** (§7.4): t_fixed 2–3 ms, P 47–49, Q6_K 80%; Maya measured |
| C-M3 the ≤ 720 launch gate | **Accepted** (acceptance 3): reported, inventoried at step 4, in the roofline |
| C-M4 1K prompts: staging, not the ring | **Accepted** (D9; lead call 4). N0: a 1K prompt routes 89% of experts (C est. 70–85%) |
| C-M5 N2 needs prompt traces and the bench's exact workload | **Accepted** (N2) |
| C-M6 the cover sees prompt rows only; the decode plant's catcher is thin | **Accepted** (acceptance 1f, §6.3): decode rows, router, `mla_attend`, KDA at step starts; FN's b → b+1 and p50 pre-registered |
| C-M7 GPU-only plants never run in the tiny test | **Accepted** (§6.1 a′, §6.3) |
| C-M8 Sinkhorn iteration-count bugs hide under mild inputs | **Accepted** (D2, D3, §6.3): peaked inputs; `mhc_sinkhorn_short` |
| C-M9 mode (ii) hygiene | **Accepted** (D10) |
| C-M10 D11's statistics | **Accepted** (D11): the run's statistic, the pair median, a second allocation within 3%, Maya's CV at N1 |
| C-m1 RAM misses must not wait behind SSD | **Accepted** (D8c) |
| C-m2 absorbed vs non-absorbed prefill; `q̃` precision | **Accepted** (D5, N5) |
| C-m3 the indexer's prefill cost in fp32 | **Accepted** (D5): `bf16x3` allowed within the fp32 L0 budget |
| C-m4 the tail skip | **Accepted** (D9): counted as Maya's edge now; built at step 4 if core-free (`StepCtx.logits_index`) |
| C-m5 masked rows: β = 0; both KDA paths tested | **Accepted** (D4) |
| C-m6 the clamp bind rate per path; NaN | **Accepted** (D2, D6) |
| C-m7 square tiny MLA dims | **Accepted** (D2, §6.1) |
| C-m8 §7 small items | **Accepted** (§7.3); the MTP slot only if built |
| C-m9 R1's numbers against §7 | **Accepted** (R1, §7.4) |
| C-m10 MTP analytically ≈ 0.9× | **Accepted** (D7): the schedule plans on the deferral |
| C-m11 the 5080's ring cannot fit its IQ3_XXS class | **Accepted** (§7.2): the 5080 stages routed misses |
| C-m12 read ≥ 2 layers ahead | **Accepted** (D8f) |
| C-m13 near-ties in the cover's 100% rule | **Accepted** (acceptance 1f) |

No finding was rejected.
