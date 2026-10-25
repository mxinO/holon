# v1 — GLM: GLM-5.3-Flash on Holon, then its expert offload compared with Project Maya

Status: **rev 3 (2026-10-25), for the panel's third round.** Rev 2 drew A 5 new majors + 4 minors, B 1 major + 8
minors (all 26 rev-1 findings resolved), C 7 majors + 9 minors; §13 answers each, with the lead's calls (§11). Step 0
is under way (N0, N2's decode traces and N4's A5000 probes done; N1 running; §8). Nothing in the engine is built.
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
"N0 … N6" cite step 0's records (§8): `bench/.out/glm0_*`, `local/tmp/glm/{traces,n4}` until step 0's evidence
records are written.

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
   - **(a) Tiny model on the CPU** (§6.1): a random reduced config written by LC's converter (f32) matches HF 5.19.0
     in fp32: logits within 1e-4 and top-1 identical, one-shot and chunked; identical DSA token sets per (layer,
     position) under the lower-pool tie rule (ties asserted to occur); a prefix hit bitwise equal to recomputing. The
     GGUF tokenizer gives llama.cpp's ids on a multilingual round-trip corpus.
   - **(a′) A quantized tiny model on the GPU** (§6.1; A-N2, C-R2-M2, lead call 3): a wider tiny config (H = I = 512,
     power-of-two MLA dims, `lm_head` scaled to a logit std ≥ 2) converted to **Q8_0** by `llama-quantize`, so the GPU
     run reaches mmvq / mmq, the fused router and the clamp's mmvq path; through the GPU providers against the CPU
     engine on the same file, judged by KL (bound from the clean run ×3). **The IQ2_XXS / IQ2_S / IQ3_XXS / Q6_K
     kernels stay L0-only**: their 256-value blocks need widths a tiny model cannot carry cheaply.
   - **(b) Full model, by the owner's method** (FN §18; no fp32 goldens, no big-model CPU runs):
     - **(c) GPU self-consistency:** decode (1-token steps) vs prefill and chunked (250) vs one-shot on the 4
       `v1-long` prompts retokenized for GLM: KL mean ≤ 0.13 and KL p50 ≤ 0.007 (FN's bars). Two pre-registered
       must-fail plants (§6.3).
     - **(d) Task accuracy:** FN's 72-item synthetic set (`spikes/fn7/tasks`), the recall filler rebuilt to GLM's
       token lengths, rendered with GLM's template; Holon and Maya greedy on the same file and card, the same ids.
       Bar, stated now: Holon within 5 items of Maya's score, and no Holon answer degenerate where Maya's is not
       (pooled rates, the owner's ruling of 2026-10-24).
     - **(e) A sample dump** of 8 answers for a human look.
     - **(f) The cover** (FN §22.6, extended): on the real model's own inputs, over ≥ 1 prompt of ≥ 8K tokens in
       2,000-token steps (≥ 4 step boundaries, not multiples of 64) and ≥ 256 decode rows: DSA selection equal to its
       oracle on the engine's own q and pooled keys (a row whose differing pools lie within fp32 error of the 512th
       score counts as equal); the router's ids and weights; one KDA layer's output and state at each step start; one
       mHC pair's **`comb` itself** and its write-back (C-r2-m5); one `mla_attend` call at a sparse length. Bounds set
       FN's way (§6.2). Step 5 records the real comb logits' spread, and `mhc_sinkhorn_short`'s cover verdict is
       conditional on it (A-n3).
   - **On the shipped tree:** gates (c) and (f) and `exact.py` run again at step 10 on the final tree at each lock's
     step size; gate (d) runs again if any decode-path kernel changed after step 5.
2. **Exact and run-identical:** at **one fixed `max_step_tokens`** (16K, the step of both modes, D9), snapshot traces
   are byte-identical across fits (all-resident `std.moe@1` on the RTX PRO 6000, the deployment's cache, a capped
   cache), across policies (`lfu`, deferred admission, F10 spares), with lending and the ring on and off, and between
   RAM holding every expert and RAM short (exclusive tiers + SSD). Run-identical ×3 on the RTX 5090 and the RTX 5080
   at their locks. **Every existing lock advances byte-identical** at the step that lands a shared change (D9, D6, D4).
3. **Kernels:** L0 for every new kernel against its `reference.py` on sm_86, sm_89 and sm_120, memcheck clean; every
   decode step graphed; a kernel-level run-identity test for every in-place multi-CTA kernel (A-N5). **Launches per
   decode token are reported, not gated:** step 4 takes a launch inventory on a 4-layer slice (FN's method) and the
   count × the measured per-launch cost enters the roofline. `plugins/tests/test_reserve.py` covers tiny GLM from its
   first kernel, including a multi-row step past `SUB_TOKENS` with KDA, and asserts **≤ 150 KiB of live activations
   per token** (R included) on a 16K-equivalent sub-batched step (C-R2-M3; N4 measured ≈ 119 KiB on the A5000).
4. **Program bars** (`docs/v1-offload.md` §1.4; protocol and statistics D11), RTX 5090, Maya-S, each engine's
   drafting as deployed (Maya: none on one NVIDIA GPU, G-B §5; Holon: none unless D7 builds and ships MTP), 1K / 4K /
   32K:

   | Host mode | Against | Decode | Prefill |
   |---|---|---|---|
   | **(i) RAM holds every expert** | Maya's defaults (its `--calibrate` kept them, N1) | ≥ 1.0× | ≥ 0.9× |
   | **(ii) 64 GB host** (both engines under one 64 GiB step memory limit, the same RAM tier in experts, the GGUF on the node's NVMe, D10) | same | ≥ 1.0× | ≥ 0.9× |

   - 12 gated cells, each predicted in §7.4. Plan §9's kill line (< 0.8×) applies to each.
   - **Reported, not gated:** Maya's deterministic recipe; Maya's 32K arm with `MAYA_GLM_CKPT_EVERY=0` (the gate takes
     the faster arm, D11); the RTX 5080 with Maya-S, both engines, mode (i); Holon with MTP if it is built and no lock
     takes it.
5. **Memory** (§7): the 5090 fits at 33,792 tokens with 16K steps in both modes; the 5080 at 33,792 tokens in mode
   (i). Every pinned host byte is accounted against `host_memory`; in mode (ii) each engine's process stays under the
   step's memory limit, read from the step's own cgroup (`memory.peak`, N1).
6. **Core:** **no core change is planned (§4).** Any core line is escalated to the owner with a toy proof; a hard cap
   of **+5** code lines, counted at every step close by `scripts/ci/check_core_loc.py` **against the merge-base with
   `main`** (B2-m8). ABI stays 1.5; std and plugin contracts change only additively.
7. **Locks:** `rtx-5090` (both modes' step) and `rtx-5080` (the development class), by `tune --new-model`. Mode (ii)
   is one named, pre-registered override set on the 5090 lock (D10). Every other lock is propagated once, at GLM's
   close.
8. **Tests:** the default `pytest` run grows by ≤ 5% (`--durations` top 10 reported at each step close); one shared
   tiny-GLM engine per module.

**Non-goals:** the vision tower; multi-GPU; DFlash2 (CC BY-NC-ND); K4 (Q12); Maya-S24 / M / L and Unsloth's files;
NVFP4 / EXL3; Maya's approximate opt-ins (AG1–AG6; Holon's A1 `approx` stays off); the CPU split as a default (D8g).

## 2. Decisions

**D1 — One file on every card: Maya-S v2** (downloaded at `f16ee4ba`, sha256 verified against `maya.py:124-136`). It
is the bar's file and Maya's choice for the 5090 (`maya.py:1280`); the 16 GB row uses it too (Q4).

**D2 — The oracle: HF transformers 5.19.0 `glm5_next` on the tiny models only, in a golden-only venv.**
- `local/envs/golden-glm`: transformers 5.19.0, CPU torch, **no `flash-linear-attention`** (HF swaps in FLA's
  exp2-domain KDA kernels when installed, HFm:463, 517); `-e judge -e std --no-deps`, `--link-mode copy`.
- fp32 on the CPU: HF's bf16 rounding points (bf16 streams, β's sigmoid, the bf16 MoE combine, G-A §9) are no-ops,
  so Holon's fp32-residual form (LC's, Maya's) matches.
- **Documented deviations from HF**, recorded in every fixture's metadata: the indexer's `Tensor.topk` (HFm:889) and
  the router's `torch.topk` (HFm:168, 177) are wrapped per module instance (the router's inside its forward) with a
  stable descending sort (lower pool / lower expert wins).
- **Random init hides conventions** (HFm:1381-1421): the fixtures randomise `A_log`, `dt_bias`, the selection bias,
  hc `base` / `scale`, the pooling gate and APE, every norm weight and LayerNorm bias, and the pad row (pad = EOS =
  154820), and assert each is non-default. **The comb logits are scaled so that ≥ half the rows are peaked**
  (|s₂·m| ≥ 8, rows colliding on one destination), and the expert weights so that the clamp binds on ≥ 5% of rows of
  every path (routed, shared, dense, both sides of `clamp(u)`).
- **Two fixtures** (lead call 3):
  - **CPU, f32:** H = 64; MLA dims distinct powers of two (kv_lora 64, nope 32, v 16; C-R2-M2), so a transposed
    layout fails on shape and the Triton tiles see only power-of-two widths.
  - **GPU, Q8_0** (acceptance 1a′): H = I = 512 (the fused router's `hidden % 512`), 8 experts top-2, power-of-two MLA
    dims (kv_lora 128, nope 64, v 64), `lm_head` scaled to a logit std ≥ 2 (A-N2: at HF's init the median top-1 gap
    is 0.026 and any bf16-level noise flips it); written f32 by LC's converter, then `llama-quantize` to Q8_0.
- **No MTP oracle is needed while MTP is deferred** (D7). If built, the reference is transcribed from LC's
  `graph_mtp` (fresh selection per call) and Holon matches LC.
- **Fixture tool:** `testing/holon_testing/glm5next_files.py` (FN's `qwen4exp_files.py` pattern): the random HF
  models, LC's converter at `27c7a76` with the real `tokenizer.json`, `llama-quantize` for the GPU copy, pinned
  sha256; the GGUFs are rebuilt, not committed; CI skips only the GGUF-dependent checks when LC's tools are absent.

**D3 — mHC: siblings in `std.hyper`, fp32 throughout, the in-block fusion in the contract.**
- The residual is `Residual(16384, "float32")` through C1, exactly as FN's (G-C §1); 64 KiB per token.
- Units: `std.hyper.mhc_block@1(n, mixer, ffn, norms)` — the attention half's mix → RMSNorm → mixer → write-back;
  the FFN half's mix → RMSNorm → FFN → write-back — **declares no state** (the core binds the mixer's state from
  `LayerDecl.mixer`, `assembly.py:140-178`); its phases are named `mix` / `writeback`, not `pre` (`gr_block`'s PLE
  parameter). `std.hyper.mean@1(inner)` is the collapse (the stream mean, then `inner`, the model's RMSNorm), used as
  `ModelSpec.norm`. `std.hyper.expand@1` is reused unchanged.
- **Contracts:**
  - `hc_mhc_mix@1(R [T, n·H] f32, fn [24, n·H] f32, base [24], scale [3], y_prev?, post_prev?, comb_prev?) → (x [T, H]
    of dtype, post [T, n] f32, comb [T, n, n] f32)`. With `y_prev` it first applies the attention half's pending
    write-back to R (in f32), **only within one block**; then the unweighted RMS, the 24-wide product summed in a
    fixed order, the affine maps, the Sinkhorn of HFm:316-325 exactly (1 softmax, 20 + 19 normalizations), the
    weighted collapse rounded once. R is complete at every block boundary.
  - `hc_mhc_writeback@1(R, y [T, H], post, comb) → R'` with `R'_d = post_d · y + Σ_s comb[s, d] · R_s` in f32: the
    block's last write-back.
  - **The write-back is cross-stream** (A-N5): every output column h reads all n streams' old values at h, so the
    contracts state that R is written in place **only by a thread or CTA that owns column h of all n streams** (work
    split over h, never over streams); any other split writes out of place. A kernel-level run-identity test (many
    CTAs, repeated runs, against the out-of-place reference) catches a stream split, which racecheck cannot see
    across CTAs (acceptance 3).
- Precision: the 24-wide product `ieee` by default; `bf16x3` / `tf32x3` only within L0's fp32 budget, recorded as a
  kernel fact (FN D10).
- Plant `mhc_sinkhorn_short` (19 iterations) must fail L0 (peaked inputs) and the tiny tests; the cover's verdict is
  conditional on the real combs' spread (acceptance 1f).

**D4 — KDA: `unit.std.mixers.kda@1` and `op.kda@1` inside the `std.mixers.gdn` package**, sharing GDN's `Planner`,
state specs, replay buffers, in-step conv publishes and rewind flow; kernels from the FLA pin we already carry.
- State: GDN's two `Slot` specs — the recurrent `[64, 128, 128]` f32 and the conv windows of q, k, v (3 × 8,192
  channels × 3) in the model dtype, as GDN's — `alignment = in_step = 64`, replay rewind, checkpoints through
  `std.state.recurrent`. 140.9 MiB a sequence.
- Conv weights stay F32 as stored: `causal_conv1d@1`'s contract states that weights are used in their stored dtype;
  L0 gains an f32-weight case.
- Op `kda@1`: `(q, k, v, f [T, H·Dk], β [T, H], A_log [H], dt_bias [H·Dk])`, fact `lower_bound` (−5); chunked and
  recurrent paths, in-step snapshots, per-token states as GDN's.
- **Kernels** (A-N3, lead call 4):
  - **chunked:** `chunk_kda` and `fused_kda_gate_chunk_cumsum` from `ops/kda.py` at vLLM `6dd813e0`, vendored into
    `kernels.fla` with `vendor.patch` dropping their vLLM imports; FN's `SUB_TOKENS` packing and in-step snapshots in
    `chunk_delta_h` apply unchanged (it already takes per-channel `gk`);
  - **recurrent (decode, verify):** **not** the pin's `fused_recurrent.py`, whose kernel treats `state_idx <= 0` as
    padding (`fused_recurrent.py:114, 158`) — at `max_seqs` 1 the only sequence is slot 0, so every call would return
    unwritten. KDA's recurrent path goes through `fused_sigmoid_gating.py`'s `IS_KDA` branch, which already carries
    Holon's patches (slot 0 real, `has_init`, final store once, `out_slots`, per-token states; `vendor.patch:483-514`),
    with the bounded gate added there in `vendor.patch`;
  - **the lock's case as a test:** a one-sequence decode at slot 0 against the reference, which fails on the unpatched
    upstream kernel.
- **Rewind:** `gdn_rewind@1` is generalised to a per-row gate width (`a` [L, T', W], W = Hv for GDN, H·Dk for KDA);
  **one masked value for both, `−inf`** (A-n4): for KDA `g = −5 · σ(−inf) = −0` exactly for any `A_log`, for GDN
  `softplus(−inf) = 0` and `σ(−inf) = 0`, the same bits as today's −1e4 — so GDN's locks advance byte-identical (checked
  at the step). `β = 0` on masked rows. Tests: a masked row leaves the state bitwise unchanged on the chunked path
  (FLA's 16-row sub-chunks) and the recurrent path.
- **o_norm:** `gated_rmsnorm@1` gains a fact `round = "none"` (passed only for KDA): the norm stays f32 through the
  weight and gate (HFm:385-397).
- **The whole mixer is sub-batched** (D9; A-n1, C-R2-M3): the unit runs its projections, convs, gates, kernel, gated
  norm and `o_proj` over row sub-batches of `rows` (a recipe value; 0 = off), boundaries multiples of 64, carrying the
  conv window and the recurrent state across sub-batches; tested bitwise against one call of the same step as FN's
  `SUB_TOKENS` was.
- β is the fp32 sigmoid (LC, Maya, FT).

**D5 — DSA: `unit.std.mixers.dsa_mla@1`, `op.dsa_index@1` and `op.mla_attend@1` inside the `std.mixers.qsa` package**,
reusing `IndexPlanner`, `dense_limit` / `width` and the model-only path; one mixer because the indexer's query reads
MLA's `q_r`, so the indexer cannot wrap the attention as QSA's does.
- **Specs:** `"kv"` — the latent, `[page, 512]` in the model dtype, one copy read as both K and V; `"idx"` — per page
  the raw keys and gates γ (model dtype, 128 each) and the pooled keys (f32, 128 per 4 tokens): 640 B per token per
  layer. Two groups, as QSA's. `page_size % 4 == 0`.
- **Pooling** (`dsa_index@1`): a pool's key is computed when its 4th token is appended, from the stored raw keys and
  gates (rewind and prefix hits stay stateless, FN D6); a pool completed by rejected drafts is rewritten.
- **Selection:** `qsa_select@1` gains an optional per-row head-weight input `w [T, Hi]` (fact `head_weights`, default
  off) and Hi up to 32. fp32 scores, IEEE products, the exact top-k with lower-pool ties, shape-invariant,
  sub-batched workspace (FN D6). In prefill a `bf16x3` tensor-core split is allowed within the fp32 L0 budget.
- **Attention** (`mla_attend@1`): absorbed MQA over the latent for prefill, decode and verify, over the selected pools
  + tail. Decode's `q̃` is f32; prefill's `q̃` precision is set by L0 against the f32 reference (fp16 with a per-head
  power-of-two scale, Maya's form, or bf16; budget rel L2 ≤ 2e-4).
- **`attn_k_b` / `attn_v_b` are dequantized from Q6_K to BF16 at load** (+0.27 GB): a recorded deviation from the
  GGUF's exact values (A-n2), the rounding Maya's pack makes too, of tensors that are BF16 in the source checkpoint
  (`kv_b_proj`, G-A §3.1). The L0 reference takes the same rounded weights; the CPU tiny model is f32 and unaffected.
- **The dense path is the sparse kernel over every pool** (Maya's `DSA_ALL`), so a row's attention bits do not depend
  on co-scheduled rows; the planner's `dense` only skips scoring. The tiny test asserts it (one row alone vs beside a
  long row). N5 may replace it only with a kernel ≥ 10% faster on the 5090's prefill (it times the non-absorbed form
  too); the property is then dropped and recorded.
- **Sub-batching inside the mixer** (D9): `q_b`, `q̃`, the attention and `o_proj` run over row sub-batches of `rows`
  (queries sub-batched over all keys); the indexer's scores keep FN's 512-query passes.
- **Kernels** extend `kernels.qsa_triton` (`heads ≤ 32`, `head_weights` in the existing selection); `dsa_index` and
  `mla_attend` are new, unadapted kernels in **their own file** (`kernels/qsa_triton/mla.py`), so the plugin's
  provenance list (`kernel.py`, adapted from FreeToken) stays accurate (B2-m8).

**D6 — MoE: two generic contract minors, facts passed only when not the default; the router is a real kernel change.**
- **Router contract:** `moe_topk@1` and `moe_router@1` gain the facts `scoring` (`softmax` | `sigmoid`), `bias` (an
  `[E]` f32 parameter used for selection only), `scale`; `std.moe@1` gains the bias parameter. The reference
  normalises as ggml computes it, `× 1 / max(Σ, 6.1e-5)` (≤ 1 ulp from HF's `/(Σ + 1e-20)`).
- **Router kernels** (`kernels.ggml_cuda`; A-N1, B2-m5, lead call 2) — **both routing paths change**:
  - **decode** (≤ `rows` tokens): `holon_topk_select` inside `holon_moe_router` (`topk-moe.cu:468-535`) gains the
    sigmoid + bias + scale path; `holon_moe_router` takes the bias pointer and scale; `HOLON_ROUTER_CASE(288)`;
  - **prompt chunks:** the router GEMM's logits go to `MoeTopk` (`plugin.py:694-697`), whose wrapper
    `holon_ggml_topk_moe` hard-codes softmax with a null bias (`csrc/api.cu:476-488`): **`MoeTopk` and
    `holon_ggml_topk_moe` take scoring, bias and scale** (the vendored `topk_moe_cuda` has all three);
  - **the prompt GEMM takes E = 288 and stays row-invariant:** `router_chains` is defined as the largest divisor of
    hidden / 32 that is ≤ 2048 / E (128 → 16 and 256 → 8, unchanged; 288 → 4 at H = 4096), and the `prefill`
    predicate's `experts % 64` (`plugin.py:686`, from `ROUTER_BN = 64`) follows the per-E tile (32 / 96 for 288). For
    whole chunks there is no fallback to `F.linear`; a step's last partial chunk (more than `rows`, fewer than 512
    rows) keeps `F.linear` as today, deterministic per step shape (A-N1's nit);
  - `_supports_moe_topk` admits 288 (`plugin.py:764-766`);
  - **L0, bounded not bitwise** (C-r2-m1): `kernels.ggml_cuda` builds with `-use_fast_math` (`extension.py:28`), so the
    sigmoid is `__expf`. As today's form (`plugins/tests/test_ggml_cuda_moe.py:896-952`): the fused router bitwise
    against `moe_topk` on the GPU, both against the reference within an ulp budget that allows near-tie ids;
  - **the vehicle's and FN's MoE locks advance byte-identical** after the router change (step 4's check): the only
    case that catches a chain or tile slip for the existing E.
- **Clamp:** `linear_experts@2`, `linear_experts_arena@1` and `silu_mul@1` gain `glu_limit`: `silu(min(g, L)) ·
  clamp(u, −L, L)`. ggml's `GGML_GLU_OP_SWIGLU_CLAMP` is vendored in mmvq (`mmvq.cu:863, 1043`); the fused-SwiGLU mmq
  store and `kernels.ggml_cpu` take the same form; L0 records that ggml's `fminf` / `fmaxf` drop NaN where HF's clamp
  propagates it.
- The shared expert composes as `std.ffn.sum@1(moe, swiglu)` (FN D7), ungated, after the routed gate/up.

**D7 — MTP: deferred unless step 0 predicts ≥ 1.0×.**
- Maya loads no MTP on one NVIDIA GPU (`glm_mtp.cu:22-24, 78-98`). **N2 measured the verify window's cost**: on Maya's
  decode routing, 2 / 3 / 4 consecutive tokens route 13.9 / 19.1 / 23.9 distinct experts per layer against 8 for one
  (both traces, 35,936 tokens), so a 4-row verify pays ≈ 3× a token's routes. With Holon's mode (i) token (§7.4) and
  0.65–0.75 acceptance per draft, MTP comes out **≈ 0.89–1.09× plain** at 1–3 drafts: not ≥ 1.0× with confidence.
  **MTP is deferred** (the lead's default, Q3); the schedule plans on it.
- If ever built: `std.spec.mtp_head@1` with GLM's `blk.45`; each draft sub-step selects its own pools (a deviation
  from VL's `index_share_for_mtp_iteration`); `drafters/mtp` unwraps the embed when `target.spec.residual is not
  None`; `MTPHead.write_kv` with `PreNorm.write_kv` and `dsa_mla.write_kv`; experts as `aux.*` through D8(d).

**D8 — Offload: X's store; plan §10's std minors in step 1, in this order, each tested with RAM capped before GLM uses
it.** The store gains **no new options**; provider options live in their kernel plugins.
- **(b)+(e) as one change — lending and the prefill ring under exclusive tiers, and the staging that follows.**
  Deleting `or self.exclusive` (`store.py:240`) and `not self.exclusive` (`:435`) needs: a lent piece is demoted
  first — to a free RAM unit, else over RAM's coldest expert when hotter (Maya's `LEND_PARK_LRU`), else SSD-only —
  before its slot is overwritten; staging per class becomes `k × max_seqs × (1 + drafts)` (8 at GLM's lock),
  returning 2.94 GB of VRAM and as much pinned host memory in mode (ii). Pipelined parts stay as in the RAM-holds-all
  layout. Tested on Qwen3-30B-A3B with `moe.ram` capped (`exact.py`: capped vs uncapped identical) and the plant
  `exclusive_lent_not_demoted`, forced to bite by a RAM tier small enough that the stale unit is reused.
- **(a) F10 with RAM short — the full requirement set:**
  - a **heat**: `moe_tiers@1`'s heat input gains a fact `heat = "i32"` (default f32, so today's facts do not move)
    and `placement.promote@1` passes its recency (`last`, int32 device state) unconverted (B2-m8: no f32 copy, no
    precision limit on a long-running server);
  - a **victim rank** for the spare refill: the layer's LRU resident (Maya's rule);
  - **refill evictions go through `moe_tiers@1`**: each victim is demoted D2H into a free RAM unit, else over RAM's
    coldest when hotter, else SSD-only, in the graph, before the spare's next promotion;
  - `kernels.promote`'s CUDA resolve emits the delta record "push b's slots onto the layer's empty stack", replayed by
    `ResidencyMap.commit` outside borrowing steps;
  - `LOG` (`store.py:56`) sized for the records;
  - plants: `promote_spare_unlogged` and `promote_victim_code`.
- **(c) The high-priority split fetch under exclusive tiers:** two fetch launches — the RAM misses at once at high
  priority, the SSD misses after their read completes — so a RAM miss never waits behind an SSD read.
- **(f) The SSD path, rebuilt for decode and prefill** — D8f below.
- **(d) Aux experts under exclusive tiers — only if D7 builds MTP:** the residency, piece and owner tables move to
  `before_materialize`; the `cpu_task` I/O worker and each group's read binding stay at `resolver()`.
- **(g) The CPU split (F5b)** stays reported only, decided at step 0's end (R3).
- **Placement defaults:** mode (i) `lfu` (F11); mode (ii) the least modelled time among F11 with deferred admission
  and F10 spares, from N2's replay at each engine's own cache share, confirmed by step 6's run.
- **The seed** is a GLM routing profile from `route_dump` on held-out prompts (FN D12f), shipped in the bundle.

**D8f — The SSD read path: one submitting thread, asynchronous I/O, depth 16–32.**
- **Where:** `kernels.cpu_task`'s I/O worker (the provider behind `std.tiered`'s file tier,
  `std/holon_std/tiered/gguf_direct.py`, and behind `std.tiered.rows@1`'s SSD placement). The pread worker pool is
  replaced by **one submitting thread**. `cpu_task@1`'s contract is unchanged.
- **The interface, chosen automatically at worker start and logged** (C-R2-M4, B2-m1, lead call 7): **io_uring** (raw
  syscalls, no liburing, **no `IORING_SETUP_SQPOLL`** — its kernel thread spins a core, FN §22.1's −32%; the submitter
  blocks in `io_uring_enter`), else **Linux native AIO** (`io_setup` / `io_submit` / `io_getevents` on the O_DIRECT
  fds), which default container seccomp profiles allow where they block io_uring. No option chooses it.
- **Options:** `cpu_task.io_depth` (default 16; a plugin option, B2-m1) replaces `cpu_task.io_threads`, which is
  **deleted**; `io_cores` reduces to the submitting thread's one CPU.
- **Per layer call:** every SSD miss of the call is submitted at once, each expert's three extents as chunks
  (N3 picks the size); completions set mapped flags the SSD fetch launch waits on. RAM misses go first in their own
  launch (D8c).
- **Alignment** (B2-m3, C-r2-m8): 125 of 129 `_exps` tensors start 32 B past a 4 KiB boundary (G-B §1), so an
  extent is read as its 4 KiB-aligned superset (≤ 8 KiB more) into a **landing unit**, and the GPU fetch reads the
  piece at its offset there — no memmove. A read goes straight into a packed RAM-tier unit only when the file offset,
  the length and the destination are all 4 KiB-aligned; otherwise an expert kept in RAM after an SSD read (D10's
  `keep` rule) is copied from its landing unit into its packed RAM unit by the submitting thread. N3 measures that
  copy on the same thread (`uringb`) inside the 1.3 ms target.
- **Landing units and their fence** (A-N4, B2-m4, lead call 5): a named host pool `io_landing` (pinned, counted
  against `host_memory` and mode (ii)'s limit) holds **one set of landing units and one request buffer per layer in
  flight** (depth 1 for decode; 2 for prefill read-ahead; set g uses `g % depth`). A set is refilled only after a
  device-side "copied" signal — a mapped word the reading fetch kernel writes after its last load from the set — so
  io_uring never overwrites a unit a pending gather still reads. A read-ahead expert that has landed is found by the
  next layer's resolve (its tier code says landed, with its unit), so it is not read twice. Plant
  `landing_reuse_unfenced` (the refill not waiting for the signal) must make `exact.py` differ.
- **Prefill read-ahead, one threshold** (C-R2-M5, lead call 5): for steps **≥ 2,048 tokens** (`moe.stream_tokens`,
  the same line as D9's staging), each MoE layer reads the **whole** SSD-only set of the next layer ahead, 2 layers
  deep (≈ 0.5 GB of `io_landing`), if N2's prompt traces show it pays; below 2,048, each layer reads its own routed
  SSD-only set once routed. Threshold and depth are constants.
- **Target:** ≤ 1.3 ms per 7.1 MB expert in the engine (≥ 5.5 GB/s), the bounce copy included, **proven first by N3
  on the bar node's NVMe** (io_uring available in the container there: N3's probe), before step 1 builds it.
- **Core:** none.

**D9 — Prefill: 16K steps in both modes, with opt-in sub-batched activations.**
- **Why:** N0 measured Maya's 32K prompt at 3,195 tok/s, read as 2 × 16K chunks that are compute-bound (≈ 0.29 ms a
  token; its copy lane busy 4.3 s of a 4.7 s chunk). A 4K-step Holon pays the ≈ 68 GB stream eight times; from 8K
  steps it hides under compute (§7.4). **One step size, 16K, in both modes** (B2-m7): it is not in mode (ii)'s
  override set.
- **Where sub-batching lives, all opt-in, set only by GLM's recipe** (B2-M1, lead call 1):
  - `std.ffn.swiglu@1(rows = 0)`: 0 (the default) runs the whole step as today; GLM's recipe sets 2,048 for the dense
    FFN and the shared expert;
  - `std.mixers.dsa_mla@1` and `std.mixers.kda@1` (new, GLM-only): the whole mixer in sub-batches of `rows` (D4, D5);
  - the indexer's scores (FN's 512-query passes) and the MoE (`pass_rows`) are already split.
  - `rows` is a recipe value, not an engine option.
- **Bits:** sub-batching is **deterministic at a given step size, not identical to an unsplit call** — a matmul's
  bits may depend on its row count (`qlinear@2` promises nothing else; `linear_experts@2` only whole `CHUNK`s up to
  `pass_rows`). So no existing lock's computation may change: **every existing lock advances byte-identical at the
  step that lands `rows`** (its default 0), and acceptance 2's identity is at one step size. The KDA and DSA mixers'
  sub-batched forms are tested bitwise against an unsplit call only where the contract promises it (the recurrence
  with its carried state).
- **Measured (N4, A5000, one KDA and one DSA layer in their mHC blocks, `spikes/glm0/n4_peak.py`):** unsplit, the live
  set beyond R is **≈ 326 KiB a token** at every step size (5.1 GiB at 16K); sub-batched at 2,048 rows it is **0.67 /
  0.73 / 0.86 / 1.55 GiB at 4K / 8K / 16K / 32K** — with R's 64 KiB, **≈ 119 KiB a token at 16K (1.9 GiB)**, inside
  acceptance 3's 150 KiB. The routed experts' pass buffers are the store's (lent).
- **The loan:** what exceeds the reserve during a prompt step is lent by `moe.lend_peak` from **the gate/up class's
  tail** (the largest class, ≈ 13.6 GB on the 5090), never from the IQ3_XXS down class (≈ 2 GB; FN §21.12's 8K loan
  outgrew its class and the store refused the layout).
- **FN's lesson (§22.9):** every sub-batched op keeps the reserve true for a step holding several prompts' rows;
  `test_reserve.py`'s tiny-GLM case covers it from step 4 (acceptance 3).
- **1K prompts: stage the routed experts, not the ring:** steps below 2,048 tokens use FN's ring of parts
  (`moe.stream_tokens = 2048`, `moe.pipeline = 8`). N0: a 1K prompt routes 89% of all experts, so the saving is ≈ 10%
  of the bytes, more in mode (ii).
- **N4 still measures** on the 5090: the prefill compute per token and the live set at each step size.
  **Already measured on the A5000** (`spikes/moe_prefill/gemm.py` on Maya-S's layers 10 and 3): a 4,096-token pass of
  one MoE layer's routed experts takes 24.0 ms (gate / up, IQ2_XXS) + 15.8 / 13.1 ms (down, IQ2_S / IQ3_XXS) at a
  grouping overhead of 1.5 against a dense GEMM of the same MACs; `pass_rows` stays 4,096 at top-8 (`ID_PAIRS`). The
  grouping at 4K passes is a prefill-compute risk against Maya's whole-chunk MMQ launch (R2).
- **Prefill tail skip** (B2-m6, lead call 9): in the last layer, the FFN half runs on the rows
  `ctx.batch.dev.logits_index` names (the field is `DeviceBatch.logits_index`, in the ABI: `core/abi.txt:69`;
  prompt-logprob rows are already in it); **the MoE call is never skipped** (it may run on few or zero rows: the
  store's `layer_done` returns lent slots there); the mixer runs whole. Conditions: no drafter (the only residual-tap
  user); the block learns it is the last from a recipe parameter. Built at step 4.

**D10 — The two host modes.**
- **(i) RAM holds every expert:** the bar desktop pins all 86.03 GB of routed experts plus the process; F11 on; no SSD
  tier. (N0: Maya pinned a 65.89 GiB RAM tier beside its expert pool; every expert held.)
- **(ii) 64 GB:** each engine runs in its own job step with a 64 GiB memory limit (the step's cgroup `memory.max`;
  checked inside the container: 68,719,476,736). Pinned host memory is counted against it; N1 reads `memory.peak`.
  **The RAM tier is equal in experts**: N1 reads Maya's RAM-tier slot count at `MAYA_GLM_RAM_GB` set to fit under the
  limit (default 50); Holon's `moe.ram` is set to the same number of experts, per class proportionally. Each engine
  keeps its own VRAM cache, so Holon's smaller cache shows as more SSD-only experts — its own cost.
- **Hygiene** (built: `local/tmp/glm/evict_wrap.sh`, every mode (ii) arm's `wrap`): the GGUF's pages are evicted
  (`posix_fadvise(POSIX_FADV_DONTNEED)`) before every arm; the engine's GGUF fds are checked for `O_DIRECT`
  (`/proc/<pid>/fdinfo`); the device's read bytes (`/sys/block/<nvme>/stat`) are recorded around the arm against each
  engine's own read counter; `AnonHugePages`, the cgroup's `memory.current` and `memory.peak` are logged; each
  engine's SSD-only expert count is logged.
- **The pre-registered mode (ii) override set** on the 5090 lock, one named configuration in the evidence: `moe.tiers`
  with the file tier; `moe.ram` (experts per class, as above); `--host-memory`; the placement policy (D8's rule);
  `kernels.cpu_task.io_depth` (N3). The step size is the lock's (16K). Nothing else differs from the lock.

**D11 — The comparison protocol: FN's D11 as closed, with the run as the unit.**
- **The Maya backend** (`bench/offload_ctx.py`, built and checked at N0): the engine's stdin protocol with raw ids
  (`GEN <n> temperature=0 <ids>` → `T <id>` … `DONE …`, `generate.cpp:2063, 2213-2222, 7005-7041`); `--glm-pack` from
  `tools/iq_pack.py --compat-bf16` (the pack reads the GGUF from its parent directory); `--max-context 33792
  --kv-resident 33792`; `--eos-ids 2147483647` (N0: `produced` = n on every request); `MAYA_GLM_SLOTS=0`; loop guards
  off; its setup's defaults (`--expert-cache auto --prefill auto --kv-streaming on --expert-profile
  data/expert-profile-glm.bin`); **its own `--calibrate` kept the defaults on the 5090** (N1: PCIe share 0.64 and 8
  lane threads; no sweep beat them by 3%). Built at `4615f34`, sm_120, on the bench host.
- **Maya's checkpoint cut:** a 32K raw-id prompt is read as 16,384 + 16,384 by default; the 32K cells take the faster
  of Maya's default and `MAYA_GLM_CKPT_EVERY=0` arms (Q5).
- **Holon:** its lock's configuration (mode (ii): the override set), `--ctx 33792`, `ignore_eos`; 0 reused tokens per
  timed request.
- **Corpus and spans:** FN's text corpus tokenized with GLM's tokenizer (184,531 tokens); 8 spans per context (FN's
  six starts and two midpoints), the warm-up span excluded; all 24 requests in one fresh process per run.
- **Statistics** (fixed now):
  - **the run's statistic**, per context and metric: total tokens over total time across its 8 prompts (decode:
    Σ(n − 1) / Σ(t_last − t_first); prefill: Σ C / Σ prompt time), each engine timed from the client;
  - **runs** alternate in ABBA order within an allocation, a discarded warm-up run first;
  - **the cell:** the median over adjacent (Holon, Maya) run pairs of the per-pair ratio;
  - **rounds:** at least 3 (6 pairs); more if N1's measured Maya run-to-run CV needs them for a ratio SE ≤ 1%;
  - **pass** ≥ bar; **a cell within 3% of its bar is re-run on a second allocation** (another node), and the cell
    becomes **the median of both allocations' pair ratios together** (C-r2-m6); below the bar → the owner.
- **Per cell:** expert hit rates (all, and the first 32 decode tokens after a prompt), SSD reads per token and device
  read bytes (mode (ii)), Maya's `glm stat:` and `glm prefill:` lines, peak VRAM and RSS.

**D12 — Kernels: Triton in the existing kernel plugins; CUDA only by FN D10's rule.** `kernels.qsa_triton` (selection
with weights; `dsa_index`, `mla_attend` in `mla.py`), `kernels.hc_triton` (the mHC pair: decode split-K over h with
all four streams per CTA, D3; the prefill product in fixed-order passes), `kernels.fla` (KDA), `kernels.ggml_cuda`
(router and clamp), `kernels.cpu_task` (D8f). Adapted code from FreeToken (Apache-2.0) is declared per AGENTS.md. A
kernel moves to CUDA in step 7 if it is over 5% of a step and a prototype gains ≥ 10%.

**D13 — Tokenizer and frontend.** `std.weights.gguf`'s `_PRE` gains `glm4` only: BPE with `ignore_merges = True`
(llama.cpp's CHATGLM4), no normalizer, GLM's split regex (`\p{N}{1,3}`); `eom` (`<|observation|>`) read as a stop
token. `glm_xml@1` (`<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>…`, My `serve/frontend.py:8-10`) is
checked by a real-client smoke with a tool call at every merge. The template opens `<think>`, so `think@1` is checked
on a prompt that ends in it.

**D14 — Text only.**

**D15 — Locks and propagation.**
- **New:** `rtx-5090` (Maya-S, 16K steps, `max_seqs` 1) and `rtx-5080` (mode (i); the development lock class, where
  GLM's perf iteration runs). Mode (ii) is the 5090 lock plus D10's pre-registered override set, one named
  configuration recorded in the evidence; no lock is keyed on a host's RAM (Q10).
- **Shared code** is listed by `holon_tools impact <base>..HEAD` at every step close. Expected: `std.moe`, `std.ffn`,
  `std.tiered`, `std.hyper`, `std.mixers.gdn`, `std.mixers.qsa`, `std.weights.gguf` (every GGUF lock),
  `kernels.ggml_cuda`, `kernels.ggml_cpu`, `kernels.fla`, `kernels.qsa_triton`, `kernels.hc_triton`,
  `kernels.expert_io`, `kernels.cpu_task`, `placement.promote`, `kernels.promote`, `frontends.http`. New facts and
  parameters are passed only when not the default, so traces advance byte-identical (checked at each landing step,
  acceptance 2); one propagation at the close (release §22).
- **Merge order:** step 1's `std.tiered` / `std.moe/offload.py` / `kernels.cpu_task` changes merge into the branch
  first; steps 3–4 rebase on them, and `std.moe/units.py` and `offload.py` are edited by one step at a time.

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
| — | Tail skip | `DeviceBatch.logits_index` (`ctx.batch.dev.logits_index`, `core/abi.txt:69`) → unchanged | in the ABI; the MoE call is never skipped (D9) |

- **Generality:** `check_core_denylist.py` gains `glm|mhc|sinkhorn|nextn` (a CI change, not core).
- **Import contracts** (B-m4): the new modules live in existing packages; step 3 checks the import-linter contracts
  (`pyproject.toml:100-115`) with them.

**Std and plugin contract minors** (additive; `std/contracts.txt` / `plugins/contracts.txt` regenerated):

| Contract or unit | Today → change |
|---|---|
| `moe_topk@1`, `moe_router@1` | softmax, power-of-two E → facts `scoring`, `bias`, `scale`; E = 288 in both router paths (D6) |
| `linear_experts@2`, `linear_experts_arena@1`, `silu_mul@1` | no clamp → fact `glu_limit` (D6) |
| `qsa_select@1` | `Σ_h relu(q_h·K̄)`, ≤ 16 heads → optional `w [T, Hi]` (fact `head_weights`), Hi ≤ 32 (D5) |
| `gdn_rewind@1` | gate width Hv, masked value −1e4 → any per-row width, masked value −inf (same bits for GDN) (D4) |
| `std.ffn.swiglu@1` | whole step → parameter `rows` (0 = whole step, the default; D9) |
| `qsa_triton`'s selection kernel | ≤ 16 heads → ≤ 32, head weights; `dsa_index`, `mla_attend` in a new `mla.py` (D5) |
| `gated_rmsnorm@1` | rounds before the weight → fact `round = "none"` (D4) |
| `causal_conv1d@1` | weight dtype unstated → weights in their stored dtype (D4) |
| new `kda@1` (gdn), `dsa_index@1`, `mla_attend@1` (qsa), `hc_mhc_mix@1`, `hc_mhc_writeback@1` (hyper) | — |
| `moe_tiers@1` | f32 heat → fact `heat = "i32"` (D8a) |
| `ResidencyMap`, the store, `placement.promote`, `kernels.promote` | D8 (a)–(d) |
| `kernels.ggml_cuda`'s `MoeTopk` / `holon_ggml_topk_moe` and fused router | softmax only, power-of-two E → scoring, bias, scale; E = 288 (D6) |
| `kernels.cpu_task` | a pread pool (`io_threads`) → one submitting thread (io_uring, else native AIO, chosen and logged), option `io_depth`, host pool `io_landing` with a device-side fence; `io_threads` deleted (D8f) |
| `std.weights.gguf` | `glm4` pre-tokenizer, `eom` (D13) |

## 5. Units, contracts, model plugin

| Unit | Package | Role |
|---|---|---|
| `std.hyper.mhc_block@1`, `std.hyper.mean@1` | `std.hyper` | D3 |
| `std.mixers.kda@1` | `std.mixers.gdn` | D4 |
| `std.mixers.dsa_mla@1` | `std.mixers.qsa` | D5 |
| `std.moe@1` (+ bias), `std.ffn.sum@1`, `std.ffn.swiglu@1` (+ limit, + `rows`) | `std.moe`, `std.ffn` | D6, D9 |
| `std.spec.mtp_head@1` | `std.spec.mtp_head` | D7, if built |

**Contracts** (each with `reference.py` and a CPU provider in `testing/holon_testing/ref_kernels`): `hc_mhc_mix@1`,
`hc_mhc_writeback@1`, `kda@1` (with `gdn_rewind@1` generalised), `dsa_index@1`, `qsa_select@1` (weights),
`mla_attend@1` (prefill, decode, verify), the router and clamp facts.

**`models.glm5_next`**, written fresh (no `derived_from`: it shares no names, HC, PLE or rules with `qwen4_exp`;
B-m9): the recipe reads `cfg["gguf"]` (`glm5-next.*` keys; `head_count_kv` per layer and `key_length` = 512 read raw,
G-C §8), sets `residual = Residual(16384, "float32")`, `embed = std.hyper.expand@1(inner = std.tiered.rows@1)` (the
token table on the host, 0.52 GB of VRAM back), `norm = std.hyper.mean@1(rmsnorm)`, per layer `LayerDecl(kda |
dsa_mla, swiglu(limit) | sum(moe(sigmoid, bias, 2.5, limit), swiglu(limit)))` inside `mhc_block`, with `rows = 2048` on
the mixers and the dense / shared FFN (D9) and the last block marked for the tail skip; GGUF rules per
LC's names (`glm5-next.cpp:61-184`; `ssm_a` = −exp(A_log) from the converter); the bundle (chat template,
`glm_xml@1`, `think@1`).


## 6. Numerics and judge (thresholds and predicted verdicts fixed now)

### 6.1 Tiny models: CPU (f32 vs HF) and GPU (Q8_0 vs the CPU engine)

**CPU fixture (f32):** 8 layers (DSA at 3 and 7; dense FFN at 0; KDA elsewhere), H = 64, n = 4 streams, KDA 2 heads ×
32, MLA 2 heads (q_lora 32, kv_lora 64, nope 32, v 16), indexer 4 heads × 16 with kpool 4 and top-k 8 (sparse from
t + 1 > 11), 8 experts top-2, I = 64, shared expert, swiglu_limit 10 with the clamp binding on ≥ 5% of every path's
rows, peaked comb logits on ≥ half the rows, vocab 154,880 (the real tokenizer). Prompts: 100 tokens with an EOS,
chunked at 7 and 13, then 24 decode steps.

**GPU fixture (Q8_0):** the same layer pattern at H = I = 512, 8 experts top-2, MLA 2 heads (q_lora 128, kv_lora 128,
nope 64, v 64), indexer 4 heads × 64, `lm_head` scaled to a logit std ≥ 2; Q8_0 by `llama-quantize`; 3 prompts of 300
tokens (sparse DSA from t + 1 > 11), chunked at 100 / 250, then 32 decode steps; graphs and eager.

| Check | Where | Threshold | Predicted |
|---|---|---|---|
| logits (the GGUF's head), one-shot and chunked, vs HF | CPU | max \|Δ\| ≤ 1e-4, top-1 identical | pass |
| DSA token sets per (layer, position) | CPU | identical under the lower-pool rule; ties asserted | pass |
| prefix hit vs recompute | CPU | bitwise | pass |
| one row alone vs beside a long row (the dense path is the sparse kernel) | CPU, GPU | bitwise | pass |
| clamp bind rate per path | CPU, GPU | ≥ 5% | pass |
| tokenizer vs `llama-tokenize` (LC `27c7a76`) on a multilingual round-trip corpus | CPU | identical ids | pass |
| the Q8_0 fixture through the GPU providers vs the CPU engine on the same file: per-position KL | GPU (A5000; sm_120 at step 4's L0 round) | KL mean and p99 ≤ the bound set from the clean run ×3 (each ≈ 3× the clean maximum), written before any plant runs | pass |

### 6.2 Full model (Maya-S)

| Check | Where | Predicted |
|---|---|---|
| (c) decode vs prefill; chunked (250) vs one-shot: KL mean ≤ 0.13, p50 ≤ 0.007 | 5090 deployment, step 5 and step 10 | pass (FN: 0.025–0.031 / 0.002–0.003) |
| (d) 72 items vs Maya, greedy, thinking on, 16,384 new tokens at most, the template's default effort | both engines' 5090 mode (i) deployments | pass (within 5 items) |
| (f) the cover (acceptance 1f). Bounds FN's way: each set at ≈ 10× the clean run's maximum and ≥ 10× below its plant's effect, both measured before the bound is written | RTX PRO 6000 and 5090 | pass |
| exactness (acceptance 2) at the 16K step | RTX PRO 6000; 5090; 5080 | byte-identical |
| the clamp's bind rate, the real comb logits' spread, zero-score ties at the selection boundary per 1,000 rows | reported at step 5 | — |

### 6.3 Plants

Each changes only its target path; "everywhere" means every provider but not `reference.py`. **A plant's verdict on a
statistical check (gate (c), the GPU tiny run, the cover) is pre-registered only after its effect is measured at ≥ 10×
that check's bound** (lead call 3, A-N2); until then the table gives the runner and the expected direction.

| Plant | Breaks | Run by | Predicted |
|---|---|---|---|
| `dsa_decode_select_shift` | decode and verify selection shifted one pool (b → b + 1, FN's form) | the cover (decode rows; deterministic), (c), (a′) | **the cover fails** (C-r2-m4); (c) fails on its p50 (pre-registered statistic: ≈ 0.01–0.015 against 0.007) once measured; (a′) fails once measured |
| `kda_gate_softplus` | the recurrent (decode) KDA kernel uses the unbounded softplus gate | L0, the cover (KDA at decode rows), (c), (a′) | L0 and the cover fail; (c) and (a′) once measured |
| `kda_state_not_carried` | a chunked prefill step (or sub-batch) starts from a zero state, everywhere | tiny CPU, cover, (c) | tiny chunked fails; cover fails; (c) chunked vs one-shot once measured |
| `mhc_comb_transposed` | `comb` applied as C, not Cᵀ, in every provider | tiny CPU, cover, L0 | all three fail; (c) passes (both paths planted) |
| `mhc_sinkhorn_short` | 19 Sinkhorn iterations, everywhere | L0 (peaked), tiny CPU, cover (`comb` itself) | L0 and tiny fail; the cover's verdict is conditional on the real combs' spread (step 5); (c) passes |
| `mhc_writeback_stream_split` | the in-place write-back split over streams, not h | the kernel run-identity test (acceptance 3) | fails (results vary run to run) |
| `dsa_pool_ape_off` | APE left out of pooling, everywhere | tiny CPU, cover | both fail; (c) passes |
| `router_bias_in_weights` | the bias added to the weights too, everywhere | tiny CPU, cover (router ids and weights) | both fail; (c) passes |
| `router_prompt_softmax` | the prompt path's `MoeTopk` left at softmax | L0 (prompt path), (a′) | L0 fails; (a′) once measured |
| `glu_clamp_off` | the mmvq (decode) path without the clamp | L0 (inputs that bind it), (a′) (Q8_0 reaches mmvq) | L0 fails; (a′) once measured; (c) fails only if the real bind rate (§6.2) is ≥ 1e-3 of rows |
| `kda_slot0_skipped` | the recurrent kernel treats slot 0 as padding (the unpatched upstream rule) | the one-sequence slot-0 test (D4) | fails |
| `exclusive_lent_not_demoted` | a lent piece's tier code left at RAM | `exact.py` with a small RAM tier that reuses the unit | differs |
| `promote_spare_unlogged` | F10's delta record dropped | the host mirror check, `exact.py` | fails, differs |
| `promote_victim_code` | a refill victim's tier code not moved | `exact.py` RAM-short | differs |
| `landing_reuse_unfenced` | a landing set refilled without the "copied" signal | `exact.py` RAM-short with prefill read-ahead | differs |
| `aux_row_padding`, `lent-slot-still-mapped` (existing) | as FN / X | `exact.py` | differs |

## 7. Memory fit and predictions

### 7.1 The RTX 5090 (the measured total, statics included)

The 5090's measured total is **31.448 GiB** (FN `evidence/fn_close_2026-10-24.json`). All sizes in GiB.

| Item | GiB | Note |
|---|---|---|
| outside the allocator | 0.95 | FN's 0.92 + the larger host mapping |
| parameters (non-expert, token table on the host) | 6.89 | 7.13 GB + `attn_k_b` / `attn_v_b` as BF16 (+0.27 GB) |
| statics (FlashInfer workspace, mmq id workspaces) | 0.30 | FN's 5090: 0.40–0.48 GiB with RoPE tables GLM lacks |
| state pools | 0.98 | latent 0.38 GB + `idx` 0.24 GB + KDA 2 slots and 1 checkpoint 0.43 GB |
| graphs' reserve | 0.12 | FN |
| activation reserve | 1.02 | a decode or short step's peak; a 16K step's ≈ 1.9 GiB (N4) is the reserve plus the loan (D9) |
| slack | 0.44 | FN |
| **the arena** | **≈ 20.7 (22.3 GB)** | |
| staging and null slots | 0.11 | 3 classes × (8 + 1) |
| **experts cached** | **≈ 3,120 (25.8%)** | **Maya: 3,377 slots / ≈ 3,300 resident (27.3%)**: its log's "expert pool 22.36 GB" is GiB (24.0 GB / 3,377 = 7.11 MB, C-r2-m7). The gap is ≈ 1.6 GiB, **≈ 7.5% fewer experts** for Holon |

- **The gap's value:** `lfu` at Maya's share gives h 0.737 against 0.720 at Holon's (N2), ≈ +4% decode. Levers, in
  §7.4's list: `moe.lend_peak` for prompt steps, FN §23.6's stack trim (151 MiB), `--memory-fraction` 0.99.
- **Mode (ii):** the same VRAM; RAM tier equal to Maya's in experts (D10), est. ≈ 7,550 at 50 GiB; SSD-only ≈ 1,430
  for Holon against ≈ 1,250 for Maya (+14%).
- **Host, mode (i):** experts 86.03 GB pinned (THP), + the token table 0.52 GB.

### 7.2 The RTX 5080 (16 GB), mode (i)

Visible 15.47 GiB; parameters 6.89, statics 0.2, state 0.98, outside 0.6, activation 0.75, graphs 0.1, slack 0.4 →
arena ≈ 5.6 GiB → **≈ 840 experts, 6.9%**. Its IQ3_XXS down class (8 layers × ≈ 20 slots) cannot hold a ring, so the
5080 stages routed misses (the ring of parts) at every step size, and its loan stays within the gate/up class. A
reported row and the development lock (Q4).

### 7.3 The RTX PRO 6000 (96 GB), step 5

≈ 102.6 GB visible less the same items leaves ≈ 91 GB for experts against 86.03 GB: all-resident fits at
`--memory-fraction` ≥ 0.98, acceptance 2's reference fit. If it does not, the store at its largest fit is the
reference instead.

### 7.4 Predicted Holon / Maya ratios per gated cell

**Maya's side** comes from N2's 16-prompt capture (C-r2-m9): its 17 `glm stat:` lines fit **ms a token = 10.3 + 0.189
× misses** (C's `maya_fit.py`, residual sd 0.38 ms), **≈ 28.1 ms a token at h 0.72**; prompts 1.40–1.50 s at 1K and
1.78–1.80 s at 4K (engine), 10.25 s at 32K (N0, two 16K chunks). Its `--calibrate` kept its defaults (N1). N1's
calibrated runs replace these when they land.

**Holon's side** is the first-order model with C's corrected inputs: dense 7.13 GB at 1.55 TB/s × 0.8 = 5.75 ms;
misses 336 × (1 − h) × 7.11 MB at 47–49 GB/s (Holon's measured in-engine fetch, FN §21.1, §23.8); fixed 2–3 ms
(`k_resolve` 0.32, ≈ 180 mHC calls 0.6–0.9, 11 selections 0.3, KDA state 0.18, glue 0.6). At h 0.72: **22–23 ms**.
**The fixed part is a model, not a measurement** (C-R2-M1): at Maya's fitted 10.3 ms Holon would be 24.5–24.9 ms.

**The hit rate** (N2, held out: each trace's seed from the other trace): Holon's `lfu` at 25.8% gives **0.719 / 0.723**
(bench / tasks); Maya's own policy at 27.3% replays **0.714 / 0.718** against its live **0.722 / 0.725** — the replay
is within 1 point. **N2's replays leave prompt lending out** (C-R2-M6): Maya's live h includes its ≈ 1,020-expert lend
a prompt; `replay.py`'s `lfu` cannot fill an emptied slot, and FN's `lend_min.py` needs a Holon engine that serves
GLM. **Lending's effect on Holon's h is measured at the first Holon-GLM engine (step 5/6); the decode bands carry it
as ≈ −3%** (rev 1 panel's after-prompt dip estimate: ≈ 1,300 experts refilled over ≈ 16 tokens of a 256-token row).

| Cell (bar) | Holon | Maya | Predicted ratio: central [band] | What decides it |
|---|---|---|---|---|
| (i) decode 1K / 4K / 32K (1.0) | 22–25 ms a token | 28.1 ms (fit); 21.8–33.2 by prompt | **≤ 1.15 [0.85–1.25]** (lead call 8: two-sided until N1 lands) | Holon's fixed part (a model); Maya's calibrated lane; lending (−3%); R3 decides the lane lever at step 0's end |
| (i) prefill 1K (0.9) | routed misses ≈ 55 GB by the ring of parts' gather at ≈ 45.6 GB/s (FN §21.9) + compute under it: 1.3–1.45 s | 1.40–1.50 s engine, ≈ 1.59 s client | **1.1 [0.95–1.25]** (C-r2-m2) | the gather rate; N4's compute |
| (i) prefill 4K (0.9) | ≈ 68 GB stream 1.3–1.4 s ∥ compute ≈ 1.2 s: 1.45–1.6 s | 1.78–1.80 s engine, ≈ 2.03 s client | **1.3 [1.0–1.4]** | the same |
| (i) prefill 32K (0.9), 16K steps | 2 × max(stream ≈ 1.4 s, compute ≈ 4.7 s × c) | 10.25 s (2 × 16K); the faster `CKPT_EVERY=0` arm est. ≈ 5% less | **1.0 [0.8–1.2]** (low end −5% for the faster arm, C-r2-m3) | c = Holon's prefill compute per token against Maya's (N4; the grouping overhead at 4K passes, D9) |
| (ii) decode (1.0) | the mode (i) token + 1.5 r × 1.3 ms (r_Holon ≈ 1.5 r_Maya: +14% SSD-only experts; Maya's own data, +850 RAM experts cut reads ×7.5, C-R2-M7) | 28.1 ms + r × 1.2 ms (8 chunk workers) | **1.05–1.10 [0.85–1.2]** (r_Maya 2 → 1.13–1.18; 5 → 1.04–1.07; 8 → 0.98–1.00) | D8f's rate (N3); r (N1); the cache levers (each +100 VRAM experts ≈ −25% of Holon's reads) |
| (ii) prefill 1K (0.9) | the routed SSD-only set: a 1K window routes ≈ 80% of the coldest 1,250–1,430 (C's `ssd_routed.py`): ≈ 1,140 experts, 8.1 GB, at N3's rate, beside ≈ 55 GB of PCIe | its routed ≈ 1,000 experts, 7.1 GB | **0.95 [0.75–1.15]** (lead call 5: re-predicted with the 2,048 threshold; Holon reads +14% SSD bytes; both overlap PCIe) | both SSD rates; Holon's cache levers |
| (ii) prefill 4K (0.9) | the whole SSD-only set (a 4K window routes 98% of it): ≈ 10.2 GB | its ≈ 8.9 GB | **1.0 [0.8–1.2]** | as 1K |
| (ii) prefill 32K (0.9), 16K steps | 2 × max(SSD set ≈ 10.2 GB / rate, stream, compute) | its 2 × 16K | **0.95 [0.75–1.15]** | compute-bound if the rate ≥ ≈ 2.5 GB/s; SSD-bound below |

**Cells with a band below their bar go to the lead at step 0's end** (N4's rule), with levers and costs:
- **(i) decode** — R3's exact CPU row (decided at step 0's end); the cache levers (+4% at Maya's share); the measured
  fixed part (step 4's launch inventory).
- **(ii) decode** — D8f's rate (N3); the cache levers (each +100 VRAM experts ≈ −25% of Holon's SSD reads).
- **(ii) prefill 1K / 4K / 32K** — the cache levers; D8b's demote-into-RAM rule (fewer SSD-only experts after
  prompts); the read-ahead rule if N2 shows it pays.
- **(i) and (ii) 32K prefill** — 16K steps (D9); prefill compute (step 7: the 4K-pass grouping overhead); the tail
  skip (+2.4%).

## 8. Step 0 — measure before freezing

On the RTX 5090 desktop (and the 5080 for the reported row), one heavy job at a time; the A5000s for N4's kernel
probes.

| # | Measure | Status | Decides |
|---|---|---|---|
| N0 | The harness: Maya built at `4615f34` (sm_120) and its pack; `offload_ctx`'s Maya backend; `produced` = n; `reused` = 0; the corpus retokenized | **done** (one run: decode 35.9 / 35.6 / 35.9, prefill 646 / 2,016 / 3,195 tok/s at 1K / 4K / 32K; pool 3,360–3,377; h 0.72–0.74) | D11 runnable |
| N1 | Maya's rows: `--calibrate`, then mode (i) 5 rounds × 2 arms (its run-to-run CV), mode (ii) in a 64 GiB job step (pinned memory in `memory.peak`; the tier in experts; hygiene), the deterministic recipe; the 5080 mode (i); Maya's ledger lines | **calibrate done (kept the defaults); mode (i) running; mode (ii) queued** | the denominators; D10's tier; D11's rounds |
| N2 | Maya's route traces: decode (`STRATA_GLM_ROUTE_LOG`) and prompts (`--dump-routing`) on the bench plan and the 72 prompts; the held-out replay at each engine's share; F10 vs deferred admission in mode (ii); MTP verify windows; the routed fraction of the SSD-only set by window | **decode traces done** (bench 8,256 + tasks 27,680 tokens): h above; windows 13.9 / 19.1 / 23.9 distinct experts per layer at 2 / 3 / 4 tokens (D7). **Prompt traces queued.** Lending: not replayable at step 0 (§7.4) | D8's mode (ii) default; D7 (deferred); D9's rules; h |
| N3 | The SSD microbench on the bar node's NVMe: io_uring (with and without the bounce copy), native AIO, `pread`, an 8-thread pool; depth 8 / 16 / 32; 256 KiB–1 MiB chunks; 1–8 experts a call (`spikes/glm0/n3_ssd.c`) | **io_uring and native AIO available in the container** (kernel 6.17, `io_uring_disabled` 0); the bench queued after N1 | D8f's interface, chunk and depth; its rate |
| N4 | The roofline per §7.4 row; **the prefill compute probe** and **the live set at 4K–32K** | **A5000 done**: MoE GEMMs per 4K pass (D9), the live set unsplit ≈ 326 KiB / sub-batched ≈ 55 KiB a token beyond R at 16K (D9); **the 5090's queued** | every gated row frozen; a row predicted below its bar goes to the lead before step 1 ends |
| N5 | (½ day) FlashInfer 0.6.12's fa2 MLA with `head_dim_kpe = 0` on sm_120, and the non-absorbed prefill form, against a Triton prototype of `mla_attend` | to build | D5's dense path |
| N6 | Maya's answers to the 72 items on the 5090 (its defaults, the same rendered ids), saved as gate (d)'s reference | after N1 | gate (d) ready at step 5 |

## 9. Steps

| Step | Work | Checks | Days |
|---|---|---|---|
| 0 | N0–N6 | evidence; §7.4 frozen; escalations | 4 |
| 1 | D8 in order: (b)+(e), then (a), then (c); D8f (after N3): the one-thread async I/O worker, `io_depth`, `io_landing` and its fence, `io_threads` deleted — on the Qwen3-30B-A3B vehicle with RAM capped | `exact.py` capped vs uncapped (A5000, then the 5090 node's NVMe); **FN's rows RAM-vs-SSD identity** (B2-m2); the vehicle's offload lock trace byte-identical; the D8 plants incl. `landing_reuse_unfenced`; core diff empty | 6 |
| 2 | The oracle: `golden-glm`; both fixtures (tie wrappers, randomised init, peaked combs, power-of-two MLA dims; the Q8_0 copy); the tokenizer round-trip corpus | the HF driver bitwise vs HF's whole forward on the tiny model | 3 |
| 3 | Std units and contracts with references and CPU providers (D3–D6, D13) in their packages, `std.ffn.swiglu@1(rows)`, `gdn_rewind@1` generalised (−inf); the model plugin; the import-linter contracts | §6.1 CPU rows; **every existing lock's trace byte-identical** (`rows` = 0, the masked value); core diff empty | 6 |
| 4 | Kernels (D12): the router on both paths (`MoeTopk` included) and the clamp; KDA (chunked vendored; recurrent through the `IS_KDA` branch); mHC (split over h); `dsa_index`, `qsa_select` weights, `mla_attend`; the mixers' sub-batching; the tail skip; the launch inventory; `test_reserve.py`'s tiny-GLM case | acceptance 3; §6.1's GPU row (a′); **the vehicle's and FN's MoE locks byte-identical** after the router change; the slot-0 test | 7 |
| 5 | The full model on the RTX PRO 6000 (all-resident) then the 5090: gates (c), (d), (e), (f); exactness; the clamp bind rate, comb spread, tie counts; `route_dump` profile; Holon's own traces re-check N2, **and lending's effect on h** | acceptance 1(b), 2; §6.3's plants | 4 |
| 6 | Offload on the bar cards: mode (i), and mode (ii) with the pre-registered override set; the placement default by N2's rule confirmed; the SSD path on the node's NVMe | acceptance 2 on both cards; fit vs §7 | 4 |
| 7 | Perf closure: per-component breakdown vs Maya (nsys), fusions, CUDA rewrites by D12's rule; a per-change perf smoke | every gated row within its band, or the gap explained | 6 |
| 8 | Locks (`tune --new-model`): `rtx-5090`, `rtx-5080` | snapshot; fit vs §7 | 2 |
| 9 | The comparison (D11), a second allocation for any cell within 3% | acceptance 4 | 3 |
| 10 | Close: gates (c), (f) and `exact.py` on the final tree at each lock's step, (d) if a decode kernel changed after step 5; the GPU-sharded suite; panel code review; propagation; plan §10 | acceptance 1 (shipped tree), 5–8 | 3 |

**Schedule:** 48 days of work with MTP deferred. **Critical path** with two to three agents: 0 ∥ 1 ∥ 2 (6) → 3 (6;
step 4's kernels start beside it) → 4 (5) → 5 (4) → 6 (4) → 7 (6) → 8 (2) → 9 (3) → 10 (3) = **39 days**. Step 1 is
independent of the model and merges into the branch first (D15).

## 10. Risks

- **R1 — Mode (ii) decode rests on D8f's rate and on Holon's +14% SSD-only experts** (§7.4: 1.05–1.10 central). N3
  proves the rate before step 1 builds it (io_uring and native AIO both available in the bar node's container).
- **R2 — 32K prefill** in both modes needs 16K steps (D9) and Holon's prefill compute at Maya's: at 4K passes the
  routed GEMMs run at a grouping overhead of 1.5 (N4, A5000) against Maya's whole-chunk MMQ launch. If the 5090's
  compute probe puts c > 1.15, the rows go to the lead with the curve.
- **R3 — Maya's CPU lane** (36 experts a token at N0; calibrated at 8 threads, 0.64 PCIe share). Holon stays exact by
  default. If (i) decode's band after N1 still reaches below 1.0, the lever — an exact fast CPU row under F5b's bounded
  contract (Maya's G9) — is decided at step 0's end with the lead, not at step 7.
- **R4 — Kernel scale and races:** MLA's 64 heads × 512 accumulator in Triton on sm_120's ≈ 99 KB of shared memory;
  the indexer's 32 heads in fp32; KDA's temporaries against the reserve (`test_reserve.py` from step 4); the mHC
  write-back's cross-stream dependency (D3).
- **R5 — Holon's smaller cache** (≈ 7.5% fewer experts than Maya's measured pool): ≈ −4% decode in mode (i); +14%
  SSD-only experts in mode (ii).
- **R6 — Oracle gaps:** HF's FLA swap (venv without FLA); exact-zero ties (the wrapper, ties counted).
- **R7 — Maya moves fast** (v1.1.15 → v1.1.27 in 8 days). Pinned at `4615f34`, re-checked at the close; a newer Maya
  that changes the GLM one-GPU path re-runs N1 only with the owner's go.
- **R8 — Host RAM for mode (i):** ≈ 86 GB pinned plus the process (N0: Maya pinned 88 GiB of tiers on the bar node).
- **R9 — Shared code:** step 1 (offload, the I/O worker), step 3 (`std.ffn`, `gdn_rewind`) and step 4 (the router)
  touch paths every existing lock runs; each is held to byte-identical traces at its landing step (acceptance 2).
- **R10 — Variance:** Maya's defaults are timing-dependent; N1 measures its CV and D11 sizes the rounds.
- **R11 — Lending's effect on h** is unmeasured until step 5/6 (§7.4 carries −3%).
- **R12 — Disk:** 275 GB free on scratch after the download; the bar node's NVMe has 2.2 TB free (N0).

**Deferred:** K4 (Q12); MTP (D7); an exact fast CPU row (R3); Maya-S24 / M / L; DFlash2; `--parallel` batching rows
(`max_seqs` 1 in the locks); the vision tower.

## 11. Lead decisions applied and owner questions

**The lead's calls on rev 1's review (2026-10-25):** 1. per-cell predictions with bands, Maya from measurement; 2. the
SSD path (one thread, async, depth 16–32, ≤ 1.3 ms, proven by N3 first; the routed read-ahead dropped); 3. 16K steps
with sub-batched activations, N4's probes, `test_reserve.py` from day one; 4. 1K prompts stage routed experts; 5. the
router as a real, row-invariant kernel change; 6. MTP selects fresh per sub-step, deferred unless ≥ 1.0×; 7. F10's full
requirement set and step 1's order; 8. KDA in `std.mixers.gdn`, DSA in `std.mixers.qsa`, `qsa_triton` extended;
9. gates on the shipped tree, plant verdicts fixed, `mhc_sinkhorn_short`, the cover on decode rows; 10. mode (ii)'s
limit, override set, hygiene, the tier in experts, one step for identity; 11. the run as the unit; 12. the measured
ledger, launches reported; 13. Q12.

**The lead's calls on rev 2's review (2026-10-25):** 1. sub-batching opt-in (`std.ffn.swiglu@1(rows = 0)`, set by
GLM's recipe), the whole KDA mixer and MLA inside `dsa_mla`, the bit claim corrected, the loan's class named, every
existing lock byte-identical at the landing step (D9, D4, D5); 2. the prompt path's `MoeTopk` too, the chain-count rule,
the `% 64` predicate, the vehicle's and FN's MoE locks byte-identical, a bounded L0 (D6); 3. the GPU tiny run on a
Q8_0 copy at H = 512, power-of-two MLA dims, a relative metric, plants measured before pre-registration, IQ kernels
L0-only (acceptance 1a′, D2, §6.3); 4. KDA's recurrent path through the patched `IS_KDA` branch, the slot-0 test (D4);
5. per-layer-in-flight landing sets in `io_landing` with a device-side fence and its plant, one 2,048 threshold, the
(ii) 1K cell re-predicted (D8f, §7.4); 6. the mHC write-back split over h, a kernel run-identity test (D3);
7. native AIO as the automatic fallback, no SQPOLL, `io_threads` deleted, `io_depth` an option, FN's rows identity at
step 1, alignment per piece (D8f, §9); 8. two-sided (i) decode band, (ii) decode central 1.05–1.10, lending not
replayable at step 0 (§7.4); 9. the tail skip runs the FFN on the logit rows, never skips the MoE call (D9); 10. 16K in
both modes, off the override list (D9, D10); 11. the remaining minors as proposed, the core cap against the merge-base
(acceptance 6).

**Owner questions** (each with the recommended default, which the build follows until answered):

| # | Question | Recommended default |
|---|---|---|
| Q1 | Gate both host modes on the 5090 (12 cells), mode (ii) as a 64 GiB job-step memory limit on a larger desktop? | **Yes** |
| Q2 | Mode (ii)'s RAM tier equal in experts for both engines, sized to fit both under the limit? | **Yes** |
| Q3 | MTP deferred (N2 predicts ≈ 0.89–1.09× plain); built only if a later measurement predicts ≥ 1.0×? | **Yes (defer)** |
| Q4 | The 16 GB row: RTX 5080, Maya-S, mode (i), reported (not gated), with a lock because it is the development lock class? | **Yes** |
| Q5 | Maya's 32K cells: gated against the faster of its default and `CKPT_EVERY=0`? | **Yes** |
| Q6 | The prefill bar stays 0.9× (v1-offload §1.4; FN closed with it), decode 1.0× in tok/s? | **Yes** |
| Q7 | Holon stays exact (GPU-only) by default; a CPU split only by `docs/v1-offload.md` D4's rule, reported; R3's lever decided at step 0's end? | **Yes** |
| Q8 | Core: 0 planned, a hard cap of +5 lines against the merge-base, each line escalated to the owner? | **Yes** |
| Q9 | Disk: Maya-S only (done, 96.5 GB); no HF FP8 checkpoint (328 GB); keep FN's GGUFs (≈ 115 GB) for FN's propagations? | **Yes** |
| Q10 | Mode (ii) = the 5090 lock + one pre-registered override set (file tier, `moe.ram` in experts, `--host-memory`, placement, `io_depth`; D10), not a lock keyed on host RAM? | **Yes** |
| Q11 | Maya's arms at its own `--calibrate` result on each card (on the 5090 it kept its defaults)? | **Yes** |
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

### Rev 2 → rev 3 (the same panel)

Rev 2 drew A 5 majors + 4 minors (A's rev-1 M1 and M6 partly open, the rest resolved), B 1 major + 8 minors (all 26
rev-1 findings resolved), C 7 majors + 9 minors (every rev-1 finding answered; five answers incomplete). The lead's
calls (§11) settle the choices. Dispositions:

| Finding | Disposition |
|---|---|
| A-N1 the prompt path's `MoeTopk` stays softmax | **Accepted** (D6): `MoeTopk` and `holon_ggml_topk_moe` take scoring, bias, scale; plant `router_prompt_softmax`; the partial-chunk `F.linear` stated |
| A-N2 (a′) cannot reach the deployed paths; its bound is fragile | **Accepted** (acceptance 1a′, D2, §6.1, §6.3; lead call 3): a Q8_0 fixture at H = I = 512, power-of-two MLA dims, logit std ≥ 2, KL bound from the clean run; plants measured before pre-registration; IQ kernels L0-only |
| A-N3 the pin's `fused_recurrent.py` skips slot 0 | **Accepted** (D4; lead call 4): the recurrent path through the patched `IS_KDA` branch; the one-sequence slot-0 test; plant `kda_slot0_skipped` |
| A-N4 landing-unit reuse unfenced with read-ahead | **Accepted** (D8f; lead call 5): per-layer-in-flight sets in `io_landing`, the device-side "copied" signal, plant `landing_reuse_unfenced`; landed read-ahead found by the next resolve |
| A-N5 the mHC write-back races when split over streams | **Accepted** (D3, D12, acceptance 3; lead call 6): in place only with column h of all streams in one CTA, else out of place; a kernel run-identity test; plant `mhc_writeback_stream_split` |
| A-n1 D9's live set needs the whole KDA mixer sub-batched, state carried | **Accepted** (D4, D9): the whole mixer, boundaries multiples of 64, conv window and state carried; N4 measured the sub-batched live set |
| A-n2 Q6_K → BF16 for `attn_k_b` / `attn_v_b` is lossy | **Accepted as a recorded deviation** (D5): the tensors are BF16 in the source checkpoint and Maya's pack rounds them the same way; L0's reference takes the rounded weights; f32 (+0.27 GB more) or in-kernel Q6_K is not worth ≈ 40 experts of cache for a rounding the reference engine shares |
| A-n3 the cover's `mhc_sinkhorn_short` verdict depends on the real combs | **Accepted** (acceptance 1f, §6.2, §6.3): step 5 records the spread; the cover's verdict is conditional |
| A-n4 one masked value for `gdn_rewind@1` | **Accepted** (D4): −inf for both (the same bits as −1e4 for GDN), checked byte-identical at step 3 |
| B2-M1 which units sub-batch; the bit claim | **Accepted** (D9, D4, D5, §4; lead call 1): `std.ffn.swiglu@1(rows = 0)` set only by GLM's recipe, the mixers inside `kda` / `dsa_mla`; "deterministic at a step size, not identical to unsplit"; every existing lock byte-identical at the landing step |
| B2-m1 the depth is an option; delete `io_threads`; an automatic fallback | **Accepted** (D8f) |
| B2-m2 the I/O change also moves FN's rows SSD reads | **Accepted** (§9 step 1): FN's rows RAM-vs-SSD identity |
| B2-m3 "no bounce copy" needs an exact rule | **Accepted** (D8f): direct only when offset, length and destination are 4 KiB-aligned; landing units hold the superset, the fetch reads at the piece's offset; the bounce into packed RAM units is on the submitting thread and in N3 (`uringb`) |
| B2-m4 one request buffer for read-ahead depth 2; landing pool unnamed | **Accepted** (D8f): request buffers and landing sets per layer in flight (`g % depth`) in the named pool `io_landing`; threshold and depth constants |
| B2-m5 the router's chain and tile rule for the existing E | **Accepted** (D6): the largest divisor of hidden / 32 ≤ 2048 / E; the predicate per E's tile; the vehicle's and FN's MoE locks byte-identical at step 4 |
| B2-m6 the tail skip's field and the store's per-layer protocol | **Accepted** (D9, §4): `DeviceBatch.logits_index`; the FFN half on the logit rows; the MoE call never skipped; conditions trimmed; the last block by a recipe parameter |
| B2-m7 the step size in both modes vs the override set | **Accepted** (D9, D10; lead call 10): 16K in both modes; off the override set |
| B2-m8 core count base; F10 heat dtype; provenance | **Accepted** (acceptance 6, D8a, D5): the merge-base with `main`; `moe_tiers@1` fact `heat = "i32"`; `dsa_index` / `mla_attend` in their own file |
| C-R2-M1 the (i) decode band one-sided and its central optimistic | **Accepted** (§7.4; lead call 8): central ≤ 1.15, band [0.85, 1.25]; Maya's denominator from the 16-prompt fit (28.1 ms); (i) decode at risk for R3's step-0 decision |
| C-R2-M2 (a′) on an f32 GGUF cannot reach mmvq / mmq; MLA dims not powers of two | **Accepted** (as A-N2) |
| C-R2-M3 16K needs whole-mixer sub-batching; the loan needs a class | **Accepted** (D9; lead call 1): N4 measured ≈ 119 KiB a token at 16K sub-batched (R included) against ≈ 390 unsplit; `test_reserve.py` asserts ≤ 150 KiB; the loan from the gate/up class's tail |
| C-R2-M4 the `preadv2` fallback cannot reach depth | **Accepted** (D8f; lead call 7): native AIO, chosen automatically; no SQPOLL; N3 benches both (both available in the bar node's container) |
| C-R2-M5 the 1,024 read-ahead threshold contradicts staging below 2,048 | **Accepted** (D8f; lead call 5): one threshold, 2,048; the (ii) 1K cell re-predicted (0.95 [0.75–1.15]) |
| C-R2-M6 Holon's h leaves out lending; the lend replay cannot run | **Accepted** (§7.4, R11; lead call 8): stated; measured at the first Holon-GLM engine (step 5); the bands carry −3% |
| C-R2-M7 (ii) decode assumes equal SSD reads | **Accepted** (§7.4): r_Holon ≈ 1.5 r_Maya; central 1.05–1.10; the cache levers listed for this cell |
| C-r2-m1 the router L0 cannot be bitwise under fast-math | **Accepted** (D6): bounded, in today's form |
| C-r2-m2 (i) 1K central ≈ 1.1 | **Accepted** (§7.4) |
| C-r2-m3 the 32K denominator is the faster arm | **Accepted** (§7.4): the band's low end −5% until N1 measures `CKPT_EVERY=0` |
| C-r2-m4 `dsa_decode_select_shift`'s runner includes the cover | **Accepted** (§6.3) |
| C-r2-m5 the mHC cover should compare `comb` | **Accepted** (acceptance 1f, §6.3) |
| C-r2-m6 "the two medians pooled" undefined | **Accepted** (D11): the median of both allocations' pair ratios together |
| C-r2-m7 Maya's "GB" are GiB; the cache gap | **Accepted** (§7.1): ≈ 7.5% fewer experts, ≈ −4% decode; levers listed |
| C-r2-m8 landing alignment | **Accepted** (as B2-m3) |
| C-r2-m9 cite the 16-prompt capture | **Accepted** (§7.4) |

No finding was rejected.

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
