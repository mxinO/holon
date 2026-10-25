# v1 — GLM: GLM-5.3-Flash on Holon, then its expert offload compared with Project Maya

Status: **design closed 2026-10-25, rev 10, panel-clean** (panel B confirmed B9-M1's fix; B's build notes B10-m1 and
B10-m2 are in §4 and D9). **The build proceeds per §9** (step 1's rest — the parking commit, D8f's SSD path with its
landing pool, then F10 and promote's `lend_tail` — then steps 2 on), the panel reviewing code at each step's close. Rev
9 drew A 2 minors + 2 nits, B 1 major + 2 minors, C 3 minors; §13 answers each with the lead's calls (§11). (Rev 8 drew
A 1 + 2, B 2 minors, C 1 + 2; rev 7 drew A 2 majors, B 1 + 2, C 2 + 2; rev 6 drew A 1 major + 2 minors, B 1 + 1, C 2
minors; rev 5 drew A 2 + 1 + nit, B 3 minors, C 1 + 1; rev 4 drew A 1 major + 2 minors + 1 nit, B 1 + 3, C 2 notes; rev
3 drew A 4 minors + 3 nits, B 2 majors + 2 minors, C 1 major + 2 minors; rev 2 drew A 5 majors + 4 minors, B 1 + 8, C 7
+ 9; rev 1, A 6 + 12, B 9 + 17, C 3 blockers + 10 + 13.) Step 0 is under way (N0, N1's mode (i), N2's decode and prompt
traces, N3 and N4's A5000 probes done; §8). In the engine only step 1's (b)+(e) is built. Parameters marked "est."
freeze when step 0's evidence lands.

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
2. **Exact and run-identical:** at **one fixed `max_step_tokens`** (32K, the step of both modes, D9), snapshot traces
   are byte-identical across fits (all-resident `std.moe@1` on the RTX PRO 6000, the deployment's cache, a capped
   cache), across policies (`lfu`, deferred admission, F10 spares), with lending and the ring on and off, and between
   RAM holding every expert and RAM short (exclusive tiers + SSD). Run-identical ×3 on the RTX 5090 and the RTX 5080
   at their locks. **Every existing lock advances byte-identical** at the step that lands a shared change (D9, D6, D4).
3. **Kernels:** L0 for every new kernel against its `reference.py` on sm_86, sm_89 and sm_120, memcheck clean; every
   decode step graphed; a kernel-level run-identity test for every in-place multi-CTA kernel (A-N5). **Launches per
   decode token are reported, not gated:** step 4 takes a launch inventory on a 4-layer slice (FN's method) and the
   count × the measured per-launch cost enters the roofline. `plugins/tests/test_reserve.py` covers tiny GLM from its
   first kernel, including a multi-row step past `SUB_TOKENS` with KDA, and asserts **≤ 150 KiB of live activations
   per token** (R included) on a sub-batched step at 16K and at 32K (C-R2-M3; N4 measured ≈ 119 / ≈ 113 KiB on the
   A5000).
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
5. **Memory** (§7): the 5090 fits at 33,792 tokens with 32K steps in both modes; the 5080 at 33,792 tokens in mode
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
    dims (kv_lora 128, nope 64, v 32; C m2), KDA 2 heads × 128 (pow2 head dim, heads = attention heads as LC
    asserts), `lm_head` scaled to a logit std ≥ 2 (A-N2: at HF's init the median top-1 gap
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
    `kernels.fla` with `vendor.patch` dropping their vLLM imports and `kda.py`'s `.fused_recurrent` import with
    `fused_recurrent_kda` (A nit: the file is not vendored); FN's `SUB_TOKENS` packing and in-step snapshots in
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
- **The whole mixer is sub-batched** (D9): the unit runs its projections, convs, gates, kernel, gated norm and
  `o_proj` over row sub-batches of `rows` (a recipe value; 0 = off), boundaries multiples of 64, carrying the conv
  window and the recurrent state across sub-batches. **The shared `Planner`'s in-step snapshot rows and conv publishes
  are re-based per sub-batch only behind `rows > 0`** (B3-m2), so GDN's path is untouched; GDN's locks advance
  byte-identical at the landing step. **The bitwise test covers the recurrence only** (A-m3): `chunk_kda` with the
  carried f32 state and conv window at 64-multiple boundaries against one call; the projections and `o_proj` are
  `qlinear` calls whose bits may depend on the row count.
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
it.** The store gains **one option**, `moe.read_ahead` (D8f; B7-m1); provider options live in their kernel plugins.
- **(b)+(e) as one change — lending and the prefill ring under exclusive tiers, and the staging that follows.**
  Deleting `or self.exclusive` (`store.py:240`) and `not self.exclusive` (`:435`) needs: a lent piece is demoted
  first — to a free RAM unit, else over RAM's coldest expert when hotter (Maya's `LEND_PARK_LRU`), else SSD-only —
  before its slot is overwritten; staging per class becomes `k × max_seqs × (1 + drafts)` (8 at GLM's lock),
  returning 2.94 GB of VRAM and as much pinned host memory in mode (ii). Pipelined parts stay as in the RAM-holds-all
  layout. Tested on Qwen3-30B-A3B with `moe.ram` capped (`exact.py`: capped vs uncapped identical) and the plant
  `exclusive_lent_not_demoted`, forced to bite by a RAM tier small enough that the stale unit is reused.
  - **Built first (step 1, rev 8):** a prompt step under exclusive tiers borrows its staging from the cache and a lent
    piece goes **SSD-only** (its code −2: the file always holds it, so its slot is free at once; no RAM unit is
    written); the graphed steps' staging slots return to the decode bound; the host I/O staging units stay a pass's
    until D8f sizes them. Exact on the GPU tiny model at one sequence (`test_moe_engine.py`'s `ssd1`: bitwise the
    resident store, borrowing asserted; it fails on the store before the change) and on the vehicle (A5000, one
    sequence, `moe.ram = *=0.25`, prompts of 5,000 / 700 / 3,000 tokens and 64 generated each: capped vs uncapped
    identical in streams and top-3 prompt logprobs, the capped arm borrowing its staging in 9 prompt steps). Under
    exclusive tiers the ring stays off (it copies RAM units only) and pipelined parts stay off (`resolver.py:81-82`)
    until D8f, which runs SSD rounds before the parts. **Parking a lent piece in RAM** by the rule above is the item's
    next commit (a `moe_tiers@1` demote entry); C's loan model assumes it, and the SSD-only count is the same either way
    (parking moves RAM's coldest to the SSD instead). **The host mirror records a lent piece's code** (−2 under
    exclusive tiers; B8-m1, `bbe68fd9`): `ssd1` checks the mirror equals the device map, tier codes too, once its prompt
    steps are committed (it failed before). **The parking commit makes the mirror tier-exact** under exclusive tiers:
    a parked lend, like `moe_tiers@1`'s demotions and keeps, places pieces unlogged, so the step's map rides in its
    telemetry and the commit adopts that map, still replaying the step's records for the per-layer empty-slot lists
    (≈ 100 KB a step at GLM's sizes); every mirror reader still reads slots ≥ 0 only
    (the policies' `after_commit`, the bench's tap, `route_dump`'s `device_is_mirror`).
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
  - **`lend_tail`** in `placement.promote@1`'s resolve (rev 9, B7-M1's first option, the lead's call): its CPU
    reference and `kernels.promote`'s CUDA resolve release the policy's spares in the tail's range and unmap the
    experts holding it, as `moe_resolve.lfu@1` does (`kernels/expert_io/plugin.py:81-89`; the contract's spares
    clause, `moe/kernels.py:121-130`); an L0 against the reference and `exact.py` (F10 + exclusive tiers + tail lending,
    capped vs uncapped) — so F10 takes D9's 32K loan; the reference asserts ownership after a lending step's return
    (no slot both a spare and on the layer's empty stack or lent; A's nit);
  - plants: `promote_spare_unlogged` and `promote_victim_code`.
- **(c) The high-priority split fetch under exclusive tiers:** two fetch launches — the RAM misses at once at high
  priority, the SSD misses after their read completes — so a RAM miss never waits behind an SSD read.
- **(f) The SSD path, rebuilt for decode and prefill** — D8f below.
- **(d) Aux experts under exclusive tiers — only if D7 builds MTP:** the residency, piece and owner tables move to
  `before_materialize`; the `cpu_task` I/O worker and each group's read binding stay at `resolver()`.
- **(g) The CPU split (F5b)** stays reported only, decided at step 0's end (R3).
- **Placement defaults:** mode (i) `lfu` (F11); **mode (ii) F10 spares with the keep rule, once (a)'s `lend_tail` is
  built** (rev 9, the lead's call; B7-M1's first option): C's held-out replay at GLM's fit reads **19–23% fewer SSD
  experts a token** than F11 with deferred admission and keep (`local/tmp/glm-panel-c/ssd_cap_replay.py`,
  `mode2_decode.py`), ≈ +8% on the (ii) decode cell. Until `lend_tail` lands, mode (ii) runs F11 with deferred
  admission and keep (it has `lend_tail`; the store refuses `ring_lend = tail` without it, `store.py:221-223`). Step 6
  runs both with prompt lending and reports the other.
- **The seed** is a GLM routing profile from `route_dump` on held-out prompts (FN D12f), shipped in the bundle.

**D8f — The SSD read path: one submitting thread, asynchronous I/O, depth 16–32.**
- **Where** (B3-M2, lead call 2): the reads are issued by `kernels.cpu_task`'s I/O worker (the provider behind
  `std.tiered`'s file tier, `std/holon_std/tiered/gguf_direct.py`, and `std.tiered.rows@1`'s SSD placement), whose
  pread pool is replaced by **one submitting thread**; `cpu_task@1` gains one additive flag, `bind(task, urgent=False)`,
  **a hint** (B8-m2): a worker serves an urgent slot's request before requests it has not yet started, and may ignore
  it (the compute pool does). **The landing memory, its
  sizing and its fence are `std.tiered`'s** (below): the worker owns no store state.
- **The interface, chosen automatically at worker start and logged** (C-R2-M4, B2-m1, lead call 7): **io_uring** (raw
  syscalls, no liburing, **no `IORING_SETUP_SQPOLL`** — its kernel thread spins a core, FN §22.1's −32%; the submitter
  blocks in `io_uring_enter`), else **Linux native AIO** (`io_setup` / `io_submit` / `io_getevents` on the O_DIRECT
  fds), which default container seccomp profiles allow where they block io_uring. No option chooses it.
- **Options:** `cpu_task.io_depth` (default 16; a plugin option) replaces `cpu_task.io_threads`, which is **deleted**
  (an option removal, noted in the plugin's changelog; no lock under `plans/` sets it; `state_tier.io_threads` is a
  separate option and stays); `io_cores` reduces to the submitting thread's one CPU. **The worker takes a constructor
  argument that forces a backend** (not an engine option; B3-m1), and the copy-task L0 runs under io_uring and under
  native AIO.
- **The worker is completion-driven** (A R8-m2): one loop submits each request's extents as chunks (N3: the chunk
  size does not matter on the bar drive), up to `io_depth` in flight, reaps completions, answers a slot once all its
  request's chunks are in, and looks for new requests between submissions. **An urgent request goes ahead of every
  chunk not yet submitted**, so a large read-ahead in flight delays an on-demand round by at most `io_depth` chunks
  (≈ 1.6 ms at 16 × 256 KiB and 2.57 GB/s). Its test: an urgent round answered while a large read-ahead is in
  flight, before that read-ahead's DONE. RAM misses go first in their own launch (D8c).
- **Landing: a landing pool per class, at the superset width; the I/O bounded, not placement** (rev 10: B9-M1, the
  lead's call; rev 7: A R6-M1, B6-M1): every SSD read — kept pieces included — lands in a **landing pool of its own per
  class** in `std.tiered.pinned` (mapped, pinned, under `host_memory`), its rows at the class's superset width, in two
  disjoint ranges: **on demand** `[0, A_od)` and **read-ahead** `[A_od, A_od + depth × A_ra)` (depth sets, set g on
  `g % depth`). **The RAM pool stays exactly as today** — canonical `unit_bytes` rows, its I/O staging units no longer
  used — and so does every RAM-miss gather (the main fetch, the staging ring of parts, D8c's split) and
  `cpu_experts@1`'s whole-unit reads.
  - **Placement is unchanged:** `moe_tiers@1` places exactly as today — no per-layer cap, no window rule. (Rev 6's cap
    was infeasible under lending: a 16K step's loan sends ≈ 1,400–1,850 more experts out of VRAM, ≈ 70 SSD-only a
    layer against A ≈ 51, and undefined for fill arenas, where the cache size is unknown at `before_materialize`.)
  - **Two sizes, staging choices, not invariants** (rev 8: C R7-M1, B7-m1); nothing's correctness depends on either:
    - **A_ra, the read-ahead set:** the store option **`moe.read_ahead`** (units a class and layer), generic default
      the decode bound (top_k × max_seqs × (1 + drafts)). GLM's lock sets it at step 0's end to **the largest
      per-layer SSD-only set after the measured prompt loan** (N2's prompt traces, C's replay method: max 101–127
      after a 16K loan, ≈ 135 after a 32K one with the ring's halves; the ring of parts lends ≈ 540 fewer, so est.
      ≈ 120–135); it is inert without a file tier, and D10's override set lists it. At that size a layer's set past
      A_ra is rare; at ≈ 70 it would expose 9–19% of every mode (ii) prompt step to on-demand reads (C).
    - **A_od, the on-demand range:** max(32, the decode bound), not an option; it takes a spill past A_ra and every
      call that is not read ahead, in rounds. **Bind asserts A_od ≥ the decode bound** (8 at GLM's lock) and refuses
      the layout otherwise.
  - **Read-ahead** (a prompt step ≥ 2,048 tokens **with no decode or verify row**): the **first A_ra of the next
    layer's SSD-only set, lowest ids first** (deterministic), 2 layers deep; **layers 3 and 4 (the first two MoE
    layers) are read at the step's start**, under the dense layers 0–2's ≈ 0.3 s of compute (C r7-m2), each later
    layer g + 2 after layer g's last pass.
  - **No read-ahead in a step that carries a decode or verify row** (A R7-M2, the simplest fix): `moe_tiers@1`'s keep
    rule lands a kept SSD miss's read in its RAM unit (`moe/kernels.py:180-183`), and a read-ahead expert's bytes sit
    only in its read-ahead unit, so a kept read-ahead expert's RAM unit would never be filled. Such a step (`max_seqs`
    > 1, a long chunk beside a decode row) reads on demand; at the locks' `max_seqs` 1 it never arises.
  - **On demand: the rest, in rounds, in eager prompt steps only.** A call's SSD misses not read ahead (a set past
    A_ra; anything that turned SSD-only after the snapshot, such as a mixed step's admission, A R6-m2; prompt steps
    below 2,048 tokens; steps with a decode row) are read in rounds of at most A_od units.
    - **Outside the ring** (rev 9: A R8-M1 + C R8-M1, the lead's call): a call's SSD entries — its on-demand rounds
      and its read-ahead hits — get device staging slots **outside the ring**, a region of A_od slots a class (A_ra +
      A_od in a step that reads ahead) after the ring's, lent with the step's staging: counted in the loan's need and
      in `test_reserve.py`'s tiny-GLM cases (≈ 0.23 GB at A_od 32, ≈ 1.1 GB with A_ra 120: ≈ 30 / ≈ 150 more experts
      lent in a mode (ii) prompt step). The ring's slots carry RAM misses only.
    - **The order of a layer call** (C's): (1) round 0 is signalled at the layer's resolve; (2) the RAM misses run
      through the staging ring of parts, and the read-ahead hits, gathered on main into their outside slots, are
      computed with the resident entries (part −1); (3) then, on main, round 0 is waited, gathered into its outside
      slots and computed as its own part; round r + 1 is signalled after round r's gather and computed after round r
      (the same slots, main-stream order). **No part ever waits on a round**, so A R7-M1's cycle cannot form, and the
      round's read overlaps the RAM stream (C: serialized, the (ii) 1K cell would fall from ≈ 1.2 to ≈ 0.9).
    - **Rounds of exactly A_od** (A R8-m1): `moe_tiers@1` writes the call's SSD-miss fetch entries in rank order j,
      contiguously after its RAM misses (D8c's split); entry j lands in on-demand unit `capacity + (j mod A_od)` in
      round ⌊j / A_od⌋ (a kept piece in its RAM unit, as today). Round r's gather is `expert_fetch@1`'s
      `gather_roles(..., part = r, size = A_od)`, a small additive form (a fixed part size; §4). Before round r's
      signal, `std.tiered` copies round r's count and pairs from the full list into the on-demand view.
    - **No host sync:** R, the call's rounds, is its bound ⌈min(E, k × T) / A_od⌉ (9 at GLM's E = 288, A_od = 32);
      a trailing round is empty — count 0, answered DONE at once, an empty gather and part.
    - **A graphed step never needs a second round** (its misses ≤ the decode bound ≤ A_od, asserted in the graph's
      resolve: a second round in a graphed step sets the store's error word).
  - Counted under `host_memory` and mode (ii)'s limit: the landing pool's (depth × A_ra + A_od) rows a class,
    **≈ 272–302 at A_ra 120–135 (≈ 1.9–2.1 GB pinned; §7.1)**; the RAM pool's I/O staging units go (`moe_tiers@1`'s
    staging indices name landing rows now).
  - **No bounce copy** (rev 9: D8f's own trigger fired, N3: the bounce costs 7% of a one-expert read; rev 10's
    layout, B9-M1). Each extent — **a gate/up piece is two extents (the gate and up tensors), a down piece one** — is
    read as its 4 KiB-aligned superset straight into its landing unit, and **the SSD fetch launch alone** (D8c's split:
    its sources the landing pool) reads each extent at its offset there through a per-piece run table:
    `expert_fetch@1`'s std minor (§4). **Sizing is general** (A's nit): a landing row is the sum over the piece's
    extents of each superset's bound, ≤ the extent + 2 × 4 KiB − 2 bytes (GLM's slices are 4 KiB multiples, so +4 KiB
    an extent there), rounded to 4 KiB; each extent's destination is asserted 4 KiB-aligned.
  - **A kept piece** (D10's keep rule) is read into its landing unit like any other SSD miss, and **the submitting
    thread copies it landing → its canonical RAM unit** once its extents are in, before it answers the request (a
    7.1 MB copy, ≈ 0.5 ms, beside the request's other reads; decode keeps only). `moe_tiers@1`'s `io` rows name the
    landing unit; its keep moves the RAM unit's owner as today. One launch, one source pool, one layout (B9-m1).
- **Two slots a layer; the fence is stream order** (A R5-M2, R6-m1):
  - **Slots.** Each MoE layer has an **on-demand** `cpu_task` slot, signalled **once per round of an expert pass**,
    and a **read-ahead** slot, signalled **once per step, after the layer's last pass** (a partial-chunk pass still finds
    its read-ahead units intact). (`cpu_task@1`'s wait passes on `word ≥ DONE` with no sequence number,
    `cpu_task.cu:4-8, 407-419`.)
  - **Only the main-stream gather reads SSD units** (rev 9, A R8-c1): read-ahead hits and on-demand rounds are
    gathered on main into the outside slots, never by a part, the ring or the split fetch. Every wait is on main after
    its own signal, so a wait can never pass on an earlier request's DONE; the next round's signal follows the previous
    round's gather in main-stream order.
  - **The read-ahead signal** for layer g + depth is enqueued on main at the end of layer g's MoE call, after its last
    pass (and, with F5b on, its CPU share's wait).
  - **Request rows in their own views:** the store's request pool (`io_pool`, `store.py:402-403`) becomes 1 + depth
    views — one on-demand view, waited within its round before the next round writes it, and depth read-ahead views,
    group g's read-ahead task bound to view `g % depth` — in `std.tiered`, no contract change.
  - Plants, each must make `exact.py` differ (or the store's check fail) on a RAM-short prompt with the staging ring
    of parts and a capped RAM tier with the maximum loan: `landing_reuse_unfenced` (the read-ahead signal before the
    reading layer's gather), `ssd_slots_in_ring` (rev 8's form: SSD entries staged in ring slots the RAM parts
    overwrite), `ondemand_round_unjoined` (round r + 1 signalled before round r's gather; its target configuration
    forces two rounds: a 250-token chunk with parts on and the on-demand range forced small by a test-only
    constructor argument), `graphed_round` (a graphed step forced past one round: the store's error word), and
    `keep_read_ahead_unfilled` (A R7-M2's case: `max_seqs` 2, a 4K chunk beside a decode row, keep on), which **must
    pass**: the case cannot arise. Rev 8's `landing_reuse_side_stream` and `landing_fill_unjoined` are dropped: no
    side stream reads SSD units now. **The two-round test runs under `exact.py` against the uncapped run.**
- **A read-ahead expert is a plain SSD miss for placement** (A-m2): admission, the `keep` rule and heat see it exactly
  as an SSD miss; "already read ahead" changes only which staging unit the fetch reads and that no on-demand read is
  issued. So placement does not depend on whether the read finished first.
- **Prefill read-ahead, one threshold:** for steps **≥ 2,048 tokens** (a constant, the line D9 used for its staging
  until rev 9), each MoE layer reads the next layer's SSD-only set ahead (its first A_ra: the whole set at GLM's A_ra,
  sized to the largest), 2 layers deep; below 2,048, each layer reads its own routed SSD-only set once routed, in
  rounds. Threshold and depth are constants.
- **Target: the drive's rate** (rev 9). N3 measured the bar node's NVMe at **2.57 GB/s** (≈ 2.77 ms of bandwidth a
  7.1 MB expert; 2.86 ms for a one-expert call) on io_uring, native AIO and an 8-thread pool alike, so rev 1's 1.3 ms
  cannot be met there; both engines read the same drive. The engine's SSD path is held to within 5% of N3's rate at
  k ≥ 4 experts a call (step 6).
- **Core:** none.

**D9 — Prefill: one 32K step per prompt in both modes (passes of 16,384), with opt-in sub-batched activations.**
- **Why:** N0 measured Maya's 32K prompt at 3,195 tok/s, read as 2 × 16K chunks that are compute-bound (≈ 0.29 ms a
  token; its copy lane busy 4.3 s of a 4.7 s chunk). A 4K-step Holon pays the ≈ 68 GB stream eight times; from 8K
  steps it hides under compute (§7.4). **One step size, 32K (`max_step_tokens` 32,768), in both modes** (rev 9, the
  lead's call; B2-m7), passes of 16,384 rows: a 32K prompt is one step, so mode (ii) reads its SSD-only set once
  (C's mode (ii) analysis: two 16K steps read it twice, ≈ 0.53–0.64 against Maya's one-chunk arm; one step ≈ 0.86–1.0).
  It is not in mode (ii)'s override set, so acceptance 2's identity is at 32K. **Mode (i) is checked not to lose**:
  step 0's and step 4's data show its 4K and 32K prefill rows and the decode transient after a prompt (≈ −1–2% at
  32K, C) at 16K vs 32K steps; if mode (i) loses, 32K becomes a mode (ii) override and the identity claim stays at
  16K, said so in the evidence.
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
  0.73 / 0.86 / 1.55 GiB at 4K / 8K / 16K / 32K** — with R's 64 KiB, **≈ 119 KiB a token at 16K (1.9 GiB) and
  ≈ 113 KiB at 32K (3.55 GiB)**, inside acceptance 3's 150 KiB. The routed experts' pass buffers are the store's (lent).
- **The loan:** what exceeds the reserve during a prompt step is lent by `moe.lend_peak` from **the gate/up class's
  tail** (the largest class, ≈ 13.6 GB on the 5090), never from the IQ3_XXS down class (≈ 2 GB; FN §21.12's 8K loan
  outgrew its class and the store refused the layout). **At a 32K step** the activation loan is ≈ 5.06 GiB (3.55 GiB
  live against the 1.02 GiB reserve, × `LOAN_MARGIN` 2.0), ≈ +3.3 GiB and ≈ +500 lent experts over a 16K step; the
  tail must hold it beside ≈ 2.2 GB of pass buffers and its staging (step 4 checks it; `test_reserve.py` gains a 32K
  case). **The lock refuses to start without it** (C r9-m3): where the tail cannot lend the loan, the store today
  drops it and reserves the worst step's peak (`store.py:699-702`: 3.55 GiB, ≈ −420 experts, silently); GLM's lock
  sets `moe.lend_peak = require`, a new value of the existing option under which the store raises instead, on
  **every** path where `lend_activations` would lend nothing, with a message naming the condition (B10-m2). Step 1
  restores `ring_tokens` under exclusive tiers first (its ring off there today), and a test starts a mode (ii)-shaped
  layout with `require` and sees it lend.
- **FN's lesson (§22.9):** every sub-batched op keeps the reserve true for a step holding several prompts' rows;
  `test_reserve.py`'s tiny-GLM case covers it from step 4 (acceptance 3).
- **Mode (ii): the staging ring of parts at every step size** (rev 9, the lead's call; `moe.stream_tokens` above
  the step, in D10's override set): it lends ≈ two parts of a layer's misses instead of the ring's two halves, ≈ 540
  fewer experts a step, each one SSD read fewer (C); its gather (≈ 45.6 GB/s against the ring's 53) hides under a
  16K–32K step's compute. FN proved it bitwise the resident store (§21.16).
- **1K prompts: stage the routed experts, not the ring:** steps below 2,048 tokens use FN's ring of parts
  (`moe.stream_tokens = 2048`, `moe.pipeline = 8`). N0: a 1K prompt routes 89% of all experts, so the saving is ≈ 10%
  of the bytes, more in mode (ii).
- **The MoE pass size** (C R3-M1, B4-M1, A R4-m2, lead call 2): mmq computes each expert's rows in 64–128-wide column
  tiles; at a 4,096-row pass a layer's routed GEMMs fill ≈ 0.61–0.63 of them on real routing, at 16K ≈ 0.87 (C's
  `tile_fill.py`, `pass_fill.py`; Maya's 16K chunk ≈ 0.88).
  - **Measured, A5000** (N4, `gemm.py` on Maya-S's layers 10 and 3, through a spike copy of the plugin at the larger
    pair limit, outside the tree): per token per layer, **real routes (N2's tasks trace): 10.50 / 8.24 / 7.41 µs at 4K
    / 8K / 16K passes — the 16K pass 1.42× faster** (C's fill ratio: 1.44); synthetic routes 9.82 / 8.52 / 7.41 µs (the
    generator overstates the 4K fill by ≈ 9%, so the 4,096-pass fallback centrals are ≈ 1 point optimistic). The 16K
    calls ran clean. The 5090's probe repeats it on the same real routes.
  - **The change, two parts:**
    - **vendor:** `GGML_CUDA_MM_IDS_MAX_PAIRS` 32,768 → 131,072 (`vendor/ggml/src/ggml-cuda/mmid.cuh:8`); calls ≤ 32,768
      pairs keep their paths (the device-wide sort past one block's 16,384 pairs already runs for today's 4K passes);
    - **plugin** (`kernels.ggml_cuda/plugin.py:79-93, 261-274`; A R5-M1, B5-m3, lead call 2): **whenever the option is
      unset or ≤ `PASS_ROWS` (4,096), today's arithmetic exactly** — `min(_cap, ID_PAIRS // top_k)` with `ID_PAIRS`
      32,768, so FN's explicit 4,096 at top-10 stays **3,072** and the 30B locks stay 2,048 / 512; **only an option
      above 4,096 takes the new rule**, `min(option, MAX_PAIRS // top_k, max_step_tokens, the entry's bound)`, where
      **each `PASS_PROVEN` (SM, type, mode) entry carries its proven row bound** (4,096 for today's entries; 16,384 for
      IQ2_XXS / IQ2_S / IQ3_XXS on sm_86 and sm_120 once step 4's L0 proves them) and **an option above an entry's
      bound is refused with a message**.
  - **GLM's lock alone sets `ggml_cuda.pass_rows` = 16,384** (N4's pick).
  - **Step 4 checks:** an assert (and the launch inventory) that GLM's prompt passes really are 16,384 rows; **FN's
    locks byte-identical with their 3,072-row passes at top-10** and the vehicle's at 2,048 / 512; an L0 of
    whole-`CHUNK` row invariance at 16,384 rows on IQ2_XXS / IQ2_S / IQ3_XXS on sm_86 and sm_120; an option above an
    entry's bound refused; the gate/up class's tail holding the loan, the ≈ 4× pass buffers (≈ +1.5–2 GB lent during
    prompt steps) and the ring halves (≈ 6–8 GB of ≈ 13.6).
- **The prefill tail skip is deferred** (B3-M1, A-m4, lead call 1): as designed it would compute prompt logprobs from
  stale rows (`logits_index` covers sampled rows only; prompt logprobs read `h` by core-internal `PromptRows`,
  `executor.py:36-43, 376, 624-640`) and break std.hyper's "R complete at every block boundary". It is worth ≈ 2.4% of
  PCIe- or SSD-bound prefill; §7.4 counts it as Maya's edge. A deferred lever beside the CPU lane (R3) and MTP (D7); no
  core line for it.

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
  with the file tier; `moe.ram` (experts per class, as above); `--host-memory`; the placement policy (F10 spares with
  keep once its `lend_tail` lands, else F11 with deferred admission and keep, D8), with `moe.ring_lend = tail` and
  `moe.lend_peak` stated explicitly at the lock's values; `moe.stream_tokens` above the step (the staging ring of
  parts at every step size, D9); `moe.read_ahead` (A_ra, D8f); `kernels.cpu_task.io_depth` (N3). The step size is the
  lock's (32K). Nothing else differs from the lock.

**D11 — The comparison protocol: FN's D11 as closed, with the run as the unit.**
- **The Maya backend** (`bench/offload_ctx.py`, built and checked at N0): the engine's stdin protocol with raw ids
  (`GEN <n> temperature=0 <ids>` → `T <id>` … `DONE …`, `generate.cpp:2063, 2213-2222, 7005-7041`); `--glm-pack` from
  `tools/iq_pack.py --compat-bf16` (the pack reads the GGUF from its parent directory); `--max-context 33792
  --kv-resident 33792`; `--eos-ids 2147483647` (N0: `produced` = n on every request); `MAYA_GLM_SLOTS=0`; loop guards
  off; its setup's defaults (`--expert-cache auto --prefill auto --kv-streaming on --expert-profile
  data/expert-profile-glm.bin`); **its own `--calibrate` kept the defaults on the 5090** (N1: PCIe share 0.64 and 8
  lane threads; no sweep beat them by 3%). Built at `4615f34`, sm_120, on the bench host.
- **Maya's checkpoint cut:** a 32K raw-id prompt is read as 16,384 + 16,384 by default; **each 32K cell, prefill and
  decode alike, takes the faster of** Maya's default and `MAYA_GLM_CKPT_EVERY=0` arms for that metric (Q5; N1:
  `CKPT_EVERY=0` is faster at 32K in both, +6.8% prefill, C r5-m1).
- **Holon:** its lock's configuration (mode (ii): the override set), `--ctx 33792`, `ignore_eos`; 0 reused tokens per
  timed request.
- **Generated ids** (C r6-m1): both workers keep each request's generated ids in the run's rows (from N1's third
  round on), so the degenerate screen and the matched-text ratio run offline. **The screen is validated before the
  gated runs** (C r6-m2): on every N1 (prompt, run) whose hit rate is more than 0.08 above that prompt's median across
  Maya's runs.
  - **N1's result** (120 (prompt, run) rows with ids, `local/tmp/glm/n1_stats.py`): rev 7's end rule (≤ 32 tokens
    repeated ≥ 4 times at the end) **misses one of the three outliers** — 32K prompt 1, h +0.11, a 175-token period-2
    loop that broke off before token 256 — and drops 20 of 120 rows, most at their prompt's median h (long list-like
    periods of 20–30 tokens that do not raise h). Widening the period to 64 catches neither problem.
  - **The rule, pre-registered** (the lead's call, 2026-10-25, before any gated run; not tuned once Holon's runs
    exist): a (prompt, run) is degenerate if **some run of ≥ 128 consecutive generated tokens has a period ≤ 16**,
    anywhere in its 256. Its evidence is N1's validation: of the five rows more than 0.08 above their prompt's
    median h, it flags **the three with ids** (+0.14 at period 1, +0.11 at 2, +0.08 at 9) and nothing else; **two are
    unscreenable**, run before ids were kept (1K prompt 0 `CKPT_EVERY=0` round 0, h 0.868, the period-1 row's prompt
    and h; 32K prompt 7 round 1, +0.087) (C r8-m2). The rule was fitted on those three rows and is frozen before the
    gated runs. **Its effect is read on the pair ratios**, the prompt dropped from both engines, never on a per-run
    statistic of mixed prompt sets (C: Maya's 4K run CV rises from 1.5% to 3.0% screened that way).
  - **Maya's greedy text varies run to run** (N1: two runs of one prompt diverge after 1–143 tokens, even within one
    arm), so its decode rate carries a path variance: most of the 1K decode CV.
- **Corpus and spans:** FN's text corpus tokenized with GLM's tokenizer (184,531 tokens); 8 spans per context (FN's
  six starts and two midpoints), the warm-up span excluded; all 24 requests in one fresh process per run.
- **Statistics** (fixed now; C R5-M1, lead call 6):
  - **the run's statistic**, per context and metric: total tokens over total time across its 8 prompts (decode:
    Σ(n − 1) / Σ(t_last − t_first); prefill: Σ C / Σ prompt time), each engine timed from the client;
  - **runs** alternate in ABBA order within an allocation, a discarded warm-up run first;
  - **the cell:** the median over adjacent (Holon, Maya) run pairs of the per-pair ratio;
  - **rounds per context, sized from N1's runs:** Maya's run-to-run CV of the run statistic (N1, 5 runs an arm:
    decode 2.5% at 1K, 1.2–1.7% at 4K and 32K; prefill ≤ 0.2%) and Holon's timing noise give the rounds for a ratio
    SE ≤ 1% — **≈ 6–7 for 1K decode, 3 elsewhere**; the counts are written into the evidence before the gated runs;
  - **a pre-registered degenerate-continuation screen:** a (prompt, run) whose 256 generated tokens hold a run of
    ≥ 128 consecutive tokens with a period ≤ 16 (N1's validation, above) is dropped from that context's cell **for both
    engines in that pair**, and recorded (N1: such a Maya continuation at 1K moved its run by ≈ 4–8%);
  - **the gate is each engine's own greedy text** (the owner's "speed ≥ Maya"), unchanged;
  - **report-only diagnostics:** a **matched-hit-rate ratio for every cell** (each engine's per-prompt time moved to
    the other's per-prompt h by its own fitted slope; Maya's is 0.19–0.20 ms a miss, N1 / N2) and, **for a cell within
    3% of its bar, a matched-text ratio** (Holon decoding forced along Maya's continuations, FN §23.5's
    `draft_quality.py` method). Holon is deterministic, so its 8-prompt cell carries a fixed path-luck bias (≈ 1–1.3%
    sd) that more rounds cannot remove; the diagnostics expose it. **If a diagnostic disagrees with the gated ratio by
    more than the cell's band, the cell goes to the lead with all three;**
  - **pass** ≥ bar; **a cell within 3% of its bar is re-run on a second allocation** (another node), the cell becoming
    the median of both allocations' pair ratios together; below the bar → the owner.
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
- **New:** `rtx-5090` (Maya-S, 32K steps, `max_seqs` 1) and `rtx-5080` (mode (i); the development lock class, where
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
| `ResidencyMap`, the store, `placement.promote`, `kernels.promote` | D8 (a)–(d); `lend_tail` in `placement.promote@1`'s resolve, its reference and `kernels.promote` (D8a) |
| `kernels.ggml_cuda`'s `MoeTopk` / `holon_ggml_topk_moe` and fused router | softmax only, power-of-two E → scoring, bias, scale; E = 288 (D6) |
| `kernels.cpu_task` | a pread pool (`io_threads`) → one completion-driven submitting thread (io_uring, else native AIO, chosen and logged; a constructor argument forces either for tests), option `io_depth`; `io_threads` deleted; an urgent request ahead of every chunk not yet submitted (D8f) |
| `cpu_task@1` | `bind(task)` → `bind(task, urgent=False)`, additive, a hint: a worker serves an urgent slot's request before requests it has not yet started, and may ignore it (D8f); the copy task takes a second destination pool, the RAM pool for a kept piece's copy (B10-m1); signal / wait unchanged |
| `expert_fetch@1` | `gather_roles(..., part, parts)` → also `gather_roles(..., part, size)`: part r the entries [r·size, (r + 1)·size); a call passes exactly one of `parts` and `size` (additive, B9); a per-piece run table for a source pool of superset rows: each extent read at its offset in the piece's aligned superset (D8f; only the SSD fetch launch, from the landing pool) |
| `std.tiered` (store, `pinned`) | I/O staging units: a pass's per class in the RAM pool (exclusive) → **a landing pool per class at the superset width** (general sizing, alignment asserted), the RAM pool unchanged: an on-demand range of A_od = max(32, the decode bound) rows (≥ the graphed steps' SSD misses, checked at bind) and depth read-ahead sets of A_ra; a kept piece copied landing → RAM unit by the submitting thread; `moe.lend_peak = require` (the store raises where the tail cannot lend); **one new option, `moe.read_ahead`** (A_ra, default the decode bound); two `cpu_task` slots a layer (the on-demand one urgent); SSD entries in device slots outside the ring (A_od, + A_ra when reading ahead, lent); round 0 signalled at the resolve and waited after the RAM parts, rounds of exactly A_od in eager prompt steps only, each a `gather_roles(part, size)` slice with its rows copied into the on-demand view; no read-ahead in a step with a decode or verify row; layers 3–4 read at the step's start; the request pool as 1 + depth views; read-ahead signals on main after the reading layer's last pass (D8f); the host mirror from the step's map under exclusive tiers (D8) |
| `moe_tiers@1` | an SSD miss → staging unit `capacity[c] + i` of the RAM pool (its fetch entry i; a kept piece its RAM unit) → **its landing row** (a separate pool, a fact of the call): an SSD miss not read ahead → its fetch entry in rank order j after the RAM misses, landing row `j mod A_od` in round ⌊j / A_od⌋, a kept piece too (its RAM unit's owner moved as today; the copy is the worker's: `io`'s request rows gain the keep unit, B10-m1), or, read ahead, row `A_od + (g % depth)·A_ra + its rank among the first A_ra of the layer's SSD-only set` (no read issued); placement unchanged (D8f) |
| `kernels.ggml_cuda` vendor | `GGML_CUDA_MM_IDS_MAX_PAIRS` 32,768 → 131,072 (D9) |
| `kernels.ggml_cuda` plugin (`plugin.py:79-93, 261-274`) | `pass_rows` from `ID_PAIRS // top_k` → today's arithmetic whenever the option is unset or ≤ 4,096; above 4,096 up to `MAX_PAIRS // top_k` and the `PASS_PROVEN` entry's own row bound, a larger option refused (D9) |
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
the mixers and the dense / shared FFN (D9); GGUF rules per
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
nope 64, v 32), KDA 2 heads × 128, indexer 4 heads × 64, `lm_head` scaled to a logit std ≥ 2; Q8_0 by
`llama-quantize`; 3 prompts of 300
tokens (sparse DSA from t + 1 > 11), chunked at 100 / 250, then 32 decode steps; graphs and eager.

| Check | Where | Threshold | Predicted |
|---|---|---|---|
| logits (the GGUF's head), one-shot and chunked, vs HF | CPU | max \|Δ\| ≤ 1e-4, top-1 identical | pass |
| DSA token sets per (layer, position) | CPU | identical under the lower-pool rule; ties asserted | pass |
| prefix hit vs recompute | CPU | bitwise | pass |
| one row alone vs beside a long row (the dense path is the sparse kernel) | CPU, GPU | bitwise | pass |
| clamp bind rate per path | CPU, GPU | ≥ 5% | pass |
| tokenizer vs `llama-tokenize` (LC `27c7a76`) on a multilingual round-trip corpus | CPU | identical ids | pass |
| the quantized loads `llama-quantize` leaves f32 in the fixtures (A-m1): a synthetic GGUF with Q6_K `attn_k_b` / `attn_v_b` (3-D, widths 256 / 512) and Q8_0 indexer and KDA-gate tensors, loaded through the recipe, against gguf-py's dequantization (rounded to BF16 where D5 rounds), per head | CPU (step 2) | bitwise | pass |
| the Q8_0 fixture through the GPU providers vs the CPU engine on the same file: per-position KL | GPU (A5000; sm_120 at step 4's L0 round) | KL mean and p99 ≤ the bound set from the clean run ×3 (each ≈ 3× the clean maximum), written before any plant runs | pass |

### 6.2 Full model (Maya-S)

| Check | Where | Predicted |
|---|---|---|
| (c) decode vs prefill; chunked (250) vs one-shot: KL mean ≤ 0.13, p50 ≤ 0.007 | 5090 deployment, step 5 and step 10 | pass (FN: 0.025–0.031 / 0.002–0.003) |
| (d) 72 items vs Maya, greedy, thinking on, 16,384 new tokens at most, the template's default effort | both engines' 5090 mode (i) deployments | pass (within 5 items) |
| (f) the cover (acceptance 1f). Bounds FN's way: each set at ≈ 10× the clean run's maximum and ≥ 10× below its plant's effect, both measured before the bound is written | RTX PRO 6000 and 5090 | pass |
| exactness (acceptance 2) at each lock's step: 32K on the 5090 against the RTX PRO 6000 (§7.3); the 5080 at its own lock's step against the RTX PRO 6000 at that step (A R9-m1: its ≈ 5.6 GiB arena cannot lend a 32K loan) | RTX PRO 6000; 5090; 5080 | byte-identical |
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
| `landing_reuse_unfenced` | a read-ahead signal enqueued before the reading layer's gather | `exact.py` RAM-short, a 32K prompt, the staging ring of parts | differs |
| `ssd_slots_in_ring` | rev 8's form: SSD entries staged in ring slots the RAM parts then overwrite | `exact.py` RAM-short, a 250-token chunk with parts on, two rounds | differs |
| `ondemand_round_unjoined` | round r + 1 of an on-demand read signalled before round r's gather on main | `exact.py` RAM-short, a 250-token chunk with parts on, the on-demand range forced small (two rounds) | differs |
| `keep_read_ahead_unfilled` | not a mutation: A R7-M2's case, `max_seqs` 2, a 4K chunk beside a decode row, keep on | `exact.py` RAM-short | **passes** (no read-ahead in a step with a decode row: the case cannot arise) |
| `graphed_round` | a graphed step forced past one on-demand round | the store's error word | fails |
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
| activation reserve | 1.02 | a decode or short step's peak; a 32K step's ≈ 3.55 GiB (N4) is the reserve plus the loan (D9). **The trim is the store's `lend_activations` threshold** (B9-m2): the engine reserves the worst step below the size from which the store lends (`hooks.py:206-210`, `executor.py:214-217`), so lending from `ring_tokens` (512) leaves the decode / verify / sub-512-row peak, ≈ 0.35–0.45 GiB (C r9-m1; N4 measures it), checked by `test_reserve.py`'s cases (FN §22.9) |
| slack | 0.44 | FN |
| **the arena** | **≈ 20.7 (22.3 GB)** | |
| staging and null slots | 0.11 | 3 classes × (8 + 1) |
| **experts cached** | **≈ 3,120 (25.8%)** | **Maya: 3,377 slots / ≈ 3,300 resident (27.3%)**: its log's "expert pool 22.36 GB" is GiB (24.0 GB / 3,377 = 7.11 MB, C-r2-m7). The gap is ≈ 1.6 GiB, **≈ 7.5% fewer experts** for Holon |

- **The gap's value:** `lfu` at Maya's share gives h 0.737 against 0.720 at Holon's (N2), ≈ +4% decode. Levers
  taken (rev 9–10, no core lines): `moe.lend_peak` for prompt steps from `ring_tokens`, the activation reserve then
  the decode / verify / sub-512-row peak (≈ +0.6 GiB, C r9-m1, measured by N4), `--memory-fraction` 0.99 (+0.16, A/B
  against 0.985 on the 32K cells at step 7, C r9-m3): ≈ +95–120 experts, a decode share ≈ 26.5–26.8%. FN §23.6's
  stack trim (151 MiB) costs core lines and is not taken.
- **Mode (ii):** the same VRAM; SSD-only ≈ 1,426 for Holon against ≈ 1,170–1,250 for Maya at rest, and during a prompt ≈
  1,726 / 2,116 at 1K / 4K and ≈ 2,800–3,250 at a 32K step with the staging ring of parts, against Maya's ≈ 2,114 at any
  length (it lends ≈ 914 at every prompt; C's `c-mode2.md`). **Host memory under the 64 GiB limit** (est. until N1's
  mode (ii) run reads `memory.peak`; B7-m1):

  | Item | GiB | Note |
  |---|---|---|
  | RAM tier | ≈ 50.0 | equal to Maya's in experts (D10): ≈ 7,550 at its default 50 |
  | I/O staging (D8f) | ≈ 1.8–2.0 | (2 × A_ra + A_od) = 272–302 units a class at A_ra 120–135 (1.9–2.1 GB) |
  | token table | 0.5 | `rows.place = token_embd:host` |
  | the process | ≈ 3–5 | est. |
  | **total** | **≈ 55–58 of 64** | |

  The staging does not shrink the tier (equal in experts by D10). Were the tier sized to fill the limit, each GB of
  staging would cost ≈ 140 tier experts, ≈ 0.2 s of extra SSD reading a prompt step, against the 0.4–0.9 s a
  too-small A_ra exposes (C).
- **Host, mode (i):** experts 86.03 GB pinned (THP), + the token table 0.52 GB.

### 7.2 The RTX 5080 (16 GB), mode (i)

Visible 15.47 GiB; parameters 6.89, statics 0.2, state 0.98, outside 0.6, activation 0.75, graphs 0.1, slack 0.4 →
arena ≈ 5.6 GiB → **≈ 840 experts, 6.9%**. Its IQ3_XXS down class (8 layers × ≈ 20 slots) cannot hold a ring, so the
5080 stages routed misses (the ring of parts) at every step size, and its loan stays within the gate/up class. A
reported row and the development lock (Q4).

### 7.3 The RTX PRO 6000 (96 GB), step 5

≈ 102.6 GB visible less the same items leaves ≈ 91 GB for experts against 86.03 GB at a 1.02 GiB reserve. **At a
32K step** (A R9-m2) the all-resident `std.moe@1` has no store to lend from: its reserve holds the step's ≈ 3.55 GiB of
activations and ≈ 2.2 GB of 16,384-row pass buffers, ≈ +4.9 GB, leaving ≈ 0.1 GB at `--memory-fraction` 0.98 and
≈ 1.1 GB at 0.99. **If it does not fit at step 5, the reference runs at 16K steps** (≈ +2 GB, it fits), and acceptance
2's all-resident identity is stated at 16K, the 32K identity across the store's fits only (the 5090's cache, a capped
cache, RAM short).

### 7.4 Predicted Holon / Maya ratios per gated cell

**Maya's side is N1's measured mode (i) rows** (`bench/.out/glm0_n1_mode1_rtx-5090`: 5 rounds × 2 arms, each run a
fresh process over the 24 requests, ABBA; D11's run statistic, median over runs; its `--calibrate` kept the defaults,
so these are its calibrated rows):

| Maya, mode (i), RTX 5090 | 1K | 4K | 32K |
|---|---|---|---|
| decode, default (tok/s; ms a token) | 35.9; 27.8 | 34.0; 29.4 | 34.1; 29.3 |
| decode, `CKPT_EVERY=0` | 36.7 | 33.9 | 34.3 |
| prefill, default (tok/s; s, engine ≈ client) | 656; 1.56 | 2,124; 1.93 | 3,233; 10.12 |
| prefill, `CKPT_EVERY=0` | 658 | 2,122 | **3,458; 9.48** (+6.9%) |
| hit rate (median) | 0.73 | 0.71 | 0.71 |
| run-to-run CV, decode / prefill | 2.5% / ≤ 0.2% | 1.5–1.7% / 0.1% | 1.2–1.4% / 0.0% |

Its per-token cost fits **9.96 + 0.197 × misses** ms over N1's prompt-runs (residual sd 0.43 ms, C's `n1_var.py`) and
**10.3 + 0.189 × misses** over N2's 17 `glm stat:` lines (C's `maya_fit.py`): **≈ 28.1 ms a token at h 0.72**. The
4K and 32K decode rows are slower than 1K by their lower h, not by context.

**Holon's side** is the first-order model with C's corrected inputs: dense 7.13 GB at 1.55 TB/s × 0.8 = 5.75 ms;
misses 336 × (1 − h) × 7.11 MB at 47–49 GB/s (Holon's measured in-engine fetch, FN §21.1, §23.8); fixed 2–3 ms
(`k_resolve` 0.32, ≈ 180 mHC calls 0.6–0.9, 11 selections 0.3, KDA state 0.18, glue 0.6). At h 0.72: **22–23 ms**.
**The fixed part is a model, not a measurement** (C-R2-M1): at Maya's fitted 10.3 ms Holon would be 24.5–24.9 ms.

**The hit rate** (N2, held out: each trace's seed from the other trace): Holon's `lfu` at 25.8% gives **0.719 / 0.723**
(bench / tasks); Maya's own policy at 27.3% replays **0.714 / 0.718** against its live **0.722 / 0.725** — the replay
is within 1 point. **N2's replays leave prompt lending out** (C-R2-M6): Maya's live h includes its ≈ 1,020-expert lend
a prompt; `replay.py`'s `lfu` cannot fill an emptied slot, and FN's `lend_min.py` needs a Holon engine that serves
GLM. **Lending's effect on Holon's h is measured at the first Holon-GLM engine (step 5/6); the decode cells carry it
per context** (C r8-m1): **≈ −1 to −2% at 1K, ≈ −4 to −6% at 32K** (Maya lends ≈ 914 at every prompt; Holon little at
1K and ≈ 1,900–2,350 at a 32K step, refilled by the decode's misses: ≈ +2–3.5 misses a token over 256 tokens).

| Cell (bar) | Holon | Maya | Predicted ratio: central [band] | What decides it |
|---|---|---|---|---|
| (i) decode 1K / 4K / 32K (1.0) | 22–25 ms a token | 27.8 / 29.4 / 29.3 ms (N1, h 0.73 / 0.71 / 0.71); 28.1 at h 0.72 (fit) | **≈ 1.15 / 1.13 / 1.10–1.12 [0.95–1.3]** (N1: Maya measured; Holon's side a model; lending per context) | Holon's fixed part (a model); Maya's calibrated lane; lending; R3 decides the lane lever at step 0's end |
| (i) prefill 1K (0.9) | routed misses ≈ 55 GB by the ring of parts' gather at ≈ 45.6 GB/s (FN §21.9) + compute under it: 1.3–1.45 s | 1.56 s (N1: 656 tok/s) | **1.1 [0.95–1.25]** (C-r2-m2) | the gather rate; N4's compute |
| (i) prefill 4K (0.9) | ≈ 68 GB stream 1.3–1.4 s ∥ compute ≈ 1.2 s: 1.45–1.6 s | 1.93 s (N1: 2,124 tok/s) | **1.25 [1.0–1.4]** | the same |
| (i) prefill 32K (0.9), one 32K step | max(stream ≈ 1.4 s, compute ≈ 9.5 s × c), two 16,384-row passes a layer; c ≈ 1.2 at `pass_rows` 4,096 (D9: 0.63 column fill against Maya's 0.88) | 10.12 s (N1: 3,233 tok/s, 2 × 16K); the faster `CKPT_EVERY=0` arm 9.48 s (3,458 tok/s) | **≈ 1.0 [0.85–1.15] at the design's 16,384-row pass** (the A5000's measured 1.42× per-token gain on real routes; against the faster arm, N1's `CKPT_EVERY=0` at +6.8%, ≈ 9.48 s; C r5-m1); **≈ 0.83–0.87 at the 4,096-row fallback** — on the owner list if the 5090's probe does not confirm the 16K gain | the pass size (N4, step 4); the rest of c |
| (ii) decode (1.0) | the mode (i) token + r_H × 2.86 ms (N3, no bounce); r_H from C's replay (`mode2_decode.py`): as designed (F11 deferred + keep) ≈ +54 / +77% of a Maya-like policy's reads (bench / tasks); F10 spares + keep −19 / −23% | 28.1 ms + r_M × 2.89 ms (its 8-thread pool; N3) | **as designed ≈ 0.94–1.04; with the levers ≈ 1.07–1.15 [0.9–1.2]** | r_M (N1 mode (ii): `stat.disk_reads`, 9001fdb8); F10 + `lend_tail`; the VRAM levers |
| (ii) prefill 1K (0.9) | the routed SSD-only set, (1,426 + 300 lent) × 0.75 ≈ 1,295 experts (9.2 GB), 3.6 s beside the RAM stream (rev 9's order) | (1,200 + 914) × 0.75 ≈ 1,590 (11.3 GB), ≈ 4.4–4.8 s | **as designed ≈ 1.15–1.25; with the levers ≈ 1.2 [0.95–1.4]** | both engines' SSD sets (N1 mode (ii)); the order (serialized ≈ 0.9) |
| (ii) prefill 4K (0.9) | ≈ 2,075 as designed (14.8 GB, ≈ 5.8 s); with the ring of parts ≈ 1,545 (11.0 GB, 4.3 s) | 2,114 × 0.98 ≈ 2,070 (14.7 GB), ≈ 5.8 s | **as designed ≈ 1.0; with the levers ≈ 1.3 [1.0–1.45]** | the staging ring of parts in mode (ii) (D9) |
| (ii) prefill 32K (0.9) | per prompt: as designed two 16K steps read 5,652–6,552 experts (40–47 GB), 15.6–18.1 s; **one 32K step with the ring of parts** ≈ 2,936–3,386 with the out-of-ring slots' ≈ 150 lent (C r9-m2), 8.1–9.4 s under compute ≈ 9.5 s: **a ≈ 1% margin at the top end** | its faster arm, one 32K chunk (`CKPT_EVERY=0`): 2,114 (15.0 GB), 5.9 s, under its 9.5 s of compute: ≈ 9.5–10 s | **as designed ≈ 0.53–0.64; with the levers ≈ 0.95–1.0 [0.85–1.05]** (the low end: SSD-bound if the measured loan lands at the top) | one 32K step (D9); the ring of parts; compute c ≈ 1 (the 16,384-row pass); the measured loan |

**Mode (ii) per prompt** (rev 9, C's `c-mode2.md`; the drive at N3's 2.57 GB/s, ≈ 2.77 ms an expert; every Maya mode
(ii) figure modelled from its code and logs until N1's mode (ii) run measures its lend, parking, disk rate and r).
**The levers, all taken in rev 9, no core lines:** R8-M1's order (D8f), one 32K step in both modes and the staging
ring of parts in mode (ii) (D9), F10 spares + keep with `lend_tail` (D8a), no bounce copy (D8f), the VRAM levers
(§7.1). **No cell is centred below its bar with the levers**; (ii) 32K sits nearest (≈ 0.95–1.0 against 0.9). A cell
still centred below its bar after the levers goes on the owner list.

**Cells with a band below their bar go to the lead at step 0's end** (N4's rule), with levers and costs:
- **(i) decode** — R3's exact CPU row (decided at step 0's end); MTP with its head's experts pinned (D7: 0.89–1.09×,
  ≈ 4 days; C m1); the cache levers (+4% at Maya's share); the measured fixed part (step 4's launch inventory).
- **(ii) decode** — the levers above; then the cache levers (each +100 VRAM experts ≈ −25% of Holon's SSD reads).
- **(ii) prefill 1K / 4K / 32K** — the levers above; the cache levers; D8b's demote-into-RAM rule.
- **(i) and (ii) 32K prefill** — on the owner list unless N4's pass probe lifts them: the 16,384-row pass (D9, step
  4); one 32K step; the deferred tail skip (+2.4%, D9).

## 8. Step 0 — measure before freezing

On the RTX 5090 desktop (and the 5080 for the reported row), one heavy job at a time; the A5000s for N4's kernel
probes.

| # | Measure | Status | Decides |
|---|---|---|---|
| N0 | The harness: Maya built at `4615f34` (sm_120) and its pack; `offload_ctx`'s Maya backend; `produced` = n; `reused` = 0; the corpus retokenized | **done** (one run: decode 35.9 / 35.6 / 35.9, prefill 646 / 2,016 / 3,195 tok/s at 1K / 4K / 32K; pool 3,360–3,377; h 0.72–0.74) | D11 runnable |
| N1 | Maya's rows: `--calibrate`, then mode (i) 5 rounds × 2 arms (its run-to-run CV), mode (ii) in a 64 GiB job step (pinned memory in `memory.peak`; the tier in experts; hygiene), the deterministic recipe; the 5080 mode (i); Maya's ledger lines | **calibrate done (kept the defaults); mode (i) done** (5 rounds × 2 arms, §7.4; the screen validated, D11); **mode (ii) queued behind N4's 5090 probes**, recording Maya's SSD reads a token (`stat.disk_reads`) | the denominators; D10's tier; D11's rounds |
| N2 | Maya's route traces: decode (`STRATA_GLM_ROUTE_LOG`) and prompts (`--dump-routing`) on the bench plan and the 72 prompts; the held-out replay at each engine's share; F10 vs deferred admission in mode (ii); MTP verify windows; the routed fraction of the SSD-only set by window | **decode traces done** (bench 8,256 + tasks 27,680 tokens): h above; windows 13.9 / 19.1 / 23.9 distinct experts per layer at 2 / 3 / 4 tokens (D7). **Prompt traces queued.** Lending: not replayable at step 0 (§7.4) | D8's mode (ii) default; D7 (deferred); D9's rules; h |
| N3 | The SSD microbench on the bar node's NVMe: io_uring (with and without the bounce copy, which copies each extent as it completes into a rotating page-locked 2 GiB region; C n2), native AIO, `pread`, an 8-thread pool; depth 8 / 16 / 32; 256 KiB–1 MiB chunks; 1–8 experts a call (`spikes/glm0/n3_ssd.c`) | **done** (`local/tmp/glm/n3_ssd_rtx-5090.txt`): **2.57 GB/s at k ≥ 4** on io_uring, native AIO and an 8-thread pool alike (2.45 at k = 1, 2.86 ms an expert); plain `pread` 1.62; depth 8–32 and chunks 256 KiB–1 MiB within 1%; the bounce copy −7% at k = 1, −1% at k = 8. The drive is the limit (a PCIe 4.0 x4 NVMe) | D8f's interface (io_uring, AIO fallback), depth (16) and chunk (any); its rate decided the mode (ii) levers (§7.4, rev 9); the bounce dropped |
| N4 | The roofline per §7.4 row; **the prefill compute probe**, **4K / 8K / 16K MoE passes** (A5000 done on real routes: 10.50 / 8.24 / 7.41 µs a token a layer) (a spike build of the extension at `GGML_CUDA_MM_IDS_MAX_PAIRS` 131,072) and **the live set at 4K–32K** | **A5000 done**: MoE GEMMs per 4K pass (D9), the live set unsplit ≈ 326 KiB / sub-batched ≈ 55 KiB a token beyond R at 16K (D9); **the 5090's queued** (real routes from N2's tasks trace; 4K pass, and 8K / 16K with the spike build) | every gated row frozen; a row predicted below its bar goes to the lead before step 1 ends; **the non-lending peak** (decode / verify / sub-512-row steps) that the reserve trims to (C r9-m1) |
| N5 | (½ day) FlashInfer 0.6.12's fa2 MLA with `head_dim_kpe = 0` on sm_120, and the non-absorbed prefill form, against a Triton prototype of `mla_attend` | to build | D5's dense path |
| N6 | Maya's answers to the 72 items on the 5090 (its defaults, the same rendered ids), saved as gate (d)'s reference | after N1 | gate (d) ready at step 5 |

## 9. Steps

| Step | Work | Checks | Days |
|---|---|---|---|
| 0 | N0–N6 | evidence; §7.4 frozen; escalations | 4 |
| 1 | D8 in order: (b)+(e), then (a), then (c); D8f (after N3): the one-thread async I/O worker (both backends tested), `io_depth`, staging units sized for the read-ahead depth, the completion-driven worker, SSD entries outside the ring with the round waited after the RAM parts, rounds of exactly A_od, the run table (no bounce), `moe.read_ahead`, `io_threads` deleted; `lend_tail` in `placement.promote@1` — on the Qwen3-30B-A3B vehicle with RAM capped | `exact.py` capped vs uncapped (A5000, then the 5090 node's NVMe); **FN's rows RAM-vs-SSD identity** (B2-m2); the vehicle's offload lock trace byte-identical; the two-round test (a 250-token chunk, parts on) under `exact.py` against the uncapped run; the urgent-round test; F10 + tail lending exact; the D8 plants incl. `ssd_slots_in_ring` and `keep_read_ahead_unfilled`; core diff empty | 8 |
| 2 | The oracle: `golden-glm`; both fixtures (tie wrappers, randomised init, peaked combs, power-of-two MLA dims; the Q8_0 copy); the tokenizer round-trip corpus; the quantized-load test (A-m1) | the HF driver bitwise vs HF's whole forward on the tiny model; the load test | 3 |
| 3 | Std units and contracts with references and CPU providers (D3–D6, D13) in their packages, `std.ffn.swiglu@1(rows)`, `gdn_rewind@1` generalised (−inf); the model plugin; the import-linter contracts | §6.1 CPU rows; **every existing lock's trace byte-identical** (`rows` = 0, the masked value); core diff empty | 6 |
| 4 | Kernels (D12), 32K steps (D9): the router on both paths (`MoeTopk` included) and the clamp; KDA (chunked vendored; recurrent through the `IS_KDA` branch); mHC (split over h); `dsa_index`, `qsa_select` weights, `mla_attend`; the mixers' sub-batching (behind `rows > 0`); `GGML_CUDA_MM_IDS_MAX_PAIRS` 131,072 and the plugin's explicit-only larger `pass_rows`; the launch inventory; `test_reserve.py`'s tiny-GLM case | acceptance 3; §6.1's GPU row (a′); **the vehicle's (2,048 / 512) and FN's (3,072 at top-10) MoE locks byte-identical** after the router and pass-size changes; an option above a `PASS_PROVEN` bound refused; GLM's passes asserted at 16,384 rows; the 16,384-row whole-`CHUNK` invariance L0 (IQ2_XXS / IQ2_S / IQ3_XXS, sm_86, sm_120); the gate/up tail holds the 32K loan + passes + staging; `test_reserve.py`'s 32K case; mode (i) at 16K vs 32K steps (4K, 32K prefill and the decode after a prompt); the slot-0 test | 7 |
| 5 | The full model on the RTX PRO 6000 (all-resident) then the 5090: gates (c), (d), (e), (f); exactness; the clamp bind rate, comb spread, tie counts; `route_dump` profile; Holon's own traces re-check N2, **and lending's effect on h** | acceptance 1(b), 2; §6.3's plants | 4 |
| 6 | Offload on the bar cards: mode (i), and mode (ii) with the pre-registered override set (F10 spares + keep with `lend_tail`); F11 with deferred admission and keep as a reported row, both with prompt lending (D8); the SSD path on the node's NVMe | acceptance 2 on both cards; fit vs §7 | 4 |
| 7 | Perf closure: per-component breakdown vs Maya (nsys), fusions, CUDA rewrites by D12's rule; a per-change perf smoke; `--memory-fraction` 0.99 against 0.985 on the 32K cells (kept only if they hold, C r9-m3) | every gated row within its band, or the gap explained | 6 |
| 8 | Locks (`tune --new-model`): `rtx-5090`, `rtx-5080` | snapshot; fit vs §7 | 2 |
| 9 | The comparison (D11), a second allocation for any cell within 3% | acceptance 4 | 3 |
| 10 | Close: gates (c), (f) and `exact.py` on the final tree at each lock's step, (d) if a decode kernel changed after step 5; the GPU-sharded suite; panel code review; propagation; plan §10 | acceptance 1 (shipped tree), 5–8 | 3 |

**Schedule:** 50 days of work with MTP deferred (step 1 grew by `lend_tail` and the run table, rev 9). **Critical path**
with two to three agents: 0 ∥ 1 ∥ 2 (6; step 1's 8 days run on beside step 3, which does not need it) → 3 (6; step 4's
kernels start beside it) → 4 (5) → 5 (4) → 6 (4) → 7 (6) → 8 (2) → 9 (3) → 10 (3) = **39 days**. Step 1 is independent
of the model and merges into the branch first (D15).

## 10. Risks

- **R1 — Mode (ii) rests on the drive** (N3: 2.57 GB/s, both engines on it) and on Holon's SSD-only set against
  Maya's. As designed the (ii) 32K cell is ≈ 0.53–0.64 per prompt and (ii) decode ≈ 0.94–1.04; rev 9's levers (one
  32K step, the staging ring of parts, F10 + `lend_tail`, no bounce, VRAM) put them at ≈ 0.95–1.0 and ≈ 1.07–1.15.
  Every Maya mode (ii) figure is modelled until N1's mode (ii) run.
- **R2 — 32K prefill** in both modes needs one 32K step (D9) and Holon's prefill compute at Maya's: at 4,096-row
  passes the routed GEMMs fill 0.63 of mmq's column tiles against Maya's 0.88 (c ≈ 1.2). The lever is the 16,384-row
  pass (step 4; 1.42× faster per token on the A5000, real routes); if N4's 5090 probe does not lift the cells, they go
  on the step-0 owner list with the curve. The 32K step's loan must fit the gate/up tail (step 4).
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

**Deferred:** K4 (Q12); MTP (D7); an exact fast CPU row (R3); the prefill tail skip (D9); Maya-S24 / M / L; DFlash2;
`--parallel` batching rows
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
replayable at step 0 (§7.4); 9. ~~the tail skip runs the FFN on the logit rows, never skips the MoE call~~ (superseded
by rev 4: the tail skip is deferred, D9); 10. 16K in
both modes, off the override list (D9, D10); 11. the remaining minors as proposed, the core cap against the merge-base
(acceptance 6).

**The lead's calls on rev 3's review (2026-10-25):** 1. the prefill tail skip deferred, a lever beside the CPU lane and
MTP, no core line (D9); 2. B's landing design: the RAM tier's staging units sized for the depth, in `std.tiered`; the
fence by stream order, no new word; the bounce into whole units kept unless N3 breaks 1.3 ms (then an `expert_fetch@1`
run table); a read-ahead expert is a plain SSD miss for placement; gate/up as two extents (D8f); 3. the 4K / 8K / 16K
pass probe in N4, `GGML_CUDA_MM_IDS_MAX_PAIRS` 131,072 at step 4 with GLM's lock alone on the larger pass, the 32K
centrals at ≈ 0.86–0.90 until N4, the gate/up tail's capacity checked (D9, §7.4); 4. the quantized-load CPU test at
step 2 (§6.1); 5. a forced-backend constructor argument, KDA's sub-batching behind `rows > 0`, D4's bitwise test on the
recurrence, MTP as a lever, v = 32, A's nits (D8f, D4, D2, §7.4).

**The lead's calls on rev 4's review (2026-10-25):** 1. the read-ahead signal after layer g's MoE pass has joined every
reader of the set (split fetch, parts, ring, F5b), side-stream readers waiting on the slot, two new plants (D8f);
2. the pass cap reconciled: the header raised, the plugin's default exactly today's, only an explicit `pass_rows`
above it, GLM's passes asserted at 16K, §4 lists the plugin change (D9); 3. a 16,384-row invariance L0 on GLM's IQ
types (D9, §9); 4. a deterministic read-ahead cap, disjoint on-demand units, a second signal after the read-ahead's
wait, the request pool × depth (D8f); 5. real routes in the 5090 probe, N3's bounce fixed, the 4K fallback ≈ 1 point
lower (§7.4, §8); 6. §11's rev-3 tail-skip line marked superseded.

**The lead's calls on rev 5's review (2026-10-25), to converge by simplifying:** 1. two `cpu_task` slots a layer (on
demand, read-ahead), one request each per step, every wait ordered after its signal; no rounds — a per-layer SSD-only
cap of A and a read-ahead window with no victims make every call fit and remove late experts (D8f); 2. today's pass
arithmetic whenever the option is ≤ 4,096 (FN's 3,072 at top-10 named in step 4), the new rule only above, each
`PASS_PROVEN` entry's own bound (D9); 3. A derived from class sizes (D8f); 4. the `moe_tiers@1` mapping in §4, request
rows in their own views (D8f); 5. layers 3 and 4 on demand (D8f); 6. D11's rounds from N1, the paired degenerate
screen, report-only matched-h and matched-text ratios, the gate unchanged (D11); 7. the 32K centrals ≈ 1.0 / 0.83–0.87,
the faster arm per metric (§7.4, D11).

**The lead's calls on rev 6's review (2026-10-25):** 1. drop the placement cap, bound the I/O: `moe_tiers@1` places as
today; read ahead the first A of a layer's SSD-only set; the rest on demand in rounds of A, eager prompt steps only,
each round after the previous round's readers joined; the read-ahead signal after the layer's last pass; graphed steps
never past one round, asserted; A sized from the measured traces with lending; plants `ondemand_round_unjoined` and
`graphed_round` (D8f); 2. mode (ii)'s default policy F10 spares + keep (D8, D10); 3. the bench workers keep generated
ids (D11, done: `dd8b178f`); 4. the screen validated on N1's high-hit-rate outliers before the gated runs (D11).

**The lead's calls on rev 7's review (2026-10-25), the simplest option every time:** 1. SSD misses gathered on main,
round by round, before the parts pipeline; the parts carry RAM misses only; a test forces two rounds for a 250-token
chunk (D8f, A R7-M1); 2. no read-ahead in a step that carries decode rows; `keep_read_ahead_unfilled` at `max_seqs` 2
must pass (D8f, A R7-M2); 3. mode (ii) stays on F11 with deferred admission and keep (`lend_tail`); F10 spares
reported with its measured saving and the lending gap that blocks it (D8, D10, B7-M1); 4. the read-ahead size a store
option (`moe.read_ahead`, default the decode bound, GLM's ≈ 120 from the largest set after the loan) in the override
set; the on-demand range its own size (≈ 32); the pinned bytes in §7.1; per-round `gather_roles` slices with each
round's rows written before its signal (D8f, §7.1, B7-m1, B7-m2, C R7-M1); 5. the (ii) 32K threshold with the loan
(≈ 4.2–4.9 GB/s), the band down to ≈ 0.65, N3 decisive, the owner list below ≈ 4.5 GB/s with its levers (§7.4, C
R7-M2); 6. read-ahead held while a round is outstanding; layers 3–4 read at step start; step 6's F10 row with lending
(C r7-m1, r7-m2).

**The lead's calls on rev 8's review and N3 (2026-10-25):** 1. SSD entries and read-ahead hits in staging slots
outside the ring, counted in the loan and `test_reserve.py`; round 0 signalled at the resolve, the RAM parts, then the
round waited, gathered and computed as its own part; the two-round test under `exact.py` against the uncapped run
(D8f, A R8-M1, C R8-M1); 2. rounds of exactly A_od by `gather_roles(part, size)` (A R8-m1); 3. a completion-driven I/O
worker with an urgent test, `urgent` a hint (A R8-m2, B8-m2); 4. the host mirror's lent code fixed in code with the
`ssd1` check (B8-m1, done); 5. lending per context, the screen's record of its unscreenable rows, its effect on pair
ratios (C r8-m1, r8-m2); 6. D8f's fence: only the main-stream gather reads SSD units (A R8-c1); 7. after N3 and C's
lever analysis: `max_step_tokens` 32,768 in both modes with passes of 16,384, mode (i) checked not to lose (else a
mode (ii) override); the staging ring of parts at every step size in mode (ii); `lend_tail` in `placement.promote@1`
and F10 spares + keep as mode (ii)'s default; no bounce copy (a run table); the reserve trim and `--memory-fraction`
0.99, not the stack trim; §7.4 per prompt, as designed and with the levers (D9, D8, D8f, §7).

**The lead's calls on rev 9's review (2026-10-25), the final design:** 1. a landing pool per class at the superset
width; every SSD piece, kept ones included, read into it, a kept piece copied to its canonical RAM unit by the
submitting thread; the RAM pool and every RAM-miss gather exactly as today; `moe_tiers@1`'s staging indices name
landing rows (D8f, §4, B9-M1, B9-m1); 2. the reserve trim is the store's `lend_activations` threshold (§7.1, B9-m2);
3. N4 measures the non-lending peak; the (ii) 32K set with the out-of-ring slots and its ≈ 1% margin; the lock
refuses to start without the 32K loan; step 7 A/Bs 0.99 against 0.985 (§7, §8, §9, D9, C r9-m1–m3); 4. the 5080's
exactness at its own lock's step; §7.3 at 32K with its 16K fallback; general landing sizing with an alignment
assert; `lend_tail`'s reference asserts ownership (§6.2, §7.3, D8f, D8a, A R9-m1, R9-m2, nits); 5. `gather_roles`
takes exactly one of `parts` and `size` (§4). After B checks B9-M1's fix the design is closed.

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
| Q10 | Mode (ii) = the 5090 lock + one pre-registered override set (file tier, `moe.ram` in experts, `--host-memory`, placement, `moe.read_ahead`, `io_depth`; D10), not a lock keyed on host RAM? | **Yes** |
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

### Rev 9 → rev 10, the final design (the same panel)

Rev 9 drew A 2 minors + 2 nits (no major), B 1 major + 2 minors, C 3 minors (no major). Dispositions (lead's calls,
§11); after rev 10 only B checks B9-M1's fix, and the design is closed:

| Finding | Disposition |
|---|---|
| B9-M1 widening the class's pool unit changes every RAM-unit fetch | **Accepted** (D8f, §4): a landing pool per class at the superset width; the RAM pool and every RAM-miss gather exactly as today; only the SSD fetch launch uses the run table |
| B9-m1 one decode SSD launch mixes two layouts | **Dissolved** (D8f): kept pieces land in the landing pool too, copied to their RAM unit by the submitting thread |
| B9-m2 name the reserve trim's mechanism | **Accepted** (§7.1): the store's `lend_activations` threshold; `test_reserve.py`'s cases check it |
| B: `gather_roles` takes one of `parts` and `size` | **Accepted** (§4) |
| A R9-m1 the 5080 cannot take a 32K step | **Accepted** (§6.2): its exactness at its own lock's step, against the RTX PRO 6000 at that step |
| A R9-m2 §7.3's all-resident reference at 32K | **Accepted** (§7.3): ≈ 0.1–1.1 GB to spare; if it does not fit, the reference at 16K and the identity stated there |
| A nits: general superset sizing; `lend_tail`'s ownership assert | **Accepted** (D8f, D8a) |
| C r9-m1 the reserve can go to the non-lending peak | **Accepted** (§7.1, §8): ≈ 0.35–0.45 GiB, N4 measures it |
| C r9-m2 the out-of-ring slots in the (ii) 32K set | **Accepted** (§7.4): 2,936–3,386 reads, 8.1–9.4 s, a ≈ 1% top margin |
| C r9-m3 startup guards for the loan and the 0.99 slack | **Accepted** (D9, §9): `moe.lend_peak = require`; step 7 A/Bs 0.99 against 0.985 |

No finding was rejected.

### Rev 8 → rev 9 (the same panel), with N3 and C's mode (ii) lever analysis

Rev 8 drew A 1 major + 2 minors + 1 consistency fix, B 2 minors, C 1 major + 2 minors (A's and C's majors at the same
spot). N3 then measured the drive at 2.57 GB/s, and C's lever analysis (`local/tmp/glm-panel/c-mode2.md`) put the (ii)
32K cell at ≈ 0.53–0.64 per prompt as designed. Dispositions (lead's calls, §11):

| Finding | Disposition |
|---|---|
| A R8-M1 SSD entries gathered first land in ring slots the RAM parts overwrite | **Accepted** (D8f, §6.3): SSD entries and read-ahead hits in slots outside the ring, counted in the loan and `test_reserve.py`; plant `ssd_slots_in_ring`; the two-round test under `exact.py` against the uncapped run |
| C R8-M1 rounds before the parts serialize the SSD read with the RAM stream | **Accepted** (D8f): round 0 signalled at the resolve, the RAM parts, then the round waited and computed as its own part (C's order) |
| A R8-m1 a fixed R splits small sets into tiny serial rounds | **Accepted** (D8f, §4): rounds of exactly A_od via `gather_roles(part, size)` |
| A R8-m2 the urgent hold needs a completion-driven worker | **Accepted** (D8f, §4): completion-driven loop, urgent ahead of unsubmitted chunks, a test |
| A R8-c1 the fence paragraph predates rev 8 | **Accepted** (D8f): only the main-stream gather reads SSD units; the side-stream plants dropped |
| B8-m1 the host mirror records a lent piece as RAM | **Accepted, fixed in code** (`bbe68fd9`; D8): the lent code in the mirror, `ssd1` checks it |
| B8-m2 `urgent` as a hint | **Accepted** (D8f, §4) |
| C r8-m1 lending per context | **Accepted** (§7.4): −1 to −2% at 1K, −4 to −6% at 32K; (i) 32K decode ≈ 1.10–1.12 |
| C r8-m2 the unscreenable rows | **Accepted** (D11): three with ids flagged, two unscreenable, the effect read on pair ratios |
| C's lever analysis: the (ii) 32K cell as designed ≈ 0.53–0.64 per prompt | **Levers taken** (D8, D8f, D9, §7): one 32K step in both modes, the staging ring of parts in mode (ii), F10 + `lend_tail`, no bounce, the VRAM levers; §7.4 restated per prompt, as designed and with the levers |

No finding was rejected.

### Rev 7 → rev 8 (the same panel)

Rev 7 drew A 2 majors (R6 items resolved), B 1 major + 2 minors (B6 items resolved), C 2 majors + 2 minors (r6 items
resolved, F10 + keep adopted). Dispositions, each the simplest option (lead's calls, §11):

| Finding | Disposition |
|---|---|
| A R7-M1 on-demand rounds and the ring of parts deadlock when a part straddles two rounds | **Accepted** (D8f): SSD misses gathered on main, round by round, before the parts pipeline; the parts carry RAM misses only; a test forces two rounds for a 250-token chunk with parts on; `ondemand_round_unjoined`'s target configuration forces them |
| A R7-M2 a kept read-ahead expert's RAM unit is never filled | **Accepted** (D8f): no read-ahead in a step that carries a decode or verify row; `keep_read_ahead_unfilled` at `max_seqs` 2 must pass (§6.3) |
| B7-M1 F10 spares has no `lend_tail`, so mode (ii)'s 16K loan is refused | **Accepted, second option** (D8, D10): mode (ii) stays on F11 with deferred admission and keep; F10 reported with C's 19–23% saving and the `lend_tail` gap that blocks it; `ring_lend` / `lend_peak` stated in the override set |
| B7-m1 how A reaches the store | **Accepted** (D8, D8f, D10, §7.1): `moe.read_ahead`, a store option with the decode bound as its generic default, GLM's lock sets it, the override set lists it; "no new options" corrected; the pinned bytes shown against mode (ii)'s RAM tier |
| B7-m2 per-round slices of the fetch list and the request rows | **Accepted** (D8f, §4): SSD-miss entries in rank order after the RAM misses; round r a `gather_roles(part = r, parts = R)` slice; its count and pairs copied into the on-demand view before its signal |
| C R7-M1 A too small after the loan; size read-ahead and on-demand apart | **Accepted** (D8f): A_ra ≈ the largest per-layer set after the measured loan (≈ 120), A_od = max(32, the decode bound) |
| C R7-M2 the (ii) 32K threshold ignores the loan's SSD bytes | **Accepted** (§7.4, §8, R1): ≥ ≈ 4.2–4.9 GB/s; band [0.65–1.15]; N3 decisive; below ≈ 4.5 GB/s the owner list with a smaller loan or pass and their costs |
| C r7-m1 on-demand rounds share the SSD with read-ahead | **Accepted** (D8f, §4): new read-ahead submissions held while a round is outstanding (`cpu_task@1`'s `urgent` flag) |
| C r7-m2 layers 3–4 on demand | **Accepted** (D8f): read at the step's start, under dense layers 0–2 |
| C: step 6's F10 confirmation should cover lending | **Accepted** (D8, §9): its reported row runs with prompt lending |

No finding was rejected.

### Rev 6 → rev 7 (the same panel)

Rev 6 drew A 1 major + 2 minors (R5-M1 resolved), B 1 major + 1 minor (the same major; the round-5 items resolved),
C 2 minors (the cap's replayed cost ≈ 1%). Dispositions:

| Finding | Disposition |
|---|---|
| A R6-M1 / B6-M1 the per-layer cap A is infeasible under lending and undefined for fill arenas | **Accepted: the cap dropped** (D8f; lead call 1, B's recommendation): placement unchanged; the I/O bounded — the first A of a layer's SSD-only set read ahead, the rest on demand in rounds of A in eager prompt steps only; A a sizing choice from the measured traces with lending, ≥ the graphed steps' SSD misses (checked at bind) |
| A R6-m1 "once per step" false for the on-demand slot | **Accepted** (D8f): once per round of an expert pass; the read-ahead signal after the layer's last pass |
| A R6-m2 mixed steps: a victim made SSD-only after the snapshot | **Covered** (D8f): anything SSD-only not read ahead is read on demand |
| B6-m1 the window rule is vacuous at `max_seqs` 1 | **Accepted: dropped** (D8f); its plant with it |
| C r6-m1 the bench keeps no generated ids | **Accepted, done** (D11; `dd8b178f`): both workers keep them, from N1's third round on |
| C r6-m2 validate the screen on the high-hit-rate outliers | **Accepted** (D11): before the gated runs; widen to period 64 or add the hit-rate trigger if any escapes |
| C's side result: F10 spares + keep reads 19–23% fewer SSD experts | **Adopted** (D8, D10; lead call 2): mode (ii)'s default |

No finding was rejected.

### Rev 5 → rev 6 (the same panel)

Rev 5 drew A 2 majors + 1 minor + 1 nit (R4-M1 resolved), B 3 minors (no major; every round-4 item resolved), C 1 major
+ 1 minor (its round-4 notes resolved). The lead asked to converge by simplifying the SSD path. Dispositions:

| Finding | Disposition |
|---|---|
| A R5-M1 the explicit-option rule changes FN's 3,072-row passes | **Accepted** (D9, §4, §9; lead call 2): today's arithmetic whenever the option is ≤ 4,096; the new rule only above; step 4 names FN's 3,072 at top-10 and the vehicle's 2,048 / 512 |
| A R5-M2 a side-stream wait cannot tell a slot's second request from its first | **Accepted, removed structurally** (D8f; lead call 1): two slots a layer, each signalled at most once per step; every wait ordered after its signal; no second request exists — a per-layer SSD-only cap of A bounds every call, and no victim is taken inside the read-ahead window, so no late experts; plants `read_ahead_window_victim`, `ssd_cap_ignored` replace the late-read and round-reuse cases, which cannot occur |
| A R5-m1 rounds of F need the read-ahead's fence | **Moot** (D8f): no rounds |
| A nit the step's first `depth` MoE layers | **Accepted** (D8f): layers 3 and 4 on demand |
| B5-m1 F and A fixed from GLM's traces in generic std code | **Accepted** (D8f): A from class sizes at `before_materialize` (1.5 × the mean SSD-only set a layer), an enforced invariant; N2 validates it |
| B5-m2 the rounds and read-ahead rows change `moe_tiers@1`; request write order | **Accepted** (§4, D8f): the mapping listed in §4 (no rounds); request rows in their own views, 1 + depth |
| B5-m3 an option above 4,096 for unproven (SM, type, mode) | **Accepted** (D9, §4; lead call 2): each `PASS_PROVEN` entry carries its row bound; a larger option refused |
| C R5-M1 D11's decode statistic: rounds, outliers, Holon's path luck | **Accepted** (D11; lead call 6): rounds per context from N1 (≈ 4–5 for 1K decode, 3 elsewhere); a pre-registered paired degenerate screen; report-only matched-h (every cell) and matched-text (cells within 3%) ratios; the gate unchanged; disagreement beyond the band goes to the lead |
| C r5-m1 N1's measured `CKPT_EVERY=0` gain; the faster arm per metric | **Accepted** (§7.4, D11): the 32K centrals ≈ 1.0 at 16K passes, 0.83–0.87 at the fallback; the faster arm taken per metric |

No finding was rejected.

### Rev 4 → rev 5 (the same panel)

Rev 4 drew A 1 major + 2 minors + 1 nit (every round-3 item resolved; the fence sound on one stream and across
replays), B 1 major + 3 minors (every round-3 item resolved), C 2 probe notes (no majors). Dispositions:

| Finding | Disposition |
|---|---|
| A R4-M1 the fence covers only the signalling stream; parts, the ring and F5b also read or fill the set | **Accepted** (D8f; lead call 1): the signal at the end of layer g's MoE call after every reader's join; side-stream readers wait on the slot; plants `landing_reuse_side_stream`, `landing_fill_unjoined` |
| A R4-m1 the set is sized for the average; late SSD-only experts | **Accepted** (D8f; lead call 4): a deterministic cap (lowest ids first), the rest on demand; a second signal after the read-ahead's wait; tested on the vehicle |
| A R4-m2 the bit check must name the plugin's `ID_PAIRS` | **Accepted** (D9, §4; lead call 2): the default arithmetic unchanged, only an explicit option above it |
| A nit §11's rev-3 tail-skip line | **Accepted** (§11): marked superseded |
| B4-M1 `pass_rows` silently capped at 4,096 by the plugin | **Accepted** (D9, §4, §9; lead call 2): explicit-only larger passes up to the header's limit; GLM's passes asserted at 16K |
| B4-m1 row invariance at 16K passes unproven | **Accepted** (D9, §9; lead call 3): the L0 at 16,384 rows on IQ2_XXS / IQ2_S / IQ3_XXS, sm_86 and sm_120 |
| B4-m2 staging sized by the average; on-demand and read-ahead units collide | **Accepted** (D8f; lead call 4): disjoint ranges, F on demand then depth sets of A |
| B4-m3 per-depth request rows need a bind change | **Accepted** (D8f): `io_pool` × depth, group g on view `g % depth` |
| C n1 the pass probe's synthetic routing | **Accepted and measured** (D9, §7.4, §8): `gemm.py --routes`; on real routes the 16K pass is 1.42× faster than 4K on the A5000 (C predicted 1.44); the 5090 probe uses them; the 4K fallback centrals lowered ≈ 1 point |
| C n2 N3's bounce mode | **Accepted** (§8): per completed extent, into a rotating page-locked region |

No finding was rejected.

### Rev 3 → rev 4 (the same panel)

Rev 3 drew A 4 minors + 3 nits (every round-2 item resolved; no major), B 2 majors + 2 minors (7 of 9 round-2 items
resolved; B3-M1 corrects B's own round-2 advice), C 1 major + 2 minors (every round-2 item resolved). Dispositions:

| Finding | Disposition |
|---|---|
| A-m1 the fixtures never load quantized `attn_k_b` / `attn_v_b`, indexer or KDA gates | **Accepted** (§6.1, §9 step 2; lead call 4): a CPU load test of a synthetic GGUF with those tensors quantized, against gguf-py's dequantization, per head |
| A-m2 the "copied" word must be an epoch; "landed" must not feed placement | **Accepted, the word removed** (D8f; lead call 2): the fence is stream order, so no word exists; a read-ahead expert is a plain SSD miss for placement |
| A-m3 D4 and D9 disagree on the bitwise test | **Accepted** (D4): the recurrence only |
| A-m4 the tail skip breaks "R complete at every block boundary" | **Moot**: the tail skip is deferred (B3-M1) |
| A nits: `kda.py`'s `.fused_recurrent` import; the GPU fixture's KDA shape; `io_threads`' removal | **Accepted** (D4, D2, D8f) |
| B3-M1 the tail skip gives wrong prompt logprobs | **Accepted: deferred** (D9; lead call 1); a lever beside the CPU lane and MTP; no core line |
| B3-M2 `io_landing` implies unlisted std changes and sits in the wrong owner | **Accepted** (D8f, §4; lead call 2): the RAM tier's staging units sized for the depth, in `std.tiered`; the stream-order fence; the bounce into whole units kept (an `expert_fetch@1` run table only if N3 requires it); gate/up as two extents; `moe_tiers@1`'s read-ahead mapping listed in §4 |
| B3-m1 test both I/O backends | **Accepted** (D8f): a forced-backend constructor argument; the copy-task L0 under both |
| B3-m2 KDA's sub-batching runs through GDN's shared Planner | **Accepted** (D4): behind `rows > 0`; GDN's locks byte-identical at the landing step |
| C R3-M1 the 4K pass's column fill puts the 32K cells at ≈ 0.86–0.91 | **Accepted** (D9, §7.4, §8, §9, R2; lead call 3): N4 times 4K / 8K / 16K passes on the 5090; step 4 raises `GGML_CUDA_MM_IDS_MAX_PAIRS`; GLM's lock alone sets the larger `pass_rows`; the 32K centrals at ≈ 0.86–0.90 until N4, on the owner list if it does not lift them; the gate/up tail's capacity checked |
| C m1 MTP as a lever for (i) decode | **Accepted** (§7.4); the deferral rule stands |
| C m2 nope = v in the GPU fixture | **Accepted** (D2, §6.1): v = 32 |

No finding was rejected.

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
