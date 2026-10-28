"""llama.cpp's quantized CUDA kernels (vendored at a pinned commit, M0.1 design §4.4, M0.2 design §4.3) as
`qlinear@2`, `qembedding@1` and `linear_experts@2` kernels for ggml block layouts (`"ggml.<type>/gguf.native"`) and
the HF FP4 layouts (`"nvfp4/hf"`, `"mxfp4/hf"`, repacked by `prepare`), and its MoE routing and combine as
`moe_topk@1`, `moe_router@1` and `moe_combine@1` (op contracts: std.ops.kernels, std.moe.kernels).

`qlinear` dispatches from the row count: `mmvq` up to `mmvq_rows`, else `mmq` in calls of whole 512-row chunks
(llama.cpp's ubatch) up to `call_rows`, the last partial chunk a call of its own (docs/dev/v1-moe-prefill.md D5:
bounded,
as mmq's stream-k split points move with a call's rows); both write `out_dtype` (bf16 or f32) directly, mmq's
stream-k partials kept in f32 until its one rounding (docs/dev/m07-perf.md D4).
`mmvq_rows` is llama.cpp's threshold for the type at the class's compute capability, or the class's entry for the
shape where a repeated sweep found mmq faster (docs/dev/m07-perf.md D2, `CROSSOVER`); the judge's floors ask it too.
Both quantize the activations in a workspace shared by every instance with the same `in`. `qembedding` is
llama.cpp's `get_rows`.

Shared quantization (D6): `prepare_input(x)` quantizes `x` once, on the mmvq path, as mmvq would (block_q8_1, the
same for every type), into the buffer every instance with the same `in` shares (`ggml_cuda.q8_1[in=…]`, its own:
not the workspace above); the handle (`Prepared`) names it with the host-side generation of that write, and an
mmvq call of the same `in` and rows reads it while the buffer's last write is still that generation (`Extension`
counts them), else quantizes `x` itself. `qlinear_glu` (D6): mmvq's fused gate at one row on a merged gate/up
(`silu(gate) · up` from the f32 sums, NVFP4's scale2 on both halves), None at more rows (std's SwiGLU composes).

`linear_experts` is llama.cpp's MUL_MAT_ID per call of up to `pass_rows` rows (docs/dev/v1-moe-prefill.md D0): mmvq with
ids
up to the type's limit (`get_mmvq_mmid_max_batch`), for `glu="swiglu"` through its fused gate on the merged stack, else
mmq with ids into f32, followed for swiglu by our f32 SwiGLU — or, where mmq can (`mmq_glu`: the stack's rows in whole
tiles of 128, MMA layouts), with SwiGLU in mmq's store, bitwise the same (docs/dev/v1-moe-prefill.md D4: no f32 gate/up
output). `pass_rows` is 512 (llama.cpp's ubatch), or on an (SM, type, mode) whose L0 proves a row's bits the same in any
call of whole 512-row chunks (`PASS_PROVEN`) up to `PASS_ROWS`, capped by the id helper's pairs, the step budget and the
`ggml_cuda.pass_rows` option; a call's last partial chunk runs on its own. Its workspace (`ggml_cuda.id[…]`, sized for
`pass_rows`) holds mmq's id maps besides the activations; gate/up's f32 output, where SwiGLU is not in mmq's store,
comes from the caller's `scratch` (`scratch_bytes`, docs/dev/v1-moe-prefill.md D10) or is allocated, and a call writes
into
the caller's `out` if given. `moe_topk` is `topk-moe.cu` (softmax, top k, renormalised), `moe_router` our router kernel
beside it (the f32 router GEMV from bf16 activations, then `moe_topk`'s selection, in one launch), and `moe_combine`
`moe-weighted-reduction.cu`, both patched to the contracts' layouts (ids at row stride k; the combine stores bf16 into
the caller's `out`, and its `add` the residual stream plus the rounded sum, rounded again: docs/dev/v1-offload.md §14,
step
3c).

Activation modes (docs/dev/m06-lowp.md D3, D8, §4.5, §4.7): one spec per mode, `config["act"]` naming each path's
activation format. `q8_1` (every class, every type, `act_quant` a4 or a8): q8_1 activations on both paths, so
W4A8 for the FP4 types. `fp4` (sm_120, FP4 types, `act_quant` a4): mmq at llama.cpp's native W4A4 (its own
dynamic FP4 quantizer, which reads f32 only: bf16 activations are first copied, exactly, into the named workspace
`ggml_cuda.f32[…]`), mmvq at q8_1. Every mmq call passes its mode's precision in dst's `op_params[3]` (zero is
llama.cpp's default, native W4A4 on Blackwell), and each instance asks llama.cpp, before any launch or capture,
which precision it will use (`mmq_prec`): a mismatch is an error.

FP4 (D8): `prepare` repacks the HF layouts losslessly into ggml blocks (`repack`, llama.cpp's converter layouts);
NVFP4's per-tensor or per-expert `scale2` ([1]; [E]; [2, E] for a merged gate/up stack) is applied in f32 before SwiGLU
on every path: mmvq in llama.cpp's fused epilogue (one row dense; any tokens with ids), dense mmvq above one row in the
unfused kernel's epilogue (docs/dev/m07-perf.md D11), dense mmq in its store (D4) — each before the one rounding to
`out_dtype` — and mmq with ids by our SwiGLU (gate/up) or `y *= scale2[ids]` (down). With `in` a multiple of 256, or of
a `ROW_BLOCKS` type's block (its loaders keep the last K step inside the row: T6), no q8_1 kernel reads past the weight
blocks, so they need no padding (L0 poisons the bytes after them); native W4A4 steps K by 512 with no tail, so the `fp4`
spec's `prepare` appends llama.cpp's allocation padding, the blocks of one row's values up to a multiple of 512 zeroed
(`tail_bytes`), after each FP4 tensor or stack, and its calls check it is there.

Calls are counted per path and mode (`calls["mmq/fp4"]`, …; counted when called, not on graph replay).

The native extension is built ahead of time (`python -m holon_plugins.kernels.ggml_cuda.prebuild`); without it an
instance cannot be created and the error names that command.

`repack` and `tail_bytes` are adapted from llama.cpp cbe28bb3 (conversion/base.py `_nvfp4_pack`,
`repack_mxfp4_blocks`; ggml-cuda.cu's row padding), MIT License, Copyright (c) 2023-2026 The ggml authors.
"""
from __future__ import annotations

import collections
import dataclasses
import functools
import itertools

import torch
from holon.api import Constraint, KernelSpec, Layout, QTensor, Scheme

from . import extension

QLINEAR, QLINEAR_GLU, QEMBEDDING = "qlinear", "qlinear_glu", "qembedding"
LINEAR_EXPERTS, MOE_TOPK, MOE_ROUTER, MOE_COMBINE = "linear_experts", "moe_topk", "moe_router", "moe_combine"
LINEAR_EXPERTS_ARENA = "linear_experts_arena"     # the same kernel over an offload store's arena (ids are its slots)
CHUNK = 512                     # mmq rows per call: llama.cpp's ubatch; sizes the workspace and the f32 buffer
ID_PAIRS = 32768                # the most (token, expert) pairs mmq's id helper sorts per call (mmid.cuh)
PASS_ROWS = 4096                # the largest expert pass of a prefill chunk (docs/dev/v1-offload.md D8)
PASS_OPTION = "ggml_cuda.pass_rows"     # a lock's cap on `pass_rows` (manifest [options]; KV capacity per class)
# An expert pass of a prefill chunk (docs/dev/v1-moe-prefill.md D0, D1; v1-offload.md D8): mmq with ids takes the pass's
# rows in one call where the id helper sorts them (4,096 at top-8: its device-wide path past one block's 16,384
# pairs), in whole CHUNKs, so a row's chunk and path stay the reference's (a call's last partial chunk runs as a call
# of its own: `LinearExperts.__call__` cuts it, whatever its caller passes). That holds on the (SM, ggml type, mmq
# activation mode) pairs whose L0 shows a row's bits the same in a call of any whole number of CHUNKs as in its own
# chunk's call (holon_testing.pass_rows, every entry run on its SM by plugins/tests/test_ggml_cuda_moe.py): the lock
# classes' expert types per SM (from plans/: the 3090's, the 4090's, the 5080's and 5090's) and FN's MoE types
# (docs/dev/v1-flash-next.md D1′). Every other pair keeps CHUNK.
_FN_TYPES = ("iq1_m", "iq2_s", "iq2_xxs", "iq3_s", "iq3_xxs", "iq4_nl", "iq4_xs", "q2_0")
PASS_PROVEN = frozenset({(86, t, "q8_1") for t in ("q4_k", "q6_k", *_FN_TYPES)}
                        | {(89, t, "q8_1") for t in ("q4_k", "q6_k", "mxfp4", "nvfp4", *_FN_TYPES)}
                        | {(120, t, "q8_1") for t in ("q4_k", "q5_k", "q6_k", *_FN_TYPES)})
# The bits mmq's id helper ors into `LinearExperts.errors` for a violated precondition (GGML_CUDA_MM_IDS_ERR_*)
ID_ERRORS = {"slots": 1,        # more distinct slots in a call's ids than `LinearExperts.slots`
             "range": 2,        # an id outside the stack's slots
             "columns": 4}      # a slot with more pairs than mmq's column tiles (an id repeated in a row)
# Supported ggml types (design §4.6) and their ggml type ids. IQ1_M's mmq is vendor.patch's tile loader
# (docs/dev/v1-moe-prefill.md D9).
TYPES = {"q4_0": 2, "q4_1": 3, "q5_0": 6, "q5_1": 7, "q8_0": 8, "q2_k": 10, "q3_k": 11, "q4_k": 12, "q5_k": 13,
         "q6_k": 14, "iq2_xxs": 16, "iq2_xs": 17, "iq3_xxs": 18, "iq1_s": 19, "iq4_nl": 20, "iq3_s": 21, "iq2_s": 22,
         "iq4_xs": 23, "iq1_m": 29, "mxfp4": 39, "nvfp4": 40, "q2_0": 42}
FP4 = ("mxfp4", "nvfp4")
# ggml type -> (values per block, block bytes)
BLOCKS = {"mxfp4": (32, 17), "nvfp4": (64, 36)}
# Holon: the row widths qembedding serves, as multiples (QK_K = 256 unless listed): the legacy types' get_rows reads
# whole 32-value blocks; IQ4_NL's reads 256-value groups, so a row of whole blocks narrower than that (FN: PLE's
# 160-wide rows) is gathered as bytes and converted (`QEmbedding`), bitwise the same values.
QK = {"q4_0": 32, "q4_1": 32, "q5_0": 32, "q5_1": 32, "q8_0": 32, "iq4_nl": 32}
# The `in` widths qlinear and linear_experts serve, as multiples (256 unless listed; T6, docs/dev/v1-moe-prefill.md
# §10):
# these types' mmq tile loaders keep a row's last K step (MMQ_ITER_K = 256 values) inside the row, so any whole number
# of their blocks serves (FN: the down projections at 640, Q2_0 and IQ4_NL); mmvq reads whole blocks only. The
# 256-value types and FP4 keep 256 (their blocks, or native W4A4's K step).
ROW_BLOCKS = {"q4_0": 32, "q4_1": 32, "q5_0": 32, "q5_1": 32, "q8_0": 32, "iq4_nl": 32, "q2_0": 64}
DTYPES = ("bfloat16", "float32")
ACT_QUANT = ("a4", "a8")                    # q8_1 activations (8 bits) serve either
# Each spec's activation format per path, and the ggml_prec each format asks mmq for (GGML_PREC_Q8, GGML_PREC_Q4).
Q8_1 = {"mmvq": "q8_1", "mmq": "q8_1"}
NATIVE_FP4 = {"mmvq": "q8_1", "mmq": "fp4"}
PREC = {"q8_1": 30, "fp4": 40}
ROW_PADDING = 512               # llama.cpp's MATRIX_ROW_PADDING: native W4A4's K step
MMVQ_MAX = 8                    # llama.cpp's MMVQ_MAX_BATCH_SIZE: mmvq's most activation rows
POOL_ALIGN = 256                # csrc/api.cu's workspace pool: its base, and each of its allocations, at this boundary
# The types whose merged gate/up runs mmvq's fused gate (qlinear_glu, docs/dev/m07-perf.md D6, §13.4, §17): where
# bench/glu_sweep.py found it faster per call than the matmul and FlashInfer's silu_and_mul at the locks' gate/up
# shapes on the RTX 5080 and 4090 by more than a sweep's noise, and within 0.4% on the 5090
# (docs/dev/evidence/m07-step2_glu-sweep_<class>), and the step-level A/B kept it (the 8B rows). Q5_K (+0.6 / +1.5 /
# −0.3%, within noise, no L2 of its own) composes, as do most i-quants and Q3_K, which ran it 1–10% slower.
GLU_TYPES = frozenset({"q4_k", "nvfp4"})
# The crossover per shape (docs/dev/m07-perf.md D2): (GPU class, ggml type, in, out, mmq's activation mode) -> mmvq up
# to
# this many rows, mmq above, where repeated graph-timed sweeps (bench/qmm_sweep.py) found mmq faster per call by at
# least 3% and by more than 3 sd at every graphed row the entry moves (4, 7, 8), and the entries save at least 1% of
# some lock's decode or verify step (ten sweeps of the 27B's shapes). mmq writes bf16 itself (D4), so the table is
# the sweeps' no-cast column (docs/dev/evidence/m07-step1_d2-no-cast_<class>_2026-10-14.json). Pinned by this
# plugin's `src`; every other shape keeps llama.cpp's rule.
CROSSOVER: dict[tuple[str, str, int, int, str], int] = {
    ("rtx-5080", "iq1_s", 5120, 17408, "q8_1"): 7,
    ("rtx-5080", "iq2_s", 17408, 5120, "q8_1"): 7,
    ("rtx-5080", "iq2_s", 5120, 17408, "q8_1"): 6,
    ("rtx-5080", "iq2_s", 5120, 34816, "q8_1"): 6,
    ("rtx-5080", "iq2_s", 5120, 6144, "q8_1"): 7,
    ("rtx-5080", "iq2_xs", 5120, 17408, "q8_1"): 6,
    ("rtx-5080", "iq2_xs", 5120, 34816, "q8_1"): 7,
    ("rtx-5080", "iq2_xs", 5120, 6144, "q8_1"): 7,
    ("rtx-5080", "iq2_xxs", 17408, 5120, "q8_1"): 7,
    ("rtx-5080", "iq2_xxs", 5120, 12288, "q8_1"): 6,
    ("rtx-5080", "iq2_xxs", 5120, 17408, "q8_1"): 6,
    ("rtx-5080", "iq2_xxs", 5120, 34816, "q8_1"): 6,
    ("rtx-5080", "iq3_s", 17408, 5120, "q8_1"): 7,
    ("rtx-5080", "iq3_s", 5120, 10240, "q8_1"): 7,
    ("rtx-5080", "iq3_s", 5120, 12288, "q8_1"): 7,
    ("rtx-5080", "iq3_s", 5120, 17408, "q8_1"): 7,
    ("rtx-5080", "iq3_s", 5120, 34816, "q8_1"): 7,
    ("rtx-5080", "iq3_s", 5120, 6144, "q8_1"): 7,
    ("rtx-5080", "iq4_xs", 17408, 5120, "q8_1"): 7,
    ("rtx-5080", "iq4_xs", 5120, 10240, "q8_1"): 6,
    ("rtx-5080", "iq4_xs", 5120, 12288, "q8_1"): 6,
    ("rtx-5080", "iq4_xs", 5120, 17408, "q8_1"): 7,
    ("rtx-5080", "iq4_xs", 5120, 34816, "q8_1"): 7,
    ("rtx-5080", "iq4_xs", 5120, 6144, "q8_1"): 7,
    ("rtx-5080", "iq4_xs", 6144, 5120, "q8_1"): 7,
    ("rtx-5090", "iq3_s", 17408, 5120, "q8_1"): 6,
    ("rtx-5090", "iq3_s", 5120, 17408, "q8_1"): 6,
    ("rtx-5090", "iq4_nl", 17408, 5120, "q8_1"): 6,
    ("rtx-5090", "iq4_nl", 5120, 17408, "q8_1"): 7,
    ("rtx-5090", "iq4_nl", 5120, 34816, "q8_1"): 7,
    ("rtx-5090", "iq4_xs", 17408, 5120, "q8_1"): 6,
    ("rtx-5090", "iq4_xs", 5120, 10240, "q8_1"): 7,
    ("rtx-5090", "iq4_xs", 5120, 12288, "q8_1"): 7,
    ("rtx-5090", "iq4_xs", 5120, 17408, "q8_1"): 6,
    ("rtx-5090", "iq4_xs", 5120, 34816, "q8_1"): 7,
    ("rtx-5090", "q6_k", 5120, 248320, "q8_1"): 6,
}

# D2's batched-decode rule per SM (docs/dev/v1-moe-prefill.md B-dec): mmq with ids picks its tile width against a call's
# tokens up to this many, above against the launch's mean rows per entry. 16 where D2's per-call sweep and the paired
# batched-decode rounds measured it (sm_86 RTX A5000, sm_120 RTX 5080 / 5090); step 2's 64 elsewhere — on sm_89
# provisional until the 4090's sweep runs.
IDS_J_TOKENS: dict[int, int] = {86: 16, 120: 16}
IDS_J_TOKENS_DEFAULT = 64


def ggml_type(layout: str) -> int | None:
    """The ggml type id of a `"ggml.<type>/gguf.native"` layout, None for any other layout."""
    scheme, _, name = layout.partition("/")
    return TYPES.get(scheme.removeprefix("ggml.")) if scheme.startswith("ggml.") and name == "gguf.native" else None


def weight_type(layout: str) -> str | None:
    """The ggml type a matmul reads for `layout`: the GGUF one's, or what `prepare` repacks an HF FP4 layout into."""
    elem, _, name = layout.partition("/")
    if name == "hf":
        return elem if elem in FP4 else None
    return elem.removeprefix("ggml.") if ggml_type(layout) is not None else None


def repack(elem: str, parts: dict[str, torch.Tensor], tail: int = 0, rows: int = 4096) -> torch.Tensor:
    """An HF FP4 weight's codes and block scales (docs/dev/m06-lowp.md D2: `weight` [..., out, in/2] u8, element 2i in
    the low nibble; `scale` [..., out, in/g] u8) as ggml blocks [..., out, row_bytes], bit for bit what llama.cpp's
    converter writes (cbe28bb3 conversion/base.py): NVFP4 per 64 values the four e4m3 scales of its sub-blocks (sign
    bit cleared, as `_nvfp4_pack`), then per sub-block 8 bytes, element j low and j + 8 high; MXFP4 per 32 values the
    e8m0 scale, then 16 bytes, element j low and j + 16 high (`repack_mxfp4_blocks`). Same device; `rows` at a time,
    so the transient beyond the output stays small for any stack; `tail` zeroed bytes after it in the same buffer
    (`with_tail`'s)."""
    group, block = BLOCKS[elem]
    lead, half = parts["weight"].shape[:-1], parts["weight"].shape[-1]
    w = parts["weight"].reshape(-1, half)
    s = parts["scale"].reshape(len(w), -1)
    out = with_tail(torch.empty(len(w), half * 2 // group * block, dtype=torch.uint8, device=w.device), tail, False)
    for a in range(0, len(w), rows):
        out[a:a + rows] = _blocks(elem, w[a:a + rows], s[a:a + rows])
    return out.view(*lead, -1)


def _blocks(elem: str, w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    n = len(w)
    sub = 16 if elem == "nvfp4" else 32                                   # values per scale
    codes = torch.stack((w & 0x0F, w >> 4), dim=-1).reshape(n, -1, sub)   # [rows, scales, sub]
    qs = codes[..., :sub // 2] | (codes[..., sub // 2:] << 4)
    if elem == "mxfp4":
        return torch.cat((s[..., None], qs), dim=-1).reshape(n, -1)
    return torch.cat(((s & 0x7F).reshape(n, -1, 4), qs.reshape(n, -1, 32)), dim=-1).reshape(n, -1)


def tail_bytes(name: str, k: int) -> int:
    """llama.cpp's padding of a quantized tensor (ggml-cuda.cu, `ggml_backend_cuda_buffer_type_get_alloc_size`): the
    blocks of one row's values up to a multiple of 512, zeroed, which native W4A4's last K step reads."""
    group, block = BLOCKS[name]
    return (-k % ROW_PADDING) // group * block


def with_tail(blocks: torch.Tensor, nbytes: int, copy: bool = True) -> torch.Tensor:
    """A view of `blocks`' shape over a buffer `nbytes` longer whose end is zeroed, holding a copy of `blocks` (or,
    not `copy`, uninitialized); `blocks` itself when `nbytes` is 0."""
    if not nbytes:
        return blocks
    buf = torch.empty(blocks.numel() + nbytes, dtype=torch.uint8, device=blocks.device)
    buf[blocks.numel():] = 0
    out = buf[:blocks.numel()].view(blocks.shape)
    return out.copy_(blocks) if copy else out


def mmvq_rows(fingerprint, name: str, mode: str, k: int, out: int, lib=None) -> int:
    """The most activation rows a `qlinear` of ggml type `name`, `in` = `k`, `out` rows, whose mmq path quantizes
    activations in `mode` (`"q8_1"`, `"fp4"`), runs on mmvq on `fingerprint`'s class: its `CROSSOVER` entry, else
    llama.cpp's threshold at the class's compute capability (an entry may only lower it, so mmvq runs every row count
    up to it). `lib`: a loaded extension module (llama.cpp's rule is host code: any SM's module answers for any
    class, no device needed); default the class's SM's module, else any prebuilt one."""
    lib = lib or rule_module(fingerprint.sm)
    cc = 10 * fingerprint.sm                         # llama.cpp's compute capability: 100 * major + 10 * minor
    llama = max((n for n in range(1, MMVQ_MAX + 1) if lib.use_mmvq(cc, TYPES[name], n)), default=0)
    entry = CROSSOVER.get((fingerprint.gpu, name, k, out, mode))
    if entry is None:
        return llama
    if not 1 <= entry <= llama:
        raise RuntimeError(f"kernels.ggml_cuda: the {fingerprint.gpu} entry for {name} {k}x{out} ({mode}) is "
                           f"{entry} rows, not within 1..{llama} (llama.cpp's)")
    return entry


def _cap(ctx) -> int:
    """The most rows one mmq call may take in this build: PASS_ROWS within the build's `max_step_tokens` and the
    `ggml_cuda.pass_rows` option (each where set)."""
    caps = [PASS_ROWS, ctx.options.get("max_step_tokens"), ctx.options.get("plugins", {}).get(PASS_OPTION)]
    return min(int(c) for c in caps if c is not None)


def pass_rows(ctx, name: str, mode: str, top_k: int) -> int:
    """The most rows a `linear_experts` call of ggml type `name` in mmq activation mode `mode` takes at its CHUNK-row
    calls' bits (D0): CHUNK unless (SM, type, mode) is in PASS_PROVEN, else the most whole CHUNKs within `_cap` and
    the id helper's pairs."""
    if (ctx.fingerprint.sm, name, mode) not in PASS_PROVEN:
        return CHUNK
    return max(CHUNK, min(_cap(ctx), ID_PAIRS // top_k) // CHUNK * CHUNK)


def calls(rows: int, size: int) -> list[tuple[int, int]]:
    """A call of `rows` rows cut into calls of whole CHUNKs, up to `size` (a multiple of CHUNK) rows each, then the
    last partial chunk as a call of its own: each row's kernel path is its chunk's, whatever the caller's cut."""
    full = rows // CHUNK * CHUNK
    return [(a, min(a + size, full)) for a in range(0, full, size)] + [(full, rows)] * (full < rows)


def rule_module(sm: int):
    """A module that answers `mmvq_rows` for SM `sm`: its own if prebuilt, else any built here (NotBuilt if none is)."""
    for arch in (sm, *extension.ARCHS):
        try:
            return extension.load(arch)
        except extension.NotBuilt:
            continue
    raise extension.NotBuilt(f"kernels.ggml_cuda: no prebuilt extension; build one with: {extension.PREBUILD} --sm N")


def _device(ctx) -> int:
    """The index of the build's CUDA device."""
    index = torch.device(ctx.device).index
    return torch.cuda.current_device() if index is None else index


class Extension:
    """The prebuilt module, loaded once per SM on first use (never compiled here); and the generation of each prepared
    input buffer's last write (`QLinear.prepare_input`), by its address, from one count."""

    def __init__(self) -> None:
        self.modules: dict[int, object] = {}
        self.generations, self.written = itertools.count(1), {}

    def __call__(self, sm: int):
        if sm not in self.modules:
            self.modules[sm] = extension.load(sm)
        return self.modules[sm]


@dataclasses.dataclass(frozen=True, eq=False)
class Prepared:
    """`QLinear.prepare_input`'s handle (std.ops.kernels, `qlinear@2`): `x`, and, when its rows take mmvq, the buffer
    holding its q8_1 activations as written at `generation` for `key` (path, activation mode, rows, `in`)."""

    x: torch.Tensor
    key: tuple | None = None
    buffer: torch.Tensor | None = None
    generation: int = 0


class _Matmul:
    """What `qlinear` and `linear_experts` instances share: the type they read, their spec's activation mode per
    path, the precision check, the f32 activation copy of native FP4 and `prepare`."""

    def __init__(self, ext: Extension, act: dict[str, str], ctx, facts: dict) -> None:
        self.name, self.k, self.out, self.act = weight_type(facts["layout"]), facts["in"], facts["out"], act
        self.type, self.hf, self.prec = TYPES[self.name], facts["layout"].endswith("/hf"), PREC[act["mmq"]]
        self.lib, self.device = ext(ctx.fingerprint.sm), _device(ctx)
        self.tail = tail_bytes(self.name, self.k) if act["mmq"] == "fp4" else 0
        self.calls: collections.Counter[str] = collections.Counter()

    def _uses_mmq(self, after: int, layout: str) -> None:
        """llama.cpp runs mmq above `after` rows here; and asks it, before any launch or capture, for the activation
        precision it will use: a spec whose mode it would not run is an error, not a silent other mode."""
        if not self.lib.use_mmq(self.device, self.type, after + 1):
            raise RuntimeError(f"kernels.ggml_cuda: llama.cpp uses neither mmvq nor mmq for {layout} here")
        got = self.lib.mmq_prec(self.device, self.type, self.prec)
        if got != PREC[self.act["mmq"]]:
            raise RuntimeError(f"kernels.ggml_cuda: mmq would quantize {layout} activations at ggml_prec {got}, its "
                               f"spec's mode {self.act['mmq']!r} is {PREC[self.act['mmq']]}")

    def _count(self, path: str) -> None:
        self.calls[f"{path}/{self.act[path]}"] += 1

    def _mmq_input(self, x: torch.Tensor, f32: torch.Tensor | None) -> torch.Tensor:
        """mmq's activations: as given, or copied (exactly) into the f32 workspace for native FP4's quantizer."""
        if f32 is None or x.dtype == torch.float32:
            return x
        out = f32.view(torch.float32)[:x.numel()].view(x.shape)
        out.copy_(x)
        return out

    def _f32_workspace(self, ctx, rows: int, act_dtype: str):
        """Native FP4's f32 copy of `rows` activation rows (named by its size: shared, fixed for graph capture)."""
        if self.act["mmq"] != "fp4" or act_dtype == "float32":
            return None
        return ctx.workspace(f"ggml_cuda.f32[in={self.k},rows={rows}]", rows * self.k * 4)

    def _mmq_blocks(self, parts: dict[str, torch.Tensor]) -> torch.Tensor:
        blocks = parts["blocks"]
        if self.tail and blocks.untyped_storage().nbytes() < (blocks.storage_offset() + blocks.numel() + self.tail):
            raise RuntimeError("kernels.ggml_cuda: native W4A4 reads a zeroed tail after the blocks (prepare's)")
        return blocks

    def prepare(self, q: QTensor) -> QTensor:
        """The weight as this instance reads it (docs/dev/m06-lowp.md D4, D8): an HF FP4 layout repacked into ggml
        blocks
        (`{"blocks", "scale2"}` for NVFP4, its scale2 [1] dense, [E] or [2, E] per expert), under the `fp4` spec
        followed by the zeroed tail; a GGUF layout as adopted (`q` itself), under `fp4` copied with the tail."""
        if not (self.hf or self.tail):
            return q
        blocks = repack(self.name, q.parts, self.tail) if self.hf else with_tail(q.parts["blocks"], self.tail)
        parts = {"blocks": blocks}
        if "scale2" in q.parts:                         # one value per tensor, or per expert ([E] or [E, 1]; [2, E])
            s = q.parts["scale2"].float()
            parts["scale2"] = s.reshape(1) if blocks.dim() == 2 else s.reshape(-1, len(blocks)).squeeze(0)
        return QTensor(Layout(Scheme(f"ggml.{self.name}"), "gguf.native"), q.logical_shape, parts)


class QLinear(_Matmul):
    def __init__(self, ext: Extension, act: dict[str, str], ctx, **facts) -> None:
        super().__init__(ext, act, ctx, facts)
        self.ext, self.out_dtype = ext, getattr(torch, facts["out_dtype"])
        # mmvq up to `mmvq_rows` (the class's, for this shape and mode; fixed per graph bucket), else mmq
        self.mmvq_rows = mmvq_rows(ctx.fingerprint, self.name, act["mmq"], self.k, self.out, self.lib)
        self._uses_mmq(self.mmvq_rows, facts["layout"])
        # mmq's rows per call (D5), which size its workspaces
        self.call_rows = max(CHUNK, _cap(ctx) // CHUNK * CHUNK)
        self.workspace = ctx.workspace(f"ggml_cuda[in={self.k},rows={self.call_rows}]",
                                       self.lib.workspace_bytes(self.device, self.k, self.call_rows))
        self.f32 = self._f32_workspace(ctx, self.call_rows, facts["act_dtype"])
        self.q8_1 = ctx.workspace(f"ggml_cuda.q8_1[in={self.k}]", q8_1_bytes(MMVQ_MAX, self.k))   # prepared inputs

    def prepare_input(self, x: torch.Tensor) -> Prepared:
        """`x` with its q8_1 activations, when its rows take mmvq here (D6; std.ops.kernels, `qlinear@2`)."""
        rows = len(x)
        if not 0 < rows <= self.mmvq_rows:
            return Prepared(x)
        buffer = self.q8_1[:q8_1_bytes(rows, self.k)]
        self.lib.quantize_q8_1(self.type, x.contiguous(), buffer)
        generation = self.ext.written[self.q8_1.data_ptr()] = next(self.ext.generations)
        return Prepared(x, ("mmvq", "q8_1", rows, self.k), buffer, generation)

    def _prepared(self, x: Prepared) -> torch.Tensor | None:
        """The q8_1 activations `x` carries, if this call's mmvq would compute exactly them and no later
        `prepare_input` has overwritten them; else None."""
        if x.key != ("mmvq", "q8_1", len(x.x), self.k) or len(x.x) > self.mmvq_rows:
            return None
        return x.buffer if self.ext.written.get(x.buffer.data_ptr()) == x.generation else None

    def __call__(self, x, parts: dict[str, torch.Tensor]) -> torch.Tensor:
        xq = None
        if not isinstance(x, torch.Tensor):               # a prepare_input handle: ours (maybe ready), or another's
            xq, x = self._prepared(x) if isinstance(x, Prepared) else None, x.x
        x, rows, blocks, scale2 = x.contiguous(), len(x), parts["blocks"], parts.get("scale2")
        y = torch.empty(rows, self.out, dtype=self.out_dtype, device=x.device)
        if rows == 0:
            return y
        if rows <= self.mmvq_rows:                       # scale2 in its epilogue at any rows (D11)
            self._count("mmvq")
            self.lib.mmvq(self.type, blocks, x, y, scale2, self.workspace, xq=xq)
            return y
        self._count("mmq")
        blocks = self._mmq_blocks(parts)
        for start, end in calls(rows, self.call_rows):     # scale2 in f32, then one rounding to out_dtype (D4)
            self.lib.mmq(self.type, self.prec, blocks, self._mmq_input(x[start:end], self.f32), y[start:end], scale2,
                         self.workspace)
        return y


class QLinearGlu(QLinear):
    """`qlinear_glu@1` (glu "swiglu") on a merged gate/up of `out` rows: mmvq's fused gate at one row for the types
    where it pays (`GLU_TYPES`, D6), else None (the caller composes) — and None for halves that are not a multiple of
    4 rows: mmvq reads whole blocks of up to 4 rows at one column (small K: `nwarps` rows), so the up half's last
    block would read past the stack (memcheck)."""

    def __call__(self, x: torch.Tensor, parts: dict[str, torch.Tensor]) -> torch.Tensor | None:
        if len(x) != 1 or self.mmvq_rows < 1 or self.out // 2 % 4 or self.name not in GLU_TYPES:
            return None
        self._count("mmvq")
        blocks, half = parts["blocks"], self.out // 2
        y = torch.empty(1, half, dtype=self.out_dtype, device=x.device)
        self.lib.mmvq(self.type, blocks[half:], x.contiguous(), y, parts.get("scale2"), self.workspace,
                      gate=blocks[:half])
        return y


def _overlap(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Whether two contiguous tensors share a byte."""
    return a.data_ptr() < b.data_ptr() + b.nbytes and b.data_ptr() < a.data_ptr() + a.nbytes


def q8_1_bytes(rows: int, k: int) -> int:
    """Bytes of mmvq's q8_1 activations of `rows` rows of `k` values (block_q8_1 per 32, rows padded to 512)."""
    return rows * -(-k // ROW_PADDING) * ROW_PADDING // 32 * 36


class QEmbedding:
    def __init__(self, ext: Extension, ctx, *, layout: str, vocab: int, hidden: int, dtype: str) -> None:
        self.type, self.hidden, self.dtype = ggml_type(layout), hidden, getattr(torch, dtype)
        self.lib = ext(ctx.fingerprint.sm)
        self.gather = hidden % 256 != 0 and self.type == TYPES["iq4_nl"]     # Holon: see QK

    def __call__(self, ids: torch.Tensor, blocks: torch.Tensor) -> torch.Tensor:
        y = torch.empty(len(ids), self.hidden, dtype=self.dtype, device=blocks.device)
        if len(ids) and self.gather:
            self.lib.dequantize(self.type, blocks.index_select(0, ids), y)
        elif len(ids):
            self.lib.get_rows(self.type, blocks, ids.to(torch.int32), y)
        return y


class LinearExperts(_Matmul):
    def __init__(self, ext: Extension, act: dict[str, str], ctx, arena: bool = False, **facts) -> None:
        super().__init__(ext, act, ctx, facts)
        self.ext, self.rows_per_token = ext, facts["rows_per_token"]
        self.experts, self.top_k, self.swiglu = facts["experts"], facts["top_k"], facts["glu"] == "swiglu"
        # over an arena, slots below top_k are null slots (zero weights, std.moe.kernels): mmvq stores their entries'
        # zeros without reading them (a pass of the resident entries, or of the fetched ones, costs its own work only)
        self.null_slots = self.top_k if arena else 0
        self.out_y = self.out // 2 if self.swiglu else self.out
        # llama.cpp's choice per chunk from its token count (fixed per graph bucket): mmvq with ids, else mmq
        self.mmvq_tokens = max((n for n in range(1, 9) if self.lib.use_mmvq_id(self.device, self.type, n)), default=0)
        # rows per mmq call (std.moe.kernels' `pass_rows`), which sizes `scratch` and native FP4's f32 copy
        self.pass_rows = pass_rows(ctx, self.name, act["mmq"], self.top_k)
        self._uses_mmq(self.mmvq_tokens, facts["layout"])
        # the most distinct slots a call's ids name: one per expert, plus per routing rank one standing in for a miss
        # or a padding row (an offload arena's null slots, docs/dev/v1-offload.md D5); it bounds mmq's compact launch,
        # which
        # is how a stack of more slots than experts (a class arena) costs no more than the layer's own stack
        self.slots = self.experts + self.top_k
        # D4: SwiGLU in mmq's store (bitwise our SwiGLU of its unfused output), where mmq can for this stack
        self.glu_store = self.swiglu and self.lib.mmq_glu(self.device, self.type, self.prec, self.out)
        self.ids_j_tokens = IDS_J_TOKENS.get(ctx.fingerprint.sm, IDS_J_TOKENS_DEFAULT)      # D2's batched-decode rule
        # a device int32 word, or'd with ID_ERRORS bits by every mmq call with ids (the rows a violation names are not
        # computed; mmvq with ids does not check), one per build context, zero until a violation; the reader zeroes it
        self.errors = ctx.static("ggml_cuda.id_errors", (1,), torch.int32)
        # The persistent id workspace holds the mmvq calls' id maps and q8_1 activations (up to `mmvq_tokens` rows): one
        # per `in`. An mmq call's take a part of the caller's `scratch` past gate/up's output (an offload store's lent
        # tail in a streaming step: FN step 7, its 4,096-row workspaces were 120 MiB of statics), or a fresh buffer:
        # docs/dev/v1-moe-prefill.md §10 L3-2 measured that at 0.2% of the 5080 offload lock's prefill. Its size depends
        # on in, k (the id maps, the activations per slot), E (expert_bounds) and the rows: all four name it.
        rows = max(1, self.mmvq_tokens)
        self.workspace = ctx.workspace(
            f"ggml_cuda.id[in={self.k},top_k={self.top_k},experts={self.experts},rows={rows}]",
            self.lib.workspace_bytes_id(self.device, self.k, rows, self.top_k, self.slots))
        self.f32 = self._f32_workspace(ctx, self.pass_rows * facts["rows_per_token"], facts["act_dtype"])
        # prepared inputs (`MoeRouter.route_prepare`): QLinear's buffer for this `in`, the same generations
        self.q8_1 = ctx.workspace(f"ggml_cuda.q8_1[in={self.k}]", q8_1_bytes(MMVQ_MAX, self.k))

    def prepare_buffer(self, tokens: int) -> torch.Tensor | None:
        """Where `tokens` tokens' q8_1 activations go for this instance's mmvq to read (one input row per token, on
        the mmvq path), else None."""
        return self.q8_1[:q8_1_bytes(tokens, self.k)] if self.rows_per_token == 1 and 0 < tokens <= self.mmvq_tokens \
            else None

    def prepared(self, x: torch.Tensor, buffer: torch.Tensor) -> Prepared:
        """`x` [T, in] with its q8_1 activations just written into `buffer` (`prepare_buffer`'s): a handle standing for
        `x[:, None]`, valid until the buffer's next write."""
        generation = self.ext.written[self.q8_1.data_ptr()] = next(self.ext.generations)
        return Prepared(x[:, None], ("mmvq", "q8_1", len(x), self.k), buffer, generation)

    def _prepared(self, x: Prepared) -> torch.Tensor | None:
        """The q8_1 activations `x` carries, if this call's mmvq computes exactly them and no later write has
        overwritten them; else None."""
        if x.key != ("mmvq", "q8_1", len(x.x), self.k) or self.prepare_buffer(len(x.x)) is None:
            return None
        return x.buffer if self.ext.written.get(x.buffer.data_ptr()) == x.generation else None

    def _mmq_rows(self, tokens: int) -> int:
        """The rows of a call's largest mmq launch (0: it runs mmvq only)."""
        return max((b - a for a, b in calls(tokens, self.pass_rows) if b - a > self.mmvq_tokens), default=0)

    def scratch_bytes(self, tokens: int) -> int:
        """`scratch`'s bytes for a call of `tokens` tokens (std.moe.kernels, docs/dev/v1-moe-prefill.md D10): its mmq
        launches' transient, one launch at a time: gate/up's f32 output before our SwiGLU (none where D4's store
        applies SwiGLU), then the launch's id maps and q8_1 rows; 0 where it runs mmvq only."""
        rows = self._mmq_rows(tokens)
        return self._gate_up_bytes(rows) + self._id_bytes(rows)

    def _id_bytes(self, rows: int) -> int:
        """An mmq launch's id workspace of `rows` rows (0: none)."""
        if not rows:
            return 0
        nbytes = self.lib.workspace_bytes_id(self.device, self.k, rows, self.top_k, self.slots)
        return -(-nbytes // POOL_ALIGN) * POOL_ALIGN

    def _id_workspace(self, tokens: int, scratch: torch.Tensor | None) -> torch.Tensor | None:
        """The mmq launches' id workspace: `scratch` past gate/up's output, or a fresh buffer (None: mmvq only)."""
        rows = self._mmq_rows(tokens)
        if not (nbytes := self._id_bytes(rows)):
            return None
        at = self._gate_up_bytes(rows)
        if scratch is not None and scratch.numel() >= at + nbytes:
            return scratch[at:at + nbytes]
        return torch.empty(nbytes, dtype=torch.uint8, device=self.workspace.device)

    def _gate_up_bytes(self, rows: int) -> int:
        return -(-rows * self.top_k * self.out * 4 // POOL_ALIGN) * POOL_ALIGN if self.swiglu and not self.glu_store \
            else 0

    def _check_buffers(self, x: torch.Tensor, tokens: int, out, scratch) -> None:
        """D10's terms on passed buffers, before any launch."""
        if out is not None and (out.dtype != torch.float32 or out.device != x.device or not out.is_contiguous()
                                or tuple(out.shape) != (tokens, self.top_k, self.out_y)):
            raise ValueError(f"kernels.ggml_cuda: out {out.dtype} {tuple(out.shape)} on {out.device}: a contiguous f32 "
                             f"[{tokens}, {self.top_k}, {self.out_y}] on {x.device}")
        if scratch is not None and (scratch.dtype != torch.uint8 or scratch.device != x.device or scratch.dim() != 1
                                    or not scratch.is_contiguous() or scratch.data_ptr() % POOL_ALIGN
                                    or len(scratch) < self.scratch_bytes(tokens)):
            raise ValueError(f"kernels.ggml_cuda: scratch {scratch.dtype} of {scratch.numel()} bytes at "
                             f"{scratch.data_ptr() % POOL_ALIGN} past {POOL_ALIGN}: contiguous uint8 on {x.device}, "
                             f"{POOL_ALIGN}-byte aligned, at least {self.scratch_bytes(tokens)} bytes")
        if out is not None and (_overlap(out, x) or scratch is not None and _overlap(out, scratch)):
            raise ValueError("kernels.ggml_cuda: out overlaps x or scratch")
        if scratch is not None and _overlap(scratch, x):
            raise ValueError("kernels.ggml_cuda: scratch overlaps x")

    def _gate_up_out(self, tokens: int, device: torch.device, scratch: torch.Tensor | None) -> torch.Tensor | None:
        """Gate/up's f32 output before our SwiGLU (D10) from `scratch`, or a new buffer where the caller passed none;
        None where D4's store applies SwiGLU or the call runs mmvq only."""
        rows = self._mmq_rows(tokens)
        if not (nbytes := self._gate_up_bytes(rows)):
            return None
        buf = scratch if scratch is not None else torch.empty(nbytes, dtype=torch.uint8, device=device)
        return buf[:nbytes].view(torch.float32)[:rows * self.top_k * self.out].view(rows, self.top_k, self.out)

    def __call__(self, x, parts: dict[str, torch.Tensor], ids: torch.Tensor, *, out: torch.Tensor | None = None,
                 scratch: torch.Tensor | None = None) -> torch.Tensor:
        xq = None
        if not isinstance(x, torch.Tensor):            # a prepared handle: ours (maybe stale: then x), or another's
            xq, x = self._prepared(x) if isinstance(x, Prepared) else None, x.x
        x, ids, blocks = x.contiguous(), ids.contiguous(), parts["blocks"]
        # NVFP4's scale2 per expert: [E], or [2, E] (gate's row, up's) for a merged gate/up stack
        scale2 = parts.get("scale2")
        if scale2 is not None and scale2.shape != ((2, self.experts) if self.swiglu else (self.experts,)):
            raise ValueError(f"kernels.ggml_cuda: scale2 {tuple(scale2.shape)} for {self.experts} experts, glu "
                             f"{'swiglu' if self.swiglu else 'none'}")
        gate_scale, up_scale = (scale2[0], scale2[1]) if scale2 is not None and self.swiglu else (None, scale2)
        tokens = len(ids)
        # Decode and verify: mmvq only, which reads no `scratch`. An eager small step is host-bound, so this path does
        # none of D10's carving.
        if tokens <= self.mmvq_tokens:
            self._check_buffers(x, tokens, out, None)
            y = torch.empty(tokens, self.top_k, self.out_y, device=x.device) if out is None else out
            if tokens:
                self._count("mmvq")
                self.lib.mmvq_id(self.type, blocks.view(-1, blocks.shape[-1]), x, ids, y, self.swiglu, gate_scale,
                                 up_scale, self.workspace, xq=xq, null_slots=self.null_slots)
            return y
        self._check_buffers(x, tokens, out, scratch)
        y = torch.empty(tokens, self.top_k, self.out_y, device=x.device) if out is None else out
        gate_up = self._gate_up_out(tokens, x.device, scratch)
        ws = self._id_workspace(tokens, scratch)
        for start, end in calls(tokens, self.pass_rows):
            xs, ids_c, ys = x[start:end], ids[start:end], y[start:end]
            if len(ids_c) <= self.mmvq_tokens:
                self._count("mmvq")
                self.lib.mmvq_id(self.type, blocks.view(-1, blocks.shape[-1]), xs, ids_c, ys, self.swiglu,
                                 gate_scale, up_scale, self.workspace, xq=xq, null_slots=self.null_slots)
                continue
            self._count("mmq")
            w, xs = self._mmq_blocks(parts).view(-1, blocks.shape[-1]), self._mmq_input(xs, self.f32)
            if not self.swiglu:
                self.lib.mmq_id(self.type, self.prec, w, xs, ids_c, ys, ws, self.slots, self.errors,
                                self.ids_j_tokens)
                if up_scale is not None:
                    self._scale_slots(ys, up_scale, ids_c)
                continue
            if self.glu_store:
                self.lib.mmq_id_glu(self.type, self.prec, w, xs, ids_c, ys, ws, self.slots, self.errors,
                                    gate_scale, up_scale, self.ids_j_tokens)
                continue
            self.lib.mmq_id(self.type, self.prec, w, xs, ids_c, gate_up[:len(ids_c)], ws, self.slots,
                            self.errors, self.ids_j_tokens)
            self._swiglu(gate_up[:len(ids_c)], ys, gate_scale, up_scale, ids_c)
        return y

    @staticmethod
    def _scale_slots(y: torch.Tensor, scale: torch.Tensor, ids: torch.Tensor) -> None:
        """y [T, k, out] *= the scale of each slot's expert (a device gather: graph-safe); a −1 entry's row × 1, so
        untouched (the contract: rows another call of the pass wrote)."""
        y.mul_(torch.where(ids >= 0, scale[ids.clamp_min(0)], 1.0)[..., None])

    def _swiglu(self, gate_up: torch.Tensor, y: torch.Tensor, gate_scale, up_scale, ids: torch.Tensor) -> None:
        """Our f32 SwiGLU of mmq's gate/up rows, each half first scaled by its row's expert scale (if any); rows of −1
        entries untouched."""
        self.lib.swiglu(gate_up, y, gate_scale, up_scale, ids.reshape(-1))


class MoeTopk:
    def __init__(self, ext: Extension, ctx, *, experts: int, top_k: int, renormalize: bool) -> None:
        self.top_k, self.renormalize, self.lib = top_k, renormalize, ext(ctx.fingerprint.sm)

    def __call__(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = logits.contiguous()
        weights = torch.empty(len(logits), self.top_k, device=logits.device)
        ids = torch.empty(len(logits), self.top_k, dtype=torch.int32, device=logits.device)
        if len(logits):
            self.lib.topk_moe(logits, weights, ids, self.renormalize)
        return weights, ids


class MoeRouter:
    """`moe_router@1`: decode and verify calls (up to `rows` tokens) in one launch — topk-moe.cu's router kernel, ours
    through vendor.patch: per row and expert four warps' f32 fma chains over quarters of hidden, each closed by the
    warp's xor tree, added in order, then `moe_topk`'s selection on those logits in the last block of the row's tile.
    Prompt chunks (calls of at least CHUNK rows, experts a multiple of 64; docs/dev/v1-moe-prefill.md D3): our router
    GEMM, per logit a fixed number of f32 fma chains (2,048 / experts) over equal parts of hidden ascending, added in
    order, from x's values as given (`moe_router_logits`, no f32 copy), then `moe_topk`'s selection; calls between, as
    `moe_topk@1` composed with the f32 linear, as before (a row's logits follow its call's path, as `qlinear`'s mmvq /
    mmq: batched decode and verify keep their bits). The GEMM's rows do not depend on the call, so it declares
    `pass_rows` (D3′): std.moe routes an expert pass of whole CHUNKs in one launch. Its logits scratch is a
    workspace, its tile counters a static buffer each call leaves zeroed."""

    def __init__(self, ext: Extension, ctx, *, hidden: int, experts: int, top_k: int, renormalize: bool,
                 dtype: str, router_dtype: str = "float32") -> None:
        self.top_k, self.renormalize, self.lib = top_k, renormalize, ext(ctx.fingerprint.sm)
        self.topk = MoeTopk(ext, ctx, experts=experts, top_k=top_k, renormalize=renormalize)
        self.rows = self.lib.moe_router_rows()
        self.logits = ctx.workspace(f"ggml_cuda.router[experts={experts}]", self.rows * experts * 4)
        self.done = ctx.static("ggml_cuda.router_done", (self.lib.moe_router_tiles(self.rows),), torch.int32)
        # the router GEMM's expert tile and hidden steps
        self.prefill = experts % 64 == 0 and hidden % (32 * self.lib.moe_router_partials(experts)) == 0
        # D3′: the GEMM is row-invariant, so a call of whole CHUNKs gives each row its chunk's bits, up to an expert
        # pass (std.moe.kernels' `pass_rows` term: std routes a pass in one call)
        self.pass_rows = max(CHUNK, _cap(ctx) // CHUNK * CHUNK) if self.prefill else CHUNK

    def __call__(self, x: torch.Tensor, router: torch.Tensor, xq: torch.Tensor | None = None):
        if len(x) >= CHUNK and self.prefill:
            return self.topk(self._logits(x, router))
        if not 0 < len(x) <= self.rows:
            return self.topk(torch.nn.functional.linear(x.float(), router.float()))
        x = x.contiguous()
        if x.data_ptr() % 16:                                # the kernel's vector loads
            x = x.clone()
        weights = torch.empty(len(x), self.top_k, device=x.device)
        ids = torch.empty(len(x), self.top_k, dtype=torch.int32, device=x.device)
        logits = self.logits.view(torch.float32)[:len(x) * router.shape[0]].view(len(x), -1)
        self.lib.moe_router(x, router, logits, self.done, weights, ids, self.renormalize, xq=xq)
        return weights, ids

    def _logits(self, x: torch.Tensor, router: torch.Tensor) -> torch.Tensor:
        """A prompt chunk's f32 logits [T, E] (the router GEMM, D3)."""
        logits = torch.empty(len(x), router.shape[0], device=x.device)
        partials = torch.empty(self.lib.moe_router_partials(len(router)), *logits.shape, device=x.device)
        self.lib.moe_router_logits(x.contiguous(), router.contiguous(), partials, logits)
        return logits

    def route_prepare(self, x: torch.Tensor, router: torch.Tensor, experts):
        """`moe_router@1`'s result and the experts' input `x[:, None]`: for this provider's `linear_experts` instance
        `experts` on its mmvq path, a handle to x's q8_1 activations written by the router's launch (no quantize of
        its own; std.moe.kernels), else the tensor."""
        buffer = experts.prepare_buffer(len(x)) if isinstance(experts, LinearExperts) and len(x) <= self.rows else None
        weights, ids = self(x, router, buffer)
        return weights, ids, x[:, None] if buffer is None else experts.prepared(x, buffer)


class MoeCombine:
    def __init__(self, ext: Extension, ctx, **facts) -> None:
        self.lib = ext(ctx.fingerprint.sm)

    def __call__(self, y: torch.Tensor, weights: torch.Tensor, out: torch.Tensor) -> None:
        if len(y):
            self.lib.moe_combine(y.contiguous(), weights.contiguous(), out)

    def add(self, y: torch.Tensor, weights: torch.Tensor, residual: torch.Tensor, out: torch.Tensor) -> None:
        """`residual` + the combine into `out`, in the combine's kernel (bitwise the unfused add)."""
        if len(y):
            self.lib.moe_combine(y.contiguous(), weights.contiguous(), out, residual.contiguous())


def _supports_matmul(f: dict, act_quant: tuple[str, ...], types) -> bool:
    name = weight_type(f["layout"])
    return (name in types and f["in"] % ROW_BLOCKS.get(name, 256) == 0 and f["act_dtype"] in DTYPES
            and f["act_quant"] in act_quant)


def _supports_qlinear(f: dict, act_quant=ACT_QUANT, types=TYPES) -> bool:
    return _supports_matmul(f, act_quant, types) and f["out_dtype"] in DTYPES


def _supports_qlinear_glu(f: dict, act_quant=ACT_QUANT, types=TYPES) -> bool:
    return _supports_qlinear(f, act_quant, types) and f["glu"] == "swiglu" and f["out"] % 2 == 0


def _supports_qembedding(f: dict) -> bool:
    # get_rows has no NVFP4
    name = weight_type(f["layout"])
    return (ggml_type(f["layout"]) not in (None, TYPES["nvfp4"]) and f["hidden"] % QK.get(name, 256) == 0
            and f["dtype"] in DTYPES)


def _supports_linear_experts(f: dict, act_quant=ACT_QUANT, types=TYPES) -> bool:
    # out_y a multiple of 4: mmvq with ids at one token bounds a block's rows (up to 4) by the slot stride, not out_y
    swiglu = f["glu"] == "swiglu"
    return (_supports_matmul(f, act_quant, types) and f["out_dtype"] == "float32" and f["glu"] in ("none", "swiglu")
            and f["out"] % (8 if swiglu else 4) == 0 and f["rows_per_token"] in (1, f["top_k"])
            and CHUNK * f["top_k"] <= ID_PAIRS)


def _supports_moe_topk(f: dict) -> bool:
    e = f["experts"]
    return 0 < e <= 512 and e & (e - 1) == 0 and 0 < f["top_k"] <= e     # topk-moe.cu's instances


def _supports_moe_router(f: dict) -> bool:
    return (_supports_moe_topk(f) and f["top_k"] <= 32 and f["hidden"] % 512 == 0 and f["dtype"] in DTYPES
            and f.get("router_dtype", "float32") in ("float32", "bfloat16"))


def _supports_moe_combine(f: dict) -> bool:
    return f["dtype"] == "bfloat16" and f["y_dtype"] == "float32"


class Plugin:
    def register(self, regs) -> None:
        ext = Extension()

        def add(op: str, impl, supports, sms=frozenset(extension.ARCHS), priority: int = 10, act=None) -> None:
            regs.kernels.add(op, KernelSpec(
                op=op, provider="kernels.ggml_cuda", impl=impl, requires=Constraint(sms=sms), supports=supports,
                props=frozenset({"graph_safe", "deterministic"}), priority=priority,
                config=({"act": act} if act else {}) | ({"fused": ["qlinear_glu@1"]} if op == QLINEAR else {})))

        # q8_1 everywhere; native W4A4 on Blackwell for the FP4 types, opt-in (`quant.act = a4`) and then preferred
        for act, sms, priority, act_quant, types in ((Q8_1, frozenset(extension.ARCHS), 10, ACT_QUANT, TYPES),
                                                     (NATIVE_FP4, frozenset({120}), 11, ("a4",), FP4)):
            for op, impl, supports in ((QLINEAR, QLinear, _supports_qlinear),
                                       (QLINEAR_GLU, QLinearGlu, _supports_qlinear_glu),
                                       (LINEAR_EXPERTS, LinearExperts, _supports_linear_experts),
                                       (LINEAR_EXPERTS_ARENA, functools.partial(LinearExperts, arena=True),
                                        _supports_linear_experts)):
                add(op, functools.partial(impl, ext, act),
                    functools.partial(supports, act_quant=act_quant, types=types), sms, priority, act)
        add(QEMBEDDING, functools.partial(QEmbedding, ext), _supports_qembedding)
        add(MOE_TOPK, functools.partial(MoeTopk, ext), _supports_moe_topk)
        add(MOE_ROUTER, functools.partial(MoeRouter, ext), _supports_moe_router)
        add(MOE_COMBINE, functools.partial(MoeCombine, ext), _supports_moe_combine)
