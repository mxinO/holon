#!/usr/bin/env bash
# Vendor SGLang's ggml-derived GGUF CUDA headers (unmodified) into third_party/sglang_gguf.
# Source: sgl-project/sglang @ $SGLANG_COMMIT, python/sglang/kernels/aot/csrc/quantization/gguf (Apache-2.0;
# derived via vLLM from llama.cpp ggml-cuda b2899, MIT). gguf_kernel.cu (the torch wrapper) is not vendored:
# csrc/ggml_ext.cu is our own wrapper. patches/*.patch are applied on top (see README).
set -euo pipefail
SGLANG_COMMIT=968726f3cea0f6d6f44bea84797e5e5914658f4e
SUB=python/sglang/kernels/aot/csrc/quantization/gguf
HERE=$(cd "$(dirname "$0")/.." && pwd)
DST=$HERE/third_party/sglang_gguf
SRC=${SGLANG_SRC:-/tmp/cmp/sglang}
if [ ! -d "$SRC/.git" ]; then
  SRC=$HERE/third_party/sglang-src
  git clone --filter=blob:none --no-checkout https://github.com/sgl-project/sglang.git "$SRC"
  git -C "$SRC" sparse-checkout set "$SUB" python/sglang/kernels/aot/LICENSE
  git -C "$SRC" checkout "$SGLANG_COMMIT"
fi
test "$(git -C "$SRC" rev-parse HEAD)" = "$SGLANG_COMMIT" || { echo "sglang not at $SGLANG_COMMIT"; exit 1; }
mkdir -p "$DST"
for f in ggml-common.h vecdotq.cuh dequantize.cuh mmvq.cuh mmq.cuh moe.cuh moe_vec.cuh; do
  cp "$SRC/$SUB/$f" "$DST/$f"
done
cp "$SRC/python/sglang/kernels/aot/LICENSE" "$DST/LICENSE"
(cd "$DST" && sha256sum *.h *.cuh > SHA256SUMS.upstream)
for p in "$HERE"/patches/*.patch; do patch -s -d "$DST" -p1 < "$p"; done
echo "vendored into $DST"
