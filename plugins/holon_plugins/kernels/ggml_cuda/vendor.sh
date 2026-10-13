#!/usr/bin/env bash
# llama.cpp's quantized CUDA matmul kernels and MoE routing / combine kernels (MIT), vendored at a pinned commit
# (M0.1 design §4.4, M0.2 design §4.3).
#
#   ./vendor.sh [llama.cpp checkout]   refresh vendor/: copy the files below from a checkout at $COMMIT (cloned when
#                                       omitted), record their upstream SHA256SUMS, apply vendor.patch
#   ./vendor.sh --check                verify vendor/ == upstream files (SHA256SUMS) + vendor.patch, offline
#
# A refresh to a new commit: change COMMIT (and [provenance] in plugin.toml), run ./vendor.sh, fix vendor.patch until
# it applies, then the link test and L0 (plugins/tests/test_ggml_cuda.py) on every SM.
set -euo pipefail
COMMIT=cbe28bb3cb0758f0c566a9a3561766821aeb0748
HERE=$(cd "$(dirname "$0")" && pwd)
CUDA=ggml/src/ggml-cuda
FILES="LICENSE
ggml/include/ggml.h ggml/include/ggml-alloc.h ggml/include/ggml-backend.h ggml/include/ggml-cuda.h ggml/include/gguf.h
ggml/src/ggml-common.h ggml/src/ggml-impl.h
$CUDA/common.cuh $CUDA/vendors/cuda.h $CUDA/mma.cuh $CUDA/vecdotq.cuh $CUDA/dequantize.cuh $CUDA/unary.cuh
$CUDA/mmq.cu $CUDA/mmq.cuh $CUDA/mmq-load-tiles.cuh $CUDA/mmq-vec-dot.cuh $CUDA/mmid.cu $CUDA/mmid.cuh
$CUDA/mmvq.cu $CUDA/mmvq.cuh $CUDA/quantize.cu $CUDA/quantize.cuh $CUDA/getrows.cu $CUDA/getrows.cuh
$CUDA/convert.cu $CUDA/convert.cuh
$CUDA/topk-moe.cu $CUDA/topk-moe.cuh $CUDA/moe-weighted-reduction.cu $CUDA/moe-weighted-reduction.cuh"
for c in ampere blackwell cdna gcn pascal-dp4a pascal-older rdna2 rdna3 rdna3-5 rdna4; do
  FILES="$FILES $CUDA/mmq-config-$c.cuh"
done
for t in q1_0 q2_0 q4_0 q4_1 q5_0 q5_1 q8_0 q2_k q3_k q4_k q5_k q6_k \
         iq1_s iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_nl iq4_xs mxfp4 nvfp4; do
  FILES="$FILES $CUDA/template-instances/mmq-instance-$t.cu"
done

if [ "${1:-}" = "--check" ]; then
  tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
  cp -r "$HERE/vendor" "$tmp/vendor"
  patch -s -R -p1 -d "$tmp/vendor" < "$HERE/vendor.patch"
  (cd "$tmp/vendor" && sha256sum --quiet -c "$HERE/SHA256SUMS")
  diff <(cd "$tmp/vendor" && find . -type f | sed 's|^\./||' | sort) <(awk '{print $2}' "$HERE/SHA256SUMS" | sort)
  echo "ok   vendor/ = llama.cpp ${COMMIT:0:8} + vendor.patch"
  exit 0
fi

SRC=${1:-}
if [ -z "$SRC" ]; then
  SRC=$(mktemp -d); trap 'rm -rf "$SRC"' EXIT
  git clone -q --filter=blob:none https://github.com/ggml-org/llama.cpp.git "$SRC"
  git -C "$SRC" checkout -q "$COMMIT"
fi
test "$(git -C "$SRC" rev-parse HEAD)" = "$COMMIT" || { echo "$SRC is not at $COMMIT" >&2; exit 1; }
rm -rf "$HERE/vendor"
for f in $FILES; do
  git -C "$SRC" diff --quiet HEAD -- "$f" || { echo "$SRC/$f has local changes" >&2; exit 1; }
  install -D -m 644 "$SRC/$f" "$HERE/vendor/$f"
done
(cd "$HERE/vendor" && sha256sum $FILES) > "$HERE/SHA256SUMS"
patch -s -p1 -d "$HERE/vendor" < "$HERE/vendor.patch"
echo "vendored $(echo $FILES | wc -w) files from llama.cpp ${COMMIT:0:8}"
