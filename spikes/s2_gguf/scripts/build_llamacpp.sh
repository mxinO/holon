#!/usr/bin/env bash
# Clone llama.cpp at the pinned commit (MIT), apply the S2 perf-case patch, build for one SM:
#   scripts/build_llamacpp.sh 86|120
set -euo pipefail
SM=$1
LLAMA_COMMIT=51ce9c11a6f2dfa895696c0048c4333e8953b728
HERE=$(cd "$(dirname "$0")/.." && pwd)
SRC=$HERE/third_party/llama.cpp
BUILD=$HERE/out/llama-sm$SM
if [ ! -d "$SRC" ]; then
  git clone https://github.com/ggml-org/llama.cpp.git "$SRC"
  git -C "$SRC" checkout "$LLAMA_COMMIT"
fi
test "$(git -C "$SRC" rev-parse HEAD)" = "$LLAMA_COMMIT" || { echo "llama.cpp not at $LLAMA_COMMIT"; exit 1; }
git -C "$SRC" diff --quiet && git -C "$SRC" apply "$HERE/patches/llama.cpp-s2-perf-cases.patch"
cmake -S "$SRC" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$SM -DCMAKE_CUDA_COMPILER=$(which nvcc) \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_SERVER=OFF
cmake --build "$BUILD" -j 48 --target test-backend-ops llama-bench
