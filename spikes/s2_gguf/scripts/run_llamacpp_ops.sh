#!/usr/bin/env bash
# llama.cpp per-kernel perf on the S2 shapes: scripts/run_llamacpp_ops.sh <sm> (GPU pinned by CUDA_VISIBLE_DEVICES)
set -euo pipefail
SM=$1
HERE=$(cd "$(dirname "$0")/.." && pwd)
: "${CUDA_VISIBLE_DEVICES:?pin the GPU}"
TAG=${S2_TAG:-sm$SM}
S2_SHAPES=1 "$HERE/out/llama-$TAG/bin/test-backend-ops" perf -b CUDA0 -o MUL_MAT,MUL_MAT_ID --output sql \
  > "$HERE/results/llamacpp_ops_$TAG.sql"
