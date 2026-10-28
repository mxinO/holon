#!/usr/bin/env bash
# Whole S2 measurement sequence on one GPU: scripts/run_all_sm.sh <sm>  (GPU pinned by CUDA_VISIBLE_DEVICES)
set -euo pipefail
SM=$1
HERE=$(cd "$(dirname "$0")/.." && pwd)
PY=${S2_PY:-$HERE/../../.venv/bin/python}
: "${CUDA_VISIBLE_DEVICES:?pin the GPU}"
cd "$HERE"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=name,memory.used,utilization.gpu --format=csv
# ninja must be on PATH for torch cpp_extension
$PY scripts/build_ext.py "$SM"
$PY scripts/correctness.py
$PY scripts/graph_check.py
$PY scripts/bench_ops.py
scripts/run_llamacpp_ops.sh "$SM"
scripts/run_llama_bench.sh "$SM"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=name,memory.used,utilization.gpu --format=csv
