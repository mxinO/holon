#!/usr/bin/env bash
# End-to-end llama.cpp baseline (pp512 / tg128, all layers on GPU, flash attention):
#   scripts/run_llama_bench.sh <sm>   (GPU pinned by CUDA_VISIBLE_DEVICES)
set -euo pipefail
SM=$1
HERE=$(cd "$(dirname "$0")/.." && pwd)
: "${CUDA_VISIBLE_DEVICES:?pin the GPU}"
PY=.venv/bin/python
for key in 8B-Q4_K_M 8B-UD-IQ3_XXS 30B-A3B-UD-IQ3_XXS; do
  model=$(cd "$HERE" && $PY -c "from models import path; print(path('$key'))")
  "$HERE/out/llama-sm$SM/bin/llama-bench" -m "$model" -p 512 -n 128 -ngl 99 -fa on -r 5 -o json \
    > "$HERE/results/llama_bench_${key}_sm$SM.json"
done
