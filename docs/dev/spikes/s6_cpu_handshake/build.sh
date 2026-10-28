#!/usr/bin/env bash
# Build the S6 extension for the visible GPU into ./build (bench.py also builds on first use).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:?pin the GPU, e.g. CUDA_VISIBLE_DEVICES=0} "$PY" bench.py --build-only
