#!/usr/bin/env bash
# Port-feasibility probe: compile llama.cpp's current quantized matmul kernels standalone (no ggml-cuda.cu,
# no ggml-base) and list the symbols a shim would have to provide. Usage: scripts/probe_llamacpp_port.sh [sm]
set -euo pipefail
SM=${1:-86}
HERE=$(cd "$(dirname "$0")/.." && pwd)
L=$HERE/third_party/llama.cpp/ggml
OUT=$HERE/out/port-probe-sm$SM
mkdir -p "$OUT"
FILES="mmvq.cu quantize.cu mmid.cu mmq.cu"
for t in iq3_xxs iq3_s iq4_xs q3_k q4_k q8_0; do FILES="$FILES template-instances/mmq-instance-$t.cu"; done
for f in $FILES; do
  nvcc -std=c++17 -O3 -arch=sm_$SM -I"$L/include" -I"$L/src" -I"$L/src/ggml-cuda" \
    -c "$L/src/ggml-cuda/$f" -o "$OUT/$(basename "$f" .cu).o" &
done
wait
nm -C --defined-only "$OUT"/*.o | awk '$2 ~ /[TWtw]/ {$1=$2=""; sub(/^ +/, ""); print}' | sort -u > "$OUT/defined.txt"
nm -C --undefined-only "$OUT"/*.o | awk '{$1=""; sub(/^ +/, ""); print}' | sort -u | grep -E "ggml" | grep -vxFf "$OUT/defined.txt" \
  | tee "$OUT/undefined_ggml.txt"
