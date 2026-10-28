#!/usr/bin/env bash
# Record the fp32 CPU golden and all GPU candidates for one model, then compute its noise-floor table.
#   ./run_all.sh Qwen/Qwen3-0.6B qwen3-0.6b
# GPUs 4,5 only; a fixed CPU thread count for the fp32 recorder, 2 per GPU process.
set -euo pipefail
cd "$(dirname "$0")"
PY=../../.venv/bin/python
MODEL=$1
DIR=out/$2
mkdir -p "$DIR" results

rec() {  # rec <name> <args...>
  local name=$1; shift
  "$PY" golden.py --model "$MODEL" --full-stride 32 --out "$DIR/$name.safetensors" "$@" > "$DIR/$name.log" 2>&1
  echo "$name done ($(grep -o '"wall_s": [0-9.]*' "$DIR/$name.log"))"
}
gpu() {  # gpu <device> <name> <args...>
  local dev=$1; shift
  CUDA_VISIBLE_DEVICES=$dev OMP_NUM_THREADS=2 rec "$@"
}

(gpu 4 bf16 --device cuda --dtype bfloat16 --threads 2
 gpu 4 bf16-r2 --device cuda --dtype bfloat16 --threads 2
 gpu 4 bf16-chunk256 --device cuda --dtype bfloat16 --chunk 256 --threads 2
 gpu 4 bug-eps --device cuda --dtype bfloat16 --threads 2 --override 'rms_norm_eps=1e-5') &
(gpu 5 bf16-eager --device cuda --dtype bfloat16 --attn eager --threads 2
 gpu 5 fp16 --device cuda --dtype float16 --threads 2
 gpu 5 bug-rope --device cuda --dtype bfloat16 --threads 2 \
   --override 'rope_parameters={"rope_type":"default","rope_theta":900000.0}') &
CUDA_VISIBLE_DEVICES= rec fp32 --device cpu --dtype float32 --threads 44
wait
"$PY" noise_floor.py --dir "$DIR" --out "results/noise_floor_$2.json"
