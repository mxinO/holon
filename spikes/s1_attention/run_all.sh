#!/bin/sh
# Spike S1 end to end on one GPU.  Usage: run_all.sh <gpu index> <log suffix> [step ...]
#   A5000 host:  spikes/s1_attention/run_all.sh 0 a5000
#   the Blackwell workstation:   S1_WAIT_FOR_IDLE=s2_gguf spikes/s1_attention/run_all.sh 6 sm120
# Results: results/*_<class>.json; logs: out/*_<suffix>.log.
set -u
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=$1 FLASHINFER_WORKSPACE_BASE=$PWD/out/fi_ws TRITON_CACHE_DIR=$PWD/out/triton_cache
export PATH="$(python3 -c 'import sys; print(sys.base_prefix)')/bin:$PATH"          # ninja for the FlashInfer JIT
suffix=$2
shift 2
PY=../../.venv/bin/python

run() {   # run <log name> <script> [args]
    if [ -n "${S1_WAIT_FOR_IDLE:-}" ]; then          # shared GPU: wait while another job benchmarks on it
        while pgrep -f "$S1_WAIT_FOR_IDLE" > /dev/null; do sleep 30; done
    fi
    log=$1
    shift
    echo "== $*"
    $PY "$@" > "out/${log}_${suffix}.log" 2>&1
}

steps=${*:-"jit prebuild sweep graph det prep bf16 fp8 q8"}
mkdir -p out results
for s in $steps; do
    case $s in
        jit) run jit jit_time.py ;;
        prebuild) run prebuild prebuild.py ;;
        sweep) run sweep sweep_triton.py ;;
        graph) run graph graph_test.py ;;
        det) run det determinism.py ;;
        prep) run prep prep_cost.py ;;
        bf16) run bench_bf16 bench.py --kv bf16 ;;
        fp8) run bench_fp8 bench.py --kv fp8 --modes decode verify ;;
        q8) run bench_q8 bench_q8.py ;;
    esac
done
