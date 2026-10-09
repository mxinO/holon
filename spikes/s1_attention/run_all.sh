#!/bin/sh
# Spike S1 end to end on one GPU.  Usage: run_all.sh <gpu index> <log suffix> [step ...]
#   A5000 host:  spikes/s1_attention/run_all.sh 0 a5000
#   while another spike uses the GPU: S1_WAIT_FOR_IDLE=s2_gguf spikes/s1_attention/run_all.sh 6 sm120
# Results: results/*_<class>.json; logs: out/*_<suffix>.log.
set -u
cd "$(dirname "$0")"
suffix=$2
# S1_PER_CLASS_CACHE=1: one JIT/Triton cache per class, so hosts sharing this directory do not race.
if [ -n "${S1_PER_CLASS_CACHE:-}" ]; then fi_ws=$PWD/out/fi_ws_$suffix tri=$PWD/out/triton_cache_$suffix; else fi_ws=$PWD/out/fi_ws tri=$PWD/out/triton_cache; fi
export CUDA_VISIBLE_DEVICES=$1 FLASHINFER_WORKSPACE_BASE=$fi_ws TRITON_CACHE_DIR=$tri
# ninja must be on PATH for the FlashInfer JIT; S1_PY overrides the interpreter
PY=${S1_PY:-../../.venv/bin/python}
shift 2

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
        bf16rep) run bench_bf16_rep2 bench.py --kv bf16 --modes decode verify --tag _rep2 ;;
        fp8) run bench_fp8 bench.py --kv fp8 --modes decode verify ;;
        q8) run bench_q8 bench_q8.py ;;
    esac
done
