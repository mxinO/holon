#!/usr/bin/env bash
# Pinned baseline engines under bench/.envs/ (gitignored): one venv or build per baseline. Holon's own .venv is never
# touched: each baseline brings its own torch stack, and run.py starts it with that venv's python.
# Usage: bash bench/setup_baselines.sh [minisgl] [nanovllm] [llamacpp] [sglang] [freetoken]   (default: all)
# llamacpp and freetoken compile CUDA code: run them where nvcc (CUDA 13) is on PATH.
set -euo pipefail
cd "$(dirname "$0")"

MINISGL_COMMIT=9a91cfafe754aa85daee49998176275667eb58f2    # 2026-05-17
NANOVLLM_COMMIT=bb823b3e06983d71485a8e1f23715ebd87d98ef8   # 2026-04-26
# llama.cpp: the gate runs at the commit kernels.ggml_cuda vendors once M0.6 Gate A merges (release design D5); the
# newest upstream is a second, ungated column, re-pinned when the suite runs.
LLAMA_GATE=cbe28bb3cb0758f0c566a9a3561766821aeb0748
LLAMA_UPSTREAM=df18bee11e96557b07aa7b12db0976f3c7ac9165   # upstream HEAD, 2026-10-14
LLAMA_ARCHS="86;89;120a"                                   # the release classes: sm_86, sm_89, sm_120 (fp4 needs 120a)
FREETOKEN_COMMIT=74a387777708b9650a027494bcb37d2ab9cf2443  # 2026-10-10 (torch 2.14.1, sglang-kernel 0.4.9)
SGLANG_VERSION=0.5.21                                      # 2026-10-01 release (torch 2.13.0, CUDA 13)
RESOLVED_BEFORE=2026-10-07   # uv --exclude-newer: dependency resolution is reproducible from this date
RELEASE_RESOLVED_BEFORE=2026-10-14   # the same for the release design's baselines (sglang, freetoken)
# nano-vllm needs flash-attn; a prebuilt wheel for torch 2.9 + CUDA 12 avoids a long source build.
FLASH_ATTN="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

checkout() {  # name url commit
  [ -d ".envs/src/$1" ] || git clone -q "$2" ".envs/src/$1"
  git -C ".envs/src/$1" fetch -q origin
  git -C ".envs/src/$1" checkout -q "$3"
}

venv() {  # name requirements...
  local name=$1; shift
  [ -x ".envs/$name/bin/python" ] || uv venv -q --python 3.12 ".envs/$name"
  uv pip install -q --python ".envs/$name/bin/python" --exclude-newer "${EXCLUDE_NEWER:-$RESOLVED_BEFORE}" "$@"
}

llamacpp() {  # name commit: llama-bench and llama-server in .envs/llama.cpp-<name>/bin
  checkout "llama.cpp-$1" https://github.com/ggml-org/llama.cpp.git "$2"
  cmake -S ".envs/src/llama.cpp-$1" -B ".envs/llama.cpp-$1" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="$LLAMA_ARCHS" -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF > /dev/null
  cmake --build ".envs/llama.cpp-$1" -j "${JOBS:-16}" --target llama-bench llama-server > /dev/null
  echo "llama.cpp-$1: $(".envs/llama.cpp-$1/bin/llama-server" --version 2>&1 | grep version)"
}

targets=("$@")
[ $# -gt 0 ] || targets=(minisgl nanovllm llamacpp sglang freetoken)
for b in "${targets[@]}"; do
  case $b in
    minisgl)   # pins torch < 2.10 (the shared stack is 2.11), so it installs its own: torch 2.9.1 + cu128
      checkout mini-sglang https://github.com/sgl-project/mini-sglang.git "$MINISGL_COMMIT"
      venv minisgl -e .envs/src/mini-sglang ;;
    nanovllm)
      checkout nano-vllm https://github.com/GeeeekExplorer/nano-vllm.git "$NANOVLLM_COMMIT"
      venv nanovllm torch==2.9.1 "flash-attn @ $FLASH_ATTN" -e .envs/src/nano-vllm ;;
    llamacpp)
      llamacpp gate "$LLAMA_GATE"
      llamacpp upstream "$LLAMA_UPSTREAM"
      continue ;;
    sglang)    # HF checkpoints only (release design §5.1); token ids in, so no tokenizer process is needed
      EXCLUDE_NEWER=$RELEASE_RESOLVED_BEFORE venv sglang "sglang==$SGLANG_VERSION" ;;
    freetoken)  # its C++ extension builds against the torch in its venv: torch first, then no build isolation
      checkout FreeToken https://github.com/FlashML-org/FreeToken.git "$FREETOKEN_COMMIT"
      EXCLUDE_NEWER=$RELEASE_RESOLVED_BEFORE venv freetoken torch==2.14.1 setuptools wheel
      EXCLUDE_NEWER=$RELEASE_RESOLVED_BEFORE venv freetoken --no-build-isolation -e ".envs/src/FreeToken[accel]" ;;
    *) echo "unknown baseline: $b" >&2; exit 2 ;;
  esac
  echo "$b: $(.envs/"$b"/bin/python -c 'import torch; print("torch", torch.__version__)')"
done
