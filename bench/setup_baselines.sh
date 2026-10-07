#!/usr/bin/env bash
# Create one pinned venv per baseline engine under bench/.envs/ (gitignored). Holon's own .venv is never touched:
# each baseline brings its own torch stack, and run.py starts it with that venv's python.
# Usage: bash bench/setup_baselines.sh [minisgl] [nanovllm]     (default: both)
set -euo pipefail
cd "$(dirname "$0")"

MINISGL_COMMIT=9a91cfafe754aa85daee49998176275667eb58f2    # 2026-05-17
NANOVLLM_COMMIT=bb823b3e06983d71485a8e1f23715ebd87d98ef8   # 2026-04-26
RESOLVED_BEFORE=2026-10-07   # uv --exclude-newer: dependency resolution is reproducible from this date
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
  uv pip install -q --python ".envs/$name/bin/python" --exclude-newer "$RESOLVED_BEFORE" "$@"
}

targets=("$@")
[ $# -gt 0 ] || targets=(minisgl nanovllm)
for b in "${targets[@]}"; do
  case $b in
    minisgl)   # pins torch < 2.10 (the shared stack is 2.11), so it installs its own: torch 2.9.1 + cu128
      checkout mini-sglang https://github.com/sgl-project/mini-sglang.git "$MINISGL_COMMIT"
      venv minisgl -e .envs/src/mini-sglang ;;
    nanovllm)
      checkout nano-vllm https://github.com/GeeeekExplorer/nano-vllm.git "$NANOVLLM_COMMIT"
      venv nanovllm torch==2.9.1 "flash-attn @ $FLASH_ATTN" -e .envs/src/nano-vllm ;;
    *) echo "unknown baseline: $b" >&2; exit 2 ;;
  esac
  echo "$b: $(.envs/"$b"/bin/python -c 'import torch; print("torch", torch.__version__)')"
done
