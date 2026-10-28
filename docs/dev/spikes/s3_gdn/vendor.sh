#!/usr/bin/env bash
# Recreate third_party/fla_ops (gitignored): copy the GDN subset of vLLM's flash-linear-attention copy at the
# pinned commit and apply vendor.patch (vllm.* imports and env switches stripped, autotune pinned, v_head_tiled).
# Usage: spikes/s3_gdn/vendor.sh /path/to/vllm-checkout
set -euo pipefail
VLLM=${1:?path to a vLLM checkout}
COMMIT=6dd813e0b7ca8314fed28c13a73cfb5fc5310163
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=vllm/third_party/flash_linear_attention
FILES="__init__ chunk chunk_delta_h chunk_o chunk_scaled_dot_kkt cumsum fused_recurrent fused_sigmoid_gating index l2norm op solve_tril utils wy_fast"
DST="$HERE/third_party/fla_ops"
rm -rf "$DST" && mkdir -p "$DST"
git -C "$VLLM" show "$COMMIT:$SRC/LICENSE" > "$DST/LICENSE"
for f in $FILES; do git -C "$VLLM" show "$COMMIT:$SRC/ops/$f.py" > "$DST/$f.py"; done
patch -s -d "$DST" -p1 < "$HERE/vendor.patch"
echo "vendored $(cat "$DST"/*.py | wc -l) lines into $DST"
