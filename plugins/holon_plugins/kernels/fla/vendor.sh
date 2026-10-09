#!/usr/bin/env bash
# vLLM's copy of flash-linear-attention's Gated DeltaNet Triton kernels (vLLM: Apache-2.0; flash-linear-attention:
# MIT), vendored at a pinned commit (M0.3 design D3; the subset spike S3 measured).
#
#   ./vendor.sh [vLLM checkout]   refresh vendor/: copy the files below from a checkout at $COMMIT (cloned when
#                                  omitted), record their upstream SHA256SUMS, apply vendor.patch
#   ./vendor.sh --check           verify vendor/ == upstream files (SHA256SUMS) + vendor.patch, offline
#
# A refresh to a new commit: change COMMIT (and [provenance] in plugin.toml), run ./vendor.sh, fix vendor.patch until
# it applies, then L0 (plugins/tests/test_gdn_kernels.py) on every SM.
set -euo pipefail
COMMIT=6dd813e0b7ca8314fed28c13a73cfb5fc5310163
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=vllm/third_party/flash_linear_attention
FILES="LICENSE"                                      # vendor/<name> <- $SRC/ops/<name> (LICENSE: $SRC/LICENSE)
for f in __init__ chunk chunk_delta_h chunk_o chunk_scaled_dot_kkt cumsum fused_sigmoid_gating index l2norm op \
         solve_tril utils wy_fast; do
  FILES="$FILES $f.py"
done

if [ "${1:-}" = "--check" ]; then
  tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
  cp -r "$HERE/vendor" "$tmp/vendor"
  find "$tmp/vendor" -name __pycache__ -prune -exec rm -rf {} +
  patch -s -R -p1 -d "$tmp/vendor" < "$HERE/vendor.patch"
  (cd "$tmp/vendor" && sha256sum --quiet -c "$HERE/SHA256SUMS")
  diff <(cd "$tmp/vendor" && find . -type f | sed 's|^\./||' | sort) <(awk '{print $2}' "$HERE/SHA256SUMS" | sort)
  echo "ok   vendor/ = vLLM ${COMMIT:0:8} flash-linear-attention + vendor.patch"
  exit 0
fi

REPO=${1:-}
if [ -z "$REPO" ]; then
  REPO=$(mktemp -d); trap 'rm -rf "$REPO"' EXIT
  git clone -q --filter=blob:none --no-checkout https://github.com/vllm-project/vllm.git "$REPO"
fi
rm -rf "$HERE/vendor" && mkdir -p "$HERE/vendor"
for f in $FILES; do
  up="$SRC/ops/$f"; [ "$f" = LICENSE ] && up="$SRC/LICENSE"
  git -C "$REPO" show "$COMMIT:$up" > "$HERE/vendor/$f"
done
(cd "$HERE/vendor" && sha256sum $FILES) > "$HERE/SHA256SUMS"
patch -s -p1 -d "$HERE/vendor" < "$HERE/vendor.patch"
echo "vendored $(echo $FILES | wc -w) files from vLLM ${COMMIT:0:8}"
