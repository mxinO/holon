#!/usr/bin/env bash
# ggml-cpu's quantization and dot-product sources (MIT), vendored at kernels.ggml_cuda's pin: the activation
# quantizers and `vec_dot`s of the CPU experts (docs/v1-offload.md D7), unmodified (no patch).
#
#   ./vendor.sh [llama.cpp checkout]   refresh vendor/: copy the files below from a checkout at $COMMIT (cloned when
#                                       omitted) and record their upstream SHA256SUMS
#   ./vendor.sh --check                verify vendor/ == upstream files (SHA256SUMS), offline
#
# A refresh to a new commit: change COMMIT (and [provenance] in plugin.toml, kept with kernels.ggml_cuda's), run
# ./vendor.sh, fix csrc/shim.c and csrc/cpu_experts.cpp's type table until the link test passes, then L0
# (plugins/tests/test_ggml_cpu.py) on every ISA tier.
set -euo pipefail
COMMIT=cbe28bb3cb0758f0c566a9a3561766821aeb0748
HERE=$(cd "$(dirname "$0")" && pwd)
FILES="LICENSE
ggml/include/ggml.h ggml/include/ggml-cpu.h ggml/include/ggml-backend.h ggml/include/ggml-alloc.h ggml/include/gguf.h
ggml/src/ggml-common.h ggml/src/ggml-impl.h ggml/src/ggml-quants.h ggml/src/ggml-quants.c
ggml/src/ggml-cpu/ggml-cpu-impl.h ggml/src/ggml-cpu/simd-mappings.h ggml/src/ggml-cpu/arch-fallback.h
ggml/src/ggml-cpu/quants.h ggml/src/ggml-cpu/quants.c ggml/src/ggml-cpu/arch/x86/quants.c"

if [ "${1:-}" = "--check" ]; then
  (cd "$HERE/vendor" && sha256sum --quiet -c "$HERE/SHA256SUMS")
  diff <(cd "$HERE/vendor" && find . -type f | sed 's|^\./||' | sort) <(awk '{print $2}' "$HERE/SHA256SUMS" | sort)
  echo "ok   vendor/ = llama.cpp ${COMMIT:0:8} (ggml-cpu subset)"
  exit 0
fi

SRC=${1:-}
if [ -z "$SRC" ]; then
  SRC=$(mktemp -d); trap 'rm -rf "$SRC"' EXIT
  git clone -q --filter=blob:none https://github.com/ggml-org/llama.cpp.git "$SRC"
  git -C "$SRC" checkout -q "$COMMIT"
fi
test "$(git -C "$SRC" rev-parse HEAD)" = "$COMMIT" || { echo "$SRC is not at $COMMIT" >&2; exit 1; }
rm -rf "$HERE/vendor"
for f in $FILES; do
  git -C "$SRC" diff --quiet HEAD -- "$f" || { echo "$SRC/$f has local changes" >&2; exit 1; }
  install -D -m 644 "$SRC/$f" "$HERE/vendor/$f"
done
(cd "$HERE/vendor" && sha256sum $FILES) > "$HERE/SHA256SUMS"
echo "vendored $(echo $FILES | wc -w) files from llama.cpp ${COMMIT:0:8}"
