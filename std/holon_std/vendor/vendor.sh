#!/usr/bin/env bash
# gguf-py's type tables and dequantization (MIT), vendored from llama.cpp at the commit the workspace's own gguf-py
# comes from (docs/dev/v1-pip.md D7): std reads GGUF without a `gguf` dependency, which PyPI's gguf (no Q2_0) cannot
# be. Unmodified (no patch); gguf/ is a namespace package, so its relative imports work as upstream wrote them.
#
#   ./vendor.sh [llama.cpp checkout]   refresh gguf/: copy the files below from a checkout at $COMMIT (cloned when
#                                       omitted) and record their upstream SHA256SUMS
#   ./vendor.sh --check                verify gguf/ == upstream files (SHA256SUMS), offline, and that every stamp names
#                                       $COMMIT: the [provenance] of the units that use it (a re-vendor outdates their
#                                       locks) and the workspace's gguf source (the judge's oracle reads the same tables)
#
# A refresh to a new commit: change COMMIT, the three units' [provenance] commit and the root pyproject's gguf rev,
# run ./vendor.sh, then the GGUF tests and a propagation of the locks those units pin.
set -euo pipefail
COMMIT=27c7a769da1671c97b2e3f6cb88704a6e1e0d0cd
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
FILES="LICENSE constants.py quants.py lazy.py"
STAMPED="std/holon_std/weights/gguf/plugin.toml std/holon_std/mixers/qsa/plugin.toml std/holon_std/moe/plugin.toml"

if [ "${1:-}" = "--check" ]; then
  (cd "$HERE/gguf" && sha256sum --quiet -c "$HERE/SHA256SUMS")
  diff <(cd "$HERE/gguf" && find . -type f ! -path '*/__pycache__/*' | sed 's|^\./||' | sort) \
       <(awk '{print $2}' "$HERE/SHA256SUMS" | sort)
  for m in $STAMPED; do
    grep -q "^commit = \"$COMMIT\"" "$ROOT/$m" || { echo "FAIL $m: [provenance] does not name gguf-py's $COMMIT" >&2; exit 1; }
  done
  grep -q "llama.cpp\", rev = \"$COMMIT\", subdirectory = \"gguf-py\"" "$ROOT/pyproject.toml" ||
    { echo "FAIL pyproject.toml: the workspace's gguf source is not at $COMMIT" >&2; exit 1; }
  echo "ok   std/holon_std/vendor/gguf/ = llama.cpp ${COMMIT:0:8} gguf-py subset; stamps and the gguf source agree"
  exit 0
fi

SRC=${1:-}
if [ -z "$SRC" ]; then
  SRC=$(mktemp -d); trap 'rm -rf "$SRC"' EXIT
  git clone -q --filter=blob:none https://github.com/ggml-org/llama.cpp.git "$SRC"
  git -C "$SRC" checkout -q "$COMMIT"
fi
test "$(git -C "$SRC" rev-parse HEAD)" = "$COMMIT" || { echo "$SRC is not at $COMMIT" >&2; exit 1; }
rm -rf "$HERE/gguf"
for f in $FILES; do
  up=gguf-py/$([ "$f" = LICENSE ] && echo LICENSE || echo "gguf/$f")
  git -C "$SRC" diff --quiet HEAD -- "$up" || { echo "$SRC/$up has local changes" >&2; exit 1; }
  install -D -m 644 "$SRC/$up" "$HERE/gguf/$f"
done
(cd "$HERE/gguf" && sha256sum $FILES) > "$HERE/SHA256SUMS"
echo "vendored $(echo $FILES | wc -w) files from llama.cpp ${COMMIT:0:8} (gguf-py)"
