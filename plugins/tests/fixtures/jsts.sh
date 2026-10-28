#!/usr/bin/env bash
# The JSON-Schema-Test-Suite (MIT) at a pinned commit: the tests of every draft (draft3 … draft2020-12), unmodified,
# for the grammar plugin's model-free oracle (plugins/tests/test_grammar.py; docs/dev/m05-grammar.md D9).
#
#   ./jsts.sh [checkout]   refresh jsts/ from a checkout at $COMMIT (cloned when omitted), record jsts.SHA256SUMS
#   ./jsts.sh --check      verify jsts/ == the upstream files (jsts.SHA256SUMS), offline
set -euo pipefail
COMMIT=8568bcc477cc3aaeed5bc397c9c8648e631317fc
HERE=$(cd "$(dirname "$0")" && pwd)
DRAFTS="draft3 draft4 draft6 draft7 draft2019-09 draft2020-12"

if [ "${1:-}" = "--check" ]; then
  (cd "$HERE/jsts" && sha256sum --quiet -c "$HERE/jsts.SHA256SUMS")
  diff <(cd "$HERE/jsts" && find . -type f | sed 's|^\./||' | sort) <(awk '{print $2}' "$HERE/jsts.SHA256SUMS" | sort)
  echo "ok   jsts/ = JSON-Schema-Test-Suite ${COMMIT:0:8}"
  exit 0
fi

REPO=${1:-}
if [ -z "$REPO" ]; then
  REPO=$(mktemp -d); trap 'rm -rf "$REPO"' EXIT
  git clone -q --filter=blob:none --no-checkout https://github.com/json-schema-org/JSON-Schema-Test-Suite.git "$REPO"
fi
rm -rf "$HERE/jsts" && mkdir -p "$HERE/jsts"
git -C "$REPO" show "$COMMIT:LICENSE" > "$HERE/jsts/LICENSE"
for d in $DRAFTS; do
  git -C "$REPO" ls-tree -r --name-only "$COMMIT" "tests/$d" | while read -r f; do
    mkdir -p "$HERE/jsts/$(dirname "${f#tests/}")"
    git -C "$REPO" show "$COMMIT:$f" > "$HERE/jsts/${f#tests/}"
  done
done
(cd "$HERE/jsts" && find . -type f | sed 's|^\./||' | sort | xargs sha256sum) > "$HERE/jsts.SHA256SUMS"
echo "fetched $(wc -l < "$HERE/jsts.SHA256SUMS") files from JSON-Schema-Test-Suite ${COMMIT:0:8}"
