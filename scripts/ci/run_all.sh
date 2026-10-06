#!/usr/bin/env bash
# Run all static CI checks. Usage: scripts/ci/run_all.sh [base-ref]
set -uo pipefail
cd "$(dirname "$0")"
rc=0
for c in check_core_denylist.py check_core_loc.py check_no_env.py check_torch_cuda.py; do
  python3 "$c" || rc=1
done
(cd ../.. && ruff check core std plugins judge scripts) || rc=1
bash check_core_diff.sh "${1:-origin/main}" || true
if command -v lint-imports >/dev/null; then (cd ../.. && lint-imports) || rc=1; else echo "skip import contracts (import-linter not installed)"; fi
exit $rc
