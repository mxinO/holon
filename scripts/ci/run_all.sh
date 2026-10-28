#!/usr/bin/env bash
# Run all static CI checks. Usage: scripts/ci/run_all.sh [base-ref]
set -uo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"   # the checks import this checkout's packages, never an installed copy
export PYTHONPATH="$root/core:$root/std:$root/plugins:$root/judge:$root/testing:$root/tools${PYTHONPATH:+:$PYTHONPATH}"
cd "$root/scripts/ci"
rc=0
for c in check_core_denylist.py check_core_loc.py check_no_env.py check_torch_cuda.py check_private.py check_locks.py check_provenance.py \
         check_manifest_deps.py check_docs.py check_version.py; do
  python3 "$c" || rc=1
done
(cd ../.. && ruff check core std plugins judge tools testing scripts bench) || rc=1
bash ../../plugins/holon_plugins/kernels/ggml_cuda/vendor.sh --check || rc=1   # vendored llama.cpp = upstream + patch
bash ../../plugins/holon_plugins/kernels/fla/vendor.sh --check || rc=1         # vendored vLLM fla = upstream + patch
bash ../../plugins/holon_plugins/kernels/ggml_cpu/vendor.sh --check || rc=1    # vendored ggml-cpu subset = upstream
bash ../../plugins/tests/fixtures/jsts.sh --check || rc=1                       # JSON-Schema-Test-Suite = upstream
python3 check_abi.py --base "${1:-origin/main}" || rc=1                           # ABI listings, bumps vs base
python3 check_commit_messages.py "${1:-origin/main}" || rc=1                    # commit messages vs base: no private names
(cd ../.. && uv lock --check -q) || rc=1                                         # uv.lock matches the pyprojects
bash check_core_diff.sh "${1:-origin/main}" || true
python3 check_lock_impact.py "${1:-origin/main}" || true                         # warns: the locks a merge outdates
if command -v lint-imports >/dev/null; then (cd ../.. && lint-imports) || rc=1; else echo "skip import contracts (import-linter not installed)"; fi
exit $rc
