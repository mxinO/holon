#!/usr/bin/env bash
# Invariant 9 (informational in v0): report changes under core/ relative to a base ref.
set -euo pipefail
base="${1:-origin/main}"
git rev-parse --verify -q "$base" >/dev/null || { echo "skip core-diff: base '$base' not found"; exit 0; }
changed=$(git diff --name-only "$base"...HEAD -- ':/core/' || true)   # (from the top: run_all.sh runs in scripts/ci)
if [ -n "$changed" ]; then echo "info core-diff vs $base:"; echo "$changed" | sed 's/^/  /'; else echo "ok   core-diff (empty)"; fi
