#!/usr/bin/env bash
# Run a build_wheels.py step inside an image, from `git archive <ref>` unpacked at /holon (docs/dev/v1-pip.md D4, D10):
# never a bind-mounted checkout, so what a binary embeds names /holon. <out> is copied in as /out and back out after
# (no bind mount: the daemon need not reach <out>), so a step reads what earlier steps wrote there.
#
#   scripts/release/in_image.sh overlay|reference <ref> <out> <build_wheels.py args, with /out for paths>
set -euo pipefail
kind=$1 ref=$2 out=$3; shift 3
here=$(cd "$(dirname "$0")" && pwd)
root=$(git -C "$here" rev-parse --show-toplevel)         # git archive in a subdirectory archives only it
image=holon-$kind:$(sha256sum "$here/$kind.Dockerfile" | cut -c1-12)
docker image inspect "$image" > /dev/null 2>&1 || docker build -q -t "$image" -f "$here/$kind.Dockerfile" "$here"
mkdir -p "$out"
cid=$(docker create -i -e HOLON_COMMIT="$(git -C "$root" rev-parse "$ref")" \
  -e HOLON_IMAGE="$(sed -n 's/^FROM //p' "$here/$kind.Dockerfile")" -e MAX_JOBS="${MAX_JOBS:-3}" "$image" \
  bash -c 'mkdir /holon && tar -x -C /holon && cd /holon && uv python install -q 3.12 &&
           uv run --no-project --python 3.12 --with packaging==26.3 python scripts/release/build_wheels.py "$@"' -- "$@")
trap 'docker rm -f "$cid" > /dev/null' EXIT
docker cp -q "$out/." "$cid:/out"
git -C "$root" archive --format=tar "$ref" | docker start -a -i "$cid"
docker cp -q "$cid:/out/." "$out"
