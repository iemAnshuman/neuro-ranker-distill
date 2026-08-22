#!/usr/bin/env bash

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

if [[ -n "$(git status --porcelain=v1)" ]]; then
    echo "refusing to build from a dirty Git worktree" >&2
    exit 2
fi

revision=$(git rev-parse HEAD)
output=${1:-"neuro-ranker-${revision:0:8}.sif"}
exec apptainer build --fakeroot \
    --build-arg "SOURCE_REVISION=$revision" \
    "$output" containers/neuro-ranker.def
