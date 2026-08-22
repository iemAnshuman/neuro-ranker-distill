#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${WORK:-}" ]]; then
    echo "WORK is not set; run this script on Rostam" >&2
    exit 1
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
venv_path=${NEURORANKER_VENV:-"$WORK/venvs/neuro-ranker-0.2"}
export HF_HOME=${HF_HOME:-"$WORK/.cache/neuro-ranker/huggingface"}
export TORCH_HOME=${TORCH_HOME:-"$WORK/.cache/neuro-ranker/torch"}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-"$WORK/.cache/neuro-ranker/pip"}

module reset
module load python/3.12.3
mkdir -p "$(dirname "$venv_path")" "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR"

if [[ ! -x "$venv_path/bin/python" ]]; then
    python3 -m venv "$venv_path"
fi

"$venv_path/bin/python" -m pip install --require-hashes \
    -r "$repo_root/requirements-runtime.lock"
"$venv_path/bin/python" -m pip install --no-build-isolation --no-deps -e "$repo_root"

echo "Environment ready: $venv_path"
