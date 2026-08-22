#!/usr/bin/env bash
#SBATCH --job-name=neurorank-smoke
#SBATCH --partition=cuda-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
if [[ "${MSMARCO_TERMS_ACCEPTED:-0}" != "1" ]]; then
    echo "Review https://microsoft.github.io/msmarco/ and submit with MSMARCO_TERMS_ACCEPTED=1" >&2
    exit 2
fi

module reset
module load python/3.12.3
repo_root=${SLURM_SUBMIT_DIR:?submit from the repository root}
venv_path=${NEURORANKER_VENV:-"$WORK/venvs/neuro-ranker-0.2"}
run_dir="$WORK/neuro-ranker-runs/smoke-$SLURM_JOB_ID"
export HF_HOME=${HF_HOME:-"$WORK/.cache/neuro-ranker/huggingface"}
export TORCH_HOME=${TORCH_HOME:-"$WORK/.cache/neuro-ranker/torch"}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false

source "$venv_path/bin/activate"
cd "$repo_root"
neuro-ranker run \
    --config configs/smoke.json \
    --work-dir "$run_dir" \
    --repo-root "$repo_root" \
    --device cuda \
    --accept-msmarco-terms
echo "$run_dir"
