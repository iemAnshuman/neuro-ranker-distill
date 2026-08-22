#!/usr/bin/env bash
#SBATCH --job-name=neurorank-fixture
#SBATCH --partition=cuda-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
module purge
module load python/3.12.3

repo_root=${SLURM_SUBMIT_DIR:?submit from the repository root}
venv_path=${NEURORANKER_VENV:-"$WORK/venvs/neuro-ranker-0.2"}
run_dir="$WORK/neuro-ranker-runs/fixture-$SLURM_JOB_ID"
export HF_HOME=${HF_HOME:-"$WORK/.cache/neuro-ranker/huggingface"}
export TORCH_HOME=${TORCH_HOME:-"$WORK/.cache/neuro-ranker/torch"}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=42
export TOKENIZERS_PARALLELISM=false

source "$venv_path/bin/activate"
cd "$repo_root"
neuro-ranker run \
    --config configs/fixture.json \
    --work-dir "$run_dir" \
    --repo-root "$repo_root" \
    --device cuda
echo "$run_dir"
