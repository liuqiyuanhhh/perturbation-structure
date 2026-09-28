#!/bin/bash -l
#SBATCH --job-name=evaluate
#SBATCH --output=logs/%x_%j.out
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#
# Weighted, the linear models, the training mean and the five metrics for one target (or TARGET=all).
#
#   TARGET=VCC sbatch 4_prediction_benchmark/slurm/evaluate.sbatch
set -euo pipefail
: "${TARGET:?set TARGET, e.g. TARGET=VCC}"
source "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-.}}/4_prediction_benchmark/slurm/_env.sh"
activate gears

python "${PRED}/evaluation/evaluate.py" --target "${TARGET}"
