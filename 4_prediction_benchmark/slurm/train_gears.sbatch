#!/bin/bash -l
#SBATCH --job-name=train_gears
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --time=50:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=500G
#SBATCH --array=1-5
#
# GEARS under 5-fold CV (array index = fold, not a random seed).
#
#   BUILD=VCC sbatch 4_prediction_benchmark/slurm/train_gears.sbatch
#   BUILD='Huang-HCT116-third-f{fold}' sbatch 4_prediction_benchmark/slurm/train_gears.sbatch
#
# "{fold}" in BUILD is replaced by the array index (per-fold builds).
set -euo pipefail
: "${BUILD:?set BUILD, e.g. BUILD=VCC}"
source "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-.}}/4_prediction_benchmark/slurm/_env.sh"
activate gears

FOLD="${SLURM_ARRAY_TASK_ID}"
NAME="${BUILD//\{fold\}/${FOLD}}"
python "${PRED}/gears_scgpt/run_gears.py" \
    --dataset "${NAME}-cv5" --seed "${FOLD}" \
    --epochs 50 --hiddendim 64 --batchsize 32 --lr 1e-3
