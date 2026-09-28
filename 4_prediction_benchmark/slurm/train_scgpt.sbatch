#!/bin/bash -l
#SBATCH --job-name=train_scgpt
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=500G
#SBATCH --array=1-5
#
# scGPT fine-tuned from the whole-human checkpoint, under 5-fold CV
# (array index = fold).  24 GB of GPU memory is enough.
#
#   BUILD=VCC sbatch 4_prediction_benchmark/slurm/train_scgpt.sbatch
#   BUILD='Huang-HCT116-third-f{fold}' sbatch 4_prediction_benchmark/slurm/train_scgpt.sbatch
set -euo pipefail
: "${BUILD:?set BUILD, e.g. BUILD=VCC}"
source "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-.}}/4_prediction_benchmark/slurm/_env.sh"
activate scgpt

FOLD="${SLURM_ARRAY_TASK_ID}"
NAME="${BUILD//\{fold\}/${FOLD}}"
python "${PRED}/gears_scgpt/run_scgpt.py" \
    --dataset "${NAME}-cv5" --seed "${FOLD}" \
    --epochs 20 --finetune --train
