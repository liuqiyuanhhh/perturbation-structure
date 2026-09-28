#!/bin/bash -l
#SBATCH --job-name=presage_collect
#SBATCH --output=logs/%x_%j.out
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#
# PRESAGE step 04 for one target: stack the five folds' test predictions and
# check that every perturbation is predicted exactly once.  Run after all
# folds of presage_train.sbatch.
#
#   PP_DATASET=VCC sbatch 4_prediction_benchmark/slurm/presage_collect.sbatch
set -euo pipefail
: "${PP_DATASET:?set PP_DATASET, e.g. PP_DATASET=VCC}"
source "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-.}}/4_prediction_benchmark/slurm/_env.sh"
activate presage

cd "${PRED}/presage_pipeline"
python 04_collect_predictions.py
