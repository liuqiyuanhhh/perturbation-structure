#!/bin/bash -l
#SBATCH --job-name=presage_train
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --array=1-5
#
# PRESAGE (+Perturb-seq), one CV fold per array task, on CPU.
#
# Fold 1 writes the dense <target>_processed.h5ad and the knowledge-tensor cache
# that the other folds reuse, so run it alone first:
#
#   PP_DATASET=VCC sbatch --array=1   4_prediction_benchmark/slurm/presage_train.sbatch
#   PP_DATASET=VCC sbatch --array=2-5 4_prediction_benchmark/slurm/presage_train.sbatch
#   PP_DATASET=VCC sbatch 4_prediction_benchmark/slurm/presage_collect.sbatch
set -euo pipefail
: "${PP_DATASET:?set PP_DATASET, e.g. PP_DATASET=VCC}"
source "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-.}}/4_prediction_benchmark/slurm/_env.sh"
activate presage

cd "${PRED}/presage_pipeline"
python 03_train.py --fold "${SLURM_ARRAY_TASK_ID}"
