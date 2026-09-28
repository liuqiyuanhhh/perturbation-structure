#!/bin/bash -l
#SBATCH --job-name=gears_prep
#SBATCH --output=logs/%x_%j.out
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=500G
#
# Build one GEARS dataset from its batch-corrected h5ad, then cut its 5 folds.
#
#   DATASET=VCC sbatch 4_prediction_benchmark/slurm/gears_prep.sbatch
#
# DATASET is a key of DATASETS in gears_scgpt/prepare_data.py (not a -third-f*
# build; those come from gears_fold_subset.sbatch).  Memory per dataset: README.md.
set -euo pipefail
: "${DATASET:?set DATASET, e.g. DATASET=VCC}"
source "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-.}}/4_prediction_benchmark/slurm/_env.sh"
activate gears

python "${PRED}/gears_scgpt/prepare_data.py" --dataset "${DATASET}"
