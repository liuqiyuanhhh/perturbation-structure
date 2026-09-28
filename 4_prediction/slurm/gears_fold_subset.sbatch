#!/bin/bash -l
#SBATCH --job-name=gears_fold_subset
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=350G
#SBATCH --array=1-5
#
# Per-fold reduced builds of a large screen (array index = fold).
#
#   PARENT=Huang-HCT116 sbatch 4_prediction/slurm/gears_fold_subset.sbatch
#
# Run after gears_prep.sbatch has built PARENT.  Huang-HEK293T needs ~550G.
set -euo pipefail
: "${PARENT:?set PARENT, e.g. PARENT=Huang-HCT116}"
source "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-.}}/4_prediction/slurm/_env.sh"
activate gears

python "${PRED}/gears_scgpt/prepare_data.py" --parent "${PARENT}" --fold "${SLURM_ARRAY_TASK_ID}"
