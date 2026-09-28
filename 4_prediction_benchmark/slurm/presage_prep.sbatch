#!/bin/bash -l
#SBATCH --job-name=presage_prep
#SBATCH --output=logs/%x_%j.out
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#
# PRESAGE steps 00-02 for one target: link the GEARS build, write folds and
# DEGs, build the Perturb-seq knowledge sources, write the source list.
#
#   PP_DATASET=VCC sbatch 4_prediction_benchmark/slurm/presage_prep.sbatch
set -euo pipefail
: "${PP_DATASET:?set PP_DATASET, e.g. PP_DATASET=VCC}"
source "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-.}}/4_prediction_benchmark/slurm/_env.sh"
activate presage

cd "${PRED}/presage_pipeline"
python 00_prep_dataset.py
python 01_build_pert_sources.py
python 02_make_source_list.py
