#!/usr/bin/env bash
# QC, primary outcomes, L1 perturbation strength, strong perturbations,
# prediction QC tables and the effect dict.
#
# Usage: bash 1_preprocessing/run.sh [--dry-run]
#
# Not included (PLACEHOLDER folders; their outputs are inputs in data/):
#   crispyx_pseudobulk_de    moments and Wilcoxon DE (data/pseudobulk, data/de, data/sc)
#   vcc_subsampling          VCC-subsampled moments and DE
#   split_halves             split-half moments (data/split)
# The DepMap pan-essentiality score is computed in 2_global_trend.
# No figure notebook.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../utils/run.sh" "$@"

section "QC, primary outcomes, L1 strength, strong perturbations"
threads 4
py "$HERE/build_qc_primary_features.py" \
  --primary-expression-cutoff 0.08 --target-control-cutoff 0.08 \
  --minimum-inactivation-efficiency 0.10 --n-feature-genes 2000
py "$HERE/compute_perturbation_strength.py" --adjusted-p-cutoff 0.05 \
  --default-maximum 2000 --vcc-maximum 150

section "prediction QC tables"
M="$DATA/pseudobulk"
PREDICTION_QC="$RESULTS/1_preprocessing/filter_result_bulk"
# the 12 screens; VCC-subsampled uses the QC of VCC
QC_DATASETS=(--dataset Feng-GW "$M/Feng-gwsf_moments.h5ad" "$M/Feng-gwsnf_moments.h5ad")
for name in Feng-ts Huang-HCT116 Huang-HEK293T Nadig-HEPG2 Nadig-JURKAT \
  Nourreddine-GW-ipsc Pan-GW-hESC Replogle-E-k562 Replogle-E-rpe1 \
  Replogle-GW-k562 VCC; do
  QC_DATASETS+=(--dataset "$name" "$M/${name}_moments.h5ad")
done
run "$PYTHON" -m utils.qc "${QC_DATASETS[@]}" \
  --primary-expression-cutoff 0.08 --target-control-cutoff 0.08 \
  --minimum-inactivation-efficiency 0.10 --output-dir "$PREDICTION_QC"

section "effect dict"
py "$HERE/build_effect_dict.py" --qc-dir "$PREDICTION_QC"
