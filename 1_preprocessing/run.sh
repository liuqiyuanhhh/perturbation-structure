#!/usr/bin/env bash
# QC, primary outcomes, L1 perturbation strength, strong perturbations,
# prediction QC tables and the effect dict.
#
# Usage: bash 1_preprocessing/run.sh
#
# Reads the processed data in data/ (from the data deposit, or rebuilt by the
# folders below; each has a README):
#   crispyx_pseudobulk_de    single cells, moments and Wilcoxon DE (data/sc, data/pseudobulk, data/de)
#   vcc_subsampling          VCC-subsampled, for S1b only
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
# the 12 screens
QC_DATASETS=(--dataset Feng-GW "$M/effect_Feng-gwsf_moments.h5ad" "$M/effect_Feng-gwsnf_moments.h5ad")
for name in Feng-ts Huang-HCT116 Huang-HEK293T Nadig-HEPG2 Nadig-JURKAT \
  Nourreddine-GW-ipsc Pan-GW-hESC Replogle-E-k562 Replogle-E-rpe1 \
  Replogle-GW-k562 VCC; do
  QC_DATASETS+=(--dataset "$name" "$M/effect_${name}_moments.h5ad")
done
run "$PYTHON" -m utils.qc "${QC_DATASETS[@]}" \
  --primary-expression-cutoff 0.08 --target-control-cutoff 0.08 \
  --minimum-inactivation-efficiency 0.10 --output-dir "$PREDICTION_QC"

section "effect dict"
py "$HERE/build_effect_dict.py" --qc-dir "$PREDICTION_QC"
