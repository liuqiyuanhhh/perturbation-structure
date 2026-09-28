#!/usr/bin/env bash
# Cross-split rank-1 energy of the residual effects.
#
# Usage: bash 6_magnitude_structure/run.sh
#
# Needs 1_preprocessing and the split-half moments (data/split).
# Figure notebook: magnitude_structure_figures.ipynb (Fig 2c, S5a).

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../utils/run.sh" "$@"

# Fig 2c (first three) and S5a (last seven), as in the paper jobs.
CROSS_SPLIT_DATASETS=(
  Replogle-E-k562 Replogle-E-rpe1 Pan-GW-hESC
  Replogle-GW-k562 Nadig-HEPG2 Nadig-JURKAT Huang-HCT116 Huang-HEK293T VCC Feng-ts
)

section "cross-split rank-1 energy"
threads 4
for dataset in "${CROSS_SPLIT_DATASETS[@]}"; do
  py "$HERE/analyze_cross_split_dataset.py" --dataset "$dataset"
done
