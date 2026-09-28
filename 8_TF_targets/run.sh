#!/usr/bin/env bash
# Residual magnitudes of the SCENIC+ cell-line-active targets of TF perturbations.
#
# Usage: bash 8_TF_targets/run.sh
#
# Needs the all-QC factor of 2_global_trend and 5_response_hierarchy.  Figure notebook:
# TF_figures.ipynb (Fig 2f, S5d).

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../utils/run.sh" "$@"

section "SCENIC+ active targets and TF target magnitudes"
py "$HERE/compare_active_target_residual_magnitude.py"
