#!/usr/bin/env bash
# Per-perturbation noise-corrected cosine audits of total and residual effects
# between datasets.
#
# Usage: bash 3_cross_dataset_similarity/run.sh
#
# Needs 1_preprocessing only: the residual mode refits the leading factor
# itself.  Figure notebook: similarity_figures.ipynb (Fig 1a, 1c, S1a-S1c),
# which computes the plotted values from the audits and the L1 strength scores.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../utils/run.sh" "$@"

threads 4

section "noise-corrected cosine audits (Fig 1a, 1c, S1a, S1b)"
py "$HERE/compute_noise_corrected_cosine.py" --effect-mode total
py "$HERE/compute_noise_corrected_cosine.py" --effect-mode first_svd_residual
