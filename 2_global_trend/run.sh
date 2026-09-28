#!/usr/bin/env bash
# Leading global trend: leading-factor fits, noise-corrected signal energy and
# the pan-essentiality score.
#
# Usage: bash 2_global_trend/run.sh
#
# Needs 1_preprocessing and the DepMap gene effect
# (data/essential/CRISPRGeneEffect.csv).
# Figure notebook: global_trend_figures.ipynb (Fig 1b left, 1b right, S1d).

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../utils/run.sh" "$@"

threads 4

section "leading factor, all-QC and L1-selected (Fig 1b right, S1d)"
py "$HERE/fit_first_factor.py"

section "noise-corrected signal energy (Fig 1b left)"
py "$HERE/compute_noise_corrected_signal_energy.py"

section "pan-essentiality score (Fig 1b right)"
py "$HERE/compute_pan_essentiality_score.py"
