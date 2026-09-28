#!/usr/bin/env bash
# Discovery matrices, nestedness, response breadth.
#
# Usage: bash 5_response_hierarchy/run.sh
#
# Needs 1_preprocessing and the all-QC factor of 2_global_trend.  Figure
# notebooks: nestedness_figures.ipynb (Fig 2a, S3a, S3b) and DES_figures.ipynb
# (Fig 2b, 2d, S3c, S4, S5b).

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../utils/run.sh" "$@"

section "discovery matrices and nestedness"
nest=(--n-splits 100 --train-fraction 0.70 --master-seed 20260824
  --minimum-rejections 20 --minimum-rows-per-split 60 --bins 20)
threads 2
# --dataset-index: 12 datasets for the matrices, 11 without Feng-GW for the nestedness
for i in $(seq 0 11); do
  py "$HERE/build_discovery_matrices.py" --mode analyze --dataset-index "$i" --alpha 0.05
done
py "$HERE/build_discovery_matrices.py" --mode aggregate --alpha 0.05
for i in $(seq 0 10); do
  py "$HERE/compute_matched_nestedness.py" --mode analyze --dataset-index "$i" "${nest[@]}"
done
py "$HERE/compute_matched_nestedness.py" --mode aggregate "${nest[@]}"

section "DES scorer (placeholder; external inputs)"
# PLACEHOLDER 5_response_hierarchy/des_scoring: per-perturbation DES scorer and threshold
#   sweep on the prediction pickles; writes des_per_pert.csv and
#   des_threshold_per_pert.csv, which DES_figures.ipynb summarizes.
skip "the DES scorer is a placeholder"

section "response breadth (prediction pickles)"
# Reads the saved prediction pickles (data/prediction_result; 4_prediction_benchmark);
# DES_figures.ipynb pools its per-dataset tables (S4).
threads 2
for i in $(seq 0 10); do
  py "$HERE/compute_breadth_stratified_des.py" --dataset-index "$i"
done
