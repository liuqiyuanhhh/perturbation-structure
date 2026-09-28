#!/usr/bin/env python3
"""Noise-corrected cosine sum_g E_d E_d' / sqrt(Q_d Q_d'), Q_d = sum_g (E_d^2 - SE_d^2), of
each perturbation shared by two datasets, on total effects or first-factor residuals.
Reads MOMENTS_DIR, QC_GENE_PANELS and L1_SELECTION; writes SIMILARITY_TOTAL_AUDIT or
SIMILARITY_RESIDUAL_AUDIT (--effect-mode).  Fig 1a, 1c, S1a, S1b."""

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from utils.effects import fit_and_remove_first_svd_factor, load_effects, load_selections
from utils.paths import (
    L1_SELECTION,
    MOMENTS_DIR,
    QC_GENE_PANELS,
    REGISTRY,
    SIMILARITY_RESIDUAL_AUDIT,
    SIMILARITY_TOTAL_AUDIT,
)

# analysis panel: (perturbations, outcome genes)
PANELS = {"A": ("qc", "primary"), "B": ("qc", "feature"), "C": ("L1", "primary"), "D": ("L1", "feature")}
OUTPUT_DIRS = {"total": SIMILARITY_TOTAL_AUDIT, "first_svd_residual": SIMILARITY_RESIDUAL_AUDIT}


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pseudobulk-dir", type=Path, default=MOMENTS_DIR)
    parser.add_argument(
        "--gene-selection-file", type=Path,
        default=QC_GENE_PANELS / "outcome_gene_scores_and_selections.csv.gz",
    )
    parser.add_argument(
        "--quality-filter-file", type=Path,
        default=QC_GENE_PANELS / "perturbation_quality_filter.csv.gz",
    )
    parser.add_argument(
        "--l1-membership-file", type=Path,
        default=L1_SELECTION / "L1_strong_perturbation_membership.csv.gz",
    )
    parser.add_argument("--effect-mode", choices=list(OUTPUT_DIRS), default="total")
    parser.add_argument("--output-dir", type=Path, help="default: set by --effect-mode")
    return parser.parse_args()


def load_dataset(label, chosen, args):
    """Effects and SEs on the primary genes, direct targets set to NaN."""
    paths = [args.pseudobulk_dir / f"{c}_moments.h5ad" for c in REGISTRY[label].components]
    perturbations, genes, effects, se = load_effects(paths, chosen["qc"], chosen["primary"], se=True)
    if args.effect_mode == "first_svd_residual":
        fit_and_remove_first_svd_factor(effects)  # in place
    column = {gene: j for j, gene in enumerate(genes)}
    for i, name in enumerate(perturbations):
        if name in column:
            effects[i, column[name]] = np.nan  # direct target left out
    return {
        "label": label, "perturbations": pd.Index(perturbations), "genes": pd.Index(genes),
        "effects": effects, "se": se,
        "qc": set(perturbations), "L1": chosen["L1"] & set(perturbations),
        "primary": chosen["primary"] & set(genes), "feature": chosen["feature"] & set(genes),
    }


def submatrices(dataset, perturbations, genes):
    index = np.ix_(dataset["perturbations"].get_indexer(perturbations), dataset["genes"].get_indexer(genes))
    return dataset["effects"][index], dataset["se"][index]


def cosines(x1, s1, x2, s2):
    """Raw cosine, Q and Q / sum(E^2) of both sides, and the noise-corrected cosine, per row."""
    keep = np.isfinite(x1) & np.isfinite(s1) & np.isfinite(x2) & np.isfinite(s2)
    x1, s1, x2, s2 = (np.where(keep, m, 0).astype(np.float64) for m in (x1, s1, x2, s2))
    energy1, energy2 = np.sum(x1 * x1, 1), np.sum(x2 * x2, 1)
    q1, q2 = energy1 - np.sum(s1 * s1, 1), energy2 - np.sum(s2 * s2, 1)
    dot = np.sum(x1 * x2, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            "raw_cosine_on_se_complete_outcomes": dot / np.sqrt(energy1 * energy2),
            "dataset_1_corrected_norm_squared": q1,
            "dataset_1_corrected_signal_fraction": np.where(energy1 > 0, q1 / energy1, np.nan),
            "dataset_2_corrected_norm_squared": q2,
            "dataset_2_corrected_signal_fraction": np.where(energy2 > 0, q2 / energy2, np.nan),
            "would_be_noise_corrected_cosine_unclipped":
                np.where((q1 > 0) & (q2 > 0), dot / np.sqrt(q1 * q2), np.nan),
        }


def main():
    args = parse_args()
    selections = load_selections(args.quality_filter_file, args.gene_selection_file,
                                 args.l1_membership_file)
    datasets = []
    for label in REGISTRY:  # the 12 screens and VCC-subsampled (S1b)
        print(f"Loading {label}")
        datasets.append(load_dataset(label, selections[label], args))

    frames = []
    for panel, (pool, outcomes) in PANELS.items():
        for left, right in combinations(datasets, 2):
            perturbations = sorted(left[pool] & right[pool])
            genes = sorted(left[outcomes] & right[outcomes])
            frames.append(pd.DataFrame({
                "panel": panel, "dataset_1": left["label"], "dataset_2": right["label"],
                "perturbation": perturbations,
                **cosines(*submatrices(left, perturbations, genes), *submatrices(right, perturbations, genes)),
            }))
    output_dir = args.output_dir or OUTPUT_DIRS[args.effect_mode]
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(
        output_dir / "pairwise_per_perturbation_corrected_norm_audit.csv.gz",
        index=False, compression="gzip", float_format="%.12g")


if __name__ == "__main__":
    main()
