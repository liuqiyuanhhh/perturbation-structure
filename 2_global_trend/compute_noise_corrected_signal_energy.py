#!/usr/bin/env python3
"""Noise-corrected signal energy 100 lambda_max(K_hat) / tr(K_hat), with
K_hat = Y Y^T - diag(rowSums(S^2)) on the L1-selected perturbations x primary genes.
Reads MOMENTS_DIR, QC_GENE_PANELS and L1_SELECTION; writes FIRST_FACTOR (Fig 1b left)."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils.effects import load_effects, load_selections
from utils.paths import (
    DATASET_COMPONENTS,
    FIRST_FACTOR,
    L1_SELECTION,
    MOMENTS_DIR,
    QC_GENE_PANELS,
    moments_file,
    paper_name,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pseudobulk-dir", type=Path, default=MOMENTS_DIR)
    parser.add_argument("--gene-selection-file", type=Path,
                        default=QC_GENE_PANELS / "outcome_gene_scores_and_selections.csv.gz")
    parser.add_argument("--quality-filter-file", type=Path,
                        default=QC_GENE_PANELS / "perturbation_quality_filter.csv.gz")
    parser.add_argument("--l1-membership-file", type=Path,
                        default=L1_SELECTION / "L1_strong_perturbation_membership.csv.gz")
    parser.add_argument("--output-dir", type=Path, default=FIRST_FACTOR)
    return parser.parse_args()


def signal_energy(y, s):
    """100 lambda_max(K_hat) / tr(K_hat) over the rows with every SE, and their number."""
    complete = np.isfinite(s).all(axis=1)
    y, s = y[complete].astype(np.float64), s[complete].astype(np.float64)
    k_hat = y @ y.T - np.diag(np.sum(s**2, axis=1))
    return 100 * np.linalg.eigvalsh(k_hat)[-1] / np.trace(k_hat), int(complete.sum())


def main():
    args = parse_args()
    selections = load_selections(args.quality_filter_file, args.gene_selection_file,
                                 args.l1_membership_file)
    records = []
    for dataset, components in DATASET_COMPONENTS.items():
        paths = [args.pseudobulk_dir / moments_file(c) for c in components]
        chosen = selections[dataset]
        _, _, y, s = load_effects(paths, chosen["L1"], chosen["primary"], se=True)
        percent, n = signal_energy(y, s)
        records.append({"dataset": dataset, "display_dataset": paper_name(dataset),
                        "perturbation_set": "L1_selected", "n_perturbations": n,
                        "se_adjusted_rank1_energy_percent": percent})
        print(f"{paper_name(dataset)}: {percent:.2f}% ({n:,} perturbations)", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(
        args.output_dir / "first_svd_global_trend_strength_comparison_se_adjusted.csv",
        index=False, float_format="%.12g")


if __name__ == "__main__":
    main()
