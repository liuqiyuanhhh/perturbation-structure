#!/usr/bin/env python3
"""L1 strength sum_{g in D(p)} |tau(p, g)| / G_dp of each QC-passing perturbation, where
D(p) are the BH (0.05) discoveries among its G_dp eligible primary genes, and the 2,000
strongest per dataset (VCC: 150).  Reads MOMENTS_DIR, DE_DIR and QC_GENE_PANELS; writes
STRENGTH_SCORES and L1_SELECTION.  Fig 1a-c, S1a-d."""

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from utils.effects import axis_names, de_perturbations, load_quality_audit, unique_columns
from utils.paths import (
    DATASET_COMPONENTS,
    DE_DIR,
    L1_SELECTION,
    MOMENTS_DIR,
    QC_GENE_PANELS,
    STRENGTH_SCORES,
)
from utils.qc import load_primary_outcome_lists
from utils.stats import bh_rejection_rows


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pseudobulk-dir", type=Path, default=MOMENTS_DIR)
    parser.add_argument("--de-dir", type=Path, default=DE_DIR)
    parser.add_argument("--selection-file", type=Path,
                        default=QC_GENE_PANELS / "outcome_gene_scores_and_selections.csv.gz")
    parser.add_argument("--quality-filter-file", type=Path,
                        default=QC_GENE_PANELS / "perturbation_quality_filter.csv.gz")
    parser.add_argument("--strength-dir", type=Path, default=STRENGTH_SCORES)
    parser.add_argument("--l1-selection-dir", type=Path, default=L1_SELECTION)
    parser.add_argument("--adjusted-p-cutoff", type=float, default=0.05)
    parser.add_argument("--default-maximum", type=int, default=2000)
    parser.add_argument("--vcc-maximum", type=int, default=150)
    return parser.parse_args()


def strength_l1(moments_path, de_path, primary, keep, alpha):
    """The perturbations in `keep`, in moments-file order, and their strength_L1."""
    with h5py.File(moments_path, "r") as moments, h5py.File(de_path, "r") as de:
        obs = axis_names(moments, "obs")
        names = [name for name in obs if name in keep]
        moments_row = {name: i for i, name in enumerate(obs)}
        de_row = {name: i for i, name in enumerate(de_perturbations(de))}
        moments_column = unique_columns(axis_names(moments, "var"))
        de_column = unique_columns(axis_names(de, "var"))
        genes = sorted(primary & moments_column.keys() & de_column.keys())
        tested = [name for name in names if name in de_row]
        tau = moments["X"][:][np.ix_([moments_row[name] for name in tested],
                                     [moments_column[gene] for gene in genes])].astype(np.float64)
        p_values = de["layers/pvalue"][:][np.ix_([de_row[name] for name in tested],
                                                 [de_column[gene] for gene in genes])]

    column = {gene: j for j, gene in enumerate(genes)}
    eligible = np.isfinite(tau)
    for i, name in enumerate(tested):
        if name in column:
            eligible[i, column[name]] = False
    p_values[~eligible] = np.nan
    rejected, _ = bh_rejection_rows(p_values, alpha)
    score = {name: np.abs(t[r]).sum() / n
             for name, t, r, n in zip(tested, tau, rejected, eligible.sum(axis=1))}
    return names, [score.get(name, np.nan) for name in names]


def main():
    args = parse_args()
    primary = load_primary_outcome_lists(args.selection_file)
    _, passing = load_quality_audit(args.quality_filter_file)

    tables = []
    for dataset, components in DATASET_COMPONENTS.items():
        for c in components:
            names, strength = strength_l1(args.pseudobulk_dir / f"{c}_moments.h5ad",
                                          args.de_dir / f"{c}_wilcoxon_batch_corrected.h5ad",
                                          set(primary[dataset]), passing[dataset],
                                          args.adjusted_p_cutoff)
            tables.append(pd.DataFrame({"dataset": dataset, "perturbation": names,
                                        "target_gene": names, "strength_L1": strength}))
    args.strength_dir.mkdir(parents=True, exist_ok=True)
    strength_file = args.strength_dir / "perturbation_strength_all_datasets.csv"
    pd.concat(tables, ignore_index=True).to_csv(strength_file, index=False)

    # strong perturbations, from the saved scores as they read back
    table = pd.read_csv(strength_file)
    ranked = table[table["strength_L1"] > 0].sort_values(
        ["strength_L1", "perturbation"], ascending=[False, True], kind="stable")
    maximum = np.where(ranked["dataset"].eq("VCC"), args.vcc_maximum, args.default_maximum)
    table["selected_L1_strong"] = table.index.isin(
        ranked.index[ranked.groupby("dataset").cumcount() < maximum])
    args.l1_selection_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.l1_selection_dir / "L1_strong_perturbation_membership.csv.gz", index=False,
                 compression="gzip")
    print(table.groupby("dataset", sort=False).agg(
        QC_passing=("perturbation", "size"), scored=("strength_L1", "count"),
        selected=("selected_L1_strong", "sum")).to_string(), flush=True)


if __name__ == "__main__":
    main()
