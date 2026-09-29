#!/usr/bin/env python3
"""Leading factor sigma1 u1 v1' of each dataset, fitted on all
QC-passing and on the L1-selected perturbations x primary genes.  Reads MOMENTS_DIR,
QC_GENE_PANELS and L1_SELECTION; writes loadings and a summary to FIRST_FACTOR.
Fig 1b right, S1d; via the residuals, Fig 2a, 2e, 2f, S3a, S3b, S4, S5c, S5d."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils.effects import fit_and_remove_first_svd_factor, load_effects, load_selections
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


def fit(label, paths, perturbations, genes):
    """Perturbation and gene loading tables of the leading factor, and the summary row."""
    rows, columns, effects = load_effects(paths, perturbations, genes)
    sigma, u1, v1 = fit_and_remove_first_svd_factor(effects)
    names = {"dataset": label, "display_dataset": paper_name(label)}
    u = pd.DataFrame({**names, "perturbation": rows, "left_singular_vector_u1": u1,
                      "perturbation_factor_score_sigma_u1": sigma * u1,
                      "absolute_u1": np.abs(u1), "singular_value_sigma1": sigma})
    v = pd.DataFrame({**names, "gene": columns, "right_singular_vector_v1": v1,
                      "singular_value_sigma1": sigma})
    summary = {**names, "n_perturbations_used_to_fit_svd": len(rows),
               "n_primary_outcomes_used_to_fit_svd": len(columns), "singular_value_sigma1": sigma}
    return u, v, summary


def main():
    args = parse_args()
    selections = load_selections(args.quality_filter_file, args.gene_selection_file,
                                 args.l1_membership_file)
    tables = {"first_svd": ([], []), "first_svd_L1_selected": ([], [])}
    summaries = []
    for label, components in DATASET_COMPONENTS.items():
        paths = [args.pseudobulk_dir / moments_file(c) for c in components]
        chosen = selections[label]
        for prefix, pool in (("first_svd", "qc"), ("first_svd_L1_selected", "L1")):
            u, v, summary = fit(label, paths, chosen[pool], chosen["primary"])
            tables[prefix][0].append(u)
            tables[prefix][1].append(v)
            print(f"{paper_name(label)} ({pool}): {len(u):,} x {len(v):,}, "
                  f"sigma1 {summary['singular_value_sigma1']:.6g}", flush=True)
            if pool == "qc":
                summaries.append(summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for prefix, (u, v) in tables.items():
        # the all-QC tables are read downstream at full precision
        float_format = None if prefix == "first_svd" else "%.12g"
        for kind, frames in (("perturbation", u), ("gene", v)):
            pd.concat(frames, ignore_index=True).to_csv(
                args.output_dir / f"{prefix}_{kind}_loadings_all_datasets.csv.gz",
                index=False, compression="gzip", float_format=float_format)
    pd.DataFrame(summaries).to_csv(args.output_dir / "first_svd_factor_summary.csv", index=False)


if __name__ == "__main__":
    main()
