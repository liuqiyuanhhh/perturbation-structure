#!/usr/bin/env python3
"""Leading factor sigma1 u1 v1' of each dataset, fitted on all QC-passing
perturbations x primary genes.  Reads MOMENTS_DIR and QC_GENE_PANELS; writes
loadings and a summary to FIRST_FACTOR.
Fig 1b right, S1d; via the residuals, Fig 2a, 2e, 2f, S3a, S3b, S4, S5c, S5d."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils.effects import fit_and_remove_first_svd_factor, load_effects, load_selections
from utils.paths import (
    DATASET_COMPONENTS,
    FIRST_FACTOR,
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
    selections = load_selections(args.quality_filter_file, args.gene_selection_file)
    u_tables, v_tables, summaries = [], [], []
    for label, components in DATASET_COMPONENTS.items():
        paths = [args.pseudobulk_dir / moments_file(c) for c in components]
        chosen = selections[label]
        u, v, summary = fit(label, paths, chosen["qc"], chosen["primary"])
        u_tables.append(u)
        v_tables.append(v)
        summaries.append(summary)
        print(f"{paper_name(label)}: {len(u):,} x {len(v):,}, "
              f"sigma1 {summary['singular_value_sigma1']:.6g}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for kind, frames in (("perturbation", u_tables), ("gene", v_tables)):
        pd.concat(frames, ignore_index=True).to_csv(
            args.output_dir / f"first_svd_{kind}_loadings_all_datasets.csv.gz",
            index=False, compression="gzip")
    pd.DataFrame(summaries).to_csv(args.output_dir / "first_svd_factor_summary.csv", index=False)


if __name__ == "__main__":
    main()
