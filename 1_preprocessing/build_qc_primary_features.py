#!/usr/bin/env python3
"""Perturbation QC, primary outcome genes and the top-2,000 feature genes (largest effect
variance over the QC-passing perturbations) of the 12 screens.
Reads MOMENTS_DIR; writes the perturbation and gene tables to QC_GENE_PANELS."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils.paths import DATASET_COMPONENTS, MOMENTS_DIR, QC_GENE_PANELS, moments_file
from utils.qc import load_qc_dataset, perturbation_qc, primary_outcomes


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input-dir", type=Path, default=MOMENTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=QC_GENE_PANELS)
    parser.add_argument("--primary-expression-cutoff", type=float, default=0.08)
    parser.add_argument("--target-control-cutoff", type=float, default=0.08)
    parser.add_argument("--minimum-inactivation-efficiency", type=float, default=0.10)
    parser.add_argument("--n-feature-genes", type=int, default=2000)
    return parser.parse_args()


def main():
    args = parse_args()
    quality_tables, gene_tables = [], []
    for label, components in DATASET_COMPONENTS.items():
        data = load_qc_dataset([args.input_dir / moments_file(c) for c in components])
        genes, perturbations = data["genes"], data["perturbations"]
        audit = perturbation_qc(label, data, args.target_control_cutoff,
                                args.minimum_inactivation_efficiency)
        primary = primary_outcomes(data["control_mean"], data["mean_treated"],
                                   args.primary_expression_cutoff)

        # variance over the QC-passing perturbations, each one's target left out
        passing = np.flatnonzero(audit["passes_perturbation_quality_filter"])
        column = {gene: j for j, gene in enumerate(genes)}
        effects = data["effects"][passing]
        effects[np.arange(len(passing)), [column[perturbations[i]] for i in passing]] = np.nan
        variance = np.nanvar(effects, axis=0, dtype=np.float64)
        ranked = sorted(np.flatnonzero(np.isfinite(variance) & primary),
                        key=lambda j: (-variance[j], genes[j]))
        features = np.zeros(len(genes), dtype=bool)
        features[ranked[:args.n_feature_genes]] = True

        quality_tables.append(audit)
        gene_tables.append(pd.DataFrame({
            "dataset": label,
            "gene": genes,
            "control_mean": data["control_mean"],
            "reconstructed_mean_treated_across_all_pre_qc_perturbations": data["mean_treated"],
            "effect_variance_across_qc_passing_perturbations": variance,
            "selected_primary_control_or_treated_ge_0.08": primary,
            "selected_top2000_effect_variance": features,
        }))
        print(f"{label}: {len(passing):,}/{len(perturbations):,} perturbations pass QC, "
              f"{int(primary.sum()):,} primary genes", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(quality_tables, ignore_index=True).to_csv(
        args.output_dir / "perturbation_quality_filter.csv.gz", index=False, compression="gzip")
    pd.concat(gene_tables, ignore_index=True).to_csv(
        args.output_dir / "outcome_gene_scores_and_selections.csv.gz", index=False,
        compression="gzip")


if __name__ == "__main__":
    main()
