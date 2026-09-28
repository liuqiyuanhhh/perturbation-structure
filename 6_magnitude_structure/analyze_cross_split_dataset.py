#!/usr/bin/env python3
"""Rank-1 share F_1 = sigma_1^2 / ||.||_F^2 of the mean residual A_k and the cross-split product M_k,
with R_k = E - sum_{j<=k} sigma_j u_j v_j' in each of the 10 split halves, k = 0..15, for one dataset.
Reads SPLIT_DIR and QC_GENE_PANELS; writes CROSS_SPLIT/per_dataset/<dataset>/. Panels: Fig 2c, S5a.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds

from utils.effects import load_effects, load_moments, load_quality_audit
from utils.paths import CROSS_SPLIT, DATASET_COMPONENTS, QC_GENE_PANELS, SPLIT_DIR, paper_name
from utils.qc import load_primary_outcome_lists

N_SEEDS = 5
MAX_FACTORS = 15
PRODUCT_K = 20  # sigma_1 of M_k from a k = 20 fit with vectors, as in the paper runs
CAP_QUANTILE = 1 - 1e-4
TOLERANCE = 1e-7


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", required=True, choices=list(DATASET_COMPONENTS))
    parser.add_argument("--split-dir", type=Path, default=SPLIT_DIR)
    parser.add_argument("--gene-selection-file", type=Path,
                        default=QC_GENE_PANELS / "outcome_gene_scores_and_selections.csv.gz")
    parser.add_argument("--quality-filter-file", type=Path,
                        default=QC_GENE_PANELS / "perturbation_quality_filter.csv.gz")
    parser.add_argument("--output-dir", type=Path, default=CROSS_SPLIT)
    return parser.parse_args()


def start(matrix):
    return np.linspace(1.0, 2.0, min(matrix.shape), dtype=np.float32)


def half_paths(split_dir, dataset, seed, half):
    return [split_dir / f"{component}_seed{seed}_half{half}_moments.h5ad"
            for component in DATASET_COMPONENTS[dataset]]


def load_half(paths, perturbations, genes):
    """E of one half on the given axes (load_effects zeroes the direct targets)."""
    rows, columns, effects = load_effects(paths, set(perturbations), set(genes))
    row = {name: i for i, name in enumerate(rows)}
    column = {name: j for j, name in enumerate(columns)}
    return effects[np.ix_([row[name] for name in perturbations], [column[name] for name in genes])]


def leading_factors(matrix):
    """u, sigma, v' of the MAX_FACTORS leading triplets, largest first, float64."""
    u, s, vt = svds(matrix, k=MAX_FACTORS, v0=start(matrix), tol=TOLERANCE)
    order = np.argsort(s)[::-1]
    return u[:, order].astype(np.float64), s[order].astype(np.float64), vt[order].astype(np.float64)


def cap(matrix, target):
    """Clip in place at the CAP_QUANTILE quantile of |entry| off the direct targets."""
    cutoff = float(np.quantile(np.abs(matrix[~target]), CAP_QUANTILE))
    return np.clip(matrix, -cutoff, cutoff, out=matrix)


def f1_percent(matrix, sigma1):
    """100 F_1 = 100 sigma_1^2 / ||matrix||_F^2."""
    return 100 * (float(sigma1) ** 2 / float(np.sum(np.square(matrix, dtype=np.float64))))


def main():
    args = parse_args()
    dataset = args.dataset
    qc = load_quality_audit(args.quality_filter_file)[1][dataset]
    primary = load_primary_outcome_lists(args.gene_selection_file)[dataset]
    axes = [load_moments(half_paths(args.split_dir, dataset, seed, half), layers=())
            for seed in range(N_SEEDS) for half in (0, 1)]
    perturbations = sorted(set(qc).intersection(*(a["perturbations"] for a in axes)))
    genes = sorted(set(primary).intersection(*(a["genes"] for a in axes)))
    target = np.equal.outer(perturbations, genes)
    print(f"{dataset}: {len(perturbations):,} perturbations x {len(genes):,} primary outcomes",
          flush=True)

    effect = [np.zeros(target.shape, dtype=np.float32) for _ in range(MAX_FACTORS + 1)]
    product = [np.zeros(target.shape, dtype=np.float32) for _ in range(MAX_FACTORS + 1)]
    for seed in range(N_SEEDS):
        halves = [load_half(half_paths(args.split_dir, dataset, seed, half), perturbations, genes)
                  for half in (0, 1)]
        factors = [leading_factors(r) for r in halves]
        for k in range(MAX_FACTORS + 1):
            if k > 0:
                for r, (u, s, vt) in zip(halves, factors):
                    r -= np.outer(u[:, k - 1] * s[k - 1], vt[k - 1]).astype(np.float32)
                    r[target] = 0
            # weight 1/10 per half, 1/5 per seed product
            effect[k] += halves[0] * np.float32(0.1)
            effect[k] += halves[1] * np.float32(0.1)
            product[k] += halves[0] * halves[1] * np.float32(1 / N_SEEDS)

    rows = []
    for k in range(MAX_FACTORS + 1):
        a, m = cap(effect[k], target), cap(product[k], target)
        sigma_a = svds(a, k=1, v0=start(a), tol=TOLERANCE, return_singular_vectors=False)[0]
        sigma_m = svds(m, k=PRODUCT_K, v0=start(m), tol=TOLERANCE)[1].max()
        rows.append({
            "dataset": dataset,
            "display_dataset": paper_name(dataset),
            "n_factors_removed": k,
            "n_qc_perturbations": len(perturbations),
            "n_primary_outcomes": len(genes),
            "mean_split_effect_rank1_percent": f1_percent(a, sigma_a),
            "mean_signed_product_rank1_percent": f1_percent(m, sigma_m),
        })

    output = args.output_dir / "per_dataset" / dataset.replace("-", "_")
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "rank1_comparison.csv", index=False)
    parameters = {"dataset": dataset, "n_split_seeds": N_SEEDS, "maximum_factors": MAX_FACTORS,
                  "product_svd_k": PRODUCT_K, "cap_quantile": CAP_QUANTILE,
                  "svd_tolerance": TOLERANCE}
    (output / "run_parameters.json").write_text(json.dumps(parameters, indent=2) + "\n")


if __name__ == "__main__":
    main()
