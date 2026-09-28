"""Discovery matrices per dataset: BH within each perturbation of the Wilcoxon p (total), and of
max(Wilcoxon p, 2 Phi(-|E - sigma1 u1 v1'| / SE)) (residual).
Reads DE_DIR, MOMENTS_DIR, FIRST_FACTOR and QC_GENE_PANELS; writes TOTAL_DISCOVERIES and
RESIDUAL_DISCOVERIES(_CANONICAL). Panels: Fig 2a, 2e, 2f, S3a, S3b, S4, S5c, S5d.
"""

import argparse
import json

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import ndtr

from utils.effects import axis_names, de_perturbations, load_moments, load_quality_audit, unique_columns
from utils.paths import (
    DATASET_COMPONENTS,
    DE_DIR,
    FIRST_FACTOR,
    MOMENTS_DIR,
    QC_GENE_PANELS,
    RESIDUAL_DISCOVERIES,
    RESIDUAL_DISCOVERIES_CANONICAL,
    TOTAL_DISCOVERIES,
    paper_name,
    slug,
)
from utils.qc import load_primary_outcome_lists
from utils.stats import bh_rejection_rows


DATASETS = list(DATASET_COMPONENTS)
TOTAL = "rejection_matrix_target_missing_primary_BH.npz"
RESIDUAL = "rejection_matrix_global_trend_adjusted_max_p_primary_BH.npz"
MISSING = "missing_entry_matrix.npz"


def wilcoxon_pvalues(dataset, passing, primary):
    """Rows, genes and Wilcoxon p (direct target NaN) on the primary genes found in every DE file."""
    components = DATASET_COMPONENTS[dataset]
    paths = [DE_DIR / f"{component}_wilcoxon_batch_corrected.h5ad" for component in components]
    columns = []
    for path in paths:
        with h5py.File(path, "r") as handle:
            columns.append(unique_columns(axis_names(handle, "var")))
    genes = [gene for gene in primary if all(gene in column for column in columns)]
    rows, blocks = [], []
    for component, path, column in zip(components, paths, columns):
        with h5py.File(path, "r") as handle:
            names = de_perturbations(handle)
            keep = [i for i, name in enumerate(names) if name in passing]
            pvalues = handle["layers/pvalue"][:]
        blocks.append(pvalues[np.ix_(keep, [column[gene] for gene in genes])].astype(np.float64))
        rows += [(component, names[i]) for i in keep]
    rows = pd.DataFrame(rows, columns=["component", "perturbation"])
    rows.insert(0, "dataset", dataset)
    rows.insert(1, "display_dataset", paper_name(dataset))
    p = np.concatenate(blocks)
    target = {gene: j for j, gene in enumerate(genes)}
    for i, name in enumerate(rows["perturbation"]):
        if name in target:
            p[i, target[name]] = np.nan
    return rows, genes, p


def moment_matrices(dataset, perturbations, genes, layers=("X", "layers/effect_se")):
    """Moments layers (E = X, SE = layers/effect_se), float64, on the given rows and genes."""
    paths = [MOMENTS_DIR / f"{component}_moments.h5ad" for component in DATASET_COMPONENTS[dataset]]
    moments = load_moments(paths, layers, set(perturbations), set(genes))
    return [pd.DataFrame(moments[layer], moments["perturbations"], moments["genes"]).loc[perturbations, genes]
            .to_numpy(np.float64) for layer in layers]


def leading_factor(dataset, perturbations, genes):
    """sigma1 u1 v1' of the all-QC fit of 2_global_trend."""
    u = pd.read_csv(FIRST_FACTOR / "first_svd_perturbation_loadings_all_datasets.csv.gz")
    v = pd.read_csv(FIRST_FACTOR / "first_svd_gene_loadings_all_datasets.csv.gz")
    sigma_u1 = u[u["dataset"].eq(dataset)].set_index("perturbation").loc[perturbations]
    v1 = v[v["dataset"].eq(dataset)].set_index("gene").loc[genes]
    return np.outer(sigma_u1["perturbation_factor_score_sigma_u1"].to_numpy(np.float64),
                    v1["right_singular_vector_v1"].to_numpy(np.float64))


def z_pvalue(z, se, testable):
    """2 Phi(-|z| / SE) where testable, 1 elsewhere."""
    p = np.ones(z.shape)
    p[testable] = 2.0 * ndtr(-np.abs(z[testable] / se[testable]))
    return p


def save(directory, matrices, tables):
    directory.mkdir(parents=True, exist_ok=True)
    for name, matrix in matrices.items():
        sparse.save_npz(directory / name, sparse.csr_matrix(matrix, dtype=np.uint8))
    for name, table in tables.items():
        table.to_csv(directory / name, index=False)


def analyze(dataset, passing, primary, alpha):
    rows, genes, p_wilcoxon = wilcoxon_pvalues(dataset, passing, primary)
    perturbations = rows["perturbation"].tolist()
    effect, se = moment_matrices(dataset, perturbations, genes)
    finite = np.isfinite(p_wilcoxon)
    testable = finite & np.isfinite(effect) & np.isfinite(se) & (se > 0)

    def conjunction(z):
        return np.where(finite, np.maximum(p_wilcoxon, z_pvalue(z, se, testable)), np.nan)

    trend = leading_factor(dataset, perturbations, genes)
    total, n_tests = bh_rejection_rows(p_wilcoxon, alpha)
    residual = bh_rejection_rows(conjunction(effect - trend), alpha)[0]
    unshifted = bh_rejection_rows(conjunction(effect), alpha)[0]

    missing = ~finite
    columns = pd.DataFrame({"dataset": dataset, "gene": genes})
    counts = rows.assign(
        n_primary_tests=n_tests,
        n_original_primary_BH_rejections=total.sum(axis=1),
        n_global_trend_adjusted_primary_BH_rejections=residual.sum(axis=1),
    )
    folder = f"per_dataset/{slug(dataset)}"
    save(TOTAL_DISCOVERIES / folder, {TOTAL: total, MISSING: missing},
         {"row_metadata.csv.gz": rows, "column_metadata.csv.gz": columns})
    save(RESIDUAL_DISCOVERIES / folder, {RESIDUAL: residual, MISSING: missing},
         {"row_metadata_and_rejection_counts.csv.gz": counts, "column_metadata.csv.gz": columns})
    if dataset != "Feng-GW":
        save(RESIDUAL_DISCOVERIES_CANONICAL / folder, {TOTAL: residual, MISSING: missing},
             {"row_metadata.csv.gz": counts, "column_metadata.csv.gz": columns})
    audit = {
        "dataset": dataset,
        "display_dataset": paper_name(dataset),
        "n_perturbations": total.shape[0],
        "n_primary_outcomes": len(genes),
        "n_original_rejections": int(total.sum()),
        "n_global_trend_adjusted_rejections": int(residual.sum()),
        "n_unshifted_z_wilcoxon_max_p_rejections": int(unshifted.sum()),
    }
    (RESIDUAL_DISCOVERIES / folder / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(f"{paper_name(dataset)}: {audit['n_original_rejections']:,} -> "
          f"{audit['n_global_trend_adjusted_rejections']:,} discoveries", flush=True)


def aggregate():
    counts = pd.DataFrame([
        json.loads((RESIDUAL_DISCOVERIES / "per_dataset" / slug(dataset) / "audit.json").read_text())
        for dataset in DATASETS
    ])
    counts.drop(columns="n_unshifted_z_wilcoxon_max_p_rejections").to_csv(
        RESIDUAL_DISCOVERIES / "rejection_count_reduction_by_dataset.csv", index=False)
    counts.drop(columns="n_global_trend_adjusted_rejections").to_csv(
        RESIDUAL_DISCOVERIES / "unshifted_effect_se_test_rejection_counts_by_dataset.csv", index=False)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("analyze", "aggregate", "all"), default="all")
    parser.add_argument("--dataset-index", type=int, default=-1, help="one dataset (0-11); -1 analyzes all")
    parser.add_argument("--alpha", type=float, default=0.05, help="per-perturbation BH level")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode != "aggregate":
        primary = load_primary_outcome_lists(QC_GENE_PANELS / "outcome_gene_scores_and_selections.csv.gz")
        passing = load_quality_audit(QC_GENE_PANELS / "perturbation_quality_filter.csv.gz")[1]
        for dataset in DATASETS if args.dataset_index < 0 else [DATASETS[args.dataset_index]]:
            analyze(dataset, passing[dataset], primary[dataset], args.alpha)
    if args.mode != "analyze":
        aggregate()


if __name__ == "__main__":
    main()
