"""Held-out nestedness of the total and residual discovery matrices over stratified train/test splits,
with genes ranked by training discovery frequency: block discovery rates (triangles) and the share of
each held-out perturbation's discoveries in its top-ranked genes (capture curves).
Reads TOTAL_DISCOVERIES and RESIDUAL_DISCOVERIES_CANONICAL; writes under the latter. Panels: Fig 2a, S3b.
"""

import argparse
import hashlib
import json
import math

import numpy as np
import pandas as pd
from scipy import sparse

from utils.paths import (
    DATASET_COMPONENTS,
    RESIDUAL_DISCOVERIES_CANONICAL,
    TOTAL_DISCOVERIES,
    paper_name,
    slug,
)


DATASETS = [dataset for dataset in DATASET_COMPONENTS if dataset != "Feng-GW"]
CAPTURE_DATASETS = [dataset for dataset in DATASETS if dataset != "Nourreddine-GW-ipsc"]
FRACTIONS = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00)
CURVES = ["original_fraction_rejections_captured", "adjusted_fraction_rejections_captured"]
TRIANGLES = RESIDUAL_DISCOVERIES_CANONICAL / "triangle_visualization_polish"
CAPTURE = RESIDUAL_DISCOVERIES_CANONICAL / "capture_curve_comparison"
MATRIX = "rejection_matrix_target_missing_primary_BH.npz"


def load(dataset):
    """Total- and residual-effect matrices, the missing entries, perturbations and genes."""
    total = TOTAL_DISCOVERIES / "per_dataset" / slug(dataset)
    residual = RESIDUAL_DISCOVERIES_CANONICAL / "per_dataset" / slug(dataset)
    return (
        sparse.load_npz(total / MATRIX),
        sparse.load_npz(residual / MATRIX),
        sparse.load_npz(total / "missing_entry_matrix.npz"),
        pd.read_csv(total / "row_metadata.csv.gz")["perturbation"].astype(str).to_numpy(),
        pd.read_csv(total / "column_metadata.csv.gz")["gene"].astype(str).to_numpy(),
    )


def row_counts(matrix):
    return np.asarray(matrix.sum(axis=1)).ravel().astype(np.int64)


def column_counts(matrix, rows):
    return np.asarray(matrix[rows].sum(axis=0)).ravel().astype(np.int64)


def splits(dataset, original, missing, perturbations, args):
    """(split, training rows, held-out rows) of each split."""
    density = row_counts(original) / (original.shape[1] - row_counts(missing))
    density = np.where(np.isfinite(density), density, -np.inf)
    order = np.lexsort((np.asarray(perturbations, dtype=str), density))
    strata = np.empty(len(order), dtype=np.int16)
    strata[order] = np.minimum(4, np.arange(len(order)) * 5 // len(order)) + 1
    key = int.from_bytes(hashlib.sha256(dataset.encode("utf-8")).digest()[:4], "little")  # stable_integer
    for split in range(1, args.n_splits + 1):
        rng = np.random.default_rng(np.random.SeedSequence([args.master_seed, key, split, 1]))
        train, test = [], []
        for stratum in np.unique(strata):
            members = rng.permutation(np.flatnonzero(strata == stratum))
            n_train = min(max(round(args.train_fraction * len(members)), 1), len(members) - 1)
            train.append(members[:n_train])
            test.append(members[n_train:])
        yield split, np.sort(np.concatenate(train)), np.sort(np.concatenate(test))


def gene_order(matrix, missing, train, genes):
    """Genes by decreasing f_g over the training rows, ties by name."""
    observed = len(train) - column_counts(missing, train)
    frequency = np.divide(column_counts(matrix, train), observed,
                          out=np.full(len(genes), np.nan), where=observed > 0)
    alphabetical = np.argsort(genes, kind="mergesort")
    return alphabetical[np.argsort(-frequency[alphabetical], kind="mergesort")]


# ---- triangles


def block_means(matrix, missing, rows, row_order, column_order, bins):
    """Discoveries / observed entries in each of the bins x bins equal-count blocks."""
    row_bin, gene_bin = np.empty(len(row_order), np.int64), np.empty(len(column_order), np.int64)
    for b, (r, g) in enumerate(zip(np.array_split(row_order, bins), np.array_split(column_order, bins))):
        row_bin[r], gene_bin[g] = b, b

    def block_sums(indicator):
        entries = indicator[rows].tocoo()
        cells = row_bin[entries.row] * bins + gene_bin[entries.col]
        return np.bincount(cells, weights=entries.data, minlength=bins * bins).reshape(bins, bins)

    size = np.outer(np.bincount(row_bin, minlength=bins), np.bincount(gene_bin, minlength=bins))
    return block_sums(matrix) / (size - block_sums(missing))


def triangles(dataset, original, adjusted, missing, perturbations, genes, args):
    counts = {"original": row_counts(original), "adjusted": row_counts(adjusted)}
    observed = len(genes) - row_counts(missing)
    blocks, used = {"original": [], "adjusted": []}, []
    for _, train, test in splits(dataset, original, missing, perturbations, args):
        rows = test[counts["original"][test] >= args.minimum_rejections]
        if len(rows) < max(args.minimum_rows_per_split, args.bins):
            continue
        for name, matrix in (("original", original), ("adjusted", adjusted)):
            row_order = np.lexsort((perturbations[rows], -(counts[name] / observed)[rows]))
            blocks[name].append(block_means(matrix, missing, rows, row_order,
                                            gene_order(matrix, missing, train, genes), args.bins))
        n_original, n_adjusted = counts["original"][rows].sum(), counts["adjusted"][rows].sum()
        n_observed = observed[rows].sum()
        used.append({"original": n_original / n_observed, "adjusted": n_adjusted / n_observed,
                     "retained": n_adjusted / n_original})

    bins, grid = args.bins, f"{args.bins}x{args.bins}"
    mean = {name: np.asarray(stack, dtype=np.float64).mean(axis=0) if stack else np.full((bins, bins), np.nan)
            for name, stack in blocks.items()}
    output_dir = TRIANGLES / "per_dataset" / slug(dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "dataset": dataset,
        "display_dataset": paper_name(dataset),
        "row_bin": np.repeat(np.arange(1, bins + 1), bins),
        "gene_bin": np.tile(np.arange(1, bins + 1), bins),
        "original_mean_rejection_indicator": mean["original"].ravel(),
        "adjusted_mean_rejection_indicator_own_order": mean["adjusted"].ravel(),
    }).to_csv(output_dir / f"matched_original_adjusted_mean_triangle_{grid}.csv", index=False)
    summary = {"dataset": dataset, "display_dataset": paper_name(dataset), f"n_splits_used_{grid}": len(used)}
    for key, name in (("original", "mean_original_rejection_density"),
                      ("adjusted", "mean_adjusted_rejection_density"),
                      ("retained", "mean_fraction_original_calls_retained")):
        summary[f"{name}_{grid}"] = float(np.mean([split[key] for split in used])) if used else np.nan
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{paper_name(dataset)}: {grid} from {len(used)} splits", flush=True)


# ---- capture curves


def capture(discoveries, missing, position):
    """Share of the discoveries among the top FRACTIONS of the observed genes in the ranking `position`."""
    rank = position[discoveries] - np.searchsorted(np.sort(position[missing]), position[discoveries])
    n = len(position) - len(missing)
    return [np.sum(rank < math.ceil(x * n)) / len(discoveries) for x in FRACTIONS]


def capture_curves(dataset, original, adjusted, missing, perturbations, genes, args):
    columns = [np.split(matrix.indices, matrix.indptr[1:-1]) for matrix in (original, adjusted, missing)]
    records = []
    for split, train, test in splits(dataset, original, missing, perturbations, args):
        positions = []
        for matrix in (original, adjusted):
            position = np.empty(len(genes), dtype=np.int64)
            position[gene_order(matrix, missing, train, genes)] = np.arange(len(genes))
            positions.append(position)
        curves = []
        for row in test:
            o, a, m = (column[row] for column in columns)
            n = len(genes) - len(m)
            if 0 < len(o) < n and 0 < len(a) < n:
                curves.append([capture(o, m, positions[0]), capture(a, m, positions[1])])
        for x, o, a in zip(FRACTIONS, *np.mean(curves, axis=0)):
            records.append({"dataset": dataset, "display_dataset": paper_name(dataset), "split_id": split,
                            "top_gene_fraction": x, CURVES[0]: o, CURVES[1]: a})
    output_dir = CAPTURE / "per_dataset" / slug(dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output_dir / "matched_capture_curves_by_split.csv", index=False)


def aggregate(args):
    grid = f"{args.bins}x{args.bins}"
    folders = [TRIANGLES / "per_dataset" / slug(dataset) for dataset in DATASETS]
    pd.DataFrame([json.loads((folder / "dataset_summary.json").read_text()) for folder in folders]).to_csv(
        TRIANGLES / "matched_triangle_dataset_summary.csv", index=False)
    name = f"matched_original_adjusted_mean_triangle_{grid}.csv"
    pd.concat([pd.read_csv(folder / name) for folder in folders], ignore_index=True).to_csv(
        TRIANGLES / f"matched_original_adjusted_mean_triangle_all_datasets_{grid}.csv", index=False)

    by_split = pd.concat(
        [pd.read_csv(CAPTURE / "per_dataset" / slug(dataset) / "matched_capture_curves_by_split.csv")
         for dataset in CAPTURE_DATASETS],
        ignore_index=True,
    )
    by_dataset = (
        by_split.groupby(["dataset", "display_dataset", "top_gene_fraction"], as_index=False)[CURVES]
        .mean()
        .sort_values(["dataset", "top_gene_fraction"])
    )
    pooled = by_dataset.groupby("top_gene_fraction", as_index=False)[CURVES].mean().rename(columns={
        "original_fraction_rejections_captured": "original_equal_dataset_mean_capture",
        "adjusted_fraction_rejections_captured": "adjusted_equal_dataset_mean_capture",
    })
    by_dataset.to_csv(CAPTURE / "matched_capture_curves_by_dataset.csv", index=False)
    pooled.to_csv(CAPTURE / "matched_capture_curves_equal_dataset_mean.csv", index=False)
    top20 = pooled.loc[np.isclose(pooled["top_gene_fraction"], 0.20)].iloc[0]
    print(f"Top-20% capture, equal-dataset mean: original {top20.original_equal_dataset_mean_capture:.1%}, "
          f"adjusted {top20.adjusted_equal_dataset_mean_capture:.1%}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("analyze", "aggregate", "all"), default="all")
    parser.add_argument("--dataset-index", type=int, default=-1, help="one dataset (0-10); -1 analyzes all")
    parser.add_argument("--n-splits", type=int, default=100)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--master-seed", type=int, default=20260824)
    parser.add_argument("--minimum-rejections", type=int, default=20)
    parser.add_argument("--minimum-rows-per-split", type=int, default=60)
    parser.add_argument("--bins", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode != "aggregate":
        for dataset in DATASETS if args.dataset_index < 0 else [DATASETS[args.dataset_index]]:
            matrices = load(dataset)
            triangles(dataset, *matrices, args)
            if dataset in CAPTURE_DATASETS:
                capture_curves(dataset, *matrices, args)
    if args.mode != "analyze":
        aggregate(args)


if __name__ == "__main__":
    main()
