"""DES(p) = |top_k(p) & D_p| / k, k = |D_p|, of each method's held-out predictions for total and
residual discoveries D_p, plus the median |E| and |R| over each perturbation's discoveries.
Reads the 4_prediction pickles, discovery matrices and moments; writes RESPONSE_BREADTH. Panels: S4a-c.
"""

import argparse
import hashlib

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.utils.extmath import randomized_svd

from utils.effects import load_quality_audit
from utils.paths import (
    DATASET_ORDER,
    PAPER_PREDICTION_RESULTS,
    QC_GENE_PANELS,
    RESIDUAL_DISCOVERIES_CANONICAL,
    RESPONSE_BREADTH,
    TOTAL_DISCOVERIES,
    effect_key,
    paper_name,
    sisters,
    slug,
)

from build_discovery_matrices import leading_factor, moment_matrices


DATASETS = [dataset for dataset in DATASET_ORDER if dataset != "Feng-GW"]
# prediction-pickle method -> output name
METHOD_KEYS = {
    "weighted": "weighted",
    "presage": "presage",
    "paper_linear_embedding_from_training_10": "linear_train",
    "linear_embedding_from_otherds_10": "linear_otherds",
    "gears": "gears",
    "scGPT-ft": "scGPT-ft",
    "train_mean": "train_mean",
}
ZERO_TOL = 1e-5


def load_discoveries(root, dataset):
    folder = root / "per_dataset" / slug(dataset)
    return (
        sparse.load_npz(folder / "rejection_matrix_target_missing_primary_BH.npz"),
        pd.read_csv(folder / "row_metadata.csv.gz")["perturbation"].astype(str).tolist(),
        pd.read_csv(folder / "column_metadata.csv.gz")["gene"].astype(str).tolist(),
    )


def source_covered(dataset):
    """QC-passing perturbations of the source screens of a target dataset."""
    _, passing = load_quality_audit(QC_GENE_PANELS / "perturbation_quality_filter.csv.gz")
    excluded = {dataset, *sisters(dataset)}
    return set().union(*(members for name, members in passing.items() if name not in excluded))


def leading_factor_residual(values):
    """A prediction block minus its leading SVD factor; entries below ZERO_TOL set to 0."""
    if not values.any():
        return np.zeros_like(values)
    u, s, vt = randomized_svd(values, n_components=1, n_iter=7, random_state=0)
    residual = values - (u * s) @ vt
    residual[np.abs(residual) < ZERO_TOL] = 0.0
    return residual


def by_magnitude(values, target, k):
    """The k columns of largest |values| other than the target, ties in column order."""
    order = np.argsort(-np.abs(values), kind="stable")
    return order[order != target][:k]


def at_random(n_genes, target, k, *labels):
    """k genes other than the target, drawn without replacement; seeded by the labels."""
    seed = int.from_bytes(hashlib.sha256("\x1f".join(map(str, labels)).encode()).digest()[:8], "little")
    genes = np.arange(n_genes, dtype=np.int64)
    return np.random.default_rng(seed).choice(genes[genes != target], size=k, replace=False)


def des(top, discoveries):
    return np.intersect1d(top, discoveries).size / len(discoveries) if len(discoveries) else np.nan


def des_scores(dataset, truths, perturbations, genes):
    row_of = {name: i for i, name in enumerate(perturbations)}
    gene_of = {gene: j for j, gene in enumerate(genes)}
    splits = joblib.load(PAPER_PREDICTION_RESULTS / f"{effect_key(dataset)}_splits_by_seed.pkl")
    predictions = joblib.load(PAPER_PREDICTION_RESULTS / f"{effect_key(dataset)}_predictions_by_seed.pkl")
    covered = source_covered(dataset)

    records = []
    for fold in sorted(splits):
        train = [row_of[name] for name in map(str, splits[fold]["train"]) if name in row_of]
        test = [name for name in map(str, splits[fold]["test"]) if name in covered and name in row_of]
        targets = [gene_of.get(name, -1) for name in test]
        blocks = {"original": {}, "conjunction": {}}
        for key, method in METHOD_KEYS.items():
            if key in predictions[fold]:
                frame = predictions[fold][key].rename(index=str, columns=str)
                values = frame.loc[test, genes].to_numpy(np.float32)
                blocks["original"][method] = values
                # train_mean is one repeated profile, so it has no residual
                if method != "train_mean":
                    blocks["conjunction"][method] = leading_factor_residual(values)

        for condition, truth in truths.items():
            discoveries = [truth[row_of[name]].indices for name in test]
            frequency = np.argsort(-np.asarray(truth[train].mean(axis=0)).ravel(), kind="stable")
            tops = {method: [by_magnitude(v, t, len(d)) for v, t, d in zip(values, targets, discoveries)]
                    for method, values in blocks[condition].items()}
            tops["rejfreq"] = [frequency[frequency != t][:len(d)] for t, d in zip(targets, discoveries)]
            tops["random"] = [at_random(len(genes), t, len(d), dataset, fold, condition, name)
                              for name, t, d in zip(test, targets, discoveries)]
            for method, top in tops.items():
                records += [{"dataset": dataset, "seed": int(fold), "method": method, "condition": condition,
                             "perturbation": name, "n_truth_rejections": len(d), "DES": des(t, d)}
                            for name, t, d in zip(test, top, discoveries)]
        print(f"{paper_name(dataset)} fold {fold}: {len(test)} held-out perturbations scored", flush=True)
    return pd.DataFrame(records)


def median_abs(values):
    values = np.abs(values[np.isfinite(values)])
    return float(np.median(values)) if len(values) else np.nan


def effect_magnitudes(dataset, total, residual, perturbations, genes):
    """Median |E| over the total-effect and median |R| over the residual-effect discoveries."""
    effect = moment_matrices(dataset, perturbations, genes, ("X",))[0]
    residual_effect = effect - leading_factor(dataset, perturbations, genes)
    return pd.DataFrame([{
        "dataset": dataset,
        "perturbation": name,
        "original_rejection_count": len(total[i].indices),
        "conjunction_rejection_count": len(residual[i].indices),
        "median_abs_total_effect_among_original_rejections": median_abs(effect[i, total[i].indices]),
        "median_abs_residual_effect_among_conjunction_rejections":
            median_abs(residual_effect[i, residual[i].indices]),
    } for i, name in enumerate(perturbations)])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-index", type=int, required=True, help="0-10, order of DATASETS")
    dataset = DATASETS[parser.parse_args().dataset_index]

    total, perturbations, genes = load_discoveries(TOTAL_DISCOVERIES, dataset)
    residual = load_discoveries(RESIDUAL_DISCOVERIES_CANONICAL, dataset)[0]
    output_dir = RESPONSE_BREADTH / "per_dataset" / slug(dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    des_scores(dataset, {"original": total, "conjunction": residual}, perturbations, genes).to_csv(
        output_dir / "DES_by_perturbation.csv.gz", index=False)
    effect_magnitudes(dataset, total, residual, perturbations, genes).to_csv(
        output_dir / "rejected_effect_magnitude_by_perturbation.csv.gz", index=False)


if __name__ == "__main__":
    main()
