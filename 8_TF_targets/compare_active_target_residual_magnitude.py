#!/usr/bin/env python3
"""n, mean and median |R|, R = X - sigma1 u1 v1', over each TF perturbation's residual discoveries,
split into its SCENIC+ targets active in the cell line and all other genes (K562, HepG2, HCT116).
Reads SCENICPLUS_LOOM, FIRST_FACTOR, RESIDUAL_DISCOVERIES and MOMENTS_DIR; writes TF_RESULTS_DIR.
Panels: Fig 2f, S5d.
"""
import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from utils.effects import load_moments
from utils.paths import (
    DATASET_COMPONENTS,
    FIRST_FACTOR,
    MOMENTS_DIR,
    RESIDUAL_DISCOVERIES,
    SCENICPLUS_LOOM,
    TF_RESULTS_DIR,
    moments_file,
    paper_name,
    slug,
)

# dataset: cell line of its network
DATASETS = {"Replogle-GW-k562": "K562", "Nadig-HEPG2": "HepG2", "Huang-HCT116": "HCT116"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--loom", type=Path, default=SCENICPLUS_LOOM)
    parser.add_argument("--first-factor-dir", type=Path, default=FIRST_FACTOR)
    parser.add_argument("--residual-discovery-dir", type=Path, default=RESIDUAL_DISCOVERIES)
    parser.add_argument("--pseudobulk-dir", type=Path, default=MOMENTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=TF_RESULTS_DIR)
    return parser.parse_args()


def decode(values):
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in values])


def active_targets(loom_path):
    """{cell line: TF x gene boolean frame, True for the genes of the TF's active eRegulons}."""
    with h5py.File(loom_path, "r") as loom:
        genes = decode(loom["row_attrs/Gene"][:])
        membership = loom["row_attrs/Regulons"][:]  # one field per eRegulon
        auc = loom["col_attrs/RegulonsAUC"][:]
        cell_line = decode(loom["col_attrs/ACC_Cell_type"][:])
        metadata = json.loads(loom["attrs/MetaData"][()].decode())
    threshold = {r["regulon"]: r["defaultThresholdValue"] for r in metadata["regulonThresholds"]}
    regulons = list(membership.dtype.names)
    tf = np.array([re.split(r"_[+-]_", name)[0] for name in regulons])
    in_regulon = np.column_stack([membership[name] for name in regulons]).astype(bool)  # genes x eRegulons
    on = np.column_stack([auc[name] > threshold[name] for name in regulons])  # cells x eRegulons
    networks = {}
    for line in DATASETS.values():
        active = on[cell_line == line].mean(0) > 0.5
        regulon_genes = pd.DataFrame(in_regulon[:, active].T, index=tf[active], columns=genes)
        networks[line] = regulon_genes.groupby(level=0).any()  # union over the TF's eRegulons
    return networks


def magnitudes(values):
    """n, mean and median of |values|."""
    if len(values) == 0:
        return 0, np.nan, np.nan
    return len(values), np.abs(values).mean(), np.median(np.abs(values))


def tf_records(dataset, line, network, u, v, args):
    """One record per TF perturbation of the dataset in the network."""
    folder = args.residual_discovery_dir / "per_dataset" / slug(dataset)
    rows = pd.read_csv(folder / "row_metadata_and_rejection_counts.csv.gz").perturbation.astype(str)
    genes = pd.read_csv(folder / "column_metadata.csv.gz").gene.astype(str).tolist()
    in_network = np.flatnonzero(rows.isin(network.index))
    tfs = rows.iloc[in_network].tolist()
    rejected, missing = (
        sparse.load_npz(folder / name).tocsr()[in_network].toarray().astype(bool)
        for name in ["rejection_matrix_global_trend_adjusted_max_p_primary_BH.npz", "missing_entry_matrix.npz"]
    )

    paths = [args.pseudobulk_dir / moments_file(c) for c in DATASET_COMPONENTS[dataset]]
    moments = load_moments(paths, perturbations=set(tfs), genes=set(genes))
    x = pd.DataFrame(moments["X"], moments["perturbations"], moments["genes"]).reindex(index=tfs, columns=genes)
    sigma_u1 = u[u.dataset.eq(dataset)].set_index("perturbation").perturbation_factor_score_sigma_u1
    v1 = v[v.dataset.eq(dataset)].set_index("gene").right_singular_vector_v1
    residual = x.to_numpy(np.float64) - np.outer(sigma_u1.loc[tfs], v1.loc[genes])
    residual[np.array(tfs)[:, None] == np.array(genes)[None, :]] = np.nan  # direct target
    found = rejected & ~missing & np.isfinite(residual)
    targets = network.reindex(index=tfs, columns=genes, fill_value=False).to_numpy()

    records = []
    for i, name in enumerate(tfs):
        n_target, mean_target, median_target = magnitudes(residual[i, found[i] & targets[i]])
        n_other, mean_other, median_other = magnitudes(residual[i, found[i] & ~targets[i]])
        records.append({
            "dataset": dataset,
            "display_dataset": paper_name(dataset),
            "TF": name,
            "network": f"SCENICplus_{line}_active",
            "test": "conjunction",
            "effect_type": "residual",
            "condition": "conjunction_residual",
            "n_rejected_target_genes": n_target,
            "n_rejected_non_target_genes": n_other,
            "target_mean_abs_effect": mean_target,
            "non_target_mean_abs_effect": mean_other,
            "target_median_abs_effect": median_target,
            "non_target_median_abs_effect": median_other,
        })
    return records


def main():
    args = parse_args()
    networks = active_targets(args.loom)
    u = pd.read_csv(args.first_factor_dir / "first_svd_perturbation_loadings_all_datasets.csv.gz")
    v = pd.read_csv(args.first_factor_dir / "first_svd_gene_loadings_all_datasets.csv.gz")
    records = []
    for dataset, line in DATASETS.items():
        print(f"{paper_name(dataset)} with {line}-active targets")
        records += tf_records(dataset, line, networks[line], u, v, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.output_dir / "conjunction_residual_magnitude_by_TF.csv", index=False)


if __name__ == "__main__":
    main()
