#!/usr/bin/env python3
"""The 27 curated CORUM complexes and their complex-specific genes: g outside complex A with s > 0.25,
s_adj > 0.25, two-proportion z > 2.5 and >= 3 discoveries in A; s = residual discovery rate in A minus
outside, s_adj = s minus the same difference of mu_pg = d_p c_g / rho.
Reads CORUM_GMT and RESIDUAL_DISCOVERIES; writes two tables to COMPLEX_ENRICHMENT. Panels: Fig 2e, S5c.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from utils.paths import (
    COMPLEX_ENRICHMENT,
    CORUM_GMT,
    DATASET_COMPONENTS,
    RESIDUAL_DISCOVERIES,
    slug,
)

COMPLEX_TABLE = "corum_curated_complex_genes.csv"
DATASETS = list(DATASET_COMPONENTS)

CURATED = {"Mediator": "230", "SAGA": "6643", "TFIID": "509", "TFIIH": "1029", "BAF": "189",
           "Cohesin": "5432", "Condensin_I": "10", "APC/C": "96", "MCM": "387", "ORC": "1031",
           "Fanconi": "2739", "Integrator": "1153", "LSM2-8": "562", "SMN": "1142",
           "Exosome": "788", "eIF3": "4403", "EIF2B": "7292", "Multisynthetase": "3040",
           "20S_prot": "8850", "CCT": "126", "COP9": "2174", "Exocyst": "6167", "ARP2/3": "27",
           "TRAPP": "6468", "gamma-TuRC": "6893", "MitoRibo": "320", "RespCplxI": "178"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gmt", type=Path, default=CORUM_GMT, help="human CORUM GMT")
    parser.add_argument("--adjusted-rejection-dir", type=Path,
                        default=RESIDUAL_DISCOVERIES / "per_dataset")
    parser.add_argument("--enrichment-dir", type=Path, default=COMPLEX_ENRICHMENT)
    return parser.parse_args()


def complex_table(gmt):
    """One row per curated complex x member gene."""
    corum = {}
    with open(gmt) as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            corum[fields[0].split("#")[-1]] = (fields[0], fields[2:])
    return pd.DataFrame(
        [(label, corum_id, corum[corum_id][0], gene)
         for label, corum_id in CURATED.items() for gene in sorted(corum[corum_id][1])],
        columns=["complex", "corum_id", "corum_name", "gene"])


def load_complexes(path):
    """{complex: member genes}, in CURATED order."""
    table = pd.read_csv(path)
    return {label: set(genes.astype(str))
            for label, genes in table.groupby("complex", sort=False)["gene"]}


def load_discoveries(dataset, root):
    """Perturbation x gene residual discoveries: 1, 0, NaN = untested."""
    folder = root / slug(dataset)
    values = sparse.load_npz(folder / "rejection_matrix_global_trend_adjusted_max_p_primary_BH.npz")
    values = values.toarray().astype(np.float32)
    values[sparse.load_npz(folder / "missing_entry_matrix.npz").nonzero()] = np.nan
    rows = pd.read_csv(folder / "row_metadata_and_rejection_counts.csv.gz")["perturbation"]
    columns = pd.read_csv(folder / "column_metadata.csv.gz")["gene"]
    return pd.DataFrame(values, index=rows.astype(str).tolist(),
                        columns=columns.astype(str).tolist())


@np.errstate(divide="ignore", invalid="ignore")
def complex_specific_genes(dataset, frame, members):
    """The complex-specific genes of one dataset, one row per (complex, gene)."""
    frame = frame[frame.index.isin(set().union(*members.values()))]
    r = frame.to_numpy(dtype=np.float64)
    genes = frame.columns
    d, c, rho = np.nanmean(r, axis=1), np.nanmean(r, axis=0), np.nanmean(r)
    mu = np.outer(d, c / rho)
    mu[np.isnan(r)] = np.nan

    tables = []
    for label, subunits in members.items():
        inside = frame.index.isin(subunits)
        if inside.sum() < 3:
            continue
        m_in, m_out = np.isfinite(r[inside]).sum(axis=0), np.isfinite(r[~inside]).sum(axis=0)
        rate_in, rate_out = np.nanmean(r[inside], axis=0), np.nanmean(r[~inside], axis=0)
        s = rate_in - rate_out
        s_adj = s - (np.nanmean(mu[inside], axis=0) - np.nanmean(mu[~inside], axis=0))
        pooled = (m_in * rate_in + m_out * rate_out) / (m_in + m_out)
        z = s / np.sqrt(pooled * (1.0 - pooled) * (1.0 / m_in + 1.0 / m_out))
        n_discoveries = np.nansum(r[inside], axis=0).astype(np.int64)
        specific = ((s > 0.25) & (s_adj > 0.25) & (z > 2.5) & (n_discoveries >= 3)
                    & ~genes.isin(subunits))
        table = pd.DataFrame({"dataset": dataset, "complex": label, "gene": genes,
                              "raw_specificity": s, "adjusted_specificity": s_adj,
                              "z_enrich": z, "n_rejections_in_complex": n_discoveries})[specific]
        tables.append(table.sort_values("adjusted_specificity", ascending=False, kind="stable"))
    return pd.concat(tables)


def main():
    args = parse_args()
    args.enrichment_dir.mkdir(parents=True, exist_ok=True)
    complex_table(args.gmt).to_csv(args.enrichment_dir / COMPLEX_TABLE, index=False)
    members = load_complexes(args.enrichment_dir / COMPLEX_TABLE)

    tables = []
    for dataset in DATASETS:
        frame = load_discoveries(dataset, args.adjusted_rejection_dir)
        tables.append(complex_specific_genes(dataset, frame, members))
        print(f"{dataset}: {len(tables[-1])} complex-specific (complex, gene) pairs", flush=True)
    pd.concat(tables).to_csv(args.enrichment_dir / "complex_specific_genes.csv", index=False)


if __name__ == "__main__":
    main()
