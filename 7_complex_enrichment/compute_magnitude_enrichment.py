#!/usr/bin/env python3
"""enrichment(q) = complex-specific share in the top q% of residual discoveries by |R| / share among all,
R = E - sigma_1 u_1 v_1'; pooled over complexes (Fig 2e) and per complex at q = 5 (S5c).
Reads PAPER_EFFECT_DICT, RESIDUAL_DISCOVERIES and the tables of define_complex_specific_genes.py;
writes both results to COMPLEX_ENRICHMENT.
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds

from utils.paths import (
    COMPLEX_ENRICHMENT,
    PAPER_EFFECT_DICT,
    RESIDUAL_DISCOVERIES,
    effect_key,
    paper_name,
)

from define_complex_specific_genes import COMPLEX_TABLE, DATASETS, load_complexes, load_discoveries

CURVES_FILE = "complex_specific_magnitude_enrichment/complex_specific_magnitude_enrichment_curves.csv"
TOP5_FILE = "complex_level_top5_enrichment/complex_top5_enrichment_long.csv"
QGRID = np.geomspace(100.0, 1.0, 61)
TOP5_DATASETS = [dataset for dataset in DATASETS if dataset != "Feng-GW"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--effect-pkl", type=Path, default=PAPER_EFFECT_DICT)
    parser.add_argument("--adjusted-rejection-dir", type=Path,
                        default=RESIDUAL_DISCOVERIES / "per_dataset")
    parser.add_argument("--enrichment-dir", type=Path, default=COMPLEX_ENRICHMENT,
                        help="tables of define_complex_specific_genes.py; the output goes below it")
    return parser.parse_args()


def direct_targets(perturbations, genes):
    """Row and column positions of the entries whose outcome is the perturbed gene."""
    rows = np.flatnonzero(perturbations.isin(genes))
    return rows, genes.get_indexer(perturbations[rows])


def residual(frame):
    """R = E - sigma_1 u_1 v_1', E the effects with the direct targets set to 0."""
    effect = frame.to_numpy(dtype=np.float64, copy=True)
    effect[direct_targets(frame.index, frame.columns)] = 0.0
    u, s, vt = svds(effect, k=6, tol=0, v0=np.ones(min(effect.shape)))
    i = np.argmax(s)
    effect -= np.outer(u[:, i] * s[i], vt[i])
    return effect


def excluded_pairs(perturbations, genes, members):
    """True where the outcome is the perturbed gene or shares a curated complex with it."""
    excluded = np.zeros((len(perturbations), len(genes)), dtype=bool)
    for subunits in members.values():
        excluded[np.ix_(perturbations.isin(subunits), genes.isin(subunits))] = True
    excluded[direct_targets(perturbations, genes)] = True
    return excluded


def enrichment(magnitude, specific, q):
    """Complex-specific share among the top q% by magnitude / share among all entries."""
    n_specific_in_top = np.cumsum(specific[np.argsort(-magnitude, kind="stable")])
    n = np.maximum(1, np.floor(len(magnitude) * q / 100).astype(np.int64))
    return n_specific_in_top[n - 1] / n / specific.mean()


def curve(dataset, perturbations, genes, magnitude, usable, selected, members):
    """Fig 2e rows of one dataset."""
    eligible = [label for label, subunits in members.items()
                if perturbations.isin(subunits).sum() >= 3]
    rows = perturbations.isin(set().union(*(members[label] for label in eligible)))
    specific = np.zeros((rows.sum(), len(genes)), dtype=bool)
    for label in eligible:
        specific[np.ix_(perturbations[rows].isin(members[label]),
                        genes.isin(selected.get(label, ())))] = True
    ranked = usable[rows]
    values = enrichment(magnitude[rows][ranked], specific[ranked], QGRID)
    return pd.DataFrame({"dataset": dataset, "display_dataset": paper_name(dataset),
                         "q_percent": QGRID, "enrichment": values})


@np.errstate(divide="ignore")
def top5(dataset, perturbations, genes, magnitude, usable, selected, members):
    """S5c rows of one dataset, one per complex."""
    records = []
    for label, subunits in members.items():
        rows = perturbations.isin(subunits)
        columns = genes.isin(selected.get(label, ()))
        ranked = usable[rows]
        specific = np.broadcast_to(columns, ranked.shape)[ranked]
        value = np.nan
        if rows.sum() < 3:
            status = "ineligible_lt3_perturbations"
        elif not columns.any():
            status = "no_selected_complex_specific_genes"
        elif not ranked.any():
            status = "no_eligible_rejections"
        elif not specific.any():
            status = "no_complex_specific_rejections"
        else:
            value = enrichment(magnitude[rows][ranked], specific, 5.0)
            status = "available" if value > 0 else "available_zero_top5_share"
        records.append({"dataset": dataset, "display_dataset": paper_name(dataset),
                        "complex": label, "n_measured_perturbations": int(rows.sum()),
                        "status": status, "top5_enrichment": value,
                        "log2_top5_enrichment": np.log2(value)})
    return records


def main():
    args = parse_args()
    members = load_complexes(args.enrichment_dir / COMPLEX_TABLE)
    selected_genes = pd.read_csv(args.enrichment_dir / "complex_specific_genes.csv")
    effects = joblib.load(args.effect_pkl)

    curves, top5_rows = [], []
    for dataset in DATASETS:
        frame = effects[effect_key(dataset)]
        perturbations, genes = frame.index, frame.columns
        discoveries = load_discoveries(dataset, args.adjusted_rejection_dir).reindex(
            index=perturbations, columns=genes).to_numpy()
        usable = (discoveries == 1) & ~excluded_pairs(perturbations, genes, members)
        magnitude = np.abs(residual(frame))
        chosen = selected_genes[selected_genes["dataset"].eq(dataset)]
        selected = {label: set(names.astype(str))
                    for label, names in chosen.groupby("complex")["gene"]}

        curves.append(curve(dataset, perturbations, genes, magnitude, usable, selected, members))
        if dataset in TOP5_DATASETS:
            top5_rows += top5(dataset, perturbations, genes, magnitude, usable, selected, members)
        print(paper_name(dataset), flush=True)

    for name, table in [(CURVES_FILE, pd.concat(curves, ignore_index=True)),
                        (TOP5_FILE, pd.DataFrame(top5_rows))]:
        path = args.enrichment_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)


if __name__ == "__main__":
    main()
