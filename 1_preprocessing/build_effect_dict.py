#!/usr/bin/env python3
"""Effect dict {dataset: perturbations x genes X} of the prediction benchmark, on the
QC-passing perturbations and primary genes (Fig 1d, 2e, S2, S5c).
Reads MOMENTS_DIR and the prediction QC tables (--qc-dir); writes EFFECT_DICT."""

import argparse
from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd

from utils.effects import axis_names, load_quality_audit
from utils.paths import (
    DATASET_COMPONENTS,
    EFFECT_DICT,
    MOMENTS_DIR,
    PAPER_PREDICTION_QC_DIR,
    effect_key,
)
from utils.qc import load_primary_outcome_lists


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--moments-dir", type=Path, default=MOMENTS_DIR)
    parser.add_argument("--qc-dir", type=Path, default=PAPER_PREDICTION_QC_DIR,
                        help="prediction QC tables; default: the paper tables")
    parser.add_argument("--output", type=Path, default=EFFECT_DICT)
    return parser.parse_args()


def read_effects(path, genes, perturbations):
    """X of one moments file on `genes` and `perturbations`, both axes sorted."""
    with h5py.File(path, "r") as handle:
        names = np.array(axis_names(handle, "obs"))
        file_genes = np.array(axis_names(handle, "var"))
        first = ~pd.Index(file_genes).duplicated()  # a repeated symbol keeps its first column
        columns = np.flatnonzero(np.isin(file_genes, list(genes)) & first)
        rows = np.flatnonzero(np.isin(names, list(perturbations)))
        values = handle["X"][:][np.ix_(rows, columns)].astype(np.float32)
    frame = pd.DataFrame(values, index=pd.Index(names[rows], name="perturbation"),
                         columns=file_genes[columns])
    return frame.sort_index(axis=0).sort_index(axis=1)


def merge_parts(frames):
    """Feng-GW: union of perturbations on the shared genes, a perturbation in both parts averaged."""
    genes = sorted(set.intersection(*(set(frame.columns) for frame in frames)))
    merged = pd.concat([frame[genes] for frame in frames])
    if merged.index.duplicated().any():
        merged = merged.groupby(level=0).mean()
    merged = merged.astype("float32")
    merged.index.name = "perturbation"
    return merged.sort_index(axis=0).sort_index(axis=1)


def main():
    args = parse_args()
    primary = load_primary_outcome_lists(args.qc_dir / "outcome_expression_qc.csv.gz")
    _, passing = load_quality_audit(args.qc_dir / "perturbation_quality_filter.csv.gz")
    qc_names = {name.lower(): name for name in primary}

    effect = {}
    for dataset, components in DATASET_COMPONENTS.items():
        qc = qc_names[dataset.lower()]
        frames = [read_effects(args.moments_dir / f"{component}_moments.h5ad", primary[qc], passing[qc])
                  for component in components]
        effect[effect_key(dataset)] = frames[0] if len(frames) == 1 else merge_parts(frames)

    effect = {key: effect[key] for key in sorted(effect)}
    for key, frame in effect.items():
        # a perturbation with no estimable effect on the panel cannot enter any analysis
        all_nan = np.isnan(frame.values).all(axis=1)
        if all_nan.any():
            effect[key] = frame.loc[~all_nan]
        print(f"{key}: {effect[key].shape[0]} perturbations x {effect[key].shape[1]} genes",
              flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(effect, args.output, compress=0)


if __name__ == "__main__":
    main()
