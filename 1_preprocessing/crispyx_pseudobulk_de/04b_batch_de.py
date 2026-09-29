"""Step 04b: batch-stratified Wilcoxon DE (van Elteren test).

    data/sc/sc_<dataset>.h5ad -> data/de/DE_<dataset>_wilcoxon_batch_corrected.h5ad

Each perturbation is compared with control cells within batches
(datasets.strat_col: the batch column, or the joint batch x donor key for the Feng
screens). Log-fold changes and detection fractions stay pooled; p-values, U
statistics and z-scores are stratified.

    python 04b_batch_de.py --dataset Replogle-E-k562
    python 04b_batch_de.py --dataset all
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import anndata as ad
import crispyx as cx
import numpy as np

import datasets
from paths import DE_DIR, SC_DIR, TMP_DIR, de_file, sc_file
from util_csc_cache import ensure_csc
from util_de import check_crispyx, set_scratch_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SC_CSC_DIR = TMP_DIR / "sc_csc"


def validate_batch_metadata(path: Path, pert_col: str, batch_col: str, control: str) -> dict:
    """Require >= 2 batches and, for every perturbation, a batch shared with control cells."""
    backed = ad.read_h5ad(path, backed="r")
    try:
        obs = backed.obs[[pert_col, batch_col]]
    finally:
        backed.file.close()
    valid = obs.notna().all(axis=1).to_numpy()
    labels = obs[pert_col].astype(str).to_numpy()[valid]
    batches = obs[batch_col].astype(str).to_numpy()[valid]
    if len(set(batches)) < 2:
        raise ValueError(f"{path.name}: batch column {batch_col!r} has fewer than two levels")
    control_batches = set(batches[labels == control])
    if not control_batches:
        raise ValueError(f"{path.name}: control label {control!r} not found in {pert_col!r}")
    shared = {p: len(set(batches[labels == p]) & control_batches) for p in set(labels) - {control}}
    no_control = [p for p, n in shared.items() if n == 0]
    if no_control:
        raise ValueError(f"{path.name}: {len(no_control)} perturbations share no batch with controls: {no_control[:10]}")
    return {"n_cells": int(valid.sum()), "n_batches": len(set(batches)),
            "n_perturbations": len(shared), "min_shared_batches": int(min(shared.values())),
            "median_shared_batches": float(np.median(list(shared.values())))}


def process_dataset(name: str, force: bool) -> None:
    cfg = datasets.get(name)
    sc_path = SC_DIR / sc_file(name)
    out_h5ad = DE_DIR / de_file(name)
    if not sc_path.exists():
        raise FileNotFoundError(f"{name}: {sc_path} not found (run 03_sc_preprocess.py)")
    if out_h5ad.exists() and not force:
        log.info("SKIP %s: outputs exist (pass --force to rebuild)", name)
        return
    strat = datasets.strat_col(name)
    info = validate_batch_metadata(sc_path, cfg["pert_col"], strat, cfg["control"])
    log.info("%s: stratified by %r: %s", name, strat, info)
    cx.de.wilcoxon_test(
        ensure_csc(name, sc_path, SC_CSC_DIR),
        perturbation_column=cfg["pert_col"], control_label=cfg["control"],
        batch_column=strat, corr_method="benjamini-hochberg",
        output_path=out_h5ad, verbose=1, force=force,
    )
    log.info("%s: done -> %s", name, out_h5ad)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="dataset name, or 'all'")
    ap.add_argument("--force", action="store_true", help="rebuild existing outputs")
    args = ap.parse_args(argv)
    check_crispyx()
    set_scratch_dir()
    DE_DIR.mkdir(parents=True, exist_ok=True)
    failed = []
    for name in datasets.resolve(args.dataset):
        try:
            process_dataset(name, args.force)
        except Exception:
            log.exception("FAILED: %s", name)
            failed.append(name)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
