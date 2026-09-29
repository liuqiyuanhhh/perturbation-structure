"""Build Nadig-HEPG2/Nadig-JURKAT origin files from raw GEO downloads.

GEO GSE264667's deposited per-cell-line files already carry this project's
expected native obs schema verbatim (pert_col="gene", batch_col="gem_group")
-- confirmed against the raw downloads. The one content transformation
needed is on the *var* axis: the raw files key var_names on Ensembl
``gene_id`` (unique) and carry the HGNC symbol in a separate ``gene_name``
column that has one duplicate pair (``HSPA14`` -> 2 Ensembl IDs, the same
kind of reference-annotation artifact as Feng's -- see setup/feng.py's
docstring). All other datasets use var_names as the gene-symbol axis, so
this script promotes ``gene_name`` to ``var_names``: raw counts are
collapsed (summed) across the duplicate symbol via
``util_gene_symbols.collapse_duplicate_gene_symbols``, and the original
Ensembl id is kept as ``var["ensembl_id"]`` (mirroring Feng's convention).

Usage:
    python 02_setup.py --dataset Nadig-HEPG2 --force
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import anndata as ad
import numpy as np

from util_helpers import validate_schema

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from util_gene_symbols import collapse_duplicate_gene_symbols  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger(__name__)

from paths import DATA_DIR  # noqa: E402
RAW_DIR = DATA_DIR / "raw"
ORIGIN_DIR = DATA_DIR / "origin"

DATASETS = ("Nadig-HEPG2", "Nadig-JURKAT")
PERT_COL = "gene"
CONTROL_LABEL = "non-targeting"
GENE_NAME_COL = "gene_name"


def build_dataset(name: str, raw_path: Path) -> ad.AnnData:
    validate_schema(raw_path, pert_col=PERT_COL, control_label=CONTROL_LABEL)

    adata = ad.read_h5ad(raw_path)
    if GENE_NAME_COL not in adata.var.columns:
        raise ValueError(f"{name}: var is missing expected column {GENE_NAME_COL!r}")

    var = adata.var.copy()
    var["ensembl_id"] = adata.var_names.astype(str).to_numpy()
    gene_symbols = var.pop(GENE_NAME_COL).astype(str)

    X, var_collapsed, collapse_info = collapse_duplicate_gene_symbols(
        np.asarray(adata.X, dtype=np.float32), gene_symbols, var, log1p_space=False
    )
    if collapse_info["n_collapsed_symbols"]:
        log.info(
            "%s: collapsed %d duplicate gene-symbol groups (%d features removed): %s",
            name, collapse_info["n_collapsed_symbols"],
            collapse_info["n_removed_features"], collapse_info["collapsed_symbols"],
        )

    return ad.AnnData(X=X, obs=adata.obs, var=var_collapsed)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    for name in DATASETS:
        raw_path = RAW_DIR / f"{name}.h5ad"
        origin_path = ORIGIN_DIR / f"{name}.h5ad"
        if origin_path.exists() and not args.force:
            log.info("SKIP %s: origin already exists (pass --force to overwrite)", name)
            continue
        if not raw_path.exists():
            log.warning("SKIP %s: raw input not found: %s (run downloaders/nadig.sh)",
                        name, raw_path)
            continue
        log.info("Building %s ...", name)
        adata = build_dataset(name, raw_path)
        origin_path.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(origin_path)
        log.info("Wrote %s: %d cells, %d genes", origin_path, adata.n_obs, adata.n_vars)


if __name__ == "__main__":
    main()
