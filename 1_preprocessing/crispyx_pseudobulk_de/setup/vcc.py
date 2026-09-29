"""Convert the full VCC 2025 release to standardized origin format.

Step 02:
  data/raw/vcc_data/2025/{train,validation,test}/adata_*.h5ad
      -> data/origin/VCC.h5ad

The output has the standard origin schema:
  - X: raw counts
  - obs["perturbation"]: target gene label
  - obs["is_control"]: perturbation == "non-targeting"
  - obs["cell_type"]: "hESC"
  - obs["dataset"]: "VCC"

It also records the release split of each cell in obs["vcc_split"].

Usage:
    python 02_setup.py --dataset VCC --force
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable

import anndata as ad
import h5py
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATA_DIR  # noqa: E402
RAW_DIR = DATA_DIR / "raw" / "vcc_data" / "2025"
ORIGIN_PATH = DATA_DIR / "origin" / "VCC.h5ad"

SPLIT_FILES = {
    "train": "adata_Training.h5ad",
    "validation": "adata_Validation.h5ad",
    "test": "adata_Test.h5ad",
}
COUNT_FILES = {
    "train": "pert_counts_Training.csv",
    "validation": "pert_counts_Validation.csv",
    "test": "pert_counts_Test.csv",
}


def split_paths(raw_dir: Path, splits: Iterable[str]) -> dict[str, Path]:
    """Return h5ad paths for requested VCC splits."""
    paths = {}
    for split in splits:
        paths[split] = raw_dir / split / SPLIT_FILES[split]
    return paths


def _standardize_split(adata: ad.AnnData, split: str) -> ad.AnnData:
    """Return a VCC split with the standard obs columns."""
    if "target_gene" not in adata.obs.columns:
        raise ValueError(f"VCC {split} split is missing obs['target_gene']")

    out = adata.copy()
    out.obs_names = [f"{split}__{idx}" for idx in out.obs_names.astype(str)]
    out.obs["vcc_split"] = split
    out.obs["perturbation"] = out.obs["target_gene"].astype(str)
    out.obs["is_control"] = out.obs["perturbation"] == "non-targeting"
    out.obs["cell_type"] = "hESC"
    out.obs["dataset"] = "VCC"
    return out


def load_standardized_splits(paths: dict[str, Path]) -> list[ad.AnnData]:
    """Load and standardize requested VCC split h5ad files."""
    adatas = []
    first_var_names = None
    for split, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Required VCC {split} file not found: {path}\n"
                "Run: python 01_download.py --dataset VCC"
            )

        log.info("Loading %s split: %s", split, path)
        adata = ad.read_h5ad(path)
        log.info("  shape=%s obs=%s var=%s", adata.shape, list(adata.obs.columns),
                 list(adata.var.columns))

        if first_var_names is None:
            first_var_names = adata.var_names.astype(str).tolist()
        elif adata.var_names.astype(str).tolist() != first_var_names:
            raise ValueError(f"VCC {split} var_names do not match the first split")

        adatas.append(_standardize_split(adata, split))
    return adatas


def build_vcc_adata(paths: dict[str, Path]) -> ad.AnnData:
    """Build the full-release VCC AnnData from standardized split files."""
    adatas = load_standardized_splits(paths)
    combined = ad.concat(
        adatas,
        axis=0,
        join="inner",
        merge="same",
        label=None,
        index_unique=None,
    )
    combined.obs_names_make_unique()
    combined.uns["vcc_2025_source"] = {
        "bucket_prefix": (
            "gs://arc-institute-virtual-cell-atlas/"
            "virtual-cell-challenge/2025"
        ),
        "splits": {
            split: {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "n_cells": int((combined.obs["vcc_split"] == split).sum()),
                "n_perturbations": int(
                    combined.obs.loc[
                        combined.obs["vcc_split"] == split, "perturbation"
                    ].nunique()
                ),
            }
            for split, path in paths.items()
        },
    }
    return combined


def _read_count_summaries(raw_dir: Path, splits: Iterable[str]) -> dict:
    """Read released pert-count CSV summaries when present."""
    out = {}
    for split in splits:
        path = raw_dir / split / COUNT_FILES[split]
        if path.exists():
            counts = pd.read_csv(path)
            out[split] = {
                "path": str(path),
                "rows": int(len(counts)),
                "columns": list(counts.columns),
            }
    return out


def write_vcc_origin(
    raw_dir: Path = RAW_DIR,
    output_path: Path = ORIGIN_PATH,
    splits: Iterable[str] = ("train", "validation", "test"),
    force: bool = False,
) -> Path:
    """Write the standardized full-release VCC origin h5ad."""
    splits = tuple(splits)
    if output_path.exists() and not force:
        raise FileExistsError(
            f"Output already exists: {output_path}\n"
            "Pass --force to overwrite it."
        )

    paths = split_paths(raw_dir, splits)
    adata = build_vcc_adata(paths)
    adata.uns["vcc_2025_source"]["pert_counts"] = _read_count_summaries(
        raw_dir, splits
    )

    n_ctrl = int(adata.obs["is_control"].sum())
    n_perts = int(adata.obs["perturbation"].nunique())
    log.info("Combined VCC shape: %s", adata.shape)
    log.info("Control cells (non-targeting): %d", n_ctrl)
    log.info("Unique perturbations: %d", n_perts)
    log.info("Cells by split: %s", adata.obs["vcc_split"].value_counts().to_dict())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Writing %s ...", output_path)
    adata.write_h5ad(output_path)

    with h5py.File(output_path, "r") as f:
        enc = f["X"].attrs.get("encoding-type", b"unknown")
        if isinstance(enc, bytes):
            enc = enc.decode()
        log.info("Validated output X encoding: %s", enc)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert the full VCC 2025 release to data/origin/VCC.h5ad"
    )
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output", type=Path, default=ORIGIN_PATH)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=tuple(SPLIT_FILES),
        default=("train", "validation", "test"),
        help="VCC release splits to include; default is all splits",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing output h5ad"
    )
    args = parser.parse_args()

    write_vcc_origin(
        raw_dir=args.raw_dir,
        output_path=args.output,
        splits=args.splits,
        force=args.force,
    )
    log.info("Done: %s", args.output)


if __name__ == "__main__":
    main()
