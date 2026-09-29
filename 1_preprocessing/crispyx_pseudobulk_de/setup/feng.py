"""Convert Feng et al. CRISPRi screen CSVs to standardized origin format.

Source: three parallel screens (raw CSV/TSV pairs from downloaders/feng.sh):
  GenomeWideScreen_FitnessGenes     -> Feng-gwsf
  GenomeWideScreen_NonFitnessGenes  -> Feng-gwsnf
  TargetedScreen                    -> Feng-ts

Each screen ships as ``{name}_RNA-UMI-Counts.csv`` (genes x cells, raw UMI
counts) + ``{name}_Cell-Metadata.tsv`` (per-cell metadata, including the raw
``Guide_Call`` guide identifier).

**Perturbation-label parsing must never split ``Guide_Call`` on the first
underscore.** The guide library encodes hyphenated HGNC symbols with
underscores (``HLA-B`` -> ``HLA_B``), so a naive first-underscore split
truncates at the first hyphen and merges distinct genes sharing a prefix
(e.g. collapsing ``HLA-B``/5 HLA genes and a readthrough locus into one
shared label per dataset). This script instead strips the trailing guide
spacer sequence (``SPACER``, an 18-25bp ACGT run) and restores the hyphen,
so a from-scratch rebuild never produces the merged label. The pre-strip
value is still kept as ``perturbation_raw_label`` (this project's standard
audit convention), holding the **naive first-underscore split** for
comparison (not a copy of ``Guide_Call`` itself, which is separately
preserved verbatim).

**Memory note.** These CSVs are large (the resulting origin files are 47-170 GB)
and each screen's full count matrix is read into memory with ``pandas.read_csv``.
This builder has been checked on small synthetic input and against the structure of
the origin files used in the paper, but has not been re-run end to end on the
full source CSVs.

**Duplicate gene symbols in the CellRanger reference.** The reference this
screen's ``ensembl_id:gene_name`` var strings were annotated against assigns
10 HGNC symbols (``TBCE``, ``LINC01238``, ``CYB561D2``, ``MATR3``,
``LINC01505``, ``HSPA14``, ``GOLGA8M``, ``GGT1``, ``ARMCX5-GPRASP2``,
``TMSB15B``) to two distinct Ensembl gene models each -- a known artifact of
that reference, not a parsing bug here. Raw UMI counts are collapsed
(summed) across each duplicate symbol via
``util_gene_symbols.collapse_duplicate_gene_symbols`` before the origin
``AnnData`` is built, so ``var_names`` are unique from this point forward in
the pipeline (matching every other dataset's convention) and no downstream
step needs to special-case it.

Usage:
    python 02_setup.py --dataset Feng-gwsf --force
    python setup/feng.py --force  # all three
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from util_gene_symbols import collapse_duplicate_gene_symbols  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger(__name__)

from paths import DATA_DIR  # noqa: E402
RAW_DIR = DATA_DIR / "raw" / "feng"
ORIGIN_DIR = DATA_DIR / "origin"

# source screen name -> (origin dataset name, index-cleanup prefix/suffix)
SCREENS = {
    "GenomeWideScreen_FitnessGenes": ("Feng-gwsf", ("MP-", None)),
    "GenomeWideScreen_NonFitnessGenes": ("Feng-gwsnf", ("MP-", None)),
    "TargetedScreen": ("Feng-ts", ("PC-", "-D3")),
}

# Trailing guide spacer sequence appended to Guide_Call; stripping it and
# restoring the hyphen recovers the true HGNC symbol (see module docstring).
SPACER = re.compile(r"_[ACGT]{18,25}$")


def guide_to_target(guide_call: str) -> str | None:
    """Corrected Guide_Call -> gene-symbol rule. Returns None if unparseable."""
    if guide_call == "unassigned" or guide_call.startswith("NonTarget"):
        return "control"
    if not SPACER.search(guide_call):
        return None
    return SPACER.sub("", guide_call).replace("_", "-")


def naive_first_underscore_split(guide_call: str) -> str:
    """The ORIGINAL (buggy) rule -- kept only to populate the audit column
    perturbation_raw_label, documenting what the naive pipeline would have
    produced. Never used for the promoted `perturbation` column."""
    if guide_call == "unassigned":
        return "control"
    if guide_call.startswith("NonTarget"):
        return "control"
    return guide_call.split("_", 1)[0]


def build_screen(screen_name: str, dataset_name: str, prefix: str, suffix: str | None) -> ad.AnnData:
    counts_path = RAW_DIR / f"{screen_name}_RNA-UMI-Counts.csv"
    meta_path = RAW_DIR / f"{screen_name}_Cell-Metadata.tsv"
    if not counts_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"{dataset_name}: expected {counts_path.name} and {meta_path.name} "
            f"under {RAW_DIR} (run downloaders/feng.sh first)"
        )

    log.info("Reading %s (genes x cells) ...", counts_path.name)
    # Source CSV is genes (rows) x cells (columns); transpose to cells x genes.
    counts = pd.read_csv(counts_path, index_col=0).T

    log.info("Reading %s ...", meta_path.name)
    meta = pd.read_csv(meta_path, sep="\t", index_col=0)
    meta.index = meta.index.to_series().str.replace(prefix, "", regex=False)
    if suffix:
        meta.index = meta.index.to_series().str.replace(suffix, "", regex=False)

    missing = meta.index.difference(counts.index)
    if len(missing) > 0:
        raise ValueError(
            f"{dataset_name}: {len(missing)} metadata barcodes have no matching "
            f"count-matrix row after index cleanup (prefix={prefix!r}, "
            f"suffix={suffix!r}) -- e.g. {list(missing[:5])}"
        )
    counts = counts.loc[meta.index]

    var_split = counts.columns.to_series().str.split(":", n=1, expand=True)
    if var_split.shape[1] != 2:
        raise ValueError(
            f"{dataset_name}: expected var names shaped 'ensembl_id:gene_name', "
            f"got e.g. {counts.columns[0]!r}"
        )
    var = pd.DataFrame(
        {"ensembl_id": var_split[0].to_numpy(), "info": "Gene-Expression"},
        index=pd.Index(var_split[1].to_numpy(), name=None),
    )

    # Raw counts are additive, so collapsing (summing) here -- before the
    # AnnData is ever built -- is the scientifically correct combine for the
    # CellRanger reference's known duplicate-symbol pairs (see module
    # docstring). Every downstream var_names consumer then sees a unique axis.
    counts_matrix, var, collapse_info = collapse_duplicate_gene_symbols(
        counts.to_numpy(dtype=np.float32), var.index, var, log1p_space=False
    )
    if collapse_info["n_collapsed_symbols"]:
        log.info(
            "%s: collapsed %d duplicate gene-symbol groups (%d features removed): %s",
            dataset_name, collapse_info["n_collapsed_symbols"],
            collapse_info["n_removed_features"], collapse_info["collapsed_symbols"],
        )

    if "Guide_Call" not in meta.columns:
        raise ValueError(f"{dataset_name}: Cell-Metadata is missing 'Guide_Call'")
    guide_call = meta["Guide_Call"].astype(str)

    perturbation = guide_call.map(guide_to_target)
    unparsed = perturbation.isna()
    if unparsed.any():
        examples = guide_call[unparsed].unique()[:5].tolist()
        raise ValueError(
            f"{dataset_name}: {int(unparsed.sum())} Guide_Call values didn't match "
            f"the control pattern or the SPACER shape -- e.g. {examples}. "
            f"Refusing to silently pass an unparsed string through as a label."
        )

    obs = meta.copy()
    obs["perturbation"] = perturbation.to_numpy()
    obs["perturbation_raw_label"] = guide_call.map(naive_first_underscore_split).to_numpy()
    if "Cell_Line" in obs.columns:
        obs["donor_line"] = obs["Cell_Line"]

    # Donor line is not nested inside Batch (each 10x inlet pools cells from
    # many lines) -- name must match datasets.strat_col().
    if {"Batch", "donor_line"} <= set(obs.columns):
        obs["Batch_x_donor_line"] = (
            obs["Batch"].astype(str) + "_x_" + obs["donor_line"].astype(str)
        )

    for c in ("perturbation", "perturbation_raw_label", "donor_line", "Cell_Line",
              "Batch", "Batch_x_donor_line"):
        if c in obs.columns:
            obs[c] = obs[c].astype("category")

    n_ctrl = int((obs["perturbation"] == "control").sum())
    log.info(
        "%s: %d cells, %d genes, %d perturbations (incl. control), %d control cells",
        dataset_name, counts_matrix.shape[0], counts_matrix.shape[1],
        obs["perturbation"].nunique(), n_ctrl,
    )

    return ad.AnnData(X=counts_matrix, obs=obs, var=var)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--screen", choices=[*SCREENS, "all"], default="all",
                    help="Which source screen to build (default: all three).")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    targets = SCREENS if args.screen == "all" else {args.screen: SCREENS[args.screen]}
    for screen_name, (dataset_name, (prefix, suffix)) in targets.items():
        out_path = ORIGIN_DIR / f"{dataset_name}.h5ad"
        if out_path.exists() and not args.force:
            log.info("SKIP %s: origin already exists (pass --force to overwrite)", dataset_name)
            continue
        adata = build_screen(screen_name, dataset_name, prefix, suffix)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(out_path)
        log.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
