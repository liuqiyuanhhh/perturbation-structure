"""Step 03: QC, normalization and pooled Wilcoxon DE.

    data/origin/<dataset>.h5ad -> data/sc/sc_<dataset>.h5ad
                                  data/de/DE_<dataset>_wilcoxon.h5ad

For each dataset:
  0. check perturbation labels for truncation defects;
  1. QC (datasets.QC): drop cells with < 100 detected genes, then perturbations
     carried by < 20 remaining cells (control cells are always kept), then genes
     detected in < 100 kept cells;
  1b. merge gene symbols that differ only in letter case (sum of raw counts);
  2. store each cell's total count as obs['library_size'], normalize counts to
     1e4 per cell and apply log1p (counts = expm1(X) * library_size / 1e4);
  3. Wilcoxon rank-sum test of each perturbation against control cells
     (Benjamini-Hochberg adjustment within each perturbation);
  4. add the per-cell columns perturbation_type, self_pert_lfc and
     self_pert_zscore.

    python 03_sc_preprocess.py --dataset Replogle-E-k562
    python 03_sc_preprocess.py --dataset all
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import anndata as ad
import crispyx as cx
import h5py
import numpy as np
import pandas as pd

import datasets
from paths import DE_DIR, ORIGIN_DIR, SC_DIR, TMP_DIR, de_file, sc_file
from util_csc_cache import ensure_csc
from util_de import check_crispyx, set_scratch_dir
from util_gene_symbols import collapse_duplicate_gene_symbols, has_case_duplicate_gene_symbols
from util_validate_perturbation_labels import validate_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SC_CSC_DIR = TMP_DIR / "sc_csc"  # column-major copies for gene-chunked DE
EXTRA_OBS_COLS = ("perturbation_type", "self_pert_lfc", "self_pert_zscore")


def _is_log1p(path: Path) -> bool:
    """True if X already holds non-integer (log-normalized) values."""
    with h5py.File(path, "r") as f:
        x = f["X"]
        if isinstance(x, h5py.Group):
            sample = x["data"][: min(100_000, x["data"].shape[0])]
        else:
            sample = x[: min(50, x.shape[0]), :].ravel()
    nz = sample[sample != 0]
    return len(nz) > 0 and np.mean(np.abs(nz - np.round(nz)) < 1e-6) < 0.5


def add_library_size(path: Path) -> None:
    """Store each cell's total count (over the genes in the file) as obs['library_size'].

    Called on the QC-filtered raw counts right before normalization, so that
    counts = expm1(X) * library_size / 1e4 can be recovered from the normalized file.
    """
    with h5py.File(path, "a") as f:
        X = f["X"]
        n = X.attrs["shape"][0] if isinstance(X, h5py.Group) else X.shape[0]
        sums = np.zeros(n, dtype=np.float64)
        if isinstance(X, h5py.Group) and X.attrs.get("encoding-type") in ("csr_matrix", b"csr_matrix"):
            indptr = X["indptr"][:]
            for s in range(0, n, 20_000):
                e = min(n, s + 20_000)
                data = X["data"][indptr[s]:indptr[e]]
                row = np.repeat(np.arange(e - s), np.diff(indptr[s:e + 1]))
                sums[s:e] = np.bincount(row, weights=data, minlength=e - s)
        elif isinstance(X, h5py.Group):  # CSC
            indptr, m = X["indptr"][:], X.attrs["shape"][1]
            for s in range(0, m, 2_000):
                e = min(m, s + 2_000)
                np.add.at(sums, X["indices"][indptr[s]:indptr[e]], X["data"][indptr[s]:indptr[e]])
        else:  # dense
            for s in range(0, n, 20_000):
                sums[s:min(n, s + 20_000)] = X[s:min(n, s + 20_000)].sum(axis=1)
        obs = f["obs"]
        if "library_size" in obs:
            del obs["library_size"]
        ds = obs.create_dataset("library_size", data=sums.astype(np.float64))
        ds.attrs["encoding-type"] = "array"
        ds.attrs["encoding-version"] = "0.2.0"
        order = [c.decode() if isinstance(c, bytes) else str(c) for c in obs.attrs.get("column-order", [])]
        if "library_size" not in order:
            obs.attrs.create("column-order", np.array(order + ["library_size"], dtype=object),
                             dtype=h5py.string_dtype())


def append_obs_extra_cols(sc_path: Path, pert_col: str, control: str, de_path: Path) -> None:
    """Add perturbation_type, self_pert_lfc and self_pert_zscore to obs in place.

    self_pert_lfc / self_pert_zscore are the log-fold change and Wilcoxon z-score of
    the targeted gene in its own perturbation-vs-control test (NaN for control cells
    or when the target gene is not measured). A categorical perturbation_type from the
    source file (scPerturb: "CRISPR") is relabelled; other existing columns are kept. New columns are
    written with AnnData
    encoding attributes and registered in obs's column order, so anndata/scanpy
    read them.
    """
    with h5py.File(sc_path, "a") as hf:
        obs = hf["obs"]
        col = obs[pert_col]
        if isinstance(col, h5py.Group):  # categorical
            perts = col["categories"][:].astype(str)[col["codes"][:]]
        else:
            perts = col[:].astype(str)
        with h5py.File(de_path, "r") as de:
            groups = de["obs"][de["obs"].attrs["_index"]].asstr()[:]
            de_genes = de["var"][de["var"].attrs["_index"]].asstr()[:]
            lfc_df = pd.DataFrame(de["layers/logfoldchanges"][:], index=groups, columns=de_genes)
            z_df = pd.DataFrame(de["layers/z_score"][:], index=groups, columns=de_genes)
        genes = set(lfc_df.columns)

        def lookup(p: str, df: pd.DataFrame) -> float:
            return float("nan") if p == control or p not in df.index or p not in genes else float(df.loc[p, p])

        values = {
            "perturbation_type": (np.array([datasets.PERTURBATION_TYPE] * len(perts), dtype=object), "string-array"),
            "self_pert_lfc": (np.array([lookup(p, lfc_df) for p in perts], dtype=np.float32), "array"),
            "self_pert_zscore": (np.array([lookup(p, z_df) for p in perts], dtype=np.float32), "array"),
        }
        for name, (arr, enc) in values.items():
            if name in obs:
                if name == "perturbation_type" and isinstance(obs[name], h5py.Group):
                    # source column (scPerturb: categorical "CRISPR"): relabel its single category
                    cats = obs[name]["categories"]
                    attrs = dict(cats.attrs)
                    del obs[name]["categories"]
                    new = obs[name].create_dataset("categories", data=np.array([datasets.PERTURBATION_TYPE], dtype=object),
                                                   dtype=h5py.string_dtype())
                    for k, v in attrs.items():
                        new.attrs[k] = v
                continue
            ds = obs.create_dataset(name, data=arr, dtype=h5py.string_dtype() if enc == "string-array" else arr.dtype)
            ds.attrs["encoding-type"] = enc
            ds.attrs["encoding-version"] = "0.2.0"
        order = [c.decode() if isinstance(c, bytes) else str(c) for c in obs.attrs.get("column-order", [])]
        order += [c for c in EXTRA_OBS_COLS if c not in order and "encoding-type" in obs[c].attrs]
        obs.attrs.create("column-order", np.array(order, dtype=object), dtype=h5py.string_dtype())


def process_dataset(name: str, force: bool) -> None:
    cfg = datasets.get(name)
    pert_col, control = cfg["pert_col"], cfg["control"]
    origin_path = ORIGIN_DIR / f"{name}.h5ad"
    sc_path = SC_DIR / sc_file(name)
    de_h5ad = DE_DIR / de_file(name, batch_corrected=False)

    if not origin_path.exists():
        raise FileNotFoundError(f"{name}: {origin_path} not found (run 02_setup.py)")
    if sc_path.exists() and de_h5ad.exists() and not force:
        log.info("SKIP %s: outputs exist (pass --force to rebuild)", name)
        return
    for p in (sc_path, de_h5ad):
        p.unlink(missing_ok=True)

    # 0. perturbation-label check (obs/var only)
    validation = validate_dataset(origin_path)
    if validation.failed:
        raise ValueError(f"{name}: {len(validation.merges)} perturbation labels look truncated/merged: "
                         f"{dict(list(validation.merges.items())[:5])}")
    for w in validation.warnings:
        log.warning("%s: label check: %s", name, w)

    # 1. QC
    qc = cx.qc.quality_control_summary(
        origin_path, perturbation_column=pert_col, control_label=control,
        min_genes=datasets.QC["min_genes"],
        min_cells_per_gene=datasets.QC["min_cells_per_gene"],
        min_cells_per_perturbation=datasets.QC["min_cells_per_pert"],
    )
    log.info("%s: QC keeps %d cells, %d genes", name, int(qc.cell_mask.sum()), int(qc.gene_mask.sum()))
    SC_DIR.mkdir(parents=True, exist_ok=True)
    cx.qc.write_filtered_subset(origin_path, cell_mask=qc.cell_mask, gene_mask=qc.gene_mask, output_path=sc_path)

    # 1b. gene symbols differing only in case
    peek = ad.read_h5ad(sc_path, backed="r")
    gene_names = list(peek.var_names)
    peek.file.close()
    if has_case_duplicate_gene_symbols(gene_names):
        full = ad.read_h5ad(sc_path)
        X, var, summary = collapse_duplicate_gene_symbols(full.X, full.var_names, full.var,
                                                          log1p_space=_is_log1p(origin_path))
        tmp = sc_path.with_suffix(".genedup.tmp.h5ad")
        ad.AnnData(X=X, obs=full.obs, var=var, uns=full.uns).write_h5ad(tmp)
        tmp.rename(sc_path)
        log.info("%s: merged case-duplicate gene symbols %s", name, summary["collapsed_symbols"])

    # 2. normalization (library sizes stored first, so counts can be recovered)
    if not _is_log1p(origin_path):
        add_library_size(sc_path)
        tmp = sc_path.with_suffix(".tmp.h5ad")
        cx.pp.normalize_total_log1p(sc_path, output_path=tmp, target_sum=1e4)
        tmp.rename(sc_path)

    # 3. pooled Wilcoxon DE
    de_result = cx.de.wilcoxon_test(
        ensure_csc(name, sc_path, SC_CSC_DIR),
        perturbation_column=pert_col, control_label=control,
        corr_method="benjamini-hochberg", output_path=de_h5ad,
    )
    log.info("%s: DE tested %d perturbations", name, len(de_result.groups))

    # 4. per-cell columns
    append_obs_extra_cols(sc_path, pert_col, control, de_h5ad)
    log.info("%s: done -> %s", name, sc_path)


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
