"""Convert the KOLF2.1J genome-scale CRISPRi atlas to standardized origin format.

Reference: Nourreddine S, Doctor Y, Dailamy A, ... Mali P.
  "A genome-scale CRISPRi perturbation atlas of human induced pluripotent stem
   cells." Nature Biotechnology (2026). DOI 10.1038/s41587-026-03199-w
  Figshare+ DOI: 10.25452/figshare.plus.27261219

Input:  data/raw/KOLF2.1J_atlas/KOLF_Pan_Genome_QC_Filtered.h5ad
        (2,659,209 cells x 37,567 genes; CSC raw counts + layers['counts'];
         obs.gene_target = target gene symbol with 'NTC' for controls.)
Output: data/origin/Nourreddine-GW-ipsc.h5ad  (standard origin schema)

The source file is already a valid, QC-filtered AnnData with raw counts. To avoid
holding the ~63 GB expression matrix in memory, we stream-copy the /X group
on disk with h5py and only rebuild the (small) obs/var frames to the repo's
standard schema, then convert X to CSR with crispyx's bounded-memory streaming
converter (the atlas ships CSC, which makes crispyx's cell-streaming QC/normalize
~100x slower -- see util_csc_cache.py's docstring for the general form of this
issue):

  obs: perturbation (= gene_target; controls = 'NTC'), is_control, cell_type,
       dataset, batch (= channel), plus gRNA / chip for provenance.
  var: index = gene symbols, gene_ids = Ensembl IDs.

Usage:
    python 02_setup.py --dataset Nourreddine-GW-ipsc --force
"""

import argparse
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from anndata.io import write_elem

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATA_DIR  # noqa: E402
RAW_DIR = DATA_DIR / "raw" / "KOLF2.1J_atlas"
ORIGIN_DIR = DATA_DIR / "origin"

DEFAULT_INPUT = RAW_DIR / "KOLF_Pan_Genome_QC_Filtered.h5ad"
DATASET_NAME = "Nourreddine-GW-ipsc"
CELL_TYPE = "iPSC"
CONTROL_LABEL = "NTC"

# Source obs columns -> standard origin columns
GENE_TARGET_COL = "gene_target"   # target gene symbol (controls already 'NTC')
BATCH_COL = "channel"             # GEM-well / lane (90 categories) -> batch
CHIP_COL = "batch"               # coarse chip batch: ALPHA / BETA / GAMMA
GRNA_COL = "gRNA"                # individual guide id (provenance)
ENSEMBL_COL = "gene_target_ensembl_id"


def read_obs_column(obs_grp: h5py.Group, key: str) -> np.ndarray | None:
    """Read an obs column (categorical or plain array) into a decoded ndarray."""
    if key not in obs_grp:
        return None
    col = obs_grp[key]
    if isinstance(col, h5py.Group) and "codes" in col:  # pandas Categorical
        codes = col["codes"][:]
        cats = col["categories"][:]
        cats = np.array([c.decode() if isinstance(c, bytes) else c for c in cats])
        out = np.empty(codes.shape[0], dtype=object)
        valid = codes >= 0
        out[valid] = cats[codes[valid]]
        out[~valid] = None
        return out
    arr = col[:]
    if arr.dtype.kind in ("S", "O"):
        arr = np.array([a.decode() if isinstance(a, bytes) else a for a in arr])
    return arr


def read_index(grp: h5py.Group) -> np.ndarray:
    """Read the pandas index (_index) of an anndata obs/var group as strings."""
    idx_key = grp.attrs.get("_index", "_index")
    if isinstance(idx_key, bytes):
        idx_key = idx_key.decode()
    if idx_key not in grp:
        idx_key = "_index" if "_index" in grp else list(grp.keys())[0]
    raw = grp[idx_key][:]
    return np.array([x.decode() if isinstance(x, bytes) else x for x in raw])


def build_obs(src: h5py.File, dataset_name: str) -> pd.DataFrame:
    obs_grp = src["obs"]
    barcodes = read_index(obs_grp)
    n = barcodes.shape[0]

    gene_target = read_obs_column(obs_grp, GENE_TARGET_COL)
    if gene_target is None:
        log.error("Source obs missing '%s' column.", GENE_TARGET_COL)
        sys.exit(1)
    perturbation = np.array([str(g) for g in gene_target], dtype=object)

    is_control = perturbation == CONTROL_LABEL

    data = {
        "perturbation": perturbation,
        "is_control": is_control,
        "cell_type": np.array([CELL_TYPE] * n, dtype=object),
        "dataset": np.array([dataset_name] * n, dtype=object),
    }

    batch = read_obs_column(obs_grp, BATCH_COL)
    data["batch"] = (np.array([str(b) for b in batch], dtype=object)
                     if batch is not None else np.array(["unknown"] * n, dtype=object))

    chip = read_obs_column(obs_grp, CHIP_COL)
    if chip is not None:
        data["chip"] = np.array([str(c) for c in chip], dtype=object)

    grna = read_obs_column(obs_grp, GRNA_COL)
    if grna is not None:
        data["gRNA"] = np.array([str(g) for g in grna], dtype=object)

    ens = read_obs_column(obs_grp, ENSEMBL_COL)
    if ens is not None:
        data["gene_target_ensembl_id"] = np.array([str(e) for e in ens], dtype=object)

    obs = pd.DataFrame(data, index=pd.Index(barcodes, name=None))
    # Categoricals compress well and match other origins
    for c in ("perturbation", "cell_type", "dataset", "batch", "chip"):
        if c in obs:
            obs[c] = obs[c].astype("category")

    n_ctrl = int(is_control.sum())
    log.info("  obs: %d cells | %d perturbations (incl. control) | %d control cells (NTC)",
             n, obs["perturbation"].nunique(), n_ctrl)
    return obs


def build_var(src: h5py.File) -> pd.DataFrame:
    var_grp = src["var"]
    symbols = read_index(var_grp)
    var = pd.DataFrame(index=pd.Index(symbols, name=None))
    if "gene_ids" in var_grp:
        gids = var_grp["gene_ids"][:]
        var["gene_ids"] = np.array([g.decode() if isinstance(g, bytes) else g for g in gids])
    ft = read_obs_column(var_grp, "feature_types")
    if ft is not None:
        var["feature_types"] = np.array([str(x) for x in ft], dtype=object)
    log.info("  var: %d genes (index = gene symbols; gene_ids kept)", len(symbols))
    return var


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                    help="Source atlas .h5ad (default: pan-genome QC-filtered).")
    ap.add_argument("--dataset", default=DATASET_NAME,
                    help="Origin dataset name / output stem.")
    ap.add_argument("--force", action="store_true", help="Overwrite existing origin.")
    args = ap.parse_args()

    ORIGIN_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ORIGIN_DIR / f"{args.dataset}.h5ad"

    if out_path.exists() and not args.force:
        log.info("SKIP: %s already exists (%.1f GB). Use --force to overwrite.",
                 out_path.name, out_path.stat().st_size / 1e9)
        return

    if not args.input.exists():
        log.error("Input not found: %s", args.input)
        log.error("Run: python 01_download.py --dataset Nourreddine-GW-ipsc")
        sys.exit(1)

    log.info("Source: %s (%.1f GB)", args.input, args.input.stat().st_size / 1e9)

    with h5py.File(args.input, "r") as src:
        if "X" not in src or "obs" not in src or "var" not in src:
            log.error("Source is not a valid AnnData h5ad (missing X/obs/var).")
            sys.exit(1)

        log.info("Building obs / var frames ...")
        obs = build_obs(src, args.dataset)
        var = build_var(src)

        # sanity: X shape must match (n_obs, n_var)
        x = src["X"]
        xshape = x.attrs.get("shape", None)
        if xshape is not None:
            xshape = tuple(int(v) for v in xshape)
            if xshape != (len(obs), len(var)):
                log.error("X shape %s != (n_obs=%d, n_var=%d)", xshape, len(obs), len(var))
                sys.exit(1)
            log.info("  X shape %s (encoding=%s)", xshape,
                     x.attrs.get("encoding-type", "?"))

        tmp_path = out_path.with_suffix(".stage.h5ad")
        if tmp_path.exists():
            tmp_path.unlink()
        log.info("Writing staging origin -> %s (stream-copying /X on disk) ...", tmp_path.name)
        with h5py.File(tmp_path, "w") as dst:
            dst.attrs["encoding-type"] = "anndata"
            dst.attrs["encoding-version"] = "0.1.0"
            # Disk-to-disk copy of the expression matrix (no full RAM load).
            src.copy(src["X"], dst, name="X")
            write_elem(dst, "obs", obs)
            write_elem(dst, "var", var)

    # The atlas stores X as CSC. crispyx's cell-(row-)streaming QC/normalize is
    # O(total_nnz) per chunk on a backed CSC matrix (~100x slower at genome
    # scale). Convert to CSR (matching every other origin) via crispyx's
    # bounded-memory two-pass streaming converter. If the source is already
    # CSR, this is a no-op rename.
    import crispyx as cx

    fmt = cx.data.get_matrix_storage_format(str(tmp_path))
    if fmt == "csr":
        log.info("Staging X already CSR; finalizing without conversion.")
        tmp_path.replace(out_path)
    else:
        log.info("Converting X %s -> CSR (crispyx streaming, bounded memory) ...", fmt)
        csr_tmp = out_path.with_suffix(".csr.tmp.h5ad")
        if csr_tmp.exists():
            csr_tmp.unlink()
        cx.data.convert_to_csr(str(tmp_path), output_path=str(csr_tmp), verbose=False)
        tmp_path.unlink()
        csr_tmp.replace(out_path)
    log.info("Done: %s (%.1f GB, X=CSR)", out_path.name, out_path.stat().st_size / 1e9)


if __name__ == "__main__":
    main()
