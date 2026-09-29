"""Convert GSE295214 genome-scale CRISPRi Perturb-seq (hESC) to standardized origin format.

Reference: Pan X, Saunders R, Replogle J, Weissman J, Zhuang X.
  "Global cell-state and gene-program representations reveal conserved and
   context-specific perturbation responses of cells"
  GEO: GSE295214  BioProject: PRJNA1252779

Dataset: 85 × 10x Cell Ranger filtered_feature_bc_matrix.h5 files.
Each H5 contains two feature types:
  - "Gene Expression"       → gene × cell UMI counts
  - "CRISPR Guide Capture"  → guide × cell UMI counts

Guide-to-gene assignment: each cell is assigned the guide with highest UMI
count. Non-targeting guides are mapped to "NTC". Cells with no guide UMI
or ambiguous assignments are discarded.

Usage:
    python 02_setup.py --dataset Pan-GW-hESC

Input: data/raw/GSE295214/*.h5  (85 files, ~7.8 GB total)
Output: data/origin/Pan-GW-hESC.h5ad
"""

import logging
import re
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATA_DIR  # noqa: E402
RAW_DIR = DATA_DIR / "raw" / "GSE295214"
ORIGIN_DIR = DATA_DIR / "origin"

DATASET_NAME = "Pan-GW-hESC"
CELL_TYPE = "hESC"
CONTROL_LABEL = "NTC"

# Guide assignment thresholds
MIN_GUIDE_UMI = 5      # min UMI for the top guide to be considered valid
MIN_RATIO = 3.0        # top guide UMI must be ≥ this multiple of second-best


# ---------------------------------------------------------------------------
# Guide name → target gene parsing
# ---------------------------------------------------------------------------

def sgrna_name_to_gene(name: str) -> str:
    """Map a guide name to its target gene symbol.

    Handles Weissman/Replogle lab naming conventions observed in 10x-based
    Perturb-seq experiments:
      - "non-targeting_{N}"         → "NTC"
      - "safe-targeting_{N}"        → "NTC"
      - "non_targeting_{N}"         → "NTC"
      - "{GENE}_sg{N}"              → "{GENE}"
      - "{GENE}_{N}"  (N is int)    → "{GENE}"
      - "{GENE}-{N}"  (N is int)    → "{GENE}"
      - Plain gene name             → as-is
    """
    name_lower = name.lower()

    # Non-targeting / safe-targeting controls
    if (name_lower.startswith("non-targeting")
            or name_lower.startswith("non_targeting")
            or name_lower.startswith("safe-targeting")
            or name_lower.startswith("safe_targeting")
            or name_lower.startswith("nontargeting")):
        return CONTROL_LABEL

    # '{GENE}_sg{N}' pattern (most common in Replogle/Weissman data)
    if "_sg" in name:
        return name.split("_sg")[0]

    # Weissman genome-wide library: '{GENE}_{strand}_{position}.{ver}-P1P2'
    # e.g. 'AAMP_+_219134841.23-P1P2'
    parts = name.split("_")
    if len(parts) >= 3 and parts[1] in ("+", "-"):
        return parts[0]

    # '{GENE}_{N}' where N is purely numeric
    parts = name.rsplit("_", 1)
    if len(parts) == 2 and re.match(r"^\d+$", parts[1]):
        return parts[0]

    # '{GENE}-{N}' where N is purely numeric
    parts = name.rsplit("-", 1)
    if len(parts) == 2 and re.match(r"^\d+$", parts[1]):
        return parts[0]

    # Fallback: return as-is (rare / unexpected format)
    return name


# ---------------------------------------------------------------------------
# Per-sample processing
# ---------------------------------------------------------------------------

def _batch_from_filename(fname: str) -> str:
    """Extract batch label from filename.

    e.g. 'GSM8943713_jan_10_GEX_A7_sg_H6_filtered_feature_bc_matrix.h5'
         → 'jan_10'
    """
    # Pattern: GSM{ID}_{month}_{day}_GEX_...
    m = re.search(r"GSM\d+_(jan_\d+)_", fname)
    if m:
        return m.group(1)
    # Fallback: first two underscore-fields after GSM{ID}
    parts = fname.split("_")
    if len(parts) >= 3:
        return f"{parts[1]}_{parts[2]}"
    return "unknown"


def process_sample(h5_path: Path) -> tuple[sp.csr_matrix, pd.DataFrame, pd.Index] | None:
    """Load one Cell Ranger H5 file and return (gex_matrix, obs_df, var_index).

    Returns None if the file has no usable CRISPR capture data.
    """
    try:
        import scanpy as sc
    except ImportError:
        log.error("scanpy is required. Install with: pip install scanpy")
        sys.exit(1)

    sample_name = h5_path.name
    batch = _batch_from_filename(sample_name)
    log.info("  Loading %s (batch=%s)...", sample_name, batch)

    # Read full multi-modal H5 (GEX + CRISPR)
    adata = sc.read_10x_h5(str(h5_path), gex_only=False)

    # Log feature overview for the first sample
    if not hasattr(process_sample, "_logged_features"):
        process_sample._logged_features = True
        ft_counts = adata.var["feature_types"].value_counts()
        log.info("  Feature types in first sample: %s", ft_counts.to_dict())
        crispr_names = adata.var_names[
            adata.var["feature_types"] == "CRISPR Guide Capture"
        ]
        log.info("  Guide name examples (first 10): %s",
                 list(crispr_names[:10]))

    # Split feature types
    gex_mask = adata.var["feature_types"] == "Gene Expression"
    crispr_mask = adata.var["feature_types"] == "CRISPR Guide Capture"

    if crispr_mask.sum() == 0:
        log.warning("  No CRISPR Guide Capture features in %s — skipping", sample_name)
        return None

    gex = adata[:, gex_mask].copy()
    crispr = adata[:, crispr_mask].copy()

    n_cells = gex.n_obs
    log.info("  %d cells, %d GEX genes, %d CRISPR guides",
             n_cells, gex.n_vars, crispr.n_vars)

    # ------------------------------------------------------------------
    # Assign guide per cell
    # ------------------------------------------------------------------
    guide_names = crispr.var_names.values.astype(str)

    # Check for 'target_gene_name' column (explicit mapping from Cell Ranger)
    if "target_gene_name" in crispr.var.columns:
        guide_gene_map = dict(zip(guide_names,
                                  crispr.var["target_gene_name"].values.astype(str)))
        # Replace NaN / empty with name-based parsing
        for k, v in guide_gene_map.items():
            if pd.isna(v) or v.strip() == "" or v.lower() in ("nan", "na", "none"):
                guide_gene_map[k] = sgrna_name_to_gene(k)
        log.info("  Using 'target_gene_name' column for guide→gene mapping")
    else:
        guide_gene_map = {g: sgrna_name_to_gene(g) for g in guide_names}

    # CRISPR count matrix: cells × guides (ensure CSR for row ops)
    crispr_mat = sp.csr_matrix(crispr.X)

    perturbations = []
    n_filtered = 0

    for i in range(n_cells):
        row = crispr_mat.getrow(i)
        if row.nnz == 0:
            n_filtered += 1
            perturbations.append(None)
            continue

        counts = row.toarray().ravel()
        sorted_idx = np.argsort(counts)[::-1]
        top_umi = counts[sorted_idx[0]]
        second_umi = counts[sorted_idx[1]] if len(sorted_idx) > 1 else 0

        # Discard cells with no confident guide assignment
        if top_umi < MIN_GUIDE_UMI:
            n_filtered += 1
            perturbations.append(None)
            continue
        if second_umi > 0 and (top_umi / second_umi) < MIN_RATIO:
            n_filtered += 1
            perturbations.append(None)
            continue

        perturbations.append(guide_gene_map[guide_names[sorted_idx[0]]])

    n_kept = sum(1 for p in perturbations if p is not None)
    log.info("  Guide assignment: %d/%d cells kept (%d filtered for low/ambiguous UMI)",
             n_kept, n_cells, n_filtered)

    # Filter to cells with valid guide assignments
    keep_mask = np.array([p is not None for p in perturbations])
    gex_filtered = gex[keep_mask].copy()
    pert_array = np.array([p for p in perturbations if p is not None], dtype=str)

    # Build obs
    barcodes = gex_filtered.obs_names.values.astype(str)
    # Prefix barcodes with batch to avoid collisions across samples
    unique_barcodes = pd.Index([f"{batch}__{bc}" for bc in barcodes])

    obs = pd.DataFrame({
        "perturbation": pert_array,
        "is_control": (pert_array == CONTROL_LABEL),
        "cell_type": CELL_TYPE,
        "dataset": DATASET_NAME,
        "batch": batch,
    }, index=unique_barcodes)

    # Gene var: use gene symbols as index, keep gene_ids
    var_idx = gex_filtered.var_names.copy()  # gene symbols from Cell Ranger
    gene_ids = (gex_filtered.var["gene_ids"].values.astype(str)
                if "gene_ids" in gex_filtered.var.columns
                else var_idx.values)

    return sp.csr_matrix(gex_filtered.X), obs, var_idx, gene_ids


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="overwrite an existing origin file")
    args = ap.parse_args()

    ORIGIN_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ORIGIN_DIR / f"{DATASET_NAME}.h5ad"

    if out_path.exists() and not args.force:
        log.info("SKIP: %s already exists (%s)", out_path.name,
                 f"{out_path.stat().st_size / 1e9:.1f} GB")
        return

    # Find all H5 files
    h5_files = sorted(RAW_DIR.glob("*.h5"))
    if len(h5_files) == 0:
        log.error("No H5 files found in %s", RAW_DIR)
        log.error("Run: python 01_download.py --dataset Pan-GW-hESC")
        sys.exit(1)

    log.info("Found %d H5 files in %s", len(h5_files), RAW_DIR)

    # ------------------------------------------------------------------
    # Process samples one at a time; accumulate matrices
    # ------------------------------------------------------------------
    all_mats: list[sp.csr_matrix] = []
    all_obs: list[pd.DataFrame] = []
    var_index: pd.Index | None = None
    var_gene_ids: np.ndarray | None = None

    n_ok = 0
    n_skip = 0

    for h5_path in h5_files:
        result = process_sample(h5_path)
        if result is None:
            n_skip += 1
            continue

        mat, obs, v_idx, g_ids = result

        # On first valid sample, record gene var info
        if var_index is None:
            var_index = v_idx
            var_gene_ids = g_ids
            log.info("Gene set: %d genes (from first valid sample)", len(var_index))
        else:
            # Validate gene sets match
            if len(v_idx) != len(var_index) or not (v_idx == var_index).all():
                log.warning(
                    "Gene set mismatch in %s (%d genes vs expected %d). "
                    "Aligning to reference...",
                    h5_path.name, len(v_idx), len(var_index)
                )
                # Reindex to reference gene set
                shared = var_index.intersection(v_idx)
                if len(shared) < len(var_index) * 0.9:
                    log.warning("  < 90%% gene overlap — skipping %s", h5_path.name)
                    n_skip += 1
                    continue
                # Build index mapping for sparse reindex
                ref_pos = {g: i for i, g in enumerate(var_index)}
                src_pos = {g: i for i, g in enumerate(v_idx)}
                col_src = np.array([src_pos[g] for g in var_index if g in src_pos])
                col_dst = np.array([ref_pos[g] for g in var_index if g in src_pos])
                n_rows = mat.shape[0]
                reindexed = sp.lil_matrix((n_rows, len(var_index)), dtype=mat.dtype)
                mat_csc = mat.tocsc()
                for j_src, j_dst in zip(col_src, col_dst):
                    reindexed[:, j_dst] = mat_csc[:, j_src]
                mat = sp.csr_matrix(reindexed)

        all_mats.append(mat)
        all_obs.append(obs)
        n_ok += 1
        log.info("  Accumulated: %d samples, %d total cells so far",
                 n_ok, sum(m.shape[0] for m in all_mats))

    if n_ok == 0:
        log.error("No valid samples processed. Check H5 files in %s", RAW_DIR)
        sys.exit(1)

    log.info("Processed %d/%d samples (%d skipped)", n_ok, len(h5_files), n_skip)

    # ------------------------------------------------------------------
    # Concatenate all samples
    # ------------------------------------------------------------------
    log.info("Concatenating %d sample matrices...", n_ok)
    X_full = sp.vstack(all_mats, format="csr")
    obs_full = pd.concat(all_obs, axis=0)
    assert X_full.shape[0] == len(obs_full), "Row count mismatch after concat"

    # Ensure unique obs_names (barcodes already prefixed with batch)
    if obs_full.index.duplicated().any():
        n_dup = obs_full.index.duplicated().sum()
        log.warning("Deduplicating %d duplicate barcodes with suffix", n_dup)
        obs_full.index = pd.Index(obs_full.index).astype(str)
        # Append integer suffix for duplicates
        counts: dict[str, int] = {}
        new_idx = []
        for name in obs_full.index:
            if name in counts:
                counts[name] += 1
                new_idx.append(f"{name}_{counts[name]}")
            else:
                counts[name] = 0
                new_idx.append(name)
        obs_full.index = pd.Index(new_idx)

    # Build var
    var_full = pd.DataFrame(
        {"gene_id": var_gene_ids},
        index=pd.Index(var_index, name="gene_name"),
    )
    if var_full.index.duplicated().any():
        log.info("Making %d duplicate gene names unique", var_full.index.duplicated().sum())
        var_full = var_full.loc[~var_full.index.duplicated(keep="first")]
        X_full = X_full[:, ~pd.Index(var_index).duplicated(keep="first")]

    # Build final AnnData (raw integer counts)
    adata = ad.AnnData(X=X_full, obs=obs_full, var=var_full)
    adata.obs["is_control"] = adata.obs["is_control"].astype(bool)

    # Summary stats
    n_perts = adata.obs["perturbation"].nunique()
    n_ctrl = adata.obs["is_control"].sum()
    n_pert_cells = (~adata.obs["is_control"]).sum()
    log.info("")
    log.info("=" * 60)
    log.info("FINAL ADATA: %d cells × %d genes", adata.n_obs, adata.n_vars)
    log.info("  Perturbations  : %d unique labels", n_perts)
    log.info("  Control cells  : %d (%.1f%%)",
             n_ctrl, 100 * n_ctrl / adata.n_obs)
    log.info("  Perturbed cells: %d (%.1f%%)",
             n_pert_cells, 100 * n_pert_cells / adata.n_obs)
    log.info("  Batches        : %s", sorted(adata.obs["batch"].unique()))

    # Sample guide genes
    pert_labels = sorted(set(adata.obs["perturbation"].unique()) - {CONTROL_LABEL})
    log.info("  Sample target genes (first 20): %s", pert_labels[:20])

    log.info("=" * 60)
    log.info("Writing %s...", out_path.name)
    adata.write_h5ad(out_path)
    log.info("Done: %s (%.1f GB)", out_path.name, out_path.stat().st_size / 1e9)


if __name__ == "__main__":
    main()
