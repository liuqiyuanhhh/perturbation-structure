"""Configuration of the 12 screens analysed in the paper (13 files: Feng-GW-iPSC
consists of the two sub-screens Feng-gwsf and Feng-gwsnf), plus VCC-subsampled,
a downsampled version of VCC-H1 derived from data/origin/VCC.h5ad
(../vcc_subsampling/downsample_vcc.py).

Fields per dataset:
  control     control label in obs[pert_col]
  pert_col    obs column holding the perturbation (target-gene) label
  batch_col   obs column with the experimental batch used for stratification
  donor_col   donor/cell-line column; when set, batch-aware steps stratify on
              the joint (batch, donor) key "<batch_col>_x_<donor_col>"
  download    downloaders/<download>.sh fetching the raw files (None: derived dataset)
  setup       setup/<setup>.py building data/origin/<dataset>.h5ad (None: derived dataset)
  split       has split-half replicates (step 05)
"""
from __future__ import annotations

# Quality control, identical for every dataset.
QC = {
    "min_genes": 100,            # a cell must express at least this many genes
    "min_cells_per_pert": 20,    # a perturbation must be carried by at least this many cells
    "min_cells_per_gene": 100,   # a gene must be detected in at least this many kept cells
}
PERTURBATION_TYPE = "CRISPRi"

DATASETS: dict[str, dict] = {
    "Replogle-GW-k562":    {"control": "control", "pert_col": "perturbation", "batch_col": "batch",
                            "download": "scperturb_zenodo", "setup": "passthrough"},
    "Replogle-E-k562":     {"control": "control", "pert_col": "perturbation", "batch_col": "batch",
                            "download": "scperturb_zenodo", "setup": "passthrough"},
    "Replogle-E-rpe1":     {"control": "control", "pert_col": "perturbation", "batch_col": "batch",
                            "download": "scperturb_zenodo", "setup": "passthrough"},
    "Nadig-JURKAT":        {"control": "non-targeting", "pert_col": "gene", "batch_col": "gem_group",
                            "download": "nadig", "setup": "nadig"},
    "Nadig-HEPG2":         {"control": "non-targeting", "pert_col": "gene", "batch_col": "gem_group",
                            "download": "nadig", "setup": "nadig"},
    "Huang-HCT116":        {"control": "Non-Targeting", "pert_col": "gene_target", "batch_col": "sample",
                            "download": "huang_xatlas", "setup": "passthrough"},
    "Huang-HEK293T":       {"control": "Non-Targeting", "pert_col": "gene_target", "batch_col": "sample",
                            "download": "huang_xatlas", "setup": "passthrough"},
    "Feng-gwsf":           {"control": "control", "pert_col": "perturbation", "batch_col": "Batch",
                            "donor_col": "donor_line", "download": "feng", "setup": "feng"},
    "Feng-gwsnf":          {"control": "control", "pert_col": "perturbation", "batch_col": "Batch",
                            "donor_col": "donor_line", "download": "feng", "setup": "feng"},
    "Feng-ts":             {"control": "control", "pert_col": "perturbation", "batch_col": "Batch",
                            "donor_col": "donor_line", "download": "feng", "setup": "feng"},
    "Pan-GW-hESC":         {"control": "NTC", "pert_col": "perturbation", "batch_col": "batch",
                            "download": "pan_hesc", "setup": "pan_hesc"},
    "VCC":                 {"control": "non-targeting", "pert_col": "perturbation", "batch_col": "batch",
                            "download": "vcc_2025", "setup": "vcc"},
    "Nourreddine-GW-ipsc": {"control": "NTC", "pert_col": "perturbation", "batch_col": "batch",
                            "download": "kolf_ipsc", "setup": "kolf_ipsc"},
    "VCC-subsampled":      {"control": "non-targeting", "pert_col": "perturbation", "batch_col": "batch",
                            "download": None, "setup": None, "split": False},
}


def get(name: str) -> dict:
    if name not in DATASETS:
        raise SystemExit(f"Unknown dataset {name!r}. Available: {', '.join(DATASETS)}")
    return {"donor_col": None, "split": True, **DATASETS[name]}


def resolve(name: str) -> list[str]:
    """'all' -> the 13 screen files (not the derived VCC-subsampled); otherwise the named dataset."""
    return [n for n, d in DATASETS.items() if d["setup"]] if name == "all" else [get(name) and name]


def strat_col(name: str) -> str:
    """obs column used for batch stratification (joint batch x donor when donor_col is set)."""
    d = get(name)
    return f"{d['batch_col']}_x_{d['donor_col']}" if d["donor_col"] else d["batch_col"]


def fallback_strat_col(name: str) -> str | None:
    """Plain batch column, used by step 04 for perturbations without a control in any joint stratum."""
    d = get(name)
    return d["batch_col"] if d["donor_col"] else None
