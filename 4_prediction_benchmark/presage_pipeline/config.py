"""Paths and constants for the PRESAGE (+Perturb-seq) runs.

PRESAGE (Littman, Levine et al., bioRxiv 2025.06.03.657653;
https://github.com/genentech/PRESAGE) is used from an unmodified upstream checkout
(``paths.PRESAGE_REPO``); every change is a runtime patch in ``03_train.py``.

Relative to upstream's example notebooks, three things are specific to this
pipeline:

1. TRAINING DATA is the single-cell GEARS build of the target
   (``<GEARS_DATA_DIR>/<build>/<build>/perturb_processed.h5ad``) with the same
   cv5 folds GEARS and scGPT-ft use.

2. CROSS-DATASET PERTURB-SEQ PRIORS: every screen in the target's source pool
   (``targets.source_pool``) is its own knowledge source, i.e. its own channel
   with its own MLP and attention weight, as in the paper's
   "PRESAGE (+Perturb-seq)" variant.  The perturbation x gene effect matrices go
   in as they are; ``read_and_embed`` runs the PCA.

3. ``read_and_embed`` IS PATCHED so that each knowledge source gets its own zero
   matrix (``patched_read_and_embed.py``, not included; see README.md).
   Upstream reuses one matrix across the source loop, so a source that covers
   few genes -- as a Perturb-seq screen does -- would inherit the previous
   sources' rows.

The target comes from the PP_DATASET environment variable:

    PP_DATASET=VCC python 03_train.py --fold 1
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT.parent))

import paths  # noqa: E402
import targets  # noqa: E402

# --------------------------------------------------------------------------
# Upstream inputs (read only)
# --------------------------------------------------------------------------
PRESAGE_REPO = paths.PRESAGE_REPO
PRESAGE_SRC = PRESAGE_REPO / "src"
PRESAGE_CONFIGS = PRESAGE_REPO / "configs"
PATHWAY_EMB_DIR = PRESAGE_REPO / "cache" / "pathway_embeddings"
OTHER_EMB_DIR = PRESAGE_REPO / "cache" / "other_embeddings"

EFFECT_DICT = paths.EFFECT_DICT

# --------------------------------------------------------------------------
# Target
# --------------------------------------------------------------------------
DATASET = os.environ.get("PP_DATASET")
if DATASET is None:
    raise SystemExit(
        "PP_DATASET is not set. Choose one of: " + ", ".join(targets.TARGETS) + "\n"
        "e.g.  PP_DATASET=VCC python 03_train.py --fold 1"
    )
targets.get(DATASET)

EXCLUDE_DATASETS = targets.excluded(DATASET)

GEARS_DIR = paths.GEARS_DATA_DIR / targets.presage_build(DATASET) / targets.presage_build(DATASET)
SC_H5AD = GEARS_DIR / "perturb_processed.h5ad"
_SPLIT_BUILD = targets.split_build(DATASET)
SPLIT_SRC_DIR = paths.GEARS_DATA_DIR / _SPLIT_BUILD / _SPLIT_BUILD / "splits"
SPLIT_PKL_GLOB = f"{_SPLIT_BUILD}_cv5_{{fold}}.pkl"

N_FOLDS = 5                   # array index 1..5 == CV fold, not a random seed

# --------------------------------------------------------------------------
# Outputs, all target-scoped
# --------------------------------------------------------------------------
WORK = paths.PRESAGE_WORK_DIR

# PRESAGE's data_dir.  prepare_data writes <DATASET>_processed.h5ad here with a
# DENSE X on the first fold; the other folds reuse it.
DATA_DIR = WORK / "data"
DATASET_DIR = DATA_DIR / DATASET
RAW_H5AD = DATASET_DIR / "perturb_processed.h5ad"      # symlink to SC_H5AD
DEG_FILE = DATASET_DIR / "degs" / "merged.degs.json"

SPLIT_DIR = WORK / "splits" / f"{DATASET}_random_splits"
PERT_LIST_FILE = SPLIT_DIR.parent / f"{DATASET}_random_splits.perturbations.json"

ARTIFACT_DIR = WORK / "artifacts"
# One pkl per external screen, shared across targets.  Which of them a target may
# use is recorded in its manifest and source list, never found by globbing.
PERT_SRC_DIR = ARTIFACT_DIR / "pert_sources"

# The knowledge-source list is content-addressed (sources.<sha8>.txt, with
# current.<dataset>.txt naming the live one) because read_and_embed keys its
# tensor cache on the list's basename, not its contents.
SOURCE_LIST_DIR = ARTIFACT_DIR / "source_lists"
SOURCE_LIST_POINTER = SOURCE_LIST_DIR / f"current.{DATASET}.txt"

# read_and_embed's knowledge-tensor cache.
CACHE_DIR = WORK / "cache" / "pathway_embeddings"

OUTPUT_DIR = WORK / "output"
PRED_DIR = OUTPUT_DIR / "predictions"
CKPT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "lightning_logs"
ATTN_DIR = OUTPUT_DIR / "attention"

RUN = DATASET


def split_json(fold: int) -> Path:
    return SPLIT_DIR / f"seed_{fold}.json"


def split_pkl(fold: int) -> Path:
    return SPLIT_SRC_DIR / SPLIT_PKL_GLOB.format(fold=fold)


def pert_source_pkl(key: str) -> Path:
    """Where the Perturb-seq knowledge source for external screen `key` lives."""
    return PERT_SRC_DIR / f"perturbseq.{key}.pkl"


def kept_pert_sources(effect_keys) -> list:
    """External screens usable as priors for DATASET, in a stable order."""
    return targets.source_pool(effect_keys, DATASET)


def current_source_list() -> Path:
    """Path to the live knowledge-source list, per current.<dataset>.txt."""
    if not SOURCE_LIST_POINTER.exists():
        raise SystemExit(
            f"{SOURCE_LIST_POINTER} missing -- run 02_make_source_list.py first"
        )
    path = SOURCE_LIST_DIR / SOURCE_LIST_POINTER.read_text().strip()
    if not path.exists():
        raise SystemExit(f"{SOURCE_LIST_POINTER} points at missing {path}")
    return path


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
N_EMB = 128                   # must equal model.n_nmf_embedding in the config
CONTROL_KEY = "control"       # PRESAGE's harmonized control label
DEG_TOP_N = 1000              # matches ReplogleDataModule.compute_degs' loc[:999]

# Perturbation labels treated as controls in the effect matrices, dropped before
# a matrix becomes a knowledge source.
CONTROL_LABELS = {"control", "ctrl", "non-targeting", "non_targeting", "nt", "ntc"}


def harmonize(key: str) -> str:
    """GEARS condition key -> PRESAGE perturbation key.

    Reproduces presage_datamodule.prepare_data:
        "RRM1+ctrl" -> "RRM1",  "ctrl+ctrl" -> "" -> "control"
    """
    out = key.replace("+", "_").replace("ctrl", "").strip("_")
    return out if out else CONTROL_KEY


# --------------------------------------------------------------------------
# Knowledge sources
# --------------------------------------------------------------------------
# The 40 non-Perturb-seq sources, in the order of upstream's
# sample_files/prior_files/sample.knowledge_experimental.txt.  Enumerated rather
# than globbed: the cache directories also hold files that are not sources.
#
# STRING (Szklarczyk et al. 2023) and MSigDB 2023.2 (Subramanian et al. 2005;
# Liberzon et al. 2015):
PATHWAY_EMB_FILES = (
    "stringdb.human.highest.pkl",
    "stringdb.human.high.pkl",
    "stringdb.human.medium.pkl",
    "c1.all.v2023.2.Hs.symbols.pkl",
    "c2.cgp.v2023.2.Hs.symbols.pkl",
    "c2.cp.pid.v2023.2.Hs.symbols.pkl",
    "c2.cp.reactome.v2023.2.Hs.symbols.pkl",
    "c2.cp.wikipathways.v2023.2.Hs.symbols.pkl",
    "c3.all.v2023.2.Hs.symbols.pkl",
    "c3.mir.mirdb.v2023.2.Hs.symbols.pkl",
    "c3.mir.mir_legacy.v2023.2.Hs.symbols.pkl",
    "c3.mir.v2023.2.Hs.symbols.pkl",
    "c3.tft.gtrd.v2023.2.Hs.symbols.pkl",
    "c3.tft.tft_legacy.v2023.2.Hs.symbols.pkl",
    "c3.tft.v2023.2.Hs.symbols.pkl",
    "c4.3ca.v2023.2.Hs.symbols.pkl",
    "c4.all.v2023.2.Hs.symbols.pkl",
    "c4.cgn.v2023.2.Hs.symbols.pkl",
    "c4.cm.v2023.2.Hs.symbols.pkl",
    "c5.all.v2023.2.Hs.symbols.pkl",
    "c5.go.bp.v2023.2.Hs.symbols.pkl",
    "c5.go.cc.v2023.2.Hs.symbols.pkl",
    "c5.go.mf.v2023.2.Hs.symbols.pkl",
    "c5.go.v2023.2.Hs.symbols.pkl",
    "c5.hpo.v2023.2.Hs.symbols.pkl",
    "c6.all.v2023.2.Hs.symbols.pkl",
    "c7.all.v2023.2.Hs.symbols.pkl",
    "c7.immunesigdb.v2023.2.Hs.symbols.pkl",
    "c7.vax.v2023.2.Hs.symbols.pkl",
    "c8.all.v2023.2.Hs.symbols.pkl",
    "h.all.v2023.2.Hs.symbols.pkl",
)

# Periscope (Ramezani et al. 2025) x3; DepMap 24Q2; Funk et al. 2022 OPS;
# GenePT (Chen & Zou 2024) x2; BioGPT (Luo et al. 2022); ESM2 (Lin et al. 2023).
OTHER_EMB_FILES = (
    "HeLa_DMEM.pkl",
    "A549_CP186.pkl",
    "HeLa_HPLM.pkl",
    "CRISPRGeneEffectDepMap.pkl",
    "funk.cellprofiler.embeddings.pkl",
    "GenePT_ada.pkl",
    "GenePT_protein_embedding.pkl",
    "biogpt_emb_gene2biogpt.pkl",
    "esm_emb_gene2esm.pkl",
)
