"""Locations and the dataset registry, used by every stage.

Paths are relative to the repo root: data/ (inputs), results/ (script outputs) and
results_reference/ (the frozen paper results, read only by the notebooks)."""

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"
RESULTS = REPO_ROOT / "results"
REFERENCE_RESULTS = REPO_ROOT / "results_reference"

# inputs
MOMENTS_DIR = DATA / "pseudobulk"  # <component>_moments.h5ad
DE_DIR = DATA / "de"  # <component>_wilcoxon_batch_corrected.h5ad
SPLIT_DIR = DATA / "split"  # {base}_seed{0-4}_half{0,1}_moments.h5ad
SC_DIR = DATA / "sc"  # <screen>.h5ad, single cells for GEARS and scGPT
DEPMAP_GENE_EFFECT = DATA / "essential" / "CRISPRGeneEffect.csv"
CORUM_GMT = DATA / "gene_set_list" / "corum_human.gmt"
SCENICPLUS_LOOM = DATA / "scenicplus" / "SCENIC+_DPCL_grnboost_gene_based_autoreg.loom"
SCGPT_MODEL_DIR = DATA / "models" / "scGPT_human"
GEARS_SUPPORT_DIR = DATA / "gears"  # GEARS GO graph and essential-gene files
PRESAGE_REPO = DATA / "external" / "PRESAGE"
DES_PER_PERT = DATA / "external" / "des" / "des_per_pert.csv"
DES_THRESHOLD_PER_PERT = DATA / "external" / "des" / "des_threshold_per_pert.csv"
# paper copies of tables that stages 1, 2 and 4 also write; later stages read these
PAPER_PREDICTION_QC_DIR = DATA / "filter_result_bulk"
PAPER_EFFECT_DICT = DATA / "effect_dict" / "bc_bulk_qc_effect.pkl"
PAPER_PAN_ESSENTIALITY = DATA / "essential" / "pan_essentiality_score.csv"
PAPER_PREDICTION_RESULTS = DATA / "prediction_result"  # {screen}_summary_metrics.csv

# 1_preprocessing
QC_GENE_PANELS = RESULTS / "01_qc_strength_features" / "qc_gene_panels"
STRENGTH_SCORES = RESULTS / "01_qc_strength_features" / "perturbation_strength" / "scores"
L1_SELECTION = RESULTS / "01_qc_strength_features" / "perturbation_strength" / "L1_selection"
PREDICTION_QC_DIR = RESULTS / "1_preprocessing" / "filter_result_bulk"
EFFECT_DICT = RESULTS / "1_preprocessing" / "bc_bulk_qc_effect.pkl"

# 2_global_trend
FIRST_FACTOR = RESULTS / "03_global_trend" / "first_factor"
PAN_ESSENTIALITY = RESULTS / "03_global_trend" / "pan_essentiality_score.csv"

# 3_cross_dataset_similarity
SIMILARITY_TOTAL_AUDIT = RESULTS / "02_data_similarity" / "se_corrected_total_effect_cosine" / "norm_audit"
SIMILARITY_RESIDUAL_AUDIT = RESULTS / "02_data_similarity" / "se_corrected_residual_cosine" / "norm_audit"

# 4_prediction
PREDICTION_WORK_DIR = RESULTS / "prediction"  # GEARS builds and predictions
PREDICTION_EVALUATION = RESULTS / "prediction" / "evaluation"  # <target>/summary_metrics.csv

# 5_response_hierarchy
TOTAL_DISCOVERIES = RESULTS / "05_rejection_matrix_structure" / "cross_validated_nestedness"
RESIDUAL_DISCOVERIES = RESULTS / "05_rejection_matrix_structure" / "global_trend_adjusted_rejections"
RESIDUAL_DISCOVERIES_CANONICAL = RESIDUAL_DISCOVERIES / "cross_validated_nestedness"
DES_SOURCES = RESULTS / "07_prediction_benchmarking" / "des_replication" / "source_data"
RESPONSE_BREADTH = RESULTS / "09_des_threshold" / "rejection_count_stratification"

# 6_magnitude_structure
CROSS_SPLIT = RESULTS / "04_cross_split_magnitude_rank1" / "capped_top_0.01_percent_k0_to_k15"

# 7_complex_enrichment
COMPLEX_ENRICHMENT = RESULTS / "10_modular_enrichment" / "02_margin_rank1_specificity" / "data"

# 8_TF_targets
TF_RESULTS_DIR = RESULTS / "tf_analysis" / "08_scenicplus_active_magnitude"


def reference(path):
    """The frozen paper copy of a results/ path."""
    return REFERENCE_RESULTS / Path(path).relative_to(RESULTS)


# Dataset registry.  The keys name the input files and seed the splits, so keep
# them and their order.  components, effect_key and label default to the key.
#   paper       name in the paper and figures
#   components  moments and DE files
#   effect_key  key in the effect dict and prediction results
#   label       name in the frozen results (per_dataset/<label>)
#   sisters     same-cell-type screens left out of its prediction source pool
#   variants    other spellings in the frozen tables
# paper_name(), effect_key(), sisters() and slug() accept any spelling.

@dataclass
class Dataset:
    key: str
    paper: str
    components: tuple = ()
    effect_key: str = ""
    label: str = ""
    sisters: tuple = ()
    variants: tuple = ()

    def __post_init__(self):
        self.components = self.components or (self.key,)
        self.effect_key = self.effect_key or self.key
        self.label = self.label or self.key


REGISTRY = {dataset.key: dataset for dataset in (
    Dataset("Replogle-E-k562", "Replogle-E-K562", sisters=("Replogle-GW-k562",)),
    Dataset("Replogle-GW-k562", "Replogle-GW-K562", sisters=("Replogle-E-k562",)),
    Dataset("Replogle-E-rpe1", "Replogle-E-RPE1"),
    Dataset("Nadig-HEPG2", "Nadig-HepG2"),
    Dataset("Nadig-JURKAT", "Nadig-Jurkat"),
    Dataset("Huang-HCT116", "Huang-HCT116"),
    Dataset("Huang-HEK293T", "Huang-HEK293T"),
    Dataset("Pan-GW-hESC", "Pan-GP-H1", label="Pan-GP-hESC", sisters=("VCC",)),
    Dataset("VCC", "VCC-H1", label="VCC-hESC", sisters=("Pan-GW-hESC",)),
    # VCC-H1 downsampled in UMIs and cells; used only in 3_cross_dataset_similarity (S1b)
    Dataset("VCC-subsampled", "VCC-subsampled-H1", label="VCC-subsampled-hESC"),
    Dataset("Feng-ts", "Feng-TS-iPSC", label="Feng-ts-ipsc",
            sisters=("Feng-GW", "Nourreddine-GW-ipsc"), variants=("Feng-ts-iPSC",)),
    Dataset("Feng-GW", "Feng-GW-iPSC", components=("Feng-gwsf", "Feng-gwsnf"),
            effect_key="Feng-gw", label="Feng-GW-ipsc", sisters=("Feng-ts", "Nourreddine-GW-ipsc")),
    Dataset("Nourreddine-GW-ipsc", "Nourreddine-iPSC", sisters=("Feng-GW", "Feng-ts"),
            variants=("Nourreddine-GW-iPSC",)),
)}

# spelling -> key
KEYS = {
    spelling: dataset.key
    for dataset in REGISTRY.values()
    for spelling in (dataset.key, dataset.paper, dataset.effect_key, dataset.label,
                     *dataset.variants)
}


def paper_name(name):
    return REGISTRY[KEYS[name]].paper


def effect_key(name):
    return REGISTRY[KEYS[name]].effect_key


def sisters(name):
    """Registry keys of the sister screens."""
    return REGISTRY[KEYS[name]].sisters


def slug(name):
    """The label of the frozen results; do not change it."""
    return REGISTRY[KEYS[name]].label


# the 12 screens
DATASET_COMPONENTS = {
    name: dataset.components for name, dataset in REGISTRY.items() if name != "VCC-subsampled"
}
DATASET_ORDER = tuple(DATASET_COMPONENTS)

# VCC-subsampled uses the QC, gene panels and L1 selection of VCC; its effects are its own
SELECTION_REFERENCE_DATASET = {"VCC-subsampled": "VCC"}
