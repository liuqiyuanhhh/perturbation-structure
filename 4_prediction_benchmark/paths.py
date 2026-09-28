"""Every location the prediction code reads or writes (utils.paths).

The inputs are in data/, everything the pipelines write in results/prediction
(WORK_ROOT), the evaluation in results/prediction/evaluation (EVAL_DIR).
"""

from utils import paths as repo

# Inputs (read only)
# batch-corrected single-cell matrices {screen}.h5ad, log1p(CPTT)
SC_DIR = repo.SC_DIR
# pseudobulk moments {screen}_moments.h5ad; only uns/control_profile is read
MOMENTS_DIR = repo.MOMENTS_DIR
# Wilcoxon DE {screen}_wilcoxon_batch_corrected.pkl, a 'scores' frame
DE_DIR = repo.DE_DIR
# gene panel and perturbation filter of the benchmark
QC_DIR = repo.PAPER_PREDICTION_QC_DIR
OUTCOME_QC = QC_DIR / "outcome_expression_qc.csv.gz"
PERT_QC = QC_DIR / "perturbation_quality_filter.csv.gz"
# {screen: DataFrame[perturbation x gene]}: ground truth of the evaluation,
# training data of Weighted, the linear models and the training mean, PRESAGE's Perturb-seq priors
EFFECT_DICT = repo.PAPER_EFFECT_DICT
# scGPT whole-human checkpoint (args.json, best_model.pt, vocab.json)
SCGPT_MODEL_DIR = repo.SCGPT_MODEL_DIR
# PRESAGE checkout with its knowledge-source cache
PRESAGE_REPO = repo.PRESAGE_REPO
# optional GEARS support files, for machines where GEARS cannot download them
GEARS_SUPPORT_DIR = repo.GEARS_SUPPORT_DIR

# Outputs
WORK_ROOT = repo.PREDICTION_WORK_DIR
# <build>/<build>/{perturb_processed.h5ad, data_pyg/, splits/}
GEARS_DATA_DIR = WORK_ROOT / "gears_data"
# <build>-cv5_<fold>_{gears,scgpt_ft}_post-{gt,pred}.csv
GEARS_PRED_DIR = WORK_ROOT / "predictions" / "gears_scgpt"
# PRESAGE's data/, splits/, artifacts/, cache/, output/
PRESAGE_WORK_DIR = WORK_ROOT / "presage"
PRESAGE_PRED_DIR = PRESAGE_WORK_DIR / "output" / "predictions"
# one folder per target
EVAL_DIR = repo.PREDICTION_EVALUATION
