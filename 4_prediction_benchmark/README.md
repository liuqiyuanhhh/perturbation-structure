# 4_prediction_benchmark

Perturbation-response prediction benchmark: three published models (GEARS,
scGPT-ft, PRESAGE), Weighted, two linear models and the training mean, scored under 5-fold cross-validation on 12 batch-corrected
Perturb-seq targets (Fig 1d, Supplementary Fig S2). The per-fold predictions
are also inputs of the DES panels (`5_response_hierarchy`).

| Method | Predicts from | Code |
|---|---|---|
| GEARS | gene-gene graphs (GO, co-expression) + the target's training perturbations | `gears_scgpt/run_gears.py` (README inside) |
| scGPT-ft | the scGPT whole-human model, fine-tuned on the target's training perturbations | `gears_scgpt/run_scgpt.py` |
| PRESAGE (+Perturb-seq) | 40 knowledge sources + one source per other Perturb-seq screen | `presage_pipeline/` (README inside) |
| Weighted | the same perturbation's effect in the other screens, weighted by similarity to the target | `weighted_and_linear/weighted.py` |
| Linear (P from other ds / from training) | bilinear ridge model; perturbation embedding from the other screens or from the target's training effects | `weighted_and_linear/linear.py` |
| Train mean | mean effect of the fold's training perturbations | `weighted_and_linear/train_mean.py` |

The five test folds of a target are disjoint, cover every perturbation of its
GEARS build and are shared by all methods. `targets.py` defines the 12 targets
and the pool of other screens each may learn from (used by Weighted, the linear models, the
PRESAGE priors and the evaluation): the 12 screens minus the target and
its sisters, screens that would leak it (VCC and Pan-GW-hESC; the two Replogle K562
screens; Feng-ts, Feng-gw and Nourreddine-GW-ipsc), taken from the dataset
registry in `utils/paths.py`. `paths.py` holds every
location, `slurm/` the job templates.

GEARS is the installed `cell-gears==0.1.2` package (`envs/`, `gears_scgpt/README.md`).

## Setup

Three environments; `envs/*.txt` pin the package versions used. The jobs
activate them by the names in `GEARS_ENV`, `SCGPT_ENV` and `PRESAGE_ENV`
(default `perturbation-structure-gears`, `-scgpt`, `-presage`).

| Environment | Python | Used for |
|---|---|---|
| `envs/gears.txt` | 3.10 | GEARS builds and training, evaluation |
| `envs/scgpt.txt` | 3.10 | scGPT fine-tuning (`scgpt==0.2.4`) |
| `envs/presage.txt` | 3.11 | PRESAGE; created from upstream PRESAGE's `environment.yml` |

External resources:

- **PRESAGE**: clone https://github.com/genentech/PRESAGE into
  `data/external/PRESAGE` and unpack its knowledge-source cache as its README
  describes. Step 03 also needs `presage_pipeline/patched_read_and_embed.py`,
  which is not included (`presage_pipeline/README.md`).
- **scGPT**: the whole-human checkpoint `scGPT_human` (`args.json`,
  `best_model.pt`, `vocab.json`) from the
  [scGPT repository](https://github.com/bowang-lab/scGPT), in
  `data/models/scGPT_human`.
- **GEARS support files** (`gene2go_all.pkl`,
  `essential_all_data_pert_genes.pkl`, `go_essential_all/`): GEARS downloads
  them on first use; without network access put them in `data/gears`.

Other inputs: single cells in `data/sc` (log1p(CPTT); placeholder
`1_preprocessing/crispyx_pseudobulk_de`), `uns/control_profile` of the
moments in `data/pseudobulk`, Wilcoxon DE scores
`data/de/{screen}_wilcoxon_batch_corrected.pkl`, and the paper QC tables
(`data/filter_result_bulk`) and effect dict
(`data/effect_dict/bc_bulk_qc_effect.pkl`) of `1_preprocessing`. Everything
is written to `results/prediction`.

## Running

```bash
bash 4_prediction_benchmark/run.sh
TARGETS="VCC Feng-gw" bash 4_prediction_benchmark/run.sh     # some targets only
```

`run.sh` submits the jobs below with `sbatch`, chained by `afterok`, for the
12 targets (or `TARGETS`) and the GEARS builds they need. `sbatch` reads
`SBATCH_ACCOUNT` and `SBATCH_PARTITION`; `GPU_PARTITION` sends the GEARS and
scGPT-ft jobs elsewhere; `JOB_SETUP` holds shell lines each job runs before
`conda activate`. Logs go to `results/logs/slurm`. By hand, set the variable
a template names: `DATASET=VCC sbatch 4_prediction_benchmark/slurm/gears_prep.sbatch`.

| Step | Template (variable) | Script | Per | Writes (in `results/prediction`) |
|---|---|---|---|---|
| 1a | `gears_prep.sbatch` (`DATASET`) | `gears_scgpt/prepare_data.py --dataset` | GEARS build | `gears_data/<build>/` |
| 1b | `gears_fold_subset.sbatch` (`PARENT`) | `gears_scgpt/prepare_data.py --parent --fold` | fold of Huang-HCT116, Huang-HEK293T, Nourreddine-GW-ipsc | `gears_data/<parent>-third-f<fold>/` |
| 2 | `train_gears.sbatch`, `train_scgpt.sbatch` (`BUILD`), one GPU | `gears_scgpt/run_gears.py`, `run_scgpt.py` | fold | `predictions/gears_scgpt/` |
| 3a | `presage_prep.sbatch` (`PP_DATASET`) | `presage_pipeline/00`-`02` | target, one at a time | `presage/` |
| 3b | `presage_train.sbatch` (`PP_DATASET`) | `presage_pipeline/03_train.py` | fold; fold 1 first | `presage/output/predictions/` |
| 3c | `presage_collect.sbatch` (`PP_DATASET`) | `presage_pipeline/04_collect_predictions.py` | target | `presage/output/predictions/` |
| 4 | `evaluate.sbatch` (`TARGET`) | `evaluation/evaluate.py`, with Weighted, the linear models and the training mean (`weighted_and_linear/`) | target | `evaluation/<target>/` |

- Huang-HCT116, Huang-HEK293T and Nourreddine-GW-ipsc train GEARS and
  scGPT-ft on one reduced build per fold (its test perturbations, a third of
  the training perturbations, 50,000 control cells), Feng-gw on
  `Feng-gw-control-cap` (50,000 control cells, same folds).
- GEARS and scGPT-ft predict absolute expression; the evaluation subtracts the
  control profile of their training data: for targets with `control_build` in
  `targets.py`, the pooled control mean of that build, computed in
  `evaluate.py`; else `uns/control_profile` of the moments file.

## Compute

`run.sh` requests the memory and time of the paper runs (tables at its top):
up to 1000G for a GEARS build, 550G and 120 h per GEARS or scGPT-ft fold,
440G per PRESAGE fold (CPU), 200G per evaluation.

## Evaluation

`evaluation/evaluate.py --target <target|all>` scores every method per fold,
excluding the perturbed gene. Residual space removes the leading SVD factor,
fitted separately for the truth and for each method on the rows scored. A
test perturbation is scored if a screen in the target's pool measured it; one
missing from a prediction file counts as a zero prediction.

| Column | Space | Genes |
|---|---|---|
| `mean_weighted_mse` | effect | all panel genes, weighted by DE strength |
| `mean_pds` | effect | all panel genes; perturbation discrimination score (cosine) |
| `mean_resid_cosine` | residual | all panel genes |
| `mean_resid_top20_pval_cosine` | residual | top 20 DE genes |
| `mean_resid_top100_pval_cosine` | residual | top 100 DE genes |

Outputs in `results/prediction/evaluation/<target>/`:

| File | Content |
|---|---|
| `summary_metrics.csv` | one row per fold (`seed`) and method, the five metrics |
| `models_and_seeds.csv` | model predictions found, and the folds where all exist |
| `predictions_by_seed.pkl`, `splits_by_seed.pkl` | `{fold: {method: test perturbation x gene}}` predicted effects; `{fold: {train, test}}` |

Method keys: `gears`, `scGPT-ft`, `presage`, `weighted`,
`linear_embedding_from_otherds_10`, `paper_linear_embedding_from_training_10`,
`train_mean`. The paper copies of the metrics, predictions and splits are in
`data/prediction_result`, prefixed with the target
(`{target}_summary_metrics.csv`); `5_response_hierarchy` reads them there.

## Figure notebook

`prediction_figures.ipynb` draws Fig 1d (VCC-H1, Replogle-GW-K562) and S2
(the other ten biological datasets) into `figures/` and `data/`: per fold,
1 - wMSE / wMSE(train mean), PDS - 0.5 and the three residual cosines, as
mean ± SD over folds. It reads `data/prediction_result`; with
`USE_REFERENCE=0` it reads `results/prediction/evaluation`.
