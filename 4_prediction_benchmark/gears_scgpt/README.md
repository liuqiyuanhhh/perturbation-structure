# GEARS and scGPT-ft

GEARS data builds, the 5-fold cross-validation splits every method uses, and
per-fold training and prediction for GEARS and scGPT fine-tuned (Fig 1d, S2).
Builds and GEARS run in the GEARS environment (`../envs/gears.txt`), scGPT in
the scGPT environment (`../envs/scgpt.txt`). Both use the installed
`cell-gears==0.1.2` (`from gears import PertData, GEARS`); the scGPT
environment installs it with `--no-deps`, since `scgpt==0.2.4` declares
`cell-gears<0.0.3`.

## Scripts

| Script | What it does | Reads | Writes |
|---|---|---|---|
| `prepare_data.py --dataset B` | Builds B: maps the obs columns to `perturbation` / `batch` / `guide_id`; keeps control cells and QC-passing perturbations; optionally caps the controls (batch-stratified, seed 0); keeps the primary-outcome gene panel; writes GEARS conventions and runs `PertData.new_data_process` (DE rankings, cell graphs). Then cuts its 5 folds: 5 disjoint test folds (seed 0), 10% of each fold's training pool becomes validation (seed 100 + fold); conditions outside the GEARS perturbation graph are dropped first, as `PertData.load` would | `data/sc`, `data/filter_result_bulk` | `gears_data/B/B/{perturb_processed.h5ad, data_pyg/}`, `gears_data/B/B/splits/B_cv5_{1..5}.pkl`, `B_cv5_manifest.json`, `gears_data/B/B_qc_report.json` |
| `prepare_data.py --parent P --fold F` | Reduced build of one fold: the fold's test perturbations with all of their cells, a third of its training and validation perturbations (seed 0 + fold), 50,000 control cells stratified by batch. QC, gene panel and DE rankings are inherited from the parent | the parent build and its fold file | `gears_data/P-third-fF/...` |
| `prepare_data.py --go-graph` | Rebuilds `go_essential_all/go_essential_all.csv` offline from `gene2go_all.pkl` and `essential_all_data_pert_genes.pkl` (Jaccard of GO terms > 0.1, as in GEARS) | the two pickles in `data/gears` | `data/gears/go_essential_all/` |
| `run_gears.py --dataset B-cv5 --seed F` | Trains GEARS on fold F and predicts the fold's test conditions | the build | `predictions/gears_scgpt/B-cv5_F_gears_post-{gt,pred}.csv`, `checkpoints/` |
| `run_scgpt.py --dataset B-cv5 --seed F --finetune --train` | Fine-tunes the scGPT whole-human checkpoint on fold F and predicts each test perturbation from a pool of control cells | the build, `data/models/scGPT_human` | `predictions/gears_scgpt/B-cv5_F_scgpt_ft_post-{gt,pred}.csv`, `checkpoints/` |

`prepare_data.py` also holds the registry of builds (`DATASETS`), the CV
constants and seeds, and `get_pert_data`, which the two runners use to load
`<build>-cv5` under one fold. The QC tables are the paper tables of
`1_preprocessing`, read with `utils.qc` and `utils.effects`; nothing is
recomputed. `--force` rebuilds and rewrites the fold files; without it an
existing build or complete set of fold files is left alone.

`gears_data/` and `predictions/gears_scgpt/` are under `results/prediction`
(`paths.GEARS_DATA_DIR`, `paths.GEARS_PRED_DIR`).
`--seed` is the fold number (1-5), not a random seed.

## Builds

| Build | Single-cell file(s) | Notes |
|---|---|---|
| `VCC`, `Pan-GW-hESC`, `Replogle-E-k562`, `Replogle-E-rpe1`, `Replogle-GW-k562`, `Nadig-HEPG2`, `Nadig-JURKAT`, `Feng-ts`, `Nourreddine-GW-ipsc`, `Huang-HCT116`, `Huang-HEK293T` | `{screen}.h5ad` | |
| `Feng-gw` | `Feng-gwsf.h5ad` + `Feng-gwsnf.h5ad` | two halves of one genome-wide screen, concatenated on their common genes |
| `Feng-gw-control-cap` | same | `Feng-gw` with 50,000 of its ~500,000 control cells; identical folds |
| `{Huang-HCT116, Huang-HEK293T, Nourreddine-GW-ipsc}-third-f{1..5}` | the parent build | one reduced build per fold (`prepare_data.py --parent`) |

## Training

- **GEARS** (`slurm/train_gears.sbatch`): 50 epochs, hidden size 64, batch
  size 32, learning rate 1e-3; the epoch with the lowest validation top-20
  DE MSE is kept.
- **scGPT-ft** (`slurm/train_scgpt.sbatch`): starts from `best_model.pt` of
  the whole-human checkpoint (encoder, value encoder and transformer
  weights); up to 20 epochs, learning rate 1e-4 decayed by 0.9 per epoch,
  batch size 32, mixed precision, early stopping on validation MSE
  (patience 5). Training batches use a random 1,536 of the panel genes when
  the panel is larger.
  Predictions come from the best-validation weights, averaged over 100
  control cells per perturbation. 24 GB of GPU memory is enough.

Both write, for every test condition, the mean observed expression
(`post-gt`) and the prediction (`post-pred`): absolute post-perturbation
expression on the panel genes, indexed by `(condition, n_train)`. The
evaluation turns them into effects by subtracting a control profile
(`../evaluation/evaluate.py`, `../README.md`). GEARS skips a
condition whose gene is outside its perturbation graph.

## Running

`../run.sh` submits all of it. By hand, from the repository root:

```bash
DATASET=VCC sbatch 4_prediction_benchmark/slurm/gears_prep.sbatch          # prepare_data.py --dataset
PARENT=Huang-HCT116 sbatch 4_prediction_benchmark/slurm/gears_fold_subset.sbatch  # prepare_data.py --parent --fold
BUILD=VCC sbatch 4_prediction_benchmark/slurm/train_gears.sbatch
BUILD=VCC sbatch 4_prediction_benchmark/slurm/train_scgpt.sbatch
BUILD='Huang-HCT116-third-f{fold}' sbatch 4_prediction_benchmark/slurm/train_gears.sbatch
```

Build the fold subsets after their parent. Memory per build: `../run.sh`.

## GEARS support files

`PertData` downloads `gene2go_all.pkl`, and `prepare_split` downloads
`essential_all_data_pert_genes.pkl` and the `go_essential_all` GO graph,
from Harvard Dataverse. On a node without network access the download
writes a zero-byte file and the next `pickle.load` fails. `prepare_data.py`
therefore copies the pickles and links the GO graph from
`data/gears` (or from a build that already has them) before GEARS
looks for them; `prepare_data.py --go-graph` rebuilds the GO graph when only
the two pickles are at hand.

## GEARS package

Two details of `cell-gears==0.1.2` are handled in the scripts:
`PertData.new_data_process` lower-cases the build folder name, so
`prepare_data.py` renames the folder back; and `GEARS.train()` evaluates the
test set at the end of training, so `run_gears.py` removes the test loader
first and writes the test predictions itself.
