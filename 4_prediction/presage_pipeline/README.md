# PRESAGE (+Perturb-seq)

[PRESAGE](https://github.com/genentech/PRESAGE) (Littman, Levine et al.,
bioRxiv 2025.06.03.657653) trained on the single-cell GEARS build of each
target, on the same 5 CV folds as GEARS and scGPT-ft, with one extra
knowledge source per other Perturb-seq screen in the target's source pool:
the paper's "PRESAGE (+Perturb-seq)" variant (Fig 1d, S2). CPU only, in the
PRESAGE environment (`../envs/presage.txt`).

The upstream checkout (`data/external/PRESAGE`, commit `2c7b231`, with its
knowledge-source cache unpacked) is used unmodified; `03_train.py` patches it
at run time. One patch needs a file that is not distributed here: see
"Required change to PRESAGE" before running step 03.

The target is set with `PP_DATASET` (a key of `../targets.py`). `config.py`
takes every path from `../paths.py`; the working tree is
`paths.PRESAGE_WORK_DIR` (`results/prediction/presage`).

## Steps

| Step | Script | What it does | Writes (under the working tree) |
|---|---|---|---|
| 00 | `00_prep_dataset.py` | Links the GEARS build as PRESAGE's raw input; converts the cv5 folds of `targets.split_build` into PRESAGE split files (GEARS condition names harmonized, control dropped); writes the top-1,000 DE genes of every perturbation from the build's `rank_genes_groups_cov_all` | `data/<target>/perturb_processed.h5ad` (symlink), `data/<target>/degs/merged.degs.json`, `splits/<target>_random_splits/seed_{1..5}.json`, `splits/<target>_random_splits.perturbations.json` |
| 01 | `01_build_pert_sources.py [--force]` | One knowledge source per screen in the target's pool (`targets.source_pool`): its effect-dict matrix with control rows dropped and NaNs set to 0 (more than 1% NaN stops the step). No PCA here; `read_and_embed` does it | `artifacts/pert_sources/perturbseq.<screen>.pkl` (shared across targets), `manifest.<target>.json` |
| 02 | `02_make_source_list.py` | The knowledge-source list: 31 STRING and MSigDB 2023.2 sources and 9 others (Periscope x3, DepMap, Funk et al. OPS, GenePT x2, BioGPT, ESM2) in upstream's order, then the Perturb-seq sources of the manifest. Content-addressed, because `read_and_embed` keys its cache on the list's file name | `artifacts/source_lists/sources.<sha8>.txt`, `current.<target>.txt` |
| 03 | `03_train.py --fold F` | Trains PRESAGE on fold F and predicts the fold's test perturbations with the best-validation checkpoint; writes the global source weights | `output/predictions/<target>_seed_<F>_{all,test}_predictions.pkl`, `output/checkpoints/`, `output/lightning_logs/`, `output/attention/<target>_seed_<F>_global_source_weights.json`, `cache/pathway_embeddings/`, `data/<target>/<target>_processed.h5ad` |
| 04 | `04_collect_predictions.py` | Stacks the five folds' test predictions and checks that every perturbation is predicted exactly once | `output/predictions/<target>_presage_test_{predictions,concat}.pkl` |

Predictions are effects (control-subtracted) on the log1p(CPTT) scale of the
build. The evaluation reads the per-fold `<target>_seed_<F>_test_predictions.pkl`.
Fold 1 writes the dense `<target>_processed.h5ad` and the knowledge-tensor
cache that folds 2-5 reuse, so it runs alone first.

Training (step 03): upstream's `defaults_config.json` and
`singles_config.json`, overridden by `configs/default_config.json` (learning
rate 1.23e-3, weight decay 1e-15, batch size 16); 128 embedding dimensions
per source; early stopping on validation loss (patience 10), gradient
clipping at 0.1. The model has one channel per line of the source list plus
two computed from the training data (perturbation x gene pseudobulk and
co-expression).

Runtime patches installed by `03_train.py` (nothing in the checkout is
edited):

1. the dataloaders run in-process (`num_workers=0`);
2. `presage.read_and_embed` is replaced by the version described below;
3. a `var` index name that collides with a `var` column is cleared on read;
4. `obs` gets the `gene` column that PRESAGE's evaluator groups by.

## Running

`../run.sh` submits all of it (the `presage_prep` jobs one at a time, since
step 01 writes files shared across targets). By hand, from the repository
root:

```bash
PP_DATASET=VCC sbatch 4_prediction/slurm/presage_prep.sbatch             # 00-02
PP_DATASET=VCC sbatch --array=1   4_prediction/slurm/presage_train.sbatch
PP_DATASET=VCC sbatch --array=2-5 4_prediction/slurm/presage_train.sbatch  # after fold 1
PP_DATASET=VCC sbatch 4_prediction/slurm/presage_collect.sbatch          # 04
```

Step 00 needs the GEARS builds of `targets.presage_build` and
`targets.split_build` (`../gears_scgpt`). Memory per fold: `../run.sh`.

## Required change to PRESAGE

`03_train.py` imports `make_patched_read_and_embed` from
`presage_pipeline/patched_read_and_embed.py` and installs the function it
returns as `presage.read_and_embed`. That file is a modified copy of
`read_and_embed` from PRESAGE's `src/presage.py` (commit `2c7b231`). PRESAGE
is distributed under the Genentech Non-Commercial Software License v1.0, so
the file is not included here, and `03_train.py` stops with a message that
points to this section when it is missing. To run step 03, write it from your
own PRESAGE checkout, keeping Genentech's copyright and license notice:

- **Interface.** A factory `make_patched_read_and_embed(presage_mod,
  cache_dir_override=None)` returns a replacement with the same arguments as
  upstream's `read_and_embed` (the source-list file, the genes to keep, the
  embedding dimension, the model config and the datamodule) and the same
  return value, the genes x dimensions x channels knowledge tensor.
  `presage_mod` is the imported `presage` module: the replacement calls the
  module's helpers (the embeddings from the training data, the
  knowledge-graph processing, the sparse-frame reader) through it at call
  time instead of importing them, so a patch on any of them still applies.
- **The change to the computation.** Upstream allocates the per-source
  genes x dimensions zero matrix once, before the loop over the lines of the
  source list. Each source then overwrites only the rows of the genes it
  covers, and the whole matrix is appended as that source's channel. Channel
  i therefore holds sources 1 to i stacked, with the last write winning, and
  the per-source mask PRESAGE derives from "this channel is all zero for this
  gene" no longer marks which sources cover a gene. The copy allocates a
  fresh zero matrix, indexed by the genes to keep, inside the loop, once per
  source. This is the only change to what is computed. Upstream's 40 sources
  each cover nearly every gene, so it rarely matters there; a Perturb-seq
  source covers only the genes its screen perturbed (a few hundred to a few
  thousand), and without the change its channel would mostly hold the values
  of earlier sources.
- **Edits that leave the tensor unchanged.** The embedding cache is written
  to `cache_dir_override` when it is given (`03_train.py` passes
  `<working tree>/cache/pathway_embeddings`, created if needed) instead of
  the relative `./cache/pathway_embeddings/`; blank lines of the source list
  are skipped; after the loop, the number of genes each source covers is
  printed, so the per-source coverage can be checked in the job log.

Everything else follows upstream: the cache key, the two training-data
channels in front, the removal of constant features, jitter, standardization,
PCA and zero padding of each `.pkl` source, node2vec for knowledge-graph
sources, and the reuse of a cached tensor.
