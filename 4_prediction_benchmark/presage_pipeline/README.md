# PRESAGE (+Perturb-seq)

PRESAGE (Littman et al., *bioRxiv* 2025) with one extra knowledge source per other
Perturb-seq screen in the target's source pool, trained on the same 5 CV folds as GEARS and
scGPT-ft (Fig 1d, S2). The target is set with `PP_DATASET`; `../run.sh` submits every step.

| Step | Script | What it does |
|---|---|---|
| 00 | `00_prep_dataset.py` | PRESAGE input and split files from the GEARS build and its folds |
| 01 | `01_build_pert_sources.py` | One knowledge source per screen in the source pool |
| 02 | `02_make_source_list.py` | The knowledge-source list: PRESAGE's 40 sources, then the Perturb-seq sources |
| 03 | `03_train.py --fold F` | Trains on fold F and predicts its test perturbations |
| 04 | `04_collect_predictions.py` | Stacks the test predictions of the five folds |

## Required change to PRESAGE

`03_train.py` needs `patched_read_and_embed.py` (not included): a copy of PRESAGE's
`read_and_embed` that allocates the genes x dimensions zero matrix once per knowledge source,
inside the loop over sources, so that each channel holds only its own source. It defines
`make_patched_read_and_embed(presage_mod, cache_dir_override=None)`, which returns a function
with the arguments and return value of `read_and_embed` and writes its cache to
`cache_dir_override`.
