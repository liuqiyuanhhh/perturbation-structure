# 5_response_hierarchy

Whether perturbations share a hierarchy of responsive outcome genes (Fig 2a,
2b, 2d, S3, S4, S5b): the discovery matrices of every dataset, their nesting
across perturbations, and the Differential Expression Score (DES) of the
prediction methods.

- Total-effect discoveries: BH within each perturbation of the Wilcoxon
  p-values P^W of the primary outcomes (direct target missing).
- Residual-effect discoveries (conjunction): BH within each perturbation of
  max(P^W, P^Z), P^Z = 2 Phi(-|E - sigma1 u1 v1'| / SE), with sigma1 u1 v1'
  the all-QC leading factor of `2_global_trend`.

## Scripts

In run order. Results locations are the names in `utils/paths.py`.

| Script | What it does | Reads | Writes |
|---|---|---|---|
| `build_discovery_matrices.py` | Total- and residual-effect discovery matrices, and for S3a the counts of the total-effect conjunction max(P^W, P^0), P^0 = 2 Phi(-\|E\| / SE): `--mode analyze` per dataset (`--dataset-index` 0-11, 12 datasets), then `--mode aggregate` for the count tables | `data/de`, `data/pseudobulk`, `QC_GENE_PANELS`, `FIRST_FACTOR` | `TOTAL_DISCOVERIES`, `RESIDUAL_DISCOVERIES`, `RESIDUAL_DISCOVERIES_CANONICAL` (the residual matrices in the total-effect file layout, without Feng-GW) |
| `compute_matched_nestedness.py` | 100 stratified 70/30 splits; genes ranked by training discovery frequency; 20 x 20 total/residual triangles of the held-out perturbations with >= 20 total-effect discoveries (11 datasets) and held-out capture curves, equal-dataset mean over 10 datasets: analyze 0-10, then aggregate | `TOTAL_DISCOVERIES`, `RESIDUAL_DISCOVERIES_CANONICAL` | `triangle_visualization_polish/`, `capture_curve_comparison/` under `RESIDUAL_DISCOVERIES_CANONICAL` |
| `des_scoring/` (placeholder) | Per-perturbation DES scorer and residual-effect cutoff sweep | `data/prediction_result`, `data/effect_dict/bc_bulk_qc_effect.pkl` | `DES_PER_PERT`, `DES_THRESHOLD_PER_PERT` (`data/external/des/`) |
| `compute_breadth_stratified_des.py` | One dataset per call (`--dataset-index` 0-10): DES = \|top-k & D\| / k of every method, DE frequency and random under both truths (the leading factor of each prediction block removed by randomized SVD for the residual truth), and the median \|E\| and \|R\| over each perturbation's discoveries | `QC_GENE_PANELS`, `FIRST_FACTOR`, `TOTAL_DISCOVERIES`, `RESIDUAL_DISCOVERIES_CANONICAL`, `data/prediction_result`, `data/pseudobulk` | `per_dataset/` under `RESPONSE_BREADTH` |

## Panels

| Notebook | Panels | From |
|---|---|---|
| `nestedness_figures.ipynb` | Fig 2a, S3b | `compute_matched_nestedness.py` |
| `nestedness_figures.ipynb` | S3a (percent of Wilcoxon discoveries kept by the two conjunctions) | the count tables of `RESIDUAL_DISCOVERIES` |
| `DES_figures.ipynb` | Fig 2b, 2d, S3c, S5b (DES over unique held-out perturbations; relative DES across residual-effect cutoffs for perturbations with >= 20 total-effect discoveries) | `DES_PER_PERT`, `DES_THRESHOLD_PER_PERT` |
| `DES_figures.ipynb` | S4a-c (matched Low (1-10) / Broad (>= 20) pool, DES above random, datasets by C_d) | `compute_breadth_stratified_des.py` |

The discovery matrices are also the inputs of Fig 2e, S5c (`7_complex_enrichment`)
and Fig 2f, S5d (`8_TF_targets`). The notebooks write each panel to `figures/`
(PDF) and its plotted values to `data/` (CSV). They read `results_reference/`
by default; `USE_REFERENCE=0` reads `results/`.

## How to run

```bash
bash 5_response_hierarchy/run.sh [--dry-run]
```

`run.sh` runs the scripts in order with the paper parameters (the script
defaults), after `1_preprocessing` and `2_global_trend`. It skips the
placeholder `des_scoring/`; `DES_figures.ipynb` reads its paper tables from
`data/external/des/`. `compute_breadth_stratified_des.py` reads the prediction
pickles in `data/prediction_result`, the paper copies of the `4_prediction`
outputs.

## Notes

- **Frozen DES summaries.** With `USE_REFERENCE` on, `DES_figures.ipynb`
  plots the frozen summary tables (`DES_SOURCES`,
  `09_des_threshold/paper_figure_S5b/source_data/`, `paper_figure_S4/source_data/`
  under `RESPONSE_BREADTH`); `USE_REFERENCE=0` computes them from
  `DES_PER_PERT`, `DES_THRESHOLD_PER_PERT` and the `per_dataset/` tables of
  `RESPONSE_BREADTH`.
- **S4.** `compute_breadth_stratified_des.py` covers 11 datasets; S4 draws 10
  (no Nourreddine-iPSC).
- **Dataset keys.** The 70/30 splits are seeded from the dataset key and the
  saved row order, so the keys, the registry order and the per-dataset folder
  names must not change; stages 7 and 8 read these folders. The folders keep
  the labels of the frozen results (`slug()`, e.g. `Pan-GP-hESC`); the
  `display_dataset` columns hold the paper names.
