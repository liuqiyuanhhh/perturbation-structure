# Transcriptional perturbation effects remain structured beyond the leading global trend

Code for the analyses and figures of the paper.

## Contents

| Folder | Contents | Figures |
|---|---|---|
| `utils/` | Shared helpers: file locations and dataset registry, effect matrices and leading-factor fit, QC rules, BH, figure style of the notebooks, shell helpers of the stage `run.sh` | |
| `1_preprocessing/` | Perturbation and gene QC, L1 perturbation strength, strong perturbations, prediction QC tables and effect dict | |
| `2_global_trend/` | Leading global trend: first-factor fits, noise-corrected signal energy, pan-essentiality | 1b, S1d |
| `3_cross_dataset_similarity/` | Noise-corrected cross-dataset similarity of total and residual effects | 1a, 1c, S1a–c |
| `4_prediction_benchmark/` | Prediction benchmark: GEARS, scGPT-ft, PRESAGE, Weighted, linear models, training mean | 1d, S2 |
| `5_response_hierarchy/` | Discovery matrices, gene-response hierarchy, DES, response breadth | 2a, 2b, 2d, S3, S4, S5b |
| `6_magnitude_structure/` | Cross-split residual products and rank-1 energy | 2c, S5a |
| `7_complex_enrichment/` | CORUM complex-specific enrichment | 2e, S5c |
| `8_TF_targets/` | Residual magnitudes of SCENIC+ TF targets | 2f, S5d |

Each stage folder has its scripts, a `run.sh` (`bash <folder>/run.sh`), the figure
notebooks, the plotted values in `data/`, and a README.

## Data

The processed data (pseudobulk moments `.h5ad`, differential-expression test results
and intermediate results) will be deposited on a public server; the link will be
added here. The scripts read them from `data/` (`utils/paths.py`).

## Third-party code

- GEARS (Roohani et al., *Nat Biotechnol* 2024) is used as the unmodified
  `cell-gears==0.1.2` package.
- PRESAGE (Littman et al., *bioRxiv* 2025) is not included; clone it from
  https://github.com/genentech/PRESAGE.
- scGPT (Cui et al., *Nat Methods* 2024) is used as the unmodified `scgpt==0.2.4` package.

## Citation

Citation details will be added on publication.
