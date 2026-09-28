# 6_magnitude_structure

Whether residual effects stay organized in magnitude after their signed
low-rank structure is removed (Fig 2c, S5a). Observed magnitudes are biased by
noise, so the magnitude is measured with cross-split residual products, whose
expectation is the squared residual effect (Methods, "Cross-split
residual-product analysis").

For each dataset, five seeds give two split-half effect matrices each. After
removing k = 0..15 leading factors in every half, F_1 = sigma_1^2 / ||A||_F^2
of the mean residual matrix (signed residual effects) is compared with that of
the mean within-seed half0 x half1 product (cross-split residual product).
Each averaged matrix is capped at its 99.99th |entry| percentile (direct
targets excluded, signs kept) before its final SVD.

## Script

Results locations are the names in `utils/paths.py`.

| Script | What it does | Reads | Writes |
|---|---|---|---|
| `analyze_cross_split_dataset.py --dataset D` | One dataset per call: aligns the 10 halves, removes k = 0..15 factors, caps and computes both rank-1 energies | `data/split`, `QC_GENE_PANELS` | `CROSS_SPLIT/per_dataset/<D>/`: `rank1_comparison.csv`, `run_parameters.json` |

## Panels

`magnitude_structure_figures.ipynb` builds the panel tables from the per-dataset
`rank1_comparison.csv` files (with the paper names) and draws Fig 2c
(Replogle-E-K562, Replogle-E-RPE1, Pan-GP-H1) and Supplementary Fig S5a (the
seven other screens), writing each panel to `figures/` (PDF) and its plotted
values to `data/` (CSV). It reads `results_reference/` by default;
`USE_REFERENCE=0` reads `results/`.

## How to run

```bash
bash 6_magnitude_structure/run.sh
```

`run.sh` runs `analyze_cross_split_dataset.py` for the 10 panel datasets, after
`1_preprocessing`.

## Notes

- **Split halves not included.** The split-half moments in `data/split`
  (10 datasets x 5 seeds x 2 halves) come from the placeholder
  `1_preprocessing/split_halves`.
- **Folder names.** The per-dataset folders are the dataset keys with `_` for
  `-` (`Replogle_GW_k562`, `Pan_GW_hESC`), not the `5_response_hierarchy` folder names. The
  `display_dataset` column holds the paper names (`utils.paths.paper_name`).
