# 3_cross_dataset_similarity

Noise-corrected cosine similarity of perturbation effects between datasets,
for total effects and for residual effects after removing each dataset's
leading factor, its sensitivity to the perturbation and outcome sets and to
sequencing depth (VCC subsampling), and the cross-dataset concordance of L1
perturbation strength. Needs `1_preprocessing` only: the residual mode refits
the leading factor itself.

| # | Script | What it does | Reads | Writes (`utils/paths.py`) |
|---|---|---|---|---|
| 1 | `compute_noise_corrected_cosine.py --effect-mode total` | Per analysis panel A-D, dataset pair and shared perturbation, over the outcomes with a finite effect and SE in both (direct target excluded): Q = sum(E^2) - sum(SE^2) and Q / sum(E^2) on both sides, the raw cosine and the unclipped noise-corrected cosine sum(E E') / sqrt(Q Q') | `data/pseudobulk`, `QC_GENE_PANELS`, `L1_SELECTION` | `SIMILARITY_TOTAL_AUDIT` |
| 2 | `compute_noise_corrected_cosine.py --effect-mode first_svd_residual` | The same after removing each dataset's first SVD factor (`svds`, tol 1e-7); the total-effect SE is reused | as step 1 | `SIMILARITY_RESIDUAL_AUDIT` |

`similarity_figures.ipynb` computes the plotted values and draws them into
`figures/` (PDF) and `data/` (plotted values):

- Fig 1a, 1c (panel C) and S1a (panels A and D): matched total and residual
  matrices from steps 1 and 2. Per pair, the perturbations passing in both
  audits (finite values, Q > 0, fraction > 0.05 on both sides), cosines
  clipped to [-1, 1], equal-weight mean.
- S1b: the Panel C VCC perturbations retained against each of the 11 other
  datasets, the same IDs applied to VCC-subsampled; mean raw and mean clipped
  corrected cosines (steps 1, 2).
- S1c: pairwise Spearman correlation of `strength_L1` (`STRENGTH_SCORES`,
  `1_preprocessing`) over shared perturbations, BH q-values over the pairs;
  Pan-GW-hESC left out.

It reads `results_reference/`; with `USE_REFERENCE=0` it reads `results/`.

```bash
bash 3_cross_dataset_similarity/run.sh
```

## Notes

- The VCC-subsampled moments (steps 1, 2) come from the placeholder
  `1_preprocessing/vcc_subsampling`.
- Step 2 uses the same first-factor routine as `2_global_trend` but refits
  it.
- Two pass rules: the matched matrices require finite values, Q > 0 and a
  fraction > 0.05 on both sides; S1b checks only the fractions.
