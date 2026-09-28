# 2_global_trend

The leading global trend: the first SVD factor of each dataset's
perturbation-effect matrix, fitted on all QC-passing and on the L1-selected
perturbations. The stage measures the noise-corrected signal energy it
carries, its perturbation loading against the target's DepMap
pan-essentiality and its concordance across datasets. The all-QC factor is
also the trend removed in `5_response_hierarchy` and `8_TF_targets`;
`3_cross_dataset_similarity` refits the factor itself.

| # | Script | What it does | Reads | Writes (`utils/paths.py`) |
|---|---|---|---|---|
| 1 | `fit_first_factor.py` | Leading SVD factor (ARPACK `svds`, tol 1e-7) of the perturbations x primary outcomes matrix, direct targets zeroed, for all QC-passing and for the L1-selected perturbations: u1/v1 loadings; sigma1 of the all-QC fits | `data/pseudobulk`, `QC_GENE_PANELS`, `L1_SELECTION` | `FIRST_FACTOR` |
| 2 | `compute_noise_corrected_signal_energy.py` | 100 lambda_max(K) / tr(K), K = YY^T - diag(rowSums(SE^2)), over the L1-selected perturbations with complete SE (dense `eigvalsh`) | `data/pseudobulk`, `QC_GENE_PANELS`, `L1_SELECTION` | `FIRST_FACTOR` |
| 3 | `compute_pan_essentiality_score.py` | Per gene, the 10th percentile, over the DepMap lines that scored it, of its within-line essentiality rank (scaled to [0, 1], 1 = most essential) | `data/essential/CRISPRGeneEffect.csv` | `PAN_ESSENTIALITY` |

`global_trend_figures.ipynb` draws into `figures/` (PDF) and `data/`
(plotted values):

- Fig 1b left: the signal energy of step 2.
- Fig 1b right: the all-QC \|u1\| of step 1, ranked within dataset, in ten
  equal-count bins of the target's pan-essentiality score (step 3; targets
  from `STRENGTH_SCORES`); only targets scored in every DepMap line (1,208)
  are binned.
- Supplementary Fig S1d: the coupled cosine cos(u1, u1') cos(v1, v1') of the
  L1-selected factors of step 1, per dataset pair over the shared
  perturbations and genes.

It reads `results_reference/` and the paper score
`data/essential/pan_essentiality_score.csv` (`PAPER_PAN_ESSENTIALITY`); with
`USE_REFERENCE=0` it reads `results/`.

```bash
bash 2_global_trend/run.sh
```

## Inputs not included

- DepMap Public 26Q1 `CRISPRGeneEffect.csv`, from the DepMap Data Portal
  (https://depmap.org/portal/data_page/), in `data/essential/`.
