# 8_TF_targets

Residual-effect magnitudes of SCENIC+ targets and non-targets among the
conjunction discoveries of TF perturbations: Replogle-GW-K562 with the
K562-active network (Fig 2f), Nadig-HepG2 and Huang-HCT116 with the HepG2- and
HCT116-active networks (Supplementary Fig S5d). The residual is the moments
`X` minus the saved all-QC leading factor sigma1 u1 v1 of `2_global_trend`,
with the direct target missing; nothing is refitted.

| # | Script | What it does | Reads | Writes (`utils/paths.py`) |
|---|---|---|---|---|
| 1 | `compare_active_target_residual_magnitude.py` | Targets of each TF: the union of its eRegulons active in the cell line (AUC above the loom's `gaussian_mixture_split` threshold in more than half of the line's cells; no top-N cut). For every QC-passing TF perturbation in the network: n, mean and median \|residual\| of its rejected targets and rejected non-targets | `data/scenicplus`, `FIRST_FACTOR`, `RESIDUAL_DISCOVERIES` (`5_response_hierarchy`), `data/pseudobulk` | `TF_RESULTS_DIR` |

`TF_figures.ipynb` draws Fig 2f and S5d from the step-1 table into `figures/`
(PDF) and `data/` (plotted values), keeping TFs with >= 3 rejected targets and
>= 1 rejected non-target. It reads `results_reference/`; with
`USE_REFERENCE=0` it reads `results/`.

```bash
bash 8_TF_targets/run.sh
```

## Notes

- Not included: the SCENIC+ DPCL eRegulon loom
  `SCENIC+_DPCL_grnboost_gene_based_autoreg.loom` (Bravo Gonzalez-Blas et
  al., *Nat Methods* 2023), from SCope (scope.aertslab.org/#/scenic-v2,
  Session Looms > DPCL > SCENIC+), in `data/scenicplus/`.
- `data/pseudobulk` must hold the moments the factor was fitted on
  (`2_global_trend/fit_first_factor.py`).
- The DPCL network was inferred once, jointly over eight ENCODE cell lines;
  only the AUCell activity is line-specific. The active networks are subsets
  of it, not per-cell-line inferences.
- Non-targets are all other eligible primary outcomes, including genes absent
  from the network; they are not verified negatives.
