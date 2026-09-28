# crispyx pseudobulk moments and Wilcoxon DE (placeholder)

Missing: the crispyx 0.1.4 (https://pypi.org/project/crispyx/) run scripts,
configs and logs that made the moments, DE and single-cell files, and
`util_moments.py` (named in `uns/moments_note` of the moments files). Every
panel starts from these files.

What the step does for each screen (Methods, "Datasets and preprocessing",
"Perturbation-effect estimation"):

1. Remove cells with fewer than 100 detected genes, keep perturbations with at
   least 20 remaining cells (and all control cells), remove genes detected in
   fewer than 100 kept cells.
2. Normalize to 1e4 counts per cell and apply `log1p`.
3. Per perturbation and gene, combine the within-batch differences of means
   with the weights `w_b = n_pert n_ctrl / (n_pert + n_ctrl)` (the effect) and
   compute its standard error.
4. Run the batch-stratified Wilcoxon test against the control cells.

It must write, for each component (Replogle-E-k562, Replogle-GW-k562,
Replogle-E-rpe1, Nadig-HEPG2, Nadig-JURKAT, Huang-HCT116, Huang-HEK293T,
Pan-GW-hESC, VCC, Feng-ts, Feng-gwsf, Feng-gwsnf, Nourreddine-GW-ipsc):

- `data/pseudobulk/<component>_moments.h5ad`: `X` (effect), `layers/effect_se`,
  `layers/perturbation_profile`, `uns/control_profile`;
- `data/de/<component>_wilcoxon_batch_corrected.h5ad`: `layers/pvalue`, and
  the `.pkl` companion with the `scores` frame read by `4_prediction`;
- `data/sc/<screen>.h5ad`: QC'd, batch-corrected single cells for GEARS and
  scGPT.
