# VCC UMI and cell downsampling (placeholder)

Missing: the code that made VCC-subsampled, the technical control of
Supplementary Fig S1b (noise-correction check) and of the VCC-subsampled curve
in Fig 2e.

It downsamples VCC-H1 in both UMIs and cells (Methods, "Sensitivity
analyses") and runs the crispyx moments and Wilcoxon steps of
`../crispyx_pseudobulk_de` on the downsampled data.

It must write, with the perturbation and gene axes of VCC:

- `data/pseudobulk/VCC-subsampled_moments.h5ad` (layout of the other moments
  files);
- `data/de/VCC-subsampled_wilcoxon_batch_corrected.h5ad`.

The analysis gives VCC-subsampled the VCC QC, gene-panel and L1 memberships
(`SELECTION_REFERENCE_DATASET` in `utils/paths.py`) and recomputes only the
effect-dependent quantities.
