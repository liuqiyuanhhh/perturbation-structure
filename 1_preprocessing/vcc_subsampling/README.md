# VCC UMI and cell downsampling (placeholder)

Missing: the code that made VCC-subsampled, the technical control of
Supplementary Fig S1b (noise-correction check). No other panel uses it.

It downsamples VCC-H1 in both UMIs and cells (Methods, "Sensitivity
analyses") and runs the crispyx moments step of `../crispyx_pseudobulk_de`
on the downsampled data.

It must write, with the perturbation and gene axes of VCC,
`data/pseudobulk/VCC-subsampled_moments.h5ad` (layout of the other moments
files).

`3_cross_dataset_similarity/compute_noise_corrected_cosine.py` reads it and
gives VCC-subsampled the VCC QC, gene-panel and L1 memberships
(`SELECTION_REFERENCE_DATASET` in `utils/paths.py`); only the effects and
their SEs are its own.
