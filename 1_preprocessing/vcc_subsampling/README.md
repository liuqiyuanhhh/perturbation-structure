# VCC UMI and cell downsampling

VCC-subsampled, the technical control of Supplementary Fig S1b (noise-correction
check; Methods, "Sensitivity analyses"). No other panel uses it.

`downsample_vcc.py` downsamples the raw counts of VCC-H1
(`data/origin/VCC.h5ad`, steps 01-02 of `../crispyx_pseudobulk_de`) in cells and
UMIs, with random seed 0 throughout:

1. keep 16 of the 48 batches at random;
2. sample up to 100 cells per perturbation, keeping all control cells of these
   batches;
3. thin each cell's counts to at most 13,000 UMIs, sampling without
   replacement.

The result is then processed like the other screens:

```bash
D=1_preprocessing/crispyx_pseudobulk_de
python $D/01_download.py --dataset VCC
python $D/02_setup.py --dataset VCC
python 1_preprocessing/vcc_subsampling/downsample_vcc.py   # data/origin/VCC-subsampled.h5ad
python $D/03_sc_preprocess.py --dataset VCC-subsampled
python $D/04b_batch_de.py --dataset VCC-subsampled
python $D/04_moments.py --dataset VCC-subsampled             # data/pseudobulk/effect_VCC-subsampled_moments.h5ad
```

It uses the environment of `../crispyx_pseudobulk_de`. VCC-subsampled has no
split halves.

`3_cross_dataset_similarity/compute_noise_corrected_cosine.py` reads its moments
and gives VCC-subsampled the VCC QC, gene-panel and L1 memberships
(`SELECTION_REFERENCE_DATASET` in `utils/paths.py`); only the effects and their
SEs are its own.
