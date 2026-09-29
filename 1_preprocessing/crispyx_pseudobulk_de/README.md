# crispyx single cells, pseudobulk moments and Wilcoxon DE

Rebuilds the processed data of the 12 CRISPRi screens (13 files: Feng-GW-iPSC
is the two sub-screens Feng-gwsf and Feng-gwsnf) from the public sources: the
QC'd single cells, the per-perturbation moments (effects and standard errors)
and the Wilcoxon DE results (Methods, "Datasets and preprocessing",
"Perturbation-effect estimation"). The outputs are the files of the data
deposit, under the same names, in the `data/` folders the analysis stages read
(`utils/paths.py`). Every panel starts from these files.

## Setup

```bash
conda env create -f 1_preprocessing/crispyx_pseudobulk_de/environment.yml
conda activate perturbation-structure-preprocessing
```

Run the steps from the repo root. All of them require crispyx 0.1.4
(https://pypi.org/project/crispyx/) and check for it. The full set of screens
needs several TB of disk (make `data/` a link to a large disk if needed); the
largest screens (Huang-HCT116, Huang-HEK293T, Nourreddine-GW-ipsc) need several
hundred GB of RAM in steps 02-04.

## Steps

Each step takes `--dataset <name>` or `--dataset all` (the 13 screen files) and
skips outputs that already exist (`--force` rebuilds them).

| Step | Script | Writes |
|---|---|---|
| 01 | `01_download.py` | `data/raw/`: source files (`downloaders/*.sh`) |
| 02 | `02_setup.py` | `data/origin/<ds>.h5ad`: raw counts in a common schema (`setup/*.py`) |
| 03 | `03_sc_preprocess.py` | `data/sc/sc_<ds>.h5ad` (QC, normalized, log1p); `data/de/DE_<ds>_wilcoxon.h5ad` (pooled Wilcoxon) |
| 04b | `04b_batch_de.py` | `data/de/DE_<ds>_wilcoxon_batch_corrected.h5ad` (batch-stratified Wilcoxon) |
| 04 | `04_moments.py` | `data/pseudobulk/effect_<ds>_moments.h5ad` (effects, means, s.d., standard errors) |

```bash
D=1_preprocessing/crispyx_pseudobulk_de
python $D/01_download.py --dataset Replogle-E-k562
python $D/02_setup.py --dataset Replogle-E-k562
python $D/03_sc_preprocess.py --dataset Replogle-E-k562
python $D/04b_batch_de.py --dataset Replogle-E-k562
python $D/04_moments.py --dataset Replogle-E-k562
```

Steps 04b and 04 both read the output of step 03. The split halves
(`../split_halves`) and VCC-subsampled (`../vcc_subsampling`) reuse these
modules.

What steps 03-04 do for each screen:

1. Remove cells with fewer than 100 detected genes, keep perturbations with at
   least 20 remaining cells (and all control cells), remove genes detected in
   fewer than 100 kept cells; merge gene symbols that differ only in case.
2. Store each cell's library size (total count over the kept genes) as
   `obs['library_size']`, normalize to 1e4 counts per cell and apply `log1p`,
   so that `counts = expm1(X) * library_size / 1e4`.
3. Wilcoxon test of each perturbation against the control cells, pooled
   (step 03) and stratified by batch (van Elteren, step 04b); BH adjustment
   within each perturbation. Log fold changes and detection fractions are
   pooled in both.
4. Per perturbation and gene, combine the within-batch differences of means
   with the weights `w_b = n_pert n_ctrl / (n_pert + n_ctrl)` (the effect, `X`)
   and compute its standard error (`layers/effect_se`), with one pooled
   within-batch variance (`util_moments.py`).

## Datasets

`datasets.py` lists each dataset's control label, perturbation column and batch
column: Replogle-GW-k562, Replogle-E-k562, Replogle-E-rpe1, Nadig-JURKAT,
Nadig-HEPG2, Huang-HCT116, Huang-HEK293T, Feng-gwsf, Feng-gwsnf, Feng-ts,
Pan-GW-hESC, VCC, Nourreddine-GW-ipsc, and the derived VCC-subsampled. The QC
thresholds are the same for every dataset.

## Notes

- For the Feng screens the batch used for stratification is the joint
  batch x donor-line key `Batch_x_donor_line`. Step 04 would use `Batch` alone
  for a perturbation without a control cell in any joint stratum; no
  perturbation of the paper's data needed this (`obs['stratification']` of the
  moments files is `composite` throughout).
- Step 03 adds the per-cell columns `perturbation_type` (`CRISPRi`; the
  Replogle source label `CRISPR` is relabelled), `self_pert_lfc` and
  `self_pert_zscore` (log2 fold change and z-score of the targeted gene in its
  own pooled test), and registers them so that anndata and scanpy read them.
- `setup/feng.py` reads each Feng count matrix into memory. It was checked on
  small inputs and against the structure of the origin files of the paper, but
  not re-run end to end on the full source files.
- Steps 01 and 02 were written from the scripts that built the deposited files;
  of them, only `setup/passthrough.py` and `../vcc_subsampling` were re-run in
  the check below.

## Check against the deposited files

A test run on Replogle-E-rpe1 and VCC-subsampled compared every output of steps
02-05 with the deposited files: the origin files (step 02), the moments, both
DE files and the split-half moments (seed 0) were identical, and the single
cells had identical `X`, `var` and per-cell columns. "Identical" means every
array, AnnData element and HDF5 attribute is equal.
