# Split-half moments (placeholder)

Missing: the generator of the split-half moments, used by Fig 2c and S5a.

For each of five seeds, the cells of each perturbation and the control cells
are divided at random into two halves, and the crispyx moments
(`../crispyx_pseudobulk_de`, step 3) are computed in each half separately
(Methods, "Cross-split residual-product analysis").

It must write `data/split/{base}_seed{0-4}_half{0,1}_moments.h5ad`, in the
layout of the full-data moments: 10 datasets x 5 seeds x 2 halves = 100 files
for Replogle-E-k562, Replogle-E-rpe1, Pan-GW-hESC, Replogle-GW-k562,
Nadig-HEPG2, Nadig-JURKAT, Huang-HCT116, Huang-HEK293T, VCC and Feng-ts. `{base}` is the
moments component of each dataset (`DATASET_COMPONENTS` in `utils/paths.py`).
