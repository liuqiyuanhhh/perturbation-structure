# Split-half moments

Split-half replicates of the moments, used by Fig 2c and S5a
(`6_magnitude_structure`; Methods, "Cross-split residual-product analysis").

For each of five seeds (0-4), the cells of each perturbation and the control
cells are divided at random into two halves within batches, and the moments of
`../crispyx_pseudobulk_de` (step 04) are computed in each half separately.

| Script | Reads | Writes |
|---|---|---|
| `split_halves.py` | `data/sc/sc_<ds>.h5ad`, `data/split/split_<ds>_assignments.csv.gz` | `data/split/split_<ds>_seed<s>_half<h>_moments.h5ad` |

```bash
python 1_preprocessing/split_halves/split_halves.py --dataset Replogle-E-k562
python 1_preprocessing/split_halves/split_halves.py --dataset all
```

It uses the environment of `../crispyx_pseudobulk_de` and runs after its step 03.

- **The paper's halves.** By default the halves are rebuilt from the deposited
  cell assignments, `split_<ds>_assignments.csv.gz`: one row per cell of the
  single-cell file, the cell ID in `cell` and the half (0 or 1) of each seed in
  `seed0`-`seed4`.
- **A new split.** `--random` draws the halves with `crispyx.pp.subsample`
  (half of the cells within each batch, `random_state` = seed; half 1 is the
  complement) and writes their assignment file. The paper's split was drawn
  this way with crispyx 0.1.2.
- `--dataset all` covers the 13 screen files; each gets 10 files on the axes of
  its full-data moments. `6_magnitude_structure` uses 10 datasets:
  Replogle-E-k562, Replogle-E-rpe1, Pan-GW-hESC, Replogle-GW-k562,
  Nadig-HEPG2, Nadig-JURKAT, Huang-HCT116, Huang-HEK293T, VCC and Feng-ts.
