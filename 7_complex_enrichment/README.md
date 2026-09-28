# 7_complex_enrichment

CORUM complex specificity of the residual-effect (conjunction) discoveries of
`5_response_hierarchy`. For each dataset and curated complex, outcome genes discovered
preferentially by the complex's perturbations are called complex-specific, and
their enrichment among the largest |residual effects| is measured (Methods,
"Protein-complex enrichment").

## Scripts

In run order. Results locations are the names in `utils/paths.py`. Both
scripts run in the `$CORUM_ENV` environment (pandas 2.2.3, needed to unpickle
the effect dict).

| Script | What it does | Reads | Writes |
|---|---|---|---|
| `define_complex_specific_genes.py` | The 27 curated CORUM complexes (ids fixed in `CURATED`) and their member genes; complex-specific genes (s > 0.25, s_adj > 0.25, z > 2.5, at least 3 discovering perturbations) of every complex with >= 3 represented perturbations, 12 datasets | `data/gene_set_list/corum_human.gmt`, `RESIDUAL_DISCOVERIES` | `COMPLEX_ENRICHMENT`: `corum_curated_complex_genes.csv`, `complex_specific_genes.csv` |
| `compute_magnitude_enrichment.py` | Complex-specific share among the top q% of discoveries by \|residual effect\| over the overall share, 61 log-spaced q from 100 to 1, 12 datasets (Fig 2e); the same top-5% enrichment for each dataset x complex, 11 datasets x 27 complexes (S5c) | `data/effect_dict/bc_bulk_qc_effect.pkl`, `RESIDUAL_DISCOVERIES`, `COMPLEX_ENRICHMENT` | `complex_specific_magnitude_enrichment/` and `complex_level_top5_enrichment/` under `COMPLEX_ENRICHMENT` |

## Panels

`complex_enrichment_figures.ipynb` draws Fig 2e (11 screens and their
pointwise median) and Supplementary Fig S5c (11 datasets x 27 complexes),
writing each panel to `figures/` (PDF) and its plotted values to `data/`
(CSV). It reads `results_reference/` by default; `USE_REFERENCE=0` reads
`results/`.

## How to run

```bash
bash 7_complex_enrichment/run.sh [--dry-run]
```

`run.sh` runs both scripts in `$CORUM_ENV` (default
`perturbation-structure-corum`) with 2 threads, after `5_response_hierarchy`.

## Notes

- **Inputs not included.** The CORUM 5.0 human complexes GMT
  (`data/gene_set_list/corum_human.gmt`; Steinkamp et al., *Nucleic Acids
  Res* 53, D651 (2025)) and the paper effect dict
  (`data/effect_dict/bc_bulk_qc_effect.pkl`).
- **Leading factor.** The discoveries come from `5_response_hierarchy`, tested
  against the `2_global_trend` factor. The magnitudes are a refit on the
  effect dict: direct targets zeroed, the largest of k = 6 `svds` triplets
  removed.
- **Dataset names.** The scripts use the registry keys of `utils/paths.py`:
  `effect_key()` gives the effect-dict key (`Feng-gw` for Feng-GW), `slug()`
  the `5_response_hierarchy` folder and `paper_name()` the `display_dataset`
  column.
