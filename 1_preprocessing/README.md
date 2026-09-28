# 1_preprocessing

Perturbation QC, primary outcome genes, top-2,000 feature genes, L1
perturbation strength and the strong perturbations, plus the QC tables and
effect dict of the prediction benchmark, from the crispyx pseudobulk moments
and Wilcoxon DE files. No figure notebook: the tables feed every panel.

| # | Script | What it does | Reads | Writes (`utils/paths.py`) |
|---|---|---|---|---|
| 1 | `build_qc_primary_features.py` | Perturbation QC (unique target, target control >= 0.08, inactivation efficiency >= 0.10), primary outcome genes (control or mean treated >= 0.08), top 2,000 effect-variance feature genes | `data/pseudobulk` | `QC_GENE_PANELS` |
| 2 | `compute_perturbation_strength.py` | Per-perturbation BH (alpha 0.05) over the eligible primary genes, target removed: `strength_L1`; then the strong perturbations, positive L1, top 2,000 per dataset (VCC 150) | `data/pseudobulk`, `data/de`, step 1 | `STRENGTH_SCORES`, `L1_SELECTION` |
| 3 | `python -m utils.qc` | The same QC rules on the 12 screens (VCC-subsampled uses those of VCC): gene panel and perturbation filter of the prediction benchmark | `data/pseudobulk` | `PREDICTION_QC_DIR` |
| 4 | `build_effect_dict.py` | `{dataset: perturbation x gene}` batch-corrected effects on the step-3 panel and filter; Feng-gwsf and Feng-gwsnf merged into Feng-GW | `data/pseudobulk`, step 3 (`--qc-dir`) | `EFFECT_DICT` |

Later stages read the paper copies of the step-3 and step-4 outputs,
`data/filter_result_bulk` and `data/effect_dict/bc_bulk_qc_effect.pkl`
(`PAPER_PREDICTION_QC_DIR`, `PAPER_EFFECT_DICT`). To use a rerun, point these
two constants at `PREDICTION_QC_DIR` and `EFFECT_DICT`.

```bash
bash 1_preprocessing/run.sh [--dry-run]
```

## Inputs not included

| Placeholder (README inside) | Produces | Panels |
|---|---|---|
| `crispyx_pseudobulk_de/` | moments (`data/pseudobulk`), Wilcoxon DE (`data/de`), single cells for GEARS and scGPT (`data/sc`) | all |
| `vcc_subsampling/` | VCC-subsampled moments and DE | S1b, Fig 2e |
| `split_halves/` | split-half moments (`data/split`) | Fig 2c, S5a |
