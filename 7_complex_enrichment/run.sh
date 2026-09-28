#!/usr/bin/env bash
# Curated CORUM complexes and complex enrichment (corum environment).
#
# Usage: bash 7_complex_enrichment/run.sh [--dry-run]
#
# Needs 5_response_hierarchy, the CORUM GMT (data/gene_set_list/corum_human.gmt) and the
# effect dict (data/effect_dict/bc_bulk_qc_effect.pkl).  Both scripts run in
# the conda environment $CORUM_ENV (default perturbation-structure-corum),
# which is activated for them only.  Figure notebook:
# complex_enrichment_figures.ipynb (Fig 2e, S5c).

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../utils/run.sh" "$@"

section "CORUM complex enrichment (corum environment)"
threads 2
py_corum "$HERE/define_complex_specific_genes.py"
py_corum "$HERE/compute_magnitude_enrichment.py"
