#!/usr/bin/env bash
# VCC (Arc Virtual Cell Challenge 2025; Roohani et al., Cell 2025): full release
# (training, validation and test sets), checked against the published file sizes.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/../../../data"  # data/ of the repo
RAW_DIR="$DATA_DIR/raw/vcc_data/2025"
BASE="https://storage.googleapis.com/arc-institute-virtual-cell-atlas/virtual-cell-challenge/2025"
download_one() {
    local dest="$RAW_DIR/$1"
    mkdir -p "$(dirname "$dest")"
    if [[ -f "$dest" && "$(stat -c %s "$dest")" != "$2" ]]; then rm -f "$dest"; fi
    if [[ ! -f "$dest" ]]; then
        curl -L -f --retry 3 --retry-delay 30 --progress-bar -o "$dest" "$BASE/$1"
    fi
    [[ "$(stat -c %s "$dest")" == "$2" ]] || { echo "ERROR: size mismatch for $1" >&2; exit 1; }
}
download_one "gene_names.csv" "116023"
download_one "train/adata_Training.h5ad" "15482497461"
download_one "train/pert_counts_Training.csv" "2824"
download_one "validation/adata_Validation.h5ad" "6928967541"
download_one "validation/pert_counts_Validation.csv" "978"
download_one "test/adata_Test.h5ad" "11950739168"
download_one "test/pert_counts_Test.csv" "1894"
