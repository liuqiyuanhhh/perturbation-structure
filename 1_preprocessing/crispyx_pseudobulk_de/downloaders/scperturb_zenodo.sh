#!/usr/bin/env bash
# Replogle-GW-k562, Replogle-E-k562, Replogle-E-rpe1 (Replogle et al., Cell 2022),
# as curated by scPerturb (Peidli et al., Nature Methods 2024; Zenodo record 10044268).
# Usage: bash downloaders/scperturb_zenodo.sh      (DATASETS="Replogle-E-k562" for a subset)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/../../../data"  # data/ of the repo
RAW_DIR="$DATA_DIR/raw"
mkdir -p "$RAW_DIR"
BASE="https://zenodo.org/records/10044268/files"
declare -A ZENODO_FILE=(
    ["Replogle-GW-k562"]="ReplogleWeissman2022_K562_gwps.h5ad"
    ["Replogle-E-k562"]="ReplogleWeissman2022_K562_essential.h5ad"
    ["Replogle-E-rpe1"]="ReplogleWeissman2022_rpe1.h5ad"
)
for name in ${DATASETS:-Replogle-GW-k562 Replogle-E-k562 Replogle-E-rpe1}; do
    fname="${ZENODO_FILE[$name]:?unknown dataset $name}"
    dest="$RAW_DIR/$name.h5ad"
    if [[ -f "$dest" ]]; then echo "Already exists: $dest"; continue; fi
    echo "Downloading $name <- $fname"
    curl -L -f --retry 5 --retry-delay 30 -C - --progress-bar -o "$dest" "$BASE/$fname?download=1"
done
