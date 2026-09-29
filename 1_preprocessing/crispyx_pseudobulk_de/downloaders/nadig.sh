#!/usr/bin/env bash
# Nadig-HEPG2, Nadig-JURKAT (Nadig et al., Nature Genetics 2025), GEO GSE264667.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/../../../data"  # data/ of the repo
RAW_DIR="$DATA_DIR/raw"
mkdir -p "$RAW_DIR"
BASE="https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE264667&format=file"
download_one() {
    local dest="$RAW_DIR/$2"
    if [[ -f "$dest" ]]; then echo "Already exists: $dest"; return; fi
    echo "Downloading $1 -> $dest"
    curl -L -f --retry 5 --retry-delay 30 -C - --progress-bar -o "$dest" "$BASE&file=$1"
}
download_one "GSE264667%5Fhepg2%5Fraw%5Fsinglecell%5F01%2Eh5ad" "Nadig-HEPG2.h5ad"
download_one "GSE264667%5Fjurkat%5Fraw%5Fsinglecell%5F01%2Eh5ad" "Nadig-JURKAT.h5ad"
