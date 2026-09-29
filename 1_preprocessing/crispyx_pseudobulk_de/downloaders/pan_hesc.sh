#!/usr/bin/env bash
# Pan-GW-hESC (Pan et al., bioRxiv 2026), GEO GSE295214: 85 Cell Ranger
# filtered_feature_bc_matrix.h5 files (gene expression + guide capture), 7.8 GB tar.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/../../../data"  # data/ of the repo
RAW_DIR="$DATA_DIR/raw/GSE295214"
mkdir -p "$RAW_DIR"
TAR="$RAW_DIR/GSE295214_RAW.tar"
if [[ ! -f "$TAR" ]]; then
    curl -L -f --retry 3 --retry-delay 30 --progress-bar -o "$TAR" \
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE295nnn/GSE295214/suppl/GSE295214_RAW.tar"
fi
if [[ $(find "$RAW_DIR" -maxdepth 1 -name "*.h5" | wc -l) -lt 85 ]]; then
    tar -xf "$TAR" -C "$RAW_DIR"
fi
echo "H5 files: $(find "$RAW_DIR" -maxdepth 1 -name "*.h5" | wc -l) (expected 85)"
