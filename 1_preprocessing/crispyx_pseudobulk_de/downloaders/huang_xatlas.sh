#!/usr/bin/env bash
# Huang-HCT116, Huang-HEK293T (X-Atlas/Orion; Huang et al., bioRxiv 2025),
# Figshare+ article 29190726. About 550 GB.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/../../../data"  # data/ of the repo
RAW_DIR="$DATA_DIR/raw/huang_xatlas"
mkdir -p "$RAW_DIR"
ZIP="$RAW_DIR/Huang.zip"
if [[ ! -f "$ZIP" ]]; then
    curl -L -f --retry 5 --retry-delay 30 -C - --progress-bar -o "$ZIP" \
        "https://plus.figshare.com/ndownloader/articles/29190726/versions/1"
fi
extract() {
    local dest="$RAW_DIR/$2"
    if [[ -f "$dest" ]]; then echo "Already exists: $dest"; return; fi
    unzip -oj "$ZIP" "$1" -d "$RAW_DIR" && mv "$RAW_DIR/$1" "$dest"
}
extract "HCT116_filtered_dual_guide_cells.h5ad" "Huang-HCT116.h5ad"
extract "HEK293T_filtered_dual_guide_cells.h5ad" "Huang-HEK293T.h5ad"
