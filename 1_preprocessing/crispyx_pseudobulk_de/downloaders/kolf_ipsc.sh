#!/usr/bin/env bash
# Nourreddine-GW-ipsc (KOLF2.1J genome-scale CRISPRi atlas; Nourreddine et al.,
# Nature Biotechnology 2026), Figshare+ 10.25452/figshare.plus.27261219:
# KOLF_Pan_Genome_QC_Filtered.h5ad (189 GB), checked by MD5.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/../../../data"  # data/ of the repo
RAW_DIR="$DATA_DIR/raw/KOLF2.1J_atlas"
mkdir -p "$RAW_DIR"
DEST="$RAW_DIR/KOLF_Pan_Genome_QC_Filtered.h5ad"
MD5="afd30fde1e6ad32969c29868394385d1"
if [[ ! -f "$DEST" ]]; then
    curl -L -f --retry 5 --retry-delay 30 -C - --progress-bar -o "$DEST" "https://ndownloader.figshare.com/files/64650261"
fi
[[ "$(md5sum "$DEST" | cut -d' ' -f1)" == "$MD5" ]] || { echo "ERROR: md5 mismatch for $DEST (re-run to resume)" >&2; exit 1; }
echo "md5 OK: $DEST"
