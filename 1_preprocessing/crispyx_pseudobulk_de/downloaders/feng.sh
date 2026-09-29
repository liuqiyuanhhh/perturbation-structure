#!/usr/bin/env bash
# Feng-gwsf, Feng-gwsnf, Feng-ts (Feng et al., Cell Genomics 2026), Figshare
# 10.6084/m9.figshare.27989294.v2
# (count-matrix CSVs; the parsed origin files are 47-170 GB each).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/../../../data"  # data/ of the repo
RAW_DIR="$DATA_DIR/raw/feng"
mkdir -p "$RAW_DIR"
ZIP="$RAW_DIR/Feng.zip"
if [[ ! -f "$ZIP" ]]; then
    curl -L -f --retry 5 --retry-delay 30 -C - --progress-bar \
        -A "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36" \
        -o "$ZIP" "https://figshare.com/ndownloader/articles/27989294/versions/2"
fi
cd "$RAW_DIR" && unzip -o Feng.zip && { gunzip -f -- *.gz 2>/dev/null || true; }
