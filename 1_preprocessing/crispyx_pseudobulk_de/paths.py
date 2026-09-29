"""Data locations of the preprocessing steps.

The processed files go where the analysis stages read them (utils/paths.py), under
the file names of the data deposit; the downloads, the standardized counts and the
scratch files sit next to them in data/. The full set of screens needs several TB:
make data/ a link to a large disk if needed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repo root, for utils
from utils.paths import (  # noqa: E402,F401
    DATA as DATA_DIR,
    DE_DIR,
    MOMENTS_DIR,
    SC_DIR,
    SPLIT_DIR,
    de_file,
    moments_file,
    sc_file,
    split_assignments_file,
    split_file,
)

RAW_DIR = DATA_DIR / "raw"          # step 01: downloaded files
ORIGIN_DIR = DATA_DIR / "origin"    # step 02: standardized counts, <dataset>.h5ad
TMP_DIR = DATA_DIR / "_tmp"         # scratch (crispyx spill files, caches)
