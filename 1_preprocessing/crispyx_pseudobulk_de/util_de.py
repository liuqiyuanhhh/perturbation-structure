"""Helpers shared by the DE and moment steps."""
from __future__ import annotations

import tempfile
from importlib.metadata import version
from pathlib import Path

from paths import TMP_DIR

CRISPYX_VERSION = "0.1.4"  # version used for the paper


def check_crispyx() -> None:
    installed = version("crispyx")
    if installed != CRISPYX_VERSION:
        raise SystemExit(f"crispyx {CRISPYX_VERSION} is required, found {installed}: "
                         f"pip install crispyx=={CRISPYX_VERSION}")


def set_scratch_dir() -> Path:
    """Send crispyx's temporary spill files to TMP_DIR (same disk as the data)."""
    d = TMP_DIR / "streaming"
    d.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(d)
    return d

