"""Shared helper: cache a CSC copy of an h5ad for column-streamed DE.

crispyx's Wilcoxon DE chunks by gene (axis=1). Every dataset in this pipeline
is stored CSR (the project's convention, fast for the cell/row-streaming QC
and normalization steps), so gene-chunked DE hits crispyx's slow path:
O(total_nnz) per chunk, ~100x slower than CSC (see
``crispyx.data._warn_slow_axis``). Converting once to CSC and caching the
result (keyed by source mtime, so a stale cache from before a `--force`
rebuild is detected and redone) pays that cost once per dataset rather than
once per DE run.

Used by ``03_sc_preprocess.py`` and ``04b_batch_de.py``.
"""
from __future__ import annotations

from pathlib import Path

import crispyx as cx


def ensure_csc(name: str, source_path: Path, csc_dir: Path) -> Path:
    """Return a CSC-format path for ``source_path``.

    If ``source_path`` is already CSC, returns it unchanged. Otherwise
    creates (or reuses a fresh, mtime-valid) CSC copy under
    ``csc_dir/{name}.h5ad``.
    """
    storage = cx.data.get_matrix_storage_format(source_path)
    if storage == "csc":
        return source_path

    output_path = csc_dir / f"{name}.h5ad"
    if output_path.exists() and output_path.stat().st_mtime >= source_path.stat().st_mtime:
        return output_path
    if output_path.exists():
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    converted = cx.data.convert_to_csc(
        source_path,
        output_path=output_path,
        verbose=True,
    )
    close = getattr(converted, "close", None)
    if callable(close):
        close()
    return output_path
