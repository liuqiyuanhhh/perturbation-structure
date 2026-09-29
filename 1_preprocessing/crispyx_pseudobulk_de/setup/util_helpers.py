"""Shared setup helpers: obs decoding, schema validation and verbatim promotion.

Some source files already carry the expected obs schema (the scPerturb Replogle
releases, X-Atlas/Orion). For these, setup is a validated move from data/raw/ to
data/origin/: check that the perturbation column and control label are present,
then move the file (no content change).

full_column / column_values decode the AnnData obs encodings found across
the sources: per-column categorical (group with codes/categories), the
legacy shared obs/__categories/<column> layout, and plain arrays.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import h5py
import numpy as np


def _decode(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def full_column(obs_grp: h5py.Group, column: str) -> np.ndarray:
    """Return every (decoded) value of an obs column, not just the unique set."""
    col = obs_grp[column]
    if isinstance(col, h5py.Group):
        if "categories" not in col:
            raise ValueError(f"obs[{column!r}] is a group without 'categories'")
        cats = np.array([_decode(c) for c in col["categories"][:]])
        codes = col["codes"][:]
        out = np.empty(codes.shape[0], dtype=object)
        valid = codes >= 0
        out[valid] = cats[codes[valid]]
        out[~valid] = None
        return out

    legacy = obs_grp.get("__categories")
    if legacy is not None and column in legacy and col.dtype.kind in ("i", "u"):
        cats = np.array([_decode(c) for c in legacy[column][:]])
        return cats[col[:]]

    values = col[:]
    if values.dtype.kind in ("S", "O"):
        return np.array([_decode(v) for v in values])
    return values


def column_values(obs_grp: h5py.Group, column: str) -> np.ndarray:
    """Return the decoded unique value set of an obs column."""
    return np.unique(full_column(obs_grp, column))


def validate_schema(path: Path, *, pert_col: str, control_label: str) -> None:
    """Raise a clear error if the expected perturbation/control shape is absent."""
    with h5py.File(path, "r") as f:
        if "obs" not in f:
            raise ValueError(f"{path}: no /obs group -- not a valid AnnData h5ad")
        obs = f["obs"]
        if pert_col not in obs:
            raise ValueError(
                f"{path}: obs is missing expected perturbation column {pert_col!r} "
                f"(found: {sorted(obs.keys())})"
            )
        values = column_values(obs, pert_col)
        if not any(v == control_label for v in values):
            raise ValueError(
                f"{path}: obs[{pert_col!r}] has no value == {control_label!r} "
                f"(this dataset's control label) -- wrong file, or the "
                f"upstream schema changed."
            )


def promote_verbatim(
    name: str,
    raw_path: Path,
    origin_path: Path,
    *,
    pert_col: str,
    control_label: str,
    force: bool = False,
) -> None:
    """Validate ``raw_path`` against the expected schema, then move it to
    ``origin_path`` (no content change)."""
    if not raw_path.exists():
        raise FileNotFoundError(f"{name}: raw input not found: {raw_path}")
    if origin_path.exists() and not force:
        raise FileExistsError(f"{name}: origin exists (pass --force): {origin_path}")

    validate_schema(raw_path, pert_col=pert_col, control_label=control_label)

    origin_path.parent.mkdir(parents=True, exist_ok=True)
    if origin_path.exists():
        origin_path.unlink()
    try:
        raw_path.rename(origin_path)  # same filesystem: instant
    except OSError:
        shutil.move(str(raw_path), str(origin_path))  # cross-filesystem fallback
