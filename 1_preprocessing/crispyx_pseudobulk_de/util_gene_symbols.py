"""Utilities for harmonising gene symbols during data preprocessing."""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp


def _sum_columns(values: Any, group_index: np.ndarray, n_groups: int) -> Any:
    """Return `values` with columns summed by `group_index`."""
    if values.ndim == 1:
        out = np.zeros(n_groups, dtype=values.dtype)
        np.add.at(out, group_index, values)
        return out

    if sp.issparse(values):
        mapper = sp.csr_matrix(
            (
                np.ones(len(group_index), dtype=values.dtype),
                (np.arange(len(group_index)), group_index),
            ),
            shape=(len(group_index), n_groups),
        )
        return values @ mapper

    out = np.zeros((values.shape[0], n_groups), dtype=values.dtype)
    for src, dst in enumerate(group_index):
        out[:, dst] += values[:, src]
    return out


def collapse_duplicate_gene_symbols(
    values: Any,
    gene_symbols: Any,
    var: pd.DataFrame | None = None,
    *,
    log1p_space: bool = False,
) -> tuple[Any, pd.DataFrame, dict]:
    """Collapse duplicate gene-symbol columns by summing their values.

    Parameters
    ----------
    values
        1D control vector or 2D observation x gene matrix.
    gene_symbols
        Effective gene symbols for the columns in `values`.
    var
        Optional per-feature metadata aligned to `gene_symbols`.
    log1p_space
        Set when `values` holds ``log1p(normalize_total(X))``-scale mean
        expression (the convention documented throughout
        ``04_pseudobulk.py``/``05_prepare_bulk_data.py``), NOT a difference
        of two such values (an effect/logFC) and NOT a variance. Summing
        ``log1p(a) + log1p(b)`` approximates ``log(a*b)``, not ``log(a+b)``,
        which inflates a collapsed gene's expression and corrupts any effect
        derived from it. When True, values are mapped through ``expm1``
        before summing and back through ``log1p`` after, so two collapsed
        features' *linear-scale* expression is what actually adds. Leave
        False for quantities that are already correctly additive in the
        space they're given (e.g. raw counts, or a variance the caller has
        already converted to before calling this, since Var(X+Y) =
        Var(X)+Var(Y) for independent X, Y needs no further transform). An
        effect/logFC matrix must not be collapsed directly in either mode —
        reconstruct absolute expression, collapse that, then re-derive the
        effect from the collapsed absolute values (see
        ``05_prepare_bulk_data.py::_load_bulk_pooled``).

    Matching is **case-insensitive** (e.g. ``C2orf15`` and ``C2ORF15`` collapse
    to one column) — no official HGNC symbol is distinguished from another
    purely by letter case, so this never merges genuinely distinct genes; it
    only catches the same gene spelled inconsistently across reference
    annotations. The surviving display symbol is the first occurrence's
    original casing.

    Returns
    -------
    collapsed_values, collapsed_var, summary
        `collapsed_var.index` contains unique gene symbols (original casing
        of each symbol's first occurrence). For symbols produced from
        multiple input features, metadata is taken from the first feature
        and `collapsed_feature_ids` records all merged feature identifiers
        separated by `|`.
    """
    gene_index = pd.Index(gene_symbols).astype(str)
    if values.shape[-1] != len(gene_index):
        raise ValueError(
            f"values has {values.shape[-1]} columns, but got "
            f"{len(gene_index)} gene symbols"
        )

    if var is None:
        var_in = pd.DataFrame(index=gene_index)
    else:
        var_in = var.copy()
        if len(var_in) != len(gene_index):
            raise ValueError(
                f"var has {len(var_in)} rows, but got {len(gene_index)} gene symbols"
            )

    norm_index = gene_index.str.upper()
    counts = Counter(norm_index)
    first_seen: dict[str, str] = {}
    for original, normalized in zip(gene_index, norm_index):
        first_seen.setdefault(normalized, original)
    duplicate_symbols = sorted(
        first_seen[normalized] for normalized, n in counts.items() if n > 1
    )
    summary = {
        "n_input_genes": int(len(gene_index)),
        "n_output_genes": int(len(counts)),
        "n_collapsed_symbols": int(len(duplicate_symbols)),
        "n_removed_features": int(len(gene_index) - len(counts)),
        "collapsed_symbols": duplicate_symbols,
    }

    if not duplicate_symbols:
        var_out = var_in.copy()
        var_out.index = gene_index
        return values, var_out, summary

    unique_norm = pd.Index(pd.unique(norm_index))
    unique_symbols = pd.Index([first_seen[normalized] for normalized in unique_norm])
    group_index = unique_norm.get_indexer(norm_index)
    if log1p_space:
        if sp.issparse(values):
            linear_values = values.copy()
            linear_values.data = np.expm1(linear_values.data)
        else:
            linear_values = np.expm1(values)
        collapsed_linear = _sum_columns(linear_values, group_index, len(unique_norm))
        if sp.issparse(collapsed_linear):
            # ``log1p(0) == 0``, so the implicit zeros stay zero and the
            # transform can be applied to the stored values alone. Densifying
            # here would inflate the file roughly tenfold at ~10% density.
            collapsed_values = collapsed_linear.copy()
            collapsed_values.data = np.log1p(collapsed_values.data)
        else:
            collapsed_values = np.log1p(collapsed_linear)
    else:
        collapsed_values = _sum_columns(values, group_index, len(unique_norm))

    first_positions = ~norm_index.duplicated(keep="first")
    var_out = var_in.iloc[np.flatnonzero(first_positions)].copy()
    var_out.index = unique_symbols

    source_ids = pd.Index(var_in.index).astype(str)
    collapsed_feature_ids = []
    collapsed_n_features = []
    for normalized in unique_norm:
        positions = np.flatnonzero(norm_index == normalized)
        collapsed_n_features.append(int(len(positions)))
        collapsed_feature_ids.append("|".join(source_ids[positions]))
    var_out["collapsed_feature_ids"] = collapsed_feature_ids
    var_out["collapsed_n_features"] = collapsed_n_features

    return collapsed_values, var_out, summary


def has_case_duplicate_gene_symbols(gene_symbols) -> bool:
    """Cheap pre-check: True if any two symbols are equal case-insensitively.

    Use to gate a full :func:`collapse_duplicate_gene_symbols` pass (which
    needs the actual expression matrix in memory) behind a var-names-only
    check (cheap even against a backed/on-disk AnnData).
    """
    norm = pd.Index(gene_symbols).astype(str).str.upper()
    return bool(norm.duplicated().any())
