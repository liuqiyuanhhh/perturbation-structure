"""Row-wise Benjamini-Hochberg, used by the L1 strength (1_preprocessing) and the
discovery matrices (5_response_hierarchy)."""

import numpy as np


def bh_rejection_rows(p_values, alpha):
    """BH at level alpha in each row over its finite p-values, ties at the cutoff rejected.
    Returns the rejection matrix and each row's family size."""
    p_values = np.asarray(p_values, dtype=np.float64)
    finite = np.isfinite(p_values)
    rejected = np.zeros(p_values.shape, dtype=bool)
    for i, row in enumerate(p_values):
        ordered = np.sort(row[finite[i]])
        passing = np.flatnonzero(ordered <= alpha * np.arange(1, len(ordered) + 1) / len(ordered))
        if passing.size:
            rejected[i] = finite[i] & (row <= ordered[passing[-1]])
    return rejected, finite.sum(axis=1)
