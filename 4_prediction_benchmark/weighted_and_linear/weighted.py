"""Weighted: the perturbation's effect in the other screens, averaged per gene.

A screen's weight is its cosine similarity to the target's training
perturbations (target genes zeroed on both sides), normalised to sum to one over
the screens with a positive similarity; a gene a screen did not measure is left
out of that gene's mean.  `effects` is {screen: DataFrame[perturbation x gene]},
the target's ground truth and the screens of its source pool
(targets.source_pool).  Used by evaluate.py.
"""

import numpy as np
import pandas as pd


def screen_cosine(df_i, df_j, min_overlap_genes=2):
    """Mean over shared perturbations of the cosine between two effect frames.

    Each cosine uses the genes both frames measured and that are non-NaN in
    both; perturbations with fewer than `min_overlap_genes` such genes are skipped.
    """
    common_perts = df_i.index.intersection(df_j.index)
    common_genes = df_i.columns.intersection(df_j.columns)
    if len(common_perts) == 0 or len(common_genes) == 0:
        return float("nan")

    cosines = []
    for pert in common_perts:
        x = df_i.loc[pert, common_genes].values.astype(float)
        y = df_j.loc[pert, common_genes].values.astype(float)
        mask = ~np.isnan(x) & ~np.isnan(y)
        if mask.sum() < min_overlap_genes:
            continue
        x, y = x[mask], y[mask]
        denom = np.linalg.norm(x) * np.linalg.norm(y)
        cosines.append(0.0 if denom == 0 else float(np.dot(x, y) / denom))
    return float(np.mean(cosines)) if cosines else float("nan")


def zero_target_genes(df):
    """Copy of `df` with each perturbation's own target gene set to 0."""
    df = df.copy()
    for pert in df.index:
        if pert in df.columns:
            df.at[pert, pert] = 0.0
    return df


def screen_weights(train_df, effects, exclude):
    """{screen: weight}: cosine similarity to the target's training block.

    Target genes are zeroed on both sides, so a screen is not rewarded for the
    knockdown itself.  Screens with a non-positive or undefined similarity get
    no weight.
    """
    train_zeroed = zero_target_genes(train_df)
    raw = {}
    for ds, df in effects.items():
        if ds in exclude:
            continue
        sim = screen_cosine(train_zeroed, zero_target_genes(df))
        if np.isfinite(sim) and sim > 0:
            raw[ds] = sim
    total = sum(raw.values())
    return {ds: v / total for ds, v in raw.items()}


def weighted_prediction(effects, truth, test_pert, weights):
    """DataFrame[test_pert x genes of truth] of weighted source means.

    Only screens with a weight contribute; a perturbation no source measured is NaN.
    """
    genes = truth.columns.to_numpy()
    rows = {}
    for pert in test_pert:
        w_list, m_list = [], []
        for ds, df in effects.items():
            if ds not in weights or pert not in df.index:
                continue
            vec = df.loc[pert].reindex(genes)
            if np.all(np.isnan(vec.values)):
                continue
            w_list.append(weights[ds])
            m_list.append(vec.values.astype(float, copy=False))

        if not m_list:
            rows[pert] = np.full(len(genes), np.nan, dtype=float)
            continue

        W = np.asarray(w_list, dtype=float)[:, None]
        A = np.vstack(m_list)
        not_nan = ~np.isnan(A)
        num = np.nansum(W * np.where(not_nan, A, 0.0), axis=0)
        den = np.where(not_nan, W, 0.0).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            rows[pert] = num / den

    pred = pd.DataFrame.from_dict(rows, orient="index", columns=genes)
    pred.index.name = "perturbation"
    return pred
