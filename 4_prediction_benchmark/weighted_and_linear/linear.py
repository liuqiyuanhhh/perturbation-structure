"""The two linear models: Linear (P from training) and Linear (P from other ds).

Both fit  Y ~ G K P^T + b  on the target's training perturbations, with Y the
gene x perturbation effect matrix, G a gene embedding (PCA of the training
effects, genes as samples), P a perturbation embedding, K a k x k matrix and b
a per-gene intercept.  In Linear (P from training) a perturbation's embedding is
the G row of its target gene; in Linear (P from other ds) it is the PCA of its
effect averaged over the other screens, or its G row when no other screen
measured it.  Only perturbations named after a panel gene are predicted.
Used by evaluate.py.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

K_PERT = 10          # PCA dimension of the gene and perturbation embeddings
RIDGE_ALPHA = 0.1    # ridge penalty of the bilinear model


def _solve_bilinear_ridge(Y, A, B, alpha):
    """K = (A'A + aI)^-1 A' Yc B' (BB' + aI)^-1, with Yc = Y minus its row means.

    Y is genes x training perturbations, A genes x k, B k x training perturbations.
    Returns K and the row means (the intercept b).
    """
    center = Y.mean(axis=1)
    Yc = Y - center[:, None]
    I_k = np.eye(A.shape[1])
    left = np.linalg.solve(A.T @ A + alpha * I_k, A.T)
    right = np.linalg.solve(B @ B.T + alpha * I_k, B)
    return left @ Yc @ right.T, center


def _training_matrix(effect_df, train_targets):
    """Y: genes x training perturbations, NaN set to 0."""
    return np.nan_to_num(effect_df.loc[train_targets, :].to_numpy().T, nan=0.0)


def gene_embedding(Y, genes, k, random_state=0):
    """G: PCA of Y (genes x training perturbations), genes as samples, at most k components."""
    k = min(k, *Y.shape)
    G = PCA(n_components=k, random_state=random_state).fit_transform(Y)
    return pd.DataFrame(G, index=genes, columns=[f"PC{i + 1}" for i in range(k)])


def linear_from_training(effect_df, train_perts, test_perts, k=K_PERT, alpha=RIDGE_ALPHA):
    """Linear (P from training): [test_perts x genes], NaN where not predictable."""
    genes = effect_df.columns
    train_targets = [p for p in train_perts if p in genes]
    test_targets = [p for p in test_perts if p in genes]

    Y = _training_matrix(effect_df, train_targets)
    G_df = gene_embedding(Y, genes, k)
    G = G_df.to_numpy()
    K, b = _solve_bilinear_ridge(Y, G, G_df.loc[train_targets].to_numpy().T, alpha)
    Y_hat = (G @ K @ G_df.loc[test_targets].to_numpy().T) + b[:, None]

    pred = pd.DataFrame(np.nan, index=test_perts, columns=genes, dtype=float)
    pred.loc[test_targets, :] = Y_hat.T
    return pred


def perturbation_embedding(effects, target, n_components=K_PERT, random_state=0):
    """[perturbation x n_components] PCA of the effects averaged over the other screens.

    The perturbation embedding of Linear (P from other ds), the same for every
    fold.  The screens are aligned on the union of their genes; a
    perturbation's mean skips the screens that did not measure a gene, and
    genes no screen measured are set to 0 before the PCA.  Columns beyond the
    PCA rank are 0.
    """
    frames = [df for ds, df in effects.items() if ds != target]
    genes = frames[0].columns
    for df in frames[1:]:
        genes = genes.union(df.columns)

    stacked = []
    for df in frames:
        aligned = df.reindex(columns=genes)
        aligned["__pert__"] = aligned.index
        stacked.append(aligned.reset_index(drop=True))
    mean_df = pd.concat(stacked, axis=0, ignore_index=True).groupby(
        "__pert__", sort=False)[list(genes)].mean(numeric_only=True)

    X = np.nan_to_num(mean_df.to_numpy(dtype=float, copy=False), nan=0.0)
    n_fit = min(n_components, *X.shape)
    Z = PCA(n_components=n_fit, random_state=random_state).fit_transform(X)
    Z = np.pad(Z, ((0, 0), (0, n_components - n_fit)), mode="constant", constant_values=0.0)

    emb = pd.DataFrame(Z, index=mean_df.index, columns=[f"PC{i + 1}" for i in range(n_components)])
    emb.index.name = "perturbation"
    return emb


def linear_from_other_screens(effects, target, train_perts, test_perts, pert_emb,
                              k_gene=K_PERT, alpha=RIDGE_ALPHA, random_state=0):
    """Linear (P from other ds): [test_perts x genes], NaN where not predictable.

    `pert_emb` comes from perturbation_embedding; its rows are cut or
    zero-padded to the width of the gene embedding.
    """
    target_df = effects[target]
    genes = target_df.columns
    train_targets = [p for p in train_perts if p in genes]
    test_targets = [p for p in test_perts if p in genes]

    Y = _training_matrix(target_df, train_targets)
    gene_emb = gene_embedding(Y, genes, k_gene, random_state)
    A = gene_emb.to_numpy(dtype=float, copy=False)
    k = A.shape[1]

    def pert_vector(name):
        source = pert_emb if name in pert_emb.index else gene_emb
        v = source.loc[name].to_numpy(dtype=float, copy=False)[:k]
        return np.pad(v, (0, k - v.size))

    B_train = np.vstack([pert_vector(p) for p in train_targets]).T
    B_test = np.vstack([pert_vector(p) for p in test_targets]).T
    K, b = _solve_bilinear_ridge(Y, A, B_train, alpha)
    Y_hat = (A @ K @ B_test) + b[:, None]

    pred = pd.DataFrame(np.nan, index=test_perts, columns=genes, dtype=float)
    pred.loc[test_targets, :] = Y_hat.T
    pred.index.name = "perturbation"
    return pred
