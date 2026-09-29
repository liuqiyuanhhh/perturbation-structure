#!/usr/bin/env python3
"""Evaluate every method on one target: five metrics per method and CV fold.

    mean_weighted_mse              effect space, DE-weighted MSE
    mean_pds                       effect space, perturbation discrimination score
    mean_resid_cosine              residual space, all panel genes
    mean_resid_top20_pval_cosine   residual space, top 20 DE genes
    mean_resid_top100_pval_cosine  residual space, top 100 DE genes

Methods: GEARS, scGPT-ft and PRESAGE, from their prediction files, and Weighted,
the two linear models and the training mean, computed here (weighted_and_linear/weighted.py,
linear.py, train_mean.py).  The effect dict is cut to the target and its source
pool (targets.source_pool), so these see the screens PRESAGE's priors come from.  The test perturbations of a fold come from the
cv5 split of targets.split_build; those that no source screen measured are not
scored.  Genes are the target's full panel.  Every metric first sets the
perturbed gene's own entry to 0 in truth and prediction.  In residual space,
truth and each prediction have their own leading SVD factor removed.

Reads (paths.py): EFFECT_DICT (ground truth), DE_DIR (Wilcoxon DE z-scores: the
top-N genes and the weighted-MSE weights), GEARS_PRED_DIR (GEARS / scGPT-ft
absolute expression, minus a control profile: the pooled control mean of the
targets.py control_build in GEARS_DATA_DIR, or the one in MOMENTS_DIR), PRESAGE_PRED_DIR (PRESAGE effects) and the cv5 splits in
GEARS_DATA_DIR.  Writes EVAL_DIR/<target>/: summary_metrics.csv,
models_and_seeds.csv, predictions_by_seed.pkl, splits_by_seed.pkl.  Fig 1d and S2
(prediction_figures.ipynb) plot summary_metrics.csv.

    python evaluation/evaluate.py --target VCC
"""

import argparse
import os
import pickle
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py  # noqa: E402
import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import pairwise_distances  # noqa: E402
from sklearn.utils.extmath import randomized_svd  # noqa: E402

import paths  # noqa: E402
import targets  # noqa: E402
from weighted_and_linear.linear import K_PERT, linear_from_other_screens, linear_from_training, perturbation_embedding  # noqa: E402
from weighted_and_linear.train_mean import train_mean_prediction  # noqa: E402
from weighted_and_linear.weighted import screen_weights, weighted_prediction  # noqa: E402

FOLDS = [1, 2, 3, 4, 5]
CONTROL_BLOCK = 20000   # rows of a GEARS build's X read at a time
TOPN_LIST = [20, 100]
# A train-mean block is exactly rank 1, so its residual is float32 rounding
# residue; below this share of the Frobenius norm it is set to zero.
ZERO_TOL = 1e-5


# Inputs

def pooled_control(build):
    """Mean over the control cells of a GEARS build, pooled over batches."""
    path = os.path.join(paths.GEARS_DATA_DIR, build, build, "perturb_processed.h5ad")
    with h5py.File(path, "r") as fh:
        genes = fh["var"][fh["var"].attrs["_index"]].asstr()[:]
        control = fh["obs/control"]   # GEARS' categorical 0/1 column, 1 = control
        is_ctrl = control["categories"][:][control["codes"][:]] == 1
        indptr = fh["X/indptr"][:]
        n_row = len(indptr) - 1
        total = np.zeros(len(genes))
        for start in range(0, n_row, CONTROL_BLOCK):
            stop = min(start + CONTROL_BLOCK, n_row)
            lo, hi = int(indptr[start]), int(indptr[stop])
            idx = fh["X/indices"][lo:hi]
            dat = fh["X/data"][lo:hi].astype(np.float64)
            rows = np.repeat(np.arange(start, stop), np.diff(indptr[start:stop + 1]))
            sel = is_ctrl[rows]
            total += np.bincount(idx[sel], weights=dat[sel], minlength=len(genes))
    return pd.Series(total / int(is_ctrl.sum()), index=pd.Index(genes, name="gene"))


def control_mean(target, control_build):
    """Control profile subtracted from the GEARS / scGPT-ft predictions.

    GEARS and scGPT-ft predict absolute expression and do not model batch, so
    the targets whose training build has other controls than the moments file
    (control_build in targets.py: capped controls, per-fold builds) use the
    pooled control mean of that build; the others use uns/control_profile of
    the target's moments file.
    """
    if control_build:
        return pooled_control(control_build)
    path = os.path.join(paths.MOMENTS_DIR, paths.moments_file(target))
    # h5py rather than anndata: only this one array of the large file is needed
    with h5py.File(path, "r") as handle:
        genes = np.asarray(handle["var/_index"].asstr()[:])
        profile = handle["uns/control_profile"][:]
    return pd.Series(np.asarray(profile, dtype="float64"), index=pd.Index(genes, name="gene"))


def load_de_scores(target):
    """Batch-stratified Wilcoxon z-scores, perturbations x genes.

    A screen of several components (Feng-gw) stacks their disjoint perturbations
    on the shared genes.
    """
    frames = []
    for component in paths.components(target):
        path = os.path.join(paths.DE_DIR, paths.de_file(component))
        # h5py rather than anndata: only this one layer of the large file is needed
        with h5py.File(path, "r") as handle:
            obs, var = handle["obs"], handle["var"]
            frames.append(pd.DataFrame(handle["layers/z_score"][:],
                                       index=obs[obs.attrs["_index"]].asstr()[:],
                                       columns=var[var.attrs["_index"]].asstr()[:]))
    return pd.concat(frames, join="inner") if len(frames) > 1 else frames[0]


def pval_ranking(de_scores, genes):
    """Panel genes by decreasing |DE score|, one column per perturbation.

    Non-finite scores sort last; ties keep panel order.
    """
    panel = pd.Index(list(genes))
    block = de_scores.reindex(columns=panel)
    key = np.abs(block.to_numpy(dtype="float64"))
    key = np.where(np.isfinite(key), key, -1.0)
    order = np.argsort(-key, axis=1, kind="stable")
    return pd.DataFrame(panel.to_numpy()[order].T, columns=block.index)


def weight_matrix(de_scores):
    """Weighted-MSE gene weights, perturbations x genes.

    |score|, min-max scaled across genes, squared, normalised to sum 1; rows
    without spread get zero weights.  compute_mse renormalises over the genes
    it scores.
    """
    absolute = de_scores.abs().fillna(0.0)
    low = absolute.min(axis=1)
    span = absolute.max(axis=1) - low
    scaled = absolute.sub(low, axis=0).div(span.where(span > 0, 1.0), axis=0)
    scaled = scaled.where(span.gt(0), 0.0, axis=0)
    squared = scaled ** 2
    total = squared.sum(axis=1)
    return squared.div(total.where(total > 0, 1.0), axis=0)


def post_pred_path(build, model, fold):
    build = build.format(fold=fold)  # per-fold builds are named '...-third-f{fold}'
    return os.path.join(paths.GEARS_PRED_DIR, f"{build}-cv5_{fold}_{model}_post-pred.csv")


def presage_path(target, fold):
    return os.path.join(paths.PRESAGE_PRED_DIR, f"{target}_seed_{fold}_test_predictions.pkl")


def discover_models_and_seeds(gears_build, target):
    """(present, missing, seeds): the models with a prediction file for some
    fold, the others, and the folds where every present model has one."""
    candidates = ("gears", "scgpt_ft", "presage") if gears_build else ("presage",)

    def path_for(model, fold):
        if model == "presage":
            return presage_path(target, fold)
        return post_pred_path(gears_build, model, fold)

    present = [m for m in candidates if any(os.path.exists(path_for(m, f)) for f in FOLDS)]
    missing = [m for m in candidates if m not in present]
    seeds = [f for f in FOLDS if all(os.path.exists(path_for(m, f)) for m in present)]
    return present, missing, seeds


def load_post_pred(gears_build, model, fold, ctrl_mean, genes):
    """GEARS / scGPT-ft absolute prediction minus the control profile, on the panel."""
    frame = pd.read_csv(post_pred_path(gears_build, model, fold))
    frame["condition"] = frame["condition"].str.replace("+ctrl", "", regex=False)
    frame = frame.set_index("condition")
    common = [g for g in genes if g in frame.columns]
    return frame.loc[:, common].sub(ctrl_mean.loc[common], axis=1).fillna(0.0)


def load_model_predictions(target, spec, present, seeds, panel_genes):
    """{method: {fold: DataFrame}} of the GEARS, scGPT-ft and PRESAGE effects."""
    model_preds = {}
    if {"gears", "scgpt_ft"} & set(present):
        # absolute expression: subtract the control of the data the models trained on
        ctrl_mean = control_mean(target, spec.control_build)
        for model, label in (("gears", "gears"), ("scgpt_ft", "scGPT-ft")):
            if model in present:
                model_preds[label] = {
                    s: load_post_pred(spec.gears_build, model, s, ctrl_mean, panel_genes)
                    for s in seeds}
    if "presage" in present:
        model_preds["presage"] = {s: joblib.load(presage_path(target, s)) for s in seeds}
    return model_preds


def cv5_test(build, fold):
    """Test perturbations of one fold, as the names of the effect frames ('GPN3+ctrl' -> 'GPN3')."""
    path = os.path.join(paths.GEARS_DATA_DIR, build, build, "splits", f"{build}_cv5_{fold}.pkl")
    with open(path, "rb") as handle:
        split = pickle.load(handle)
    return {re.sub(r"\+ctrl$", "", str(c)) for c in split["test"]}


# Metrics: {perturbation: value}

def _zero_target_genes(*frames):
    """Set entry (p, p) to 0 in every frame, for each perturbation p on the panel."""
    genes = frames[0].columns
    for pert in frames[0].index:
        if pert in genes:
            for frame in frames:
                frame.at[pert, pert] = 0.0


def compute_pds(true_effects, pred_effects):
    """1 - the normalised rank of the true perturbation among all truths, by
    cosine distance to the prediction; the target gene is also left out of the
    distances."""
    pred_effects = pred_effects.reindex(index=true_effects.index, columns=true_effects.columns)
    true_effects = true_effects.fillna(0.0)
    pred_effects = pred_effects.fillna(0.0)
    _zero_target_genes(true_effects, pred_effects)

    perts = true_effects.index.to_numpy()
    genes = true_effects.columns.to_numpy()
    R_full = true_effects.to_numpy()
    P_full = pred_effects.to_numpy()

    scores = {}
    for i, pert in enumerate(perts):
        keep = genes != pert
        distances = pairwise_distances(R_full[:, keep], P_full[i, keep].reshape(1, -1),
                                       metric="cosine").flatten()
        rank = int(np.flatnonzero(np.argsort(distances) == i)[0])
        scores[pert] = 1.0 - (rank / (len(perts) - 1) if len(perts) > 1 else 0)
    return scores


def compute_mse(true_effects, pred_effects, weight_df):
    """Weighted MSE sum_g w_g (t_g - p_g)^2, with each row of weights renormalised to sum 1."""
    pred_effects = pred_effects.reindex(index=true_effects.index, columns=true_effects.columns)
    true_effects = true_effects.fillna(0.0)
    pred_effects = pred_effects.fillna(0.0)
    weight_df = weight_df.reindex(index=true_effects.index,
                                  columns=true_effects.columns).fillna(0.0)
    _zero_target_genes(true_effects, pred_effects, weight_df)

    mse = {}
    for pert in true_effects.index:
        w = weight_df.loc[pert].to_numpy()
        t = true_effects.loc[pert].to_numpy()
        p = pred_effects.loc[pert].to_numpy()
        if w.sum() > 0:
            w = w / w.sum()
        mse[pert] = float(np.sum(w * (t - p) ** 2))
    return mse


def compute_cosine_similarity(df_true, df_pred, topN=None, pval_df=None):
    """Cosine per perturbation, on all shared genes or, with `topN`, on its top-N DE genes.

    `pval_df` is the DE ranking (rank position x perturbation, most DE first);
    the perturbed gene is skipped when taking its first `topN` genes.
    """
    perts = df_true.index.intersection(df_pred.index)
    genes = df_true.columns.intersection(df_pred.columns)
    true_aligned = df_true.loc[perts, genes].astype(float).copy()
    pred_aligned = df_pred.loc[perts, genes].astype(float).copy()
    _zero_target_genes(true_aligned, pred_aligned)

    cosines = {}
    for pert in perts:
        row_true = true_aligned.loc[pert]
        row_pred = pred_aligned.loc[pert]
        if topN is not None:
            top_genes = [g for g in pval_df[pert][: topN + 1] if g != pert][:topN]
            row_true = row_true.loc[top_genes]
            row_pred = row_pred.loc[top_genes]
        distance = pairwise_distances(row_true.values.reshape(1, -1),
                                      row_pred.values.reshape(1, -1), metric="cosine")
        cosines[pert] = float(1.0 - distance.flatten()[0])
    return cosines


def mean_score(scores):
    return float(np.nanmean(list(scores.values())))


def remove_first_factor(df, random_state=0, zero_tol=None):
    """The perturbation x gene matrix minus its leading SVD factor.

    A matrix that is exactly rank 1, such as the train-mean prediction, leaves
    only float32 rounding residue, and that residue is nearly the same vector
    in every row, so a cosine would read it as a systematic signal.  When the
    residual's share of the Frobenius norm is below `zero_tol`, it is set to
    exactly zero.
    """
    V = np.asarray(df.values, dtype=np.float32)
    U, S, Vt = randomized_svd(V, n_components=1, random_state=random_state)
    R = V - (U * S) @ Vt

    total = float(np.linalg.norm(V))
    if zero_tol is not None and total > 0 and float(np.linalg.norm(R)) / total < zero_tol:
        R = np.zeros_like(R)
    return pd.DataFrame(R, index=df.index, columns=df.columns)


# Evaluation

def build_predictions(target, truth_key, effects, seeds, model_preds, panel_genes):
    """{fold: {method: DataFrame[test_pert x gene]}} and {fold: {train, test}}."""
    frame_index = list(effects[truth_key].index)
    pert_emb = perturbation_embedding(effects, truth_key)

    predictions_by_seed, splits_by_seed = {}, {}
    for seed in seeds:
        split_test = cv5_test(targets.split_build(target), seed)
        test_pert = [p for p in frame_index if p in split_test]
        train_pert = [p for p in frame_index if p not in split_test]
        splits_by_seed[seed] = {"train": train_pert, "test": test_pert}

        # a test perturbation missing from a model's file is predicted as 0
        preds = {method: per_seed[seed].reindex(index=test_pert, columns=panel_genes).fillna(0.0)
                 for method, per_seed in model_preds.items()}
        preds.update(fold_predictions(effects, truth_key, train_pert, test_pert, pert_emb))
        predictions_by_seed[seed] = preds
    return predictions_by_seed, splits_by_seed


def fold_predictions(effects, target, train_pert, test_pert, pert_emb):
    """Weighted, the two linear models and the training mean for one fold.

    {method: DataFrame[test_pert x gene]} of predicted effects, NaN set to 0.
    """
    truth = effects[target]
    weights = screen_weights(truth.loc[train_pert], effects, (target,))
    preds = {
        "weighted": weighted_prediction(effects, truth, test_pert, weights),
        f"paper_linear_embedding_from_training_{K_PERT}": linear_from_training(
            truth, train_pert, test_pert),
        f"linear_embedding_from_otherds_{K_PERT}": linear_from_other_screens(
            effects, target, train_pert, test_pert, pert_emb),
        "train_mean": train_mean_prediction(truth, train_pert, test_pert),
    }
    return {method: pred.fillna(0.0) for method, pred in preds.items()}


def evaluate(truth, predictions_by_seed, splits_by_seed, weight_df, pval_df, panel_genes, covered):
    """The five metrics per fold and method, on the test perturbations some source screen measured."""
    rows = []
    for seed, predictions in predictions_by_seed.items():
        eval_pert = [p for p in splits_by_seed[seed]["test"] if p in covered]
        truth_eff = truth.loc[eval_pert, panel_genes]
        truth_res = remove_first_factor(truth_eff, random_state=0, zero_tol=ZERO_TOL)
        weight_test = weight_df.reindex(index=truth_eff.index, columns=truth_eff.columns)

        for method, pred in predictions.items():
            pred_eff = pred.reindex(index=eval_pert, columns=panel_genes).fillna(0.0)
            pred_res = remove_first_factor(pred_eff, random_state=0, zero_tol=ZERO_TOL)

            row = {"seed": seed, "method": method,
                   "mean_weighted_mse": mean_score(compute_mse(truth_eff, pred_eff, weight_test)),
                   "mean_pds": mean_score(compute_pds(truth_eff, pred_eff)),
                   "mean_resid_cosine": mean_score(compute_cosine_similarity(truth_res, pred_res))}
            for topN in TOPN_LIST:
                row[f"mean_resid_top{topN}_pval_cosine"] = mean_score(
                    compute_cosine_similarity(truth_res, pred_res, topN=topN, pval_df=pval_df))
            rows.append(row)

    return pd.DataFrame(rows).set_index(["seed", "method"]).sort_index()


def run(target):
    spec = targets.get(target)
    truth_key = targets.truth_key(target)
    output_dir = os.path.join(paths.EVAL_DIR, target)
    os.makedirs(output_dir, exist_ok=True)

    effects = joblib.load(paths.EFFECT_DICT)
    pool = set(targets.source_pool(effects, target))
    # keep the dict's key order: the linear models' PCA depends on row order
    effects = {k: v for k, v in effects.items() if k == truth_key or k in pool}
    truth = effects[truth_key]
    panel_genes = truth.columns
    covered = set().union(*(effects[k].index for k in effects if k != truth_key))

    present, absent, seeds = discover_models_and_seeds(spec.gears_build, target)
    print(f"[{target}] {len(pool)} source screens, models {', '.join(present)}, folds {seeds}",
          flush=True)
    model_preds = load_model_predictions(target, spec, present, seeds, panel_genes)
    de_scores = load_de_scores(target)
    weight_df = weight_matrix(de_scores)
    pval_df = pval_ranking(de_scores, panel_genes)

    predictions_by_seed, splits_by_seed = build_predictions(
        target, truth_key, effects, seeds, model_preds, panel_genes)
    pd.DataFrame([{"models_present": ";".join(present),
                   "models_absent": ";".join(absent) or "-",
                   "seeds": ";".join(str(x) for x in seeds),
                   "n_seeds": len(seeds)}]).to_csv(
        os.path.join(output_dir, "models_and_seeds.csv"), index=False)
    joblib.dump(predictions_by_seed, os.path.join(output_dir, "predictions_by_seed.pkl"))
    joblib.dump(splits_by_seed, os.path.join(output_dir, "splits_by_seed.pkl"))

    summary = evaluate(truth, predictions_by_seed, splits_by_seed,
                       weight_df, pval_df, panel_genes, covered)
    summary.to_csv(os.path.join(output_dir, "summary_metrics.csv"))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", required=True, choices=list(targets.TARGETS) + ["all"])
    args = parser.parse_args()
    for target in (list(targets.TARGETS) if args.target == "all" else [args.target]):
        run(target)


if __name__ == "__main__":
    main()
