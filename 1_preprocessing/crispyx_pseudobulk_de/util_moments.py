"""Unified per-perturbation moments: avg_log mean, std, and SE in ONE pass.

Step 04 (`04_moments.py`) computes three per-perturbation quantities that used to
require four separate streaming scans (crispyx `normalized_effects` for the mean,
`util_batch_corrected_std` for the std, two `batch_process` passes for the SE):

* **avg_log** — batch-corrected absolute log-mean expression
  (`perturbation_profile`) and its effect vs control (`X`).
* **std** — within-batch population std of single cells (`pert_std`,
  `control_std_matched`): "how spread out are individual cells".
* **se** — standard error of the batch-corrected mean/effect estimate
  (`pert_mean_se`, `ctrl_mean_se`, `effect_se`): "how uncertain is the mean",
  which shrinks with more cells (unlike std).

All three are linear functions of the same per-`(group, batch)` sufficient
statistics `Sum(x)`, `Sum(x^2)`, `n` (accumuland `x` = the already
library-size-normalized + log1p sc value). crispyx **0.1.4**'s multi-channel
`BatchReducer` (`channels=(...)`, `compare` returns a dict of `BatchStatistic`)
emits all of them from a **single** streaming pass; each channel is combined
across batches independently as `sum(w*v)/sum(w)` and written to `layers[name]`
(weights to `layers[f"{name}_weight_sum"]`).

Two processes, one output schema (`compute_moments` dispatches):

* **batch-correction process** (`_moments_batch`, datasets with an audited
  `batch_col`) — the multi-channel `cx.batch_process` over the QC-filtered,
  normalized `data/sc/sc_<name>.h5ad`, harmonic-count weighted
  `w_b = n_p*n_c/(n_p+n_c)` per shared batch.
* **non-batch process** (`_moments_pooled`, no `batch_col`) — the single-stratum
  closed form computed directly from the same sc file (`std/sqrt(n)` etc.).

**Split per-channel gate in the batch process (mean `n>=1`, std/se pooled
`n>=2`).** A singleton `(pert, batch)` stratum (`n=1` on the pert or control
side) has no estimable within-batch variance, so it cannot feed the std at
all -- but dropping the whole stratum from the *mean* would bias it, because
batch membership is a structured confounder correlated with phenotype (a
sickly/toxic perturbation yields disproportionately many singleton batches).
So the mean/effect gate is `n_p>=1 and n_c>=1` per side (every batch with any
cells on both sides participates), while std is one **shared pooled
within-batch variance `sigma^2`** estimated only from the `n>=2` batches, and
SE reuses that same `sigma^2` for every participating batch (`Var(x_bar_b) =
sigma^2/n_b`, so a singleton batch is simply `n_b=1` in the SE sum) -- see
`_make_reducer`. A perturbation with zero participating batches (n_p or n_c
always 0) gets a NaN mean; one whose participating batches are *all*
singletons gets a finite mean but NaN std/se (no batch could estimate
`sigma^2`). Both are floored/dropped downstream.

Both processes read the **QC-filtered, log1p + library-size-normalized,
gene-deduped** sc file (NOT raw origin) so the moments describe the same cell
population the single-cell methods and DE see. Duplicate gene symbols are
collapsed on the small `pert x gene` output (means in log1p space; std/se by
summing variances).

**Robust (between-batch, matched-stratum) SE** — `pert_mean_se_robust`,
`ctrl_mean_se_robust`, `effect_se_robust` — is a second, independent SE
estimate alongside `pert_mean_se`/`ctrl_mean_se`/`effect_se`: the empirical
("sandwich") variance of the weighted mean from the *spread of the per-batch
point estimates themselves* across batches, standard for matched/stratified
designs in causal inference (1 treated stratum vs. its matched control(s),
many strata). It makes no shared-within-batch-variance assumption (unlike the
model-based SE above), at the cost of needing `>=2` participating batches
(NaN otherwise, and always NaN in the pooled/non-batch process, which has
only one stratum). `effect_se_robust` is computed from the per-batch matched
difference `d_b = mean_p,b - mean_c,b` directly, not derived from the
separate pert/ctrl robust SEs, so it captures the within-batch noise
cancellation that motivates a matched design in the first place.

Output: one combined `data/pseudobulk/effect_<name>_moments.h5ad` per dataset (see
`_assemble`).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import anndata as ad
import crispyx as cx
import numpy as np
import pandas as pd
import scipy.sparse as sp

from util_gene_symbols import collapse_duplicate_gene_symbols

log = logging.getLogger(__name__)

# One value per gene per channel, all from the same {sum, sqsum, n} state.
_CHANNELS = (
    "pert_mean", "ctrl_mean", "pert_var", "ctrl_var", "pert_sw2n", "ctrl_sw2n",
    # Robust (between-batch, matched-stratum) SE channels -- see _make_reducer.
    "pert_w2x", "pert_w2x2", "ctrl_w2x", "ctrl_w2x2", "eff_w2d", "eff_w2d2",
    "batch_count",
)


# ---------------------------------------------------------------------------
# Multi-channel reducer (batch-correction process)
# ---------------------------------------------------------------------------
def _initialize(width: int) -> dict[str, Any]:
    return {
        "sum": np.zeros(width, dtype=np.float64),
        "sqsum": np.zeros(width, dtype=np.float64),
        "n": 0,
    }


def _update(state: dict[str, Any], block: np.ndarray) -> None:
    b = np.asarray(block, dtype=np.float64)
    state["sum"] += b.sum(axis=0)
    state["sqsum"] += (b * b).sum(axis=0)
    state["n"] += b.shape[0]


def _finalize_unused(state: dict[str, Any]):
    # mode="comparison" never calls finalize; present only to satisfy the
    # BatchReducer dataclass (compare is what does the work).
    raise NotImplementedError("finalize is unused in comparison mode")


def _mean_var(state: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, int]:
    n = state["n"]
    mean = state["sum"] / n
    var = np.maximum(state["sqsum"] / n - mean * mean, 0.0)  # population var, clip fp noise
    return mean, var, n


def _make_reducer() -> cx.BatchReducer:
    """A comparison-mode reducer emitting the moment channels in one pass.

    Split per-channel gate: mean/ctrl_mean participate iff `n_p>=1 and
    n_c>=1` (a batch with any cells on both sides); pert_var/ctrl_var
    participate only when their own side has `n>=2` (a variance needs >=2
    observations -- structural, not a threshold). The `*_sw2n` channels
    (`w/n` per side) share the mean's `n>=1` gate; combined with the mean's
    weight-sum downstream they reconstruct `Sum_P w^2/n`, the SE's sample-size
    term over the *same* participating set as the mean (see
    `_moments_batch_pass`).

    The `*_w2x`/`*_w2x2`/`eff_w2d`/`eff_w2d2`/`batch_count` channels feed the
    robust (matched-stratum, between-batch) SE -- the empirical/"sandwich"
    variance of the weighted mean across batches, standard in stratified/
    matched causal inference, complementing the pooled-sigma^2 SE above by
    not assuming a shared within-batch variance. `eff_w2d`/`eff_w2d2` are
    computed on the per-batch *matched difference* `d_b = mean_p,b -
    mean_c,b` directly (not derived from the separate pert/ctrl robust SEs),
    so it captures the within-batch noise cancellation a matched design is
    for. See `_moments_batch_pass` for the reconstruction.
    """
    def compare(group_state: dict[str, Any], reference_state: dict[str, Any]):
        width = group_state["sum"].shape[0]
        n_p, n_c = group_state["n"], reference_state["n"]
        if n_p < 1 or n_c < 1:
            zero = cx.BatchStatistic(np.zeros(width, dtype=np.float64), weight=0.0)
            return {name: zero for name in _CHANNELS}
        mean_p, var_p, _ = _mean_var(group_state)   # var of a 1-cell state is 0; only used if n>=2 (else weight 0)
        mean_c, var_c, _ = _mean_var(reference_state)
        w = (n_p * n_c) / (n_p + n_c)  # harmonic count weight (matches the mean)
        w2 = w * w
        d = mean_p - mean_c            # matched per-batch effect estimate
        return {
            # linear-weight channels: layers[name] = sum(w*v)/sum(w) directly
            "pert_mean": cx.BatchStatistic(mean_p, weight=w),
            "ctrl_mean": cx.BatchStatistic(mean_c, weight=w),
            # var channels gate independently on their own side's n>=2, so
            # layers["pert_var"] = Sum_{n_p>=2} w*var_p / Sum_{n_p>=2} w = sigma^2_p.
            "pert_var": cx.BatchStatistic(var_p, weight=(w if n_p >= 2 else 0.0)),
            "ctrl_var": cx.BatchStatistic(var_c, weight=(w if n_c >= 2 else 0.0)),
            # squared-weight channels: report w/n so
            # layers[name] = sum(w*(w/n))/sum(w) = Sum_P w^2/n / Sum_P w;
            # multiply by the mean's weight_sum downstream to recover Sum_P w^2/n.
            "pert_sw2n": cx.BatchStatistic(np.full(width, w / n_p, dtype=np.float64), weight=w),
            "ctrl_sw2n": cx.BatchStatistic(np.full(width, w / n_c, dtype=np.float64), weight=w),
            # w^2-weighted sum/sum-of-squares of each per-batch point estimate
            # (pert mean, ctrl mean, matched difference); reconstructed
            # downstream into Sum_P w^2*(x_b - xbar)^2 via the expand-the-
            # square identity, same trick as _mean_var's population variance.
            "pert_w2x": cx.BatchStatistic(mean_p, weight=w2),
            "pert_w2x2": cx.BatchStatistic(mean_p * mean_p, weight=w2),
            "ctrl_w2x": cx.BatchStatistic(mean_c, weight=w2),
            "ctrl_w2x2": cx.BatchStatistic(mean_c * mean_c, weight=w2),
            "eff_w2d": cx.BatchStatistic(d, weight=w2),
            "eff_w2d2": cx.BatchStatistic(d * d, weight=w2),
            # constant-weight channel: layers["batch_count_weight_sum"] =
            # Sum_P 1 = B, the number of participating batches.
            "batch_count": cx.BatchStatistic(np.ones(width, dtype=np.float64), weight=1.0),
        }

    return cx.BatchReducer(
        initialize=_initialize, update=_update, finalize=_finalize_unused,
        compare=compare, channels=_CHANNELS,
    )


_MOMENT_ARRAY_KEYS = (
    "pert_mean", "ctrl_matched", "pert_var", "ctrl_var", "pert_mean_var", "ctrl_mean_var",
    "pert_mean_var_robust", "ctrl_mean_var_robust", "effect_var_robust",
)


def _moments_batch_pass(
    sc_path: Path,
    name: str,
    *,
    perturbation_column: str,
    control_label: str,
    batch_column: str,
    perturbations: list[str] | None,
    chunk_size: int | None,
    memory_limit_gb: float | None,
    resume: bool,
    checkpoint_interval: int | None,
    raw_path: Path,
    force: bool,
    verbose: int,
) -> dict[str, Any]:
    """One multi-channel batch_process pass for a single batch_column."""
    common: dict[str, Any] = dict(
        perturbation_column=perturbation_column,
        control_label=control_label,
        batch_column=batch_column,
        perturbations=perturbations,
        mode="comparison",
        statistic_name="moments",
        output_path=raw_path,
        verbose=bool(verbose),
        force=force,
        resume=resume,
        format_mismatch_policy="convert",
    )
    if chunk_size is not None:
        common["chunk_size"] = chunk_size
    if memory_limit_gb is not None:
        common["memory_limit_gb"] = memory_limit_gb
    if checkpoint_interval is not None:
        common["checkpoint_interval"] = checkpoint_interval

    raw = cx.batch_process(sc_path, _make_reducer(), **common)

    # crispyx orders groups by first appearance; sort alphabetically to match
    # the pooled path and every other per-dataset artifact.
    labels = np.asarray(raw.obs_names, dtype=str)
    order = np.argsort(labels)
    candidates = labels[order].tolist()

    def _lyr(key: str) -> np.ndarray:
        return np.asarray(raw.layers[key], dtype=np.float64)[order]

    pert_mean = _lyr("pert_mean")          # Sum_P w*mean / Sum_P w, NaN if no participating batch (n_p,n_c>=1)
    ctrl_matched = _lyr("ctrl_mean")
    sigma2_p = _lyr("pert_var")            # pooled within-batch var over n_p>=2 batches; NaN if none
    sigma2_c = _lyr("ctrl_var")
    # pert_mean/ctrl_mean/pert_sw2n/ctrl_sw2n all share the same weight w and
    # the same n_p>=1,n_c>=1 gate, so any one weight_sum recovers Sum_P w.
    W_all = _lyr("pert_mean_weight_sum")
    with np.errstate(invalid="ignore", divide="ignore"):
        S_p = _lyr("pert_sw2n") * W_all    # Sum_P w^2/n_p
        S_c = _lyr("ctrl_sw2n") * W_all    # Sum_P w^2/n_c
        pert_mean_var = sigma2_p * S_p / (W_all ** 2)   # se^2_p = sigma^2_p * Sum_P w^2/n_p / (Sum_P w)^2
        ctrl_mean_var = sigma2_c * S_c / (W_all ** 2)

        # Robust (between-batch) SE: empirical/"sandwich" variance of the
        # weighted mean across the B participating batches, no shared-sigma^2
        # assumption. B/(B-1) is the finite-sample (Bessel-type) correction.
        # NaN for B<2 (can't estimate a between-batch spread from <2 batches)
        # -- explicitly masked rather than relying on 0/0: at B=1 the
        # algebraic sum-of-squares is exactly 0, but crispyx's streaming
        # accumulation leaves ~1e-14 floating-point noise there, which
        # B/(B-1)=inf amplifies into spurious +-inf instead of NaN.
        B = _lyr("batch_count_weight_sum")
        has_2plus = B >= 2
        dof = B / np.maximum(B - 1, 1.0)   # safe denominator; masked out below when B<2
        effect = pert_mean - ctrl_matched
        W2_all = _lyr("pert_w2x_weight_sum")   # Sum_P w^2 (same for ctrl_w2x/eff_w2d -- one shared per-batch weight)

        def _robust_var(w2x_key: str, w2x2_key: str, xbar: np.ndarray) -> np.ndarray:
            mean_x = _lyr(w2x_key)             # Sum_P w^2*x / Sum_P w^2
            mean_x2 = _lyr(w2x2_key)           # Sum_P w^2*x^2 / Sum_P w^2
            # Sum_P w^2*(x-xbar)^2 = Sum_P w^2*x^2 - 2*xbar*Sum_P w^2*x + xbar^2*Sum_P w^2
            ss = np.maximum(W2_all * (mean_x2 - 2.0 * xbar * mean_x + xbar * xbar), 0.0)
            var = dof * ss / (W_all ** 2)
            return np.where(has_2plus, var, np.nan)

        pert_mean_var_robust = _robust_var("pert_w2x", "pert_w2x2", pert_mean)
        ctrl_mean_var_robust = _robust_var("ctrl_w2x", "ctrl_w2x2", ctrl_matched)
        effect_var_robust = _robust_var("eff_w2d", "eff_w2d2", effect)

    return {
        "candidates": candidates,
        "gene_symbols": np.asarray(raw.var_names, dtype=str),
        "var_meta": raw.var.copy(),
        "pert_mean": pert_mean,
        "ctrl_matched": ctrl_matched,
        "pert_var": sigma2_p,
        "ctrl_var": sigma2_c,
        "pert_mean_var": pert_mean_var,
        "ctrl_mean_var": ctrl_mean_var,
        "pert_mean_var_robust": pert_mean_var_robust,
        "ctrl_mean_var_robust": ctrl_mean_var_robust,
        "effect_var_robust": effect_var_robust,
    }


def _moments_batch(
    sc_path: Path,
    name: str,
    *,
    perturbation_column: str,
    control_label: str,
    batch_column: str,
    fallback_batch_column: str | None,
    perturbations: list[str] | None,
    chunk_size: int | None,
    memory_limit_gb: float | None,
    resume: bool,
    checkpoint_interval: int | None,
    raw_cache_dir: Path,
    force: bool,
    verbose: int,
) -> dict[str, Any]:
    """Batch-correction process: one multi-channel batch_process pass.

    If fallback_batch_column is given, a perturbation with NO stratum with
    any cells on both sides under batch_column (e.g. the fine joint
    batch x donor key) is recomputed under fallback_batch_column (the plain
    batch key) instead of being left NaN/floored -- trading donor correction
    for coverage on exactly the perturbations too thin to support it. Every
    perturbation is tagged in "stratification" with which key actually
    produced its moment: "composite", "batch_only", or "none" (NaN under
    both).
    """
    raw_cache_dir.mkdir(parents=True, exist_ok=True)
    primary_raw_path = raw_cache_dir / f"{name}.h5ad"
    primary = _moments_batch_pass(
        sc_path, name,
        perturbation_column=perturbation_column, control_label=control_label,
        batch_column=batch_column, perturbations=perturbations,
        chunk_size=chunk_size, memory_limit_gb=memory_limit_gb, resume=resume,
        checkpoint_interval=checkpoint_interval,
        raw_path=primary_raw_path, force=force, verbose=verbose,
    )
    raw_paths = [primary_raw_path]
    candidates = primary["candidates"]
    primary_nan = np.isnan(primary["pert_mean"]).all(axis=1)

    stratification = np.full(len(candidates), "composite", dtype=object)
    if fallback_batch_column and primary_nan.any():
        log.info(
            "%s: %d/%d perturbations have no valid '%s' stratum; retrying "
            "them under fallback batch_column=%r",
            name, int(primary_nan.sum()), len(candidates), batch_column, fallback_batch_column,
        )
        fallback_raw_path = raw_cache_dir / f"{name}_fallback.h5ad"
        fallback = _moments_batch_pass(
            sc_path, name,
            perturbation_column=perturbation_column, control_label=control_label,
            batch_column=fallback_batch_column, perturbations=candidates,
            chunk_size=chunk_size, memory_limit_gb=memory_limit_gb, resume=resume,
            checkpoint_interval=checkpoint_interval,
            raw_path=fallback_raw_path, force=force, verbose=verbose,
        )
        raw_paths.append(fallback_raw_path)
        if fallback["candidates"] != candidates:
            raise RuntimeError(
                f"{name}: fallback pass candidate order/set does not match the "
                "primary pass -- cannot merge"
            )
        fallback_nan = np.isnan(fallback["pert_mean"]).all(axis=1)
        use_fallback = primary_nan & ~fallback_nan
        for key in _MOMENT_ARRAY_KEYS:
            primary[key][use_fallback] = fallback[key][use_fallback]
        stratification[use_fallback] = "batch_only"
        stratification[primary_nan & fallback_nan] = "none"

    # Single control baseline vector for 05's control row: average of the
    # per-pert batch-matched control means (no extra scan; the pooled path
    # yields the identical pooled control mean for its single stratum).
    with np.errstate(invalid="ignore"):
        control_profile = np.nanmean(primary["ctrl_matched"], axis=0)

    return {
        "candidates": candidates,
        "gene_symbols": primary["gene_symbols"],
        "var_meta": primary["var_meta"],
        "pert_mean": primary["pert_mean"],
        "ctrl_matched": primary["ctrl_matched"],
        "pert_var": primary["pert_var"],
        "ctrl_var": primary["ctrl_var"],
        "pert_mean_var": primary["pert_mean_var"],
        "ctrl_mean_var": primary["ctrl_mean_var"],
        "pert_mean_var_robust": primary["pert_mean_var_robust"],
        "ctrl_mean_var_robust": primary["ctrl_mean_var_robust"],
        "effect_var_robust": primary["effect_var_robust"],
        "control_profile": control_profile,
        "stratification": stratification if fallback_batch_column else None,
        "raw_paths": raw_paths,
    }


# ---------------------------------------------------------------------------
# Single-stratum closed form (non-batch process)
# ---------------------------------------------------------------------------
def _moments_pooled(
    sc_path: Path,
    *,
    perturbation_column: str,
    control_label: str,
    perturbations: list[str] | None,
    chunk_size: int,
    verbose: int,
) -> dict[str, Any]:
    """Non-batch process: pooled single-stratum moments from the sc file.

    One "batch", so the harmonic weighting degenerates and the SE reduces to the
    closed form `std/sqrt(n)`; `effect_se = sqrt(V_p/n_p + V_c/n_c)`. Out of
    scope for the split mean/std gate (util_moments module docstring) -- QC
    (`min_cells_per_pert`) already guarantees every stratum here has `n>=2`,
    so mean and variance are gated identically; `min_cells=2` is the
    structural floor a variance needs, not a tunable threshold. The robust
    (between-batch) SE is structurally undefined here (B=1 stratum always) --
    `*_var_robust` are NaN for every perturbation, same as the batch process's
    B<2 case.
    """
    min_cells = 2
    backed = ad.read_h5ad(sc_path, backed="r")
    try:
        labels = backed.obs[perturbation_column].astype(str).to_numpy()
        gene_symbols = np.asarray(backed.var_names, dtype=str)
        var_meta = backed.var.copy()
        n_genes = backed.n_vars
        n_obs = backed.n_obs

        present = pd.Index(labels).unique().tolist()
        cands = sorted(p for p in present if p != control_label)
        if perturbations is not None:
            want = set(map(str, perturbations))
            cands = [p for p in cands if p in want]
        groups = [control_label] + cands
        gidx = {g: i for i, g in enumerate(groups)}

        n_grp = len(groups)
        gsum = np.zeros((n_grp, n_genes), dtype=np.float64)
        gsq = np.zeros((n_grp, n_genes), dtype=np.float64)
        gn = np.zeros(n_grp, dtype=np.int64)
        codes = np.array([gidx.get(l, -1) for l in labels], dtype=np.int64)

        for start in range(0, n_obs, chunk_size):
            end = min(start + chunk_size, n_obs)
            block = backed.X[start:end]
            if sp.issparse(block):
                block = block.toarray()
            block = np.asarray(block, dtype=np.float64)
            c = codes[start:end]
            for gi in range(n_grp):
                m = c == gi
                if m.any():
                    sub = block[m]
                    gsum[gi] += sub.sum(axis=0)
                    gsq[gi] += (sub * sub).sum(axis=0)
                    gn[gi] += int(m.sum())
    finally:
        close = getattr(getattr(backed, "file", None), "close", None)
        if callable(close):
            close()

    ci = gidx[control_label]
    if gn[ci] < min_cells:
        raise ValueError(
            f"control '{control_label}' has {gn[ci]} cells (< min_cells={min_cells})"
        )
    ctrl_mean = gsum[ci] / gn[ci]
    ctrl_var = np.maximum(gsq[ci] / gn[ci] - ctrl_mean * ctrl_mean, 0.0)
    ctrl_mean_var = ctrl_var / gn[ci]

    n_cand = len(cands)
    pert_mean = np.full((n_cand, n_genes), np.nan, dtype=np.float64)
    pert_var = np.full((n_cand, n_genes), np.nan, dtype=np.float64)
    pert_mean_var = np.full((n_cand, n_genes), np.nan, dtype=np.float64)
    for j, p in enumerate(cands):
        gi = gidx[p]
        n_p = gn[gi]
        if n_p < min_cells:
            continue
        mean_p = gsum[gi] / n_p
        var_p = np.maximum(gsq[gi] / n_p - mean_p * mean_p, 0.0)
        pert_mean[j] = mean_p
        pert_var[j] = var_p
        pert_mean_var[j] = var_p / n_p

    ctrl_matched = np.broadcast_to(ctrl_mean, (n_cand, n_genes)).copy()
    ctrl_var_matched = np.broadcast_to(ctrl_var, (n_cand, n_genes)).copy()
    ctrl_mean_var_matched = np.broadcast_to(ctrl_mean_var, (n_cand, n_genes)).copy()
    # NaN perts (n_p < min_cells) get NaN matched-control too, so their std/se
    # are NaN rather than falsely precise.
    nan_rows = np.isnan(pert_mean).all(axis=1)
    ctrl_matched[nan_rows] = np.nan
    ctrl_var_matched[nan_rows] = np.nan
    ctrl_mean_var_matched[nan_rows] = np.nan

    return {
        "candidates": cands,
        "gene_symbols": gene_symbols,
        "var_meta": var_meta,
        "pert_mean": pert_mean,
        "ctrl_matched": ctrl_matched,
        "pert_var": pert_var,
        "ctrl_var": ctrl_var_matched,
        "pert_mean_var": pert_mean_var,
        "ctrl_mean_var": ctrl_mean_var_matched,
        "pert_mean_var_robust": np.full((n_cand, n_genes), np.nan, dtype=np.float64),
        "ctrl_mean_var_robust": np.full((n_cand, n_genes), np.nan, dtype=np.float64),
        "effect_var_robust": np.full((n_cand, n_genes), np.nan, dtype=np.float64),
        "control_profile": ctrl_mean,
        "stratification": None,
        "raw_paths": [],
    }


# ---------------------------------------------------------------------------
# Shared assembly: gene-symbol collapse + combined AnnData + atomic write
# ---------------------------------------------------------------------------
def _collapse_linear(mat: np.ndarray, gene_symbols, var_meta):
    """Collapse duplicate gene-symbol columns of a variance-like matrix (sum)."""
    out, var_df, _ = collapse_duplicate_gene_symbols(mat, gene_symbols, var_meta)
    if sp.issparse(out):
        out = out.toarray()
    return np.asarray(out), var_df


def _collapse_log1p(mat: np.ndarray, gene_symbols, var_meta):
    """Collapse duplicate gene-symbol columns of a log1p-mean matrix/vector."""
    out, var_df, _ = collapse_duplicate_gene_symbols(
        mat, gene_symbols, var_meta, log1p_space=True
    )
    if sp.issparse(out):
        out = out.toarray()
    return np.asarray(out), var_df


def _assemble(
    arrays: dict[str, Any],
    *,
    control_label: str,
    batch_column: str | None,
) -> ad.AnnData:
    gene_symbols = arrays["gene_symbols"]
    var_meta = arrays["var_meta"]

    # Collapse duplicate gene symbols: means in log1p space, variances by sum.
    pert_mean, var_df = _collapse_log1p(arrays["pert_mean"], gene_symbols, var_meta)
    ctrl_matched, _ = _collapse_log1p(arrays["ctrl_matched"], gene_symbols, var_meta)
    control_profile, _ = _collapse_log1p(arrays["control_profile"], gene_symbols, var_meta)
    pert_var, _ = _collapse_linear(arrays["pert_var"], gene_symbols, var_meta)
    ctrl_var, _ = _collapse_linear(arrays["ctrl_var"], gene_symbols, var_meta)
    pert_mean_var, _ = _collapse_linear(arrays["pert_mean_var"], gene_symbols, var_meta)
    ctrl_mean_var, _ = _collapse_linear(arrays["ctrl_mean_var"], gene_symbols, var_meta)
    pert_mean_var_robust, _ = _collapse_linear(arrays["pert_mean_var_robust"], gene_symbols, var_meta)
    ctrl_mean_var_robust, _ = _collapse_linear(arrays["ctrl_mean_var_robust"], gene_symbols, var_meta)
    effect_var_robust, _ = _collapse_linear(arrays["effect_var_robust"], gene_symbols, var_meta)

    pert_std = np.sqrt(np.maximum(pert_var, 0.0)).astype(np.float32)
    control_std_matched = np.sqrt(np.maximum(ctrl_var, 0.0)).astype(np.float32)
    pert_mean_se = np.sqrt(np.maximum(pert_mean_var, 0.0)).astype(np.float32)
    ctrl_mean_se = np.sqrt(np.maximum(ctrl_mean_var, 0.0)).astype(np.float32)
    effect_se = np.sqrt(np.maximum(pert_mean_var + ctrl_mean_var, 0.0)).astype(np.float32)
    pert_mean_se_robust = np.sqrt(np.maximum(pert_mean_var_robust, 0.0)).astype(np.float32)
    ctrl_mean_se_robust = np.sqrt(np.maximum(ctrl_mean_var_robust, 0.0)).astype(np.float32)
    effect_se_robust = np.sqrt(np.maximum(effect_var_robust, 0.0)).astype(np.float32)
    effect = (pert_mean - ctrl_matched).astype(np.float32)
    perturbation_profile = pert_mean.astype(np.float32)

    candidates = arrays["candidates"]
    # Always "perturbation", regardless of the source dataset's own
    # perturbation_column ("gene", "gene_target", ...) -- this obs column
    # duplicates the index (also always named "perturbation"), so keying it
    # off the dataset-specific name would leak that internal convention into
    # every *_moments.h5ad's schema.
    obs_data: dict[str, Any] = {"perturbation": candidates}
    stratification = arrays.get("stratification")
    if stratification is not None:
        obs_data["stratification"] = pd.Categorical(stratification)
    obs = pd.DataFrame(
        obs_data,
        index=pd.Index([str(c) for c in candidates], name="perturbation"),
    )
    out = ad.AnnData(X=effect, obs=obs, var=var_df)
    out.layers["perturbation_profile"] = perturbation_profile
    out.layers["pert_std"] = pert_std
    out.layers["control_std_matched"] = control_std_matched
    out.layers["pert_mean_se"] = pert_mean_se
    out.layers["ctrl_mean_se"] = ctrl_mean_se
    out.layers["effect_se"] = effect_se
    out.layers["pert_mean_se_robust"] = pert_mean_se_robust
    out.layers["ctrl_mean_se_robust"] = ctrl_mean_se_robust
    out.layers["effect_se_robust"] = effect_se_robust
    out.uns["control_profile"] = np.asarray(control_profile, dtype=np.float32)
    out.uns["batch_column"] = str(batch_column) if batch_column else ""
    out.uns["control_label"] = str(control_label)
    out.uns["cell_gate_note"] = (
        "mean/effect: n>=1 per side per batch (or single pooled stratum); "
        "std/se: shared pooled within-batch variance from n>=2 batches only "
        "(structural floor, not a tunable threshold)."
    )
    out.uns["channels"] = np.asarray(list(_CHANNELS), dtype=object)
    stratification_note = (
        " obs['stratification'] marks, per perturbation, which batch_column "
        "actually produced its moment: 'composite' (the full batch_column), "
        "'batch_only' (fell back to a coarser key -- too few cells in the fine "
        "strata), or 'none' (NaN under both)."
        if stratification is not None else ""
    )
    out.uns["moments_note"] = (
        "one-pass avg_log/std/se on QC-filtered, normalized, gene-deduped sc "
        "data; mean/effect gated n>=1 per side per batch, std = pooled "
        "within-batch variance from n>=2 batches, se = that pooled variance "
        "times Sum(w^2/n)/Sum(w)^2 over all n>=1 batches (see cell_gate_note). "
        "layers: perturbation_profile=corrected absolute log-mean; X=effect; "
        "pert_std/control_std_matched=within-batch population std (cell spread); "
        "pert_mean_se/ctrl_mean_se/effect_se=SE of the mean/effect estimate "
        "(model-based, assumes one shared within-batch variance). "
        "pert_mean_se_robust/ctrl_mean_se_robust/effect_se_robust=empirical "
        "(matched-stratum, 'sandwich') SE from the weighted spread of "
        "per-batch point estimates across batches -- no shared-variance "
        "assumption, robust to batch heterogeneity, but needs >=2 "
        "participating batches (NaN otherwise, incl. always-NaN in the "
        "pooled/non-batch process). effect_se_robust is computed from the "
        "per-batch matched difference directly (not pert/ctrl combined), so "
        "it captures within-batch noise cancellation from matching. See "
        "plan/20260827_robust_between_batch_se.md."
        f"{stratification_note} See util_moments.py."
    )

    n_nan_mean = int(np.isnan(perturbation_profile).all(axis=1).sum())
    if n_nan_mean:
        log.warning(
            "%d/%d perturbations have no batch with any cells on both sides "
            "(after any fallback); their mean/std/se are all NaN (floored "
            "downstream).",
            n_nan_mean, len(candidates),
        )
    n_nan_std_only = int(
        (np.isnan(pert_std).all(axis=1) & ~np.isnan(perturbation_profile).all(axis=1)).sum()
    )
    if n_nan_std_only:
        log.warning(
            "%d/%d perturbations have a finite mean but no batch with >=2 "
            "perturbation cells (all participating batches are singletons); "
            "their std/se are NaN.",
            n_nan_std_only, len(candidates),
        )
    return out


def compute_moments(
    sc_path: str | Path,
    *,
    name: str,
    perturbation_column: str = "perturbation",
    control_label: str,
    batch_column: str | None = None,
    fallback_batch_column: str | None = None,
    perturbations: list[str] | None = None,
    chunk_size: int | None = None,
    memory_limit_gb: float | None = None,
    resume: bool = False,
    checkpoint_interval: int | None = None,
    raw_cache_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    force: bool = False,
    verbose: int = 1,
) -> ad.AnnData:
    """Compute the combined avg_log/std/se moments for one dataset.

    Dispatches to the batch-correction process (``batch_column`` given) or the
    non-batch pooled process (``batch_column is None``). Reads the QC-filtered,
    normalized ``sc_path``; writes one combined ``<name>_moments.h5ad``.

    ``fallback_batch_column``, if given (only meaningful with ``batch_column``
    set), is retried per-perturbation for any perturbation with no valid
    stratum under ``batch_column`` -- see ``_moments_batch``.
    """
    sc_path = Path(sc_path)
    if batch_column:
        raw_cache_dir = Path(raw_cache_dir) if raw_cache_dir is not None else sc_path.parent
        arrays = _moments_batch(
            sc_path, name,
            perturbation_column=perturbation_column,
            control_label=control_label,
            batch_column=batch_column,
            fallback_batch_column=fallback_batch_column,
            perturbations=perturbations,
            chunk_size=chunk_size,
            memory_limit_gb=memory_limit_gb,
            resume=resume,
            checkpoint_interval=checkpoint_interval,
            raw_cache_dir=raw_cache_dir,
            force=force,
            verbose=verbose,
        )
    else:
        arrays = _moments_pooled(
            sc_path,
            perturbation_column=perturbation_column,
            control_label=control_label,
            perturbations=perturbations,
            chunk_size=chunk_size or 4096,
            verbose=verbose,
        )

    out = _assemble(
        arrays,
        control_label=control_label,
        batch_column=batch_column,
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = output_path.with_suffix(".tmp.h5ad")
        out.write_h5ad(tmp)
        tmp.replace(output_path)
        if verbose:
            log.info("saved moments -> %s", output_path)

    # The raw multi-channel cache(s) exist only for mid-pass --resume; the
    # final combined file is what downstream reads, so reclaim them once
    # written (one per pass: primary, and fallback if it ran).
    if output_path is not None:
        for raw_path in arrays.get("raw_paths", []):
            if raw_path is None or not Path(raw_path).exists():
                continue
            for p in (Path(raw_path), Path(f"{raw_path}.progress.json")):
                try:
                    p.unlink()
                except OSError:
                    pass
    return out
