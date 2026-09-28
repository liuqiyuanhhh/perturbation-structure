"""Readers for the moments, DE and selection files, and the leading-factor fit.

A dataset with several components (Feng-GW) is merged: perturbations in more than one
component are dropped, genes unique in every component kept in first-component order."""

from collections import Counter

import h5py
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds

from .paths import L1_SELECTION, QC_GENE_PANELS, SELECTION_REFERENCE_DATASET


def axis_names(handle, axis):
    """The obs or var index of an open .h5ad file."""
    group = handle[axis]
    return group[group.attrs["_index"]].asstr()[:].tolist()


def de_perturbations(handle):
    """The perturbation of each row of an open DE file."""
    column = handle["uns"].attrs.get("perturbation_column", "perturbation")
    return handle[f"obs/{column}"].asstr()[:].tolist()


def unique_columns(names):
    """{name: position} of the names that occur once."""
    counts = Counter(names)
    return {name: position for position, name in enumerate(names) if counts[name] == 1}


def load_moments(paths, layers=("X",), perturbations=None, genes=None):
    """Merged moments of one dataset: perturbations, genes, the `layers` and control_mean
    (component controls weighted by their numbers of perturbations)."""
    axes = []
    for path in paths:
        with h5py.File(path, "r") as handle:
            axes.append((axis_names(handle, "obs"), unique_columns(axis_names(handle, "var"))))
    counts = Counter(name for names, _ in axes for name in names)
    merged = [gene for gene in axes[0][1] if all(gene in columns for _, columns in axes[1:])]
    data = {"perturbations": [], "genes": [gene for gene in merged if genes is None or gene in genes]}
    parts = {layer: [] for layer in layers}
    control_sum = control_weight = 0
    for path, (names, columns) in zip(paths, axes):
        kept = [row for row, name in enumerate(names) if counts[name] == 1]
        rows = np.array([row for row in kept if perturbations is None or names[row] in perturbations],
                        dtype=np.int64)
        cols = np.array([columns[gene] for gene in data["genes"]], dtype=np.int64)
        with h5py.File(path, "r") as handle:
            control = np.asarray(handle["uns/control_profile"], dtype=np.float64)[cols]
            for layer in layers:
                parts[layer].append(handle[layer][:][np.ix_(rows, cols)])
        data["perturbations"] += [names[row] for row in rows]
        finite = np.isfinite(control)
        control_sum = control_sum + np.where(finite, control * len(kept), 0.0)
        control_weight = control_weight + finite * len(kept)
    data["control_mean"] = np.divide(control_sum, control_weight, where=control_weight > 0,
                                     out=np.full(len(data["genes"]), np.nan))
    for layer in layers:
        data[layer] = np.concatenate(parts[layer])
    return data


def load_effects(paths, perturbations, genes, se=False):
    """Float32 effects (and SEs if se=True), each perturbation's own target set to 0."""
    layers = ("X", "layers/effect_se") if se else ("X",)
    data = load_moments(paths, layers, perturbations, set(genes))
    position = {gene: j for j, gene in enumerate(data["genes"])}
    targets = [(i, position[name]) for i, name in enumerate(data["perturbations"]) if name in position]
    rows, cols = np.array(targets, dtype=np.int64).reshape(-1, 2).T
    matrices = []
    for layer in layers:
        matrix = np.ascontiguousarray(data[layer], dtype=np.float32)
        matrix[rows, cols] = 0.0
        matrices.append(matrix)
    return (data["perturbations"], data["genes"], *matrices)


def fit_and_remove_first_svd_factor(effects, tolerance=1e-7, maxiter=None):
    """Remove the leading factor sigma1 u1 v1' in place (sign: largest |v1| positive).
    ARPACK's last digits depend on the layout: pass the float32 matrix of load_effects."""
    u, s, vt = svds(effects, k=1, which="LM", v0=np.linspace(1.0, 2.0, min(effects.shape), dtype=np.float32),
                    tol=tolerance, maxiter=maxiter, return_singular_vectors=True, solver="arpack")
    sigma = float(s[0])
    u1 = u[:, 0].astype(np.float64)
    v1 = vt[0].astype(np.float64)
    u1 /= np.linalg.norm(u1)
    v1 /= np.linalg.norm(v1)
    if v1[np.argmax(np.abs(v1))] < 0:
        u1, v1 = -u1, -v1
    effects -= np.outer(u1, sigma * v1).astype(np.float32)
    return sigma, u1, v1


def read_selection(path, column, item="perturbation"):
    """A saved table with boolean `column`, and {dataset: `item` values where it is True}."""
    frame = pd.read_csv(path)
    frame[column] = frame[column].astype(str).str.strip().str.lower().eq("true")
    return frame, {
        str(dataset): set(group.loc[group[column], item].astype(str))
        for dataset, group in frame.groupby("dataset", sort=False)
    }


def load_quality_audit(path):
    """A saved perturbation-QC table and {dataset: QC-passing perturbations}."""
    return read_selection(path, "passes_perturbation_quality_filter")


def load_selections(
    quality_filter_file=QC_GENE_PANELS / "perturbation_quality_filter.csv.gz",
    gene_selection_file=QC_GENE_PANELS / "outcome_gene_scores_and_selections.csv.gz",
    l1_membership_file=L1_SELECTION / "L1_strong_perturbation_membership.csv.gz",
):
    """The 1_preprocessing selections {dataset: {"qc", "primary", "feature", "L1": set}};
    VCC-subsampled takes those of VCC."""
    tables = {
        "qc": (quality_filter_file, "passes_perturbation_quality_filter", "perturbation"),
        "primary": (gene_selection_file, "selected_primary_control_or_treated_ge_0.08", "gene"),
        "feature": (gene_selection_file, "selected_top2000_effect_variance", "gene"),
        "L1": (l1_membership_file, "selected_L1_strong", "perturbation"),
    }
    selections = {}
    for name, (path, column, item) in tables.items():
        for dataset, members in read_selection(path, column, item)[1].items():
            selections.setdefault(dataset, {})[name] = members
    for dataset, source in SELECTION_REFERENCE_DATASET.items():
        inherited = selections.setdefault(dataset, {})
        for name, members in selections[source].items():
            inherited.setdefault(name, set(members))
    return selections
