"""Outcome-gene and perturbation QC: primary genes have control or mean treated
expression >= 0.08; a perturbation passes if its target has control expression >= 0.08
and inactivation efficiency -tau/control >= 0.10.  Run as a script, it writes the
prediction QC tables (Fig 1d, S2) from the moments files of each --dataset."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .effects import load_moments, read_selection
from .paths import PREDICTION_QC_DIR

CONTROL_LABELS = {"control", "ctrl", "non-targeting", "non_targeting", "nt"}


def load_qc_dataset(paths):
    """Effects (float32), control mean and mean treated expression of one dataset."""
    data = load_moments(paths, ("X", "layers/perturbation_profile"))
    treated = np.array([name.strip().lower() not in CONTROL_LABELS for name in data["perturbations"]])
    profile = np.asarray(data.pop("layers/perturbation_profile")[treated], dtype=np.float64)
    finite = np.isfinite(profile)
    count = finite.sum(axis=0)
    data["mean_treated"] = np.divide(np.where(finite, profile, 0.0).sum(axis=0), count,
                                     where=count > 0, out=np.full(len(count), np.nan))
    data["effects"] = np.asarray(data.pop("X"), dtype=np.float32)
    return data


def primary_outcomes(control_mean, mean_treated, cutoff=0.08):
    return (np.isfinite(control_mean) & (control_mean >= cutoff)) | (
        np.isfinite(mean_treated) & (mean_treated >= cutoff))


def perturbation_qc(dataset, data, target_cutoff=0.08, minimum_efficiency=0.10):
    """Per perturbation: target control expression, inactivation efficiency, QC pass."""
    column = {gene: j for j, gene in enumerate(data["genes"])}
    rows = [i for i, name in enumerate(data["perturbations"]) if name in column]
    columns = [column[data["perturbations"][i]] for i in rows]
    control = np.full(len(data["perturbations"]), np.nan)
    effect = np.full(len(data["perturbations"]), np.nan)
    control[rows] = data["control_mean"][columns]
    effect[rows] = data["effects"][rows, columns]
    with np.errstate(divide="ignore", invalid="ignore"):
        efficiency = np.where(np.isfinite(control) & (control != 0) & np.isfinite(effect),
                              -effect / control, np.nan)
    return pd.DataFrame({
        "dataset": dataset,
        "perturbation": data["perturbations"],
        "target_control_expression": control,
        "inactivation_efficiency": efficiency,
        "passes_perturbation_quality_filter": (
            np.isfinite(control) & (control >= target_cutoff) & (efficiency >= minimum_efficiency)),
    })


def load_primary_outcome_lists(path):
    """Primary outcome genes of each dataset, in file order."""
    column = "selected_primary_control_or_treated_ge_0.08"
    frame, _ = read_selection(path, column, "gene")
    return {str(dataset): group.loc[group[column], "gene"].astype(str).tolist()
            for dataset, group in frame.groupby("dataset", sort=False)}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", action="append", nargs="+", metavar=("LABEL", "H5AD"),
                        required=True,
                        help="dataset label followed by its moments file(s); repeat for each dataset")
    parser.add_argument("--output-dir", type=Path, default=PREDICTION_QC_DIR)
    parser.add_argument("--primary-expression-cutoff", type=float, default=0.08)
    parser.add_argument("--target-control-cutoff", type=float, default=0.08)
    parser.add_argument("--minimum-inactivation-efficiency", type=float, default=0.10)
    args = parser.parse_args()

    outcome_tables, perturbation_tables = [], []
    for label, *files in args.dataset:
        data = load_qc_dataset([Path(name) for name in files])
        primary = primary_outcomes(data["control_mean"], data["mean_treated"],
                                   args.primary_expression_cutoff)
        outcome_tables.append(pd.DataFrame({
            "dataset": label,
            "gene": data["genes"],
            "control_mean": data["control_mean"],
            "reconstructed_mean_treated_across_all_pre_qc_perturbations": data["mean_treated"],
            "selected_primary_control_or_treated_ge_0.08": primary,
        }))
        audit = perturbation_qc(label, data, args.target_control_cutoff,
                                args.minimum_inactivation_efficiency)
        perturbation_tables.append(audit)
        print(f"{label}: {int(primary.sum()):,}/{len(primary):,} primary outcomes, "
              f"{int(audit['passes_perturbation_quality_filter'].sum()):,}/{len(audit):,} "
              "perturbations pass QC", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(outcome_tables, ignore_index=True).to_csv(
        args.output_dir / "outcome_expression_qc.csv.gz", index=False, compression="gzip")
    pd.concat(perturbation_tables, ignore_index=True).to_csv(
        args.output_dir / "perturbation_quality_filter.csv.gz", index=False, compression="gzip")


if __name__ == "__main__":
    main()
