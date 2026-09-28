"""Step 01 -- one Perturb-seq knowledge source per external screen.

Methods, "PRESAGE (+Perturb-seq) embeddings from other Perturb-seq datasets": the
pseudobulk matrix of perturbations by (feature) genes of each other screen is a
knowledge source, reduced by PCA to 128 components.  Each external screen is its
own source, with its own source-specific MLP and attention weight.

The effect dict already holds that matrix for every screen (index = perturbed
gene, columns = feature genes), and read_and_embed runs the PCA itself for every
source it loads.  So this step only drops control rows and writes each matrix as a
plain pickle: no PCA (it would be done twice), no row normalisation, no alignment
of gene spaces across screens.  These screens are never trained on, so one file
serves all five folds and every target that may use it.

Which screens a target may use is decided by config.kept_pert_sources() and
recorded in manifest.<DATASET>.json.

Run: PP_DATASET=VCC python 01_build_pert_sources.py
"""

import argparse
import json
import pickle

import joblib
import numpy as np

from config import (
    CONTROL_LABELS,
    DATASET,
    EFFECT_DICT,
    EXCLUDE_DATASETS,
    PERT_SRC_DIR,
    kept_pert_sources,
    pert_source_pkl,
)


def prepare(df, key):
    """Clean one effect matrix into the form read_and_embed expects."""
    ctrl = [p for p in df.index if str(p).strip().lower() in CONTROL_LABELS]
    if ctrl:
        df = df.drop(index=ctrl)

    # read_and_embed z-scores each column and runs PCA, so a NaN would propagate
    # into every component.  Fill a trace amount; refuse more than 1%.
    arr = df.to_numpy()
    n_nan = int(np.isnan(arr).sum())
    if n_nan:
        frac = n_nan / arr.size
        print(f"      {key}: {n_nan} NaN ({frac:.2%}) -> filling with 0.0")
        if frac > 0.01:
            raise SystemExit(f"{key} is {frac:.1%} NaN, too much to fill silently.")
        df = df.fillna(0.0)
    return df, len(ctrl), n_nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="rewrite source pkls that already exist")
    args = ap.parse_args()

    PERT_SRC_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading {EFFECT_DICT}")
    effects = joblib.load(EFFECT_DICT)
    print(f"  {len(effects)} datasets in the dict")

    if DATASET not in effects:
        raise SystemExit(f"target {DATASET!r} is not a key of the effect dict; "
                         f"keys are {sorted(effects)}")

    kept = kept_pert_sources(effects.keys())
    print(f"\n  target   : {DATASET}")
    print(f"  excluded : {', '.join(EXCLUDE_DATASETS)}")
    print(f"  sources  : {len(kept)}")

    manifest = []
    for key in kept:
        out = pert_source_pkl(key)
        if out.exists() and not args.force:
            with open(out, "rb") as fh:
                df = pickle.load(fh)
            n_ctrl, n_nan = 0, 0
            status = "reused"
        else:
            df, n_ctrl, n_nan = prepare(effects[key], key)
            # plain pickle: read_and_embed loads sources with pkl.load
            with open(out, "wb") as fh:
                pickle.dump(df, fh, protocol=pickle.HIGHEST_PROTOCOL)
            status = "wrote "

        print(f"      {status} {key:<22} {df.shape[0]:>5} perts x {df.shape[1]:>5} genes")
        manifest.append(dict(dataset=key, path=str(out), n_perturbations=int(df.shape[0]),
                             n_genes=int(df.shape[1]), n_control_rows_dropped=n_ctrl,
                             n_nan_filled=n_nan))

    manifest_path = PERT_SRC_DIR / f"manifest.{DATASET}.json"
    with open(manifest_path, "w") as fh:
        json.dump(dict(target=DATASET, excluded=list(EXCLUDE_DATASETS), sources=manifest),
                  fh, indent=1)
    print(f"\nwrote {manifest_path}  ({len(manifest)} sources for {DATASET})")


if __name__ == "__main__":
    main()
