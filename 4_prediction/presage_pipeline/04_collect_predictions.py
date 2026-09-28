"""Step 04 -- gather the per-fold predictions into one file.

Writes next to the per-fold pkls:

  <DATASET>_presage_test_predictions.pkl   {fold: DataFrame}
  <DATASET>_presage_test_concat.pkl        one DataFrame, all folds stacked

The five test sets are disjoint and together cover every perturbation, so the
stack is a complete [n_perturbations x n_genes] matrix; both properties are
checked.  Values are predicted effects (control-subtracted, log1p(CPTT) scale).

Run: PP_DATASET=VCC python 04_collect_predictions.py
"""

import json

import joblib
import pandas as pd

from config import DATASET, N_FOLDS, PERT_LIST_FILE, PRED_DIR, RUN, split_json


def main():
    print(f"dataset: {DATASET}")
    per_fold = {}
    missing = []
    for fold in range(1, N_FOLDS + 1):
        path = PRED_DIR / f"{RUN}_seed_{fold}_test_predictions.pkl"
        if not path.exists():
            missing.append(fold)
            continue
        df = joblib.load(path)
        with open(split_json(fold)) as fh:
            expected = set(json.load(fh)["test"])
        if set(df.index) != expected:
            raise SystemExit(f"fold {fold}: predictions cover {len(df.index)} perturbations "
                             f"but the split's test set has {len(expected)}")
        per_fold[fold] = df
        print(f"  fold {fold}: {df.shape[0]:>5} perturbations x {df.shape[1]} genes")

    if missing:
        print(f"\nnot yet available: fold(s) {missing}")
    if not per_fold:
        raise SystemExit("no predictions found")

    concat = pd.concat(per_fold.values())
    dupes = concat.index[concat.index.duplicated()]
    if len(dupes):
        raise SystemExit(f"{len(dupes)} perturbations predicted by >1 fold: {list(dupes)[:5]}")
    if not missing:
        with open(PERT_LIST_FILE) as fh:
            all_perts = set(json.load(fh))
        gap = all_perts - set(concat.index)
        if gap:
            raise SystemExit(f"{len(gap)} perturbations predicted by no fold: {sorted(gap)[:5]}")
        print(f"\n  complete: {len(concat)} perturbations, each predicted exactly once")

    concat = concat.sort_index()
    dict_path = PRED_DIR / f"{RUN}_presage_test_predictions.pkl"
    concat_path = PRED_DIR / f"{RUN}_presage_test_concat.pkl"
    joblib.dump(per_fold, dict_path)
    joblib.dump(concat, concat_path)
    print(f"\nwrote {dict_path}")
    print(f"wrote {concat_path}  {concat.shape}")


if __name__ == "__main__":
    main()
