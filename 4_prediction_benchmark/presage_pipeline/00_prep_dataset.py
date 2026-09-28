"""Step 00 -- point PRESAGE's data_dir at the GEARS build, and write splits + DEGs.

PRESAGE looks for its raw input at data_dir/<dataset>/perturb_processed.h5ad.
The GEARS build already is exactly that file, so it is symlinked:

    data/<DATASET>/perturb_processed.h5ad -> <GEARS build>/perturb_processed.h5ad
    splits/<DATASET>_random_splits/seed_{1..5}.json
    data/<DATASET>/degs/merged.degs.json

<DATASET>_processed.h5ad is written later, by presage_datamodule.prepare_data on
the first training fold.

The DEG file has to exist and cover every perturbation even though PRESAGE's
evaluator is never run: scPerturbDataModule.load_preprocessed opens it and
ModelHarness.ind_to_pert asserts every perturbation key is present.  It is built
here from the GEARS build's uns["rank_genes_groups_cov_all"], which is what
presage_datamodule.prepare_data would otherwise do itself.

Run: PP_DATASET=VCC python 00_prep_dataset.py
"""

import json

import h5py
import joblib

from config import (
    CONTROL_KEY,
    DATASET,
    DATASET_DIR,
    DEG_FILE,
    DEG_TOP_N,
    N_FOLDS,
    PERT_LIST_FILE,
    RAW_H5AD,
    SC_H5AD,
    SPLIT_DIR,
    SPLIT_SRC_DIR,
    harmonize,
    split_json,
    split_pkl,
)


def link_h5ad():
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    if not SC_H5AD.exists():
        raise SystemExit(f"{SC_H5AD} does not exist")
    if RAW_H5AD.is_symlink():
        if RAW_H5AD.resolve() == SC_H5AD.resolve():
            print(f"  symlink already correct: {RAW_H5AD} -> {SC_H5AD}")
            return
        RAW_H5AD.unlink()
    elif RAW_H5AD.exists():
        raise SystemExit(f"{RAW_H5AD} exists and is not a symlink -- refusing to replace it")
    RAW_H5AD.symlink_to(SC_H5AD)
    print(f"  {RAW_H5AD} -> {SC_H5AD}")


def load_folds():
    """{fold: {split: [harmonized perts]}}, control dropped."""
    folds = {}
    for k in range(1, N_FOLDS + 1):
        src = split_pkl(k)
        raw = joblib.load(src)
        out = {}
        for split in ("train", "val", "test"):
            perts = [harmonize(p) for p in raw[split]]
            # PRESAGE adds control back itself in setup(); leaving it in would
            # duplicate it and, for "test", emit a control prediction row.
            out[split] = sorted({p for p in perts if p != CONTROL_KEY})
        folds[k] = out
        print(f"  fold {k}: train {len(out['train'])}  val {len(out['val'])}  "
              f"test {len(out['test'])}  (from {src.name})")
    return folds


def deg_key_to_pert(key: str) -> str:
    """'hESC_AASS+ctrl_1+1' -> 'AASS'.

    Same result as the regex in presage_datamodule.prepare_data: drop the GEARS
    dose suffix, drop the leading cell-type prefix, then harmonize.  No cell-type
    prefix here contains an underscore.
    """
    key = key.replace("_1+1", "")
    rest = key.split("_", 1)[1] if "_" in key else key
    return harmonize(rest)


def load_degs():
    with h5py.File(SC_H5AD, "r") as f:
        grp = f["uns/rank_genes_groups_cov_all"]
        degs = {}
        for key in grp.keys():
            pert = deg_key_to_pert(key)
            names = grp[key][:DEG_TOP_N]
            degs[pert] = [n.decode() if isinstance(n, bytes) else str(n) for n in names]
    return degs


def main():
    print(f"dataset: {DATASET}")

    print(f"\nLinking the GEARS build into {DATASET_DIR}")
    link_h5ad()

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    DEG_FILE.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nReading CV5 folds from {SPLIT_SRC_DIR}")
    folds = load_folds()

    all_perts = sorted({p for f in folds.values() for s in f.values() for p in s})
    print(f"\n{len(all_perts)} distinct perturbations across all folds")

    print(f"\nReading DEG rankings from {SC_H5AD.name}")
    degs = load_degs()
    print(f"  {len(degs)} perturbations, top {DEG_TOP_N} genes each")

    missing = [p for p in all_perts if p not in degs]
    if missing:
        raise SystemExit(
            f"{len(missing)} perturbations have no DEG entry, e.g. {missing[:10]}.\n"
            "ModelHarness.ind_to_pert would assert on these during predict."
        )

    for k, split in folds.items():
        with open(split_json(k), "w") as fh:
            json.dump(split, fh, indent=1)
        print(f"  wrote {split_json(k)}")

    with open(DEG_FILE, "w") as fh:
        json.dump(degs, fh)
    print(f"  wrote {DEG_FILE}")

    with open(PERT_LIST_FILE, "w") as fh:
        json.dump(all_perts, fh, indent=1)
    print(f"  wrote {PERT_LIST_FILE}")


if __name__ == "__main__":
    main()
