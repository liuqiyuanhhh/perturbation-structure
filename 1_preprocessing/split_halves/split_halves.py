"""Split-half replicates (5 seeds x 2 halves) and their moments.

    data/sc/sc_<dataset>.h5ad -> data/split/split_<dataset>_seed<s>_half<h>_moments.h5ad

For each seed, the cells of every perturbation and the control cells are divided
into two halves within batches (datasets.strat_col), and the moments of step 04 of
../crispyx_pseudobulk_de are recomputed on each half.

The halves used in the paper are given by data/split/split_<dataset>_assignments.csv.gz
(one row per cell of the single-cell file; columns seed0..seed4 hold the half, 0 or
1), and are rebuilt from it by default. With --random, a new split is drawn instead
with crispyx.pp.subsample (frac=0.5 within batches, random_state=seed; half 1 is
the complement) and its assignments are written to that file; the paper's split was
drawn this way with crispyx 0.1.2.

    python 1_preprocessing/split_halves/split_halves.py --dataset Replogle-E-k562
    python 1_preprocessing/split_halves/split_halves.py --dataset all
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import crispyx as cx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "crispyx_pseudobulk_de"))
import datasets  # noqa: E402
from paths import SC_DIR, SPLIT_DIR, TMP_DIR, sc_file, split_assignments_file, split_file  # noqa: E402
from util_de import check_crispyx, set_scratch_dir  # noqa: E402
from util_moments import compute_moments  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SPLIT_SC_DIR = TMP_DIR / "split_sc"  # single cells of each half (removed after use)
RAW_CACHE_DIR = TMP_DIR / "moments_raw_sub"


def write_halves(name: str, seed: int, assignments: pd.DataFrame | None) -> tuple[list[Path], np.ndarray]:
    """Write the two halves of the single-cell file for one seed; return their paths and each cell's half."""
    sc_path = SC_DIR / sc_file(name)
    paths = [SPLIT_SC_DIR / f"{name}_seed{seed}_half{h}.h5ad" for h in (0, 1)]
    SPLIT_SC_DIR.mkdir(parents=True, exist_ok=True)
    obs = cx.data.load_obs(sc_path)
    n_vars = len(cx.data.load_var(sc_path))
    if assignments is not None:
        half = assignments[f"seed{seed}"].reindex(obs.index)
        if half.isna().any():
            raise ValueError(f"{name}: {int(half.isna().sum())} cells missing from the assignment file")
        masks = [(half == h).to_numpy() for h in (0, 1)]
        for mask, path in zip(masks, paths):
            cx.qc.write_filtered_subset(sc_path, cell_mask=mask, gene_mask=np.ones(n_vars, dtype=bool),
                                        output_path=path)
    else:
        cx.pp.subsample(sc_path, frac=0.5, groupby=datasets.strat_col(name), unit="cell",
                        random_state=seed, chunk_size=4096, output_path=paths[0], verbose=False)
        in_half0 = obs.index.isin(cx.data.load_obs(paths[0]).index)
        cx.qc.write_filtered_subset(sc_path, cell_mask=~in_half0, gene_mask=np.ones(n_vars, dtype=bool),
                                    output_path=paths[1], chunk_size=4096)
        masks = [in_half0, ~in_half0]
    if int(masks[0].sum()) + int(masks[1].sum()) != len(obs) or (masks[0] & masks[1]).any():
        raise ValueError(f"{name} seed {seed}: halves do not partition the cells")
    log.info("%s seed %d: halves of %d and %d cells", name, seed, int(masks[0].sum()), int(masks[1].sum()))
    return paths, masks[1].astype(np.int8)


def process_dataset(name: str, args: argparse.Namespace) -> None:
    cfg = datasets.get(name)
    assignments_path = SPLIT_DIR / split_assignments_file(name)
    assignments = None if args.random else pd.read_csv(assignments_path, index_col="cell")
    drawn = {}
    for seed in args.seeds:
        outs = [SPLIT_DIR / split_file(name, seed, h) for h in (0, 1)]
        if all(o.exists() for o in outs) and not args.force:
            log.info("SKIP %s seed %d: outputs exist", name, seed)
            continue
        halves, drawn[f"seed{seed}"] = write_halves(name, seed, assignments)
        for half_path, out_path in zip(halves, outs):
            # Same estimator as step 04, on one half (no plain-batch fallback).
            compute_moments(
                half_path, name=half_path.stem,
                perturbation_column=cfg["pert_col"], control_label=cfg["control"],
                batch_column=datasets.strat_col(name),
                raw_cache_dir=RAW_CACHE_DIR, output_path=out_path,
                memory_limit_gb=args.memory_limit_gb, force=True, verbose=1,
            )
            half_path.unlink()
    if args.random and drawn:
        cells = pd.Index(cx.data.load_obs(SC_DIR / sc_file(name)).index, name="cell")
        pd.DataFrame(drawn, index=cells).to_csv(assignments_path, compression="gzip")
        log.info("%s: assignments of %s -> %s", name, ", ".join(drawn), assignments_path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="dataset name, or 'all'")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--random", action="store_true", help="draw a new split instead of using the assignment file")
    ap.add_argument("--memory-limit-gb", type=float, default=None)
    ap.add_argument("--force", action="store_true", help="rebuild existing outputs")
    args = ap.parse_args(argv)
    check_crispyx()
    set_scratch_dir()
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    failed = []
    for name in [n for n in datasets.resolve(args.dataset) if datasets.get(n)["split"]]:
        try:
            process_dataset(name, args)
        except Exception:
            log.exception("FAILED: %s", name)
            failed.append(name)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
