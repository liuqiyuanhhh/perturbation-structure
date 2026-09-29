"""Build the downsampled VCC-H1 dataset, data/origin/VCC-subsampled.h5ad, from
data/origin/VCC.h5ad (raw counts; steps 01-02 of ../crispyx_pseudobulk_de).

  1. keep 16 of the 48 batches at random (whole batches);
  2. within them, sample up to 100 cells per perturbation at random (perturbations
     with fewer are kept in full) and keep all control cells;
  3. thin each cell's counts to at most 13,000 by sampling UMIs without replacement.

All random steps use random_state=0. The result is then processed like the other
screens (steps 03, 04b and 04 of ../crispyx_pseudobulk_de with --dataset VCC-subsampled).

    python 1_preprocessing/vcc_subsampling/downsample_vcc.py
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import crispyx as cx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "crispyx_pseudobulk_de"))
import datasets  # noqa: E402
from paths import ORIGIN_DIR, TMP_DIR  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SOURCE = ORIGIN_DIR / "VCC.h5ad"
OUTPUT = ORIGIN_DIR / "VCC-subsampled.h5ad"
N_BATCHES_KEEP = 16
N_CELLS_PER_PERT = 100
MAX_COUNTS_PER_CELL = 13_000
RANDOM_STATE = 0


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="overwrite an existing output")
    args = ap.parse_args(argv)
    if OUTPUT.exists() and not args.force:
        log.info("SKIP: %s exists (pass --force to overwrite)", OUTPUT)
        return
    if not SOURCE.exists():
        raise SystemExit(f"{SOURCE} not found: run crispyx_pseudobulk_de/02_setup.py --dataset VCC first")
    cfg = datasets.get("VCC")
    work = TMP_DIR / "vcc_downsample"
    work.mkdir(parents=True, exist_ok=True)
    stage1, stage2a, stage2b = (work / f"{s}.h5ad" for s in ("batches", "per_pert", "with_controls"))

    # 1. whole batches
    cx.pp.subsample(SOURCE, n=N_BATCHES_KEEP, groupby=None, unit=cfg["batch_col"],
                    random_state=RANDOM_STATE, output_path=stage1, verbose=True)

    # 2. up to 100 cells per perturbation, then restore all control cells of the kept batches
    cx.pp.subsample(stage1, n=N_CELLS_PER_PERT, groupby=cfg["pert_col"], unit="cell",
                    drop_insufficient=False, random_state=RANDOM_STATE, output_path=stage2a, verbose=True)
    b1 = cx.read_backed(stage1)
    obs1, n_vars = b1.obs.copy(), b1.n_vars
    b1.file.close()
    b2 = cx.read_backed(stage2a)
    kept = set(b2.obs_names)
    b2.file.close()
    is_control = obs1[cfg["pert_col"]].astype(str).to_numpy() == cfg["control"]
    mask = is_control | (obs1.index.isin(kept) & ~is_control)
    log.info("control cells %d (all kept), perturbed cells %d", int(is_control.sum()), int((mask & ~is_control).sum()))
    cx.data.write_filtered_subset(stage1, cell_mask=mask, gene_mask=np.ones(n_vars, dtype=bool), output_path=stage2b)

    # 3. thin counts
    ORIGIN_DIR.mkdir(parents=True, exist_ok=True)
    cx.pp.downsample_counts(stage2b, output_path=OUTPUT, counts_per_cell=MAX_COUNTS_PER_CELL,
                            random_state=RANDOM_STATE, verbose=True)
    for p in (stage1, stage2a, stage2b):
        p.unlink(missing_ok=True)
    log.info("Wrote %s", OUTPUT)


if __name__ == "__main__":
    main()
