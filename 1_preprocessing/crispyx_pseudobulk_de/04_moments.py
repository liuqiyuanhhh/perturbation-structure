"""Step 04: per-perturbation moments (effect, mean, s.d., standard errors).

    data/sc/sc_<dataset>.h5ad -> data/pseudobulk/effect_<dataset>_moments.h5ad

One streaming pass over the log-normalized cells (util_moments.compute_moments).
Within each batch (datasets.strat_col) the perturbed and control means are
differenced and combined with harmonic weights n_pert*n_ctrl/(n_pert+n_ctrl);
standard errors use a variance pooled within batches (see util_moments.py for the
estimators and the output schema). For the Feng screens, perturbations without a
control cell in any joint batch x donor stratum fall back to the plain batch.

    python 04_moments.py --dataset Replogle-E-k562
    python 04_moments.py --dataset all
    python 04_moments.py --dataset Huang-HEK293T --memory-limit-gb 256 --resume
"""
from __future__ import annotations

import argparse
import logging
import sys

import datasets
from paths import MOMENTS_DIR, SC_DIR, TMP_DIR, moments_file, sc_file
from util_de import check_crispyx, set_scratch_dir
from util_moments import compute_moments

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

RAW_CACHE_DIR = TMP_DIR / "moments_raw"  # checkpoint for --resume (removed on success)


def process_dataset(name: str, args: argparse.Namespace) -> None:
    cfg = datasets.get(name)
    sc_path = SC_DIR / sc_file(name)
    out_path = MOMENTS_DIR / moments_file(name)
    if not sc_path.exists():
        raise FileNotFoundError(f"{name}: {sc_path} not found (run 03_sc_preprocess.py)")
    if out_path.exists() and not args.force:
        log.info("SKIP %s: %s exists (pass --force to rebuild)", name, out_path.name)
        return
    out = compute_moments(
        sc_path, name=name,
        perturbation_column=cfg["pert_col"], control_label=cfg["control"],
        batch_column=datasets.strat_col(name),
        fallback_batch_column=datasets.fallback_strat_col(name),
        chunk_size=args.chunk_size, memory_limit_gb=args.memory_limit_gb,
        resume=args.resume, checkpoint_interval=args.checkpoint_interval,
        raw_cache_dir=RAW_CACHE_DIR, output_path=out_path, force=args.force, verbose=1,
    )
    log.info("%s: %d perturbations x %d genes -> %s", name, out.n_obs, out.n_vars, out_path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="dataset name, or 'all'")
    ap.add_argument("--force", action="store_true", help="rebuild existing outputs")
    ap.add_argument("--chunk-size", type=int, default=None, help="gene chunk size (automatic if unset)")
    ap.add_argument("--memory-limit-gb", type=float, default=None, help="memory budget for automatic chunking")
    ap.add_argument("--resume", action="store_true", help="resume an interrupted pass from its checkpoint")
    ap.add_argument("--checkpoint-interval", type=int, default=None, help="gene chunks between checkpoints")
    args = ap.parse_args(argv)
    check_crispyx()
    set_scratch_dir()
    MOMENTS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    failed = []
    for name in datasets.resolve(args.dataset):
        try:
            process_dataset(name, args)
        except Exception:
            log.exception("FAILED: %s", name)
            failed.append(name)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
