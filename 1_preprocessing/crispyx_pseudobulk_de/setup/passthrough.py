"""Promote source files that already have the expected schema to data/origin/.

Covers Replogle-GW-k562, Replogle-E-k562, Replogle-E-rpe1 (scPerturb releases)
and Huang-HCT116, Huang-HEK293T (X-Atlas/Orion). The file is checked for the
perturbation column and control label, then moved unchanged.

Usage:
    python 02_setup.py --dataset Replogle-E-k562 --force
    python setup/passthrough.py --dataset all --force
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from util_helpers import promote_verbatim

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import datasets  # noqa: E402
from paths import RAW_DIR, ORIGIN_DIR  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

RAW_PATH = {
    "Replogle-GW-k562": RAW_DIR / "Replogle-GW-k562.h5ad",
    "Replogle-E-k562": RAW_DIR / "Replogle-E-k562.h5ad",
    "Replogle-E-rpe1": RAW_DIR / "Replogle-E-rpe1.h5ad",
    "Huang-HCT116": RAW_DIR / "huang_xatlas" / "Huang-HCT116.h5ad",
    "Huang-HEK293T": RAW_DIR / "huang_xatlas" / "Huang-HEK293T.h5ad",
}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=[*RAW_PATH, "all"], default="all")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    for name in RAW_PATH if args.dataset == "all" else [args.dataset]:
        raw_path, origin_path = RAW_PATH[name], ORIGIN_DIR / f"{name}.h5ad"
        if origin_path.exists() and not args.force:
            log.info("SKIP %s: origin exists (pass --force to overwrite)", name)
            continue
        if not raw_path.exists():
            log.warning("SKIP %s: raw input not found: %s", name, raw_path)
            continue
        cfg = datasets.get(name)
        promote_verbatim(name, raw_path, origin_path, pert_col=cfg["pert_col"],
                         control_label=cfg["control"], force=args.force)
        log.info("Done: %s", origin_path)


if __name__ == "__main__":
    main()
