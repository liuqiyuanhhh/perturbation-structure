"""Step 01: download the raw files of a dataset (-> data/raw/).

    python 01_download.py --dataset Replogle-E-k562
    python 01_download.py --dataset all

Runs downloaders/<handle>.sh; datasets sharing a source are fetched once.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import datasets

HERE = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="dataset name, or 'all'")
    args = ap.parse_args(argv)

    names = datasets.resolve(args.dataset)
    handles = list(dict.fromkeys(datasets.get(n)["download"] for n in names if datasets.get(n)["download"]))
    rc = 0
    for h in handles:
        print(f"[download] {h}", flush=True)
        r = subprocess.run(["bash", str(HERE / "downloaders" / f"{h}.sh")]).returncode
        if r:
            print(f"[download] {h} failed (exit {r})", file=sys.stderr)
            rc = r
    return rc


if __name__ == "__main__":
    sys.exit(main())
