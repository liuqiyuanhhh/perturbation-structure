"""Step 02: build the standardized raw-count file data/origin/<dataset>.h5ad.

    python 02_setup.py --dataset Replogle-E-k562
    python 02_setup.py --dataset all --force

Runs setup/<handle>.py; builders covering several datasets run once.
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
    ap.add_argument("--force", action="store_true", help="overwrite existing origin files")
    args = ap.parse_args(argv)

    names = datasets.resolve(args.dataset)
    jobs: dict[str, list[str]] = {}
    for n in names:
        h = datasets.get(n)["setup"]
        if h is None:
            print(f"[setup] {n}: derived dataset, built by ../vcc_subsampling/downsample_vcc.py", flush=True)
            continue
        # passthrough promotes one file per call; the other builders emit all of theirs
        extra = ["--dataset", n] if h == "passthrough" else []
        jobs.setdefault(h if not extra else f"{h}:{n}", [h, *extra])
    rc = 0
    for key, (h, *extra) in jobs.items():
        cmd = [sys.executable, str(HERE / "setup" / f"{h}.py"), *extra, *(["--force"] if args.force else [])]
        print(f"[setup] {key}", flush=True)
        r = subprocess.run(cmd).returncode
        if r:
            print(f"[setup] {key} failed (exit {r})", file=sys.stderr)
            rc = r
    return rc


if __name__ == "__main__":
    sys.exit(main())
