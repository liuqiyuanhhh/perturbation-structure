#!/usr/bin/env python3
"""Pan-essentiality score: per gene, the 10th percentile over DepMap cell lines of its
essentiality rank in [0, 1] (1 = most essential; lines that did not score it left out).
Reads DEPMAP_GENE_EFFECT; writes PAN_ESSENTIALITY (Fig 1b right)."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils.paths import DEPMAP_GENE_EFFECT, PAN_ESSENTIALITY


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gene-effect", type=Path, default=DEPMAP_GENE_EFFECT)
    parser.add_argument("--output", type=Path, default=PAN_ESSENTIALITY)
    return parser.parse_args()


def main():
    args = parse_args()
    effect = pd.read_csv(args.gene_effect, index_col=0)
    values = effect.to_numpy(np.float64)
    missing = np.isnan(values)
    n_genes = values.shape[1]

    # position 0 = most negative Chronos in the line; unscored genes sort last
    position = np.argsort(np.argsort(np.where(missing, np.inf, values), axis=1), axis=1)
    rank = (n_genes - 1 - position) / (n_genes - 1)  # 1 = most essential in that line
    rank[missing] = np.nan

    scores = pd.DataFrame({
        "gene_symbol": [column.split(" (")[0] for column in effect.columns],
        "gene_col": effect.columns,
        "pan_essentiality_score": np.nanpercentile(rank, 10, axis=0),
        "n_lines_screened": (~missing).sum(axis=0),
    }).sort_values("pan_essentiality_score", ascending=False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output, index=False)
    print(f"{len(scores)} genes, {values.shape[0]} cell lines")


if __name__ == "__main__":
    main()
