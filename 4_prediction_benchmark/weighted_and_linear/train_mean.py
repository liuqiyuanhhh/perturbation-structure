"""Training mean: the mean effect of the fold's training perturbations, for every test one.

The zero line of Fig 1d and S2.  Used by evaluate.py.
"""

import numpy as np
import pandas as pd


def train_mean_prediction(truth, train_pert, test_pert):
    """DataFrame[test_pert x gene]: the training perturbations' mean effect in every row."""
    train_mean = truth.loc[train_pert].mean(axis=0)
    return pd.DataFrame(np.tile(train_mean.values, (len(test_pert), 1)),
                        index=test_pert, columns=train_mean.index)
