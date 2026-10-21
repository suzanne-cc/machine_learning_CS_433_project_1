import numpy as np
import matplotlib.pyplot as plt
from helpers import *
from implementations import *

def fit_preprocessing(x):
    """Learn the preprocessing numbers and store them as paramters.

    Drops columns with 60% or more missing values, then computes the
    median (for filling NaNs) and the mean and standard deviation
    (for standardizing) of each remaining column.

    Args:
        x = numpy array of shape (N, D_raw). Raw training features, may contain NaNs

    Returns:
        params = dict with
            "keep" = boolean array, shape (D_raw, ), True for columns kept
            "median" = float array of shape (D_kept,), median of each kept column.
            "mean" = float array of shape (D_kept,), mean of each kept column after filling NaNs.
            "std" = float array of shape (D_kept,), standard deviation of each kept column after filling NaNs (0 replaced by 1).
    """

    fraction_missing = np.isnan(x).mean(axis=0)
    keep = fraction_missing < 0.6 # we drop any columns with more than 60% missing values

    x = x[:, keep]
    median = np.nanmedian(x, axis=0)
    x = np.where(np.isnan(x), median, x)

    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1

    params = {"keep": keep, "median": median, "mean": mean, "std": std}

    return params

def clean_preprocessing(x, params):
    """Clean the data using the parameters found above.

    Keeps the selected columns, fills NaNs with the training medians,
    standardizes with the training mean and std, and adds a bias column.

    Args:
        x = numpy array of shape (N, D_raw). Raw features (training, validation or test), may contain NaNs.
        params = returned by fit_preprocessing.

    Returns:
        tx = numpy array of shape (N, D_kept + 1). Cleaned feature matrix, with a column of ones as the first column.
    """

    x = x[:, params["keep"]]
    x = np.where(np.isnan(x), params["median"], x)
    x = (x - params["mean"]) / params["std"]

    tx = np.c_[np.ones(x.shape[0]), x]

    return tx

# When you add new steps later, like code cleaning or one-hot encoding, 
# update both docstrings and the list of keys in params at the same time, so they never fall out of sync.