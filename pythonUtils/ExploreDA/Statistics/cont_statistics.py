
"""
Continious variable statistics
------------------------------
Module which groups all the functions needed to compute the statistics and the
description of the categorical variables.

"""

import numpy as np
import pandas as pd


## Creation of the table info
def quantile_compute(x, n_bins):
    """Quantile computation.

    Parameters
    ----------
    x: pd.DataFrame
        the data variable we want to obtain its distribution.
    n_bins: int
        the number of bins we want to use to plot the distribution.

    Returns
    -------
    quantiles: np.ndarray
        the quantiles.

    """
    # aux.quantile(np.linspace(0, 1, 11)) # version = 0.15
    quantiles = [x.quantile(q) for q in np.linspace(0, 1, n_bins+1)]
    quantiles = np.array(quantiles)
    return quantiles


def ranges_compute(x, n_bins):
    """Computation of the ranges (borders of the bins).

    Parameters
    ----------
    x: pd.DataFrame
        the data variable we want to obtain its distribution.
    n_bins: int
        the number of bins we want to use to plot the distribution.

    Returns
    -------
    ranges: np.ndarray
        the borders of the bins.

    """
    mini = np.nanmin(np.array(x))
    maxi = np.nanmax(np.array(x))
    ranges = np.linspace(mini, maxi, n_bins+1)
    return ranges


## Continious hist
def cont_count(df, variable, n_bins):
    """Count of continious variable.

    Parameters
    ----------
    df: pd.DataFrame
        the data in dataframe form.
    variable: str
        the variable of the database we want to study.
    n_bins: int
        the number of bins we want to use to plot the distribution.

    Returns
    -------
    counts: pd.DataFrame
        the counts of the bins.
    bins: np.ndarray
        the bins information.

    """
    mini = np.nanmin(np.array(df[variable]))
    maxi = np.nanmax(np.array(df[variable]))
    bins = np.linspace(mini, maxi, n_bins+1)
    labels = [str(i) for i in range(int(n_bins))]
    categories = pd.cut(df[variable], bins, labels=labels)
    categories = pd.Series(np.array(categories)).replace(np.nan, 'NaN')
    counts = categories.value_counts()
    return counts, bins


def log_cont_count(df, variable, n_bins):
    """Logarithmic count of countinious variable.

    Parameters
    ----------
    df: pd.DataFrame
        the data in dataframe form.
    variable: str
        the variable of the database we want to study.
    n_bins: int
        the number of bins we want to use to plot the distribution.

    Returns
    -------
    counts: pd.DataFrame
        the counts of the bins.
    bins: np.ndarray
        the bins information.

    """
    mini = np.nanmin(np.array(df[variable]))
    mini = .001 if mini <= 0 else mini
    maxi = np.nanmax(np.array(df[variable]))
    bins = np.linspace(np.log10(mini), np.log10(maxi), n_bins+1)
    bins = np.power(10, bins)
    bins[0] = np.nanmin(np.array(df[variable]))
    labels = [str(i) for i in range(int(n_bins))]
    categories = pd.cut(df[variable], bins, labels=labels)
    categories = pd.Series(np.array(categories)).replace(np.nan, 'NaN')
    counts = categories.value_counts()
    return counts, bins
