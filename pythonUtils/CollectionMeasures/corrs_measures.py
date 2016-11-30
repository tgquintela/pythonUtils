
"""
correlation measures
--------------------
Collection of measures from correlation information.

"""

import numpy as np


def measure_corrs_mean(corrs, params=None):
    """Measure based on computing the mean of the non-auto correlation values.

    Parameters
    ----------
    corrs: np.ndarray
        the correlation matrix.
    params: optional (default=None)
        the parameters needed to compute the output measure.

    Returns
    -------
    measure: float
        the computed measure from corrs.

    """
    corrs = corrs/corrs[0, 0]
    corrs = corrs - np.eye(corrs.shape[0])
    return corrs.mean()


def measure_corrs_std(corrs, params=None):
    """Measure based on computing the std of the non-auto correlation values.

    Parameters
    ----------
    corrs: np.ndarray
        the correlation matrix.
    params: optional (default=None)
        the parameters needed to compute the output measure.

    Returns
    -------
    measure: float
        the computed measure from corrs.

    """
    corrs = corrs/corrs[0, 0]
    corrs = corrs - np.eye(corrs.shape[0])
    return corrs.std()


def measure_corrs_sumstd(corrs, params=None):
    """Measure based on computing the std of the sum of all the non-auto
    correlation values per each element.

    Parameters
    ----------
    corrs: np.ndarray
        the correlation matrix.
    params: optional (default=None)
        the parameters needed to compute the output measure.

    Returns
    -------
    measure: float
        the computed measure from corrs.

    """
    corrs = corrs/corrs[0, 0]
    corrs = corrs - np.eye(corrs.shape[0])
    return np.sum(corrs, axis=1).std()
