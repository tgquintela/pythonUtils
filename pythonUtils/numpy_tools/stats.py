
"""
Stats
-----
Tools using for making stats.

"""

import numpy as np


def weighted_counting(array, weights, n_vals):
    """Counting function.

    Parameters
    ----------
    array: numpy.ndarray of ints
        the categorical variable values.
    weights: numpy.ndarray, shape (n_vals,)
        the weights of each voting depending on the type.
    n_vals: int
        the number of values in array.

    Returns
    -------
    votation: np.ndarray, shape (n_vals,)
        the values of votation.

    """
    votation = np.zeros(n_vals)
    vals = np.unique(array)
    for i in range(n_vals):
        votation[i] = weights[i]*np.sum(array == vals[i])
    return votation
