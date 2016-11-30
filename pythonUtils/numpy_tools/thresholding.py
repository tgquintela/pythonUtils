
"""
Thresholding
------------
Module for threshold the numpy arrays.

"""

import numpy as np


def discretize_with_thresholds(array, thres, values=[]):
    """This function discretize the array to the values given.

    Parameters
    ----------
    array : array_like, shape (N,)
        the values of the signal.
    thres : float, list of floats
        thresholds values to discretize the array.
    values: list
        the values to discretize the array.

    Returns
    -------
    aux : array_like, shape (N,)
        a array with a discretized values.

    """

    ## 1. Preparing thresholds and values
    if type(thres) == float:
        thres = [thres]
    if values == []:
        values = range(len(thres)+1)
    else:
        assert len(thres) == len(values)-1
    mini = np.array([0])*(array.max()-array.min()) + array.min()
    maxi = np.array([1])*array.max()
    thres = np.hstack([mini, thres, maxi])

    ## 2. Fill the new vector discretized signal to the given values
    aux = np.zeros(array.shape)
    for i in range(len(values)):
        indices = np.logical_and(array >= thres[i], array <= thres[i+1])
        indices = np.nonzero(indices)[0].astype(int)
        aux[indices] = values[i]

    return aux
