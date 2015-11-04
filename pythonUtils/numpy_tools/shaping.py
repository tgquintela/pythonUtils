
"""
shaping
-------
Module which groups the functions for shaping properly the arrays.
"""

import numpy as np


def shaping_dim(array, ndim):
    "The dimensions of the array have to be least than ndim."

    if array.ndim >= ndim:
        return array
    dim_extend = ndim - array.ndim
    newshape = tuple(list(array.shape) + [1 for i in range(dim_extend)])
    array = array.reshape(newshape)
    return array
