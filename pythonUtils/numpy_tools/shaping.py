
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


def ensure_2dim(array, axis=0, sh_known=(None, None)):
    "TODO: move to python utils"
    ## Ensure 2 dimensions in the array
    sh = array.shape
    if len(sh) == 1:
        new_sh = (sh[0], 1) if axis == 0 else (1, sh[0])
    elif len(sh) == 2:
        new_sh = sh
    ## Ensure known shape
    notnone_ids = [i for i in range(len(sh_known)) if sh_known[i] is not None]
    if len(notnone_ids) == 0:
        pass
    if len(notnone_ids) == 1:
        if new_sh[notnone_ids] == sh_known[notnone_ids]:
            pass
        elif new_sh[::-1][notnone_ids] == sh_known[notnone_ids]:
            new_sh = new_sh[::-1]
        else:
            raise Exception("Impossible to fit the conditions.")
    elif len(notnone_ids) == 2:
        if new_sh == sh_known:
            pass
        elif new_sh[::-1] == sh_known:
            new_sh = new_sh[::-1]
        else:
            raise Exception("Impossible to fit the conditions.")
    ## Transform array
    array = array.reshape(new_sh)
    return array
