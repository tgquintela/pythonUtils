
"""
Sorting
-------
Tools using in sorting and retrieving from numpy arrays.

"""

import numpy as np


def get_kbest(values, kbest, ifpos=True):
    """Get the kbest values.

    Parameters
    ----------
    values: list of np.ndarray
        the values retrievable.
    kbest: int
        the number of kbest values we want to retrieve.
    ifpos: boolean (default=True)
        if we want the bigger ones (True) or the smaller ones (False)

    Returns
    -------
    idxs: np.ndarray
        the list of indices of the retrieved values
    values_idxs: np.ndarray
        the list of values of the kbest values.

    """
    if ifpos:
        idxs = np.argsort(values)[-kbest:][::-1]
    else:
        idxs = np.argsort(values)[:kbest]
    return idxs, values[idxs]
