
"""
Sorting
-------
Tools using in sorting and retrieving from numpy arrays.

"""

import numpy as np


def get_kbest(values, kbest, ifpos=True):
    if ifpos:
        idxs = np.argsort(Q)[-kbest:][::-1]
    else:
        idxs = np.argsort(Q)[:kbest]
    return idxs, values[idxs]

