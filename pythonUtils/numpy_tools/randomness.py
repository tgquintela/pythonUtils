
"""
Randomness
----------
The module which groups all the utils which support random tasks.

"""

import numpy as np


def wheeler_assignation(prop, r):
    """Assign the vote to the box who contains r.

    Parameters
    ----------
    prop: numpy.ndarray
        the vector of normalized probabilities for each option.
    r: float [0, 1)
        a random number between 0 and 1.

    Returns
    -------
    i: int
        the interval selected by the wheeler assignation process.

    """

    prop2 = np.cumsum(prop)
    for i in range(prop.shape[0]):
        if prop2[i] >= r:
            break
    return i
