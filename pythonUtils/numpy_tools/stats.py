
"""
Stats
-----
Tools using for making stats.

"""

votation = counting(feat_arr[neighs], weights, n_vals)

def counting(array, weights, n_vals):
    """Counting function.

    Parameters
    ----------
    array: numpy.ndarray of ints
        the categorical variable values.
    weighs: numpy.ndarray
        the 
    n_vals: int
        the number of values in array.

    Returns
    -------
    votation: np.ndarray, shape (n_vals,)
        the values of votation.

    """
    votation = np.zeros(n_vals)
    for i in range(n_vals):
        votation[i] = np.sum(array == i)
    return votation

