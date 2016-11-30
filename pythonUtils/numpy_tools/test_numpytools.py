
"""
test_numpy_tools
----------------
Test function for the numpy tools.

"""

import numpy as np

from randomness import wheeler_assignation
from shaping import shaping_dim, ensure_2dim
from sorting import get_kbest
from stats import weighted_counting
from thresholding import discretize_with_thresholds


def test():
    ## Randomness utils
    prop = np.random.random(10)
    prop = prop/prop.sum()
    wheeler_assignation(prop, np.random.random())

    ## Shaping
    shaping_dim(np.random.random(100), 1)
    shaping_dim(np.random.random(100), 2)
    shaping_dim(np.random.random((100, 2)), 1)
    shaping_dim(np.random.random((100, 2)), 2)
    shaping_dim(np.random.random((100, 2)), 3)

    ensure_2dim(np.random.random(100), axis=0, sh_known=(None, None))
    ensure_2dim(np.random.random(100), axis=1, sh_known=(None, None))
    ensure_2dim(np.random.random(100), axis=0, sh_known=(10, 10))
    try:
        boolean = False
        ensure_2dim(np.random.random(100), axis=0, sh_known=(10, 100))
        boolean = True
    except:
        if boolean:
            raise Exception("It should be raised an error.")
    try:
        boolean = False
        ensure_2dim(np.random.random(100), axis=0, sh_known=(None, 100))
        boolean = True
    except:
        if boolean:
            raise Exception("It should be raised an error.")

    ## Sorting
    values = np.random.random(100)
    kbest = 5
    get_kbest(values, kbest, ifpos=True)
    get_kbest(values, kbest, ifpos=False)

    ## Stats
    weighted_counting(np.random.randint(0, 10, 100), np.random.random(10), 10)

    ## Thresholding
    array = np.random.random(100)
    thres = 0.4
    thres_list = [0.2, 0.5, 0.7]

    discretize_with_thresholds(array, thres, values=[])
    discretize_with_thresholds(array, thres, values=[1, 2])
    discretize_with_thresholds(array, thres_list, values=[])
    discretize_with_thresholds(array, thres_list, values=[1, 2, 3, 4])
