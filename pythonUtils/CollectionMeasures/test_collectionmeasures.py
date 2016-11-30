
"""
Test_collection_measures
------------------------
Test the module functions.

"""

import numpy as np
from corrs_measures import measure_corrs_sumstd, measure_corrs_std,\
    measure_corrs_mean


def test():
    ### Measures testing
    corrs = np.random.random((10, 10))
    val = measure_corrs_sumstd(corrs)
    val = measure_corrs_std(corrs)
    val = measure_corrs_mean(corrs)
