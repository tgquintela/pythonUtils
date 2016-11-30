
"""
auxiliar_functions
------------------
Auxiliar functions for perturbations.

"""

import numpy as np


def check_int(i):
    "Check integer unit."
    if type(i) in [np.ndarray, list]:
        return False
    elif i is None:
        return False
    else:
        return lambda i: i - int(i) <= np.finfo('f').eps
