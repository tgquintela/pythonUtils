
"""
test_parallel_tools
-------------------
Testing function for the functions of the module parallel_tools.

"""

import numpy as np
from matrix_splitting import distribute_tasks, reshape_limits,\
    generate_split_array, split_parallel


def test():
    ## Parameters
    n, memlim, limits = 1250, 300, (250, 1500)
    lims = distribute_tasks(n, memlim)
    reshape_limits(lims, limits)
    generate_split_array(limits, lims)
    split_parallel(np.arange(n), memlim)

#	 n, memlim, limits = 100, 10, [0, 20]
#    distribute_tasks(n, 1000)
#    lims = distribute_tasks(n, 10)
#    reshape_limits(lims, limits)
#    generate_split_array(limits, lims)
#    split_parallel(np.arange(1000), 100)
