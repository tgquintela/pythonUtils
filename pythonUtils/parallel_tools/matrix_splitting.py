
"""
Matrix_splitting
----------------
Collection of functions to split the matrix in order to scale the computation
tasks properly in proportion with our computational resources.

"""

import numpy as np

#import multiprocessing
#from multiprocessing import Queue
#import time


def distribute_tasks(n, memlim):
    """Util function for distribute tasks in matrix computation in order to
    save memory or to parallelize computations.

    Parameters
    ----------
    n: int
        the number of rows.
    memlim: int
        the limit of rows we are able to compute at the same time.

    Returns
    -------
    lims: list
        the list of limits for each interval.

    """
    lims = []
    inflim = 0
    while True:
        if inflim + memlim >= n:
            lims.append([inflim, n])
            break
        else:
            lims.append([inflim, inflim+memlim])
            inflim = inflim+memlim
    return lims


def reshape_limits(lims, limits):
    """Reshape the limits

    Parameters
    ----------
    lims: list
        the list of limits for each interval.
    limits: list
        the list of the global limits.

    Returns
    -------
    new_limits: list
        the new limits.

    """
    new_limits = []
    xs = np.linspace(limits[0], limits[1], lims[-1][1]+1)
    for i in range(len(lims)):
        new_limits.append([xs[lims[i][0]], xs[lims[i][1]]])
    return new_limits


def generate_split_array(limits, lims):
    """Generate a splitted array.

    Parameters
    ----------
    limits: list
        the list of the global limits.
    lims: list
        the list of limits for each interval.

    Returns
    -------
    split_array: list
        the generated indices array for all the intervals.

    """
    init, endit = limits
    lims = np.array(lims) + (init - lims[0][0])
    idx_endit = np.where(np.array(lims) <= endit)[0][-1]
    split_array = []
    for i in range(idx_endit+1):
        if i == idx_endit:
            split_array.append(np.arange(lims[i][0], endit))
        else:
            split_array.append(np.arange(lims[i][0], lims[i][1]))
    return split_array


def split_parallel(indices, maxbunch):
    """The split parallel with a given indices and maxbunch.

    Parameters
    ----------
    indices: len or np.ndarray
        the indices of all the elements.
    maxbunch: int
        the limit of rows we are able to compute at the same time.

    Returns
    -------
    new_indices: list
        the generated indices array for all the intervals.

    """
    n_total = len(indices)
    if n_total <= maxbunch:
        return [indices]
    else:
        lims = distribute_tasks(n_total, maxbunch)
        new_indices = []
        for i in range(len(lims)):
            new_indices.append(indices[lims[i][0]:lims[i][1]])
        return new_indices


#import threading
#import os
#import queue
#import redis
#import rq
#
#
#def parallel_process(f, args_l):
#    return_values = Queue()
#    jobs = []
#    for i in range(len(args_l)):
#        p = multiprocessing.Process(target=some_long_process,
#                                    args=(i, i**2, return_values))
#        jobs.append(p)
#        p.start()
#
#return_values = Queue()
#some_long_process = lambda x, y, q: q.put(x**y)
#
#for i in range(8):
#    p = multiprocessing.Process(target=some_long_process,
#                                args=(i, i**2, return_values))
#    jobs.append(p)
#    p.start()
#time.sleep(0.5)
#for i in range(8):
#    print(return_values.get())
#
