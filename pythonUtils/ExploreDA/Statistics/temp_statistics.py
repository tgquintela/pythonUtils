
"""
Temporal variable statistics
----------------------------
Module which groups all the functions related with the computing of statistics
of temporal variables.

TODO
----
Not predefined date range.

"""

import numpy as np
import datetime


def count_temp_stats(tmps, date_ranges, tags=None):
    """Function used to compute stats of a temporal var.

    Parameters
    ----------
    tmps: np.ndarray or pd.Series
        the times in which each event happens.
    date_ranges: list or array_like
        the split times which define the bins.
    tags: list
        the tags for each bin.

    Returns
    -------
    counts: dict
        the counts for each bin.

    """

    ## 0. Variable needed
    mini = tmps.min()
    maxi = tmps.max()
    for i in range(len(date_ranges)):
        if type(date_ranges[i]) == str:
            aux = datetime.datetime.strptime(date_ranges[i], '%Y-%m-%d')
            date_ranges[i] = aux
    if mini < date_ranges[0]:
        date_ranges = [mini]+list(date_ranges)
    if maxi > date_ranges[-1]:
        date_ranges = list(date_ranges)+[maxi]
    if tags is None:
        tags = [str(e) for e in range(len(date_ranges)-1)]
    n_rang = len(tags)

    ## 1. Counting
    counts = {}
    for i in range(n_rang):
        counts[tags[i]] = np.logical_and(tmps >= date_ranges[i],
                                         tmps <= date_ranges[i+1]).sum()

    return counts
