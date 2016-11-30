
"""
Utils
-----
Statistics utils.

"""


def clean_dict_stats(stats):
    """Cleaner dict stats information. That function removes the plots stored
    in the stats dictionary data base.

    Parameters
    ----------
    stats: dict
        the stats dictionary database.

    Returns
    -------
    stats: dict
        the stats dictionary database without plots.

    """
    for i in range(len(stats)):
        if 'plots' in stats[i].keys():
            del stats[i]['plots']

    return stats
