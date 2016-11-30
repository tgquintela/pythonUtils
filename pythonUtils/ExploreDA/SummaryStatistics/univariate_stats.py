
"""
Univariate statisctics
----------------------
The univariate statistics exploration. It study variables 1 by 1 in order
to obtain some insighs in the data.

"""

from ..Plotting import general_plot
from ..Statistics.cont_statistics import quantile_compute, ranges_compute,\
    cont_count, log_cont_count
from ..Statistics.cat_statistics import cat_count
from ..Statistics.temp_statistics import count_temp_stats


def compute_univariate_stats(df, info_var):
    """Compute univariate statistics summary.

    Parameters
    ----------
    df: pd.DataFrame
        the data in dataframe form.
    info_var: dict
        the plot variables information.

    Returns
    -------
    summary: dict
        the summary statistics dictionary database.

    """
    typevar = info_var['type'].lower()
    if typevar in ['discrete', 'categorical']:
        summary = compute_cat_describe(df, info_var)
    elif typevar == 'continuous':
        summary = compute_cont_describe(df, info_var)
    elif typevar == 'coordinates':
        summary = compute_coord_describe(df, info_var)
    elif typevar in ['time', 'temporal']:
        summary = compute_temp_describe(df, info_var)
    elif typevar == 'tmpdist':
        summary = compute_tmpdist_describe(df, info_var)
    else:
        print typevar, info_var['variables']
    return summary


def compute_cont_describe(df, info_var):
    """Function created to aggregate all the exploratory information of the
    continuous variable studied.

    Parameters
    ----------
    df: pd.DataFrame
        the data in dataframe form.
    info_var: dict
        the plot variables information.

    Returns
    -------
    summary: dict
        the summary statistics dictionary database.

    """
    ## 0. Needed variables
    if info_var['variables'] == list:
        variable = info_var['variables'][0]
    else:
        variable = info_var['variables']

    ## 1. Summary
    summary = info_var
    summary['n_bins'] = 10 if not 'n_bins' in info_var else info_var['n_bins']
    summary['mean'] = df[variable].mean()
    summary['quantiles'] = quantile_compute(df[variable],
                                            summary['n_bins'])
    summary['ranges'] = ranges_compute(df[variable],
                                       summary['n_bins'])
    summary['hist_table'] = cont_count(df, variable,
                                       summary['n_bins'])
    if info_var['logscale'] in [True, 'True', 'TRUE']:
        summary['log_hist_table'] = log_cont_count(df, variable,
                                                   summary['n_bins'])

    if info_var['ifplot'] in [True, 'True', 'TRUE']:
        summary['plots'] = general_plot(df, info_var)

    return summary


def compute_cat_describe(df, info_var):
    """Function created to aggregate all the exploratory information of the
    categorical variable studied.

    Parameters
    ----------
    df: pd.DataFrame
        the data in dataframe form.
    info_var: dict
        the plot variables information.

    Returns
    -------
    summary: dict
        the summary statistics dictionary database.

    """
    ## 0. Needed variables
    if info_var['variables'] == list:
        variable = info_var['variables'][0]
    else:
        variable = info_var['variables']

    ## 1. Summary
    summary = info_var
    summary['count_table'] = cat_count(df, variable)
    # Plot
    if info_var['ifplot'] in [True, 'True', 'TRUE']:
        summary['plots'] = general_plot(df, info_var)

    return summary


def compute_coord_describe(df, info_var):
    """Compute the summary statistics of spatial coordinates variables.

    Parameters
    ----------
    df: pd.DataFrame
        the data in dataframe form.
    info_var: dict
        the plot variables information.

    Returns
    -------
    summary: dict
        the summary statistics dictionary database.

    """
    summary = info_var
    # Plot
    if info_var['ifplot'] in [True, 'True', 'TRUE']:
        summary['plots'] = general_plot(df, info_var)
    return summary


def compute_tmpdist_describe(df, info_var):
    """Compute temporal distribution of the variables summary.

    Parameters
    ----------
    df: pd.DataFrame
        the data in dataframe form.
    info_var: dict
        the plot variables information.

    Returns
    -------
    summary: dict
        the summary statistics dictionary database.

    """
    summary = info_var
    # Plot
    if info_var['ifplot'] in [True, 'True', 'TRUE']:
        summary['plots'] = general_plot(df, info_var)
    return summary


def compute_temp_describe(df, info_var):
    """Compute temporal description.

    Parameters
    ----------
    df: pd.DataFrame
        the data in dataframe form.
    info_var: dict
        the plot variables information.

    Returns
    -------
    summary: dict
        the summary statistics dictionary database.

    """
    ## 0. Needed variables
    if info_var['variables'] == list:
        variable = info_var['variables'][0]
    else:
        variable = info_var['variables']

    ## 1. Summary computation
    summary = info_var
    summary['pre_post'] = count_temp_stats(df[variable],
                                           **info_var['stats_pars'])
    # Plot
    if info_var['ifplot'] in [True, 'True', 'TRUE']:
        summary['plots'] = general_plot(df, info_var)
    return summary
