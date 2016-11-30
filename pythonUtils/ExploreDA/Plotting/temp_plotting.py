
"""
Temporal plotting
-----------------
Temporal plots to understand the temporal structure of the data.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import datetime
from dateutil.relativedelta import relativedelta


def temp_distrib(tmps, agg_time, logscale=False):
    """The temporal distribution plot.

    Parameters
    ----------
    tmps: pandas.Series
        The datetimes to consider.
    agg_time: str, optional
        The time of which make the aggregation. The options are:
        ['year', 'month','week','day', 'hour', 'minute', 'second',
        'microsecond', 'nanosecond']
    logscale: boolean (default=False)
        if the y axis is defined in logarithmic scale.

    Returns
    -------
    fig: matplotlib.pyplot.figure
        the figure of the temporal aggregation.

    """
    ## 0. Time aggregation
    tmps = tmps.dropna()
    # Vector of times range
    mini = tmps.min()
    maxi = tmps.max()
    delta = relativedelta(**{agg_time+'s': 1})
    ranges = range_time(mini, maxi+delta, agg_time)
    ranges = [strformat(t, agg_time) for t in ranges]
    nt = len(ranges)
    # Format as strings
    tmps = tmps.apply(lambda x: strformat(x, agg_time))
    ## 1. Counts
    counts = tmps.value_counts()
    count_values = []
    for t in ranges:
        if t in list(counts.index):
            count_values.append(counts[t])
        else:
            count_values.append(0)

    ## 2. Plot figure
    fig = plt.figure()
    ax = plt.subplot()
    ax.set_xlim([0, len(ranges)-1])
    ax.bar(range(len(ranges)), count_values, align='center', color='black',
           width=1)
    idxs = ax.get_xticks().astype(int)
    if len(idxs) < 100:
        xticks = [ranges[idxs[i]] for i in range(len(idxs)) if idxs[i] < nt]
        ax.set_xticklabels(xticks)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=310)
    if logscale:
        ax.set_yscale('log')

    return fig


def distrib_across_temp(df, variables, times=None, ylabel="", logscale=False,
                        year0=2006, delta_y=1):
    """Function to plot the value distribution across times.

    Parameters
    ----------
    df: pd.DataFrame
        the data we are studying.
    variables: str or list
        the variables we want to use from data for study the time for each
        time.
    times: str or list (default=None)
        the information of time numerical tags for each variable.
    ylabel: str (default="")
        the label of the y axis.
    logscale: boolean (default=False)
        if the axis is in form of logarithmic scale.
    year0: int or float
        the initial numeric tag for the time of the first variable. Only used
        in case of `times` is None.
    delta_y: int or float
        the time numerical tag increment for each variable.

    Returns
    -------
    fig: matplotlib.pyplot.figure
        the figure of the temporal aggregation.

    """

    ## 0. Preparing times
    if times is None:
        times = np.arange(year0, year0+delta_y*len(variables))
    times = [str(t) for t in times]

    mini = df[variables].min().min()
    maxi = df[variables].max().max()
    delta = (maxi-mini)/25.

    ## 1. Plotting
    fig = plt.figure()
    ax = plt.subplot()
    df[variables].boxplot()
    # Making up the plot
    plt.xlabel('Years')
    if logscale:
        ax.set_yscale('log')
    ax.set_ylim([mini-delta, maxi+delta])
    ax.grid(True)
    ax.set_xticklabels(times)
    plt.ylabel(ylabel)
    plt.title("Distribution by years")

    return fig


def temp_distrib_agg(tmps, agg_time, total_agg=False):
    """

    Parameters
    ----------
    tmps: pandas.Series
        The datetimes to consider.
    agg_time: str, optional
        The time of which make the aggregation. The options are:
        ['year', 'month','week','day', 'hour', 'minute', 'second',
        'microsecond', 'nanosecond']
    total_agg: boolean
        If true, only consider the level of aggregation selected in agg_time,
        else it is considered all the superiors levels up to the agg_time.
    """
    pass


###############################################################################
############################# AUXILIARY FUNCTIONS #############################
###############################################################################
def range_time(time0, time1, agg_time, increment=1):
    """Produce all the possible times at that level of aggregation between the
    selected initial and end time.

    Parameters
    ----------
    time0: float or datetime
        the initial time to generate the range.
    time1: float or datetime
        the final time to generate the range.
    agg_time: str optional
        the aggregation level of time we want to produce range.
    increment: (default=1)
        the integer increment.

    Returns
    -------
    range_times: list
        the range of times generated between `time0` and `time1` using the
        increment determined in the input.

    """
    curr = time0
    delta = relativedelta(**{agg_time+'s': increment})
    range_times = []
    while curr <= time1:
        range_times.append(curr)
        curr += delta
    return range_times


def strformat(time, agg_time, format_mode='name'):
    """Transform a datetime to a string with a level of aggregation desired.
    Now only format to a day aggregation.

    Parameters
    ----------
    time: datetime.datetime or datetime.date
        the time information we want to format as string.
    agg_time: str
        the minimun level of aggregation information we want to format the
        time.
    format_mode: (default='name')
        the format mode.

    Returns
    -------
    time_str: str
        the time expressed in a string format.

    """
    ## 0. Preparing needed variables
    if format_mode == 'name':
        format_list = ['%Y', '%b', '%d']
    elif format_mode == 'number':
        format_list = ['%Y', '%m', '%d']
    elif format_mode == 'datetime':
        format_list = ['%Y', '%m', '%d', '%H', '%M', '%S', '%f']
    agg_ts_list = ['year', 'month', 'week', 'day', 'hour', 'minute', 'second',
                   'microsecond']
    idx = agg_ts_list.index(agg_time)+1
    format_list = format_list[:idx]
    ## 1. Formatting
    format_str = '-'.join(format_list)
    time_str = time.strftime(format_str)
    return time_str


#def time_aggregator(time, agg_time):
#    """Transform a datetime to another with the level of aggregation selected.
#
#    time: datetime object
#        the datetime information
#    agg_time: str, optional
#        The time of which make the aggregation. The options are:
#        ['year', 'month', 'day', 'hour', 'minute', 'second', 'microsecond']
#
#    TODO
#    ----
#    Add week:
#        ['year', 'month','week','day', 'hour', 'minute', 'second',
#        'microsecond']
#
#    Notes
#    -----
#    Probably DEPRECATED. Better use strformat for current needs.
#
#    """
#
#    ## 1. Preparation for the aggregation
#    agg_ts_list = ['year', 'month', 'day', 'hour', 'minute', 'second',
#                   'microsecond']
#    idx = agg_ts_list.index(agg_time)+1
#
#    ### Corretc!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
#    timevalues = list(time.timetuple())
#
#    ## 2. Application of the aggregation
#    datedict = dict(zip(agg_ts_list[:idx], timevalues[:idx]))
#    if idx < 3:
#        datedict['day'] = 1
#    if idx < 2:
#        datedict['month'] = 1
#    time = datetime.datetime(**datedict)
#    return time
