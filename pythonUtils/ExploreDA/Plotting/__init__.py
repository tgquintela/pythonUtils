
"""
Plotting
--------
Plotting functions collections for exploring new data.

"""
from catdistrib_plot import barplot_plot
from contdistrib_plot import cont_distrib_plot
from geo_plotting import plot_in_map
from temp_plotting import temp_distrib, distrib_across_temp


def general_plot(df, info_var):
    """General plotting function. That function wrappes all the plotting
    functions of the module in order to make easy to automatize some process
    of exploring new data.

    Parameters
    ----------
    df: pd.DataFrame
        the data to explore.
    info_var: dict
        the information to plot the variables selected. The `info_var` is
        formed by:
            * type: the type of the variables we want to plot.
            * variables: the variables we want to plot.
            * logscale: if we want to use logarithmic scale.
            * hist_table
            * log_hist_table
            * agg_time

    Returns
    -------
    fig: matplotlib.pyplot.figure
        the figure of the plot defined by the info_var.

    """
    typevar = info_var['type'].lower()
    if typevar in ['discrete', 'categorical']:
        fig = barplot_plot(df[info_var['variables']])
        if info_var['logscale'] in [True, 'True', 'TRUE']:
            fig = [fig, barplot_plot(df[info_var['variables']], True)]
    elif typevar == 'continuous':
#        fig = cont_distrib_plot(info_var['count_hist'], info_var['n_bins'])
        fig = cont_distrib_plot(df[info_var['variables']],
                                info_var['hist_table'][1])
        if info_var['logscale'] in [True, 'True', 'TRUE']:
            fig = [fig, cont_distrib_plot(df[info_var['variables']],
                                          info_var['log_hist_table'][1], True)]
    elif typevar == 'coordinates':
        fig = plot_in_map(df[info_var['variables']])
    elif typevar in ['time', 'temporal']:
        fig = temp_distrib(df[info_var['variables']], info_var['agg_time'])
        if info_var['logscale'] in [True, 'True', 'TRUE']:
            fig = [fig, temp_distrib(df[info_var['variables']],
                                     info_var['agg_time'], True)]
    elif typevar == 'tmpdist':
        fig = distrib_across_temp(df, info_var['variables'])
        if info_var['logscale'] in [True, 'True', 'TRUE']:
            fig = [fig, distrib_across_temp(df, info_var['variables'],
                                            logscale=True)]
    else:
        print typevar, info_var['variables']

    #figures = {'figure': fig, 'plot_desc': info_var['plot_desc']}
    return fig
