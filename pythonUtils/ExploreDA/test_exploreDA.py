
"""
test_exploreDA
--------------
The module which groups the main functions to explore a new data.

"""

import pandas as pd
import numpy as np
import datetime

from Plotting.contdistrib_plot import cont_distrib_plot
from Plotting.catdistrib_plot import barplot_plot
from Plotting.net_plotting import plot_net_distribution, plot_heat_net
from Plotting.geo_plotting import compute_spatial_density_sparse,\
    clean_coordinates, plot_in_map, plot_geo_heatmap
from Plotting.temp_plotting import temp_distrib

from Statistics.cont_statistics import quantile_compute, ranges_compute,\
    cont_count, log_cont_count
from Statistics.cat_statistics import cat_count
from Statistics.temp_statistics import count_temp_stats
from Statistics.coord_statistics import mean_coord_by_values

from SummaryStatistics.univariate_stats import compute_univariate_stats


def test():
    ## Parameters data
    contdata = pd.Series(np.random.random(100))
    catdata = pd.Series(np.random.randint(0, 10, 100))
    netdata = np.random.random((10, 10))
    xt = np.random.random(100)
    timedata = pd.Series(np.random.random(100),
                         index=[datetime.datetime.now() + datetime.timedelta(e)
                                for e in np.cumsum(xt)])

    df = pd.DataFrame([np.random.random(100), np.random.random(100), catdata])
    df = df.T
    df.columns = ['a', 'b', 'c']

    ### Statitics testing
    # Categorical variable
    cat_count(df, 'c')

    # Continious variable
    quantile_compute(contdata, 5)
    ranges_compute(contdata, 5)
    cont_count(df, 'a', 5)
    log_cont_count(df, 'a', 5)

#    # Coordinate variables (TORETEST :Fail)
#    mean_coord_by_values(df, ['a', 'b'], 'c')

    # Temporal variable
    date_ranges = np.linspace(timedata.min(), timedata.max(), 5)[1:-1]
    count_temp_stats(timedata, date_ranges, tags=None)

#    ### Plotting testing
#    ## Testing univariate categorical variable plotting
#    barplot_plot(catdata, logscale=False)
#    barplot_plot(catdata, logscale=True)
#
#    ## Testing univariate continious variable plotting
#    cont_distrib_plot(contdata, n_bins=5, logscale=True)
#    cont_distrib_plot(contdata, n_bins=5, logscale=False)
#
#    ## Testing network plotting
#    plot_net_distribution(netdata, 5)
#    plot_heat_net(netdata, range(10))
#
#    ## Testing geospatial plotting
#    # Parameters
#    longs, lats = np.random.random(100), np.random.random(100)
#    n_x, n_y = 10, 10
#    n_levs = 5
#    sigma_smooth, order_smooth, null_lim = 5, 0, 0.1
#    var0, var1 = None, np.random.random(100)
#    coordinates = pd.DataFrame([longs, lats]).T
#
#    # Auxiliar function
#    clean_coordinates(coordinates)
#
#    # Computing grid data
#    compute_spatial_density_sparse(longs, lats, n_x, n_y, sigma_smooth,
#                                   order_smooth, null_lim, var0)
#    compute_spatial_density_sparse(longs, lats, n_x, n_y, sigma_smooth,
#                                   order_smooth, null_lim, var1)
#
#    ## Temporal plotting
#    temp_distrib(pd.Series(timedata.index), 'day', logscale=False)
#    temp_distrib(pd.Series(timedata.index), 'day', logscale=True)
#
#    # Main plotters
#    plot_in_map(coordinates, resolution='f', color_cont=None, marker_size=1)
#    plot_in_map(coordinates, resolution='f', color_cont='b', marker_size=1)
#    plot_geo_heatmap(coordinates, n_levs, n_x, n_y, var0)
#
#    ### Summary testing
#    # Parameters
#    df = pd.DataFrame([np.random.random(100), np.random.randint(0, 1, 100),
#                       np.random.random(100), np.random.random(100),
#                       np.random.random(100).cumsum(),
#                       [datetime.datetime(2005, 1, 1) + datetime.timedelta(e)
#                        for e in xt*3500]])
#    df = df.T
#    df.columns = ['cont', 'cat', 'x', 'y', 't', 't_y']
#
#    info_var_cat0 = {'type': 'categorical', 'variables': 'cat',
#                     'ifplot': True, 'logscale': True}
#    info_var_cat1 = {'type': 'categorical', 'variables': 'cat',
#                     'ifplot': True, 'logscale': True}
#
#    info_var_cont0 = {'type': 'continuous', 'n_bins': 10, 'ifplot': True,
#                      'logscale': False, 'variables': 'cont'}
#    info_var_cont1 = {'type': 'continuous', 'n_bins': 10, 'ifplot': True,
#                      'logscale': True, 'variables': 'cont'}
#
#    info_var_coord = {'type': 'coordinates', 'variables': ['x', 'y'],
#                      'ifplot': True}
#
#    info_var_temp = {'type': 'temporal', 'variables': 't', 'logscale': False,
#                     'ifplot': True, 'agg_time': 'second',
#                     'stats_pars': {'date_ranges': range(101)}}
#    info_var_templog = {'type': 'temporal', 'variables': 't', 'logscale': True,
#                        'ifplot': True, 'agg_time': 'second',
#                        'stats_pars': {'date_ranges': range(101)}}
#
#    info_var_ty = {'type': 'temporal', 'variables': 't_y', 'logscale': False,
#                   'ifplot': True, 'agg_time': 'month',
#                   'stats_pars': {'tags': ['pre', 'through', 'post'],
#                                  'date_ranges': ['2006-01-01', '2012-12-31']
#                                  }
#                   }
#    info_var_tylog = {'type': 'temporal', 'variables': 't_y', 'logscale': True,
#                      'ifplot': True, 'agg_time': 'month',
#                      'stats_pars': {'tags': None,
#                                     'date_ranges': ['2006-01-01',
#                                                     '2012-12-31']
#                                     }
#                      }
#
#    info_var_tempdist = {'type': 'tmpdist', 'variables': ['x', 'y', 't'],
#                         'logscale': False, 'ifplot': True}
#    info_var_tempdistlog = {'type': 'tmpdist', 'variables': ['x', 'y', 't'],
#                            'logscale': True, 'ifplot': True}
#
#    # Test
#    compute_univariate_stats(df, info_var_cat0)
#    compute_univariate_stats(df, info_var_cat1)
#    compute_univariate_stats(df, info_var_cont0)
#    compute_univariate_stats(df, info_var_cont1)
#    compute_univariate_stats(df, info_var_coord)
#    compute_univariate_stats(df, info_var_ty)
#    compute_univariate_stats(df, info_var_tylog)
##    compute_univariate_stats(df, info_var_temp)
##    compute_univariate_stats(df, info_var_templog)
##    compute_univariate_stats(df, info_var_tempdist)
##    compute_univariate_stats(df, info_var_tempdistlog)
