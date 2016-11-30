
"""
Network plots
-------------
Utils and functions to plot network data.

"""

import numpy as np
import matplotlib.pyplot as plt


def plot_net_distribution(net_mat, n_bins):
    """Plot the network distribution.

    Parameters
    ----------
    net_mat: np.ndarray
        the net represented in a matrix way.
    n_bins: int
        the number of intervals we want to use to plot the distribution.

    Returns
    -------
    fig: matplotlib.pyplot.figure
        the figure of the distribution required of the relations between
        elements defined by the `net_mat`.

    """
    net_mat = net_mat.reshape(-1)

    fig = plt.figure()
    plt.hist(net_mat, n_bins)

    l1 = plt.axvline(net_mat.mean(), linewidth=2, color='k', label='Mean',
                     linestyle='--')
    plt.legend([l1], ['Mean'])

    return fig


def plot_heat_net(net_mat, sectors):
    """Plot a heat map of the net relations.

    Parameters
    ----------
    net_mat: np.ndarray
        the net represented in a matrix way.
    sectors: list
        the name of the elements of the adjacency matrix network.

    Returns
    -------
    fig: matplotlib.pyplot.figure
        the figure of the matrix heatmap.

    """
    vmax = np.sort([np.abs(net_mat.max()), np.abs(net_mat.min())])[::-1][0]
    n_sectors = len(sectors)
    assert(net_mat.shape[0] == net_mat.shape[1])
    assert(n_sectors == len(net_mat))

    fig = plt.figure()
    plt.imshow(net_mat, interpolation='none', cmap=plt.cm.RdYlGn,
               vmin=-vmax, vmax=vmax)
    plt.xticks(range(n_sectors), sectors)
    plt.yticks(range(n_sectors), sectors)
    plt.xticks(rotation=90)
    plt.colorbar()
    return fig
