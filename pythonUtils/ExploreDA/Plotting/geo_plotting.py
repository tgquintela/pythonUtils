
"""
Geo_plotting
------------
Module which groups the geographical map plots.

"""

from mpl_toolkits.basemap import Basemap, cm
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import numpy as np
#from Mscthesis.Statistics.stats_functions import compute_spatial_density


def plot_in_map(coordinates, resolution='f', color_cont=None, marker_size=1):
    """Plot the coordinates in points in the map.

    Parameters
    ----------
    coordinates: pd.DataFrame
        the coordinates points.
    resolution: str optional ['f', ] (default='f')
        the resolution to plot the geographical map.
    color_cont:  (default=None)
        the color of the continents.
    marker_size: int or float
        the size of the marker

    Returns
    -------
    fig: matplotlib.pyplot.figure
        the figure of the points in the map.

    """

    # Delete nan
    coordinates = coordinates.dropna()
    coordinates = np.array(coordinates)
    # Delete 0,0
    idx = np.logical_and(coordinates[:, 0] != 0, coordinates[:, 1] != 0)
    coordinates = coordinates[idx, :]

    # compute coordinates
    longs = coordinates[:, 0]
    lats = coordinates[:, 1]

    lat_0 = np.mean(lats)
    llcrnrlon = np.floor(np.min(longs))
    llcrnrlat = np.floor(np.min(lats))
    urcrnrlon = np.ceil(np.max(longs))
    urcrnrlat = np.ceil(np.max(lats))

    # Set map
    fig = plt.figure()
    mapa = Basemap(projection='merc', lat_0=lat_0, llcrnrlon=llcrnrlon,
                   llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon,
                   urcrnrlat=urcrnrlat, resolution=resolution)
    mapa.drawcoastlines()
    mapa.drawcountries()
    if color_cont is not None:
        mapa.fillcontinents(color=color_cont)
    mapa.drawmapboundary()

    mapa.scatter(longs, lats, marker_size, marker='o', color='r', latlon=True)

    return fig


def plot_geo_heatmap(coordinates, n_levs, n_x, n_y, var=None):
    """Plot the coordinates in points in the map.

    Parameters
    ----------
    coordinates: pd.DataFrame
        the coordinates points.
    n_levs: int
        the number of level colors for the heat representation.
    n_x: int
        the size of the axis x in 'pixels'.
    n_y: int
        the size of the axis y in 'pixels'.
    var: np.ndarray (default=None)
        the weights of each point.

    Returns
    -------
    fig: matplotlib.pyplot.figure
        the figure of the heat in the map.

    """

    ## 00. Preprocess of the data
    # Delete nan
    coordinates = coordinates.dropna()
    coordinates = np.array(coordinates)
    # Delete 0,0
    idx = np.logical_and(coordinates[:, 0] != 0, coordinates[:, 1] != 0)
    coordinates = coordinates[idx, :]

    ## 0. Preparing needed variables
    # compute coordinates
    longs = coordinates[:, 0]
    lats = coordinates[:, 1]
    # Preparing corners
    lat_0 = np.mean(lats)
    llcrnrlon = np.floor(np.min(longs))
    llcrnrlat = np.floor(np.min(lats))
    urcrnrlon = np.ceil(np.max(longs))
    urcrnrlat = np.ceil(np.max(lats))

    ## 1. Set map
    fig = plt.figure()
    mapa = Basemap(projection='merc', lat_0=lat_0, llcrnrlon=llcrnrlon,
                   llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon,
                   urcrnrlat=urcrnrlat, resolution='h')
    mapa.drawcoastlines()
    mapa.drawcountries()
#    mapa.fillcontinents(color='gray')

    # Draw water
    #mapa.drawmapboundary(fill_color='aqua')
    #mapa.fillcontinents(color='coral')
    #mapa.drawlsmask(ocean_color='aqua', lakes=False)

    # mapa.scatter(longs, lats, 10, marker='o', color='k', latlon=True)

    ## 2. Preparing heat map
    density, l_x, l_y = compute_spatial_density(longs, lats, n_x+1, n_y+1, var)
    clevs = np.linspace(density.min(), density.max(), n_levs+1)
    l_x, l_y = mapa(l_x, l_y)

    ## 3. Computing heatmap
    #mapa.scatter(l_x, l_y, 1, marker='o', color='r', latlon=True)
    cs = mapa.contourf(l_x, l_y, density, clevs, cmap='Oranges')
    #cs = plt.contourf(l_x, l_y, density, clevs)
    # add colorbar.
    cbar = mapa.colorbar(cs, location='bottom', pad="5%")
    cbar.set_label('density')

    ## 4. Fix details

    # add title
    plt.title('Heat map of companies density')

    return fig


###############################################################################
########################## Computing spatial density ##########################
###############################################################################
def compute_spatial_density(longs, lats, n_x, n_y, var=None, sigma_smooth=5,
                            order_smooth=0):
    """Computation of the spatial density given the latitutes and longitudes of
    the points we want to count.

    Parameters
    ----------
    longs: np.ndarray
        the longitudes variable.
    lats: np.ndarray
        the latitutes variable.
    n_x: int
        the size of the axis x in 'pixels'.
    n_y: int
        the size of the axis y in 'pixels'.
    var: np.ndarray (default=None)
        the weight values of each coordinates.
    sigma_smooth: float
        the size of the gaussian sigma smoothing.
    order_smooth: int
        the spatial size of the smoothing in 'pixels'.

    Returns
    -------
    density: np.ndarray
        the density of each square grid region.
    l_x: np.ndarray
        the borders of each grid regions in x dimension.
    l_y: np.ndarray
        the borders of each grid regions in y dimension.

    TODO
    ----
    Smoothing function

    """
    ## 0. Setting initial variables
    llcrnrlon = np.floor(np.min(longs))
    llcrnrlat = np.floor(np.min(lats))
    urcrnrlon = np.ceil(np.max(longs))
    urcrnrlat = np.ceil(np.max(lats))
    l_x = np.linspace(llcrnrlon, urcrnrlon, n_x+1)
    l_y = np.linspace(llcrnrlat, urcrnrlat, n_y+1)

    ## 1. Computing density
    density = computing_density_var(longs, lats, [l_x, l_y], var)
    density = density.T

    ## 2. Smothing density
    density = ndimage.gaussian_filter(density, sigma_smooth, order_smooth)

    ## 3. Output
    l_x = np.mean(np.vstack([l_x[:-1], l_x[1:]]), axis=0)
    l_y = np.mean(np.vstack([l_y[:-1], l_y[1:]]), axis=0)
    l_x, l_y = np.meshgrid(l_x, l_y)

    return density, l_x, l_y


def compute_spatial_density_sparse(longs, lats, n_x, n_y, sigma_smooth=5,
                                   order_smooth=0, null_lim=0.1, var=None):
    """Computation of the spatial density given the latitutes and longitudes of
    the points we want to count.

    Parameters
    ----------
    longs: np.ndarray
        the longitudes variable.
    lats: np.ndarray
        the latitutes variable.
    n_x: int
        the size of the axis x in 'pixels'.
    n_y: int
        the size of the axis y in 'pixels'.
    sigma_smooth: float
        the size of the gaussian sigma smoothing.
    order_smooth: int
        the spatial size of the smoothing in 'pixels'.
    null_lim: float (default=0.1)
        the lower limit to consider null density.
    var: np.ndarray (default=None)
        the weight values of each coordinates.

    Returns
    -------
    density: np.ndarray
        the density of each square grid region.
    l_x: np.ndarray
        the borders of each grid regions in x dimension.
    l_y: np.ndarray
        the borders of each grid regions in y dimension.

    TODO
    ----
    Smoothing function

    """
#    ## 0. Setting initial variables
#    llcrnrlon = np.floor(np.min(longs))
#    llcrnrlat = np.floor(np.min(lats))
#    urcrnrlon = np.ceil(np.max(longs))
#    urcrnrlat = np.ceil(np.max(lats))
#    l_x = np.linspace(llcrnrlon, urcrnrlon, n_x+1)
#    l_y = np.linspace(llcrnrlat, urcrnrlat, n_y+1)
#
#    ## 1. Computing density
#    density = computing_density_var(longs, lats, [l_x, l_y], var)
#    #density = density.T
#
#    ## 2. Smothing density
#
#    ## 3. Output
#    l_x = np.mean(np.vstack([l_x[:-1], l_x[1:]]), axis=0)
#    l_y = np.mean(np.vstack([l_y[:-1], l_y[1:]]), axis=0)

    ## 0. Computing spatial density
    density, l_x, l_y = compute_spatial_density(longs, lats, n_x, n_y, var,
                                                sigma_smooth, order_smooth)
    l_x, l_y = l_x[0, :], l_y[:, 0]
    ## 1. Sparsing formatting
    idxs = (density > null_lim).nonzero()
    density = density[idxs]
    l_x = l_x[idxs[0]]
    l_y = l_y[idxs[1]]

    return density, l_x, l_y


def computing_density_var(longs, lats, border_grid, var=None):
    """Computing density of a variable.

    Parameters
    ----------
    longs: np.ndarray
        the longitudes variable.
    lats: np.ndarray
        the latitutes variable.
    border_grid: list of np.ndarray
        the borders of each square in the grid.
    var: np.ndarray (default=None)
        the weight values of each coordinates.

    Returns
    -------
    density: np.ndarray
        the density of each square grid region.

    """
    if var is None:
        density, _, _ = np.histogram2d(longs, lats, border_grid)
    else:
        l_x_bor = border_grid[0]
        l_y_bor = border_grid[1]
        n_x = l_x_bor.shape[0]-1
        n_y = l_y_bor.shape[0]-1

        ## Indexing part
        density = np.zeros((n_x, n_y))
        for i in range(n_x):
            idxs_i = np.logical_and(l_x_bor[i] <= longs, l_x_bor[i+1] >= longs)
            for j in range(n_y):
                idxs_j = np.logical_and(l_y_bor[j] <= lats,
                                        l_y_bor[j+1] >= lats)
                idxs = np.logical_and(idxs_i, idxs_j)
                density[i, j] = np.sum(var[idxs])
    return density


###############################################################################
############################# Auxiliar functions ##############################
###############################################################################
def clean_coordinates(coordinates):
    """Clean possible incorrect coordinates.

    Parameters
    ----------
    coordinates: pd.DataFrame
        the coordinates points.

    Returns
    -------
    coordinates: np.ndarray
        the coordinates points.

    """
    # Delete nan
    coordinates = coordinates.dropna()
    coordinates = np.array(coordinates)
    # Delete 0,0
    idx = np.logical_and(coordinates[:, 0] != 0, coordinates[:, 1] != 0)
    coordinates = coordinates[idx, :]
    return coordinates
