
"""
Coordinates statistics
----------------------
Module which groups all the related function of the compute of statistics in
coordinates data.

"""


def mean_coord_by_values(df, coordinates_vars, var2agg):
    """Compute the average positions for the values of a variable.

    Parameters
    ----------
    df: pd.DataFrame
        the data in dataframe form.
    coordinates_vars: list
        the list of the coordinates variables.
    var2agg: srt
        the id of the categorical varible which we want to use to collapse
        locations.

    Parameters
    ----------
    table: pd.DataFrame
        the data of the mean average locations.

    """
    #table = df.pivot_table(index=var2agg, values=coordinates_vars)
    table = df[coordinates_vars+[var2agg]].groupby(var2agg).mean()
    return table
