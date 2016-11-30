
"""
Categorical variable statistics
-------------------------------
Module which groups all the functions needed to compute the statistics and the
description of the categorical variables.

"""


## Categorical count
def cat_count(df, variable):
    """The catagory counts.

    Parameters
    ----------
    df: pd.DataFrame
        the data in dataframe form.
    variable: str
        the variable of the database we want to study.

    Returns
    -------
    counts: pd.DataFrame
        the counts of the possible categories in the categorical variable.

    """
    if type(variable) == list:
        variable = variable[0]
    counts = df[variable].value_counts()
    return counts
