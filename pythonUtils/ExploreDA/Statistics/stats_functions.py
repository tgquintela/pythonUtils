
"""
Calcular estadistiques.
"""


import numpy as np
import pandas as pd


## Creation of cnae index at a given level
def cnae_index_level(col_cnae, level):
    pass


# Distance
def distance_cnae(col_cnae):
    pass


def finantial_per_year(servicios):
    """Function which transform the servicios data to a data for each year.
    """
    pass


## Translate to latex summary
## Individually by variable
def extend_data_by_year(df, variables, newvars=None):
    """
    """
    #year_formation = lambda x: '0'*(2-len(str(x)))+str(x)
    #years = [year_formation(i) for i in range(6, 13)] if years is None else
    #years

    if newvars is None:
        newvars = [variables[i][0][2:] for i in range(len(variables))]

    df2 = []
    y = len(variables[0])
    for i in range(y):
        aux = df[[variables[j][i] for j in range(len(variables))]]
        aux.columns = newvars
        df2.append(aux)
    df2 = pd.concat(df2)

    return df2


###############################################################################
###############################################################################
