
"""
Generation
==========
Utils to perform combinations generation.


Example
-------
>>> fs = [run_vots1, run_vots2]
>>> params = [[5, 10, 15, 20, 25, 50], [10000]]
>>> fs2 = [disc1, disc2, disc3]
>>> params2 = [[[]]]
>>> p1, p2 = product(*params), product(*params2)
>>> r = generator_res(fs, p1, fs2, p2)
>>> r2 = generator_descnames(fs, p1, fs2, p2)

"""

from itertools import product
import numpy as np
import copy


def create_combination_parameters(l_parvals):
    """Create a combination of parameters.

    Parameters
    ----------
    l_parvals: list
        the possible parameters list for each possible parameter.

    Returns
    -------
    pr_params: list
        the possible combinations of parameters.

    """
    return product(*l_parvals)


def generator_sim_i(sim_f, params):
    """Generation of simulation by iteration.

    Parameters
    ----------
    sim_f: list
        list of funcitons to apply.
    params: list
        list of parameters to apply to the given functions.

    Returns
    -------
    v: optional
        the result of sequentially one of the sim_f for each param.

    """
    for f, p in sim_f, params:
        yield f(*p)


def second_order_combinations(sim_f, sim_p, desc_f, desc_p):
    """Generation of results.

    Parameters
    ----------
    sim_f: list
        list of functions to apply.
    params: list
        list of parameters to apply to the given functions.
    desc_f: list
        the list of functions to apply to the result of the previous
        combinations.
    desc_p: list
        the list of the parameters of the functions to apply to the previous
        combinations.

    Returns
    -------
    results: list
        the second order combination of parameters and functions.

    """
    results = []
    sim_info = product(sim_f, copy.copy(sim_p))
    for f, p in sim_info:
        v = f(*p)
        desc_info = product(desc_f, copy.copy(desc_p))
        for vf, vp in desc_info:
            results.append(vf(v, *vp))
    return results


def parameters_second_order_generation(sim_f, sim_p, desc_f, desc_p):
    """Generation of the descriptions for a combinations of functions and
    parameters.

    Parameters
    ----------
    sim_f: list
        the list for each parameters of their possible functions.
    sim_p: list
        the list for each parameters of their possible parameters.
    desc_f: list
        the list for each parameters of their possible description functions.
    desc_p: list
        the list for each parameters of their possible parameter descriptions.

    Returns
    -------
    results: list
        the combinations of functions, parameters and descriptions.

    """
    results = []
    sim_info = product(sim_f, copy.copy(sim_p))
    for f, p in sim_info:
        desc_info = product(desc_f, copy.copy(desc_p))
        for vf, vp in desc_info:
            results.append([f, p, vf, vp])
    return results
