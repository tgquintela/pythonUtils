
"""
test_combinatorics
------------------
Module to test the combinatorics utilities.

"""

from generador import generator_sim_i, create_combination_parameters,\
    second_order_combinations, parameters_second_order_generation


def test():
    ### Parameters
    l_parvals = [[0, 1, 2], [2, 3, 4]]
    params = [[0, 1], [1, 2]]
    sim_f = [lambda x, y: x+y]*2
    sim_p = [[0, 1, 2], [2, 3, 4]]
    params2 = [[0], [3], [4]]

    ### Generation
    create_combination_parameters(l_parvals)
    generator_sim_i(sim_f, params)
    second_order_combinations(sim_f, params, sim_f, params2)
    parameters_second_order_generation(sim_f, params, sim_f, params2)
