
"""
test_Testers
------------
The tests of the statistical tests and measures.

"""

import numpy as np
import networkx as nx

from sorting_measures import roc_comparison, compute_lift_curve
from plotting_metricresults import plot_roc_curves, plot_roc_curve,\
    plot_lift_curve, plot_lift_curves
from general_metrics import network_roc_comparison, compute_measure


def test():
    ## Main functions to be used
    def sim_prediction_bin(n, real, noise):
        noise = noise*(np.random.random(n)-.5)*2
        pred = real_cat+noise
        pred = (pred-pred.min())/(pred.max()-pred.min())
        return pred

    def sim_prediction_cont(n, real, noise):
        noise = noise*np.random.random(n)
        pred = real_cat+noise
        return pred

    ## Parameters
    n, m_preds, noise = 1000, 10, .6

    real_cat = np.random.randint(0, 2, n)
    real_cont = np.random.random(n)

    pred_cat = sim_prediction_bin(n, real_cat, noise)
    preds_cat = [sim_prediction_bin(n, real_cat, noise)
                 for i in range(m_preds)]
    pred_cont = sim_prediction_cont(n, real_cont, noise)
    preds_cont = [sim_prediction_cont(n, real_cont, noise)
                  for i in range(m_preds)]
    tags = ['prediction '+str(i) for i in range(m_preds)]

    x, y = np.random.random(100), np.random.random(100)
    G_x = nx.from_numpy_matrix(x.reshape((10, 10)))
    G_y = nx.from_numpy_matrix(x.reshape((10, 10)) > 0.5)
    Gs_x = [G_x for i in range(10)]

    ## Testing sorting measures
    ###########################
    fpr, tpr, _ = roc_comparison(real_cat, pred_cat)
    rocs = [roc_comparison(real_cat, preds_cat[i])
            for i in range(m_preds)]
    fprs = [rocs[i][0] for i in range(m_preds)]
    tprs = [rocs[i][1] for i in range(m_preds)]
    measures = np.random.random(len(rocs))
    compute_lift_curve(real_cat, pred_cat, 10)
    lift = compute_lift_curve(real_cont, pred_cont, 10)[1]
    lifts = [compute_lift_curve(real_cat, preds_cat[i], 10)[1]
             for i in range(m_preds)]

    ## Testing plotting
    ###################
    fig = plot_roc_curves(fprs, tprs, measures, tags)
    fig = plot_roc_curve(fpr, tpr, measures[0])
#    fig = plot_lift_curves(lifts, tags)
#    fig = plot_lift_curve(lift)
#
#    ## Testing main computing funcitons
#    ###################################
#    measures, fig = compute_measure(real_cat, pred_cat, metric="roc_curve",
#                                    create_plot=True, tags=['0'])
#    measures = compute_measure(real_cat, pred_cat, metric="roc_curve",
#                               create_plot=False, tags=['0'])
#    measures, fig = compute_measure(real_cat, preds_cat, metric="roc_curve",
#                                    create_plot=True, tags=tags)
#    measures = compute_measure(real_cat, preds_cat, metric="roc_curve",
#                               create_plot=False, tags=tags)
#    measures, fig = compute_measure(real_cat, pred_cat, metric="lift10",
#                                    create_plot=True, tags=['0'])
#    measures = compute_measure(real_cat, pred_cat, metric="lift10",
#                               create_plot=False, tags=['0'])
#    measures, fig = compute_measure(real_cat, preds_cat, metric="lift10",
#                                    create_plot=True, tags=tags)
#    measures = compute_measure(real_cat, preds_cat, metric="lift10",
#                               create_plot=True, tags=tags)
#
#    ########
#    names = ['network inferred '+str(i) for i in range(10)]
#    measure, fig = network_roc_comparison(G_x, G_y)
#    measure, fig = network_roc_comparison(G_x, G_y, names)
#    measure, fig = network_roc_comparison(Gs_x, G_y, ['network inferred 0'])
