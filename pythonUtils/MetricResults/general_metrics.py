
"""
Statistical testing mesaures
----------------------------
The compute and plot of Statistical testing measures.

"""

import numpy as np
import networkx as nx
from scipy.sparse import issparse

from sorting_measures import roc_comparison, compute_lift_curve
from plotting_metricresults import plot_roc_curves, plot_lift_curves


def network_roc_comparison(G_inferred, G_real, names=None):
    """This function is a function of distance between the real graph structure
    (the ground truth) and the inferred graph structure.

    Parameters
    ----------
    G_inferred: nx.Graph or list
        the inferred network.
    G_real: nx.Graph or list
        the real network.
    names: list (default=None)
        the list of names.

    Returns
    -------
    measure: float
        the measure computed.
    fig: matplotlib.pyplot.figure
        the figure in which is plot the measure.

    """

    # Check inputs and adapt
    if type(G_inferred) != list:
        G_inferred = [G_inferred]
    if type(names) == list:
        if len(names) > len(G_inferred):
            names = names[:len(G_inferred)]
        elif len(names) < len(G_inferred):
            names = ['Prediction_'+str(i) for i in range(len(G_inferred))]
    else:
        names = ['Prediction']

    def extend_net_values(G):
        aux = nx.adjacency_matrix(G)
        if issparse(aux):
            aux = np.asarray(aux.todense()).ravel()
        else:
            aux = np.asarray(aux).ravel()
        return aux

    # Extraction of the labels
    pred = []
    for i in range(len(G_inferred)):
        aux = extend_net_values(G_inferred[i])
        pred.append(aux)
    real = extend_net_values(G_real)

    # Computing the measure (TO CONTINUE)
    measure, fig = compute_measure(real, pred, tags=names)

    return measure, fig


def compute_measure(real, pred, metric="roc_curve", create_plot=True,
                    tags=None):
    """This function compute some given measures of fit of the given real
    labels (real) and the predicted labels (pred).

    Parameters
    ----------
    real: array_like, shape (N,)
        labels of correct real values.
    pred: array_like, shape (N,), list of arrays.
        predicted labels.
    metric: str, optional
        metric used to check how good is the prediction.
        There are available roc_curve, and lift10.
    tags: list
        the tags assigned for the possible predictions we are going to test.

    Returns
    -------
    measures: float
        the measure of divergence between both
    fig : matplotlib figure
        the plot of related to the measure.

    Examples
    --------
    >>> real = np.random.randint(0,2,50)
    >>> pred = np.random.rand(50)
    >>> measure, fig = compute_measure(real, pred, "roc_curve")
    >>> measure
    [0.6]
    >>> measure, fig = compute_measure(real, pred, "lift10")
    >>> measure
    [1.7]
    >>> measure = compute_measure(real, pred, "lift10", False)

    See also
    --------
    sklearn.metrics.roc_curve, compute_lift_curve,
    pythonUtils.TesterResults.plotting_testerresults.plot_roc_curve ,
    pythonUtils.TesterResults.plotting_testerresults.plot_lift_curve

    """
    multiple = type(pred) == list

    # ROC measure
    if metric == 'roc_curve':
        if not multiple:
            # Compute the measure of ROC curve
#            fpr, tpr, thresholds = roc_curve(real, pred)
#            # numerical measure
#            measure = auc(fpr, tpr)
            fpr, tpr, measure = roc_comparison(real, pred)
            fprs, tprs, measures = [fpr], [tpr], [measure]
        else:
            assert(len(pred) == len(tags))
            fprs, tprs, measures = [], [], []
            for i in range(len(pred)):
#                # Compute the measure of ROC curve
#                fpr, tpr, thresholds = roc_curve(real, pred[i])
#                # numerical measure
#                measure = auc(fpr, tpr)
                fpr, tpr, measure = roc_comparison(real, pred[i])
                # Appending
                fprs.append(fpr)
                tprs.append(tpr)
                measures.append(measure)
            assert(len(measures) == len(pred))

        # Plot: Handle this plot.
        if create_plot:
            # Call for the plot
            fig = plot_roc_curves(fprs, tprs, measures, tags)
        else:
            return measures

    # LIFT 10 MEASURE
    elif metric == 'lift10':
        if not multiple:
            # Compute lift10 curve
            _, lift10, _ = compute_lift_curve(real, pred, 10)
            # numerical measure
            lift10 = [lift10]
            measure = [lift10[0]/lift10[-1]]
        else:
            assert(len(pred) == len(tags))
            lift10, measures = [], []
            for i in range(len(pred)):
                # Compute lift10 curve
                _, lift10i, _ = compute_lift_curve(real, pred[i], 10)
                # numerical measure
                measure = lift10i[0]/lift10i[-1]
                # appending
                lift10.append(lift10i)
                measures.append(measure)

        # Plot
        if create_plot:
            fig = plot_lift_curves(lift10, tags)
        else:
            return measure

    return measure, fig
