
from sklearn.metrics import roc_curve, auc
import numpy as np
import networkx as nx

from pyCausality.Plotting.results_plots import *


def roc_comparison(G_inferred, G_real, names=None):
    """This function is a function of distance between the real graph structure
    (the ground truth) and the inferred graph structure.
    """

    # Check inputs and adapt
    if type(G_inferred) != list:
        G_inferred = [G_inferred]
    if type(names) == list:
        if len(names) > len(G_inferred):
            names = names[:len(G_inferred)]
        elif len(names) < len(G_inferred):
            names = ['Prediction_'+i for i in range(len(G_inferred))]
    else:
        names = ['Prediction']

    # Extraction of the labels
    pred = []
    for i in range(len(G_inferred)):
        aux = nx.adjacency_matrix(G_inferred)
        aux = aux.reshape(np.prod(aux.shape))
        aux = np.asarray(aux).reshape(-1)
        pred.append(aux)
    real = nx.adjacency_matrix(G_real)
    real = real.reshape(np.prod(real.shape))
    real = np.asarray(real).reshape(-1)

    # Computing the measure (TO CONTINUE)
    measure, fig = compute_measure(real, pred[0])

    return measure, fig


def compute_measure(real, pred, metric="roc_curve", create_plot=True,
                    tags=None):
    '''This function compute some given measures of fit of the given real
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
    0.6
    >>> measure, fig = compute_measure(real, pred, "lift10")
    >>> measure
    1.7
    >>> measure = compute_measure(real, pred, "lift10", False)

    See also
    --------
    sklearn.metrics.roc_curve, compute_lift_curve,
    pyCausality.Plotting.plot_roc_curve , pyCausality.Plotting.plot_lift_curve

    '''
    multiple = type(pred) == list

    # ROC measure
    if metric == 'roc_curve':
        if not multiple:
            # Compute the measure of ROC curve
            fpr, tpr, thresholds = roc_curve(real, pred)
            # numerical measure
            measure = auc(fpr, tpr)
            fprs, tprs, measures = [fpr], [tpr], [measure]
        else:
            fprs, tprs, measures = [], [], []
            for i in range(len(pred)):
                # Compute the measure of ROC curve
                fpr, tpr, thresholds = roc_curve(real, pred[i])
                # numerical measure
                measure = auc(fpr, tpr)
                # Appending
                fprs.append(fpr)
                tprs.append(tpr)
                measures.append(measures)

        # Plot: Handle this plot.
        if create_plot:
            # Call for the plot
            fig = plot_roc_curve(fprs, tprs, measures, tags)
        else:
            return measures

    # LIFT 10 MEASURE
    elif metric == 'lift10':
        if not multiple:
            # Compute lift10 curve
            lift10 = compute_lift_curve(real, pred, 10)
            # numerical measure
            measure = lift10[0]
        else:
            lift10s, measures = [], []
            for i in range(len(pred)):
                # Compute lift10 curve
                lift10 = compute_lift_curve(real, pred, 10)
                # numerical measure
                measure = lift10[0]
                # appending
                lift10s.append(lift10)
                measures.append(measure)

        # Plot
        if create_plot:
            fig = plot_lift_curve(lift10, tags)
        else:
            return measure

    return measure, fig


def compute_lift_curve(real, pred, n):
    """Compute the lift curve.

    Parameters
    ----------
    real : array_like shape(N,)
        the real values to be predicted.
    pred : array_like shape(N,)
        the prediction of the real values.
    n : int
        ecils to be used.

    Returns
    -------
    lift : array_like shape(n,)

    """

    # Compute extremes in order to split the sample.
    splitters = [i*len(pred)/n for i in range(1, n+1)]
    splitters = [0] + splitters
    # Indices of the ordering
    indices = np.argsort(pred)[::-1]
    # Creation vector of decils
    decil = np.zeros(len(pred))
    for i in range(n):
        decil[indices[splitters[i]:splitters[i+1]]] = i+1
    # Creation of the lift vector.
    lift = np.zeros(n)
    complement = np.zeros(n)
    for i in range(n):
        lift[i] = real[decil == i+1].mean()
        complement[i] = pred[decil == i+1].mean()
    lift = lift/(lift.mean())

    return lift





## TO BE DEPRECATED
#def calculate_ROC(true,pred):
#    fpr, tpr, thresholds = metrics.roc_curve(true,pred)
#    return fpr, tpr, thresholds
#
#
#def plotROC(fpr,tpr,fullpath):
#    pl.clf()
#    pl.plot(fpr, tpr, label='ROC curve (area = %0.4f)' % metrics.auc(fpr,tpr))
#    pl.plot([0, 1], [0, 1], '--')
#    pl.xlim([0.0, 1.0])
#    pl.ylim([0.0, 1.0])
#    pl.xlabel('False Positive Rate')
#    pl.ylabel('True Positive Rate')
#    pl.title('Receiver operating characteristic ')
#    pl.legend(loc="lower right")
#    pl.savefig(fullpath)
#    #pl.show()
#    pl.close()
