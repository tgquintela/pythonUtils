
"""
Plotting test results
---------------------
Collection of plotting results measures.

"""

import matplotlib.pyplot as plt


def plot_roc_curves(fprs, tprs, measures, tags):
    """Plot the ROC curves of the prediction done.

    Parameters
    ----------
    fprs: list
        the False Positive Rate for each considered prediction.
    tprs: list
        the True Positive Rate for each considered prediction.
    measures: list
        the area under the curve measure of roc for each prediction.
    tags: list
        the names of each predictions.

    Returns
    -------
    fig: matplotlib.pyplot.Figure
        the figure of the lift curve.

    """
    assert(len(fprs) == len(tags))
    assert(len(tprs) == len(tags))
    assert(len(measures) == len(tags))

    fig = plt.figure()
#    lines = []
#    for i in range(len(fprs)):
#        p = plt.plot(fprs[i], tprs[i],
#                     label=tags[i] + '; AUC = %0.2f' % measures[i])
#        lines.append(p[0])
#
#    plt.ylabel('True Positive Rate')
#    plt.xlabel('False Positive Rate')
#    plt.title('Roc curve measure')
#    plt.legend(loc='lower right')
    return fig


def plot_roc_curve(fpr, tpr, measure):
    """Plot the ROC curves of the prediction done.

    Parameters
    ----------
    fpr: np.ndarray
        the False Positive Rate for the considered prediction.
    tpr: np.ndarray
        the True Positive Rate for the considered prediction.
    measures: list
        the area under the curve measure of roc for the prediction.
    tags: list
        the names of each predictions.

    Returns
    -------
    fig: matplotlib.pyplot.Figure
        the figure of the lift curve.

    """
    figure = plt.figure()
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % measure)

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('Receiver Operating Characteristic curve')
    plt.legend(loc='lower right')
    return figure


def plot_lift_curves(lifts, tags):
    """Plot a lift curve.

    Parameters
    ----------
    lifts: list
        the list of numbers of the lift for each quantile for each predictions.
    tags: list
        the names of each predictions.

    Returns
    -------
    fig: matplotlib.pyplot.Figure
        the figure of the lift curve.

    """
    assert(len(lifts) == len(tags))

    fig = plt.figure()
    lines = []
    for i in range(len(lifts)):
        lines.append(plt.plot(lifts[i])[0])

    xticks = [str(i)+'Q' for i in range(len(lifts[0]))]
    plt.xticks(range(len(lifts[0])), xticks)
    plt.xlabel('Quantiles')
    plt.ylabel('Lift measure')
    plt.legend(lines, tags, loc='lower right')
    plt.title('Lift curve measure')
    return fig


def plot_lift_curve(lift):
    """Plot a lift curve.

    Parameters
    ----------
    lift: np.ndarray
        the numbers of the lift for each quantile.

    Returns
    -------
    fig: matplotlib.pyplot.Figure
        the figure of the lift curve.

    """
    fig = plt.figure()
    plt.plot(lift)
    plt.xticks(range(len(lift)), [str(i)+'Q' for i in range(len(lift))])
    plt.xlabel('Quantiles')
    plt.ylabel('Lift measure')
    plt.legend(('Lift curve', ), loc='lower left')
    plt.title('Lift curve measure')
    return fig
