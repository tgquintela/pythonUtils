
"""
sorting metrics
---------------
Compute how correct is a classification by computing how good are ordered
regarding the real value of the arrays.
"""

from sklearn.metrics import roc_curve, auc


def roc_comparison(real, pred):
    """Function to compute the roc curve. It is useful to test the performance
    of a binary classifier.

    Parameters
    ----------
    real : array_like shape(N,)
        the real values to be predicted.
    pred : array_like shape(N,)
        the prediction of the real values.

    Returns
    -------
    fpr : array_like shape(n,)
        false positive rate.
    tpr : array_like shape(n,)
        true positive rate.
    measure: float
        the common measure of performance associated.

    """
    # Compute the measure of ROC curve
    fpr, tpr, thresholds = roc_curve(array_real, array_pred)
    # numerical measure
    measure = auc(fpr, tpr)
    return fpr, tpr, measure


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
    quantiles: array_like shape(n,)
    lift : array_like shape(n,)

    measure: float
        the common measure of performance associated.

    """

    ## 0. Format inputs
    n = int(n)

    ## 1. Compute curve
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

    ## 2. Format outputs
    quantiles = np.arange(0, n)
    measure = lift[0]

    return quantiles, lift, measure

