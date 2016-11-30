
"""
"""

#import numpy as np
#from joblib import Parallel, delayed
#from sklearn.base import clone
#

#def looping_tests():
#    for i in range(len(models)):
#        for j in range(len(samplings)):
#            for k in range(len(perturbations)):
#                pass
#    return
#
#
#def test_scores(estimator, X, y, cv, scorer):
#    """Evaluate the scores of a cross-validated with perturbations.
#
#    Parameters
#    ----------
#    estimator: estimator object implementing 'fit'
#        The object to use to fit the data.
#    X: array-like of shape at least 2D
#        The data to fit.
#    y: array-like
#        The target variable to try to predict in the case of
#        supervised learning.
#    cv: integer or cross-validation generator, optional
#        If an integer is passed, it is the number of fold (default 3).
#        Specific cross-validation objects can be passed, see
#        sklearn.cross_validation module for the list of possible objects.
#    scorer: string, callable or None, optional, default: None
#        A string (see model evaluation documentation) or a scorer callable
#        object / function with signature ``scorer(estimator, X, y)``.
#
#    """
#    score = _test_scores(clone(estimator), X, y, cv, scorer)
#    return score
#
#
#def peturbation_test_scores(estimator, X, y, X_pert, cv, scorer, n_jobs=1, verbose=0):
#    """Evaluate the scores of a cross-validated with perturbations.
#
#    Parameters
#    ----------
#    estimator: estimator object implementing 'fit'
#        The object to use to fit the data.
#    X: array-like of shape at least 2D
#        The data to fit.
#    y: array-like
#        The target variable to try to predict in the case of
#        supervised learning.
#    X_pert: list
#        the list of perturbated data.
#    cv: integer or cross-validation generator, optional
#        If an integer is passed, it is the number of fold (default 3).
#        Specific cross-validation objects can be passed, see
#        sklearn.cross_validation module for the list of possible objects.
#    scorer: string, callable or None, optional, default: None
#        A string (see model evaluation documentation) or a scorer callable
#        object / function with signature ``scorer(estimator, X, y)``.
#    n_jobs: integer, optional (default=1)
#        The number of CPUs to use to do the computation. -1 means
#        'all CPUs'.
#    verbose: integer, optional (default=0)
#        The verbosity level.
#
#    """
#    n_pert = len(X_pert)
#    score = _test_scores(clone(estimator), X, y, cv, scorer)
#    perturbation_scores = Parallel(n_jobs=n_jobs, verbose=verbose)(
#        delayed(_test_scores)(clone(estimator), X_pert[i], y, cv, scorer)
#        for i in range(n_pert))
##    pvalue = (np.sum(perturbation_scores >= score) + 1.0) / (n_pert + 1)
#    return score, perturbation_scores
#
#
#def _test_scores(estimator, X, y, cv, scorer):
#    """Evaluate the scores of a cross-validated with perturbations.
#
#    Parameters
#    ----------
#    estimator: estimator object implementing 'fit'
#        The object to use to fit the data.
#    X: array-like of shape at least 2D
#        The data to fit.
#    y: array-like
#        The target variable to try to predict in the case of
#        supervised learning.
#    cv: integer or cross-validation generator, optional
#        If an integer is passed, it is the number of fold (default 3).
#        Specific cross-validation objects can be passed, see
#        sklearn.cross_validation module for the list of possible objects.
#    scorer: callable object
#        A scorer callable object / function with signature
#        ``scorer(estimator, X, y)``.
#
#    Returns
#    -------
#    score : float
#        The true score without permuting targets.
#    permutation_scores : array, shape = [n_permutations]
#        The scores obtained for each perturbations.
#
#    """
#    avg_score = []
#    for train, test in cv:
#        estimator.fit(X[train], y[train])
#        avg_score.append(scorer(estimator, X[test], y[test]))
#    return np.mean(avg_score)
