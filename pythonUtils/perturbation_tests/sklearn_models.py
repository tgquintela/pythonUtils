
"""
Sklearn models
--------------
Group of utitilies to explore the application of different perturbations and
sklearn models.

"""

import copy
import multiprocessing
from joblib import Parallel, delayed
import time


class Sklearn_permutation_test:
    """The class to operate perturbation tests using sklearn models as a dummy
    models.
    """
    def _initialization(self):
        self._num_cores = 1
        self._times_processes = 0

    def __init__(self, num_cores):
        self._initialization()
        # Number of cores setting
        if num_cores is None:
            self._num_cores = multiprocessing.cpu_count()
        elif num_cores == 0:
            self._num_cores = 1
        else:
            self._num_cores = num_cores
        # Setting computation function
        if self._num_cores == 1:
            self.compute = self._compute_sequencial
        else:
            self.compute = self._compute_parallel

    def _compute_parallel(self, X, y, parameters):
        """Computation of the scores for each case in a paraellel fashion."""
        ## 0. Creation of the parallelizable function and parameters
        f_compute_paral = create_f_compute_paral(X, y)
        global f_compute_paral
        parameters_paral = [([parameters[0][i]], parameters[1],
                             parameters[2], parameters[3], parameters[4])
                            for i in range(len(parameters[0]))]
        num_cores = self._num_cores
        ## 1. Parallel computation
        results = Parallel(n_jobs=num_cores)(delayed(f_compute_paral)(par)
                                             for par in parameters_paral)
        ## 2. Reconstruction of results
        scores, best_pars_info, times = rebuild_paral_results(results)
        self._times_processes = times
        return scores, best_pars_info

    def _compute_sequencial(self, X, y, parameters):
        """Computation of the scores for each case in a sequencial fashion."""
        ## 1. Computation
        scores, best_pars_info, times =\
            application_sklearn_models(X, y, parameters)
        ## 2. Formatting results
        self._times_processes = times
        return scores, best_pars_info


###############################################################################
############################# Scikit-based models #############################
###############################################################################
def application_sklearn_models_paral(X, y, parameters, num_cores=1):
    """Application of models computation for direct models in a parallel way.

    Parameters
    ----------
    X: np.ndarray
        the features information.
    y: np.ndarray
        the labels information.
    parameters: tuple
        the tuple of list of parameters for each type of information we need.
        That information is summarized in the possible values for:
            * Perturbations of data to test robustness.
            * Models to apply.
            * Samplings to compute scores.
            * Scorer function to compute performance of the model.
    num_cores: int (default=1)
        the number of cores we want to use to parallelize the computations.

    Returns
    -------
    scores: list
        the list of best scores for each possible combination.
    best_pars_info: list
        the list of best model parameters for each possible combination.

    """
    ## 0. Creation of the parallelizable function and parameters
    f_compute_paral = create_f_compute_paral(X, y)
    global f_compute_paral
    parameters_paral = [([parameters[0][i]], parameters[1],
                         parameters[2], parameters[3], parameters[4])
                        for i in range(len(parameters[0]))]
    ## 1. Parallel computation
    results = Parallel(n_jobs=num_cores)(delayed(f_compute_paral)(par)
                                         for par in parameters_paral)
    ## 2. Reconstruction of results
    scores, best_pars_info, times = rebuild_paral_results(results)
    return scores, best_pars_info


def application_sklearn_models(X, y, parameters):
    """Application of models computation for direct models.

    Parameters
    ----------
    X: np.ndarray
        the features information.
    y: np.ndarray
        the labels information.
    parameters: tuple
        the tuple of list of parameters for each type of information we need.
        That information is summarized in the possible values for:
            * Perturbations of data to test robustness.
            * Format data (data reduction or other types)
            * Models to apply.
            * Samplings to compute scores.
            * Scorer function to compute performance of the model.

    Returns
    -------
    scores: list
        the list of best scores for each possible combination.
    best_pars_info: list
        the list of best model parameters for each possible combination.
    times: list
        the times it was spended for each task.

    """
    ## 0. Parameters and initialization
    perturbations_info, format_info, models_info = parameters[:3]
    samplings_info, scorer_info = parameters[3:]
    scores, best_pars_info, times = [], [], []

    # For perturbations
    for i in range(len(perturbations_info)):
        # Computation results
        X_p, y_p = apply_perturbation(X, y, perturbations_info[i])
        scores_i, best_pars_info_i, times_i =\
            scores_sklearn_computation_comb(X_p, y_p, format_info, models_info,
                                            samplings_info, scorer_info)
        # Storage results
        scores.append(scores_i)
        best_pars_info.append(best_pars_info_i)
        times.append(times_i)
    return scores, best_pars_info, times


def scores_sklearn_computation_comb(X, y, format_info, models_info,
                                    samplings_info, scorer_info):
    """Scores over models, samplings and scorer functions.

    Parameters
    ----------
    X: np.ndarray
        the features information.
    y: np.ndarray
        the labels information.
    format_info: list
        the list of possible formatters of data.
    models_info: list
        the list of possible models to apply to the data.
    samplings_info: list
        the list of possible sampling information to apply.
    scorer_info: list
        the list of possible scorers to compute the performance of the models.

    Returns
    -------
    scores: list
        the list of best scores for each possible combination.
    best_pars_info: list
        the list of best model parameters for each possible combination.

    """
    scores, best_pars, times = [], [], []
    for i in range(len(format_info)):
        scores_i, best_pars_i, times_i = [], [], []
        X_new, y_new = apply_format(X, y, format_info[i])
        for j in range(len(models_info)):
            ## Initializatin for each model
            scores_j, best_pars_j, times_j = [], [], []
            model_j, pos_pars_j = create_sklearn_model(models_info[j])
            for k in range(len(samplings_info)):
                ## Creation of the cross-validation
                cv = create_cv(samplings_info[k], X_new, y_new)
                ## Model computation
                score, pars, ts = apply_sklearn_model(X_new, y_new, model_j,
                                                      pos_pars_j, cv,
                                                      scorer_info)
                ## Storage
                scores_j.append(score)
                best_pars_j.append(pars)
                times_j.append(ts)
            ## Storage
            scores_i.append(scores_j)
            best_pars_i.append(best_pars_j)
            times_i.append(times_j)
        ## Storage
        scores.append(scores_i)
        best_pars.append(best_pars_i)
        times.append(times_i)

    return scores, best_pars, times


###############################################################################
######################## Apply the instances processes ########################
###############################################################################
def apply_sklearn_model(X, y, model_j, pos_pars_j, cv, scorers):
    """Application of the sklearn model using cross-validation cv and the
    given scorer."""
    scores, t00 = [], time.time()
    for train, test in cv:
        ## Training model
        model_j.fit(X[train], y[train])
        ## Predicting labels
        y_pred = model_j.predict(X[test])
        ## Scores
        scores_i = []
        for i in range(len(scorers)):
            scorer = create_scorer(scorers[i])
            # Storage
            scores_i.append(scorer(y[test], y_pred))
        ## Storage
        scores.append(scores_i)
    ts = time.time()-t00
    return scores, pos_pars_j, ts


def apply_perturbation(X, y, perturbations_info):
    """Application of the perturbations."""

    perturb = perturbations_info[3](X, None, perturbations_info[1],
                                    perturbations_info[2])
    X_p, y_p = perturb.apply2features(copy.copy(X)).squeeze(2), copy.copy(y)
    return X_p, y_p


def apply_format(X, y, format_info):
    """Application of format of data."""
    X_new, y_new = format_info[3](X, y, format_info[1], format_info[2])
    return X_new, y_new


###############################################################################
#################### Create and instantiate objects process ###################
###############################################################################
def create_format(X, y, format_info):
    """"""
    formatter = format_info[1](**format_info[2])
    return formatter


def create_sklearn_model(model_info):
    """Instantiate model from model_info."""
    model_name, model_class, model_pars = model_info
    model = model_class(**model_pars)
    return model, model_pars


def create_scorer(scorer_info):
    """Create the scorer to measure how good is the prediction."""
    scorer = scorer_info[3](scorer_info[1], **scorer_info[2])
    return scorer


def create_cv(cv_info, X, y):
    """Create the cv."""
    ## TODO: Only for kFold
    cv_pars = copy.copy(cv_info[2])
    cv_pars['n'] = len(X)
    cv = cv_info[1](**cv_pars)
    return cv


def create_perturbation(pert_info):
    """Create perturbation."""
    pert = pert_info[3](pert_info[1], None, pert_info[2])
    return pert


###############################################################################
########################### Parallel tools functions ##########################
###############################################################################
def rebuild_paral_results(results):
    """Rebuild the correct way to store the results."""
    scores = [results[i][0] for i in range(len(results))]
    best_pars_info = [results[i][1] for i in range(len(results))]
    times = [results[i][2] for i in range(len(results))]
    return scores, best_pars_info, times


def create_f_compute_paral(X, y):
    """Creation of the function to compute in parallel fashion the scores given
    the parameters."""
    def f_compute_paral(pars):
        """Function which computes the different scores of sklearn models given
        the parameters."""
        return application_sklearn_models(copy.copy(X), copy.copy(y), pars)
    return f_compute_paral
