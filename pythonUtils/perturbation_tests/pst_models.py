
"""
pst models
----------
Group of utitilies to explore the application of different perturbations and
pst models.

"""

import copy
import multiprocessing
from joblib import Parallel, delayed
import time
import numpy as np

#
#x, y = np.arange(100), np.arange(6)
#x_i, y_i = np.random.randint(0, 100), np.random.randint(0, 6)
#locs = np.random.random((100, 2))
#regs = np.random.randint(0, 20, 100)
#feat = np.random.random((100, 5, 6))
#
#from pySpatialTools.Retrieve import KRetriever
#from pySpatialTools.Retrieve.tools_retriever import create_aggretriever, avgregionlocs_outretriever
#
#retriever_in = (KRetriever, {'info_ret': 1, 'bool_input_idx': True})
#retriever_out = (KRetriever, {'info_ret': 1, 'bool_input_idx': True, 'input_map': regs})
#aggregating = avgregionlocs_outretriever, (avgregionlocs_outretriever, )
#aggregation_info = (locs, regs), retriever_in, retriever_out,\
#    (avgregionlocs_outretriever, (avgregionlocs_outretriever, ))
#
#agg_info = retriever_in, retriever_out, aggregating
#
### For each regs, extract locs (por encima set mapper_selectors)
#retriever_in, retriever_out, aggregating = agg_info
#retriever_out['input_map'] = regs
#disc_info = (locs, regs)
#aggregation_info = disc_info, retriever_in, retriever_out, aggregating
#
#
#def apply_aggregation(self, regs, agg_info):
#    locs, feats_obj = self._extract_main_info()
#
#    if len(regs.shape) == 1:
#        regs = regs.reshape((len(regs), 1))
#    for i in range(regs.shape[1]):
#        retriever_in, retriever_out, aggregating = copy.copy(agg_info)
#        retriever_out['input_map'] = regs[:, i]
#        disc_info = (locs, regs[:, i])
#        agg_info_i = disc_info, retriever_in, retriever_out, aggregating
#        self._format_aggregations(agg_info_i, i_r=(0, 0))


class Pst_permutation_test:
    """The class to operate perturbation tests using pst models as a dummy
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

    def _compute_parallel(self, X, y, locs, regs_info, parameters):
        """Computation of the scores for each case in a paraellel fashion."""
        ## 0. Creation of the parallelizable function and parameters
        f_compute_paral = create_f_compute_paral(X, y, locs)
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

    def _compute_sequencial(self, X, y, locs, regs_info, id_info, parameters):
        """Computation of the scores for each case in a sequencial fashion."""
        ## 1. Computation
        scores, best_pars_info, times =\
            application_pst_time_models(X, y, locs, regs_info, id_info,
                                        parameters)
        ## 2. Formatting results
        self._times_processes = times
        return scores, best_pars_info


###############################################################################
############################### Pst-based models ##############################
###############################################################################
def application_pst_time_models(X, y, locs, regs_i, id_i, parameters):
    """Application of the pySpatialTools models for time."""
    ## 0. Parameters and initialization
    perturbations_info, format_info, models_info = parameters[:3]
    samplings_info, scorer_info = parameters[3:]
    scores, best_pars_info, times = [], [], []

    # For perturbations
    for i in range(len(perturbations_info)):
        # Apply perturbation
        X_p, y_p, locs_p, regs_p =\
            apply_pst_time_perturbation(X, y, locs, regs_i, id_i,
                                        perturbations_info[i])
        # Compute combinations
        scores_i, best_pars_info_i, times_i =\
            scores_pst_time_computation_comb(X_p, y_p, locs_p, regs_p,
                                             format_info, models_info,
                                             samplings_info, scorer_info)
        # Storage results
        scores.append(scores_i)
        best_pars_info.append(best_pars_info_i)
        times.append(times_i)
    return scores, best_pars_info, times


def scores_pst_time_computation_comb(X, y, locs, regs, format_info,
                                     models_info, samplings_info, scorer_info):
    """Application of the models to predict."""
    scores, best_pars, times = [], [], []
    for i in range(len(format_info)):
        scores_i, best_pars_i, times_i = [], [], []
        X_new, y_new, locs_new, regs_new =\
            apply_format(X, y, locs, regs, format_info[i])
        for j in range(len(models_info)):
            ## Initializatin for each model
            scores_j, best_pars_j, times_j = [], [], []
            model_j, pos_pars_j =\
                create_pst_time_model(X_new, locs_new, regs_new,
                                      models_info[j])
            for k in range(len(samplings_info)):
                ## Creation of the cross-validation
                cv = create_cv(samplings_info[k], X_new, y_new,
                               locs_new, regs_new)
                ## Model computation
                score, pars, ts = apply_pst_time_model(y_new, model_j,
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
def apply_pst_time_model(y, model_j, pos_pars_j, cv, scorers):
    """Application of the sklearn model using cross-validation cv and the
    given scorer."""
    scores, t00 = [], time.time()
    for train, test in cv:
        ## Training model
        model_j.fit(train, y[train])
        ## Predicting labels
        y_pred = model_j.predict(test)
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


def apply_pst_time_perturbation(X, y, locs, regs, id_i, perturbations_info):
    """Application of the perturbations."""
    ## 0. Parse inputs
    _, _, _, _, perturb_flag = perturbations_info
    print type(X), type(X[0])
    print X[0].shape
    ## Select
    if perturb_flag:
        X_p, y_p, locs_p, regs_p =\
            apply_unique_id_perturb(X, y, locs, regs, id_i, perturbations_info)
    else:
        X_p, y_p, locs_p, regs_p =\
            apply_independent_perturb(X, y, locs, regs, id_i,
                                      perturbations_info)
    return X_p, y_p, locs_p, regs_p


def apply_independent_perturb(X, y, locs, reg, id_i, perturbations_info):
    """Apply independent perturbation for each time.
    """
    ## 0. Parse inputs
    _, perturb_obj, perturb_pars, perturb_f, _ = perturbations_info
    if type(perturb_obj) == list:
        assert(len(perturb_obj) == len(X))
        if type(perturb_pars) != list:
            perturb_pars = [perturb_pars for i in range(len(X))]
    ## 1. Computation
    X_p, y_p, locs_p, regs_p = [], [], [], []
    for i in range(len(X)):
        if type(perturb_obj) == list:
            perturb = perturb_f(X[i], locs[i], reg[i], perturb_obj[i],
                                perturb_pars[i])
        else:
            perturb = perturb_f(X[i], locs[i], reg[i], perturb_obj,
                                perturb_pars)
        ## Computation
        X_pi, y_pi, locs_pi, regs_pi =\
            apply_independent_perturb_i(X[i], y[i], locs[i], reg[i],
                                        id_i[i], perturb)
        ## Storage
        X_p.append(X_pi)
        y_p.append(y_pi)
        locs_p.append(locs_pi)
        regs_p.append(regs_pi)
    return X_p, y_p, locs_p, regs_p


def apply_independent_perturb_i(X, y, locs, regs, id_i, perturb):
    """Apply independent perturb for a specific selected time.
    """
    X_p, y_p = perturb.apply2features(copy.copy(X)).squeeze(2), copy.copy(y)
    locs_p = perturb.apply2locations(copy.copy(locs))
    if perturb._perturbtype == 'element_permutation':
        regs_p = perturb.apply2features(copy.copy(regs))
    else:
        regs_p = copy.copy(regs)

    return X_p, y_p, locs_p, regs_p


def apply_unique_id_perturb(X, y, locs, reg, id_i, perturbations_info):
    """TODO"""
    _, perturb_obj, perturb_pars, perturb_f, _ = perturbations_info
    perturb = perturb_f(perturb_obj, perturb_pars)
    assert(perturb._perturbtype == 'element_permutation')
    ids_u = np.unique(np.concatenate(id_i))
    ids_rei = perturb.apply2indices(copy.copy(ids_u))
    reindices = []
    for i in range(len(X)):
        reindices_i = []
        for u in range(len(ids_u)):
            rei = np.where(id_i == ids_u[u])[0]
            if rei:
                reindices_i.append(rei)
        reindices.apend(reindices_i)
    ## TODO
    return X_p, y_p, locs_p, regs_p


def apply_format(X, y, locs, regs, format_info):
    """Application of format of data."""
    _, format_obj, format_pars, format_f = format_info
    X_nf, y_nf, locs_nf, regs_nf = [], [], [], []
    for i in range(len(X)):
        X_new, y_new, locs_new, regs_new = format_f(X, y, locs, regs,
                                                    format_obj, format_pars)
        X_nf.append(X_new)
        y_nf.append(y_new)
        locs_nf.append(locs_new)
        regs_nf.append(regs_new)
    return X_nf, y_nf, locs_nf, regs_nf


###############################################################################
#################### Create and instantiate objects process ###################
###############################################################################
#def create_sklearn_model(model_info):
#    """Instantiate model from model_info."""
#    model_name, model_class, model_pars = model_info
#    model = model_class(**model_pars)
#    return model, model_pars


def create_pst_time_model(X, locs, regs, models_info):
    """Instantiate model from model_info."""
    model_name, f_model, spdesc_info, agg_info, selectors = models_info

    model = f_model(locs, X, spdesc_info)
    model.apply_aggregation(regs, agg_info, selectors)

    spdesc_pars = extract_spdesc_pars(spdesc_info)
    model_pars = {'model_name': model_name, 'spdesc_info': spdesc_pars}

    return model, model_pars


def create_scorer(scorer_info):
    """Create the scorer to measure how good is the prediction."""
    scorer = scorer_info[3](scorer_info[1], **scorer_info[2])
    return scorer


def create_cv(samplings_info, X, y, locs, regs):
    """Create the cv for the temporal spatial data."""
    ## TODO: Only for kFold_ts
    cv_pars = copy.copy(samplings_info[2])
    cv_pars['n'] = [len(X[i]) for i in range(len(X))]
    cv = samplings_info[1](**cv_pars)
    return cv



#
#def create_scorer(scorer_info):
#    """Create the scorer to measure how good is the prediction."""
#    scorer = scorer_info[3](scorer_info[1], **scorer_info[2])
#    return scorer
#
#
#def create_cv(cv_info, X, y):
#    """Create the cv."""
#    ## TODO: Only for kFold
#    cv_pars = copy.copy(cv_info[2])
#    cv_pars['n'] = len(X)
#    cv = cv_info[1](**cv_pars)
#    return cv
#
#
#def create_perturbation(pert_info):
#    """Create perturbation."""
#    pert = pert_info[3](pert_info[1], None, pert_info[2])
#    return pert


###############################################################################
########################### Parallel tools functions ##########################
###############################################################################
def rebuild_paral_results(results):
    """Rebuild the correct way to store the results."""
    scores = [results[i][0] for i in range(len(results))]
    best_pars_info = [results[i][1] for i in range(len(results))]
    times = [results[i][2] for i in range(len(results))]
    return scores, best_pars_info, times


def create_f_compute_paral(X, y, locs, id_i, times_i, regs_i):
    """Creation of the function to compute in parallel fashion the scores given
    the parameters."""
    def f_compute_paral(pars):
        """Function which computes the different scores of sklearn models given
        the parameters."""
        return application_pst_time_models(copy.copy(X), copy.copy(y),
                                           copy.copy(locs), id_i, times_p,
                                           regs_i, pars)
    return f_compute_paral

#
#
#class Filterer_(object):
#    """
#    """
#
#    def _format_inputs(self, indices, y=None):
#        """Format input indices.
#
#        Parameters
#        ----------
#        indices: np.ndarray or list
#            the indices of the samples used to compute the model.
#        y: np.ndarray or list
#            the target we want to predict.
#
#        Returns
#        -------
#        indices: np.ndarray or list
#            the indices of the samples used to compute the model.
#        y: np.ndarray or list
#            the target we want to predict.
#
#        """
#
#        ## 0. Create indices formatting
#        assert(np.max(indices) <= np.sum(self.n_inputs))
#        ranges = np.cumsum([0]+self.n_inputs)
#        spdesc_i, spdesc_k = -1*np.ones(len(indices)), -1*np.ones(len(indices))
#        spdesc_i, spdesc_k = spdesc_i.astype(int), spdesc_k.astype(int)
#        new_indices, new_y = [], []
#        for i in range(len(self.n_inputs)):
#            logi = np.logical_and(ranges[i] >= indices, ranges[i+1] <= indices)
#            spdesc_k[logi] = i
#            spdesc_i[logi] = indices-ranges[i]
#            new_indices.append(indices[logi])
#            if y is not None:
#                new_y.append(y[logi])
#        self._spdesc_i = spdesc_i.astype(int)
#        self._spdesc_k = spdesc_k.astype(int)
#        ## 1. Output format
#        if y is None:
#            return new_indices
#        else:
#            return new_indices, new_y
#
#    def _format_output(self, y_pred):
#        """Re-format the output in the same way the input is given.
#
#        Parameters
#        ----------
#        y_pred: list
#            the list of prediction targets for each possible spatial model.
#
#        Return
#        ------
#        y_pred: np.ndarray
#            the target predictions in the format is given the input.
#
#        """
#        n = sum([len(e) for e in y_pred])
#        new_y_pred = np.zeros(n)
#        for i in range(len(y_pred)):
#            logi = self._spdesc_k == i
#            assert(len(y_pred[i]) == np.sum(logi))
#            new_y_pred[logi] = y_pred[i]
#        return new_y_pred
#
#
#
#
#
#
#
#
#
#
#
#
#def application_pst_models(pfeatures, qvalue, loc_ref, year_ref, regions,
#                           parameters):
#    """Application of models computation for direct spatial-based models.
#
#    Parameters
#    ----------
#    pfeatures: np.ndarray
#        the element features information.
#    qvalue: np.ndarray
#        the value to predict.
#    loc_ref: np.ndarray
#        the locations of the nif activate.
#    year_ref: np.ndarray
#        the years actived of each location.
#    parameters: tuple
#        the parameters which descriptves the model we want to apply.
#        That information is summarized in the possible values for:
#            * The information for create the pySpatialTools descriptors.
#            * The sampling, perturbation and sklearn model information possible
#            to test.
#
#    Returns
#    -------
#    scores: list
#        the list of best scores for each possible combination.
#    best_pars_info: list
#        the list of best model parameters for each possible combination.
#
#    """
#    # Initialization
#    parameters =\
#        create_pstmodel_parameters(loc_ref, year_ref, pfeatures, parameters)
##    names_comb = names_parameters_computation(parameters)
#    models_info, samplings_info, perturbations_info, scorer_info = parameters
#    models_j = create_pst_models(loc_ref, pfeatures, qvalue, parameters)
#
#
#    pfeatures, qvalue, loc_ref, year_ref, regions
#
#    ## Computation of scores and parameters by combination of possibilities
#    # Computation results
#    scores, best_pars_info, ts =\
#        scores_pst_computation_comb(pfeatures, qvalue, loc_ref, models_j,
#                                    samplings_info, scorer_info)
#
#    return scores, best_pars_info, ts
#
#
#def scores_pst_computation_comb(X, y, locs, models_info, samplings_info,
#                                scorer_info):
#    """Scores over models, samplings and scorer functions.
#
#    Parameters
#    ----------
#    X: np.ndarray
#        the features information.
#    y: np.ndarray
#        the labels information.
#    models_info: list
#        the list of possible models to apply to the data.
#    samplings_info: list
#        the list of possible sampling information to apply.
#    scorer_info: list
#        the list of possible scorers to compute the performance of the models.
#
#    Returns
#    -------
#    scores: list
#        the list of best scores for each possible combination.
#    best_pars_info: list
#        the list of best model parameters for each possible combination.
#
#    """
#    scores, best_pars, times = [], [], []
#    for k in range(len(samplings_info)):
#        ## Creation of the cross-validation
#        cv = create_cv(samplings_info[k], X, y, locs)
#        ## Model computation
#        score, pars, ts =\
#            apply_pst_model(y, models_info, pos_pars_j, cv, scorer_info)
#        ## Storage
#        scores.append(score)
#        best_pars.append(pars)
#        times.append(ts)
#
#    return scores, best_pars, times
#
#
#def apply_pst_model(y, model_j, pos_pars_j, cv, scorers):
#    """Application of the sklearn model using cross-validation cv and the
#    given scorer."""
#    scores, t00 = [], time.time()
#    for train, test in cv:
#        ## Training model
#        model_j.fit(train, y[train])
#        ## Predicting labels
#        y_pred = model_j.predict(test)
#        ## Scores
#        scores_i = []
#        for i in range(len(scorers)):
#            scorer = create_scorer(scorers[i])
#            # Storage
#            scores_i.append(scorer(y[test], y_pred))
#        ## Storage
#        scores.append(scores_i)
#    ts = time.time()-t00
#    return scores, pos_pars_j, ts
