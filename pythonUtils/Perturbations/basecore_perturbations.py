
"""
Core Perturbations
------------------
Group of core perturbations.

"""

import numpy as np
from auxiliar_functions import check_int


class BaseCoreArray(object):
    """BaseCore for array type data.

    Parameters
    ----------
    _global_computation: boolean
        if it is needed the whole data to compute the perturbation.
    _predefined_k: boolean
        if we have to pre-define the number of perturbations we want to use or
        it could be set on the fly.

    """

    ##################### Main  administrative functions ######################
    ###########################################################################
    def _check_array(self, array):
        "Check proper format of array."
        assert(type(array) == np.ndarray)
        assert(len(array.shape) == 2)

    def _assert_set(self):
        assert(self.is_set())


class BaseCorePermutationArray(BaseCoreArray):
    """Class to apply permutation.
    """
    _global_computation = True
    _predefined_k = True

    def __init__(self, reindices=None, auto=True):
        self.set_parameters(reindices, auto)

    ##################### Main  administrative functions ######################
    ###########################################################################
    def is_set(self):
        return ('reindices' in dir(self)) and (self.reindices is not None)

    def set_parameters(self, reindices=None, auto=True):
        if reindices is not None:
            self._format_reindices(reindices, auto)

    def get_parameters(self):
        return self.reindices,

    def _check_and_format_inputs(self, array, k, args):
        self._check_array(array)
        try:
            self._assert_set()
        except:
            if len(args):
                self.set_parameters(*args)
            else:
                self.set_data(array)
            self._assert_set()
        assert(type(k) == list)

    def _format_reindices(self, reindices, auto=True):
        """Format reindices.

        Parameters
        ----------
        reindices: np.ndarray or tuple
            the reindices to apply permutation perturbations.

        """
        if type(reindices) == np.ndarray:
            self.k_perturb = reindices.shape[1]
            self.reindices = reindices
        elif type(reindices) == tuple:
            n, k_perturb = reindices
            if check_int(n) and check_int(k_perturb):
                auto_indices = [np.arange(n)] if auto else []
                self.k_perturb = k_perturb
                n_pert = k_perturb-1 if auto else k_perturb
                self.reindices = np.vstack(auto_indices +
                                           [np.random.permutation(n)
                                            for i in xrange(n_pert)]).T

    ############################# Main functions ##############################
    ###########################################################################
    def apply(self, array, k, *args):
        """Apply the permutation perturbation to the given array.

        Parameters
        ----------
        array: np.ndarray
            the array we want to perturbate.
        k: list
            the indices list of perturbations.

        Returns
        -------
        array_p: np.ndarray, shape (n, nfeats, k_perturb)
            the perturbated array.

        """
        self._check_and_format_inputs(array, k, args)
        array_p = array[self.reindices[:, k]].swapaxes(1, 2)
        return array_p

    def set_data(self, data):
        """Set the main parameters computed from the data.

        Parameters
        ----------
        data: np.ndarray
            the data we want to perturbate.

        """
        self._check_array(data)


class BaseCoreJitteringArray(BaseCoreArray):
    """Class to apply gaussian jittering to data.
    """
    _global_computation = False
    _predefined_k = False

    def __init__(self, stds=None):
        self.set_parameters(stds)

    ##################### Main  administrative functions ######################
    ###########################################################################
    def is_set(self):
        return ('_stds' in dir(self)) and (self._stds is not None)

    def set_parameters(self, stds=None):
        if stds is not None:
            self._stds = stds

    def get_parameters(self):
        return self._stds,

    def _check_and_format_inputs(self, array, k, args):
        self._check_array(array)
        try:
            self._assert_set()
        except:
            if len(args):
                self.set_parameters(*args)
            else:
                self.set_data(array)
            self._assert_set()
        assert(type(k) == list)

    ############################# Main functions ##############################
    ###########################################################################
    def apply(self, array, k, *args):
        """Apply the jittering perturbation to the given array.

        Parameters
        ----------
        array: np.ndarray
            the array we want to perturbate.
        k: list
            the indices list of perturbations.

        Returns
        -------
        array_p: np.ndarray, shape (n, nfeats, k_perturb)
            the perturbated array.

        """
        ## Check inputs
        self._check_and_format_inputs(array, k, args)

        ## Compute perturbations
        sh = len(array), len(array[0]), len(k)
        marg_array = self._compute_marginal_jittering(sh)
        array_p = self._apply_marginal_jittering(array, marg_array)
        return array_p

    ##################### Specific functions of jittering #####################
    ###########################################################################
    def set_data(self, data):
        """Set the main parameters computed from the data.

        Parameters
        ----------
        data: np.ndarray
            the data we want to perturbate.

        """
        self._check_array(data)
        if not self.is_set():
            self._stds = data.std(0)

    def _compute_marginal_jittering(self, sh):
        """Compute marginal jittering.

        Parameters
        ----------
        sh: tuple
            the shape of the marginal jittering.

        Returns
        -------
        marg_array: np.ndarray, shape (n, nfeats, k_perturb)
            the marginal jitter perturbations.

        """
        jitter_d = np.random.random(sh).swapaxes(1, 2)
        marg_array = np.multiply(jitter_d, self._stds).swapaxes(1, 2)
        return marg_array

    def _apply_marginal_jittering(self, array, marg_array):
        """Add marginal jittering to the real array.

        Parameters
        ----------
        array: np.ndarray, shape (n, nfeats)
            the array we want to perturbated.
        marg_array: np.ndarray, shape (n, nfeats, k_perturb)
            the marginal jitter perturbations.

        Returns
        -------
        array_p: np.ndarray
            the perturbated array.

        """
        array_p = np.add(marg_array, array[:, :, None])
        return array_p


class BaseCoreSubstitutionArray(BaseCoreArray):
    """Base class to apply substitution perturbations in a discrete type
    data.
    """
    _global_computation = True
    _predefined_k = True

    def __init__(self, probs=None, labels=None):
        self.set_parameters(probs, labels)

    ##################### Main  administrative functions ######################
    ###########################################################################
    def is_set(self):
        logi_probs = ('probs' in dir(self)) and (self.probs is not None)
        logi_labels = ('labels' in dir(self)) and (self.labels is not None)
        return logi_probs and logi_labels

    def set_parameters(self, probs, labels=None):
        if probs is None:
            self.probs = None
            self.labels = None
        else:
            if type(probs) != list:
                assert(type(probs) == np.ndarray)
                if np.all(probs.sum(1) != 1):
                    raise TypeError("Not correct probs input.")
                if probs.shape[0] != probs.shape[1]:
                    raise IndexError("Probs is noot a square matrix.")
                probs = [probs.cumsum(1)]
            else:
                assert(all([type(p) == np.ndarray for p in probs]))
                assert(all([p.shape[0] == p.shape[1] for p in probs]))
                if all([all(np.round(p.sum(1)) == 1) for p in probs]):
                    probs = [p.cumsum(1) for p in probs]
                else:
                    assert(all([all(np.round(p[:, -1], 1) == 1.)
                                for p in probs]))
            self.probs = probs
            if labels is None:
                self.labels = [np.arange(len(probs[i]))
                               for i in range(len(probs))]
            else:
                if type(labels) == np.ndarray:
                    labels = [labels]
                assert(len(labels) == len(self.probs))
                assert(all([len(labels[i]) == len(self.probs[i])
                            for i in range(len(labels))]))
                self.labels = labels

    def get_parameters(self):
        return self.probs, self.labels

    def _check_and_format_inputs(self, array, k, args):
        ## Check inputs
        self._check_array(array)
        try:
            self._assert_set()
        except:
            if len(args):
                self.set_parameters(*args)
            else:
                self.set_data(array)
            self._assert_set()
        assert(type(k) == list)

    ############################# Main functions ##############################
    ###########################################################################
    def apply(self, array, k, *args):
        """Apply the substitution perturbation to the given array.

        Parameters
        ----------
        array: np.ndarray
            the array we want to perturbate.
        k: list
            the indices list of perturbations.

        Returns
        -------
        array_p: np.ndarray, shape (n, nfeats, k_perturb)
            the perturbated array.

        """
        ## Check inputs
        self._check_and_format_inputs(array, k, args)
        ## Compute perturbations
        array_p = []
        for i in range(len(array[0])):
            array_p.append(self._compute_substitution(array[:, [i]], k,
                                                      self.probs[i],
                                                      self.labels[i]))
        array_p = np.concatenate(array_p, axis=1)
        return array_p

    def set_data(self, data):
        """Set the main parameters computed from the data.

        Parameters
        ----------
        data: np.ndarray
            the data we want to perturbate.

        """
        self._check_array(data)
        ## Set random substitution
        if not self.is_set():
            probs, labels = [], []
            for i in range(len(data[0])):
                labels_i = np.unique(data[:, i])
                n_l = len(labels_i)
                probs_i = np.ones((n_l, n_l)).astype(float)/n_l
                probs.append(probs_i.cumsum(1))
                labels.append(labels_i)
            self.probs = probs
            self.labels = labels

    ################### Specific functions of substitution ####################
    ###########################################################################
    def _compute_substitution(self, array, k, probs, labels):
        categories = np.unique(array)
        if len(categories) > len(probs):
            msg = "Not matching dimension between probs and array."
            raise IndexError(msg)
        ## Compute each change
        array_p = np.zeros((len(array), 1, len(k))).astype(int)
        for i_k in range(len(k)):
            for i in xrange(len(array)):
                r = np.random.random()
                idx = np.where(array[i] == labels)[0]
                idx2 = np.where(probs[idx].ravel() > r)[0][0]
                array_p[i, 0, i_k] = labels[idx2]
        return array_p
