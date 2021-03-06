
"""
Perturbations
-------------
Module oriented to perform a perturbation of the system in order to carry out
with statistical testing of models.
The main function of this module is grouping functions which are able to
change the system to other statistically probable options in order to explore
the sample space.


TODO
----
-Aggregation perturbation:
--- Discretization perturbed.
--- Fluctuation of features between borders.
- Fluctuation of borders
--- Fluctuation of edge points
--- Fluctuation over sampling points

set functions for individual perturbations in jitter type.

"""


import numpy as np
from auxiliar_functions import check_int
from basecore_perturbations import BaseCorePermutationArray,\
    BaseCoreJitteringArray, BaseCoreSubstitutionArray


############################## Globals definition #############################
###############################################################################


###############################################################################
############################ Location perturbation ############################
###############################################################################
class BasePerturbation(object):
    """General perturbation. It constains default functions for perturbation
    objects.
    """

    ########################## Base class functions ###########################
    def _initialization(self):
        ## Main classes stored
        self.basecoreperturbation = []
        self.k_perturb = 1
        ## Main precomputed data stored
        self.locations_p = None
        self.features_p = None
        self.relations_p = None
        self.discretizations_p = None
        ## Ensure correctness
        self.assert_correctness()

    def _format_k_perturb(self, k):
        """Format the input k used."""
        if k is None:
            k = range(self.k_perturb)
        return k

    def _check_features(self, features):
        """Check the proper format of features.

        Parameters
        ----------
        features: np.ndarray, shape (n, m)
            the features to apply perturbation.

        """
        assert(len(features.shape) == 2)

    def _check_locations(self, locations):
        """Check the proper format of locations.

        Parameters
        ----------
        locations: np.ndarray, shape (n, dim)
            the locations to apply perturbation.

        """
        assert(len(locations.shape) == 2)

    def assert_correctness(self):
        """Assert the correct Perturbation class."""
        assert('_categorytype' in dir(self))
        assert('_perturbtype' in dir(self))

    ##################### Transformations of the indices ######################
    def apply2indices(self, i, k=None):
        """Apply the transformation to the indices.

        Parameters
        ----------
        i: int, list or np.ndarray
            the indices of the elements `i`.
        k: int, list
            the perturbation indices.

        Returns
        -------
        i: int, list or np.ndarray
            the indices of the elements `i`.

        """
        k = self._format_k_perturb(k)
        if type(k) == list:
            if type(i) == np.ndarray:
                i = np.stack([i for ki in k], axis=1)
            elif type(i) == list:
                i = [i for ki in k]
            elif check_int(i):
                i = [i for ki in k]
        return i

    ################## Transformations of the main elements ###################
    def apply2locations(self, locations, k=None):
        """Apply perturbation to locations.

        Parameters
        ----------
        locations: np.ndarray or others
            the spatial information to be perturbed.

        Returns
        -------
        locations: np.ndarray or others
            the spatial information perturbated.

        """
        self._check_locations(locations)
        k = self._format_k_perturb(k)
        if type(k) == list:
            locations = np.stack([locations for ki in k], axis=2)
        return locations

    def apply2features(self, features, k=None):
        """Apply perturbation to features.

        Parameters
        ----------
        features: np.ndarray
            the element features collection to be perturbed.

        Returns
        -------
        features: np.ndarray or others
            the element features collection perturbated.

        """
        self._check_features(features)
        k = self._format_k_perturb(k)
        if type(k) == list:
            axis = len(features.shape)
            features = np.stack([features for ki in k], axis=axis)
        return features

    def apply2relations(self, relations, k=None):
        """Apply perturbation to relations.

        Parameters
        ----------
        relations: np.ndarray or others
            the relations between elements to be perturbated.

        Returns
        -------
        relations: np.ndarray or others
            the relations between elements perturbated.

        """
        ## TODO
        return relations

    def apply2discretizations(self, discretization, k=None):
        """Apply perturbation to discretization.

        Parameters
        ----------
        discretization: np.ndarray or others
            the discretization perturbation.

        Returns
        -------
        discretization: np.ndarray or others
            the discretization perturbation.

        """
        ## TODO
        return discretization

    ######################### Precomputed applications ########################
    def apply2features_ind_precomputed(self, i, k):
        """Apply perturbation to features individually for precomputed
        applications.

        Parameters
        ----------
        i: int or list
            the element indices.
        k: int or list
            the perturbation indices.

        Returns
        -------
        locations: np.ndarray or others
            the element features perturbated.

        """
        if check_int(i):
            return self.features_p[i][:, k]
        else:
            return self.features_p[i][:, :, k]

    def apply2features_ind(self, features, i, k):
        """Apply perturbation to features individually.

        Parameters
        ----------
        features: np.ndarray or others
            the element features to be perturbed.
        i: int or list
            the element indices.
        k: int or list
            the perturbation indices.

        Returns
        -------
        locations: np.ndarray or others
            the element features perturbated.

        """
        if self.features_p is None:
            return self.apply2features(features, k)[i]
        else:
            return self.apply2features_ind_precomputed(i, k)

    def apply2locs_ind_precomputed(self, i, k):
        """Apply perturbation to locations individually for precomputed
        applications.

        Parameters
        ----------
        i: int or list
            the element indices.
        k: int or list
            the perturbation indices.

        Returns
        -------
        locations: np.ndarray or others
            the spatial information perturbated.

        """
        if check_int(i):
            return self.locations_p[i][:, k]
        else:
            return self.locations_p[i][:, :, k]

    def apply2locs_ind(self, locations, i, k):
        """Apply perturbation to locations individually for precomputed
        applications.

        Parameters
        ----------
        locations: np.ndarray or others
            the spatial information to be perturbed.
        i: int or list
            the element indices.
        k: int or list
            the perturbation indices.

        Returns
        -------
        locations: np.ndarray or others
            the spatial information perturbated.

        """
        if self.locations_p is None:
            return self.apply2locations(locations, k)[i]
        else:
            return self.apply2locs_ind_precomputed(i, k)

    def apply2relations_ind(self, relations, i, k):
        """For precomputed applications. Apply perturbation to relations.

        Parameters
        ----------
        relations: np.ndarray or others
            the relations between elements to be perturbated.

        Returns
        -------
        relations: np.ndarray or others
            the relations between elements perturbated.

        """
        if check_int(i):
            return self.relations_p[i][:, k]
        else:
            return self.relations_p[i][:, :, k]

    ##################### Selfcomputation of main elements ####################
    def selfcompute_features(self, features):
        self.features_p = self.apply2features(features)

    def selfcompute_locations(self, locations):
        self.locations_p = self.apply2locations(locations)

    def selfcompute_relations(self, relations):
        pass

    def selfcompute_discretizations(self, discretizations):
        pass

    ################################# Examples ################################
#    def selfcompute_locations(self, locations):
#        self.locations_p = self.apply2locations(locations)
#
#    def selfcompute_features(self, features):
#        self.features_p = self.apply2features(features)


###############################################################################
############################## Base permutation ###############################
###############################################################################
class BasePermutation(BasePerturbation):
    """Base permutation. Main structure to manage the permutation perturbation.
    """
    _perturbtype = "element_permutation"

    def __init__(self, reindices, auto=True):
        """Perturbations by permuting elements.

        Parameters
        ----------
        reindices: np.ndarray
            the reindices to apply permutation perturbations.

        """
        self._initialization()
        self.basecoreperturbation = BaseCorePermutationArray(reindices, auto)
        #self._format_reindices(reindices, auto)

    ###################### Administrative class functions #####################
    def _check_features(self, features):
        """Check the proper format of features.

        Parameters
        ----------
        features: np.ndarray, shape (n, m)
            the features to apply perturbation.

        """
        assert(len(features) == len(self.basecoreperturbation.reindices))
        assert(len(features.shape) == 2)
#
#    def _format_reindices(self, reindices, auto=True):
#        """Format reindices.
#
#        Parameters
#        ----------
#        reindices: np.ndarray or tuple
#            the reindices to apply permutation perturbations.
#
#        """
#        if type(reindices) == np.ndarray:
#            self.k_perturb = reindices.shape[1]
#            self.reindices = reindices
#        elif type(reindices) == tuple:
#            n, k_perturb = reindices
#            if check_int(n) and check_int(k_perturb):
#                auto_indices = [np.arange(n)] if auto else []
#                self.k_perturb = k_perturb
#                self.reindices = np.vstack(auto_indices +
#                                           [np.random.permutation(n)
#                                            for i in xrange(k_perturb)]).T

    def _filter_indices(self, i, k):
        """Filter indices to get the transformed data.

        Parameters
        ----------
        i: int, list, np.ndarray
            the indices of elements.
        k: int, list, np.ndarray
            the indices of perturbations.

        Returns
        -------
        i: int, list, np.ndarray
            the indices of elements.
        k: int, list, np.ndarray
            the indices of perturbations.
        info_input: list
            the boolean information about the if the input is sequencial or
            only a unique index.

        """
        info_input = [True, True]
        ## Check i
        if type(i) == np.ndarray:
            i = list(i)
        elif check_int(i):
            info_input[0] = False
            i = [i]
        assert(type(i) == list)
        ## Check k
        k = self._format_k_perturb(k)
        if type(k) == np.ndarray:
            k = list(k)
        elif check_int(k):
            info_input[1] = False
            k = [k]
        assert(type(k) == list)
        return i, k, info_input

    def _filter_output(self, result, info_input):
        """Filter output for 2d array results (locations and features array).

        Parameters
        ----------
        result: np.ndarray, shape: (ni, ndim, nk)
            the result we want to format to be output.

        Returns
        -------
        result: np.ndarray
            properly formatted adapted to the format of the indices input.

        """
        if not info_input[1]:
            result = result[:, :, 0]
        if not info_input[0]:
            result = result[0]
        return result

    ########################### Main class functions ##########################
    def apply2indices(self, i, k=None):
        """Apply the transformation to the indices.

        Parameters
        ----------
        i: int, list or np.ndarray
            the indices of the elements `i`.
        k: int, list
            the perturbation indices.

        Returns
        -------
        i: int, list or np.ndarray
            the indices of the elements `i`.

        """
        k = self._format_k_perturb(k)
        if check_int(i):
            return self.basecoreperturbation.reindices[i, k]
        elif type(i) == list:
            return list(self.basecoreperturbation.reindices[i][:, k])
        else:
            return self.basecoreperturbation.reindices[list(i)][:, k]

    def _apply2someelements(self, array, i, k):
        """Apply perturbation to array for individual elements.

        Parameters
        ----------
        array: np.ndarray
            the spatial information to be perturbed.
        k: int (default=None)
            the perturbation indices.

        Returns
        -------
        array_p: np.ndarray
            the spatial information perturbated.

        """
        ## Apply permutation to array
        rei = self.basecoreperturbation.reindices[i][:, k]
        array_p = array[rei].swapaxes(1, 2)
        return array_p

    def _apply2allelements(self, array, k):
        """Apply perturbation to array of complete elements.

        """
        ## Apply permutation to array
        array_p = self.basecoreperturbation.apply(array, k)
#        array_p = array[self.reindices[:, k]].swapaxes(1, 2)
        return array_p


###############################################################################
########################## Base Jitter perturbation ###########################
###############################################################################
class BaseJitterPerturbation(BasePerturbation):
    """Base Jitter Perturbation class.
    """

    def __init__(self, stds=0, k_perturb=1):
        """The jitter locations apply to locations a jittering perturbation.

        Parameters
        ----------
        stds: float, np.ndarray
            the dispersion measure of the jittering.
        k_perturb: int (default=1)
            the number of perturbations applied.

        """
        self._initialization()
        self.basecoreperturbation = BaseCoreJitteringArray(stds)
#        self._stds = np.array(stds)
        self.k_perturb = k_perturb


###############################################################################
############################## None perturbation ##############################
###############################################################################
class NonePerturbation(BasePerturbation):
    """None perturbation. Default perturbation which not alters the system."""
    _categorytype = "general"
    _perturbtype = "none"

    def __init__(self, k_perturb=1):
        """The none perturbation, null perturbation where anything happens.

        Parameters
        ----------
        k_perturb: int (default=1)
            the number of perturbations applied.

        """
        self._initialization()
        self.k_perturb = k_perturb


###############################################################################
############################ Location perturbation ############################
###############################################################################
class JitterLocations(BaseJitterPerturbation):
    """Jitter module to perturbe locations of the system in order of testing
    methods.
    TODO: Fit some model for infering stds.
    """
    _categorytype = "location"
    _perturbtype = "jitter_coordinate"

    ########################### Main class functions ##########################
    def apply2locations(self, locations, k=None):
        """Apply perturbation to locations.

        Parameters
        ----------
        locations: np.ndarray
            the spatial information to be perturbed.
        k: int (default=None)
            the perturbation indices.

        Returns
        -------
        locations: np.ndarray
            the spatial information perturbated.

        """
        ## 0. Prepare inputs
        # Check proper locations
        self._check_locations(locations)
        # Preparation of ks
        ks = range(self.k_perturb) if k is None else k
        ks = [k] if check_int(k) else ks
        ## 1. Main computation
        locations_p = self.basecoreperturbation.apply(locations, ks)
        ## 2. Format output
        if check_int(k):
            return locations_p[:, :, 0]
        return locations_p


class PermutationPerturbationLocations(BasePermutation):
    """Reindice perturbation for the whole locations."""
    _categorytype = "location"
    _perturbtype = "element_permutation"

    ########################### Main class functions ##########################
    def apply2locations(self, locations, k=None):
        """Apply perturbation to locations.

        Parameters
        ----------
        locations: np.ndarray
            the spatial information to be perturbed.
        k: int (default=None)
            the perturbation indices.

        Returns
        -------
        locations: np.ndarray
            the spatial information perturbated.

        """
        ## Check locations
        self._check_locations(locations)
        i, k, info_input = self._filter_indices([0], k)
        ## Apply permutation to locations
        locations_p = self._apply2allelements(locations, k)
        ## Format output
        locations_p = self._filter_output(locations_p, info_input)
        return locations_p

    def apply2locs_ind(self, locations, i, k=None):
        """Apply perturbation to locations for individual elements.

        Parameters
        ----------
        locations: np.ndarray
            the spatial information to be perturbed.
        k: int (default=None)
            the perturbation indices.

        Returns
        -------
        locations: np.ndarray
            the spatial information perturbated.

        """
        ## Check locations
        self._check_locations(locations)
        i, k, info_input = self._filter_indices(i, k)
        ## Apply permutation to locations
        locations_p = self._apply2someelements(locations, i, k)
        ## Format output
        locations_p = self._filter_output(locations_p, info_input)
        return locations_p


###############################################################################
########################### Permutation perturbation ##########################
###############################################################################
class PermutationPerturbation(BasePermutation):
    """Reindice perturbation for the whole features variables."""
    _categorytype = "feature"
    _perturbtype = "element_permutation"

    ########################### Main class functions ##########################
    def apply2features(self, features, k=None):
        """Apply perturbation to features.

        Parameters
        ----------
        features: np.ndarray or others
            the element features collection to be perturbed.
        k: int (default=None)
            the perturbation indices.

        Returns
        -------
        features: np.ndarray or others
            the element features collection perturbated.

        """
        ## Check features
        self._check_features(features)
        i, k, info_input = self._filter_indices([0], k)
        ## Apply permutation to features
        features_p = self._apply2allelements(features, k)
        ## Format output
        features_p = self._filter_output(features_p, info_input)
        return features_p

    def apply2features_ind(self, features, i, k):
        """Apply perturbation to features individually for precomputed
        applications.

        Parameters
        ----------
        features: np.ndarray or others
            the element features to be perturbed.
        i: int or list
            the element indices.
        k: int or list
            the perturbation indices.

        Returns
        -------
        features_p: np.ndarray or others
            the element features perturbated.

        """
        ## Check features
        self._check_features(features)
        i, k, info_input = self._filter_indices(i, k)
        ## Apply permutation to features
        features_p = self._apply2someelements(features, i, k)
        ## Format output
        features_p = self._filter_output(features_p, info_input)
        return features_p


class PermutationPerturbationGeneration(PermutationPerturbation):
    """Reindice perturbation for the whole features variables."""

    def __init__(self, n, k_perturb=1, seed=None, auto=True):
        """Element perturbation for all permutation perturbation.

        Parameters
        ----------
        n: int
            the size of the sample to create the reindices.
        m: int (default=1)
            the number of permutations we want to generate.
        seed: int (default=Npne)
            the seed to initialize and create the same reindices.

        """
        self._initialization()
        if seed is not None:
            np.random.seed(seed)
        self.basecoreperturbation =\
            BaseCorePermutationArray((n, k_perturb), auto)


class PartialPermutationPerturbationGeneration(PermutationPerturbation):
    """Reindice perturbation for the whole features variables. It can control
    the proportion of the whole sample is going to be permuted.
    """

    def __init__(self, n, rate_pert=1., k_perturb=1, seed=None, auto=True):
        """Element perturbation for all permutation perturbation.

        Parameters
        ----------
        n: int
            the size of the sample to create the reindices.
        rate_pert: float
            the partial proportion of indices which will be changed.
        k_perturb: int (default=1)
            the number of permutations we want to generate.
        seed: int (default=Npne)
            the seed to initialize and create the same reindices.
        auto: boolean
            if we want to keep the first permutation as a null permutation.

        """
        self._initialization()
        if seed is not None:
            np.random.seed(seed)
        if rate_pert == 1.:
            self.basecoreperturbation =\
                BaseCorePermutationArray((n, k_perturb), auto)
        else:
            reindices = self._format_partial_permutation(n, rate_pert,
                                                         k_perturb, auto)
            self.basecoreperturbation = BaseCorePermutationArray(reindices)

    def _format_partial_permutation(self, n, rate_pert=1., k_perturb=1,
                                    auto=True):
        """Format the reindices.

        Parameters
        ----------
        n: int
            the size of the sample to create the reindices.
        rate_pert: float
            the partial proportion of indices which will be changed.
        k_perturb: int (default=1)
            the number of permutations we want to generate.

        """
        n_sample = int(n*rate_pert)
        indices = np.random.permutation(n)[:n_sample]
        auto_indices = [indices] if auto else []
        reindices = np.vstack([np.arange(n) for i in xrange(k_perturb+1)]).T
        reindices[indices] = np.vstack(auto_indices +
                                       [np.random.permutation(n_sample)
                                        for i in xrange(1, k_perturb+1)]).T
        return reindices


###############################################################################
############################# Element perturbation ############################
###############################################################################
class MixedFeaturePertubation(BasePerturbation):
    """An individual-column-created perturbation of individual elements."""
    _categorytype = "feature"
    _perturbtype = "element_mixed"

    def __init__(self, perturbations):
        """The MixedFeaturePertubation is the application of different
        perturbations to features.

        perturbations: list
            the list of pst.BasePerturbation objects.

        """
        msg = "Perturbations is not a list of individual perturbation methods."
        self._initialization()
        if type(perturbations) != list:
            raise TypeError(msg)
        try:
            self.typefeats = [p._perturbtype for p in perturbations]
            k_perturbs = [p.k_perturb for p in perturbations]
            assert all([k == k_perturbs[0] for k in k_perturbs])
            self.k_perturb = k_perturbs[0]
            self.perturbations = perturbations
        except:
            raise TypeError(msg)

    def apply2features(self, features, k=None):
        """Apply perturbation to features.

        Parameters
        ----------
        features: np.ndarray or others
            the element features collection to be perturbed.
        k: int (default=None)
            the perturbation indices.

        Returns
        -------
        features: np.ndarray or others
            the element features collection perturbated.

        """
        assert features.shape[1] == len(self.perturbations)
        logi_int = check_int(k)
        k = self._format_k_perturb(k)
        if logi_int:
            k = [k]
        ## Apply individual perturbation for each features
        features_p = []
        for i in range(len(self.perturbations)):
            features_p_k =\
                self.perturbations[i].apply2features(features[:, [i]], k)
            features_p.append(features_p_k)
        features_p = np.concatenate(features_p, axis=1)
        if logi_int:
            features_p = features_p[:, :, 0]
        return features_p


########################### Individual perturbation ###########################
###############################################################################
class DiscreteIndPerturbation(BasePerturbation):
    """Discrete perturbation of a discrete feature variable."""
    _categorytype = "feature"
    _perturbtype = "discrete"

    def __init__(self, probs, labels=None):
        """The discrete individual perturbation to a feature variable.

        Parameters
        ----------
        probs: np.ndarray
            the probabilities to change from a value of a category to another
            value.

        """
        self._initialization()
        self.basecoreperturbation = BaseCoreSubstitutionArray(probs, labels)
#        if np.all(probs.sum(1) != 1):
#            raise TypeError("Not correct probs input.")
#        if probs.shape[0] != probs.shape[1]:
#            raise IndexError("Probs is noot a square matrix.")
#        self.probs = probs.cumsum(1)
#        self.labels = np.arange(len(probs)) if labels is None else labels

    def apply2features(self, feature, k=None):
        """Apply perturbation to features.

        Parameters
        ----------
        features: np.ndarray or others
            the element features collection to be perturbed.
        k: int (default=None)
            the perturbation indices.

        Returns
        -------
        features: np.ndarray or others
            the element features collection perturbated.

        """
        ## Prepare inputs
#        categories = np.unique(feature)
#        if len(categories) > len(self.probs):
#            msg = "Not matching dimension between probs and features."
#            raise IndexError(msg)
        k = self._format_k_perturb(k)
        logi_int = check_int(k)
        if logi_int:
            k = [k]
        if len(feature.shape) == 1:
            feature = feature.reshape((len(feature), 1))
        ## Compute each change
        features_p = self.basecoreperturbation.apply(feature, k)
#        features_p = np.zeros((len(feature), 1, len(k)))
#        for i_k in k:
#            for i in xrange(len(feature)):
#                r = np.random.random()
#                idx = np.where(feature[i] == self.labels)[0]
#                idx2 = np.where(self.probs[idx] > r)[0][0]
#                features_p[i, 0, i_k] = self.labels[idx2]

        if logi_int:
            features_p = features_p[:, :, 0]
        return features_p


class ContiniousIndPerturbation(BaseJitterPerturbation):
    """Continious perturbation for an individual feature variable."""
    _categorytype = "feature"
    _perturbtype = "continious"

    def apply2features(self, feature, k=None):
        """Apply perturbation to features.

        Parameters
        ----------
        features: np.ndarray or others
            the element features collection to be perturbed.
        k: int (default=None)
            the perturbation indices.

        Returns
        -------
        features: np.ndarray or others
            the element features collection perturbated.

        """
        ## Prepare inputs
        k = self._format_k_perturb(k)
        logi_int = check_int(k)
        if logi_int:
            k = [k]
        if len(feature.shape) == 1:
            feature = feature.reshape((len(feature), 1))
        ## Compute each change
        features_p = self.basecoreperturbation.apply(feature, k)
#        ## Compute perturbation
#        features_p = np.zeros((len(feature), 1, len(k)))
#        for i_k in k:
#            jitter_d = np.random.random(len(feature))
#            features_p[:, 0, i_k] = np.multiply(self._stds, jitter_d)
        ## Format output
        if logi_int:
            features_p = features_p[:, :, 0]
        return features_p


class PermutationIndPerturbation(PermutationPerturbation):
    """Reindice perturbation for an individual feature variable."""
    _categorytype = "feature"
    _perturbtype = "permutation_ind"

    ####################### Main perturbation functions #######################
    def apply2features(self, feature, k=None):
        """Apply perturbation to features.

        Parameters
        ----------
        features: np.ndarray or others
            the element features collection to be perturbed.
        k: int (default=None)
            the perturbation indices.

        Returns
        -------
        features: np.ndarray or others
            the element features collection perturbated.

        """
        ## Prepare input
        if len(feature.shape) == 1:
            feature = feature.reshape((len(feature), 1))
        ## Compute perturbation
        features_p =\
            super(PermutationIndPerturbation, self).apply2features(feature, k)
        return features_p

    def apply2features_ind(self, features, i, k):
        """Apply perturbation to features individually for precomputed
        applications.

        Parameters
        ----------
        features: np.ndarray or others
            the element features to be perturbed.
        i: int or list
            the element indices.
        k: int or list
            the perturbation indices.

        Returns
        -------
        features: np.ndarray or others
            the element features collection perturbated.

        """
        ## Prepare input
        if len(features.shape) == 1:
            features = features.reshape((len(features), 1))
        ## Compute perturbation
        features_p = super(PermutationIndPerturbation, self).\
            apply2features_ind(features, i, k)
        return features_p


###############################################################################
########################### Aggregation perturbation ##########################
###############################################################################
class JitterRelationsPerturbation(BasePerturbation):
    """Jitter module to perturbe relations of the system in order of testing
    methods.
    """
    _categorytype = "relations"
