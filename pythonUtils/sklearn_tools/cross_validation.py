
"""
cross-validation
----------------
Module which groups utils for the extension of cross-validation functions
from sklearn.

"""

import warnings
import numpy as np
from sklearn.grid_search import BaseSearchCV
from sklearn.cross_validation import _BaseKFold
from sklearn.utils import check_random_state
from abc import abstractmethod


############################## Collection of cv ###############################
###############################################################################
class _PartitionIterator_list(object):
    """Base class for CV iterators where train_mask = ~test_mask

    Implementations must define `_iter_test_masks` or `_iter_test_indices`.

    Parameters
    ----------
    n : list of int
        Total number of elements in each dataset.
    indices : boolean, optional (default True)
        Return train/test split as arrays of indices, rather than a boolean
        mask array. Integer indices are required when dealing with sparse
        matrices, since those cannot be indexed by boolean masks.

    """

    def __init__(self, n, indices=True):
        assert(type(n) in [list, int])
        n = n if type(n) == list else [n]
        for n_i in n:
            if abs(n_i - int(n_i)) >= np.finfo('f').eps:
                raise ValueError("n must be an integer")
        self.n = [int(n_i) for n_i in n]
        self.indices = indices

    def __iter__(self):
        indices = self.indices
        if indices:
            ind = []
            for i in range(len(self.n)):
                ind.append(np.arange(self.n[i]))
        for test_index in self._iter_test_masks():
            train_index = [np.logical_not(e) for e in test_index]
            if indices:
                train_index_aux, test_index_aux = [], []
                print '0'*10
                for i in range(len(self.n)):
                    train_index_aux.append(ind[i][train_index[i]])
                    test_index_aux.append(ind[i][test_index[i]])
                print '1'*10
                yield train_index_aux, test_index_aux
            else:
                print '2'*10
                yield train_index, test_index

    # Since subclasses must implement either _iter_test_masks or
    # _iter_test_indices, neither can be abstract.
    def _iter_test_masks(self):
        """Generates boolean masks corresponding to test sets.

        By default, delegates to _iter_test_indices()
        """
        for test_index in self._iter_test_indices():
                test_mask = self._empty_mask()
                for i in range(len(self.n)):
                    test_mask[i][test_index[i]] = True
                yield test_mask

    def _iter_test_indices(self):
        """Generates integer indices corresponding to test sets."""
        raise NotImplementedError

    def _empty_mask(self):
        return [np.zeros(self.n[i], dtype=np.bool) for i in range(len(self.n))]


class _BaseKFold_list(_PartitionIterator_list):
    """Base class to validate KFold approaches"""

    @abstractmethod
    def __init__(self, n, n_folds, indices, k=None):
        super(_BaseKFold_list, self).__init__(n, indices)
        if k is not None:  # pragma: no cover
            warnings.warn("The parameter k was renamed to n_folds and will be"
                          " removed in 0.15.", DeprecationWarning)
            n_folds = k
        if n_folds <= 1:
            raise ValueError(
                "k-fold cross validation requires at least one"
                " train / test split by setting n_folds=2 or more,"
                " got n_folds=%d.".format(n_folds))
        if n_folds > max(self.n):
            raise ValueError(
                ("Cannot have number of folds n_folds={0} greater "
                 "than the number of samples: {1}.").format(n_folds, n))
        if abs(n_folds - int(n_folds)) >= np.finfo('f').eps:
            raise ValueError("n_folds must be an integer")
        self.n_folds = int(n_folds)


class KFold_list(_BaseKFold_list):
    """KFold_list cross validation iterator.

    Provides train/test indices to split data in train test sets.

    This cross-validation object is a variation of KFold for list, which
    returns a list of splitted folds following a given values.

    Parameters
    ----------
    n : list of int
        Number of size for each part.
    n_folds : int (default=3)
        Number of folds. Must be at least 2.
    indices : boolean, optional (default=True)
        Return train/test split as arrays of indices, rather than a boolean
        mask array. Integer indices are required when dealing with sparse
        matrices, since those cannot be indexed by boolean masks.

    Examples
    --------
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    >>> skf = cross_validation.KFold_list(4, n_folds=[2, 2])
    >>> len(skf)
    2
    >>> print(skf)
    KFold_list(n_samples=4, n_folds=2)
    >>> for train_index, test_index in skf:
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...    X_train, X_test = X[train_index], X[test_index]
    ...    y_train, y_test = y[train_index], y[test_index]
    TRAIN: [0 2] TEST: [1 3]
    TRAIN: [1 3] TEST: [0 2]

    """

    def __init__(self, n, n_folds=3, indices=True, shuffle=False,
                 random_state=None, k=None):
        super(KFold_list, self).__init__(n, n_folds, indices, k)
        random_state = check_random_state(random_state)
        if shuffle:
            self.idxs = [np.arange(n[i]) for i in range(len(n))]
            for i in range(len(n)):
                random_state.shuffle(self.idxs[i])
        else:
            self.idxs = [np.arange(n[i]) for i in range(len(n))]

    def _iter_test_indices(self):
        n_folds = self.n_folds
        fold_sizes = []
        for i in range(len(self.n)):
            n = self.n[i]
            fold_sizes_i = (n // n_folds) * np.ones(n_folds, dtype=np.int)
            fold_sizes_i[:n % n_folds] += 1
            fold_sizes.append(fold_sizes_i)
        current = np.zeros(len(self.n)).astype(int)
        for nf in range(n_folds):
            ind = []
            for i in range(len(self.n)):
                fold_size = fold_sizes[i][nf]
                start, stop = current[i], current[i] + fold_size
                ind.append(self.idxs[i][start:stop])
                current[i] = stop
            yield ind

    def __repr__(self):
        return '%s.%s(n_samples=%s, n_folds=%i)' % (
            self.__class__.__module__,
            self.__class__.__name__,
            self.n,
            self.n_folds,
        )

    def __len__(self):
        return self.n_folds


################################ Particular cv ################################
###############################################################################
class QuantileKFold(_BaseKFold):
    """Quantile K-Folds cross validation iterator.

    Provides train/test indices to split data in train test sets.

    This cross-validation object is a variation of KFold, which
    returns quantile splitted folds following a given values.
    The folds are made by preserving an ordering of the values
    the percentage of samples for each class.

    Each fold is then used a validation set once while the k - 1 remaining
    fold form the training set.

    Parameters
    ----------
    y : array-like, [n_samples]
        Samples to split in K folds.
    n_folds : int, default=3
        Number of folds. Must be at least 2.
    indices : boolean, optional (default True)
        Return train/test split as arrays of indices, rather than a boolean
        mask array. Integer indices are required when dealing with sparse
        matrices, since those cannot be indexed by boolean masks.

    Examples
    --------
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([0.2, 0.7, 0.1, 0.9])
    >>> skf = cross_validation.QuantileKFold(y, n_folds=2)
    >>> len(skf)
    2
    >>> print(skf)
    QuantileKFold(n_samples=4, n_folds=2)
    >>> for train_index, test_index in skf:
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...    X_train, X_test = X[train_index], X[test_index]
    ...    y_train, y_test = y[train_index], y[test_index]
    TRAIN: [0 2] TEST: [1 3]
    TRAIN: [1 3] TEST: [0 2]

    Notes
    -----
    All the folds have size trunc(n_samples / n_folds), the last one has the
    complementary.

    See also
    --------
    StratifiedKFold: take label information into account to avoid building
    folds with imbalanced class distributions (for binary or multiclass
    classification tasks).

    """

    def __init__(self, values, n_folds=3, indices=True, k=None):
        super(QuantileKFold, self).__init__(len(values), n_folds, indices, k)
        self.values = np.asarray(values)

    def _iter_test_indices(self):
        n_folds = self.n_folds
        idx = np.argsort(self.values)
        splits = [(len(idx)/n_folds)*i for i in range(n_folds)]
        splits.append(len(idx))
        for i in range(n_folds):
            yield idx[splits[i]:splits[i+1]]

    def __repr__(self):
        return '%s.%s(n_samples=%s, n_folds=%i)' % (
            self.__class__.__module__,
            self.__class__.__name__,
            len(self.values),
            self.n_folds,
        )

    def __len__(self):
        return self.n_folds

#
#class SequencialSearchCV(BaseSearchCV):
#    """Sequencial search over specified parameter values for an estimator.
#
#    Important members are fit, predict.
#
#    SequencialSearchCV implements a "fit" method and a "predict" method like
#    any classifier except that the parameters of the classifier
#    used to predict is optimized by cross-validation.
#
#    Parameters
#    ----------
#    estimator: object type that implements the "fit" and "predict" methods
#        A object of that type is instantiated for each grid point.
#    param_grid: dict or list of dictionaries
#        Dictionary with parameters names (string) as keys and lists of
#        parameter settings to try as values, or a list of such
#        dictionaries, in which case the grids spanned by each dictionary
#        in the list are explored. This enables searching over any sequence
#        of parameter settings.
#    scoring: string, callable or None, optional, default: None
#        A string (see model evaluation documentation) or
#        a scorer callable object / function with signature
#        ``scorer(estimator, X, y)``.
#    fit_params : dict, optional
#        Parameters to pass to the fit method.
#    n_jobs : int, optional
#        Number of jobs to run in parallel (default 1).
#    pre_dispatch : int, or string, optional
#        Controls the number of jobs that get dispatched during parallel
#        execution. Reducing this number can be useful to avoid an
#        explosion of memory consumption when more jobs get dispatched
#        than CPUs can process. This parameter can be:
#
#            - None, in which case all the jobs are immediately
#              created and spawned. Use this for lightweight and
#              fast-running jobs, to avoid delays due to on-demand
#              spawning of the jobs
#
#            - An int, giving the exact number of total jobs that are
#              spawned
#
#            - A string, giving an expression as a function of n_jobs,
#              as in '2*n_jobs'
#    iid: boolean, optional
#        If True, the data is assumed to be identically distributed across
#        the folds, and the loss minimized is the total loss per sample,
#        and not the mean loss across the folds.
#    cv: integer or cross-validation generator, optional
#        If an integer is passed, it is the number of folds (default 3).
#        Specific cross-validation objects can be passed, see
#        sklearn.cross_validation module for the list of possible objects
#    refit: boolean
#        Refit the best estimator with the entire dataset.
#        If "False", it is impossible to make predictions using
#        this SequencialSearchCV instance after fitting.
#    verbose : integer
#        Controls the verbosity: the higher, the more messages.
#
#    Examples
#    --------
#    >>> from sklearn import svm, datasets
#    >>> iris = datasets.load_iris()
#    >>> parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
#    >>> svr = svm.SVC()
#    >>> clf = grid_search.SequencialSearchCV(svr, parameters)
#    >>> clf.fit(iris.data, iris.target)
#    ...                             # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
#    SequencialSearchCV(cv=None, estimator=SVC(C=1.0, cache_size=...,
#        class_weight=..., coef0=..., degree=..., gamma=...,
#        kernel='rbf', max_iter=-1, probability=False, random_state=None,
#        shrinking=True, tol=..., verbose=False),
#            fit_params={}, iid=..., loss_func=..., n_jobs=1,
#            param_grid=..., pre_dispatch=..., refit=..., score_func=...,
#            scoring=..., verbose=...)
#
#    Attributes
#    ----------
#    `grid_scores_` : list of named tuples
#        Contains scores for all parameter combinations in param_grid.
#        Each entry corresponds to one parameter setting.
#        Each named tuple has the attributes:
#
#            * ``parameters``, a dict of parameter settings
#            * ``mean_validation_score``, the mean score over the
#              cross-validation folds
#            * ``cv_validation_scores``, the list of scores for each fold
#
#    `best_estimator_` : estimator
#        Estimator that was chosen by the search, i.e. estimator
#        which gave highest score (or smallest loss if specified)
#        on the left out data.
#
#    `best_score_` : float
#        Score of best_estimator on the left out data.
#
#    `best_params_` : dict
#        Parameter setting that gave the best results on the hold out data.
#
#    Notes
#    ------
#    The parameters selected are those that maximize the score of the left out
#    data, unless an explicit score is passed in which case it is used instead.
#
#    If `n_jobs` was set to a value higher than one, the data is copied for each
#    point in the grid (and not `n_jobs` times). This is done for efficiency
#    reasons if individual jobs take very little time, but may raise errors if
#    the dataset is large and not enough memory is available.  A workaround in
#    this case is to set `pre_dispatch`. Then, the memory is copied only
#    `pre_dispatch` many times. A reasonable value for `pre_dispatch` is `2 *
#    n_jobs`.
#
#    See Also
#    ---------
#    :class:`ParameterGrid`:
#        generates all the combinations of a an hyperparameter grid.
#
#    :func:`sklearn.cross_validation.train_test_split`:
#        utility function to split the data into a development set usable
#        for fitting a GridSearchCV instance and an evaluation set for
#        its final evaluation.
#
#    """
#
#    def __init__(self, estimator, param_grid, scoring=None, loss_func=None,
#                 score_func=None, fit_params=None, n_jobs=1, iid=True,
#                 refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs'):
#        super(GridSearchCV, self).__init__(
#            estimator, scoring, loss_func, score_func, fit_params, n_jobs, iid,
#            refit, cv, verbose, pre_dispatch)
#        self.param_grid = param_grid
#        _check_param_grid(param_grid)
#
#    def fit(self, X, y=None, **params):
#        """Run fit with all sets of parameters.
#
#        Parameters
#        ----------
#        X : array-like, shape = [n_samples, n_features]
#            Training vector, where n_samples is the number of samples and
#            n_features is the number of features.
#
#        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
#            Target relative to X for classification or regression;
#            None for unsupervised learning.
#
#        """
#        if params:
#            warnings.warn("Additional parameters to GridSearchCV are ignored!"
#                          " The params argument will be removed in 0.15.",
#                          DeprecationWarning)
#        return self._fit(X, y, ParameterGrid(self.param_grid))
#
#
#class ParameterGrid(object):
#    """Grid of parameters with a discrete number of values for each.
#
#    Can be used to iterate over parameter value combinations with the
#    Python built-in function iter.
#
#    Parameters
#    ----------
#    param_grid : dict of string to sequence, or sequence of such
#        The parameter grid to explore, as a dictionary mapping estimator
#        parameters to sequences of allowed values.
#
#        An empty dict signifies default parameters.
#
#        A sequence of dicts signifies a sequence of grids to search, and is
#        useful to avoid exploring parameter combinations that make no sense
#        or have no effect. See the examples below.
#
#    Examples
#    --------
#    >>> from sklearn.grid_search import ParameterGrid
#    >>> param_grid = {'a': [1, 2], 'b': [True, False]}
#    >>> list(ParameterGrid(param_grid)) == (
#    ...    [{'a': 1, 'b': True}, {'a': 1, 'b': False},
#    ...     {'a': 2, 'b': True}, {'a': 2, 'b': False}])
#    True
#
#    >>> grid = [{'kernel': ['linear']}, {'kernel': ['rbf'], 'gamma': [1, 10]}]
#    >>> list(ParameterGrid(grid)) == [{'kernel': 'linear'},
#    ...                               {'kernel': 'rbf', 'gamma': 1},
#    ...                               {'kernel': 'rbf', 'gamma': 10}]
#    True
#
#    See also
#    --------
#    :class:`GridSearchCV`:
#        uses ``ParameterGrid`` to perform a full parallelized parameter search.
#    """
#
#    def __init__(self, param_grid):
#        if isinstance(param_grid, Mapping):
#            # wrap dictionary in a singleton list
#            # XXX Why? The behavior when passing a list is undocumented,
#            # but not doing this breaks one of the tests.
#            param_grid = [param_grid]
#        self.param_grid = param_grid
#
#    def __iter__(self):
#        """Iterate over the points in the grid.
#
#        Returns
#        -------
#        params : iterator over dict of string to any
#            Yields dictionaries mapping each estimator parameter to one of its
#            allowed values.
#        """
#        for p in self.param_grid:
#            # Always sort the keys of a dictionary, for reproducibility
#            items = sorted(p.items())
#            if not items:
#                yield {}
#            else:
#                keys, values = zip(*items)
#                for v in product(*values):
#                    params = dict(zip(keys, v))
#                    yield params
#
#    def __len__(self):
#        """Number of points on the grid."""
#        # Product function that can handle iterables (np.product can't).
#        product = partial(reduce, operator.mul)
#        return sum(product(len(v) for v in p.values()) if p else 1
#                   for p in self.param_grid)
