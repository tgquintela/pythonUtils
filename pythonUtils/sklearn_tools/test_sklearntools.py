
"""
Sklearn tools tests
-------------------
Collections of tests of sklearn tools.

"""

import numpy as np
import warnings
from cross_validation import QuantileKFold, KFold_list


def test():
    ## Artificial data
    n = 10000
    X = np.random.random((n, 4))
    y = np.random.random(n)
    X_list = [np.random.random((n, 4)) for i in range(10)]
    y_list = [np.random.random(n) for i in range(10)]

    ####################################################
    ### Testing personal KFolds
    ###########################
    ###### INDIVIDUAL SPLITTERS
    ### Quantile Kfold
    skf = QuantileKFold(y, n_folds=2)
    len(skf)
    skf.__repr__()
    for train_index, test_index in skf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    ###### LIST SPLITTERS
    #### KFold_list
    skf = KFold_list([len(y), len(y)], n_folds=2)
    len(skf)
    skf.__repr__()
    for train_index, test_index in skf:
        X_train = [X_list[i][train_index[i]] for i in range(len(train_index))]
        X_test = [X_list[i][test_index[i]] for i in range(len(test_index))]
        y_train = [y_list[i][train_index[i]] for i in range(len(train_index))]
        y_test = [y_list[i][test_index[i]] for i in range(len(test_index))]

    skf = KFold_list([len(y), len(y)], n_folds=2, indices=False, shuffle=False)
    len(skf)
    skf.__repr__()
    for train_index, test_index in skf:
        X_train = [X_list[i][train_index[i]] for i in range(len(train_index))]
        X_test = [X_list[i][test_index[i]] for i in range(len(test_index))]
        y_train = [y_list[i][train_index[i]] for i in range(len(train_index))]
        y_test = [y_list[i][test_index[i]] for i in range(len(test_index))]

    skf = KFold_list([len(y), len(y)], n_folds=2, indices=True, shuffle=True)
    len(skf)
    skf.__repr__()
    for train_index, test_index in skf:
        X_train = [X_list[i][train_index[i]] for i in range(len(train_index))]
        X_test = [X_list[i][test_index[i]] for i in range(len(test_index))]
        y_train = [y_list[i][train_index[i]] for i in range(len(train_index))]
        y_test = [y_list[i][test_index[i]] for i in range(len(test_index))]

    skf = KFold_list([len(y), len(y)], n_folds=2, indices=False, shuffle=True)
    len(skf)
    skf.__repr__()
    for train_index, test_index in skf:
        X_train = [X_list[i][train_index[i]] for i in range(len(train_index))]
        X_test = [X_list[i][test_index[i]] for i in range(len(test_index))]
        y_train = [y_list[i][train_index[i]] for i in range(len(train_index))]
        y_test = [y_list[i][test_index[i]] for i in range(len(test_index))]

    warnings.filterwarnings('ignore')
    skf = KFold_list([len(y), len(y)], k=2)
    warnings.filterwarnings('always')
    try:
        logi = False
        skf = KFold_list([len(y), len(y)], n_folds=0)
        logi = True
        raise Exception()
    except:
        if logi:
            raise Exception("Not correct instanciation")
    try:
        logi = False
        skf = KFold_list([2, 2], n_folds=100)
        logi = True
        raise Exception()
    except:
        if logi:
            raise Exception("Not correct instanciation")
    try:
        logi = False
        skf = KFold_list([20, 20], n_folds=1.5)
        logi = True
        raise Exception()
    except:
        if logi:
            raise Exception("Not correct instanciation")
    try:
        logi = False
        skf = KFold_list([20.2, 20.2], n_folds=1.5)
        logi = True
        raise Exception()
    except:
        if logi:
            raise Exception("Not correct instanciation")
