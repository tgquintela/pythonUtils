
"""
tests
-----
The module which call and centralize all the tests utilities.

"""

from parallel_tools import test_parallel_tools
from Logger import test_logger
from TUI_tools import test_tui_tools
from CodingText import test_codingtext
from ProcessTools import test_processtools
from numpy_tools import test_numpytools
from ExploreDA import test_exploreDA
from MetricResults import test_metricresults
from CollectionMeasures import test_collectionmeasures
from Combinatorics import test_combinatorics
from sklearn_tools import test_sklearntools
from Perturbations import test_perturbations
from perturbation_tests import test_perturbationtests
from NeighsManager import test_neighsmanager

## Check administrative
import release
import version

## Not inform about warnings
#import warnings
#with warnings.catch_warnings():
#    warnings.simplefilter("ignore")
#warnings.simplefilter("ignore")


def test():
    ## Tests of modules
#    test_parallel_tools.test()
#    test_logger.test()
#    test_tui_tools.test()
#    test_codingtext.test()
#    test_numpytools.test()
#    test_processtools.test()
#    test_exploreDA.test()
    test_metricresults.test()
##    test_collectionmeasures.test()
#    test_combinatorics.test()
#    test_perturbations.test()
#    test_sklearntools.test()
##    test_perturbationtests.test()
#    test_neighsmanager.test()
