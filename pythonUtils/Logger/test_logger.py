
"""
test_logger
-----------
Test to the logger utilities.

"""

import os
from logging import Logger


def test():
    logfile = 'log.log'

    log = Logger(logfile)
    log.write_log('', False)

    os.remove(logfile)
