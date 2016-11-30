
"""
testing_processer
-----------------
The testing function for test the processer utilities.

"""

import os
from processer import Processer
from ..Logger import Logger
from processer import check_subprocess, create_empty_list_from_hierarchylist,\
    store_time_subprocess, initial_message_creation


## Code a dummy processer class
#class DummyProcesser(Processer):
#    proc_name = "Dummy process"
#
#    def __init__(self, logfile, lim_rows=0, prompt_inform=False):
#        """Dummy process initialization."""
#        self._initialization()
#        # Logfile
#        self.logfile = logfile
#        # Other parameters
#        self.lim_rows = lim_rows
#        self.bool_inform = True if self.lim_rows != 0 else False
#        self.prompt_inform = prompt_inform
#        self.n_procs = 0
#        self.proc_desc = "Computation dummy processer"
#
#    def compute(self, n):
#        t00 = self.setting_global_process()
#        ## 1. Computation of the measure (parallel if)
#        # Begin to track the process
#        t0, bun = self.setting_loop(n)
#        for i in xrange(n):
#            ## Finish to track this process
#            t0, bun = self.messaging_loop(i, t0, bun)
#        # Stop tracking
#        self.close_process(t00)


class TesterProcesserClass(Processer):
    proc_name = "Tester Processer"

    def __init__(self, logfile, lim_rows=0, prompt_inform=False):
        self._initialization()
        # Logfile
        self.logfile = logfile
        # Other parameters
        self.lim_rows = lim_rows
        self.bool_inform = True if self.lim_rows != 0 else False
        self.prompt_inform = prompt_inform
        self.n_procs = 0
        self.proc_desc = "Computation %s with %s"
        self._create_subprocess_hierharchy([['prueba']])
        self.check_subprocess()

    def compute(self, n):
        # Main function to test the main utilities
        t00 = self.setting_global_process()
        # Begin to track the process
        t0, bun = self.setting_loop(n)
        t0_s = self.set_subprocess([0, 0])
        for i in range(n):
            ## Finish to track this process
            t0, bun = self.messaging_loop(i, t0, bun)
            i += 1
        self.close_subprocess([0, 0], t0_s)
        self.save_process_info('prueba')
        self.close_process(t00)


def test():
    ## Parameters
    logfile = Logger('logfile.log')

    ##### Test the auxiliar functions
    initial_message_creation(proc_name='a', proc_desc='fd')
    subprocess_desc, t_expended_subproc =\
        create_empty_list_from_hierarchylist(['', ['', '']])
    check_subprocess(subprocess_desc, t_expended_subproc)

    ### WARNING: Correct!
    store_time_subprocess([0], subprocess_desc, t_expended_subproc, 0)
    store_time_subprocess([1, 0], subprocess_desc, t_expended_subproc, 0)

    ##### Test the whole class
    dummy_proc = TesterProcesserClass(logfile)
    dummy_proc.compute(10000)
    ## Remove the files created
    try:
        os.remove('logfile.log')
        os.remove('prueba')
    except:
        pass

    dummy_proc = TesterProcesserClass(logfile, 1000)
    dummy_proc.compute(10000)
    ## Remove the files created
    try:
        os.remove('logfile.log')
        os.remove('prueba')
    except:
        pass

    dummy_proc = TesterProcesserClass(logfile, 1000, True)
    dummy_proc.compute(10000)
    ## Remove the files created
    try:
        os.remove('logfile.log')
        os.remove('prueba')
    except:
        pass
