

"""
Module which contains the abstract class of a process.
It generalize a common process in order to be easier to compute with tools
as display information of the time of the process and other things.

TODO:
-----
- Generalize the messaging
- Compulsary inputs
"""

import shelve
import time


###############################################################################
########### Global variables needed for this module
###############################################################################
message0 = """========================================
Start process %s:
--------------------
(%s)

"""
message1 = "Processing %s:"
message2 = "completed in %f seconds.\n"
message_loop = " %s bunch of %s iterations completed in %f seconds.\n"
message_last = "Total time expended computing process: %f seconds.\n"
message_close = '----------------------------------------\n'
###############################################################################
###############################################################################


###############################################################################
################################## MAIN CLASS #################################
###############################################################################
class Processer():
    """Abstract class for the processers some computations.
    """

    ### Class parameters
    ## Process descriptors
    time_expended = 0.  # Time expended along the process
    n_procs = 0  # Number of cpu used in parallelization (0 no parallel)
    proc_name = ""  # Name of the process
    proc_desc = ""  # Process description
    ## Logger info
    lim_rows = 0  # Lim of rows done in a bunch. For matrix comp or information
    logfile = None  # Log file
    ## Bool options
    bool_inform = False  # Give information of the process

    def save_process_info(self, outputfile):
        database = shelve.open(outputfile)
        out = self.to_dict()
        for e in out.keys():
            database[e] = out[e]
        database.close()

    def to_dict(self):
        "Transform the class information into a dictionary."
        out = {'time_expended': self.time_expended, 'n_procs': self.n_procs,
               'proc_name': self.proc_name, 'lim_rows': self.lim_rows,
               'logfile': self.logfile}
        return out

    def messaging_loop(self, i, t0, bun):
        "Message into the loop."
        if self.bool_inform and (i % self.lim_rows) == 0 and i != 0:
            t_sp = time.time()-t0
            bun += 1
            self.logfile.write_log(message_loop % (bun, self.lim_rows, t_sp))
            t0 = time.time()
        return t0, bun

    def close_process(self, t00):
        "Closing process."
        ## Closing process
        t_expended = time.time()-t00
        self.logfile.write_log(message_last % t_expended)
        self.logfile.write_log(message_close)
        self.time_expended = t_expended

    def setting_process(self):
        "Setting up the process."
        ## Initiating process
        message0 = initial_message_creation(self.proc_name, self.proc_desc)
        self.logfile.write_log(message0)
        t00 = time.time()
        return t00


def initial_message_creation(proc_name, proc_desc):
    line0 = "="*30
    line1 = "Start process %s:" % proc_name
    line2 = "-" * len(line1)
    line3 = "(%s)" % proc_desc
    return line0+"/n"+line1+"/n"+line2+"/n"+line3+"/n/n"
