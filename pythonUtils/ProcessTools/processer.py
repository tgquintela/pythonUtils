
"""
processer
---------
The main processer tools module. Module which contains the abstract class of a
process. It generalize a common process in order to be easier to compute with
tools as display information of the time of the process and other things.

TODO:
-----
- Generalize the messaging
- Compulsary inputs
- Process with pipelines and subprocesses

"""

import shelve
import time
import copy


###############################################################################
########### Global variables needed for this module
###############################################################################
message0 = """========================================
Start process %s:
--------------------
(%s)

"""
message1 = "%s: "
message2 = "completed in %f seconds.\n"
message_init_loop = "Total number of iterations to compute: %s"
message_loop = " %s bunch of %s iterations completed in %f seconds.\n"

message_close0 = '-'*70+"\n"
message_last = "Total time expended computing the process: %f seconds.\n"
message_close = '-'*70+'\n'
###############################################################################
###############################################################################


###############################################################################
################################## MAIN CLASS #################################
###############################################################################
class Processer(object):
    """Abstract class for the processers some computations.
    """

    def _initialization(self):
        ### Class parameters
        ## Process descriptors
        self.time_expended = 0.  # Time expended along the process
        self.t_expended_subproc = []  # Time expended in each subprocesses
        self.n_procs = 0  # Number of cpu used in parallelization
        self.proc_name = ""  # Name of the process
        self.proc_desc = ""  # Process description
        self.subproc_desc = []  # Subprocess description
        ## Logger info
        self.lim_rows = 0  # Lim of rows done in a bunch.
        self.logfile = None  # Log file
        ## Bool options
        self.bool_inform = False  # Give information of the process
        self.prompt_inform = False  # Prompt the information in the screen

    def save_process_info(self, outputfile):
        """Export function to save some variables usable.

        Parameters
        ----------
        outputfile: str
            the output file we want to store all the interesting information
            which defines and describes the process.

        """
        database = shelve.open(outputfile)
        out = self.to_dict()
        for e in out.keys():
            database[e] = out[e]
        database.close()

    def to_dict(self):
        """Transform the class information into a dictionary.

        Returns
        -------
        out: dict
            the dictionary of variables interesting to be tracked and the
            ones which defines the process.

        """
        out = {'time_expended': self.time_expended, 'n_procs': self.n_procs,
               'proc_name': self.proc_name, 'lim_rows': self.lim_rows,
               'logfile': self.logfile}
        return out

    def setting_loop(self, N_t):
        """Setting loop for the use of the processer class.

        Parameters
        ----------
        N_t: int
            the total number of loop iterations.

        Returns
        -------
        t0: float
            the time at the moment of initialization of the loop bunch.
        bun: int
            the index of the bunch.

        """
        self.logfile.write_log(message_init_loop % str(N_t),
                               self.prompt_inform)
        t0, bun = time.time(), 0
        return t0, bun

    def messaging_loop(self, i, t0, bun):
        """Message into the loop.

        Parameters
        ----------
        i: int
            the index of the loop.
        t0: float
            the time at the moment of initialization of the loop bunch.
        bun: int
            the index of the bunch.

        Returns
        -------
        t0: float
            the time at the moment of initialization of the loop bunch.
        bun: int
            the index of the bunch.

        """
        if self.bool_inform and (i % self.lim_rows) == 0 and i != 0:
            t_sp = time.time()-t0
            bun += 1
            self.logfile.write_log(message_loop % (bun, self.lim_rows, t_sp),
                                   self.prompt_inform)
            t0 = time.time()
        return t0, bun

    def close_process(self, t00):
        """Closing process.

        Parameters
        ----------
        t00: float
            the time at the moment of initialization of the process.

        """
        ## Closing process
        t_expended = time.time()-t00
        self.logfile.write_log(message_close0, self.prompt_inform)
        self.logfile.write_log(message_last % t_expended, self.prompt_inform)
        self.logfile.write_log(message_close, self.prompt_inform)
        self.time_expended = t_expended

    def setting_global_process(self):
        """Setting up the process.

        Returns
        -------
        t00: float
            the time at the moment of initialization of the process.

        """
        ## Initiating process
        message0 = initial_message_creation(self.proc_name, self.proc_desc)
        self.logfile.write_log(message0, self.prompt_inform)
        t00 = time.time()
        return t00

    def set_subprocess(self, index_sub):
        """Set subprocess.

        Parameters
        ----------
        index_sub: list
            the indices of the subprocess.

        Returns
        -------
        t0: float
            the time at the moment of initialization of the subprocess.

        """
        aux = copy.copy(self.subproc_desc)
        for i in index_sub:
            aux = aux[i]
        message1 % aux
        t0 = time.time()
        return t0

    def _create_subprocess_hierharchy(self, subprocess_desc):
        """Create the subprocess hierarchy.

        Parameters
        ----------
        subprocess_desc: list
            the subprocesses descriptions.

        """
        _, t_expended_subproc =\
            create_empty_list_from_hierarchylist(subprocess_desc)
        check_subprocess(subprocess_desc, t_expended_subproc)
        self.subproc_desc = subprocess_desc
        self.t_expended_subproc = t_expended_subproc

    def close_subprocess(self, index_sub, t0):
        """Close subprocesses at different levels.

        Parameters
        ----------
        index_sub: list
            the indices of the subprocess.
        t0: float
            the time at the moment of initialization of the subprocess.

        """
        ## Save time
        t_expended = time.time()-t0
        self.t_expended_subproc, proc_name =\
            store_time_subprocess(index_sub, self.subproc_desc,
                                  self.t_expended_subproc, t_expended)
        ## Logfile writing
        self.logfile.write_log(message1 % proc_name, self.prompt_inform)
        self.logfile.write_log(message2 % t_expended, self.prompt_inform)

    def check_subprocess(self):
        """Check if the subprocess is properly formatted.

        Returns
        -------
        bool_res: boolean
            if the subprocess information is well formatted.

        """
        n1 = len(self.t_expended_subproc)
        n2 = len(self.subproc_desc)
        bool_res = True
        bool_res = bool_res and (n1 == n2)
        for i in range(n2):
            if type(self.subproc_desc[i]) == list:
                aux = type(self.t_expended_subproc[i]) == list
                bool_res = bool_res and aux
                n1a = len(self.t_expended_subproc[i])
                n2a = len(self.subproc_desc[i])
                bool_res = bool_res and (n1a == n2a)
        return bool_res


###############################################################################
## The auxiliar functions of the processers
##
##
def initial_message_creation(proc_name, proc_desc):
    """Initial message creation.

    Parameters
    ----------
    proc_name: str
        the name of the process.
    proc_desc: str
        the description of the process.

    Returns
    -------
    message: str
        the initial message.

    """
    line0 = "="*30
    line1 = "Start process %s:" % proc_name
    line2 = "-" * len(line1)
    line3 = "(%s)" % proc_desc
    return line0+"\n"+line1+"\n"+line2+"\n"+line3+"\n\n"


def store_time_subprocess(index_sub, subproc_desc, t_expended_subproc,
                          t_expended):
    """Recursevely indexing t_expended_subproc list.

    Parameters
    ----------
    index_sub: list
        the indices of the subprocess.
    subproc_desc: list
        the descriptions of each subprocess.
    t_expended_subproc: float
        the time expended in each subprocess.
    t_expended: float
        the time expended in the whole process.

    Returns
    -------
    t_expended_subproc: float
        the time expended in each subprocess.
    proc_name: str
        the process name.

    """
    if len(index_sub) == 1:
        t_expended_subproc[index_sub[0]] = t_expended
        proc_name = subproc_desc[index_sub[0]]
        return t_expended_subproc, proc_name
    else:
        t_expended_subproc[index_sub[0]], proc_name =\
            store_time_subprocess(index_sub[1:], subproc_desc[index_sub[0]],
                                  t_expended_subproc[index_sub[0]], t_expended)
    return t_expended_subproc, proc_name


def create_empty_list_from_hierarchylist(hierarchylist):
    """Create empty list of hierarchy.

    Parameters
    ----------
    hierarchylist: list
        the hierarchy defined.

    Returns
    -------
    hierarchylist: list
        the hierarchy defined.
    tofilllist: list
        the list with the same hierarchy than the required, and empty to
        be filled.

    """
    tofilllist = []
    for i in range(len(hierarchylist)):
        if type(hierarchylist[i]) == str:
            tofilllist.append(0)
        else:
            hierarchylist_i, tofilllist_i =\
                create_empty_list_from_hierarchylist(hierarchylist[i])
            tofilllist.append(tofilllist_i)
    return hierarchylist, tofilllist


def check_subprocess(subprocess_desc, t_expended_subproc):
    """Check the subprocess recursevely.

    Parameters
    ----------
    subprocess_desc: list
        the subprocesses descriptions.
    t_expended_subproc: list
        the time expended in each subprocess.

    """
    if type(subprocess_desc) == list:
        assert(type(t_expended_subproc) == list)
        assert(len(subprocess_desc) == len(t_expended_subproc))
        for i in range(len(subprocess_desc)):
            check_subprocess(subprocess_desc[i], t_expended_subproc[i])
