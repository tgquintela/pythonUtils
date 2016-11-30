
"""
logging
-------
Main utilities to generate the log.

"""

from datetime import datetime
from os.path import exists

mark_dt_str = '\n'+'='*80+'\n'+'%s'+'\n'+'='*80+'\n'+'='*80+'\n\n'


class Logger:
    """Logger class provides a functionality to write in a logfile and display
    in screen a given message.

    """

    def __init__(self, logfile):
        """Logger instantiation.

        Parameters
        ----------
        logfile: str
            the path and name of the logfile.

        """
        self.logfile = logfile
        if not exists(logfile):
            initial = self.mark_datetime('Creation of the logfile')
            self.write_log(initial, False)

    def write_log(self, message, display=True):
        """Function to write and/or display in a screen a message.

        Parameters
        ----------
        message: srt
            the base message to add to the new line.
        display: boolean (default=True)
            if display in the terminal or not.

        """
        if display:
            print message
        append_line_file(self.logfile, message)  # +'\n')

    def mark_datetime(self, message=''):
        """Function to write the datetime in this momment.

        message: srt (default='')
            the base message to add to the new line.

        Returns
        -------
        m: str
            the message with datetime.

        """
        dtime = self.get_datetime()
        message = message+': ' if len(message) > 0 else message
        m = ' ' + message + dtime + ' '
        n = ((80-len(m))/2)
        m = n*'='+m+n*'='
        m = m+'=' if len(m) != 80 else m
        m = mark_dt_str % m
        return m

    def get_datetime(self):
        """Easy and quick way to get datetime.

        Returns
        -------
        dtime: datetime.datetime
            the datetime of the moment we execute.

        """
        dtime = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
        return dtime


def append_line_file(filename, line):
    """Append a line to the file.

    Parameters
    ----------
    filename: str
        the name of the file we want to add a line.
    line: str
        the containt of the line we want to add to the file.

    """
    f = open(filename, 'a')
    f.write(line)
    f.close()