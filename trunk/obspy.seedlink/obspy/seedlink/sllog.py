# -*- coding: utf-8 -*-
"""
Module to manage the logging of informatoin and error messages.

Part of Python implementaion of libslink of Chad Trabant and
JSeedLink of Anthony Lomax

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""


import sys

class SLLog(object):
    """ 
    Class to manage the logging of informatoin and error messages.
    
    :var log: The stream used for output of informartion messages.
    (default is sys.stdout).
    :type log: file
    /**  */
    protected PrintStream log = System.out;
    
    :var logPrefix: The prefix to prepend to informartion messages
    (default is an empty string).
    :type logPrefix: str
    
    :var err: The stream used for output of error messages.
    (default is sys.stderr).
    :type err: file
    
    :var errPrefix: The prefix to prepend to error messages.
    (default is ERROR:).
    :type errPrefix: file
    
    :var verbosity: Verbosity level, 0 is lowest (default is 0).
    :type verbosity: int

    """

    logOut = sys.stdout
    logPrefix = ""
    errOut = sys.stderr
    errPrefix = "ERROR: "
    verbosity = 0

    def __init__(self, verbosity=None, logOut=None, logPrefix=None,
                 errOut=None, errPrefix=None):
        """
        Creates a new instance of an SLLog.

        :param: verbosity verbosity level, 0 is lowest.
        :param: log stream used for output of informartion messages.
        :param: logPrefix prefix to prepend to informartion messages.
        :param: err stream used for output of error messages.
        :param: errPrefix prefix to prepend to error messages.

        """
        if verbosity is not None:
            self.verbosity = verbosity
        if logOut is not None:
            self.logOut = logOut
        else:
            self.logOut = sys.stdout
        if logPrefix is not None:
            self.logPrefix = logPrefix
        if errOut is not None:
            self.errOut = errOut
        else:
            self.errOut = sys.stderr
        if errPrefix is not None:
            self.errPrefix = errPrefix

    def log(self, isError, verbosity, message):
        """
        Logs a message in appropriate manner.

        :param: isError true if error message, false otherwise.
        :param: verbosity verbosity level for this messages.
        :param: message message text.

        """

        if verbosity > self.verbosity:
            return

        if isError:
            self.errOut.write(self.errPrefix + str(message) + '\n')
        else:
            self.logOut.write(self.logPrefix + str(message) + '\n')


