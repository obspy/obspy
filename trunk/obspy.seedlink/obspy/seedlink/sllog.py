# -*- coding: utf-8 -*-
"""
Module to manage the logging of information and error messages.

Part of Python implementation of libslink of Chad Trabant and
JSeedLink of Anthony Lomax

:copyright:
    The ObsPy Development Team (devs@obspy.org) & Anthony Lomax
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

import sys


class SLLog(object):
    """
    Class to manage the logging of information and error messages.

    :var log_out: The stream used for output of information messages.
        (default is sys.stdout).
    :type log_out: file
    :var log_prefix: The prefix to prepend to information messages
        (default is an empty string).
    :type log_prefix: str
    :var err_out: The stream used for output of error messages.
        (default is sys.stderr).
    :type err_out: file
    :var err_prefix: The prefix to prepend to error messages.
        (default is ERROR:).
    :type err_prefix: file
    :var verbosity: Verbosity level, 0 is lowest (default is 0).
    :type verbosity: int
    """
    log_out = sys.stdout
    log_prefix = ""
    err_out = sys.stderr
    err_prefix = "ERROR: "
    verbosity = 0

    def __init__(self, verbosity=None, log_out=None, log_prefix=None,
                 err_out=None, err_prefix=None):
        """
        Creates a new instance of an SLLog.

        :param verbosity: verbosity level, 0 is lowest.
        :param log_out: log stream used for output of information messages.
        :param log_prefix: prefix to prepend to information messages.
        :param err_out: stream used for output of error messages.
        :param err_prefix: errPrefix prefix to prepend to error messages.
        """
        if verbosity is not None:
            self.verbosity = verbosity
        if log_prefix is not None:
            self.log_prefix = log_prefix
        if err_prefix is not None:
            self.err_prefix = err_prefix
        self.log_out = log_out or sys.stdout
        self.err_out = err_out or sys.stderr

    def log(self, is_error, verbosity, message):
        """
        Logs a message in appropriate manner.

        :param is_error: true if error message, false otherwise.
        :param verbosity: verbosity level for this messages.
        :param message: message text.
        """
        if verbosity > self.verbosity:
            return
        if is_error:
            self.err_out.write("%s%s\n" % (self.err_prefix, message))
        else:
            self.log_out.write("%s%s\n" % (self.log_prefix, message))
