# -*- coding: utf-8 -*-
"""
Various additional utilities for ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import division
from __future__ import unicode_literals
from future.builtins import str
from future.builtins import int
from future.builtins import open
from contextlib import contextmanager
import os
import sys
import inspect
from subprocess import Popen, PIPE
import warnings
import itertools
import tempfile
import numpy as np


# The following dictionary maps the first character of the channel_id to the
# lowest sampling rate this so called Band Code should be used for according
# to: SEED MANUAL p.124
# We use this e.g. in seihub.client.getWaveform to request two samples more on
# both start and end to cut to the samples that really are nearest to requested
# start/endtime afterwards.
BAND_CODE = {'F': 1000.0,
             'G': 1000.0,
             'D': 250.0,
             'C': 250.0,
             'E': 80.0,
             'S': 10.0,
             'H': 80.0,
             'B': 10.0,
             'M': 1.0,
             'L': 1.0,
             'V': 0.1,
             'U': 0.01,
             'R': 0.0001,
             'P': 0.000001,
             'T': 0.0000001,
             'Q': 0.00000001}


def guessDelta(channel):
    """
    Estimate time delta in seconds between each sample from given channel name.

    :type channel: str
    :param channel: Channel name, e.g. ``'BHZ'`` or ``'H'``
    :rtype: float
    :return: Returns ``0`` if band code is not given or unknown.

    .. rubric:: Example

    >>> print(guessDelta('BHZ'))
    0.1

    >>> print(guessDelta('H'))
    0.0125

    >>> print(guessDelta('XZY'))  # doctest: +SKIP
    0
    """
    try:
        return 1. / BAND_CODE[channel[0]]
    except:
        msg = "No or unknown channel id provided. Specifying a channel id " + \
              "could lead to better selection of first/last samples of " + \
              "fetched traces."
        warnings.warn(msg)
    return 0


def scoreatpercentile(values, per, limit=(), issorted=True):
    """
    Calculates the score at the given per percentile of the sequence a.

    For example, the score at ``per=50`` is the median.

    If the desired quantile lies between two data points, we interpolate
    between them.

    If the parameter ``limit`` is provided, it should be a tuple (lower,
    upper) of two values.  Values of ``a`` outside this (closed) interval
    will be ignored.

    .. rubric:: Examples

    >>> a = [1, 2, 3, 4]
    >>> scoreatpercentile(a, 25)
    1.75
    >>> scoreatpercentile(a, 50)
    2.5
    >>> scoreatpercentile(a, 75)
    3.25

    >>> a = [6, 47, 49, 15, 42, 41, 7, 255, 39, 43, 40, 36, 500]
    >>> scoreatpercentile(a, 25, limit=(0, 100))
    25.5
    >>> scoreatpercentile(a, 50, limit=(0, 100))
    40
    >>> scoreatpercentile(a, 75, limit=(0, 100))
    42.5

    This function is taken from :func:`scipy.stats.scoreatpercentile`.

    Copyright (c) Gary Strangman
    """
    if limit:
        values = [v for v in values if limit[0] < v < limit[1]]

    if issorted:
        values = sorted(values)

    def _interpolate(a, b, fraction):
        return a + (b - a) * fraction

    idx = per / 100. * (len(values) - 1)
    if (idx % 1 == 0):
        return values[int(idx)]
    else:
        return _interpolate(values[int(idx)], values[int(idx) + 1], idx % 1)


def flatnotmaskedContiguous(a):
    """
    Find contiguous unmasked data in a masked array along the given axis.

    This function is taken from
    :func:`numpy.ma.extras.flatnotmasked_contiguous`.

    Copyright (c) Pierre Gerard-Marchant
    """
    np.ma.extras.flatnotmasked_contiguous
    m = np.ma.getmask(a)
    if m is np.ma.nomask:
        return slice(0, a.size, None)
    i = 0
    result = []
    for (k, g) in itertools.groupby(m.ravel()):
        n = len(list(g))
        if not k:
            result.append(slice(i, i + n))
        i += n
    return result or None


def complexifyString(line):
    """
    Converts a string in the form "(real, imag)" into a complex type.

    :type line: str
    :param line: String in the form ``"(real, imag)"``.
    :rtype: complex
    :return: Complex number.

    .. rubric:: Example

    >>> complexifyString("(1,2)")
    (1+2j)

    >>> complexifyString(" ( 1 , 2 ) ")
    (1+2j)
    """
    temp = line.split(',')
    return complex(float(temp[0].strip()[1:]), float(temp[1].strip()[:-1]))


def toIntOrZero(value):
    """
    Converts given value to an integer or returns 0 if it fails.

    :param value: Arbitrary data type.
    :rtype: int
numpy.version.version
    .. rubric:: Example

    >>> toIntOrZero("12")
    12

    >>> toIntOrZero("x")
    0
    """
    try:
        return int(value)
    except ValueError:
        return 0


# import numpy loadtxt and check if ndlim parameter is available
try:
    from numpy import loadtxt
    loadtxt(np.array([]), ndlim=1)
except TypeError:
    # otherwise redefine loadtxt
    def loadtxt(*args, **kwargs):
        """
        Replacement for older numpy.loadtxt versions not supporting ndlim
        parameter.
        """
        if not 'ndlim' in kwargs:
            return np.loadtxt(*args, **kwargs)
        # ok we got a ndlim param
        if kwargs['ndlim'] != 1:
            # for now we support only one dimensional arrays
            raise NotImplementedError('Upgrade your NumPy version!')
        del kwargs['ndlim']
        dtype = kwargs.get('dtype', None)
        # lets get the data
        try:
            data = np.loadtxt(*args, **kwargs)
        except IOError as e:
            # raises in older versions if no data could be read
            if 'reached before encountering data' in str(e):
                # return empty array
                return np.array([], dtype=dtype)
            # otherwise just raise
            raise
        # ensures that an array is returned
        return np.atleast_1d(data)


def get_untracked_files_from_git():
    """
    Tries to return a list of files (absolute paths) that are untracked by git
    in the repository.

    Returns `None` if the system call to git fails.
    """
    dir_ = os.path.abspath(
        os.path.dirname(inspect.getfile(inspect.currentframe())))
    dir_ = os.path.dirname(os.path.dirname(os.path.dirname(dir_)))
    try:
        # Check that the git root directory is actually the ObsPy directory.
        p = Popen(['git', 'rev-parse', '--show-toplevel'],
                  cwd=dir_, stdout=PIPE, stderr=PIPE)
        p.stderr.close()
        git_root_dir = p.stdout.readlines()[0].strip()
        if git_root_dir != dir_:
            raise Exception
        p = Popen(['git', 'status', '-u', '--porcelain'],
                  cwd=dir_, stdout=PIPE, stderr=PIPE)
        p.stderr.close()
        stdout = p.stdout.readlines()
        p.stdout.close()
        files = [os.path.abspath(os.path.join(dir_, line.split()[1].strip()))
                 for line in stdout
                 if line.startswith("??")]
    except:
        return None
    return files


def wrap_long_string(string, line_length=79, prefix="",
                     special_first_prefix=None, assumed_tab_width=8,
                     sloppy=False):
    """
    Reformat a long string, wrapping it to a specified length.

    :type string: str
    :param string: Input string to wrap
    :type line_length: int
    :param line_length: total target length of each line, including the
        prefix if specified
    :type prefix: str, optional
    :param prefix: common prefix used to start the line (e.g. some spaces,
        tabs for indentation)
    :type special_first_prefix: str, optional
    :param special_first_prefix: special prefix to use on the first line,
        instead of the general prefix
    :type assumed_tab_width: int
    :param assumed_tab_width: if the prefix strings include tabs the line
        length can not be computed exactly. assume a tab in general is
        equivalent to this many spaces.
    :type sloppy: bool
    :param sloppy: Controls the behavior when a single word without spaces is
        to long to fit on a single line. Default (False) is to allow a single
        line to be longer than the specified line length. If set to True,
        Long words will be force-hyphenated to fit the line.

    .. rubric:: Examples

    >>> string = ("Retrieve an event based on the unique origin "
    ...           "ID numbers assigned by the IRIS DMC")
    >>> print(wrap_long_string(string, prefix="\t*\t > ",
    ...                        line_length=50))  # doctest: +SKIP
            *        > Retrieve an event based on
            *        > the unique origin ID numbers
            *        > assigned by the IRIS DMC
    >>> print(wrap_long_string(string, prefix="\t* ",
    ...                        line_length=70))  # doctest: +SKIP
            * Retrieve an event based on the unique origin ID
            * numbers assigned by the IRIS DMC
    >>> print(wrap_long_string(string, prefix="\t \t  > ",
    ...                        special_first_prefix="\t*\t",
    ...                        line_length=50))  # doctest: +SKIP
            *        Retrieve an event based on
                     > the unique origin ID numbers
                     > assigned by the IRIS DMC
    >>> problem_string = ("Retrieve_an_event_based_on_the_unique "
    ...                   "origin ID numbers assigned by the IRIS DMC")
    >>> print(wrap_long_string(problem_string, prefix="\t\t",
    ...                        line_length=40, sloppy=True))  # doctest: +SKIP
                    Retrieve_an_event_based_on_the_unique
                    origin ID
                    numbers
                    assigned by
                    the IRIS DMC
    >>> print(wrap_long_string(problem_string, prefix="\t\t",
    ...                        line_length=40))  # doctest: +SKIP
                    Retrieve_an_event_base\
                    d_on_the_unique origin
                    ID numbers assigned by
                    the IRIS DMC
    """
    def text_width_for_prefix(line_length, prefix):
        text_width = line_length - len(prefix) - \
            (assumed_tab_width - 1) * prefix.count("\t")
        return text_width

    lines = []
    if special_first_prefix is not None:
        text_width = text_width_for_prefix(line_length, special_first_prefix)
    else:
        text_width = text_width_for_prefix(line_length, prefix)

    while len(string) > text_width:
        ind = string.rfind(" ", 0, text_width)
        # no suitable place to split found
        if ind < 1:
            # sloppy: search to right for space to split at
            if sloppy:
                ind = string.find(" ", text_width)
                if ind == -1:
                    ind = len(string) - 1
                part = string[:ind]
                string = string[ind + 1:]
            # not sloppy: force hyphenate
            else:
                ind = text_width - 2
                part = string[:ind] + "\\"
                string = string[ind:]
        # found a suitable place to split
        else:
            part = string[:ind]
            string = string[ind + 1:]
        # need to use special first line prefix?
        if special_first_prefix is not None and not lines:
            line = special_first_prefix + part
        else:
            line = prefix + part
        lines.append(line)
        # need to set default text width, just in case we had a different
        # text width for the first line
        text_width = text_width_for_prefix(line_length, prefix)
    lines.append(prefix + string)
    return "\n".join(lines)


@contextmanager
def CatchOutput():
    """
    A context manager that catches stdout/stderr for its scope.

    Always use with "with" statement. Does nothing otherwise.

    Roughly based on: http://stackoverflow.com/a/17954769

    This variant does not leak file descriptors.

    >>> with CatchOutput() as out:  # doctest: +SKIP
    ...    os.system('echo "mystdout"')
    ...    os.system('echo "mystderr" >&2')
    >>> print(out.stdout)  # doctest: +SKIP
    mystdout
    >>> print(out.stderr)  # doctest: +SKIP
    mystderr
    """
    stdout_file, stdout_filename = tempfile.mkstemp(prefix="obspy-")
    stderr_file, stderr_filename = tempfile.mkstemp(prefix="obspy-")

    try:
        fd_stdout = sys.stdout.fileno()
        fd_stderr = sys.stderr.fileno()

        # Dummy class to transport the output.
        class Output():
            pass
        out = Output()
        out.stdout = ""
        out.stderr = ""

        with os.fdopen(os.dup(sys.stdout.fileno()), "w") as old_stdout:
            with os.fdopen(os.dup(sys.stderr.fileno()), "w") as old_stderr:
                sys.stdout.flush()
                sys.stderr.flush()

                os.dup2(stdout_file, fd_stdout)
                os.dup2(stderr_file, fd_stderr)

                os.close(stdout_file)
                os.close(stderr_file)

                try:
                    yield out
                finally:
                    sys.stdout.flush()
                    sys.stderr.flush()
                    os.fsync(sys.stdout.fileno())
                    os.fsync(sys.stderr.fileno())
                    sys.stdout = sys.__stdout__
                    sys.stderr = sys.__stderr__
                    sys.stdout.flush()
                    sys.stderr.flush()
                    os.dup2(old_stdout.fileno(), sys.stdout.fileno())
                    os.dup2(old_stderr.fileno(), sys.stderr.fileno())

                    with open(stdout_filename, "r") as fh:
                        out.stdout = fh.read()
                    with open(stderr_filename, "r") as fh:
                        out.stderr = fh.read()

    finally:
        # Make sure to always close and remove the temporary files.
        try:
            os.close(stdout_file)
        except:
            pass
        try:
            os.close(stderr_file)
        except:
            pass
        try:
            os.remove(stdout_filename)
        except OSError:
            pass
        try:
            os.remove(stderr_filename)
        except OSError:
            pass


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
