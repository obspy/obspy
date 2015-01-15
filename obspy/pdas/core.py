# -*- coding: utf-8 -*-
"""
PDAS bindings to ObsPy core module.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport

import numpy as np
from obspy.core import UTCDateTime, Stream, Trace


def isPDAS(filename):
    """
    Checks whether a file is a PDAS file or not.

    :type filename: str
    :param filename: Name of file to be checked.
    :rtype: bool
    :return: ``True`` if a PDAS file.
    """
    try:
        with open(filename, "rb") as fh:
            header_fields = [fh.readline().split()[0].decode()
                             for i_ in range(11)]
        expected_headers = ['DATASET', 'FILE_TYPE', 'VERSION', 'SIGNAL',
                            'DATE', 'TIME', 'INTERVAL', 'VERT_UNITS',
                            'HORZ_UNITS', 'COMMENT', 'DATA']
        if header_fields == expected_headers:
            return True
        else:
            return False
    except:
        return False


def readPDAS(filename, **kwargs):
    """
    Reads a PDAS file and returns an ObsPy Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.stream.read` function, call this instead.

    :type filename: str
    :param filename: PDAS file to be read.
    :rtype: :class:`~obspy.core.stream.Stream`
    :return: An ObsPy Stream object.

    .. rubric:: Example

    >>> from obspy import read
    >>> st = read("/path/to/p1246001.108")
    >>> st  # doctest: +ELLIPSIS
    <obspy.core.stream.Stream object at 0x...>
    >>> print(st)  # doctest: +ELLIPSIS
    1 Trace(s) in Stream:
    ... | 1994-04-18T00:00:00.000000Z - ... | 200.0 Hz, 500 samples
    """
    extra_headers = {}
    with open(filename, "rb") as fh:
        items = [fh.readline().split() for i_ in range(11)]
        data = fh.read()
    for i_ in (0, 1, 2, 3, 7, 8, 9):
        extra_headers[items[i_][0].decode()] = items[i_][1].decode()
    month, day, year = items[4][1].decode().split("-")
    if UTCDateTime().year > 2050:
        raise NotImplementedError()
    if len(year) == 2:
        if int(year) < 50:
            year = "20" + year
        else:
            year = "19" + year
    time = items[5][1].decode()
    t = UTCDateTime("%s-%s-%sT%s" % (year, month, day, time))
    sampling_rate = 1.0 / float(items[6][1].decode())
    dtype = items[1][1].decode()
    if dtype.upper() == "LONG":
        data = np.fromstring(data, dtype=np.int16)
    elif dtype.upper() == "SHORT":
        data = np.fromstring(data, dtype=np.int8)
    else:
        raise NotImplementedError()

    tr = Trace(data=data)
    tr.stats.starttime = t
    tr.stats.sampling_rate = sampling_rate
    tr.stats._format = "PDAS"
    tr.stats.pdas = extra_headers
    st = Stream(traces=[tr])
    return st
