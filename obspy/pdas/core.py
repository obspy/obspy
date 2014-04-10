# -*- coding: utf-8 -*-
"""
PDAS bindings to ObsPy core module.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
import numpy as np
from obspy.core import UTCDateTime, Stream, Trace


def isPDAS(filename):
    """
    Checks whether a file is a PDAS file or not.

    :type filename: str
    :param filename: Name of file to be checked.
    :rtype: bool
    :return: ``True`` if a PDAS file.

    .. rubric:: Example

    >>> isPDAS("/path/to/p1246001_cropped.108")
    True
    """
    with open(filename, "rb") as fh:
        header_fields = [fh.readline().split()[0] for i_ in xrange(11)]
    expected_headers = ['DATASET', 'FILE_TYPE', 'VERSION', 'SIGNAL',
                        'DATE', 'TIME', 'INTERVAL', 'VERT_UNITS',
                        'HORZ_UNITS', 'COMMENT', 'DATA']
    if header_fields == expected_headers:
        return True
    else:
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
    >>> st = read("/path/to/p1246001_cropped.108")
    >>> st  # doctest: +ELLIPSIS
    <obspy.core.stream.Stream object at 0x...>
    >>> print(st)  # doctest: +ELLIPSIS
    1 Trace(s) in Stream:
    .AYT..BHZ | 2002-12-23T12:48:00.000100Z - ... | 100.0 Hz, 18000 samples
    """
    extra_headers = {}
    with open(filename, "rb") as fh:
        items = [fh.readline().split() for i_ in xrange(11)]
        data = fh.read()
    for i_ in (0, 1, 2, 3, 7, 8, 9):
        extra_headers[items[i_][0]] = items[i_][1]
    month, day, year = items[4][1].split("-")
    if UTCDateTime().year > 2050:
        raise NotImplementedError()
    if int(year) < 50:
        year = "20" + year
    else:
        year = "19" + year
    time = items[5][1]
    t = UTCDateTime("%s-%s-%sT%s" % (year, month, day, time))
    sampling_rate = 1.0 / float(items[6][1])
    dtype = items[1][1]
    if dtype.upper() == "LONG":
        data = np.fromstring(data, dtype='int16')
    elif dtype.upper() == "SHORT":
        data = np.fromstring(data, dtype='int8')
    else:
        raise NotImplementedError()

    tr = Trace(data=data)
    tr.stats.starttime = t
    tr.stats.sampling_rate = sampling_rate
    tr.stats._format = "PDAS"
    tr.stats.pdas = extra_headers
    st = Stream(traces=[tr])
    return st
