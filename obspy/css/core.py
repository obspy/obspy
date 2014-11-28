# -*- coding: utf-8 -*-
"""
CSS bindings to ObsPy core module.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import os

import numpy as np

from obspy import UTCDateTime, Trace, Stream
from obspy.core.compatibility import frombuffer


DTYPE = {
    # Big-endian integers
    b's4': b'>i',
    b's2': b'>h',
    # Little-endian integers
    b'i4': b'<i',
    b'i2': b'<h',
    # ASCII integers
    b'c0': (b'S12', np.int),
    b'c#': (b'S12', np.int),
    # Big-endian floating point
    b't4': b'>f',
    b't8': b'>d',
    # Little-endian floating point
    b'f4': b'<f',
    b'f8': b'<d',
    # ASCII floating point
    b'a0': (b'S15', np.float32),
    b'a#': (b'S15', np.float32),
    b'b0': (b'S24', np.float64),
    b'b#': (b'S24', np.float64),
}


def isCSS(filename):
    """
    Checks whether a file is CSS waveform data (header) or not.

    :type filename: str
    :param filename: CSS file to be checked.
    :rtype: bool
    :return: ``True`` if a CSS waveform header file.
    """
    # Fixed file format.
    # Tests:
    #  - the length of each line (283 chars)
    #  - two epochal time fields
    #    (for position of dot and if they convert to UTCDateTime)
    #  - supported data type descriptor
    try:
        with open(filename, "rb") as fh:
            lines = fh.readlines()
            # check for empty file
            if not lines:
                return False
            # check every line
            for line in lines:
                assert(len(line.rstrip(b"\n\r")) == 283)
                assert(line[26:27] == b".")
                UTCDateTime(float(line[16:33]))
                assert(line[71:72] == b".")
                UTCDateTime(float(line[61:78]))
                assert(line[143:145] in DTYPE)
    except:
        return False
    return True


def readCSS(filename, **kwargs):
    """
    Reads a CSS waveform file and returns a Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.stream.read` function, call this instead.

    :type filename: str
    :param filename: CSS file to be read.
    :rtype: :class:`~obspy.core.stream.Stream`
    :returns: Stream with Traces specified by given file.
    """
    # read metafile with info on single traces
    with open(filename, "rb") as fh:
        lines = fh.readlines()
    basedir = os.path.dirname(filename)
    traces = []
    # read single traces
    for line in lines:
        npts = int(line[79:87])
        dirname = line[148:212].strip().decode()
        filename = line[213:245].strip().decode()
        filename = os.path.join(basedir, dirname, filename)
        offset = int(line[246:256])
        dtype = DTYPE[line[143:145]]
        if isinstance(dtype, tuple):
            read_fmt = np.dtype(dtype[0])
            fmt = dtype[1]
        else:
            read_fmt = np.dtype(dtype)
            fmt = read_fmt
        with open(filename, "rb") as fh:
            fh.seek(offset)
            data = fh.read(read_fmt.itemsize * npts)
            data = frombuffer(data, dtype=read_fmt)
            data = np.require(data, dtype=fmt)
        header = {}
        header['station'] = line[0:6].strip().decode()
        header['channel'] = line[7:15].strip().decode()
        header['starttime'] = UTCDateTime(float(line[16:33]))
        header['sampling_rate'] = float(line[88:99])
        header['calib'] = float(line[100:116])
        header['calper'] = float(line[117:133])
        tr = Trace(data, header=header)
        traces.append(tr)
    return Stream(traces=traces)
