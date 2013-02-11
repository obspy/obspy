# -*- coding: utf-8 -*-
"""
CSS bindings to ObsPy core module.
"""

import os
import struct
import numpy as np
from obspy import UTCDateTime, Trace, Stream


DTYPE = {'s4': "i", 't4': "f", 's2': "h"}


def isCSS(filename):
    """
    Checks whether a file is CSS waveform data (header) or not.

    :type filename: string
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
                assert(len(line.rstrip("\n\r")) == 283)
                assert(line[26] == ".")
                UTCDateTime(float(line[16:33]))
                assert(line[71] == ".")
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

    :type filename: string
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
        dirname = line[148:212].strip()
        filename = line[213:245].strip()
        filename = os.path.join(basedir, dirname, filename)
        offset = int(line[246:256])
        dtype = DTYPE[line[143:145]]
        fmt = ">" + dtype * npts
        with open(filename, "rb") as fh:
            fh.seek(offset)
            size = struct.calcsize(fmt)
            data = fh.read(size)
            data = struct.unpack(fmt, data)
            data = np.array(data)
        header = {}
        header['station'] = line[0:6].strip()
        header['channel'] = line[7:15].strip()
        header['starttime'] = UTCDateTime(float(line[16:33]))
        header['sampling_rate'] = float(line[88:99])
        header['calib'] = float(line[100:116])
        header['calper'] = float(line[117:133])
        tr = Trace(data, header=header)
        traces.append(tr)
    return Stream(traces=traces)
