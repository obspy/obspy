# -*- coding: utf-8 -*-
"""
CSS bindings to ObsPy core module.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import os
import sys

import numpy as np

from obspy import Stream, Trace, UTCDateTime
from obspy.core.compatibility import from_buffer
from obspy.core.util.deprecation_helpers import \
    DynamicAttributeImportRerouteModule


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


def _is_css(filename):
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
                fields = line.rstrip().split()
                assert(len(fields) >= 20)  # 20 fields in CSS 3.0, commid and lddate may have spaces
                assert(fields[2][-6] == b".")
                UTCDateTime(float(fields[2]))
                assert(fields[6][-6] == b".")
                UTCDateTime(float(fields[6]))
                assert(fields[13] in DTYPE)
    except:
        return False
    return True


def _read_css(filename, **kwargs):
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
        fields = line.rstrip().split(None,19)  # 20 fields in CSS 3.0; in my experience, lldate may have spaces
        npts = int(fields[7])
        dirname = fields[15].decode()
        filename = fields[16].decode()
        filename = os.path.join(basedir, dirname, filename)
        offset = int(fields[17])
        dtype = DTYPE[fields[13]]
        if isinstance(dtype, tuple):
            read_fmt = np.dtype(dtype[0])
            fmt = dtype[1]
        else:
            read_fmt = np.dtype(dtype)
            fmt = read_fmt
        with open(filename, "rb") as fh:
            fh.seek(offset)
            data = fh.read(read_fmt.itemsize * npts)
            data = from_buffer(data, dtype=read_fmt)
            data = np.require(data, dtype=fmt)
        header = {}
        header['station'] = fields[0].decode()
        header['channel'] = fields[1].decode()
        header['starttime'] = UTCDateTime(float(fields[2]))
        header['sampling_rate'] = float(fields[8])
        header['calib'] = float(fields[9])
        header['calper'] = float(fields[10])
        tr = Trace(data, header=header)
        traces.append(tr)
    return Stream(traces=traces)


# Remove once 0.11 has been released.
sys.modules[__name__] = DynamicAttributeImportRerouteModule(
    name=__name__, doc=__doc__, locs=locals(),
    original_module=sys.modules[__name__],
    import_map={},
    function_map={
        "isCSS": "obspy.io.css.core._is_css",
        "readCSS": "obspy.io.css.core._read_css"})
