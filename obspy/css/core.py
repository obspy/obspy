# -*- coding: utf-8 -*-
"""
CSS bindings to ObsPy core module.
"""

from obspy import UTCDateTime, Stream


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
            for line in fh.readlines():
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
    :returns: Stream object containing header and data.
    """
    traces = []
    return Stream(traces=traces)
