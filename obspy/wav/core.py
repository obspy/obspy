#!/usr/bin/env python
#-----------------------------------------------------------------------
# Filename: core.py
#  Purpose: Python Class for transforming seismograms to audio WAV files
#   Author: Moritz Beyreuther
#    Email: moritz.beyreuther@geophysik.uni-muenchen.de
#
# Copyright (C) 2009-2012 Moritz Beyreuther
#-------------------------------------------------------------------------
"""
WAV bindings to ObsPy core module.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from __future__ import division
from obspy import Trace, Stream
import numpy as np
import os
import wave


# WAVE data format is unsigned char up to 8bit, and signed int
# for the remaining.
width2dtype = {
    1: '<u1',  # unsigned char
    2: '<i2',  # signed short int
    4: '<i4',  # signed int (int32)
}


def isWAV(filename):
    """
    Checks whether a file is a audio WAV file or not.

    :type filename: str
    :param filename: Name of the audio WAV file to be checked.
    :rtype: bool
    :return: ``True`` if a WAV file.

    .. rubric:: Example

    >>> isWAV("/path/to/3cssan.near.8.1.RNON.wav")  #doctest: +SKIP
    True
    """
    try:
        fh = wave.open(filename, 'rb')
        (_nchannel, width, _rate, _len, _comptype, _compname) = fh.getparams()
        fh.close()
    except:
        return False
    if width == 1 or width == 2 or width == 4:
        return True
    return False


def readWAV(filename, headonly=False, **kwargs):  # @UnusedVariable
    """
    Reads a audio WAV file and returns an ObsPy Stream object.

    Currently supports uncompressed unsigned char and short integer and
    integer data values. This should cover most WAV files.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.stream.read` function, call this instead.

    :type filename: str
    :param filename: Audio WAV file to be read.
    :rtype: :class:`~obspy.core.stream.Stream`
    :return: A ObsPy Stream object.

    .. rubric:: Example

    >>> from obspy import read
    >>> st = read("/path/to/3cssan.near.8.1.RNON.wav")
    >>> print(st) #doctest: +NORMALIZE_WHITESPACE
    1 Trace(s) in Stream:
    ... | 1970-01-01T00:00:00.000000Z - 1970-01-01T00:00:00.371143Z
    | 7000.0 Hz, 2599 samples
    """
    # read WAV file
    fh = wave.open(filename, 'rb')
    # header information
    (_nchannel, width, rate, length, _comptype, _compname) = fh.getparams()
    header = {'sampling_rate': rate, 'npts': length}
    if headonly:
        return Stream([Trace(header=header)])
    if width not in width2dtype.keys():
        raise TypeError("Unsupported Format Type, word width %dbytes" % width)
    data = np.fromstring(fh.readframes(length), dtype=width2dtype[width])
    fh.close()
    return Stream([Trace(header=header, data=data)])


def writeWAV(stream, filename, framerate=7000, rescale=False, width=4,
             **kwargs):  # @UnusedVariable
    """
    Writes a audio WAV file from given ObsPy Stream object. The seismogram is
    squeezed to audible frequencies.

    The generated WAV sound file is as a result really short. The data
    are written uncompressed as signed 4-byte integers.

    .. warning::
        This function should NOT be called directly, it registers via the
        the :meth:`~obspy.core.stream.Stream.write` method of an
        ObsPy :class:`~obspy.core.stream.Stream` object, call this instead.

    :type stream: :class:`~obspy.core.stream.Stream`
    :param stream: The ObsPy Stream object to write.
    :type filename: str
    :param filename: Name of the audio WAV file to write.
    :type framerate: int, optional
    :param framerate: Sample rate of WAV file to use. This this will squeeze
        the seismogram (default is 7000).
    :type rescale: bool, optional
    :param rescale: Maximum to maximal representable number
    :type width: int, optimal
    :param width: dtype to write, 1 for '<u1', 2 for '<i2' or 4 for '<i4'.
    """
    i = 0
    base, ext = os.path.splitext(filename)
    if width not in width2dtype.keys():
        raise TypeError("Unsupported Format Type, word width %dbytes" % width)
    for trace in stream:
        # write WAV file
        if len(stream) >= 2:
            filename = "%s%03d%s" % (base, i, ext)
        w = wave.open(filename, 'wb')
        trace.stats.npts = len(trace.data)
        # (nchannels, sampwidth, framerate, nframes, comptype, compname)
        w.setparams((1, width, framerate, trace.stats.npts, 'NONE',
                     'not compressed'))
        data = trace.data
        if rescale:
            # optimal scale, account for +/- and the zero
            data = (2 ** (width * 8 - 1) - 1) * \
                data.astype('f8') / abs(data).max()
        data = np.require(data, dtype=width2dtype[width])
        w.writeframes(data.tostring())
        w.close()
        i += 1


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
