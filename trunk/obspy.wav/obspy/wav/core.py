#!/usr/bin/env python
#-----------------------------------------------------------------------
# Filename: core.py
#  Purpose: Python Class for transforming seismograms to audio WAV files
#   Author: Moritz Beyreuther
#    Email: moritz.beyreuther@geophysik.uni-muenchen.de
#
# Copyright (C) 2009 Moritz Beyreuther
#-------------------------------------------------------------------------
""" 
WAV bindings to ObsPy core module.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from obspy.core import Trace, Stream
import numpy as np
import os
import wave


def isWAV(filename):
    """
    Checks whether a file is WAV or not. Returns True or False.

    Parameters
    ----------
    filename : string
        WAV file to be checked.
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


def readWAV(filename, headonly=False, **kwargs):
    """
    Read audio WAV file.
    
    Currently supports uncompressed unsigned char and short integer and
    integer data values. This should cover most WAV files. This function
    should NOT be called directly, it registers via the ObsPy
    :func:`~obspy.core.stream.read` function, call this instead.

    Parameters
    ----------
    filename : string
        Name of WAV file to read.

    Returns
    -------
    stream : :class:`~obspy.core.stream.Stream`
        A ObsPy Stream object.

    Basic Usage
    -----------
    >>> from obspy.core import read
    >>> st = read("/path/to/3cssan.near.8.1.RNON.wav")
    >>> print(st)
    1 Trace(s) in Stream:
    ... | 1970-01-01T00:00:00.000000Z - 1970-01-01T00:00:00.371143Z | 7000.0 Hz, 2599 samples
    """
    # read WAV file
    fh = wave.open(filename, 'rb')
    # header information
    (_nchannel, width, rate, length, _comptype, _compname) = fh.getparams()
    header = {'sampling_rate': rate, 'npts': length}
    if headonly:
        return Stream([Trace(header=header)])
    # WAVE data format is unsigned char up to 8bit, and signed int
    # for the remaining.
    if width == 1:
        fmt = '<u1' #unsigned char
    elif width == 2:
        fmt = '<i2' #signed short int
    elif width == 4:
        fmt = '<i4' #signed int (int32)
    else:
        raise TypeError("Unsupported Format Type, word width %dbytes" % width)
    data = np.fromstring(fh.readframes(length), dtype=fmt)
    fh.close()
    return Stream([Trace(header=header, data=data)])


def writeWAV(stream_object, filename, framerate=7000, **kwargs):
    """
    Write audio WAV file. The seismogram is squeezed to audible frequencies.

    The generated WAV sound file is as a result really short. The data
    are written uncompressed as signed 4-byte integers.

    This function should NOT be called directly, it registers via the
    ObsPy :meth:`~obspy.core.stream.Stream.write` method of an ObsPy
    Stream object, call this instead.

    .. note::
        The attributes `self.stats.npts` (number of samples) and
        `self.data` (array of data samples) are required

    Parameters
    ----------
    filename : string
        Name of WAV file to write.
    framerate : int, optional
        Sample rate of WAV file to use. This this will squeeze the seismogram
        (default is 7000). 
    """
    i = 0
    base , ext = os.path.splitext(filename)
    for trace in stream_object:
        # write WAV file
        if i != 0:
            filename = "%s%02d%s" % (base, i, ext)
        w = wave.open(filename, 'wb')
        trace.stats.npts = len(trace.data)
        # (nchannels, sampwidth, framerate, nframes, comptype, compname)
        w.setparams((1, 4, framerate, trace.stats.npts, 'NONE',
                     'not compressed'))
        trace.data = np.require(trace.data, '<i4')
        w.writeframes(trace.data.tostring())
        w.close()
        i += 1


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
