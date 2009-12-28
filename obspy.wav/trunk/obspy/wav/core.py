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
Python Class for transforming seismograms to audio WAV files

@copyright: The ObsPy Development Team (devs@obspy.org)
@license: GNU Lesser General Public License, Version 3 (LGPLv3)
"""

from obspy.core import Trace, Stream
import numpy as np
import wave
import os


def isWAV(filename):
    # read WAV file
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
    integer data values. This should cover most WAV files.
    
    :param filename: Name of WAV file to read.
    """
    # read WAV file
    fh = wave.open(filename, 'rb')
    # header information
    (nchannel, width, rate, length, _comptype, _compname) = fh.getparams()
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
    
    The resulting WAV sound file is as a result really short. The data
    are written uncompressed as signed 4byte integers.
    
    :note: The attributes self.stats.npts = number of samples and
           self.data = array of data samples are required
    :param filename: Name of WAV file to write.
    :param framerate: Sample rate of WAV file to use. This this will
                      squeeze the seismogram, DEFAULT=7000. 
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
