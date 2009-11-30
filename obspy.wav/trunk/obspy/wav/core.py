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

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""

from obspy.core import Trace, Stream
import numpy as N
import struct
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
    if width == 1 or width == 2:
        return True
    return False


def readWAV(filename, headonly=False, **kwargs):
    """
    Read audio WAV file.
    
    Currently supports unsigned char and short integer data values. This
    should cover most WAV files.
    
    @param filename: Name of WAV file to read.
    """
    # read WAV file
    fh = wave.open(filename, 'rb')
    # header information
    (nchannel, width, rate, length, _comptype, _compname) = fh.getparams()
    header = {'sampling_rate': rate, 'npts': length}
    if headonly:
        return Stream([Trace(header=header)])
    # set format
    if width == 1:
        format = 'B'
    elif width == 2:
        format = 'h'
    else:
        raise TypeError("Unsupported Format Type, string length %d" % length)
    data = struct.unpack("%d%s" % (length * nchannel, format),
                         fh.readframes(length))
    fh.close()
    return Stream([Trace(header=header, data=N.array(data))])


def writeWAV(stream_object, filename, framerate=7000, **kwargs):
    """
    Write audio WAV file. The seismogram is squeezed to audible frequencies.
    
    The resulting WAV sound file is as a result really short. The data
    are written uncompressed as unsigned char.
    
    @requires: The attributes self.stats.npts = number of samples; 
        self.data = array of data samples.
    @param filename: Name of WAV file to write.
    @param framerate: Sample rate of WAV file to use. This this will
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
        w.setparams((1, 1, framerate, trace.stats.npts, 'NONE',
                     'not compressed'))
        w.writeframes(struct.pack('%dB' % (trace.stats.npts * 1),
                                  *trace.data))
        w.close()
        i += 1
