#!/usr/bin/python
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

from obspy.numpy import array
from obspy.util import Stats
import os, wave, struct

class WAVTrace(object):
    __format__ = 'WAV'
    
    def __init__(self, filename=None, **kwargs):
        if filename:
            self.read(filename, **kwargs)
    
    def read(self, filename, **kwargs):
        """
        Read audio WAV file.
        
        Currently supports unsigned char and short integer data values. This
        should cover most WAV files.

        @params filename: Name of WAV file to read.
        @returns: self.data list of data values, self.stats dictionary of
            WAV parameters.
        """

        if not os.path.exists(filename):
            msg = "File not found '%s'" % (filename)
            raise IOError(msg)
        #
        # read WAV file
        # ------------------------------------------
        w = wave.open(filename, 'rb')
        (nchannel, width, rate, length, comptype, compname) = w.getparams()
        if width == 1:
            format = 'B'
        elif width == 2:
            format = 'h'
        else:
            raise TypeError("Unsupported Format Type, string length %d" % length)
        data = struct.unpack("%d%s" % (length*nchannel,format),w.readframes(length))
        w.close()
        # ------------------------------------------
        # reset header information
        self.stats = Stats()
        # wav format nchannel
        self.stats.nchannel = nchannel
        # wav format width
        self.stats.width = width
        # wav format comptype
        self.stats.comptype = comptype
        # sampling rate in Hz (float)
        self.stats.sampling_rate = rate
        # number of samples/data points (int)
        self.stats.npts = length
        # now the general trace attributes
        # station name
        self.stats.station = ""
        # starttime
        starttime = None
        # network ID
        self.stats.network = ""
        # location ID
        self.stats.location = ""
        self.data = array(data)
    
    def write(self, filename, framerate=7000, **kwargs):
        """
        Write audio WAV file. The seismogram is queezed to audible frequencies.

        The resulting wav sound file is as a result really short. The data
        are written uncompressed as unsigned char.

        @requires: The attributes self.npts = number of samples; self.data =
            list of data samples.
        @param filename: Name of WAV file to write.
        @param framerate: Samplerate of wav file to use. This this will
            squeeze the seismogram, DEFAULT=7000. 
        """
        # write WAV file
        w = wave.open(filename, 'wb')
        self.stats.npts = len(self.data)
        # Does not work with some tests because self.stats.npts = -1
#        try: self.stats.npts
#        except AttributeError:
#            self.stats.npts = len(self.data)
        # (nchannels, sampwidth, framerate, nframes, comptype, compname)
        w.setparams((1,1,framerate,self.stats.npts,'NONE', 'not compressed'))
        w.writeframes(struct.pack('%dB' % (self.stats.npts*1),*self.data))
        w.close()
