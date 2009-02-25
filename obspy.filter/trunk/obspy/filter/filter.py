#!/usr/bin/env python
#-------------------------------------------------------------------
# Filename: filter.py
#  Purpose: Various Seismogram Filtering Functions
#   Author: Tobias Megies, Moritz Beyreuther, Yannik Behr
#    Email: tobias.megies@geophysik.uni-muenchen.de
#
# Copyright (C) 2009 Tobias Megia, Moritz Beyreuther, Yannik Behr
# 
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
#---------------------------------------------------------------------

from scipy.signal import iirfilter,lfilter
from scipy.fftpack import hilbert

# NOTE: Data need to be a instance of the numpy array class

def bandpass(data,freqmin,freqmax,df=200,corners=4):
  """Butterworth-Bandpass Filter
  
  Filter data from freqmin to freqmax using
  corners corners.
  """
  fe=.5*df
  [b,a]=iirfilter(corners, [freqmin/fe, freqmax/fe], btype='band',ftype='butter',output='ba')
  return lfilter(b, a, data)

def bandpassZPHSH(data,freqmin,freqmax,df=200,corners=2):
  """Zero-Phase-Shift Butterworth-Bandpass Filter
  
  Filter data from freqmin to freqmax using corners corners and doing 2 runs
  over the data, one from left to right and one from right to left.
  Note that the default corners is 2, because of the two runs involved.
  """
  x=bandpass(data, freqmin, freqmax, df, corners)
  x=bandpass(x[::-1], freqmin, freqmax, df, corners)
  return x[::-1]

def bandstop(data,freqmin,freqmax,df=200,corners=4):
  """Butterworth-Bandstop Filter
  
  Filter data removing data between frequencies freqmin and freqmax using
  corners corners.
  """
  fe=.5*df
  [b,a]=iirfilter(corners, [freqmin/fe, freqmax/fe], btype='bandstop',ftype='butter',output='ba')
  return lfilter(b, a, data)

def bandstopZPHSH(data,freqmin,freqmax,df=200,corners=2):
  """Zero-Phase-Shift Butterworth-Bandstop Filter
  
  Filter data removing data between frequencies freqmin and freqmax using
  corners corners and doing 2 runs over the data, one from left to right and
  one from right to left.
  Note that the default corners is 2, because of the two runs involved.
  """
  x=bandstop(data, freqmin, freqmax, df, corners)
  x=bandstop(x[::-1], freqmin, freqmax, df, corners)
  return x[::-1]

def lowpass(data,freq,df=200,corners=4):
  """Butterworth-Lowpass Filter
  
  Filter data removing data over certain frequency freq using corners corners.
  """
  fe=.5*df
  [b,a]=iirfilter(corners, freq/fe, btype='lowpass',ftype='butter',output='ba')
  return lfilter(b, a, data)

def lowpassZPHSH(data,freq,df=200,corners=2):
  """Zero-Phase-Shift Butterworth-Lowpass Filter
  
  Filter data removing data over certain frequency freq using corners corners
  and doing 2 runs over the data, one from left to right and one from right
  to left.
  Note that the default corners is 2, because of the two runs involved.
  """
  x=lowpass(data, freq, df, corners)
  x=lowpass(x[::-1], freq, df, corners)
  return x[::-1]

def highpass(data,freq,df=200,corners=4):
  """Butterworth-Highpass Filter:
  
  Filter data removing data below certain frequency freq using corners corners.
  """
  fe=.5*df
  [b,a]=iirfilter(corners, freq/fe, btype='highpass',ftype='butter',output='ba')
  return lfilter(b, a, data)

def highpassZPHSH(data,freq,df=200,corners=2):
  """Zero-Phase-Shift Butterworth-Highpass Filter:
  
  Filter data removing data below certain frequency freq using corners corners
  and doing 2 runs over the data, one from left to right and one from right
  to left.
  Note that the default corners is 2, because of the two runs involved.
  """
  x=highpass(data, freq, df, corners)
  x=highpass(x[::-1], freq, df, corners)
  return x[::-1]

def envelope(data):
  """Envelope of a function:

  Computes the envelope of the given function. The envelope is determined by
  adding the squared amplitudes of the function and it's Hilbert-Transform and
  then taking the squareroot.
  (See Kanasewich: Time Sequence Analysis in Geophysics)
  The envelope at the start/end should not be taken too seriously.
  """
  hilb=hilbert(data)
  data=pow(pow(data,2)+pow(hilb,2),0.5)
  return data
