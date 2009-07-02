#!/usr/bin/env python
#-------------------------------------------------------------------
# Filename: filter.py
#  Purpose: Various Seismogram Filtering Functions
#   Author: Tobias Megies, Moritz Beyreuther, Yannik Behr
#    Email: tobias.megies@geophysik.uni-muenchen.de
#
# Copyright (C) 2009 Tobias Megies, Moritz Beyreuther, Yannik Behr
#---------------------------------------------------------------------
"""
Various Seismogram Filtering Functions

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

from numpy import array, where, fft, sin, cos, pi
from scipy.signal import iirfilter,lfilter,remez,convolve,get_window
from scipy.fftpack import hilbert

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

def remezFIR(data,freqmin,freqmax,samp_rate=200):
  """
  The minimax optimal bandpass using Remez algorithm. Zerophase bandpass?

  Finite impulse response (FIR) filter whose transfer function minimizes
  the maximum error between the desired gain and the realized gain in the
  specified bands using the remez exchange algorithm
  """
  # Remez filter description
  # ========================
  #
  # So, let's go over the inputs that you'll have to worry about.
  # First is numtaps. This parameter will basically determine how good your
  # filter is and how much processor power it takes up. If you go for some
  # obscene number of taps (in the thousands) there's other things to worry
  # about, but with sane numbers (probably below 30-50 in your case) that is
  # pretty much what it affects (more taps is better, but more expensive
  #         processing wise). There are other ways to do filters as well
  # which require less CPU power if you really need it, but I doubt that you
  # will. Filtering signals basically breaks down to convolution, and apple
  # has DSP libraries to do lightning fast convolution I'm sure, so don't
  # worry about this too much. Numtaps is basically equivalent to the number
  # of terms in the convolution, so a power of 2 is a good idea, 32 is
  # probably fine.
  #
  # bands has literally your list of bands, so you'll break it up into your
  # low band, your pass band, and your high band. Something like [0, 99, 100,
  # 999, 1000, 22049] should work, if you want to pass frequencies between
  # 100-999 Hz (assuming you are sampling at 44.1 kHz).
  #
  # desired will just be [0, 1, 0] as you want to drop the high and low
  # bands, and keep the middle one without modifying the amplitude.
  #
  # Also, specify Hz = 44100 (or whatever).
  #
  # That should be all you need; run the function and it will spit out a list
  # of coefficients [c0, ... c(N-1)] where N is your tap count. When you run
  # this filter, your output signal y[t] will be computed from the input x[t]
  # like this (t-N means N samples before the current one):
  #
  # y[t] = c0*x[t] + c1*x[t-1] + ... + c(N-1)*x[t-(N-1)]
  #
  # After playing around with remez for a bit, it looks like numtaps should be
  # above 100 for a solid filter. See what works out for you. Eventually, take
  # those coefficients and then move them over and do the convolution in C or
  # whatever. Also, note the gaps between the bands in the call to remez. You
  # have to leave some space for the transition in frequency response to occur,
  # otherwise the call to remez will complain.
  #
  # SRC: # http://episteme.arstechnica.com/eve/forums/a/tpc/f/6330927813/m/175006289731
  # See also:
  # http://aspn.activestate.com/ASPN/Mail/Message/scipy-dev/1592174
  # http://aspn.activestate.com/ASPN/Mail/Message/scipy-dev/1592172

  #take 10% of freqmin and freqmax as """corners"""
  flt = freqmin - 0.1*freqmin
  fut = freqmax + 0.1*freqmax
  #bandpass between freqmin and freqmax
  filt = remez(50, array([0, flt, freqmin, freqmax, fut,  samp_rate/2-1]), 
               array([0, 1, 0]),Hz=samp_rate)
  return convolve(filt,data)

def lowpassFIR(data,freq,samp_rate=200,winlen=2048):
  """FIR-Lowpass Filter

  Filter data by passing data only below a certain frequency.
  
  @param data: Data to filter, type numpy.ndarray.
  @param freq: Data below this frequency pass.
  @param samprate: Sampling rate in Hz; Default 200.
  @param winlen: Window length for filter in samples, must be power of 2; Default 2048
  @return: Filtered data.
  """
  # There is not currently an FIR-filter design program in SciPy.  One 
  # should be constructed as it is not hard to implement (of course making 
  # it generic with all the options you might want would take some time).
  # 
  # What kind of window are you currently using?
  # 
  # For your purposes this is what I would do:
  # SRC: Travis Oliphant
  # http://aspn.activestate.com/ASPN/Mail/Message/scipy-user/2009409]
  #
  #winlen = 2**11 #2**10 = 1024; 2**11 = 2048; 2**12 = 4096
  w = fft.fftfreq(winlen,1/float(samp_rate)) #give frequency bins in Hz and sample spacing
  myfilter = where((abs(w)< freq),1.,0.)  #cutoff is low-pass filter
  h = fft.ifft(myfilter) #ideal filter
  beta = 11.7
  myh = fft.fftshift(h) * get_window(beta,winlen)  #beta implies Kaiser
  return convolve(abs(myh),data)[winlen/2:-winlen/2]

