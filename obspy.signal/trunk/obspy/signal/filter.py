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

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
import warnings

from numpy import array, where, fft
from scipy.fftpack import hilbert
from scipy.signal import iirfilter, lfilter, remez, convolve, get_window


def bandpass(data, freqmin, freqmax, df, corners=4, zerophase=False):
    """
    Butterworth-Bandpass Filter.

    Filter data from freqmin to freqmax using
    corners corners.

    :param data: Data to filter, type numpy.ndarray.
    :param freqmin: Pass band low corner frequency.
    :param freqmax: Pass band high corner frequency.
    :param df: Sampling rate in Hz.
    :param corners: Filter corners. Note: This is twice the value of PITSA's 
        filter sections
    :param zerophase: If True, apply filter once forwards and once backwards.
        This results in twice the number of corners but zero phase shift in
        the resulting filtered trace.
    :return: Filtered data.
    """
    fe = 0.5 * df
    [b, a] = iirfilter(corners, [freqmin / fe, freqmax / fe], btype='band',
                       ftype='butter', output='ba')
    if zerophase:
        firstpass = lfilter(b, a, data)
        return lfilter(b, a, firstpass[::-1])[::-1]
    else:
        return lfilter(b, a, data)


def bandpassZPHSH(data, freqmin, freqmax, df, corners=2):
    """
    DEPRECATED. Use :func:`~obspy.signal.filter.bandpass` instead.
    """
    warnings.warn("Use bandpass(..., zerophase=True) instead.", DeprecationWarning)
    return bandpass(data, freqmin, freqmax, df, corners, zerophase=True)


def bandstop(data, freqmin, freqmax, df, corners=4, zerophase=False):
    """
    Butterworth-Bandstop Filter.

    Filter data removing data between frequencies freqmin and freqmax using
    corners corners.

    :param data: Data to filter, type numpy.ndarray.
    :param freqmin: Stop band low corner frequency.
    :param freqmax: Stop band high corner frequency.
    :param df: Sampling rate in Hz.
    :param corners: Filter corners. Note: This is twice the value of PITSA's 
        filter sections
    :param zerophase: If True, apply filter once forwards and once backwards.
        This results in twice the number of corners but zero phase shift in
        the resulting filtered trace.
    :return: Filtered data.
    """
    fe = 0.5 * df
    [b, a] = iirfilter(corners, [freqmin / fe, freqmax / fe],
                       btype='bandstop', ftype='butter', output='ba')
    if zerophase:
        firstpass = lfilter(b, a, data)
        return lfilter(b, a, firstpass[::-1])[::-1]
    else:
        return lfilter(b, a, data)


def bandstopZPHSH(data, freqmin, freqmax, df, corners=2):
    """
    DEPRECATED. Use :func:`~obspy.signal.filter.bandstop` instead.
    """
    warnings.warn("Use bandstop(..., zerophase=True) instead.", DeprecationWarning)
    return bandstop(data, freqmin, freqmax, df, corners, zerophase=True)


def lowpass(data, freq, df, corners=4, zerophase=False):
    """
    Butterworth-Lowpass Filter.

    Filter data removing data over certain frequency freq using corners 
    corners.

    :param data: Data to filter, type numpy.ndarray.
    :param freq: Filter corner frequency.
    :param df: Sampling rate in Hz.
    :param corners: Filter corners. Note: This is twice the value of PITSA's 
        filter sections
    :param zerophase: If True, apply filter once forwards and once backwards.
        This results in twice the number of corners but zero phase shift in
        the resulting filtered trace.
    :return: Filtered data.
    """
    fe = 0.5 * df
    [b, a] = iirfilter(corners, freq / fe, btype='lowpass', ftype='butter',
                       output='ba')
    if zerophase:
        firstpass = lfilter(b, a, data)
        return lfilter(b, a, firstpass[::-1])[::-1]
    else:
        return lfilter(b, a, data)


def lowpassZPHSH(data, freq, df, corners=2):
    """
    DEPRECATED. Use :func:`~obspy.signal.filter.lowpass` instead.
    """
    warnings.warn("Use lowpass(..., zerophase=True) instead.", DeprecationWarning)
    return lowpass(data, freq, df, corners, zerophase=True)


def highpass(data, freq, df, corners=4, zerophase=False):
    """
    Butterworth-Highpass Filter.

    Filter data removing data below certain frequency freq using corners.

    :param data: Data to filter, type numpy.ndarray.
    :param freq: Filter corner frequency.
    :param df: Sampling rate in Hz.
    :param corners: Filter corners. Note: This is twice the value of PITSA's 
        filter sections
    :param zerophase: If True, apply filter once forwards and once backwards.
        This results in twice the number of corners but zero phase shift in
        the resulting filtered trace.
    :return: Filtered data.
    """
    fe = 0.5 * df
    [b, a] = iirfilter(corners, freq / fe, btype='highpass', ftype='butter',
                       output='ba')
    if zerophase:
        firstpass = lfilter(b, a, data)
        return lfilter(b, a, firstpass[::-1])[::-1]
    else:
        return lfilter(b, a, data)


def highpassZPHSH(data, freq, df, corners=2):
    """
    DEPRECATED. Use :func:`~obspy.signal.filter.highpass` instead.
    """
    warnings.warn("Use highpass(..., zerophase=True) instead.", DeprecationWarning)
    return highpass(data, freq, df, corners, zerophase=True)


def envelope(data):
    """
    Envelope of a function.

    Computes the envelope of the given function. The envelope is determined by
    adding the squared amplitudes of the function and it's Hilbert-Transform 
    and then taking the squareroot. 
    (See Kanasewich: Time Sequence Analysis in Geophysics)
    The envelope at the start/end should not be taken too seriously.

    :param data: Data to make envelope of, type numpy.ndarray.
    :return: Envelope of input data.
    """
    hilb = hilbert(data)
    data = pow(pow(data, 2) + pow(hilb, 2), 0.5)
    return data


def remezFIR(data, freqmin, freqmax, samp_rate):
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
    # After playing around with remez for a bit, it looks like numtaps should 
    # be above 100 for a solid filter. See what works out for you. Eventually, 
    # take those coefficients and then move them over and do the convolution 
    # in C or whatever. Also, note the gaps between the bands in the call to 
    # remez. You have to leave some space for the transition in frequency 
    # response to occur, otherwise the call to remez will complain.
    #
    # SRC: # http://episteme.arstechnica.com/eve/forums/a/tpc/f/6330927813/m/175006289731
    # See also:
    # http://aspn.activestate.com/ASPN/Mail/Message/scipy-dev/1592174
    # http://aspn.activestate.com/ASPN/Mail/Message/scipy-dev/1592172

    #take 10% of freqmin and freqmax as """corners"""
    flt = freqmin - 0.1 * freqmin
    fut = freqmax + 0.1 * freqmax
    #bandpass between freqmin and freqmax
    filt = remez(50, array([0, flt, freqmin, freqmax, fut, samp_rate / 2 - 1]),
                 array([0, 1, 0]), Hz=samp_rate)
    return convolve(filt, data)


def lowpassFIR(data, freq, samp_rate, winlen=2048):
    """
    FIR-Lowpass Filter

    Filter data by passing data only below a certain frequency.
  
    :param data: Data to filter, type numpy.ndarray.
    :param freq: Data below this frequency pass.
    :param samprate: Sampling rate in Hz.
    :param winlen: Window length for filter in samples, must be power of 2; 
        Default 2048
    :return: Filtered data.
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
    #give frequency bins in Hz and sample spacing
    w = fft.fftfreq(winlen, 1 / float(samp_rate))
    #cutoff is low-pass filter
    myfilter = where((abs(w) < freq), 1., 0.)
    #ideal filter
    h = fft.ifft(myfilter)
    beta = 11.7
    #beta implies Kaiser
    myh = fft.fftshift(h) * get_window(beta, winlen)
    return convolve(abs(myh), data)[winlen / 2:-winlen / 2]

def integerDecimation(data, decimation_factor):
    """
    Downsampling by applying a simple integer decimation.

    Make sure that no signal is present in frequency bands above the new
    Nyquist frequency (samp_rate/2/decimation_factor), e.g. by applying a
    lowpass filter beforehand!
    New sampling rate is old sampling rate divided by decimation_factor.

    :param data: Data to filter.
    :param decimation_factor: Integer decimation factor
    :return: Downsampled data (array length: old length / decimation_factor)
    """
    if not isinstance(decimation_factor, int):
        msg = "Decimation_factor must be an integer!"
        raise TypeError(msg)
    
    data = array(data)
    
    # for reshaping the length must be a multiple of decimation factor
    while len(data) % decimation_factor:
        data = data[:-1]

    length = len(data)
    df = decimation_factor
    
    # reshape and only use every decimation_factor-th sample
    data = data.reshape((length/df, df)).swapaxes(0, 1)[0]
    return data
