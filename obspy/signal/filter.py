# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: filter.py
#  Purpose: Various Seismogram Filtering Functions
#   Author: Tobias Megies, Moritz Beyreuther, Yannik Behr
#    Email: tobias.megies@geophysik.uni-muenchen.de
#
# Copyright (C) 2009 Tobias Megies, Moritz Beyreuther, Yannik Behr
# --------------------------------------------------------------------
"""
Various Seismogram Filtering Functions

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import warnings

import numpy as np
from scipy.fftpack import hilbert
from scipy.signal import (cheb2ord, cheby2, convolve, get_window, iirfilter,
                          remez)

try:
    from scipy.signal import sosfilt
    from scipy.signal import zpk2sos
except ImportError:
    from ._sosfilt import _sosfilt as sosfilt
    from ._sosfilt import _zpk2sos as zpk2sos


def bandpass(data, freqmin, freqmax, df, corners=4, zerophase=False):
    """
    Butterworth-Bandpass Filter.

    Filter data from ``freqmin`` to ``freqmax`` using ``corners``
    corners.
    The filter uses :func:`scipy.signal.iirfilter` (for design)
    and :func:`scipy.signal.sosfilt` (for applying the filter).

    :type data: numpy.ndarray
    :param data: Data to filter.
    :param freqmin: Pass band low corner frequency.
    :param freqmax: Pass band high corner frequency.
    :param df: Sampling rate in Hz.
    :param corners: Filter corners / order.
    :param zerophase: If True, apply filter once forwards and once backwards.
        This results in twice the filter order but zero phase shift in
        the resulting filtered trace.
    :return: Filtered data.
    """
    fe = 0.5 * df
    low = freqmin / fe
    high = freqmax / fe
    # raise for some bad scenarios
    if high - 1.0 > -1e-6:
        msg = ("Selected high corner frequency ({}) of bandpass is at or "
               "above Nyquist ({}). Applying a high-pass instead.").format(
            freqmax, fe)
        warnings.warn(msg)
        return highpass(data, freq=freqmin, df=df, corners=corners,
                        zerophase=zerophase)
    if low > 1:
        msg = "Selected low corner frequency is above Nyquist."
        raise ValueError(msg)
    z, p, k = iirfilter(corners, [low, high], btype='band',
                        ftype='butter', output='zpk')
    sos = zpk2sos(z, p, k)
    if zerophase:
        firstpass = sosfilt(sos, data)
        return sosfilt(sos, firstpass[::-1])[::-1]
    else:
        return sosfilt(sos, data)


def bandstop(data, freqmin, freqmax, df, corners=4, zerophase=False):
    """
    Butterworth-Bandstop Filter.

    Filter data removing data between frequencies ``freqmin`` and ``freqmax``
    using ``corners`` corners.
    The filter uses :func:`scipy.signal.iirfilter` (for design)
    and :func:`scipy.signal.sosfilt` (for applying the filter).

    :type data: numpy.ndarray
    :param data: Data to filter.
    :param freqmin: Stop band low corner frequency.
    :param freqmax: Stop band high corner frequency.
    :param df: Sampling rate in Hz.
    :param corners: Filter corners / order.
    :param zerophase: If True, apply filter once forwards and once backwards.
        This results in twice the number of corners but zero phase shift in
        the resulting filtered trace.
    :return: Filtered data.
    """
    fe = 0.5 * df
    low = freqmin / fe
    high = freqmax / fe
    # raise for some bad scenarios
    if high > 1:
        high = 1.0
        msg = "Selected high corner frequency is above Nyquist. " + \
              "Setting Nyquist as high corner."
        warnings.warn(msg)
    if low > 1:
        msg = "Selected low corner frequency is above Nyquist."
        raise ValueError(msg)
    z, p, k = iirfilter(corners, [low, high],
                        btype='bandstop', ftype='butter', output='zpk')
    sos = zpk2sos(z, p, k)
    if zerophase:
        firstpass = sosfilt(sos, data)
        return sosfilt(sos, firstpass[::-1])[::-1]
    else:
        return sosfilt(sos, data)


def lowpass(data, freq, df, corners=4, zerophase=False):
    """
    Butterworth-Lowpass Filter.

    Filter data removing data over certain frequency ``freq`` using ``corners``
    corners.
    The filter uses :func:`scipy.signal.iirfilter` (for design)
    and :func:`scipy.signal.sosfilt` (for applying the filter).

    :type data: numpy.ndarray
    :param data: Data to filter.
    :param freq: Filter corner frequency.
    :param df: Sampling rate in Hz.
    :param corners: Filter corners / order.
    :param zerophase: If True, apply filter once forwards and once backwards.
        This results in twice the number of corners but zero phase shift in
        the resulting filtered trace.
    :return: Filtered data.
    """
    fe = 0.5 * df
    f = freq / fe
    # raise for some bad scenarios
    if f > 1:
        f = 1.0
        msg = "Selected corner frequency is above Nyquist. " + \
              "Setting Nyquist as high corner."
        warnings.warn(msg)
    z, p, k = iirfilter(corners, f, btype='lowpass', ftype='butter',
                        output='zpk')
    sos = zpk2sos(z, p, k)
    if zerophase:
        firstpass = sosfilt(sos, data)
        return sosfilt(sos, firstpass[::-1])[::-1]
    else:
        return sosfilt(sos, data)


def highpass(data, freq, df, corners=4, zerophase=False):
    """
    Butterworth-Highpass Filter.

    Filter data removing data below certain frequency ``freq`` using
    ``corners`` corners.
    The filter uses :func:`scipy.signal.iirfilter` (for design)
    and :func:`scipy.signal.sosfilt` (for applying the filter).

    :type data: numpy.ndarray
    :param data: Data to filter.
    :param freq: Filter corner frequency.
    :param df: Sampling rate in Hz.
    :param corners: Filter corners / order.
    :param zerophase: If True, apply filter once forwards and once backwards.
        This results in twice the number of corners but zero phase shift in
        the resulting filtered trace.
    :return: Filtered data.
    """
    fe = 0.5 * df
    f = freq / fe
    # raise for some bad scenarios
    if f > 1:
        msg = "Selected corner frequency is above Nyquist."
        raise ValueError(msg)
    z, p, k = iirfilter(corners, f, btype='highpass', ftype='butter',
                        output='zpk')
    sos = zpk2sos(z, p, k)
    if zerophase:
        firstpass = sosfilt(sos, data)
        return sosfilt(sos, firstpass[::-1])[::-1]
    else:
        return sosfilt(sos, data)


def envelope(data):
    """
    Envelope of a function.

    Computes the envelope of the given function. The envelope is determined by
    adding the squared amplitudes of the function and it's Hilbert-Transform
    and then taking the square-root. (See [Kanasewich1981]_)
    The envelope at the start/end should not be taken too seriously.

    :type data: numpy.ndarray
    :param data: Data to make envelope of.
    :return: Envelope of input data.
    """
    hilb = hilbert(data)
    data = (data ** 2 + hilb ** 2) ** 0.5
    return data


def remez_fir(data, freqmin, freqmax, df):
    """
    The minimax optimal bandpass using Remez algorithm. (experimental)

    .. warning:: This is experimental code. Use with caution!

    :type data: numpy.ndarray
    :param data: Data to filter.
    :param freqmin: Low corner frequency.
    :param freqmax: High corner frequency.
    :param df: Sampling rate in Hz.
    :return: Filtered data.

    Finite impulse response (FIR) filter whose transfer function minimizes
    the maximum error between the desired gain and the realized gain in the
    specified bands using the Remez exchange algorithm.

    .. versionadded:: 0.6.2
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
    # Source:
    # http://episteme.arstechnica.com/
    #         eve/forums/a/tpc/f/6330927813/m/175006289731
    #
    # take 10% of freqmin and freqmax as """corners"""
    flt = freqmin - 0.1 * freqmin
    fut = freqmax + 0.1 * freqmax
    # bandpass between freqmin and freqmax
    filt = remez(50, np.array([0, flt, freqmin, freqmax, fut, df / 2 - 1]),
                 np.array([0, 1, 0]), Hz=df)
    return convolve(filt, data)


def lowpass_fir(data, freq, df, winlen=2048):
    """
    FIR-Lowpass Filter. (experimental)

    .. warning:: This is experimental code. Use with caution!

    Filter data by passing data only below a certain frequency.

    :type data: numpy.ndarray
    :param data: Data to filter.
    :param freq: Data below this frequency pass.
    :param df: Sampling rate in Hz.
    :param winlen: Window length for filter in samples, must be power of 2;
        Default 2048
    :return: Filtered data.

    .. versionadded:: 0.6.2
    """
    # Source: Travis Oliphant
    # https://mail.scipy.org/pipermail/scipy-user/2004-February/002628.html
    #
    # There is not currently an FIR-filter design program in SciPy. One
    # should be constructed as it is not hard to implement (of course making
    # it generic with all the options you might want would take some time).
    #
    # What kind of window are you currently using?
    #
    # For your purposes this is what I would do:
    #
    # winlen = 2**11 #2**10 = 1024; 2**11 = 2048; 2**12 = 4096
    # give frequency bins in Hz and sample spacing
    w = np.fft.fftfreq(winlen, 1 / float(df))
    # cutoff is low-pass filter
    myfilter = np.where((abs(w) < freq), 1., 0.)
    # ideal filter
    h = np.fft.ifft(myfilter)
    beta = 11.7
    # beta implies Kaiser
    myh = np.fft.fftshift(h) * get_window(beta, winlen)
    return convolve(abs(myh), data)[winlen / 2:-winlen / 2]


def integer_decimation(data, decimation_factor):
    """
    Downsampling by applying a simple integer decimation.

    Make sure that no signal is present in frequency bands above the new
    Nyquist frequency (samp_rate/2/decimation_factor), e.g. by applying a
    lowpass filter beforehand!
    New sampling rate is old sampling rate divided by decimation_factor.

    :type data: numpy.ndarray
    :param data: Data to filter.
    :param decimation_factor: Integer decimation factor
    :return: Downsampled data (array length: old length / decimation_factor)
    """
    if not isinstance(decimation_factor, int):
        msg = "Decimation_factor must be an integer!"
        raise TypeError(msg)

    # reshape and only use every decimation_factor-th sample
    data = np.array(data[::decimation_factor])
    return data


def lowpass_cheby_2(data, freq, df, maxorder=12, ba=False,
                    freq_passband=False):
    """
    Cheby2-Lowpass Filter

    Filter data by passing data only below a certain frequency.
    The main purpose of this cheby2 filter is downsampling.
    #318 shows some plots of this filter design itself.

    This method will iteratively design a filter, whose pass
    band frequency is determined dynamically, such that the
    values above the stop band frequency are lower than -96dB.

    :type data: numpy.ndarray
    :param data: Data to filter.
    :param freq: The frequency above which signals are attenuated
        with 95 dB
    :param df: Sampling rate in Hz.
    :param maxorder: Maximal order of the designed cheby2 filter
    :param ba: If True return only the filter coefficients (b, a) instead
        of filtering
    :param freq_passband: If True return additionally to the filtered data,
        the iteratively determined pass band frequency
    :return: Filtered data.
    """
    nyquist = df * 0.5
    # rp - maximum ripple of passband, rs - attenuation of stopband
    rp, rs, order = 1, 96, 1e99
    ws = freq / nyquist  # stop band frequency
    wp = ws  # pass band frequency
    # raise for some bad scenarios
    if ws > 1:
        ws = 1.0
        msg = "Selected corner frequency is above Nyquist. " + \
              "Setting Nyquist as high corner."
        warnings.warn(msg)
    while True:
        if order <= maxorder:
            break
        wp = wp * 0.99
        order, wn = cheb2ord(wp, ws, rp, rs, analog=0)
    if ba:
        return cheby2(order, rs, wn, btype='low', analog=0, output='ba')
    z, p, k = cheby2(order, rs, wn, btype='low', analog=0, output='zpk')
    sos = zpk2sos(z, p, k)
    if freq_passband:
        return sosfilt(sos, data), wp * nyquist
    return sosfilt(sos, data)
