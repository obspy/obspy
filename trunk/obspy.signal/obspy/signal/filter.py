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
import numpy as np
from numpy import array, where, fft
import scipy
from scipy.fftpack import hilbert
from scipy.signal import iirfilter, lfilter, remez, convolve, get_window, \
    cheby2, cheb2ord


def bandpass(data, freqmin, freqmax, df, corners=4, zerophase=False):
    """
    Butterworth-Bandpass Filter.

    Filter data from ``freqmin`` to ``freqmax`` using ``corners`` corners.

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
    [b, a] = iirfilter(corners, [low, high], btype='band',
                       ftype='butter', output='ba')
    if zerophase:
        firstpass = lfilter(b, a, data)
        return lfilter(b, a, firstpass[::-1])[::-1]
    else:
        return lfilter(b, a, data)


def bandstop(data, freqmin, freqmax, df, corners=4, zerophase=False):
    """
    Butterworth-Bandstop Filter.

    Filter data removing data between frequencies ``freqmin`` and ``freqmax``
    using ``corners`` corners.

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
    [b, a] = iirfilter(corners, [low, high],
                       btype='bandstop', ftype='butter', output='ba')
    if zerophase:
        firstpass = lfilter(b, a, data)
        return lfilter(b, a, firstpass[::-1])[::-1]
    else:
        return lfilter(b, a, data)


def lowpass(data, freq, df, corners=4, zerophase=False):
    """
    Butterworth-Lowpass Filter.

    Filter data removing data over certain frequency ``freq`` using ``corners``
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
    f = freq / fe
    # raise for some bad scenarios
    if f > 1:
        f = 1.0
        msg = "Selected corner frequency is above Nyquist. " + \
              "Setting Nyquist as high corner."
        warnings.warn(msg)
    [b, a] = iirfilter(corners, f, btype='lowpass', ftype='butter',
                       output='ba')
    if zerophase:
        firstpass = lfilter(b, a, data)
        return lfilter(b, a, firstpass[::-1])[::-1]
    else:
        return lfilter(b, a, data)


def highpass(data, freq, df, corners=4, zerophase=False):
    """
    Butterworth-Highpass Filter.

    Filter data removing data below certain frequency ``freq`` using
    ``corners`` corners.

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
    f = freq / fe
    # raise for some bad scenarios
    if f > 1:
        msg = "Selected corner frequency is above Nyquist."
        raise ValueError(msg)
    [b, a] = iirfilter(corners, f, btype='highpass', ftype='butter',
                       output='ba')
    if zerophase:
        firstpass = lfilter(b, a, data)
        return lfilter(b, a, firstpass[::-1])[::-1]
    else:
        return lfilter(b, a, data)


def envelope(data):
    """
    Envelope of a function.

    Computes the envelope of the given function. The envelope is determined by
    adding the squared amplitudes of the function and it's Hilbert-Transform
    and then taking the square-root. (See [Kanasewich1981]_)
    The envelope at the start/end should not be taken too seriously.

    :param data: Data to make envelope of, type numpy.ndarray.
    :return: Envelope of input data.
    """
    hilb = hilbert(data)
    data = (data ** 2 + hilb ** 2) ** 0.5
    return data


def remezFIR(data, freqmin, freqmax, df):
    """
    The minimax optimal bandpass using Remez algorithm. (experimental)

    .. warning:: This is experimental code. Use with caution!

    :param data: Data to filter, type numpy.ndarray.
    :param freqmin: Low corner frequency.
    :param freqmax: High corner frequency.
    :param df: Sampling rate in Hz.
    :return: Filtered data.

    Finite impulse response (FIR) filter whose transfer function minimizes
    the maximum error between the desired gain and the realized gain in the
    specified bands using the remez exchange algorithm.

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
    filt = remez(50, array([0, flt, freqmin, freqmax, fut, df / 2 - 1]),
                 array([0, 1, 0]), Hz=df)
    return convolve(filt, data)


def lowpassFIR(data, freq, df, winlen=2048):
    """
    FIR-Lowpass Filter. (experimental)

    .. warning:: This is experimental code. Use with caution!

    Filter data by passing data only below a certain frequency.

    :param data: Data to filter, type numpy.ndarray.
    :param freq: Data below this frequency pass.
    :param df: Sampling rate in Hz.
    :param winlen: Window length for filter in samples, must be power of 2;
        Default 2048
    :return: Filtered data.

    .. versionadded:: 0.6.2
    """
    # Source: Travis Oliphant
    # http://mail.scipy.org/pipermail/scipy-user/2004-February/002628.html
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
    w = fft.fftfreq(winlen, 1 / float(df))
    # cutoff is low-pass filter
    myfilter = where((abs(w) < freq), 1., 0.)
    # ideal filter
    h = fft.ifft(myfilter)
    beta = 11.7
    # beta implies Kaiser
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

    # reshape and only use every decimation_factor-th sample
    data = array(data[::decimation_factor])
    return data


def lowpassCheby2(data, freq, df, maxorder=12, ba=False,
                  freq_passband=False):
    """
    Cheby2-Lowpass Filter

    Filter data by passing data only below a certain frequency.
    The main purpose of this cheby2 filter is downsampling.
    #318 shows some plots of this filter design itself.

    This method will iteratively design a filter, whose pass
    band frequency is determined dynamically, such that the
    values above the stop band frequency are lower than -96dB.

    :param data: Data to filter, type numpy.ndarray.
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
    wp = ws              # pass band frequency
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
    b, a = cheby2(order, rs, wn, btype='low', analog=0, output='ba')
    if ba:
        return b, a
    if freq_passband:
        return lfilter(b, a, data), wp * nyquist
    return lfilter(b, a, data)


def polarizationFilter(x, y, z, wlen, details=False):
    """
    Three component polarization filter

    Filter three component data based on polarization analysis using singular
    value decomposition following the algorithm described by [deFranco2001]_.

    .. note::

        If ``x``, ``y`` and ``z`` are not of equal length, the shortest array
        defines the length for processing and the rest of the data is
        disregarded.

    :type x: :class:`~numpy.ndarray`
    :param x: Time series of first component (e.g. Z)
    :type y: :class:`~numpy.ndarray`
    :param y: Time series of second component (e.g. R).
    :type z: :class:`~numpy.ndarray`
    :param z: Time series of third component (e.g. T).
    :type wlen: int
    :param wlen: Length of non-overlapping window for singular value
        decomposition in samples. Choice of window length depends on the signal
        characteristics that should be enhanced. As an example, [deFranco2001]_
        suggest a window of 0.5 seconds for signal with frequency content of
        2-15 Hz and a dominant frequency of 5 Hz. If set to 0, the whole time
        series is treated as one single window. Poor choice of ``wlen`` can
        result in problems in the output.
    :type details: bool
    :param details: If selected, additional information on the polarization
        filtering is returned (array of eigenimages, interpolated
        rectilinearity 1/2 and planarity, discrete rectilinearity 1/2 and
        planarity and indices of midpoints of discrete polarization
        attributes).
    :rtype: list of 3 :class:`~numpy.ndarray`
    :return: The 3 components of original time series after polarization
        filtering.
    """
    # use whole time series if window length 0 is specified
    if wlen == 0:
        wlen = len(x)

    # check lengths of arrays, use shortest length
    n = set([len(a) for a in x, y, z])
    if len(n) > 1:
        msg = "Data arrays are not of same length. Longer pieces are " + \
              "disregarded."
        warnings.warn(msg)
    n = min(n)

    raw_data = np.vstack((x[:n], y[:n], z[:n])).T

    # eigenimages
    # first index: samples
    # second index: x/y/z
    # third index: number of eigenimage
    e = np.empty(shape=(n, 3, 2), dtype="float64")

    # indices of starts of windows
    inds = np.arange(0, n, wlen)
    # indices of ends of windows
    inds2 = inds + wlen
    # if the last window is less than a certain amount of window length,
    # add it to the previous window in processing instead
    if (n - 1) % wlen < 0.6 * wlen:
        inds = inds[:-1]
        inds2 = inds2[:-1]
    inds2[-1] = n - 1
    inds_startend = np.vstack((inds, inds2)).T
    num_windows = len(inds)

    # polarization attributes stored per window first, interpolated later
    r1_disc = np.empty(num_windows, dtype="float64")
    r2_disc = np.empty(num_windows, dtype="float64")
    p_disc = np.empty(num_windows, dtype="float64")

    for i, (ind, ind2) in enumerate(inds_startend):
        u, w, v = np.linalg.svd(raw_data[ind:ind2, :], full_matrices=False)
        # sort from larger to smaller eigenvalues
        descending = w.argsort()[::-1]
        u = u[:, descending]
        w = w[descending]
        v = v[:, descending]

        # store rectilinearities and planarity for window
        r1_disc[i] = 1 - (pow(w[2], 2) / pow(w[0], 2))
        r2_disc[i] = 1 - (pow(w[2], 2) / pow(w[1], 2))
        p_disc[i] = 1 - (2 * pow(w[2], 2) / (pow(w[0], 2) + pow(w[1], 2)))

        # XXX need to transpose v?!?
        ## v = v.T
        e[ind:ind2, 0, 0] = u[:, 0] * v[0, 0] * w[0]
        e[ind:ind2, 1, 0] = u[:, 0] * v[1, 0] * w[0]
        e[ind:ind2, 2, 0] = u[:, 0] * v[2, 0] * w[0]
        e[ind:ind2, 0, 1] = u[:, 1] * v[0, 1] * w[1]
        e[ind:ind2, 1, 1] = u[:, 1] * v[1, 1] * w[1]
        e[ind:ind2, 2, 1] = u[:, 1] * v[2, 1] * w[1]
        # the third eigenimage is not used
        #e[ind:ind2, 0, 2] = u[:, 2] * v[0, 2] * w[2]
        #e[ind:ind2, 1, 2] = u[:, 2] * v[1, 2] * w[2]
        #e[ind:ind2, 2, 2] = u[:, 2] * v[2, 2] * w[2]

    # interpolate polarization attributes from midpoints to whole time series
    midpoints = inds + (wlen / 2)
    midpoints[-1] = inds[-1] + (((n - 1) % wlen) / 2)
    all_inds = np.arange(n)
    # cubic spline interpolation
    if num_windows > 3:
        spline = scipy.interpolate.InterpolatedUnivariateSpline(midpoints,
                                                                r1_disc)
        r1 = spline(all_inds)
        spline = scipy.interpolate.InterpolatedUnivariateSpline(midpoints,
                                                                r2_disc)
        r2 = spline(all_inds)
        spline = scipy.interpolate.InterpolatedUnivariateSpline(midpoints,
                                                                p_disc)
        p = spline(all_inds)
    # linear interpolation
    elif num_windows > 1:
        interp = scipy.interpolate.interp1d(midpoints, r1_disc,
                                            bounds_error=False)
        r1 = interp(all_inds)
        r1[:midpoints[0]] = r1[midpoints[0]]
        r1[midpoints[-1]:] = r1[midpoints[-1]]
        interp = scipy.interpolate.interp1d(midpoints, r2_disc,
                                            bounds_error=False)
        r2 = interp(all_inds)
        r2[:midpoints[0]] = r2[midpoints[0]]
        r2[midpoints[-1]:] = r2[midpoints[-1]]
        interp = scipy.interpolate.interp1d(midpoints, p_disc,
                                            bounds_error=False)
        p = interp(all_inds)
        p[:midpoints[0]] = p[midpoints[0]]
        p[midpoints[-1]:] = p[midpoints[-1]]
    # constant value
    else:
        r1 = np.empty(n, dtype="float64")
        r2 = np.empty(n, dtype="float64")
        p = np.empty(n, dtype="float64")
        r1.fill(r1_disc[0])
        r2.fill(r2_disc[0])
        p.fill(p_disc[0])

    # interpolation can cause values outside of theoretical bounds, clip those
    r1 = np.clip(r1, 0.0, 1.0)
    r2 = np.clip(r2, 0.0, 1.0)
    p = np.clip(p, 0.0, 1.0)

    # calculate filtered time series as weighted sum of eigenimages
    x_filt = (e[:, 0, 0] * r1 + e[:, 0, 1] * r2) * p
    y_filt = (e[:, 1, 0] * r1 + e[:, 1, 1] * r2) * p
    z_filt = (e[:, 2, 0] * r1 + e[:, 2, 1] * r2) * p
    if details:
        return x_filt, y_filt, z_filt, e, r1, r2, p, r1_disc, r2_disc, p_disc, midpoints
    else:
        return x_filt, y_filt, z_filt


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
