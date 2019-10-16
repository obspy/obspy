#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Various additional utilities for obspy.signal.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import native

import ctypes as C  # NOQA
import math

import numpy as np
from scipy import fftpack, fix, signal

from obspy.core.util.misc import factorize_int
from obspy.signal.headers import clibsignal


def util_geo_km(orig_lon, orig_lat, lon, lat):
    """
    Transform lon, lat to km with reference to orig_lon and orig_lat on the
    elliptic Earth.

    >>> util_geo_km(12.0, 48.0, 12.0, 48.0)
    (0.0, 0.0)
    >>> x, y = util_geo_km(12.0, 48.0, 13.0, 49.0)
    >>> print(round(x,7))
    73.9041417
    >>> print(round(y,7))
    111.1908262

    :param orig_lon: Longitude of reference origin
    :param orig_lat: Latitude of reference origin
    :param lat: Latitude to calculate relative coordinate in km
    :param lon: Longitude to calculate relative coordinate in km
    :return: x, y coordinate in km (in reference to origin)
    """
    # 2009-10-11 Moritz
    x = C.c_double(lon)
    y = C.c_double(lat)

    clibsignal.utl_geo_km(orig_lon, orig_lat, 0.0, C.byref(x), C.byref(y))
    return x.value, y.value


def util_lon_lat(orig_lon, orig_lat, x, y):
    """
    Transform x, y [km] to decimal degree in reference to orig_lon and orig_lat

    >>> util_lon_lat(12.0, 48.0, 0.0, 0.0)
    (12.0, 48.0)
    >>> lon, lat = util_lon_lat(12.0, 48.0, 73.9041, 111.1908)
    >>> print("%.4f, %.4f" % (lon, lat))
    13.0000, 49.0000

    :param orig_lon: Longitude of reference origin
    :param orig_lat: Latitude of reference origin
    :param x: value [km] to calculate relative coordinate in degree
    :param y: value [km] to calculate relative coordinate in degree
    :return: lon, lat coordinate in degree (absolute)
    """
    # 2009-10-11 Moritz

    clibsignal.utl_lonlat.argtypes = [C.c_double, C.c_double, C.c_double,
                                      C.c_double, C.POINTER(C.c_double),
                                      C.POINTER(C.c_double)]
    clibsignal.utl_lonlat.restype = C.c_void_p

    lon = C.c_double()
    lat = C.c_double()

    clibsignal.utl_lonlat(orig_lon, orig_lat, x, y, C.byref(lon), C.byref(lat))
    return lon.value, lat.value


def next_pow_2(i):
    """
    Find the next power of two

    >>> int(next_pow_2(5))
    8
    >>> int(next_pow_2(250))
    256
    """
    # do not use NumPy here, math is much faster for single values
    buf = math.ceil(math.log(i) / math.log(2))
    return native(int(math.pow(2, buf)))


def prev_pow_2(i):
    """
    Find the previous power of two

    >>> prev_pow_2(5)
    4
    >>> prev_pow_2(250)
    128
    """
    # do not use NumPy here, math is much faster for single values
    return int(math.pow(2, math.floor(math.log(i, 2))))


def nearest_pow_2(x):
    """
    Finds the nearest integer that is a power of 2.
    In contrast to :func:`next_pow_2` also searches for numbers smaller than
    the input and returns them if they are closer than the next bigger power
    of 2.
    """
    a = math.pow(2, math.ceil(math.log(x, 2)))
    b = math.pow(2, math.floor(math.log(x, 2)))
    if abs(a - x) < abs(b - x):
        return int(a)
    else:
        return int(b)


def enframe(x, win, inc):
    """
    Splits the vector up into (overlapping) frames beginning at increments
    of inc. Each frame is multiplied by the window win().
    The length of the frames is given by the length of the window win().
    The centre of frame I is x((I-1)*inc+(length(win)+1)/2) for I=1,2,...

    :param x: signal to split in frames
    :param win: window multiplied to each frame, length determines frame length
    :param inc: increment to shift frames, in samples
    :return f: output matrix, each frame occupies one row
    :return length, no_win: length of each frame in samples, number of frames
    """
    nx = len(x)
    nwin = len(win)
    if (nwin == 1):
        length = win
    else:
        # length = next_pow_2(nwin)
        length = nwin
    nf = int(fix((nx - length + inc) // inc))
    # f = np.zeros((nf, length))
    indf = inc * np.arange(nf)
    inds = np.arange(length) + 1
    f = x[(np.transpose(np.vstack([indf] * length)) +
           np.vstack([inds] * nf)) - 1]
    if (nwin > 1):
        w = np.transpose(win)
        f = f * np.vstack([w] * nf)
    f = signal.detrend(f, type='constant')
    no_win, _ = f.shape
    return f, length, no_win


def smooth(x, smoothie):
    """
    Smooths a given signal by computing a central moving average.

    :param x: signal to smooth
    :param smoothie: number of past/future values to calculate moving average
    :return out: smoothed signal
    """
    size_x = np.size(x)
    if smoothie > 0:
        if (len(x) > 1 and len(x) < size_x):
            # out_add = append(append([x[0,:]]*smoothie,x,axis=0),
            #                     [x[(len(x)-1),:]]*smoothie,axis=0)
            # out_add = (np.append([x[0, :]]*int(smoothie), x, axis=0))
            out_add = np.vstack(([x[0, :]] * int(smoothie), x,
                                 [x[(len(x) - 1), :]] * int(smoothie)))
            help = np.transpose(out_add)
            # out = signal.lfilter(np.ones(smoothie) / smoothie, 1, help)
            out = signal.lfilter(
                np.hstack((np.ones(smoothie) / (2 * smoothie), 0,
                           np.ones(smoothie) / (2 * smoothie))), 1, help)
            out = np.transpose(out)
            # out = out[smoothie:len(out), :]
            out = out[2 * smoothie:len(out), :]
            # out = filter(ones(1,smoothie)/smoothie,1,out_add)
            # out[1:smoothie,:] = []
        else:
            # out_add = np.append(np.append([x[0]] * smoothie, x),
            #                   [x[size_x - 1]] * smoothie)
            out_add = np.hstack(([x[0]] * int(smoothie), x,
                                 [x[(len(x) - 1)]] * int(smoothie)))
            out = signal.lfilter(np.hstack((
                np.ones(smoothie) / (2 * smoothie), 0,
                np.ones(smoothie) / (2 * smoothie))), 1, out_add)
            out = out[2 * smoothie:len(out)]
            out[0:smoothie] = out[smoothie]
            out[len(out) - smoothie:len(out)] = out[len(out) - smoothie - 1]
            # for i in xrange(smoothie, len(x) + smoothie):
            #    sum = 0
            #    for k in xrange(-smoothie, smoothie):
            #        sum = sum + out_add[i + k]
            #        suma[i - smoothie] = float(sum) / (2 * smoothie)
            #        out = suma
            #        out[0:smoothie] = out[smoothie]
            #        out[size_x - 1 - smoothie:size_x] = \
            #            out[size_x - 1 - smoothie]
    else:
        out = x
    return out


def rdct(x, n=0):
    """
    Computes discrete cosine transform of given signal.
    Signal is truncated/padded to length n.

    :params x: signal to compute discrete cosine transform
    :params n: window length (default: signal length)
    :return y: discrete cosine transform
    """
    m, k = x.shape
    if (n == 0):
        n = m
        a = np.sqrt(2 * n)
        x = np.append([x[0:n:2, :]], [x[2 * int(np.fix(n / 2)):0:-2, :]],
                      axis=1)
        x = x[0, :, :]
        z = np.append(np.sqrt(2.), 2. * np.exp((-0.5j * float(np.pi / n)) *
                                               np.arange(1, n)))
        y = np.real(np.multiply(np.transpose(fftpack.fft(np.transpose(x))),
                    np.transpose(np.array([z])) * np.ones(k))) / float(a)
        return y


def az2baz2az(angle):
    """
    Helper function to convert from azimuth to backazimuth or from backazimuth
    to azimuth.

    :type angle: float or int
    :param angle: azimuth or backazimuth value in degrees between 0 and 360.
    :return: corresponding backazimuth or azimuth value in degrees.
    """
    if 0 <= angle <= 180:
        new_angle = angle + 180
    elif 180 < angle <= 360:
        new_angle = angle - 180
    else:
        raise ValueError("Input (back)azimuth out of bounds: %s" % angle)
    return new_angle


def _npts2nfft(npts, smart=True):
    """
    Calculates number of points for fft from number of samples in trace.
    When encountering bad values with prime factors involved (that can take
    forever to compute) we try a few slightly larger numbers for a good
    factorization (computation time for factorization is negligible compared to
    fft/evalsresp/ifft) and if that fails we use the next power of 2 which is
    not fastest but robust.

    >>> _npts2nfft(1800028)  # good nfft with minimum points
    3600056
    >>> int(_npts2nfft(1800029))  # falls back to next power of 2
    4194304
    >>> _npts2nfft(1800031)  # finds suitable nfft close to minimum npts
    3600082
    """
    # The number of points for the FFT has to be at least 2 * ndat (in
    # order to prohibit wrap around effects during convolution) cf.
    # Numerical Recipes p. 429 calculate next power of 2.
    # evalresp scales directly with nfft, therefor taking the next power of
    # two has a greater negative performance impact than the slow down of a
    # not power of two in the FFT
    if npts & 0x1:  # check if uneven
        nfft = 2 * (npts + 1)
    else:
        nfft = 2 * npts

    def _good_factorization(x):
        if max(factorize_int(x)) < 500:
            return True
        return False

    # check if we have a bad factorization with large primes
    if smart and nfft > 5000 and not _good_factorization(nfft):
        # try a few numbers slightly larger for a suitable factorization
        # in most cases after less than 10 tries a suitable nfft number with
        # good factorization is found
        for i_ in range(1, 11):
            trial = int(nfft + 2 * i_)
            if _good_factorization(trial):
                nfft = trial
                break
        else:
            nfft = next_pow_2(nfft)

    return nfft


def stack(data, stack_type='linear'):
    """
    Stack data by first axis.

    :type stack_type: str or tuple
    :param stack_type: Type of stack, one of the following:
        ``'linear'``: average stack (default),
        ``('pw', order)``: phase weighted stack of given order,
        see [Schimmel1997]_,
        ``('root', order)``: root stack of given order.
    """
    if stack_type == 'linear':
        stack = np.mean(data, axis=0)
    elif stack_type[0] == 'pw':
        from scipy.signal import hilbert
        from scipy.fftpack import next_fast_len
        npts = np.shape(data)[1]
        nfft = next_fast_len(npts)
        anal_sig = hilbert(data, N=nfft)[:, :npts]
        norm_anal_sig = anal_sig / np.abs(anal_sig)
        phase_stack = np.abs(np.mean(norm_anal_sig, axis=0)) ** stack_type[1]
        stack = np.mean(data, axis=0) * phase_stack
    elif stack_type[0] == 'root':
        r = np.mean(np.sign(data) * np.abs(data)
                    ** (1 / stack_type[1]), axis=0)
        stack = np.sign(r) * np.abs(r) ** stack_type[1]
    else:
        raise ValueError('stack type is not valid.')
    return stack


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
