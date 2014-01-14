#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Various additional utilities for obspy.signal.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import division
from __future__ import unicode_literals
from future.builtins import int

from scipy import signal, fix, fftpack
import ctypes as C
import math as M
import numpy as np
from obspy.signal.headers import clibsignal


def utlGeoKm(orig_lon, orig_lat, lon, lat):
    """
    Transform lon, lat to km in reference to orig_lon and orig_lat

    >>> utlGeoKm(12.0, 48.0, 12.0, 48.0)
    (0.0, 0.0)
    >>> x, y = utlGeoKm(12.0, 48.0, 13.0, 49.0)
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


def utlLonLat(orig_lon, orig_lat, x, y):
    """
    Transform x, y [km] to decimal degree in reference to orig_lon and orig_lat

    >>> utlLonLat(12.0, 48.0, 0.0, 0.0)
    (12.0, 48.0)
    >>> lon, lat = utlLonLat(12.0, 48.0, 73.9041, 111.1908)
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


def nextpow2(i):
    """
    Find the next power of two

    >>> nextpow2(5)
    8
    >>> nextpow2(250)
    256
    """
    # do not use numpy here, math is much faster for single values
    buf = M.ceil(M.log(i) / M.log(2))
    return int(M.pow(2, buf))


def prevpow2(i):
    """
    Find the previous power of two

    >>> prevpow2(5)
    4
    >>> prevpow2(250)
    128
    """
    # do not use numpy here, math is much faster for single values
    return int(M.pow(2, M.floor(M.log(i, 2))))


def nearestPow2(x):
    """
    Finds the nearest integer that is a power of 2.
    In contrast to :func:`nextpow2` also searches for numbers smaller than the
    input and returns them if they are closer than the next bigger power of 2.
    """
    a = M.pow(2, M.ceil(M.log(x, 2)))
    b = M.pow(2, M.floor(M.log(x, 2)))
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
        #length = nextpow2(nwin)
        length = nwin
    nf = int(fix((nx - length + inc) // inc))
    #f = np.zeros((nf, length))
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
    Smoothes a given signal by computing a central moving average.

    :param x: signal to smooth
    :param smoothie: number of past/future values to calculate moving average
    :return out: smoothed signal
    """
    size_x = np.size(x)
    if smoothie > 0:
        if (len(x) > 1 and len(x) < size_x):
            #out_add = append(append([x[0,:]]*smoothie,x,axis=0),
            #                     [x[(len(x)-1),:]]*smoothie,axis=0)
            #out_add = (np.append([x[0, :]]*int(smoothie), x, axis=0))
            out_add = np.vstack(([x[0, :]] * int(smoothie), x,
                                 [x[(len(x) - 1), :]] * int(smoothie)))
            help = np.transpose(out_add)
            #out = signal.lfilter(np.ones(smoothie) / smoothie, 1, help)
            out = signal.lfilter(
                np.hstack((np.ones(smoothie) / (2 * smoothie), 0,
                          np.ones(smoothie) / (2 * smoothie))), 1, help)
            out = np.transpose(out)
            #out = out[smoothie:len(out), :]
            out = out[2 * smoothie:len(out), :]
            #out = filter(ones(1,smoothie)/smoothie,1,out_add)
            #out[1:smoothie,:] = []
        else:
            #out_add = np.append(np.append([x[0]] * smoothie, x),
            #                   [x[size_x - 1]] * smoothie)
            out_add = np.hstack(([x[0]] * int(smoothie), x,
                                 [x[(len(x) - 1)]] * int(smoothie)))
            out = signal.lfilter(np.hstack((
                np.ones(smoothie) / (2 * smoothie), 0,
                np.ones(smoothie) / (2 * smoothie))), 1, out_add)
            out = out[2 * smoothie:len(out)]
            out[0:smoothie] = out[smoothie]
            out[len(out) - smoothie:len(out)] = out[len(out) - smoothie - 1]
            #for i in xrange(smoothie, len(x) + smoothie):
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


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
