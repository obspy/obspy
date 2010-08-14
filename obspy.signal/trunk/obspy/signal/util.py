#!/usr/bin/env python
"""
Various additional utilities for obspy.signal.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from scipy import signal, fix, fftpack
import ctypes as C
import math as M
import numpy as np
import os
import platform

from obspy.core import Trace, Stream


# Import shared libsignal library depending on the platform.
# XXX: trying multiple names for now - should be removed
if platform.system() == 'Windows':
    if platform.architecture()[0] == '64bit':
        lib_names = ['libsignal.pyd', '_libsignal.win64.dll']
    else:
        lib_names = ['libsignal.pyd', '_libsignal.win32.dll']
elif platform.system() == 'Darwin':
    lib_names = ['libsignal.so', '_libsignal.dylib']
else:
    # 32 and 64 bit UNIX
    #XXX Check glibc version by platform.libc_ver()
    if platform.architecture()[0] == '64bit':
        lib_names = ['libsignal.so', '_libsignal.lin64.so']
    else:
        lib_names = ['libsignal.so', '_libsignal.so']

# initialize library
lib = None
for lib_name in lib_names:
    try:
        lib = C.CDLL(os.path.join(os.path.dirname(__file__), 'lib', lib_name))
    except:
        continue
    else:
        break
if not lib:
    msg = 'Could not load shared library "libsignal" for obspy.signal.'
    raise ImportError(msg)


def utlGeoKm(orig_lon, orig_lat, lon, lat):
    """
    Transform lon, lat to km in reference to orig_lon and orig_lat
    
    >>> utlGeoKm(12.0, 48.0, 12.0, 48.0)
    (0.0, 0.0)
    >>> utlGeoKm(12.0, 48.0, 13.0, 49.0)
    (73.904144287109375, 111.19082641601562)
    
    :param orig_lon: Longitude of reference origin
    :param orig_lat: Latitude of reference origin
    :param lat: Latitude to calculate relative coordinate in km
    :param lon: Longitude to calculate relative coordinate in km
    :return: x, y coordinate in km (in reference to origin)
    """
    # 2009-10-11 Moritz

    lib.utl_geo_km.argtypes = [C.c_float, C.c_float, C.c_float,
                               C.POINTER(C.c_float), C.POINTER(C.c_float)]
    lib.utl_geo_km.restype = C.c_void_p

    x = C.c_float(lon)
    y = C.c_float(lat)

    lib.utl_geo_km(orig_lon, orig_lat, 0.0, C.byref(x), C.byref(y))
    return x.value, y.value


def utlLonLat(orig_lon, orig_lat, x, y):
    """
    Transform x, y [km] to decimal degree in reference to orig_lon and orig_lat
    
    >>> utlLonLat(12.0, 48.0, 0.0, 0.0)
    (12.0, 48.0)
    >>> utlLonLat(12.0, 48.0, 73.904144287109375, 111.19082641601562)
    (13.0, 49.0)
    
    :param orig_lon: Longitude of reference origin
    :param orig_lat: Latitude of reference origin
    :param x: value [km] to calculate relative coordinate in degree
    :param y: value [km] to calculate relative coordinate in degree
    :return: lon, lat coordinate in degree (absolute)
    """
    # 2009-10-11 Moritz

    lib.utl_lonlat.argtypes = [C.c_float, C.c_float, C.c_float, C.c_float,
                               C.POINTER(C.c_float), C.POINTER(C.c_float)]
    lib.utl_lonlat.restype = C.c_void_p

    lon = C.c_float()
    lat = C.c_float()

    lib.utl_lonlat(orig_lon, orig_lat, x, y, C.byref(lon), C.byref(lat))
    return lon.value, lat.value


def xcorr(tr1, tr2, shift_len, full_xcorr=False):
    """
    Cross correlation of tr1 and tr2 in the time domain using window_len.
    Supports ndarrays (float32) and Trace objects.
    
    >>> tr1 = np.random.randn(10000).astype('float32')
    >>> tr2 = tr1.copy()
    >>> a, b = xcorr(tr1, tr2, 1000)
    >>> print a, round(b, 7)
    0 1.0

    ::

                                    Mid Sample
                                        |                           
        |AAAAAAAAAAAAAAA|AAAAAAAAAAAAAAA|AAAAAAAAAAAAAAA|AAAAAAAAAAAAAAA|
        |BBBBBBBBBBBBBBB|BBBBBBBBBBBBBBB|BBBBBBBBBBBBBBB|BBBBBBBBBBBBBBB|
        |<-shift_len/2->|   <- region of support ->     |<-shift_len/2->|

    
    :type tr1: numpy ndarray float32 or obspy.core.Trace
    :param tr1: Trace 1
    :type tr2: numpy ndarray float32 or obspy.core.Trace
    :param tr2: Trace 2 to correlate with trace 1
    :type shift_len: Int
    :param shift_len: Total length of samples to shift for cross correlation.
    :type full_xcorr: Bool
    :param full_xcorr: If True, the complete xcorr function will be
        returned as numpy.ndarray
    :return: (index, value, fct) index of maximum xcorr value and the value
        itself. The complete xcorr function is returned only if
        ``full_xcorr=True``.
    """
    # if we get Trace objects, use their data arrays
    for tr in [tr1, tr2]:
        if isinstance(tr, Trace):
            tr = tr.data

    # 2009-10-11 Moritz
    lib.X_corr.argtypes = [
        np.ctypeslib.ndpointer(dtype='float32', ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype='float32', ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype='float64', ndim=1, flags='C_CONTIGUOUS'),
        C.c_int, C.c_int, C.c_int,
        C.POINTER(C.c_int), C.POINTER(C.c_double)]
    lib.X_corr.restype = C.c_void_p

    # be nice and adapt type if necessary
    tr1 = np.require(tr1, 'float32', ['C_CONTIGUOUS'])
    tr2 = np.require(tr2, 'float32', ['C_CONTIGUOUS'])
    corp = np.empty(2 * shift_len + 1, dtype='float64', order='C')

    shift = C.c_int()
    coe_p = C.c_double()

    lib.X_corr(tr1, tr2, corp, shift_len, len(tr1), len(tr2),
               C.byref(shift), C.byref(coe_p))

    if full_xcorr:
        return shift.value, coe_p.value, corp
    else:
        return shift.value, coe_p.value


def xcorr_3C(st1, st2, shift_len, components=["Z", "N", "E"],
             full_xcorr=False):
    """
    Calculates the cross correlation on each of the specified components
    separately, stacks them together and estimates the maximum and shift of
    maximum on the stack.
    Basically the same as :func:`~obspy.signal.util.xcorr` but for (normally)
    three components, please also take a look at the documentation of that
    function. Useful e.g. for estimation of waveform similarity on a three
    component seismogram.
    
    :type st1: obspy.core.Stream
    :param st1: Stream 1, containing one trace for Z, N, E component (other
            component_id codes are ignored)
    :type st2: obspy.core.Stream
    :param st2: Stream 2, containing one trace for Z, N, E component (other
            component_id codes are ignored)
    :type shift_len: Int
    :param shift_len: Total length of samples to shift for cross correlation.
    :type components: List of Strings
    :param components: List of components to use in cross-correlation, defaults
            to ["Z", "N", "E"].
    :type full_xcorr: Bool
    :param full_xcorr: If True, the complete xcorr function will be
        returned as numpy.ndarray
    :return: (index, value, fct) index of maximum xcorr value and the value
        itself. The complete xcorr function is returned only if
        ``full_xcorr=True``.
    """
    streams = [st1, st2]
    # check if we can actually use the provided streams safely
    for st in streams:
        if not isinstance(st, Stream):
            raise TypeError("Expected Stream object but got %s." % type(st))
        for component in components:
            if not len(st.select(component=component)) == 1:
                msg = "Expected exactly one %s trace in stream" % component + \
                      " but got %s." % len(st.select(component=component))
                raise ValueError(msg)
    ndat = len(streams[0].select(component=components[0])[0])
    if False in [len(st.select(component=component)[0]) == ndat \
                 for st in streams for component in components]:
            raise ValueError("All traces have to be the same length.")
    # everything should be ok with the input data...

    corp = np.zeros(2 * shift_len + 1, dtype='float64', order='C')

    for component in components:
        xx = xcorr(streams[0].select(component=component)[0],
                   streams[1].select(component=component)[0],
                   shift_len, full_xcorr=True)
        corp += xx[2]

    corp /= len(components)

    shift, value = xcorr_max(corp)

    if full_xcorr:
        return shift, value, corp
    else:
        return shift, value


def xcorr_max(fct):
    """
    Return shift and value of maximum xcorr function
    
    >>> fct = np.zeros(101)
    >>> fct[50] = -1.0
    >>> xcorr_max(fct)
    (0, -1.0)
    >>> fct[50], fct[60] = 0.0, 1.0
    >>> xcorr_max(fct)
    (10, 1.0)
    >>> fct[60], fct[40] = 0.0, -1.0
    >>> xcorr_max(fct)
    (-10, -1.0)
    
    :param fct: numpy.ndarray
        xcorr function e.g. returned bei xcorr
    :return: (shift, value) Shift and value of maximum xcorr
    """
    value = fct.max()
    _min = fct.min()
    if abs(_min) > abs(value):
        value = _min

    mid = (len(fct) - 1) / 2
    shift = np.where(fct == value)[0][0] - mid
    return shift, value


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


def enframe(x, win, inc):
    nx = len(x)
    nwin = len(win)
    if (nwin == 1):
        length = win
    else:
        length = nextpow2(nwin)
    nf = int(fix((nx - length + inc) // inc))
    #f = np.zeros((nf, length))
    indf = inc * np.arange(nf)
    inds = np.arange(length) + 1
    f = x[(np.transpose(np.vstack([indf] * length)) + \
           np.vstack([inds] * nf)) - 1]
    if (nwin > 1):
        w = np.transpose(win)
        f = f * np.vstack([w] * nf)
    f = signal.detrend(f, type='constant')
    no_win, _ = f.shape
    return f, length, no_win


def smooth(x, smoothie):
    suma = np.zeros(np.size(x))
    if smoothie > 1:
        if (len(x) > 1 and len(x) < np.size(x)):
            #out_add = append(append([x[0,:]]*smoothie,x,axis=0),
            #                     [x[(len(x)-1),:]]*smoothie,axis=0)
            out_add = (np.append([x[0, :]]*int(smoothie), x, axis=0))
            help = np.transpose(out_add)
            out = signal.lfilter(np.ones(smoothie) / smoothie, 1, help)
            out = np.transpose(out)
            out = out[smoothie:len(out), :]
            #out = filter(ones(1,smoothie)/smoothie,1,out_add)
            #out[1:smoothie,:] = []
        else:
            out_add = np.append(np.append([x[0]] * smoothie, x),
                               [x[np.size(x) - 1]] * smoothie)
            for i in xrange(smoothie, len(x) + smoothie):
                sum = 0
                for k in range(-smoothie, smoothie):
                    sum = sum + out_add[i + k]
                    suma[i - smoothie] = float(sum) / (2 * smoothie)
                    out = suma
                    out[0:smoothie] = out[smoothie]
                    out[np.size(x) - 1 - smoothie:np.size(x)] = \
                        out[np.size(x) - 1 - smoothie]
    else:
        out = x
    return out


def rdct(x, n=0):
    m, k = x.shape
    if (n == 0):
        n = m
        a = np.sqrt(2 * n)
        x = np.append([x[0:n:2, :]], [x[2 * np.fix(n / 2):0:-2, :]], axis=1)
        x = x[0, :, :]
        z = np.append(np.sqrt(2.), 2. * np.exp((-0.5j * float(np.pi / n)) *
                                   np.arange(1, n)))
        y = np.real(np.multiply(np.transpose(fftpack.fft(np.transpose(x))),
                          np.transpose(np.array([z])) * np.ones(k))) / float(a)
        return y


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
