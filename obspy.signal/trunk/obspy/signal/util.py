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


def xcorr(tr1, tr2, window_len):
    """
    Cross correlation of tr1 and tr2 in the time domain using window_len.
    
    >>> tr1 = np.random.randn(10000).astype('float32')
    >>> tr2 = tr1.copy()
    >>> a, b = xcorr(tr1, tr2, 1000)
    >>> print a, round(b, 7)
    0 1.0
    
    :type tr1: numpy ndarray float32
    :param tr1: Trace 1
    :type tr2: numpy ndarray float32
    :param tr2: Trace 2 to correlate with trace 1
    :type window_len: Int
    :param window_len: Window length of cross correlation in samples
    :return: (index, value) index of maximum xcorr value and the value itself
    """
    # 2009-10-11 Moritz
    lib.X_corr.argtypes = [
        np.ctypeslib.ndpointer(dtype='float32', ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype='float32', ndim=1, flags='C_CONTIGUOUS'),
        C.c_int, C.c_int, C.c_int,
        C.POINTER(C.c_int), C.POINTER(C.c_double)]
    lib.X_corr.restype = C.c_void_p

    # be nice and adapt type if necessary
    tr1 = np.require(tr1, 'float32', ['C_CONTIGUOUS'])
    tr2 = np.require(tr2, 'float32', ['C_CONTIGUOUS'])

    shift = C.c_int()
    coe_p = C.c_double()

    lib.X_corr(tr1, tr2, window_len, len(tr1), len(tr2),
               C.byref(shift), C.byref(coe_p))
    return shift.value, coe_p.value


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
