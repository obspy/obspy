#!/usr/bin/env python

import ctypes as C
import numpy as N
import os
import platform


if platform.system() == 'Windows':
    lib_name = 'signal.win32.dll'
else:
    if platform.architecture()[0] == '64bit':
        lib_name = 'signal.lin64.so'
    else:
        lib_name = 'signal.so'

lib = C.CDLL(os.path.join(os.path.dirname(__file__), 'lib', lib_name))


def utlGeoKm(orig_lon, orig_lat, lon, lat):
    """
    Transform lon, lat to km in reference to orig_lon and orig_lat
    
    >>> utlGeoKm(12.0, 48.0, 12.0, 48.0)
    (0.0, 0.0)
    >>> utlGeoKm(12.0, 48.0, 13.0, 49.0)
    (73.904144287109375, 111.19082641601562)
    
    @param orig_lon: Longitude of reference origin
    @param orig_lat: Latitude of reference origin
    @param lat: Latitude to calculate relative coordinate in km
    @param lon: Longitude to calculate relative coordinate in km
    @return: x, y coordinate in km (in reference to origin)
    """
    # 2009-07-16 Moritz

    lib.utl_geo_km.argtypes = [C.c_float, C.c_float, C.c_float, C.c_void_p, C.c_void_p]
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
    
    @param orig_lon: Longitude of reference origin
    @param orig_lat: Latitude of reference origin
    @param x: value [km] to calculate relative coordinate in degree
    @param y: value [km] to calculate relative coordinate in degree
    @return: lon, lat coordinate in degree (absolute)
    """
    # 2009-07-24 Moritz

    lib.utl_lonlat.argtypes = [C.c_float, C.c_float, C.c_float, C.c_float,
                             C.c_void_p, C.c_void_p]
    lib.utl_lonlat.restype = C.c_void_p

    lon = C.c_float()
    lat = C.c_float()

    lib.utl_lonlat(orig_lon, orig_lat, x, y, C.byref(lon), C.byref(lat))

    return lon.value, lat.value


def xcorr(tr1, tr2, window_len):
    """
    Crosscorreltation of tr1 and tr2 in the time domain using window_len.
    
    >>> tr1 = N.random.randn(10000).astype('float32')
    >>> tr2 = tr1.copy()
    >>> a,b = xcorr(tr1, tr2, 1000)
    >>> a, round(1e6*b) # Rounding Errors
    (0, 1000000.0)
    
    @type tr1: numpy ndarray float32
    @param tr1: Trace 1
    @type tr2: numpy ndarray float32
    @param tr2: Trace 2 to correlate with trace 1
    @type window_len: Int
    @param window_len: Window length of cross correlation in samples
    """
    # 2009-07-10 Moritz
    lib.X_corr.argtypes = [
        N.ctypeslib.ndpointer(dtype='float32', ndim=1, flags='C_CONTIGUOUS'),
        N.ctypeslib.ndpointer(dtype='float32', ndim=1, flags='C_CONTIGUOUS'),
        C.c_int, C.c_int, C.c_void_p, C.c_void_p]
    lib.X_corr.restype = C.c_void_p

    shift = C.c_int()
    coe_p = C.c_double()

    lib.X_corr(tr1, tr2, window_len, len(tr1), len(tr2),
               C.byref(shift), C.byref(coe_p))

    return shift.value, coe_p.value


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
