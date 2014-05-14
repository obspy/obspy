# -*- coding: utf-8 -*-
"""
Defines the libsignal and evalresp structures and blockettes.
"""
from __future__ import unicode_literals
from future.utils import native_str

import ctypes as C
import numpy as np
from obspy.core.util.misc import _load_CDLL


# Import shared libsignal
clibsignal = _load_CDLL("signal")
# Import shared libevresp
clibevresp = _load_CDLL("evresp")

clibsignal.calcSteer.argtypes = [
    C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, C.c_float,
    np.ctypeslib.ndpointer(dtype='f4', ndim=3,
                           flags=native_str('C_CONTIGUOUS')),
    np.ctypeslib.ndpointer(dtype='c16', ndim=4,
                           flags=native_str('C_CONTIGUOUS')),
]
clibsignal.calcSteer.restype = C.c_void_p

clibsignal.generalizedBeamformer.argtypes = [
    np.ctypeslib.ndpointer(dtype='f8', ndim=2,
                           flags=native_str('C_CONTIGUOUS')),
    np.ctypeslib.ndpointer(dtype='f8', ndim=2,
                           flags=native_str('C_CONTIGUOUS')),
    np.ctypeslib.ndpointer(dtype='c16', ndim=4,
                           flags=native_str('C_CONTIGUOUS')),
    np.ctypeslib.ndpointer(dtype='c16', ndim=3,
                           flags=native_str('C_CONTIGUOUS')),
    C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int,
    C.c_double,
    C.c_int,
]
clibsignal.generalizedBeamformer.restype = C.c_int

clibsignal.X_corr.argtypes = [
    np.ctypeslib.ndpointer(dtype='float32', ndim=1,
                           flags=native_str('C_CONTIGUOUS')),
    np.ctypeslib.ndpointer(dtype='float32', ndim=1,
                           flags=native_str('C_CONTIGUOUS')),
    np.ctypeslib.ndpointer(dtype='float64', ndim=1,
                           flags=native_str('C_CONTIGUOUS')),
    C.c_int, C.c_int, C.c_int,
    C.POINTER(C.c_int), C.POINTER(C.c_double)]
clibsignal.X_corr.restype = C.c_void_p

clibsignal.recstalta.argtypes = [
    np.ctypeslib.ndpointer(dtype='float64', ndim=1,
                           flags=native_str('C_CONTIGUOUS')),
    np.ctypeslib.ndpointer(dtype='float64', ndim=1,
                           flags=native_str('C_CONTIGUOUS')),
    C.c_int, C.c_int, C.c_int]
clibsignal.recstalta.restype = C.c_void_p

clibsignal.ppick.argtypes = [
    np.ctypeslib.ndpointer(dtype='float32', ndim=1,
                           flags=native_str('C_CONTIGUOUS')),
    C.c_int, C.POINTER(C.c_int), C.c_char_p, C.c_float, C.c_int, C.c_int,
    C.c_float, C.c_float, C.c_int, C.c_int]
clibsignal.ppick.restype = C.c_int

clibsignal.ar_picker.argtypes = [
    np.ctypeslib.ndpointer(dtype='float32', ndim=1,
                           flags=native_str('C_CONTIGUOUS')),
    np.ctypeslib.ndpointer(dtype='float32', ndim=1,
                           flags=native_str('C_CONTIGUOUS')),
    np.ctypeslib.ndpointer(dtype='float32', ndim=1,
                           flags=native_str('C_CONTIGUOUS')),
    C.c_int, C.c_float, C.c_float, C.c_float, C.c_float, C.c_float,
    C.c_float, C.c_float, C.c_int, C.c_int, C.POINTER(C.c_float),
    C.POINTER(C.c_float), C.c_double, C.c_double, C.c_int]
clibsignal.ar_picker.restypes = C.c_int

clibsignal.utl_geo_km.argtypes = [C.c_double, C.c_double, C.c_double,
                                  C.POINTER(C.c_double),
                                  C.POINTER(C.c_double)]
clibsignal.utl_geo_km.restype = C.c_void_p

head_stalta_t = np.dtype([
    (native_str('N'), b'u4', 1),
    (native_str('nsta'), b'u4', 1),
    (native_str('nlta'), b'u4', 1),
], align=True)

clibsignal.stalta.argtypes = [
    np.ctypeslib.ndpointer(dtype=head_stalta_t, ndim=1,
                           flags=native_str('C_CONTIGUOUS')),
    np.ctypeslib.ndpointer(dtype='f8', ndim=1,
                           flags=native_str('C_CONTIGUOUS')),
    np.ctypeslib.ndpointer(dtype='f8', ndim=1,
                           flags=native_str('C_CONTIGUOUS')),
]
clibsignal.stalta.restype = C.c_int


STALEN = 64
NETLEN = 64
CHALEN = 64
LOCIDLEN = 64


class C_COMPLEX(C.Structure):
    _fields_ = [("real", C.c_double),
                ("imag", C.c_double)]


class RESPONSE(C.Structure):
    pass

RESPONSE._fields_ = [("station", C.c_char * STALEN),
                     ("network", C.c_char * NETLEN),
                     ("locid", C.c_char * LOCIDLEN),
                     ("channel", C.c_char * CHALEN),
                     ("rvec", C.POINTER(C_COMPLEX)),
                     ("nfreqs", C.c_int),
                     ("freqs", C.POINTER(C.c_double)),
                     ("next", C.POINTER(RESPONSE))]

clibevresp.evresp.argtypes = [
    C.c_char_p,
    C.c_char_p,
    C.c_char_p,
    C.c_char_p,
    C.c_char_p,
    C.c_char_p,
    C.c_char_p,
    np.ctypeslib.ndpointer(dtype='float64',
                           ndim=1,
                           flags=native_str('C_CONTIGUOUS')),
    C.c_int,
    C.c_char_p,
    C.c_char_p,
    C.c_int,
    C.c_int,
    C.c_int,
    C.c_int]
clibevresp.evresp.restype = C.POINTER(RESPONSE)

clibevresp.free_response.argtypes = [C.POINTER(RESPONSE)]
clibevresp.free_response.restype = C.c_void_p
