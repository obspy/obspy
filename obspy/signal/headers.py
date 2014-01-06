# -*- coding: utf-8 -*-
"""
Defines the libsignal and evalresp structures and blockettes.
"""
from __future__ import unicode_literals
from future.builtins import str
from future.utils import native_str

from distutils import sysconfig
import ctypes as C
import numpy as np
import os
import platform


# Import shared libsignal depending on the platform.
# create library names
lib_names = [
    # python3.3 platform specific library name
    'libsignal_%s_%s_py%s.cpython-%sm' % (
        platform.system(), platform.architecture()[0],
        ''.join([str(i) for i in platform.python_version_tuple()[:2]]),
        ''.join([str(i) for i in platform.python_version_tuple()[:2]])),
    # platform specific library name
    'libsignal_%s_%s_py%s' % (
        platform.system(), platform.architecture()[0],
        ''.join([str(i) for i in platform.python_version_tuple()[:2]])),
    # fallback for pre-packaged libraries
    'libsignal']
# get default file extension for shared objects
lib_extension, = sysconfig.get_config_vars('SO')
# initialize library
for lib_name in lib_names:
    try:
        clibsignal = C.CDLL(os.path.join(os.path.dirname(__file__), os.pardir,
                                         'lib', lib_name + lib_extension))
        break
    except Exception as e:
        err_msg = str(e)
        pass
else:
    msg = 'Could not load shared library for obspy.signal.\n\n %s' % err_msg
    raise ImportError(msg)

# Import shared libevresp depending on the platform.
# create library names
erlib_names = [
    # python3.3 platform specific library name
    'libevresp_%s_%s_py%s.cpython-%sm' % (
        platform.system(), platform.architecture()[0],
        ''.join([str(i) for i in platform.python_version_tuple()[:2]]),
        ''.join([str(i) for i in platform.python_version_tuple()[:2]])),
    # platform specific library name
    'libevresp_%s_%s_py%s' % (
        platform.system(), platform.architecture()[0],
        ''.join([str(i) for i in platform.python_version_tuple()[:2]])),
    # fallback for pre-packaged libraries
    'libevresp']
# initialize library
for erlib_name in erlib_names:
    try:
        clibevresp = C.CDLL(os.path.join(os.path.dirname(__file__), os.pardir,
                                         'lib', erlib_name + lib_extension))
        break
    except Exception as e:
        err_msg = str(e)
        pass
else:
    msg = 'Could not load shared library for ' + \
          'obspy.signal.invsim.evalresp\n\n %s' % err_msg
    raise ImportError(msg)

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
