# -*- coding: utf-8 -*-
"""
C wrappers for some crucial inner loops of TauPy written in C.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import native_str

import ctypes as C  # NOQA
import numpy as np

from obspy.core.util.libnames import _load_cdll
from .helper_classes import SlownessLayer, TimeDist


clibtau = _load_cdll("tau")


clibtau.tau_branch_calc_time_dist_inner_loop.argtypes = [
    # ray_params
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2,
                           flags=native_str('C_CONTIGUOUS')),
    # time
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2,
                           flags=native_str('C_CONTIGUOUS')),
    # dist
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2,
                           flags=native_str('C_CONTIGUOUS')),
    # layer, record array, 64bit floats. 2D array in memory
    np.ctypeslib.ndpointer(dtype=SlownessLayer, ndim=1,
                           flags=native_str('C_CONTIGUOUS')),
    # time_dist, record array, 64bit floats. 2D array in memory
    np.ctypeslib.ndpointer(dtype=TimeDist, ndim=1,
                           flags=native_str('C_CONTIGUOUS')),
    # max_i
    C.c_int32,
    # max_j
    C.c_int32,
    # max ray param
    C.c_double,
    # allow_turn
    C.c_int32
]
clibtau.tau_branch_calc_time_dist_inner_loop.restype = None


clibtau.seismic_phase_calc_time_inner_loop.argtypes = [
    # degree
    C.c_double,
    # max_distance
    C.c_double,
    # dist
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1,
                           flags=native_str('C_CONTIGUOUS')),
    # ray_param
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1,
                           flags=native_str('C_CONTIGUOUS')),
    # search_dist_results
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1,
                           flags=native_str('C_CONTIGUOUS')),
    # ray_num_results
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1,
                           flags=native_str('C_CONTIGUOUS')),
    # count
    C.c_int
]
clibtau.seismic_phase_calc_time_inner_loop.restype = C.c_int


clibtau.bullen_radial_slowness_inner_loop.argtypes = [
    # layer, record array, 64bit floats. 2D array in memory
    np.ctypeslib.ndpointer(dtype=SlownessLayer, ndim=1,
                           flags=native_str('C_CONTIGUOUS')),
    # p
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1,
                           flags=native_str('C_CONTIGUOUS')),
    # time
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1,
                           flags=native_str('C_CONTIGUOUS')),
    # dist
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1,
                           flags=native_str('C_CONTIGUOUS')),
    # radius
    C.c_double,
    # max_i
    C.c_int
]
clibtau.bullen_radial_slowness_inner_loop.restype = None
