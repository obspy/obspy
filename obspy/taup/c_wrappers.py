# -*- coding: utf-8 -*-
"""
C wrappers for some crucial inner loops of TauPy written in C.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import native_str

import ctypes as C
import numpy as np
from obspy.core.util.libnames import _load_CDLL

from .slowness_layer import SlownessLayer
from .helper_classes import TimeDist


clibtau = _load_CDLL("tau")

clibtau.tau_branch_calc_time_dist_inner_loop.argtypes = [
    # ray_params
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2,
                           flags=native_str('C_CONTIGUOUS')),
    # mask
    np.ctypeslib.ndpointer(dtype=np.bool, ndim=2,
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
    C.c_double
]
clibtau.tau_branch_calc_time_dist_inner_loop.restype = C.c_void_p
