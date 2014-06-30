from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import native_str

import ctypes as C
import numpy as np

from obspy.core.util.libnames import _load_CDLL


# Import shared libnoise
clibnoise = _load_CDLL("noise")

clibnoise.phase_xcorr_loop.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.complex128, ndim=1,
                           flags=native_str('C_CONTIGUOUS')),
    np.ctypeslib.ndpointer(dtype=np.complex128, ndim=1,
                           flags=native_str('C_CONTIGUOUS')),
    C.c_int,
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1,
                           flags=native_str('C_CONTIGUOUS')),
    C.c_double, C.c_int, C.c_int]
clibnoise.phase_xcorr_loop.restype = C.c_void_p
