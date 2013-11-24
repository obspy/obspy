#!/usr/bin/env python
# -*- coding: utf-8 -*-
import platform


lib_name = 'libtau_%s_%s_py%s' % (platform.system(),
        platform.architecture()[0], ''.join([str(i) for i in
        platform.python_version_tuple()[:2]]))

# Import libtau in a platform specific way.
try:
    libtau = __import__(lib_name, globals(), locals())
    libtau_ttimes = libtau.ttimes
except ImportError:
    if platform.system() != "Windows":
        raise
    # try ctypes
    import ctypes as C
    from distutils import sysconfig
    import numpy as np
    lib_extension, = sysconfig.get_config_vars('SO')
    libtau = C.CDLL(lib_name + lib_extension)

    def libtau_ttimes(delta, depth, modnam):
        delta = C.c_float(delta)
        depth = C.c_float(abs(depth))
        # initialize some arrays...
        phase_names = (C.c_char * 8 * 60)()
        flags = ['F_CONTIGUOUS', 'ALIGNED', 'WRITEABLE']
        tt = np.zeros(60, 'float32', flags)
        toang = np.zeros(60, 'float32', flags)
        dtdd = np.zeros(60, 'float32', flags)
        dtdh = np.zeros(60, 'float32', flags)
        dddp = np.zeros(60, 'float32', flags)

        libtau.ttimes_(C.byref(delta), C.byref(depth), modnam, phase_names,
               tt.ctypes.data_as(C.POINTER(C.c_float)),
               toang.ctypes.data_as(C.POINTER(C.c_float)),
               dtdd.ctypes.data_as(C.POINTER(C.c_float)),
               dtdh.ctypes.data_as(C.POINTER(C.c_float)),
               dddp.ctypes.data_as(C.POINTER(C.c_float)))
        phase_names = np.array([p.value for p in phase_names])
        return phase_names, tt, toang, dtdd, dtdh, dddp
