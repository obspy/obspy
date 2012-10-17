# -*- coding: utf-8 -*-
"""
Utilities for obspy.taup
"""

from distutils import sysconfig
import ctypes as C
import os
import platform


# Import shared libtaup depending on the platform.
# create library names
lib_names = [
     # platform specific library name
    'libtaup-%s-%s-py%s' % (platform.system(), platform.architecture()[0],
        ''.join([str(i) for i in platform.python_version_tuple()[:2]])),
     # fallback for pre-packaged libraries
    'libtaup']
# get default file extension for shared objects
lib_extension, = sysconfig.get_config_vars('SO')
# initialize library
flibtaup = None
for lib_name in lib_names:
    try:
        flibtaup = C.CDLL(os.path.join(os.path.dirname(__file__), os.pardir,
                                       'lib', lib_name + lib_extension))
    except Exception, e:
        pass
    else:
        break
if not flibtaup:
    msg = 'Could not load shared library for obspy.taup.\n\n %s' % (e)
    raise ImportError(msg)
