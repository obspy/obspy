# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from future.builtins import str

from distutils import sysconfig
from struct import unpack
import ctypes as C
import os
import platform

# Import shared libsegy depending on the platform.
# create library names
lib_names = [
    # python3.3 platform specific library name
    'libsegy_%s_%s_py%s.cpython-%sm' % (
        platform.system(), platform.architecture()[0],
        ''.join([str(i) for i in platform.python_version_tuple()[:2]]),
        ''.join([str(i) for i in platform.python_version_tuple()[:2]])),
    # platform specific library name
    'libsegy_%s_%s_py%s' % (
        platform.system(), platform.architecture()[0],
        ''.join([str(i) for i in platform.python_version_tuple()[:2]])),
    # fallback for pre-packaged libraries
    'libsegy']
# get default file extension for shared objects
lib_extension, = sysconfig.get_config_vars('SO')
# initialize library
for lib_name in lib_names:
    try:
        clibsegy = C.CDLL(os.path.join(os.path.dirname(__file__), os.pardir,
                                       'lib', lib_name + lib_extension))
        break
    except Exception as e:
        err_msg = str(e)
        pass
else:
    msg = 'Could not load shared library for obspy.segy.\n\n %s' % err_msg
    raise ImportError(msg)


def unpack_header_value(endian, packed_value, length, special_format):
    """
    Unpacks a single value.
    """
    # Use special format if necessary.
    if special_format:
        fmt = ('%s%s' % (endian, special_format)).encode('ascii', 'strict')
        return unpack(fmt, packed_value)[0]
    # Unpack according to different lengths.
    elif length == 2:
        format = ('%sh' % endian).encode('ascii', 'strict')
        return unpack(format, packed_value)[0]
    # Update: Seems to be correct. Two's complement integers seem to be
    # the common way to store integer values.
    elif length == 4:
        format = ('%si' % endian).encode('ascii', 'strict')
        return unpack(format, packed_value)[0]
    # The unassigned field. Since it is unclear how this field is
    # encoded it will just be stored as a string.
    elif length == 8:
        return packed_value
    # Should not happen
    else:
        raise Exception
