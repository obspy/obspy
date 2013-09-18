# -*- coding: utf-8 -*-

from distutils import sysconfig
from struct import unpack
import ctypes as C
import os
import platform

# Import shared libsegy depending on the platform.
# create library names
lib_names = [
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
    except Exception, e:
        pass
else:
    msg = 'Could not load shared library for obspy.segy.\n\n %s' % (e)
    raise ImportError(msg)


def unpack_header_value(endian, packed_value, length, special_format):
    """
    Unpacks a single value.
    """
    # Use special format if necessary.
    if special_format:
        format = '%s%s' % (endian, special_format)
        return unpack(format, packed_value)[0]
    # Unpack according to different lengths.
    elif length == 2:
        format = '%sh' % endian
        return unpack(format, packed_value)[0]
    # Update: Seems to be correct. Two's complement integers seem to be
    # the common way to store integer values.
    elif length == 4:
        format = '%si' % endian
        return unpack(format, packed_value)[0]
    # The unassigned field. Since it is unclear how this field is
    # encoded it will just be stored as a string.
    elif length == 8:
        return packed_value
    # Should not happen
    else:
        raise Exception
