# -*- coding: utf-8 -*-
"""
Library name handling for ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
# NO IMPORTS FROM OBSPY OR FUTURE IN THIS FILE! (file gets used at
# installation time)
import ctypes
from distutils import sysconfig
import os
import platform
import re
import warnings


def cleanse_pymodule_filename(filename):
    """
    Replace all characters not allowed in Python module names in filename with
    "_".

    See bug report:
     - http://stackoverflow.com/questions/21853678/install-obspy-in-cygwin
     - See #755

    See also:
     - http://stackoverflow.com/questions/7552311/
     - http://docs.python.org/2/reference/lexical_analysis.html#identifiers

    >>> print(cleanse_pymodule_filename("0blup-bli.554_3!32"))
    _blup_bli_554_3_32
    """
    filename = re.sub(r'^[^a-zA-Z_]', "_", filename)
    filename = re.sub(r'[^a-zA-Z0-9_]', "_", filename)
    return filename


def _get_lib_name(lib, add_extension_suffix):
    """
    Helper function to get an architecture and Python version specific library
    filename.

    :type add_extension_suffix: bool
    :param add_extension_suffix: NumPy distutils adds a suffix to
        the filename we specify to build internally (as specified by Python
        builtin `sysconfig.get_config_var("EXT_SUFFIX")`. So when loading the
        file we have to add this suffix, but not during building.
    """
    # our custom defined part of the extension file name
    libname = "lib%s_%s_%s_py%s" % (
        lib, platform.system(), platform.architecture()[0],
        ''.join([str(i) for i in platform.python_version_tuple()[:2]]))
    libname = cleanse_pymodule_filename(libname)
    # NumPy distutils adds extension suffix by itself during build (#771, #755)
    if add_extension_suffix:
        # append any extension suffix defined by Python for current platform
        ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
        # in principle "EXT_SUFFIX" is what we want.
        # "SO" seems to be deprecated on newer python
        # but: older python seems to have empty "EXT_SUFFIX", so we fall back
        if not ext_suffix:
            try:
                ext_suffix = sysconfig.get_config_var("SO")
            except Exception as e:
                msg = ("Empty 'EXT_SUFFIX' encountered while building CDLL "
                       "filename and fallback to 'SO' variable failed "
                       "(%s)." % str(e))
                warnings.warn(msg)
                pass
        if ext_suffix:
            libname = libname + ext_suffix
    return libname


def _load_CDLL(name):
    """
    Helper function to load a shared library built during ObsPy installation
    with ctypes.

    :type name: str
    :param name: Name of the library to load (e.g. 'mseed').
    :rtype: :class:`ctypes.CDLL`
    """
    # our custom defined part of the extension file name
    libname = _get_lib_name(name, add_extension_suffix=True)
    libdir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir,
                          'lib')
    libpath = os.path.join(libdir, libname)
    try:
        cdll = ctypes.CDLL(libpath)
    except Exception as e:
        msg = 'Could not load shared library "%s".\n\n %s' % (libname, str(e))
        raise ImportError(msg)
    return cdll


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
