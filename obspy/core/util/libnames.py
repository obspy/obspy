# -*- coding: utf-8 -*-
"""
Library name handling for ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import ctypes
from pathlib import Path
import re
import warnings
from distutils import sysconfig


def cleanse_pymodule_filename(filename):
    """
    Replace all characters not allowed in Python module names in filename with
    "_".

    See bug report:
     - https://stackoverflow.com/q/21853678
     - See #755

    See also:
     - https://stackoverflow.com/q/7552311
     - https://docs.python.org/3/reference/lexical_analysis.html#identifiers

    >>> print(cleanse_pymodule_filename("0blup-bli.554_3!32"))
    _blup_bli_554_3_32
    """
    filename = re.sub(r'^[^a-zA-Z_]', "_", filename)
    filename = re.sub(r'[^a-zA-Z0-9_]', "_", filename)
    return filename


def _get_lib_name(lib):
    """
    Get an architecture and Python version specific library filename.

    setuptools adds a suffix to the filename we specify to build internally (as
    specified by Python builtin `sysconfig.get_config_var("EXT_SUFFIX")`. So
    when loading the file we have to add this suffix.
    """
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
        lib += ext_suffix
    return lib


def _load_cdll(name):
    """
    Helper function to load a shared library built during ObsPy installation
    with ctypes.

    :type name: str
    :param name: Name of the library to load (e.g. 'mseed').
    :rtype: :class:`ctypes.CDLL`
    """
    # our custom defined part of the extension file name
    libname = _get_lib_name(name)
    libdir = Path(__file__).parent.parent.parent / 'lib'
    libpath = (libdir / libname).resolve()
    try:
        cdll = ctypes.CDLL(str(libpath))
    except Exception as e:
        dirlisting = sorted(libpath.parent.iterdir())
        dirlisting = '  \n'.join(map(str, dirlisting))
        msg = ['Could not load shared library "%s"' % libname,
               'Path: %s' % libpath,
               'Current directory: %s' % Path().resolve(),
               'ctypes error message: %s' % str(e),
               'Directory listing of lib directory:',
               '   %s' % dirlisting,
               ]
        msg = '\n  '.join(msg)
        raise ImportError(msg)
    return cdll


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
