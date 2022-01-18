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
import importlib.machinery
from pathlib import Path
import re


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


def _load_cdll(name):
    """
    Helper function to load a shared library built during ObsPy installation
    with ctypes.

    :type name: str
    :param name: Name of the library to load (e.g. 'mseed').
    :rtype: :class:`ctypes.CDLL`
    """
    errors = []
    libdir = Path(__file__).parent.parent.parent / 'lib'
    for ext in importlib.machinery.EXTENSION_SUFFIXES:
        libpath = (libdir / (name + ext)).resolve()
        try:
            cdll = ctypes.CDLL(str(libpath))
        except Exception as e:
            errors.append(f'    {str(e)}')
        else:
            return cdll
    # If we got here, then none of the attempted extensions worked.
    raise ImportError('\n  '.join([
        f'Could not load shared library "{name}"',
        *errors,
        'Current directory: %s' % Path().resolve(),
        'Directory listing of lib directory:',
        *(f'    {str(d)}' for d in sorted(libpath.parent.iterdir())),
    ]))


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
