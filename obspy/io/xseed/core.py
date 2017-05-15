# -*- coding: utf-8 -*-
"""
Integration with ObsPy's core classes.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport

from .parser import is_xseed


def _is_xseed(filename):
    """
    Determine if the file is an XML-SEED file.

    Does not do any schema validation but only check the root tag.
    """
    return is_xseed(filename)


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
