# -*- coding: utf-8 -*-
"""
obspy.io.pdas - PDAS file read support for ObsPy
================================================

The obspy.io.pdas package contains methods in order to read files in the PDAS
file format.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)

"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport

from .core import _is_pdas, _read_pdas


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
