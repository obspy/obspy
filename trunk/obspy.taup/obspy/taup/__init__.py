# -*- coding: utf-8 -*-
"""
obspy.taup - Travel time calculation tool
=========================================

This module contains Python wrappers for iaspei-tau - a travel time library
of Arthur Snoke (http://www.iris.edu/software/downloads/processing/).
The library iaspei-tau is written in Fortran and interfaced via Python ctypes.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    Unknown
"""

from obspy.core.util import _getVersionString


__version__ = _getVersionString("obspy.taup")


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
