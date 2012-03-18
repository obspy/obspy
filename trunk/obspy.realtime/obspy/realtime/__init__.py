# -*- coding: utf-8 -*-
"""
obspy.realtime - Real time support for ObsPy
============================================

The obspy.realtime package extends the ObsPy core classes with real time
functionalities.

:copyright:
    The ObsPy Development Team (devs@obspy.org) & Anthony Lomax
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from obspy.core.util.base import _getVersionString
from obspy.realtime.rtmemory import RtMemory
from obspy.realtime.rttrace import RtTrace, splitTrace


_version__ = _getVersionString("obspy.realtime")


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
