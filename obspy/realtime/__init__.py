# -*- coding: utf-8 -*-
"""
obspy.realtime - Real time support for ObsPy
============================================

The obspy.realtime package extends the ObsPy core classes with real time
functionality.

:copyright:
    The ObsPy Development Team (devs@obspy.org), Anthony Lomax & Alessia Maggi
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)

The obspy.realtime development has been supported by the NERA project ["Network
of European Research Infrastructures for Earthquake Risk Assessment and
Mitigation" under the European Community's Seventh Framework Programme
(FP7/2007-2013) grant agreement n° 262330] and implemented within the
activities of the JRA2/WP12 "Tools for real-time seismology, acquisition and
mining".
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from obspy.realtime.rtmemory import RtMemory
from obspy.realtime.rttrace import RtTrace


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
