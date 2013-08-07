# -*- coding: utf-8 -*-
"""
obspy.core.util.geodetics - Various geodetic utilities for ObsPy.
=================================================================

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from obspy.core.util.geodetics.base import calcVincentyInverse, \
        gps2DistAzimuth, kilometer2degrees, locations2degrees
from obspy.core.util.geodetics.flinnengdahl import FlinnEngdahl


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
