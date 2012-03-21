# -*- coding: utf-8 -*-
"""
obspy.seedlink - SeedLink client for ObsPy
==========================================

The obspy.seedlink development has been supported by the NERA project (Network
of European Research Infrastructures for Earthquake Risk Assessment and
Mitigation) under the European Community's Seventh Framework Programme
[FP7/2007-2013] grant agreement n° 262330.

:copyright:
    The ObsPy Development Team (devs@obspy.org), Anthony Lomax &
    Alberto Michelini
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from obspy.core.util.base import _getVersionString


_version__ = _getVersionString("obspy.seedlink")


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
