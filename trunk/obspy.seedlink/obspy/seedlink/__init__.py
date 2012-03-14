# -*- coding: utf-8 -*-
"""
obspy.seedlink - SeedLink client for ObsPy
==========================================

:copyright:
    The ObsPy Development Team (devs@obspy.org) & Anthony Lomax
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from obspy.core.util.base import _getVersionString


_version__ = _getVersionString("obspy.seedlink")


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
