# -*- coding: utf-8 -*-
"""
obspy.datamark - DATAMARK read support for ObsPy
================================================
This module provides read support for Datamark waveform data.

:copyright:
    The ObsPy Development Team (devs@obspy.org), Thomas Lecocq and others
    (refs will be included in the release version)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)

"""

from obspy.core.util import _getVersionString


__version__ = _getVersionString("obspy.datamark")


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
