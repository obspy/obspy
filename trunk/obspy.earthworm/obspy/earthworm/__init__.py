# -*- coding: utf-8 -*-
"""
obspy.earthworm - Earthworm Wave Server client for ObsPy.
=========================================================

:copyright:
    The ObsPy Development Team (devs@obspy.org) & Victor Kress
:license:
    GNU General Public License (GPLv2)
    (http://www.gnu.org/licenses/old-licenses/gpl-2.0.html)
"""

from obspy.core.util import _getVersionString
from client import Client


__version__ = _getVersionString("obspy.earthworm")


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
