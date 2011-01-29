# -*- coding: utf-8 -*-
"""
obspy.iris - IRIS web service request client for ObsPy
========================================================

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from obspy.core.util import _getVersionString
from client import Client


__version__ = _getVersionString("obspy.iris")
