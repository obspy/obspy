# -*- coding: utf-8 -*-
"""
obspy.events - Event handling for ObsPy.
========================================

Classes for handling events and picks.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from obspy.core.util import _getVersionString
from obspy.events.event import Event
from obspy.events.pick import Pick


__version__ = _getVersionString("obspy.events")
