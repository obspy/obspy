# -*- coding: utf-8 -*-
"""
obspy.db - A seismic waveform indexer and database for ObsPy
============================================================
The obspy.db package contains a waveform indexer collecting metadata from a
file based waveform archive and storing in into a standard SQL database.
Supported waveform formats depend on installed ObsPy packages.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from obspy.core.util import _getVersionString


__version__ = _getVersionString("obspy.db")
