# -*- coding: utf-8 -*-
"""
obspy.segy - SEG Y rev 1 read and write support
=============================================
This module provides read and write support for SEG Y rev 1 formated data.

:copyright: The ObsPy Development Team (devs@obspy.org)
:license: GNU General Public License (GPLv2)
"""

from obspy.core.util import _getVersionString

__version__ = _getVersionString("obspy.segy")

from segy import read as read
