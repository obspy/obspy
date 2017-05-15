# -*- coding: utf-8 -*-
"""
obspy.db - A seismic waveform indexer and database for ObsPy
============================================================
The obspy.db package contains a waveform indexer collecting metadata from a
file based waveform archive and storing in into a standard SQL database.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

__all__ = []

try:
    import sqlalchemy  # NOQA
except ImportError:
    msg = ("The 'sqlalchemy' module needs to be installed in order to use the "
           "'obspy.db' module.")
    raise ImportError(msg)
