# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
#  Purpose: Convenience imports for obspy
#   Author: Robert Barsch
#           Moritz Beyreuther
#           Lion Krischer
#           Tobias Megies
#
# Copyright (C) 2008-2012 Robert Barsch, Moritz Beyreuther, Lion Krischer,
#                         Tobias Megies
#------------------------------------------------------------------------------
"""
ObsPy: A Python Toolbox for seismology/seismological observatories
==================================================================

ObsPy is an open-source project dedicated to provide a Python framework for
processing seismological data. It provides parsers for common file formats
and seismological signal processing routines which allow the manipulation of
seismological time series.

The goal of the ObsPy project is to facilitate rapid application development
for seismology.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

# don't change order
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.util.base import _getVersionString
from obspy.core.trace import Trace
from obspy.core.stream import Stream, read
from obspy.core.event import readEvents


__version__ = _getVersionString()


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
