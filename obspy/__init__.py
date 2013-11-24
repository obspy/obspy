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
from obspy.core.util import _getVersionString
__version__ = _getVersionString()
from obspy.core.trace import Trace
from obspy.core.stream import Stream, read
from obspy.core.event import readEvents, Catalog
from obspy.station import read_inventory

# insert supported read/write format plugin lists dynamically in docstrings
from obspy.core.util.base import make_format_plugin_table
read.__doc__ = \
    read.__doc__ % make_format_plugin_table("waveform", "read", numspaces=4)
Stream.write.im_func.func_doc = \
    Stream.write.__doc__ % make_format_plugin_table("waveform", "write",
                                                    numspaces=8)
readEvents.__doc__ = \
    readEvents.__doc__ % make_format_plugin_table("event", "read", numspaces=4)
Catalog.write.im_func.func_doc = \
    Catalog.write.__doc__ % make_format_plugin_table("event", "write",
                                                     numspaces=8)


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
