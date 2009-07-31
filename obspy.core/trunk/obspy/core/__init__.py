# -*- coding: utf-8 -*-
#-------------------------------------------------------------------
#  Purpose: Core classes of ObsPy, Python for Seismological Observatories
#   Author: Robert Barsch
#           Moritz Beyreuther
#           Lion Krischer
#    Email: barsch@geophysik.uni-muenchen.de
#
# Copyright (C) 2008-2010 Robert Barsch, Moritz Beyreuthe, Lion Krischer
#---------------------------------------------------------------------
"""
obspy.core - Core classes of ObsPy, Python for Seismological Observatories

This class contains common methods and classes for ObsPy. It includes
a UTCDateTime, a Stats and general comment methods.


GNU General Public License (GPL)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
02110-1301, USA.
"""

# don't change order
from utcdatetime import UTCDateTime
from trace import Trace, Stats
from stream import Stream, read
from testing import runTests
