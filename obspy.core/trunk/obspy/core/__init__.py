# -*- coding: utf-8 -*-
#-------------------------------------------------------------------
#  Purpose: Core classes of ObsPy: Python for Seismological Observatories
#   Author: Robert Barsch
#           Moritz Beyreuther
#           Lion Krischer
#    Email: barsch@lmu.de
#
# Copyright (C) 2008-2010 Robert Barsch, Moritz Beyreuther, Lion Krischer
#---------------------------------------------------------------------
"""
obspy.core - Core classes of ObsPy, Python for Seismological Observatories

This class contains common methods and classes for ObsPy. It includes
UTCDateTime, Stats, Stream and Trace classes and methods for reading 
seismograms.

@copyright: The ObsPy Development Team (devs@obspy.org)
@license: GNU Lesser General Public License, Version 3 (LGPLv3)
"""

# don't change order
from utcdatetime import UTCDateTime
from trace import Trace, Stats
from stream import Stream, read
from testing import runTests
