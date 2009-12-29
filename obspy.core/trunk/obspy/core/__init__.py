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

:copyright: The ObsPy Development Team (devs@obspy.org)
:license: GNU Lesser General Public License, Version 3 (LGPLv3)
"""

# don't change order
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.trace import Stats, Trace
from obspy.core.stream import Stream, read
from obspy.core.testing import runTests
