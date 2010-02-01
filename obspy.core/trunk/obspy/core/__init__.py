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
obspy.core - Core classes of ObsPy
==================================

This class contains common methods and classes for ObsPy. It includes
UTCDateTime, Stats, Stream and Trace classes and methods for reading 
seismograms.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)

Note
----
Seismograms of the format (SAC, MiniSEED, GSE2, SEISAN, Q, SH_ASC) are
read with the :func:`~obspy.core.stream.read` function into a
:class:`~obspy.core.stream.Stream` object. The Stream object is an
list like objects which contains several :class:`~obspy.core.stream.Trace` 
objects, i.e. gap-less continuous sample junks. Each Trace object has an
attribute data, which contains the actual data and an attribute stats which
contains the header information as a dict like 
:class:`~obspy.core.trace.Stats` object. The stats attributes starttime and
endtime must hereby be :class:`~obspy.core.utcdatetime.UTCDateTime`
objects.

Example
-------

>>> st = read('gaps.mseed')
>>> print st
>>> print st
4 Trace(s) in Stream:
BW.BGLD..EHE | 2007-12-31T23:59:59.915000Z - 2008-01-01T00:00:01.970000Z | 200.0 Hz, 412 samples
BW.BGLD..EHE | 2008-01-01T00:00:04.035000Z - 2008-01-01T00:00:08.150000Z | 200.0 Hz, 824 samples
BW.BGLD..EHE | 2008-01-01T00:00:10.215000Z - 2008-01-01T00:00:14.330000Z | 200.0 Hz, 824 samples
BW.BGLD..EHE | 2008-01-01T00:00:18.455000Z - 2008-01-01T00:04:31.790000Z | 200.0 Hz, 50668 samples
>>> tr = st[0]
>>> print tr
BW.BGLD..EHE | 2007-12-31T23:59:59.915000Z - 2008-01-01T00:00:01.970000Z | 200.0 Hz, 412 samples
>>> tr.data
array([-363 -382 -388 -420 -417 ... -409 -393 -353 -360 -389])
>>> print tr.stats
Stats({'network': 'BW', 'mseed': AttribDict({'dataquality': 'D'}), 'delta': 0.0050000000000000001, 'station': 'BGLD', 'location': '', 'starttime': UTCDateTime(2007, 12, 31, 23, 59, 59, 915000), 'npts': 412, 'calib': 1.0, 'sampling_rate': 200.0, 'endtime': UTCDateTime(2008, 1, 1, 0, 0, 1, 970000), 'channel': 'EHE'})
>>> tr.stats.starttime
UTCDateTime(2007, 12, 31, 23, 59, 59, 915000)
"""

# don't change order
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.util import _getVersionString
from obspy.core.trace import Stats, Trace
from obspy.core.stream import Stream, read
from obspy.core.scripts.runtests import runTests


__version__ = _getVersionString("obspy.core")
