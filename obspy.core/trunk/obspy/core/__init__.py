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

This class contains common methods and classes for ObsPy. It includes Stream,
Trace, UTCDateTime, Stats classes and methods for reading seismogram files.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)

Summary
-------
Seismograms of the formats SAC, MiniSEED, GSE2, SEISAN, Q, etc. can be imported
into a :class:`~obspy.core.stream.Stream` object using the
:func:`~obspy.core.stream.read` function.

Streams are list-like objects which contain multiple
:class:`~obspy.core.trace.Trace` objects, i.e. gap-less continuous time series
and related header/meta information.

Each Trace object has the attribute ``data`` pointing to a NumPy_ ndarray of
the actual time series and the attribute ``stats`` which contains all meta
information in a dict-like :class:`~obspy.core.trace.Stats` object. Both
attributes ``starttime`` and ``endtime`` of the Stats object are
:class:`~obspy.core.utcdatetime.UTCDateTime` objects.

Example
-------

>>> st = read('gaps.mseed')
>>> print st
4 Trace(s) in Stream:
BW.BGLD..EHE | 2007-12-31T23:59:59.915000Z - ... | 200.0 Hz, 412 samples
BW.BGLD..EHE | 2008-01-01T00:00:04.035000Z - ... | 200.0 Hz, 824 samples
BW.BGLD..EHE | 2008-01-01T00:00:10.215000Z - ... | 200.0 Hz, 824 samples
BW.BGLD..EHE | 2008-01-01T00:00:18.455000Z - ... | 200.0 Hz, 50668 samples
>>> tr = st[0]
>>> print tr
BW.BGLD..EHE | 2007-12-31T23:59:59.915000Z - ... | 200.0 Hz, 412 samples
>>> tr.data
array([-363 -382 -388 -420 -417 ... -409 -393 -353 -360 -389])
>>> tr.stats
Stats({
    'network': 'BW', 
    'mseed': AttribDict({'dataquality': 'D'}), 
    'delta': 0.0050000000000000001, 
    'station': 'BGLD', 
    'location': '', 
    'starttime': UTCDateTime(2007, 12, 31, 23, 59, 59, 915000), 
    'npts': 412, 
    'calib': 1.0, 
    'sampling_rate': 200.0, 
    'endtime': UTCDateTime(2008, 1, 1, 0, 0, 1, 970000), 
    'channel': 'EHE'
})
>>> tr.stats.starttime
UTCDateTime(2007, 12, 31, 23, 59, 59, 915000)

.. _NumPy: http://docs.scipy.org
"""

# don't change order
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.util import _getVersionString, AttribDict
from obspy.core.trace import Stats, Trace
from obspy.core.stream import Stream, read
from obspy.core.scripts.runtests import runTests


__version__ = _getVersionString("obspy.core")
