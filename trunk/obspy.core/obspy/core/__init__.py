# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
#  Purpose: Core classes of ObsPy: Python for Seismological Observatories
#   Author: Robert Barsch
#           Moritz Beyreuther
#           Lion Krischer
#           Tobias Megies
#
# Copyright (C) 2008-2011 Robert Barsch, Moritz Beyreuther, Lion Krischer,
#                         Tobias Megies
#------------------------------------------------------------------------------
"""
obspy.core - Core classes of ObsPy
==================================

This package contains common methods and classes for ObsPy. It includes Stream,
Trace, UTCDateTime, Stats classes and methods for reading seismogram files.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)

Summary
-------
Seismograms of various formats (e.g. SAC, MiniSEED, GSE2, SEISAN, Q, etc.) can
be imported into a :class:`~obspy.core.stream.Stream` object using the
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
A :class:`~obspy.core.stream.Stream` with an example seismogram can be created
by calling :func:`~obspy.core.stream.read()` without any arguments.
Local files can be read by specifying the filename, files stored on http
servers (e.g. at http://examples.obspy.org) can be read by specifying their
URL. For details see the documentation of :func:`~obspy.core.stream.read`.

>>> from obspy.core import read
>>> st = read()
>>> print(st)  # doctest: +ELLIPSIS
3 Trace(s) in Stream:
BW.RJOB..EHZ | 2009-08-24T00:20:03.000000Z - ... | 100.0 Hz, 3000 samples
BW.RJOB..EHN | 2009-08-24T00:20:03.000000Z - ... | 100.0 Hz, 3000 samples
BW.RJOB..EHE | 2009-08-24T00:20:03.000000Z - ... | 100.0 Hz, 3000 samples
>>> tr = st[0]
>>> print(tr)  # doctest: +ELLIPSIS
BW.RJOB..EHZ | 2009-08-24T00:20:03.000000Z - ... | 100.0 Hz, 3000 samples
>>> tr.data
array([ 0.        ,  0.00694644,  0.07597424, ...,  1.93449584,
        0.98196204,  0.44196924])
>>> tr.stats  # doctest: +NORMALIZE_WHITESPACE
Stats({'network': 'BW', 'delta': 0.01, 'station': 'RJOB', 'location': '',
       'starttime': UTCDateTime(2009, 8, 24, 0, 20, 3), 'npts': 3000,
       'calib': 1.0, 'sampling_rate': 100.0, 'channel': 'EHZ'})
>>> tr.stats.starttime
UTCDateTime(2009, 8, 24, 0, 20, 3)

.. _NumPy: http://docs.scipy.org
"""

# don't change order
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.util import _getVersionString, AttribDict
from obspy.core.trace import Stats, Trace
from obspy.core.stream import Stream, read
from obspy.core.scripts.runtests import runTests


__version__ = _getVersionString("obspy.core")


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
