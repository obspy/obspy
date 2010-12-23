# -*- coding: utf-8 -*-
"""
obspy.seisan - SEISAN read support
=================================
This module provides read support for SEISAN waveform files.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)

Reading
-------
Importing SEISAN files is done similar to reading any other waveform data
format within ObsPy by using the :func:`~obspy.core.stream.read()` method of
the :mod:`obspy.core` module. Test files for the following examples may be
found at http://examples.obspy.org.

(Lines 2&3 are just to get the absolute path of our test data)

>>> from obspy.core import read
>>> from obspy.core import path
>>> filename = path("2001-01-13-1742-24S.KONO__004")
>>> st = read(filename)
>>> st #doctest: +ELLIPSIS
<obspy.core.stream.Stream object at 0x...>
>>> print st
4 Trace(s) in Stream:
.KONO.0.B0Z | 2001-01-13T17:45:01.999000Z - 2001-01-13T17:50:01.949000Z | 20.0 Hz, 6000 samples
.KONO.0.L0Z | 2001-01-13T17:42:24.924000Z - 2001-01-13T18:41:25.924000Z | 1.0 Hz, 3542 samples
.KONO.0.L0N | 2001-01-13T17:42:24.924000Z - 2001-01-13T18:41:25.924000Z | 1.0 Hz, 3542 samples
.KONO.0.L0E | 2001-01-13T17:42:24.924000Z - 2001-01-13T18:41:25.924000Z | 1.0 Hz, 3542 samples

The file format will be determined automatically. Each trace (multiple channels
are mapped to multiple traces) will have a stats attribute containing the usual
information.

>>> st[0].stats #doctest: +NORMALIZE_WHITESPACE
Stats({'network': '', '_format': 'SEISAN', 'npts': 6000, 'station': 'KONO',
       'location': '0',
       'starttime': UTCDateTime(2001, 1, 13, 17, 45, 1, 999000),
       'delta': 0.05, 'calib': 1.0, 'sampling_rate': 20.0, 'channel': 'B0Z'})

The actual data is stored as numpy.ndarray in the data attribute of each trace.

>>> print st[0].data
[  492   519   542 ..., -6960 -6858 24000]
"""

from obspy.core.util import _getVersionString


__version__ = _getVersionString("obspy.seisan")
