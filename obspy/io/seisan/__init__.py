# -*- coding: utf-8 -*-
"""
obspy.io.seisan - SEISAN read support for ObsPy
===============================================

The obspy.io.seisan package contains methods in order to read seismogram
files in the SEISAN format.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)

Reading
-------
Importing SEISAN files is done similar to reading any other waveform data
format within ObsPy by using the :func:`~obspy.core.stream.read()` method of
the :mod:`obspy.core` module. Example seismogram files may be found at
https://examples.obspy.org.

>>> from obspy import read
>>> st = read("/path/to/2001-01-13-1742-24S.KONO__004")
>>> st  # doctest: +ELLIPSIS
<obspy.core.stream.Stream object at 0x...>
>>> print(st)  # doctest: +ELLIPSIS
4 Trace(s) in Stream:
.KONO.0.B0Z | 2001-01-13T17:45:01.999000Z - ... | 20.0 Hz, 6000 samples
.KONO.0.L0Z | 2001-01-13T17:42:24.924000Z - ... | 1.0 Hz, 3542 samples
.KONO.0.L0N | 2001-01-13T17:42:24.924000Z - ... | 1.0 Hz, 3542 samples
.KONO.0.L0E | 2001-01-13T17:42:24.924000Z - ... | 1.0 Hz, 3542 samples

The file format will be determined automatically. Each trace (multiple channels
are mapped to multiple traces) will have a stats attribute containing the usual
information.

>>> print(st[0].stats)  # doctest: +NORMALIZE_WHITESPACE
             network:
             station: KONO
            location: 0
             channel: B0Z
           starttime: 2001-01-13T17:45:01.999000Z
             endtime: 2001-01-13T17:50:01.949000Z
       sampling_rate: 20.0
               delta: 0.05
                npts: 6000
               calib: 1.0
             _format: SEISAN

The actual data is stored as numpy.ndarray in the data attribute of each trace.

>>> print(st[0].data)
[  464   492   519 ..., -7042 -6960 -6858]
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
