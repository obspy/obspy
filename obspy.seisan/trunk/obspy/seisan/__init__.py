# -*- coding: utf-8 -*-
"""
obspy.seisan - SEIAN read support
=================================
 
:copyright: The ObsPy Development Team (devs@obspy.org)
:license: GNU Lesser General Public License, Version 3 (LGPLv3)


Similiar to reading any other waveform data format using obspy.core:

>>> from obspy.core import read
>>> st = read("tests/data/2001-01-13-1742-24S.KONO__004")
>>> print st
4 Trace(s) in Stream:
.KONO.0.B0Z | 2001-01-13T17:45:01.999000Z - 2001-01-13T17:50:01.949000Z | 20.0 Hz, 6000 samples
.KONO.0.L0Z | 2001-01-13T17:42:24.924000Z - 2001-01-13T18:41:25.924000Z | 1.0 Hz, 3542 samples
.KONO.0.L0N | 2001-01-13T17:42:24.924000Z - 2001-01-13T18:41:25.924000Z | 1.0 Hz, 3542 samples
.KONO.0.L0E | 2001-01-13T17:42:24.924000Z - 2001-01-13T18:41:25.924000Z | 1.0 Hz, 3542 samples

The format will be determined automatically. Each trace (multiple channels are
mapped to multiple traces) will have a stats attribute containing the usual
information.

>>> print st[0].stats
Stats({'network': '',
    'npts': 6000,
    'station': 'KONO',
    'location': '0',
    'starttime': UTCDateTime(2001, 1, 13, 17, 45, 1, 999000),
    'delta': 0.050000000000000003,
    'calib': 1.0, 
    'sampling_rate': 20.0,
    'endtime': UTCDateTime(2001, 1, 13, 17, 50, 1, 949000), 
    'channel': 'B0Z'
})

The data is stored in the data attribut.

>>> print st[0].data
[  492   519   542 ..., -6960 -6858 24000]
"""

