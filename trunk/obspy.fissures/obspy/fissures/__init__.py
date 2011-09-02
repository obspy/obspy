# -*- coding: utf-8 -*-
"""
obspy.fissures - DHI/Fissures Request Client
===========================================
See: http://www.iris.edu/dhi/

:copyright: The ObsPy Development Team (devs@obspy.org)
:license: GNU Lesser General Public License, Version 3 (LGPLv3)

.. rubric:: Examples

As obspy.fissures is still under development here a simple example which
request the SHZ channel of station APE in network GE at time t. For the
example only 10s are requested.

>>> from obspy.core import UTCDateTime, read
>>> from obspy.fissures import Client

>>> client = Client()
>>> t = UTCDateTime("2003-06-20T05:59:00.0000")
>>> st = client.getWaveform("GE", "APE", "", "SHZ", t, t + 10)
>>> print(st)  # doctest: +ELLIPSIS
1 Trace(s) in Stream:
GE.APE..SHZ | 2003-06-20T05:59:00.001000Z - ... | 50.0 Hz, 500 samples
"""

from obspy.core.util import _getVersionString
from client import Client


__version__ = _getVersionString("obspy.fissures")
