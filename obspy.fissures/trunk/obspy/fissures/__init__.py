# -*- coding: utf-8 -*-
"""
obspy.fissres - DHI/Fissures Request Client
===========================================
See: http://www.iris.edu/dhi/

:copyright: The ObsPy Development Team (devs@obspy.org)
:license: GNU Lesser General Public License, Version 3 (LGPLv3)

Example
-------
As obspy.fissures is still under development here a simple example:

>>> from obspy.core import UTCDateTime, read
>>> from obspy.fissures import Client

>>> client = Client()
>>> t = UTCDateTime("2003-06-20T05:59:00.0000")
>>> st = client.getWaveform("GE", "APE", "", "SHZ", t, t + 10)
>>> print st
1 Trace(s) in Stream:
GE.APE..SHZ | 2003-06-20T05:57:43.321000Z - 2003-06-20T06:00:34.481000Z | 50.0 Hz, 8559 samples
"""

from client import Client
