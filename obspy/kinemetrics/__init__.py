# -*- coding: utf-8 -*-
"""
obspy.kinemetrics - EVT format support for ObsPy
================================================

Evt read support for ObsPy.

This module provides read support for the EVT Kinemetrics data format.
It is based on the Kinemetrics description of the format and the provided
C code (Kw2asc.c (see "KW2ASC.SRC" File in /doc section)).

:copyright:
    The ObsPy Development Team (devs@obspy.org), Henri Martin, Thomas Lecocq,
    Kinemetrics(c)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)

Reading
-------
Similar to reading any other waveform data format using obspy.core:

>>> from obspy import read
>>> st = read("/path/to/test.evt")
>>> st
<obspy.core.stream.Stream at 0x....>
>>> print (st)
3 Trace(s) in Stream:
.MEMA..0 | 2013-08-15T09:20:28.000000Z - 2013-08-15T09:20:50.996000Z | 250.0 Hz,
 5750 samples
.MEMA..1 | 2013-08-15T09:20:28.000000Z - 2013-08-15T09:20:50.996000Z | 250.0 Hz,
 5750 samples
.MEMA..2 | 2013-08-15T09:20:28.000000Z - 2013-08-15T09:20:50.996000Z | 250.0 Hz,
 5750 samples

Each trace will have a ``stats`` attribute containing the usual information and
a ``kinemetrics_evt`` dictionary with specific attributes.
Note : All the Header's attributes are not read (can be implemented if necessary for someone)

>>> stats_evt = st[0].stats.pop('kinemetrics_evt')
>>> print(st[0].stats)
         network:
         station: MEMA
        location:
         channel: 0
       starttime: 2013-08-15T09:20:28.000000Z
         endtime: 2013-08-15T09:20:50.996000Z
   sampling_rate: 250.0
           delta: 0.004
            npts: 5750
           calib: 1.0

>>> for k, v in stats_evt.items():
        print(k, v)
comment MEMBACH PARAMETERS FAC+EEP/v3.02
chan_azimuth 0
chan_north 0
serialnumber 4823
chan_fullscale 2.5
chan_sensogain 1
installedchan 4
chan_id
duration 230
chan_range 3
chan_east 0
batteryvoltage -134
chan_sensitivity 2.5
temperature 76
instrument New Etna
latitude 50.6097946167
startime 2013-08-15T09:20:28.000000Z
chan_natfreq 196.0
chan_gain 1
elevation 298
nchannels 3
maxchannels 12
gpslastlock 2013-08-15T09:19:20.000000Z
gpsstatus Present ON
a2dbits 24
stnid MEMA
chan_damping 0.707000017166
longitude 6.00925016403
samplebytes 3
chan_up 0
triggertime 2013-08-15T09:20:34.600000Z
chan_calcoil 0.0500000007451
nscans 6

The actual data is stored as :class:`numpy.ndarray` in the ``data`` attribute
of each trace.
>>> type(st[0].data)
numpy.ndarray
>>> print(st[0].data)
[-0.02446475 -0.02453492 -0.02446709 ..., -0.02452556 -0.02450685
 -0.02442499]

Writing
-------
Not implemented


"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
