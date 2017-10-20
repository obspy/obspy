# -*- coding: utf-8 -*-
"""
obspy.clients.seishub - SeisHub database client for ObsPy
=========================================================
The obspy.clients.seishub package contains a client for the seismological
database SeisHub (http://www.seishub.org).

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)

Basic Example
-------------

>>> from obspy.clients.seishub import Client
>>> from obspy import UTCDateTime

>>> client = Client(timeout=20)
>>> t = UTCDateTime('2010-01-01T10:00:00')
>>> st = client.waveform.get_waveforms(
...     "BW", "MANZ", "", "EH*", t, t+20)  # doctest: +SKIP
>>> st.sort()  # doctest: +ELLIPSIS +SKIP
<obspy.core.stream.Stream object at ...>
>>> print(st)  # doctest: +ELLIPSIS +SKIP
3 Trace(s) in Stream:
BW.MANZ..EHE | 2010-01-01T10:00:00.000000Z - ... | 200.0 Hz, 4001 samples
BW.MANZ..EHN | 2010-01-01T10:00:00.000000Z - ... | 200.0 Hz, 4001 samples
BW.MANZ..EHZ | 2010-01-01T10:00:00.000000Z - ... | 200.0 Hz, 4001 samples

Advanced Examples
-----------------

>>> client.waveform.get_network_ids()  #doctest: +SKIP
['KT', 'BW', 'NZ', 'GR', ...]

>>> sta_ids = client.waveform.get_station_ids(network='BW')  # doctest: +SKIP
>>> sorted(sta_ids)  # doctest: +SKIP
['ALTM', 'BGLD', 'BW01',..., 'WETR', 'ZUGS']

>>> cha_ids = client.waveform.get_channel_ids(
...     network='BW', station='MANZ')  # doctest: +SKIP
>>> sorted(cha_ids)  # doctest: +NORMALIZE_WHITESPACE +SKIP
['AEX', 'AEY', 'BHE', 'BHN', 'BHZ', 'E', 'EHE', 'EHN', 'EHZ', 'HHE', 'HHN',
 'HHZ', 'LOG', 'N', 'SHE', 'SHN', 'SHZ', 'Z']

>>> paz = client.station.get_paz(
...     'BW.MANZ..EHZ', UTCDateTime('20090808'))  # doctest: +SKIP
>>> paz = paz.items()  # doctest: +SKIP
>>> sorted(paz)  # doctest: +SKIP
[('gain', 60077000.0),
 ('poles', [(-0.037004+0.037016j), (-0.037004-0.037016j), (-251.33+0j),
            (-131.04-467.29j), (-131.04+467.29j)]),
 ('sensitivity', 2516800000.0),
 ('zeros', [0j, 0j])]
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from .client import Client


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
