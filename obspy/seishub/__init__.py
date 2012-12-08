# -*- coding: utf-8 -*-
"""
obspy.seishub - SeisHub database client for ObsPy
=================================================
The obspy.seishub package contains a client for the seismological database
SeisHub (http://www.seishub.org).

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)

Basic Example
-------------

>>> from obspy.seishub import Client
>>> from obspy import UTCDateTime

>>> client = Client(timeout=20)
>>> t = UTCDateTime('2010-01-01T10:00:00')
>>> st = client.waveform.getWaveform("BW", "MANZ", "", "EH*", t, t+20)
>>> st.sort()
>>> print(st)  # doctest: +ELLIPSIS
3 Trace(s) in Stream:
BW.MANZ..EHE | 2010-01-01T10:00:00.000000Z - ... | 200.0 Hz, 4001 samples
BW.MANZ..EHN | 2010-01-01T10:00:00.000000Z - ... | 200.0 Hz, 4001 samples
BW.MANZ..EHZ | 2010-01-01T10:00:00.000000Z - ... | 200.0 Hz, 4001 samples

Advanced Examples
-----------------

>>> client.waveform.getNetworkIds()     #doctest: +SKIP
['KT', 'BW', 'NZ', 'GR', ...]

>>> sta_ids = client.waveform.getStationIds(network='BW')
>>> sorted(sta_ids)  # doctest: +SKIP
['ALTM', 'BGLD', 'BW01',..., 'WETR', 'ZUGS']

>>> cha_ids = client.waveform.getChannelIds(network='BW', station='MANZ')
>>> sorted(cha_ids)
['AEX', 'AEY', 'EHE', 'EHN', 'EHZ', 'LOG', 'SHE', 'SHN', 'SHZ']

>>> res = client.station.getResource('dataless.seed.BW_MANZ.xml',
...                                  format='metadata')
>>> print(res)  # doctest: +NORMALIZE_WHITESPACE
<?xml version="1.0" encoding="utf-8"?>
<metadata>
  <item title="Station Name">
    <text text="Manzenberg,Bavaria, BW-Net"/>
  </item>
  <item title="Station ID">
    <text text="MANZ"/>
  </item>
  <item title="Network ID">
    <text text="BW"/>
  </item>
  <item title="Channel IDs">
    <text text="EHZ"/>
    <text text="EHN"/>
    <text text="EHE"/>
  </item>
  <item title="Latitude (°)">
    <text text="+49.986198"/>
  </item>
  <item title="Longitude (°)">
    <text text="+12.108300"/>
  </item>
  <item title="Elevation (m)">
    <text text="+635.0"/>
  </item>
</metadata>

>>> paz = client.station.getPAZ('BW.MANZ..EHZ', UTCDateTime('20090808'))
>>> paz = paz.items()
>>> sorted(paz)  # doctest: +SKIP
[('gain', 60077000.0),
 ('poles', [(-0.037004+0.037016j), (-0.037004-0.037016j), (-251.33+0j),
            (-131.04-467.29j), (-131.04+467.29j)]),
 ('sensitivity', 2516800000.0),
 ('zeros', [0j, 0j])]
"""

from obspy.seishub.client import Client


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
