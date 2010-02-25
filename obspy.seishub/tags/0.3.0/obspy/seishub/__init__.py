# -*- coding: utf-8 -*-
"""
obspy.seishub - SeisHub database client for ObsPy
=================================================

:copyright: The ObsPy Development Team (devs@obspy.org)
:license: GNU Lesser General Public License, Version 3 (LGPLv3)

Basic Example
-------------

>>> from obspy.seishub import Client
>>> from obspy.core import UTCDateTime

>>> client = Client()
>>> t = UTCDateTime('20090808')
>>> st = client.waveform.getWaveform("BW", "MANZ", "", "EH*", t,
>>> print st
3 Trace(s) in Stream:
    BW.MANZ..EHZ | 2009-08-08T00:00:00.000000Z - 2009-08-08T00:30:00.000000Z | 200.0 Hz, 360001 samples
    BW.MANZ..EHN | 2009-08-08T00:00:00.000000Z - 2009-08-08T00:30:00.000000Z | 200.0 Hz, 360001 samples
    BW.MANZ..EHE | 2009-08-08T00:00:00.000000Z - 2009-08-08T00:30:00.000000Z | 200.0 Hz, 360001 samples

Advanced Examples
-----------------

>>> from obspy.seishub import Client
>>> client = Client()

>>> client.waveform.getNetworkIds()
['BW', 'CZ', 'GR', 'NZ', '']

>>> client.waveform.getStationIds(network_id='BW')
['BGLD', 'BW01', 'DHFO', 'FURT', 'HROE', 'MANZ', 'MASC', 'MGBB', 'MHAI',
'MKON', 'MROB', 'MSBB', 'MZEK', 'NORI', 'NZC2', 'NZG0', 'OBER', 'OBHA',
'PART', 'RJOB', 'RLAS', 'RMOA', 'RNHA', 'RNON', 'ROTZ', 'RTBE', 'RTBM',
'RTSH', 'RTVS', 'RWMO', 'UH1', 'UH2', 'VIEL', 'WETR', 'ZUGS']
>>> client.waveform.getChannelIds(network_id='BW',
                                  station_id='MANZ')
['AEX', 'AEY', 'EHE', 'EHN', 'EHZ', 'SHE', 'SHN', 'SHZ']

>>> client.station.getResource('dataless.seed.BW_MANZ.xml', 
                               format='metadata')
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

>>> client.station.getPAZ('BW', 'MANZ', UTCDateTime('20090808'))
{'zeros': [0j, 0j], 
 'sensitivity': 2516800000.0, 
 'poles': [(-0.037004000000000002+0.037016j), (-0.037004000000000002-0.037016j),
        (-251.33000000000001+0j), (-131.03999999999999-467.29000000000002j),
        (-131.03999999999999+467.29000000000002j)], 
 'gain': 60077000.0}
"""

from obspy.core.util import _getVersionString
from obspy.seishub.client import Client


__version__ = _getVersionString("obspy.seishub")
