# -*- coding: utf-8 -*-
"""
SeisHub database client for ObsPy.

:copyright: The ObsPy Development Team (devs@obspy.org)
:license: GNU Lesser General Public License, Version 3 (LGPLv3)

Requesting Data
---------------

>>> from from obspy.seishub import Client
>>> from obspy.core import UTCDateTime
>>> client = Client()
        data = client.waveform.getNetworkIds()
        print data
        data = client.waveform.getStationIds()
        print data
        data = client.waveform.getLocationIds()
        print data
        data = client.waveform.getChannelIds()
        print data
        data = client.waveform.getStationIds(network_id='BW')
        print data
        data = client.waveform.getChannelIds(network_id='BW',
                                             station_id='MANZ')
        print data
        data = client.waveform.getLatency(network_id='BW', station_id='MANZ')
        print data
        data = client.station.getList(network_id='BW', station_id='MANZ')
        print data
        data = client.station.getResource('dataless.seed.BW_MANZ.xml',
                                          format='metadata')
        print data
        data = client.station.getResource('dataless.seed.BW_MANZ.xml',
                                          format='metadata')
        print data
        data = client.station.getPAZ('BW', 'MANZ', UTCDateTime('20090808'))
        print data
...
XXX
"""

from obspy.seishub.client import Client
