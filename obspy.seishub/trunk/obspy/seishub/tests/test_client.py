#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The obspy.seishub.client test suite.
"""

from obspy.seishub import Client
import unittest
from obspy.core import UTCDateTime


class ClientTestCase(unittest.TestCase):
    """
    Test cases for L{obspy.seishub.client.Client}.
    """

    def test_getStations(self):
        """
        """
        client = Client("http://localhost:8080")
        data = client.waveform.getNetworkIds()
        print data
        data = client.waveform.getStationIds()
        print data
        data = client.waveform.getLocationIds()
        print data
        data = client.waveform.getChannelIds()
        print data
        t = UTCDateTime('20100102')
        st = client.waveform.getWaveform("BW", "HROE", "", "EHN", t, t + 1800)
        print st
        st = client.waveform.getPreview("BW", "HROE", "", "EHN", t, t + 1800)
        print st
        data = client.waveform.getStationIds(network_id='BW')
        print data
        data = client.waveform.getChannelIds(network_id='BW',
                                             station_id='MANZ')
        print data
        data = client.station.getList(network_id='BW', station_id='MANZ')
        print data
        data = client.waveform.getLatency(network_id='BW', station_id='HROE')
        print data
        data = client.station.getResource('dataless.seed.BW_MANZ.xml',
                                          format='metadata')
        print data
        data = client.station.getResource('dataless.seed.BW_MANZ.xml',
                                          format='metadata')
        print data
        data = client.station.getPAZ('BW', 'MANZ', UTCDateTime('20090808'))
        print data


def suite():
    return unittest.makeSuite(ClientTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
