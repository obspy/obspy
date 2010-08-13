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
    Test cases for the SeisHub client.
    """

    def setUp(self):
        self.client = Client("http://teide.geophysik.uni-muenchen.de:8080")

    def test_getWaveformApplyFilter(self):
        t = UTCDateTime("2009-09-03 00:00:00")
        #1 - w/o apply_filter
        st = self.client.waveform.getWaveform("BW", "RTPI", "", "EHZ",
                                              t, t + 20)
        self.assertEqual(len(st), 1)
        self.assertEqual(st[0].stats.network, '')
        self.assertEqual(st[0].stats.station, 'GP01')
        self.assertEqual(st[0].stats.location, '')
        self.assertEqual(st[0].stats.channel, 'SHZ')
        #2 - w/ apply_filter
        st = self.client.waveform.getWaveform("BW", "RTPI", "", "EHZ",
                                              t, t + 20, apply_filter=True)
        self.assertEqual(len(st), 1)
        self.assertEqual(st[0].stats.network, 'BW')
        self.assertEqual(st[0].stats.station, 'RTPI')
        self.assertEqual(st[0].stats.location, '')
        self.assertEqual(st[0].stats.channel, 'EHZ')

    def test_getEventList(self):
        c = self.client.event
        # UTCDateTimes
        events = c.getList(min_datetime=UTCDateTime("2009-01-01T00:00:00"),
                           max_datetime=UTCDateTime("2009-01-10T00:00:00"))
        self.assertEqual(len(events), 4)
        # time strings with T as separator 
        events = c.getList(min_datetime="2009-01-01T00:00:00",
                           max_datetime="2009-01-10T00:00:00")
        self.assertEqual(len(events), 4)
        # time strings with space as separator 
        events = c.getList(min_datetime="2009-01-01 00:00:00",
                           max_datetime="2009-01-10 00:00:00")
        self.assertEqual(len(events), 4)

    def test_getNetworkIds(self):
        items = ['BW', 'CZ', 'GR', 'NZ']
        data = self.client.waveform.getNetworkIds()
        for item in items:
            self.assertTrue(item in data)

    def test_ping(self):
        # current server
        time = self.client.ping()
        self.assertTrue(isinstance(time, float))

    def test_getStationIds(self):
        #1 - all stations
        items = ['ABRI', 'BGLD', 'BW01', 'CRLZ', 'DHFO', 'FUR', 'FURT', 'GRC1',
                 'HROE', 'MANZ', 'MASC', 'MGBB', 'MKON', 'MROB', 'MSBB',
                 'MZEK', 'NKC', 'NORI', 'NZG0', 'OBER', 'PART', 'RJOB', 'RLAS',
                 'RMOA', 'RNHA', 'RNON', 'ROTZ', 'RTAK', 'RTBE', 'RTEA',
                 'RTFA', 'RTKA', 'RTLI', 'RTPA', 'RTPI', 'RTSA', 'RTSH',
                 'RTSL', 'RTSP', 'RTSW', 'RTZA', 'RWMO', 'UH1', 'UH2', 'VIEL',
                  'WET', 'WETR', 'ZUGS']
        data = self.client.waveform.getStationIds()
        for item in items:
            self.assertTrue(item in data)
        #2 - all stations of network BW
        items = ['ABRI', 'BGLD', 'BW01', 'DHFO', 'FURT', 'HROE', 'MANZ',
                 'MASC', 'MGBB', 'MKON', 'MROB', 'MSBB', 'MZEK', 'NORI',
                 'NZG0', 'OBER', 'PART', 'RJOB', 'RLAS', 'RMOA', 'RNHA',
                 'RNON', 'ROTZ', 'RTAK', 'RTBE', 'RTEA', 'RTFA', 'RTKA',
                 'RTLI', 'RTPA', 'RTPI', 'RTSA', 'RTSH', 'RTSL', 'RTSP',
                 'RTSW', 'RTZA', 'RWMO', 'UH1', 'UH2', 'VIEL', 'WETR', 'ZUGS']
        data = self.client.waveform.getStationIds(network_id='BW')
        for item in items:
            self.assertTrue(item in data)

    def test_getLocationIds(self):
        #1 - all locations
        items = ['', '10']
        data = self.client.waveform.getLocationIds()
        for item in items:
            self.assertTrue(item in data)
        #2 - all locations for network BW
        items = ['']
        data = self.client.waveform.getLocationIds(network_id='BW')
        for item in items:
            self.assertTrue(item in data)
        #3 - all locations for network BW and station MANZ
        items = ['']
        data = self.client.waveform.getLocationIds(network_id='BW',
                                                   station_id='MANZ')
        for item in items:
            self.assertTrue(item in data)

    def test_getChannelIds(self):
        #1 - all channels
        items = ['AEX', 'AEY', 'BAN', 'BAZ', 'BHE', 'BHN', 'BHZ', 'EHE', 'EHN',
                 'EHZ', 'HHE', 'HHN', 'HHZ', 'LHE', 'LHN', 'LHZ', 'SHE', 'SHN',
                 'SHZ']
        data = self.client.waveform.getChannelIds()
        for item in items:
            self.assertTrue(item in data)
        #2 - all channels for network BW
        items = ['AEX', 'AEY', 'BAN', 'BAZ', 'BHE', 'BHN', 'BHZ', 'EHE', 'EHN',
                 'EHZ', 'HHE', 'HHN', 'HHZ', 'SHE', 'SHN', 'SHZ']
        data = self.client.waveform.getChannelIds(network_id='BW')
        for item in items:
            self.assertTrue(item in data)
        #3 - all channels for network BW and station MANZ
        items = ['AEX', 'AEY', 'EHE', 'EHN', 'EHZ', 'SHE', 'SHN', 'SHZ']
        data = self.client.waveform.getChannelIds(network_id='BW',
                                                  station_id='MANZ')
        for item in items:
            self.assertTrue(item in data)
        #4 - all channels for network BW, station MANZ and given location
        items = ['AEX', 'AEY', 'EHE', 'EHN', 'EHZ', 'SHE', 'SHN', 'SHZ']
        data = self.client.waveform.getChannelIds(network_id='BW',
                                                  station_id='MANZ',
                                                  location='')
        for item in items:
            self.assertTrue(item in data)

    def test_getPreview(self):
        # multiple channels / MiniSEED
        t1 = UTCDateTime('20080101')
        t2 = UTCDateTime('20080201')
        st = self.client.waveform.getPreview("BW", "*", "", "EHZ", t1, t2)
        self.assertEqual(len(st), 18)
        self.assertEqual(st[0].stats.network, 'BW')
        self.assertEqual(st[0].stats.channel, 'EHZ')
        self.assertEqual(st[0].stats.delta, 30.0)
        # single channel / GSE2
        t1 = UTCDateTime('20070101')
        t2 = UTCDateTime('20100101')
        st = self.client.waveform.getPreview("BW", "RTLI", "", "EHN", t1, t2)
        self.assertEqual(len(st), 1)
        self.assertEqual(st[0].id, 'BW.RTLI..EHN')
        self.assertEqual(st[0].stats.delta, 30.0)
        self.assertEqual(len(st[0]), 365679)
        self.assertEqual(st[0].stats.npts, 365679)

    def test_getPreviewByIds(self):
        # multiple channels / MiniSEED
        t1 = UTCDateTime('20080101')
        t2 = UTCDateTime('20080201')
        # via list
        st = self.client.waveform.getPreviewByIds(['BW.MANZ..EHE',
                                                   'BW.ROTZ..EHE'], t1, t2)
        st.sort()
        self.assertEqual(len(st), 2)
        self.assertEqual(st[0].id, 'BW.MANZ..EHE')
        self.assertEqual(st[1].id, 'BW.ROTZ..EHE')
        # via string
        st = self.client.waveform.getPreviewByIds('BW.MANZ..EHE,BW.ROTZ..EHE',
                                                  t1, t2)
        st.sort()
        self.assertEqual(len(st), 2)
        self.assertEqual(st[0].id, 'BW.MANZ..EHE')
        self.assertEqual(st[1].id, 'BW.ROTZ..EHE')

    def test_getPAZ(self):
        t = UTCDateTime('20090808')
        data = self.client.station.getPAZ('BW', 'MANZ', t)
        self.assertEqual(data['zeros'], [0j, 0j])
        self.assertEqual(data['sensitivity'], 2516800000.0)
        self.assertEqual(len(data['poles']), 5)
        self.assertEqual(data['poles'][0], (-0.037004000000000002 + 0.037016j))
        self.assertEqual(data['poles'][1], (-0.037004000000000002 - 0.037016j))
        self.assertEqual(data['poles'][2], (-251.33000000000001 + 0j))
        self.assertEqual(data['poles'][3],
                         (-131.03999999999999 - 467.29000000000002j))
        self.assertEqual(data['poles'][4],
                         (-131.03999999999999 + 467.29000000000002j))
        self.assertEqual(data['gain'], 60077000.0)

    def untested(self):
        """
        """
        client = self.client
        t = UTCDateTime('20100310')
        print client.waveform.getWaveform("BW", "HROE", "", "EHN", t, t + 1800)
        t = UTCDateTime("2010-03-19 00:00:01")
        print client.waveform.getWaveform("BW", "MANZ", "", "EHZ", t, t + 20)
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


def suite():
    return unittest.makeSuite(ClientTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
