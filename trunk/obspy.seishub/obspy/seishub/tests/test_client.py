#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The obspy.seishub.client test suite.
"""

from obspy.seishub import Client
import unittest
from obspy.core import UTCDateTime, AttribDict


class ClientTestCase(unittest.TestCase):
    """
    Test cases for the SeisHub client.
    """

    def setUp(self):
        self.client = Client("http://teide.geophysik.uni-muenchen.de:8080")

#    def test_getWaveformApplyFilter(self):
#        t = UTCDateTime("2009-09-03 00:00:00")
#        #1 - w/o apply_filter
#        st = self.client.waveform.getWaveform("BW", "RTPI", "", "EHZ",
#                                              t, t + 20, apply_filter=False)
#        self.assertEqual(len(st), 1)
#        self.assertEqual(st[0].stats.network, '')
#        self.assertEqual(st[0].stats.station, 'GP01')
#        self.assertEqual(st[0].stats.location, '')
#        self.assertEqual(st[0].stats.channel, 'SHZ')
#        #2 - w/ apply_filter
#        st = self.client.waveform.getWaveform("BW", "RTPI", "", "EHZ",
#                                              t, t + 20, apply_filter=True)
#        self.assertEqual(len(st), 1)
#        self.assertEqual(st[0].stats.network, 'BW')
#        self.assertEqual(st[0].stats.station, 'RTPI')
#        self.assertEqual(st[0].stats.location, '')
#        self.assertEqual(st[0].stats.channel, 'EHZ')

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
        items = ['KT', 'BW', 'CZ', 'GR', 'NZ']
        data = self.client.waveform.getNetworkIds()
        for item in items:
            self.assertTrue(item in data)

    def test_ping(self):
        # current server
        time = self.client.ping()
        self.assertTrue(isinstance(time, float))

    def test_getStationIds(self):
        #1 - some selected stations
        stations = ['FUR', 'FURT', 'ROTZ', 'RTAK', 'MANZ', 'WET']
        data = self.client.waveform.getStationIds()
        for station in stations:
            self.assertTrue(station in data)
        #2 - all stations of network BW
        stations = ['FURT', 'ROTZ', 'RTAK', 'MANZ']
        data = self.client.waveform.getStationIds(network='BW')
        for station in stations:
            self.assertTrue(station in data)

    def test_getLocationIds(self):
        #1 - all locations
        items = ['', '10']
        data = self.client.waveform.getLocationIds()
        for item in items:
            self.assertTrue(item in data)
        #2 - all locations for network BW
        items = ['']
        data = self.client.waveform.getLocationIds(network='BW')
        for item in items:
            self.assertTrue(item in data)
        #3 - all locations for network BW and station MANZ
        items = ['']
        data = self.client.waveform.getLocationIds(network='BW',
                                                   station='MANZ')
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
        data = self.client.waveform.getChannelIds(network='BW')
        for item in items:
            self.assertTrue(item in data)
        #3 - all channels for network BW and station MANZ
        items = ['AEX', 'AEY', 'EHE', 'EHN', 'EHZ', 'SHE', 'SHN', 'SHZ']
        data = self.client.waveform.getChannelIds(network='BW', station='MANZ')
        for item in items:
            self.assertTrue(item in data)
        #4 - all channels for network BW, station MANZ and given location
        items = ['AEX', 'AEY', 'EHE', 'EHN', 'EHZ', 'SHE', 'SHN', 'SHZ']
        data = self.client.waveform.getChannelIds(network='BW', station='MANZ',
                                                  location='')
        for item in items:
            self.assertTrue(item in data)

    def test_getPreview(self):
        # multiple channels / MiniSEED
        t1 = UTCDateTime('20080101')
        t2 = UTCDateTime('20080201')
        st = self.client.waveform.getPreview("BW", "M*", "", "EHZ", t1, t2)
        self.assertEqual(len(st), 4)
        self.assertEqual(st[0].stats.network, 'BW')
        self.assertEqual(st[0].stats.channel, 'EHZ')
        self.assertEqual(st[0].stats.delta, 30.0)
        # single channel / GSE2
        t1 = UTCDateTime('20090101')
        t2 = UTCDateTime('20100101')
        st = self.client.waveform.getPreview("BW", "RTLI", "", "EHN", t1, t2)
        self.assertEqual(len(st), 1)
        self.assertEqual(st[0].id, 'BW.RTLI..EHN')
        self.assertEqual(st[0].stats.delta, 30.0)
        self.assertEqual(len(st[0]), 205642)
        self.assertEqual(st[0].stats.npts, 205642)

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

    def test_getCoordinates(self):
        t = UTCDateTime("2010-05-03T23:59:30")
        data = self.client.station.getCoordinates(network="BW", station="UH1",
                                                  datetime=t, location="")
        result = {'elevation': 500.0, 'latitude': 48.081493000000002,
                  'longitude': 11.636093000000001}
        self.assertEqual(data, result)

    def test_getWaveform_with_metadata(self):
        # metadata change during t1 -> t2 !
        t1 = UTCDateTime("2010-05-03T23:59:30")
        t2 = UTCDateTime("2010-05-04T00:00:30")
        client = self.client
        self.assertRaises(Exception, client.waveform.getWaveform, "BW",
                          "UH1", "", "EH*", t1, t2, getPAZ=True,
                          getCoordinates=True)
        st = client.waveform.getWaveform("BW", "UH1", "", "EH*", t1, t2,
                                         getPAZ=True, getCoordinates=True,
                                         metadata_timecheck=False)
        result = AttribDict({'zeros': [0j, 0j, 0j], 'sensitivity': 251650000.0,
                             'poles': [(-0.88 + 0.88j), (-0.88 - 0.88j),
                                       (-0.22 + 0j)],
                             'gain': 1.0})
        self.assertEqual(st[0].stats.paz, result)
        result = AttribDict({'latitude': 48.081493000000002,
                             'elevation': 500.0,
                             'longitude': 11.636093000000001})
        self.assertEqual(st[0].stats.coordinates, result)


def suite():
    return unittest.makeSuite(ClientTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
