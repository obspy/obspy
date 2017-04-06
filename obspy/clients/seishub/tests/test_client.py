#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The obspy.clients.seishub.client test suite.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import sys
import unittest

if sys.version_info.major == 2:
    from urllib2 import urlopen
else:
    from urllib.request import urlopen

import numpy as np

from obspy.core import AttribDict, UTCDateTime
from obspy.clients.seishub import Client


TESTSERVER = "http://teide.geophysik.uni-muenchen.de:8080"
TESTSERVER_UNREACHABLE_MSG = "Seishub test server not reachable."


def _check_server_availability():
    """
    Returns an empty string if server is reachable or failure message
    otherwise.
    """
    try:
        code = urlopen(TESTSERVER, timeout=3).getcode()
        assert(code == 200)
    except Exception:
        return TESTSERVER_UNREACHABLE_MSG
    return ""


@unittest.skipIf(_check_server_availability(), TESTSERVER_UNREACHABLE_MSG)
class ClientTestCase(unittest.TestCase):
    """
    Test cases for the SeisHub client.
    """

    def setUp(self):
        self.client = Client(TESTSERVER)

    def test_get_waveforms(self):
        """
        Test fetching waveforms from the server
        """
        t = UTCDateTime(2012, 1, 1, 12)
        st = self.client.waveform.get_waveforms('GR', 'FUR', '', 'BH*',
                                                t, t + 5)
        st.sort(reverse=True)
        self.assertEqual(len(st), 3)
        for tr, cha in zip(st, ('BHZ', 'BHN', 'BHE')):
            self.assertEqual(tr.stats.network, 'GR')
            self.assertEqual(tr.stats.station, 'FUR')
            self.assertEqual(tr.stats.location, '')
            self.assertEqual(tr.stats.channel, cha)
            self.assertEqual(tr.stats.sampling_rate, 20.0)
            self.assertEqual(len(tr), 101)
        self.assertEqual(st[0].stats.starttime,
                         UTCDateTime('2012-01-01T12:00:00.019999Z'))
        self.assertEqual(st[1].stats.starttime,
                         UTCDateTime('2012-01-01T12:00:00.020000Z'))
        self.assertEqual(st[2].stats.starttime,
                         UTCDateTime('2012-01-01T12:00:00.019999Z'))
        np.testing.assert_array_equal(
            st[0].data[:5],
            np.array([-804, -736, -839, -897, -953], dtype=np.int32))
        np.testing.assert_array_equal(
            st[0].data[-5:],
            np.array([-401, -350, -400, -391, -402], dtype=np.int32))
        np.testing.assert_array_equal(
            st[1].data[:5],
            np.array([-136, -163, -208, -117, -20], dtype=np.int32))
        np.testing.assert_array_equal(
            st[1].data[-5:],
            np.array([-1233, -1254, -1227, -1213, -1288], dtype=np.int32))
        np.testing.assert_array_equal(
            st[2].data[:5],
            np.array([56, -6, -30, -38, -20], dtype=np.int32))
        np.testing.assert_array_equal(
            st[2].data[-5:],
            np.array([-1143, -1146, -1076, -998, -1050], dtype=np.int32))

#    def test_getWaveformApplyFilter(self):
#        t = UTCDateTime("2009-09-03 00:00:00")
#        #1 - w/o apply_filter
#        st = self.client.waveform.get_waveforms("BW", "RTPI", "", "EHZ",
#                                              t, t + 20, apply_filter=False)
#        self.assertEqual(len(st), 1)
#        self.assertEqual(st[0].stats.network, '')
#        self.assertEqual(st[0].stats.station, 'GP01')
#        self.assertEqual(st[0].stats.location, '')
#        self.assertEqual(st[0].stats.channel, 'SHZ')
#        #2 - w/ apply_filter
#        st = self.client.waveform.get_waveforms("BW", "RTPI", "", "EHZ",
#                                              t, t + 20, apply_filter=True)
#        self.assertEqual(len(st), 1)
#        self.assertEqual(st[0].stats.network, 'BW')
#        self.assertEqual(st[0].stats.station, 'RTPI')
#        self.assertEqual(st[0].stats.location, '')
#        self.assertEqual(st[0].stats.channel, 'EHZ')

    def test_get_event_list(self):
        c = self.client.event
        # UTCDateTimes
        events = c.get_list(min_datetime=UTCDateTime("2009-01-01T00:00:00"),
                            max_datetime=UTCDateTime("2009-01-10T00:00:00"))
        self.assertEqual(len(events), 4)
        # time strings with T as separator
        events = c.get_list(min_datetime="2009-01-01T00:00:00",
                            max_datetime="2009-01-10T00:00:00")
        self.assertEqual(len(events), 4)
        # time strings with space as separator
        events = c.get_list(min_datetime="2009-01-01 00:00:00",
                            max_datetime="2009-01-10 00:00:00")
        self.assertEqual(len(events), 4)

    def test_get_network_ids(self):
        items = ['KT', 'BW', 'CZ', 'GR', 'NZ']
        data = self.client.waveform.get_network_ids()
        for item in items:
            self.assertIn(item, data)

    def test_ping(self):
        # current server
        time = self.client.ping()
        self.assertTrue(isinstance(time, float))

    def test_get_station_ids(self):
        # 1 - some selected stations
        stations = ['FUR', 'FURT', 'ROTZ', 'RTAK', 'MANZ', 'WET']
        data = self.client.waveform.get_station_ids()
        for station in stations:
            self.assertIn(station, data)
        # 2 - all stations of network BW
        stations = ['FURT', 'ROTZ', 'RTAK', 'MANZ']
        data = self.client.waveform.get_station_ids(network='BW')
        for station in stations:
            self.assertIn(station, data)

    def test_get_location_ids(self):
        # 1 - all locations
        items = ['', '10']
        data = self.client.waveform.get_location_ids()
        for item in items:
            self.assertIn(item, data)
        # 2 - all locations for network BW
        items = ['']
        data = self.client.waveform.get_location_ids(network='BW')
        for item in items:
            self.assertIn(item, data)
        # 3 - all locations for network BW and station MANZ
        items = ['']
        data = self.client.waveform.get_location_ids(network='BW',
                                                     station='MANZ')
        for item in items:
            self.assertIn(item, data)

    def test_get_channel_ids(self):
        # 1 - all channels
        items = ['AEX', 'AEY', 'BAN', 'BAZ', 'BHE', 'BHN', 'BHZ', 'EHE', 'EHN',
                 'EHZ', 'HHE', 'HHN', 'HHZ', 'LHE', 'LHN', 'LHZ', 'SHE', 'SHN',
                 'SHZ']
        data = self.client.waveform.get_channel_ids()
        for item in items:
            self.assertIn(item, data)
        # 2 - all channels for network BW
        items = ['AEX', 'AEY', 'BAN', 'BAZ', 'BHE', 'BHN', 'BHZ', 'EHE', 'EHN',
                 'EHZ', 'HHE', 'HHN', 'HHZ', 'SHE', 'SHN', 'SHZ']
        data = self.client.waveform.get_channel_ids(network='BW')
        for item in items:
            self.assertIn(item, data)
        # 3 - all channels for network BW and station MANZ
        items = ['AEX', 'AEY', 'EHE', 'EHN', 'EHZ', 'SHE', 'SHN', 'SHZ']
        data = self.client.waveform.get_channel_ids(network='BW',
                                                    station='MANZ')
        for item in items:
            self.assertIn(item, data)
        # 4 - all channels for network BW, station MANZ and given location
        items = ['AEX', 'AEY', 'EHE', 'EHN', 'EHZ', 'SHE', 'SHN', 'SHZ']
        data = self.client.waveform.get_channel_ids(
            network='BW', station='MANZ', location='')
        for item in items:
            self.assertIn(item, data)

    def test_get_preview(self):
        # multiple channels / MiniSEED
        t1 = UTCDateTime('20080101')
        t2 = UTCDateTime('20080201')
        st = self.client.waveform.get_previews("BW", "M*", "", "EHZ", t1, t2)
        self.assertEqual(len(st), 4)
        self.assertEqual(st[0].stats.network, 'BW')
        self.assertEqual(st[0].stats.channel, 'EHZ')
        self.assertEqual(st[0].stats.delta, 30.0)
        # single channel / GSE2
        t1 = UTCDateTime('20090101')
        t2 = UTCDateTime('20100101')
        st = self.client.waveform.get_previews("BW", "RTLI", "", "EHN", t1, t2)
        self.assertEqual(len(st), 1)
        self.assertEqual(st[0].id, 'BW.RTLI..EHN')
        self.assertEqual(st[0].stats.delta, 30.0)
        self.assertEqual(len(st[0]), 205642)
        self.assertEqual(st[0].stats.npts, 205642)

    def test_get_preview_by_ids(self):
        # multiple channels / MiniSEED
        t1 = UTCDateTime('20080101')
        t2 = UTCDateTime('20080201')
        # via list
        st = self.client.waveform.get_previews_by_ids(
            ['BW.MANZ..EHE', 'BW.ROTZ..EHE'], t1, t2)
        st.sort()
        self.assertEqual(len(st), 2)
        self.assertEqual(st[0].id, 'BW.MANZ..EHE')
        self.assertEqual(st[1].id, 'BW.ROTZ..EHE')
        # via string
        st = self.client.waveform.get_previews_by_ids(
            'BW.MANZ..EHE,BW.ROTZ..EHE', t1, t2)
        st.sort()
        self.assertEqual(len(st), 2)
        self.assertEqual(st[0].id, 'BW.MANZ..EHE')
        self.assertEqual(st[1].id, 'BW.ROTZ..EHE')

    def test_get_paz(self):
        t = UTCDateTime('20090808')
        c = self.client
        # test the deprecated call too for one/two releases
        data = c.station.get_paz('BW.MANZ..EHZ', t)
        self.assertEqual(data['zeros'], [0j, 0j])
        self.assertEqual(data['sensitivity'], 2516800000.0)
        self.assertEqual(len(data['poles']), 5)
        self.assertEqual(data['poles'][0], (-0.037004 + 0.037016j))
        self.assertEqual(data['poles'][1], (-0.037004 - 0.037016j))
        self.assertEqual(data['poles'][2], (-251.33 + 0j))
        self.assertEqual(data['poles'][3],
                         (-131.03999999999999 - 467.29000000000002j))
        self.assertEqual(data['poles'][4],
                         (-131.03999999999999 + 467.29000000000002j))
        self.assertEqual(data['gain'], 60077000.0)
        # test some not allowed wildcards
        t = UTCDateTime('20120501')
        self.assertRaises(ValueError, c.station.get_paz, "BW.RLAS..BJ*", t)
        self.assertRaises(ValueError, c.station.get_paz, "BW.RLAS..*", t)
        self.assertRaises(ValueError, c.station.get_paz, "BW.RLAS..BJ?", t)
        self.assertRaises(ValueError, c.station.get_paz, "BW.R*..BJZ", t)
        # test with a XSEED file with a referenced PAZ response info (see #364)
        t = UTCDateTime("2012-05-10")
        result = AttribDict(
            {'gain': 1.0, 'poles': [0j],
             'sensitivity': 6319100000000.0, 'digitizer_gain': 1000000.0,
             'seismometer_gain': 6319100.0, 'zeros': [0j]})
        data = c.station.get_paz("BW.RLAS..BJZ", t)
        self.assertEqual(data, result)

    def test_get_coordinates(self):
        t = UTCDateTime("2010-05-03T23:59:30")
        data = self.client.station.get_coordinates(network="BW", station="UH1",
                                                   datetime=t, location="")
        result = {'elevation': 500.0, 'latitude': 48.081493000000002,
                  'longitude': 11.636093000000001}
        self.assertEqual(data, result)

    def test_get_waveform_with_metadata(self):
        # metadata change during t1 -> t2 !
        t1 = UTCDateTime("2010-05-03T23:59:30")
        t2 = UTCDateTime("2010-05-04T00:00:30")
        client = self.client
        self.assertRaises(Exception, client.waveform.get_waveforms, "BW",
                          "UH1", "", "EH*", t1, t2, get_paz=True,
                          get_coordinates=True)
        st = client.waveform.get_waveforms("BW", "UH1", "", "EH*", t1, t2,
                                           get_paz=True, get_coordinates=True,
                                           metadata_timecheck=False)
        result = AttribDict({'zeros': [0j, 0j, 0j], 'sensitivity': 251650000.0,
                             'poles': [(-0.88 + 0.88j), (-0.88 - 0.88j),
                                       (-0.22 + 0j)],
                             'gain': 1.0,
                             'seismometer_gain': 400.0,
                             'digitizer_gain': 629121.0})
        self.assertEqual(st[0].stats.paz, result)
        result = AttribDict({'latitude': 48.081493000000002,
                             'elevation': 500.0,
                             'longitude': 11.636093000000001})
        self.assertEqual(st[0].stats.coordinates, result)

    def test_localcache(self):
        """
        Tests local 'caching' of XML seed resources and station list coordinate
        information to avoid repeat requests to server.
        Tests..
            - returned information is stored with client instance in memory
            - repeat requests do not get stored duplicated locally
            - repeat requests do not issue a request to server anymore
           (- right results for example with two different metadata sets at
              different times)
        """
        net = "BW"
        sta = "RTSA"
        netsta = ".".join([net, sta])
        seed_id = ".".join([net, sta, "", "EHZ"])
        t1 = UTCDateTime("2009-09-01")
        t2 = UTCDateTime("2012-10-23")
        coords1 = dict(elevation=1022.0, latitude=47.7673, longitude=12.842417)
        coords2 = dict(elevation=1066.0, latitude=47.768345,
                       longitude=12.841651)
        paz1 = {'digitizer_gain': 16000000.0,
                'gain': 1.0,
                'poles': [(-0.88 + 0.88j), (-0.88 - 0.88j), (-0.22 + 0j)],
                'seismometer_gain': 400.0,
                'sensitivity': 6400000000.0,
                'zeros': [0j, 0j, 0j]}
        paz2 = {'digitizer_gain': 1677850.0,
                'gain': 1.0,
                'poles': [(-4.444 + 4.444j), (-4.444 - 4.444j), (-1.083 + 0j)],
                'seismometer_gain': 400.0,
                'sensitivity': 671140000.0,
                'zeros': [0j, 0j, 0j]}
        c = self.client
        # before any requests
        self.assertEqual(len(c.xml_seeds), 0)
        self.assertEqual(len(c.station_list), 0)
        # after first t1 requests
        ret = c.station.get_coordinates(net, sta, t1)
        self.assertEqual(ret, coords1)
        self.assertEqual(len(c.station_list), 1)
        self.assertEqual(len(c.station_list[netsta]), 1)
        ret = c.station.get_paz(seed_id, t1)
        self.assertEqual(ret, paz1)
        self.assertEqual(len(c.xml_seeds), 1)
        self.assertEqual(len(c.xml_seeds[seed_id]), 1)
        # after first t2 requests
        ret = c.station.get_coordinates(net, sta, t2)
        self.assertEqual(ret, coords2)
        self.assertEqual(len(c.station_list), 1)
        self.assertEqual(len(c.station_list[netsta]), 2)
        ret = c.station.get_paz(seed_id, t2)
        self.assertEqual(ret, paz2)
        self.assertEqual(len(c.xml_seeds), 1)
        self.assertEqual(len(c.xml_seeds[seed_id]), 2)
        # get_list() is called if get_paz or get_coordinates ends up making a
        # request to server so we just overwrite it and let it raise to check
        # that no request is issued
        c.station.get_list = raise_on_call
        # after second t1 requests
        ret = c.station.get_coordinates(net, sta, t1)
        self.assertEqual(ret, coords1)
        self.assertEqual(len(c.station_list), 1)
        self.assertEqual(len(c.station_list[netsta]), 2)
        ret = c.station.get_paz(seed_id, t1)
        self.assertEqual(ret, paz1)
        self.assertEqual(len(c.xml_seeds), 1)
        self.assertEqual(len(c.xml_seeds[seed_id]), 2)
        # after second t2 requests
        ret = c.station.get_coordinates(net, sta, t2)
        self.assertEqual(ret, coords2)
        self.assertEqual(len(c.station_list), 1)
        self.assertEqual(len(c.station_list[netsta]), 2)
        ret = c.station.get_paz(seed_id, t2)
        self.assertEqual(ret, paz2)
        self.assertEqual(len(c.xml_seeds), 1)
        self.assertEqual(len(c.xml_seeds[seed_id]), 2)
        # new request that needs to connect to server, just to make sure the
        # monkey patch for raising on requests really works
        self.assertRaises(RequestException, c.station.get_coordinates,
                          "GR", "FUR", t2)
        self.assertRaises(RequestException, c.station.get_paz,
                          "GR.FUR..HHZ", t2)


class RequestException(Exception):
    pass


def raise_on_call(*args, **kwargs):
    raise RequestException("Unwanted request to server.")


def suite():
    return unittest.makeSuite(ClientTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
