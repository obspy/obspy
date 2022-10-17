# -*- coding: utf-8 -*-
"""
The obspy.clients.seedlink.basic_client test suite.
"""
import unittest

import pytest

from obspy import UTCDateTime
from obspy.clients.seedlink.basic_client import Client


pytestmark = pytest.mark.network


class ClientTestCase(unittest.TestCase):
    def setUp(self):
        self.client = Client("rtserver.ipgp.fr")

    def test_get_waveform(self):
        def _test_offset_from_realtime(offset):
            t = UTCDateTime() - offset
            request = ["G", "FDFM", "00", "LHN", t, t + 20]
            st = self.client.get_waveforms(*request)
            self.assertGreater(len(st), 0)
            for tr in st:
                self.assertEqual(tr.id, ".".join(request[:4]))
            self.assertTrue(any([len(tr) > 0 for tr in st]))
            st.merge(1)
            self.assertTrue(abs(tr.stats.starttime - request[4]) < 1)
            self.assertTrue(abs(tr.stats.endtime - request[5]) < 1)

        # getting a result depends on two things.. how long backwards the ring
        # buffer stores data and how close to realtime the data is available,
        # so check some different offsets and see if we get some data
        for offset in (3600, 2000, 1000, 500):
            try:
                _test_offset_from_realtime(offset)
            except AssertionError:
                continue
            else:
                break
        else:
            raise

    def test_get_info(self):
        """
        Test fetching station information
        """
        client = self.client

        info = client.get_info(station='F*')
        self.assertIn(('G', 'FDFM'), info)
        # should have at least 7 stations
        self.assertTrue(len(info) > 2)
        # only fetch one station
        info = client.get_info(network='G', station='FDFM')
        self.assertEqual([('G', 'FDFM')], info)
        # check that we have a cache on station level
        self.assertIn(('G', 'FDFM'), client._station_cache)
        self.assertTrue(len(client._station_cache) > 20)
        self.assertEqual(client._station_cache_level, "station")

    def test_multiple_waveform_requests_with_multiple_info_requests(self):
        """
        This test a combination of waveform requests that internally do
        multiple info requests on increasing detail levels
        """
        def _test_offset_from_realtime(offset):
            # need to reinit to clean out any caches
            self.setUp()
            t = UTCDateTime() - offset
            # first do a request that needs an info request on station level
            # only
            st = self.client.get_waveforms("*", "F?FM", "??", "B??", t, t + 5)
            self.assertGreater(len(st), 2)
            self.assertTrue(len(self.client._station_cache) > 20)
            station_cache_size = len(self.client._station_cache)
            self.assertIn(("G", "FDFM"), self.client._station_cache)
            self.assertEqual(self.client._station_cache_level, "station")
            for tr in st:
                self.assertEqual(tr.stats.network, "G")
                self.assertEqual(tr.stats.station, "FDFM")
                self.assertEqual(tr.stats.channel[0], "B")
            # now make a subsequent request that needs an info request on
            # channel level
            st = self.client.get_waveforms("*", "F?FM", "*", "B*", t, t + 5)
            self.assertGreater(len(st), 2)
            self.assertTrue(
                len(self.client._station_cache) > station_cache_size)
            self.assertIn(("G", "FDFM", "00", "BHZ"),
                          self.client._station_cache)
            self.assertEqual(self.client._station_cache_level, "channel")
            for tr in st:
                self.assertEqual(tr.stats.network, "G")
                self.assertEqual(tr.stats.station, "FDFM")
                self.assertEqual(tr.stats.channel[0], "B")

        # getting a result depends on two things.. how long backwards the ring
        # buffer stores data and how close to realtime the data is available,
        # so check some different offsets and see if we get some data
        for offset in (3600, 2000, 1000, 500):
            try:
                _test_offset_from_realtime(offset)
            except AssertionError:
                continue
            else:
                break
        else:
            raise
