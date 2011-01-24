#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The DHI/Fissures client test suite.
"""

from obspy.core import UTCDateTime, read
from obspy.fissures import Client
import os
import unittest
import numpy as np


class ClientTestSuite(unittest.TestCase):
    """
    Test cases for DHI/Fissures client
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), 'data')

    def test_getWavefrom(self):
        """
        Retrieve data from DHI/Fissures, compare stat attributes.
        
        """
        client = Client()
        t = UTCDateTime("2003-06-20T05:59:00.0000")
        st = client.getWaveform("GE", "APE", "", "SHZ", t, t + 10)
        tr = st[0]
        self.assertEqual('GE', tr.stats.network)
        self.assertEqual('APE', tr.stats.station)
        self.assertEqual('', tr.stats.location)
        self.assertEqual('SHZ', tr.stats.channel)
        self.assertEqual(UTCDateTime(2003, 6, 20, 5, 59, 0, 1000),
                         tr.stats.starttime)
        self.assertEqual(50.0, tr.stats.sampling_rate)
        self.assertEqual(501, len(tr.data))
        # compare with data retrieved via ArcLink
        st2 = read(os.path.join(self.path, 'arclink.mseed'))
        st2.trim(starttime=t, endtime=t + 10)
        np.testing.assert_array_equal(st[0].data, st2[0].data)

    def test_getNetworkIds(self):
        """
        Retrieve networks_ids from DHI
        """
        client = Client()
        print "This will take a very long time"
        ids = client.getNetworkIds()
        self.assertTrue('GE' in ids)
        self.assertTrue(len(ids) > 300)

    def test_getStationIds(self):
        """
        Retrieve station_ids from DHI
        """
        client = Client()
        ids = client.getStationIds(network_id='GE')
        stations = ['BRNL', 'PMG', 'MORC', 'DSB', 'LID', 'WLF', 'STU',
                    'BGIO', 'MLR', 'KBS']
        self.assertEqual(stations, ids[0:10])
        self.assertTrue(len(ids) > 70)


def suite():
    return unittest.makeSuite(ClientTestSuite, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
