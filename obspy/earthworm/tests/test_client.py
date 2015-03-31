# -*- coding: utf-8 -*-
"""
The obspy.earthworm.client test suite.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport

import unittest

from obspy import read
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.util import NamedTemporaryFile
from obspy.core.util.decorator import skip_on_network_error
from obspy.earthworm import Client


class ClientTestCase(unittest.TestCase):
    """
    Test cases for obspy.earthworm.client.Client.
    """
    def setUp(self):
        # Monkey patch: set lower default precision of all UTCDateTime objects
        UTCDateTime.DEFAULT_PRECISION = 4
        self.client = Client("pubavo1.wr.usgs.gov", 16022, timeout=30.0)

    def tearDown(self):
        # restore default precision of all UTCDateTime objects
        UTCDateTime.DEFAULT_PRECISION = 6

    @skip_on_network_error
    def test_availability(self):
        data = self.client.availability()
        seeds = ["%s.%s.%s.%s" % (net, sta, loc, cha)
                 for net, sta, loc, cha, _start, _stop in data]
        self.assertTrue('AV.ACH.--.EHE' in seeds)

    @skip_on_network_error
    def test_getWaveform(self):
        """
        Tests getWaveform method.
        """
        client = self.client
        start = UTCDateTime(2015, 3, 22)
        end = start + 1.0
        # example 1 -- 1 channel, cleanup
        stream = client.getWaveform('AV', 'ACH', '--', 'EHE', start, end)
        self.assertEqual(len(stream), 1)
        delta = stream[0].stats.delta
        trace = stream[0]
        self.assertEqual(len(trace), 101)
        self.assertTrue(trace.stats.starttime >= start - delta)
        self.assertTrue(trace.stats.starttime <= start + delta)
        self.assertTrue(trace.stats.endtime >= end - delta)
        self.assertTrue(trace.stats.endtime <= end + delta)
        self.assertEqual(trace.stats.network, 'AV')
        self.assertEqual(trace.stats.station, 'ACH')
        self.assertEqual(trace.stats.location, '')
        self.assertEqual(trace.stats.channel, 'EHE')
        # example 2 -- 1 channel, no cleanup
        stream = client.getWaveform('AV', 'ACH', '--', 'EHE', start, end,
                                    cleanup=False)
        self.assertTrue(len(stream) >= 2)
        summed_length = sum(len(tr) for tr in stream)
        self.assertEqual(summed_length, 101)
        self.assertTrue(stream[0].stats.starttime >= start - delta)
        self.assertTrue(stream[0].stats.starttime <= start + delta)
        self.assertTrue(stream[-1].stats.endtime >= end - delta)
        self.assertTrue(stream[-1].stats.endtime <= end + delta)
        for trace in stream:
            self.assertEqual(trace.stats.network, 'AV')
            self.assertEqual(trace.stats.station, 'ACH')
            self.assertEqual(trace.stats.location, '')
            self.assertEqual(trace.stats.channel, 'EHE')
        # example 3 -- component wildcarded with '?'
        stream = client.getWaveform('AV', 'ACH', '--', 'EH?', start, end)
        self.assertEqual(len(stream), 3)
        for trace in stream:
            self.assertEqual(len(trace), 101)
            self.assertTrue(trace.stats.starttime >= start - delta)
            self.assertTrue(trace.stats.starttime <= start + delta)
            self.assertTrue(trace.stats.endtime >= end - delta)
            self.assertTrue(trace.stats.endtime <= end + delta)
            self.assertEqual(trace.stats.network, 'AV')
            self.assertEqual(trace.stats.station, 'ACH')
            self.assertEqual(trace.stats.location, '')
        self.assertEqual(stream[0].stats.channel, 'EHZ')
        self.assertEqual(stream[1].stats.channel, 'EHN')
        self.assertEqual(stream[2].stats.channel, 'EHE')

    @skip_on_network_error
    def test_saveWaveform(self):
        """
        Tests saveWaveform method.
        """
        # initialize client
        client = self.client
        start = UTCDateTime(2015, 3, 22)
        end = start + 1.0
        with NamedTemporaryFile() as tf:
            testfile = tf.name
            # 1 channel, cleanup (using SLIST to avoid dependencies)
            client.saveWaveform(testfile, 'AV', 'ACH', '--', 'EHE', start, end,
                                format="SLIST")
            stream = read(testfile)
        self.assertEqual(len(stream), 1)
        delta = stream[0].stats.delta
        trace = stream[0]
        self.assertEqual(len(trace), 101)
        self.assertTrue(trace.stats.starttime >= start - delta)
        self.assertTrue(trace.stats.starttime <= start + delta)
        self.assertTrue(trace.stats.endtime >= end - delta)
        self.assertTrue(trace.stats.endtime <= end + delta)
        self.assertEqual(trace.stats.network, 'AV')
        self.assertEqual(trace.stats.station, 'ACH')
        self.assertEqual(trace.stats.location, '')
        self.assertEqual(trace.stats.channel, 'EHE')


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(ClientTestCase, 'test'))
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
